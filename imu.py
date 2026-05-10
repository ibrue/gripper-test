"""IMU support for the gripper studio.

Important reality check on the Insta360 X3:

  The X3 does NOT expose its built-in IMU over USB in webcam mode. Only
  video is streamed live. The gyro + accelerometer samples are written
  to the recorded ``.insv`` / ``.mp4`` file as a metadata track and have
  to be extracted offline (Gyroflow, Insta360 Studio, telemetry-parser).

So this module gives you three paths:

* ``Insta360FileLoader`` — extracts IMU from a recorded .insv/.mp4 using
  the ``telemetry-parser`` Python binding (``pip install
  telemetry-parser-py``, also handles GoPro / DJI / Sony files).
  Use this for post-collection visual-inertial replay.

* ``LiveImuSource`` — abstract base for live IMU sources you wire in
  yourself (e.g. a small Adafruit/Bosch IMU board plumbed over serial
  or UDP, mounted alongside the X3). Subclass and call ``emit`` with
  each ``ImuSample``.

* ``OrientationTracker`` — Madgwick-style complementary filter that
  consumes ``ImuSample`` and maintains a current orientation quaternion.
  The studio uses this to:
    - draw a live attitude indicator
    - hand a rotation prior to the SLAM frontend (improves VO when
      visual matches are sparse).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np


@dataclass
class ImuSample:
    timestamp: float        # seconds (monotonic ok; just be consistent)
    gyro: np.ndarray        # shape (3,), rad/s, [x,y,z] in sensor frame
    accel: np.ndarray       # shape (3,), m/s^2, sensor frame
    mag: Optional[np.ndarray] = None  # optional magnetometer


# -----------------------------------------------------------------------------
# Orientation tracking
# -----------------------------------------------------------------------------

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def quat_to_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    """Returns (roll, pitch, yaw) in degrees (XYZ-intrinsic)."""
    w, x, y, z = q
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w*y - z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class OrientationTracker:
    """Madgwick-style orientation filter (no magnetometer).

    Pure gyro integration drifts over seconds; the accel correction nudges
    the estimate so gravity stays pointing down in body frame.
    """

    def __init__(self, beta: float = 0.04):
        self.beta = beta  # filter gain — higher = more accel correction
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_t: Optional[float] = None
        self.lock = threading.Lock()

    def reset(self) -> None:
        with self.lock:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])
            self._last_t = None

    def update(self, sample: ImuSample) -> np.ndarray:
        with self.lock:
            if self._last_t is None:
                self._last_t = sample.timestamp
                return self.q.copy()
            dt = max(1e-4, sample.timestamp - self._last_t)
            self._last_t = sample.timestamp
            q = self.q
            gx, gy, gz = sample.gyro
            ax, ay, az = sample.accel
            anorm = math.sqrt(ax*ax + ay*ay + az*az)
            # Gyro-only integration baseline.
            qdot = 0.5 * _quat_mul(q, np.array([0.0, gx, gy, gz]))
            if anorm > 1e-6:
                ax, ay, az = ax/anorm, ay/anorm, az/anorm
                w, x, y, z = q
                # Gradient of f = R^T g - a_body
                f = np.array([
                    2*(x*z - w*y) - ax,
                    2*(w*x + y*z) - ay,
                    2*(0.5 - x*x - y*y) - az,
                ])
                J = np.array([
                    [-2*y,  2*z, -2*w, 2*x],
                    [ 2*x,  2*w,  2*z, 2*y],
                    [   0, -4*x, -4*y,  0],
                ])
                grad = J.T @ f
                gn = np.linalg.norm(grad)
                if gn > 0:
                    qdot = qdot - self.beta * grad / gn
            self.q = _quat_normalize(q + qdot * dt)
            return self.q.copy()

    def quaternion(self) -> np.ndarray:
        with self.lock:
            return self.q.copy()

    def euler_deg(self) -> tuple[float, float, float]:
        return quat_to_euler_deg(self.quaternion())

    def rotmat(self) -> np.ndarray:
        return quat_to_rotmat(self.quaternion())


# -----------------------------------------------------------------------------
# Live IMU source scaffolding (no hardware shipped — bring your own)
# -----------------------------------------------------------------------------

class LiveImuSource:
    """Base class for any live IMU stream. Subclass and call ``emit`` per
    sample. The studio attaches a callback via ``subscribe``."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[ImuSample], None]] = []
        self._lock = threading.Lock()
        self.running = False

    def subscribe(self, fn: Callable[[ImuSample], None]) -> None:
        with self._lock:
            self._subscribers.append(fn)

    def emit(self, sample: ImuSample) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for fn in subs:
            try:
                fn(sample)
            except Exception:
                pass

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


# -----------------------------------------------------------------------------
# Insta360 .insv / .mp4 file IMU extraction (offline)
# -----------------------------------------------------------------------------

class Insta360FileLoader:
    """Read IMU samples from a recorded Insta360 (.insv / .mp4) file.

    Requires the optional dependency ``telemetry-parser-py``. We import it
    lazily so the studio still launches without it installed; trying to load
    a file then surfaces a clear error.
    """

    def __init__(self, path: str):
        self.path = path

    def load(self) -> list[ImuSample]:
        try:
            from telemetry_parser import Parser  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "IMU file loading needs the telemetry-parser-py package: "
                "pip install telemetry-parser-py"
            ) from e

        parser = Parser(self.path)
        # The library returns a flat list of dicts with keys including
        # 'timestamp_ms', 'gyro' [rad/s, xyz], 'accel' [m/s^2, xyz].
        try:
            raw = parser.normalized_imu()
        except AttributeError:
            raw = parser.telemetry()

        out: list[ImuSample] = []
        for entry in raw:
            ts_ms = entry.get("timestamp_ms") or entry.get("ts_ms") or entry.get("t")
            if ts_ms is None:
                continue
            g = entry.get("gyro") or entry.get("g")
            a = entry.get("accel") or entry.get("a")
            if g is None or a is None:
                continue
            out.append(ImuSample(
                timestamp=float(ts_ms) / 1000.0,
                gyro=np.asarray(g, dtype=np.float64),
                accel=np.asarray(a, dtype=np.float64),
            ))
        return out


@dataclass
class ImuReplay:
    """Replay a buffered list of ImuSample to a callback at wall-clock pace.

    Useful for feeding a recorded .insv IMU track into the OrientationTracker
    in lockstep with video frames during post-hoc analysis.
    """

    samples: list[ImuSample]
    callback: Callable[[ImuSample], None]
    speed: float = 1.0
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def start(self) -> None:
        if not self.samples:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        t0_data = self.samples[0].timestamp
        t0_wall = time.time()
        for sample in self.samples:
            if not self._running:
                return
            target_wall = t0_wall + (sample.timestamp - t0_data) / self.speed
            now = time.time()
            if target_wall > now:
                time.sleep(target_wall - now)
            self.callback(sample)
