"""Monocular SLAM / VO backends for the gripper studio.

Two backends are provided:

* ``VisualOdometry`` — pure OpenCV ORB + essential-matrix frame-to-frame VO.
  No external SLAM build required. Real-time on a laptop, but no loop
  closure and monocular scale ambiguity (translation magnitude is
  arbitrary), so trajectories will drift over long runs.

* ``OrbSlam3Backend`` — opt-in wrapper. Tries to import a community
  ORB-SLAM3 Python binding (``orbslam3`` or ``pyslam``). If neither is
  importable it raises with installation hints. Build instructions
  (macOS / Linux):

      git clone https://github.com/UZ-SLAMLab/ORB_SLAM3
      ... build DBoW2, g2o, Pangolin, then ORB_SLAM3 ...
      pip install git+https://github.com/jskinn/ORB_SLAM3-PythonBindings

  When wired up correctly, instantiating ``OrbSlam3Backend(vocab=..., settings=...)``
  hands frames to the C++ tracker and reports pose updates.

The Insta360 X3 in webcam mode delivers a stitched / dual-fisheye stream;
both backends accept a ``crop`` setting (``"left"``, ``"right"``, ``"full"``)
so we feed a single lens half into the SLAM frontend.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import cv2
import numpy as np


@dataclass
class Pose:
    R: np.ndarray  # 3x3 rotation, world->camera
    t: np.ndarray  # 3x1 translation, camera origin in world
    timestamp: float


@dataclass
class VOConfig:
    n_features: int = 1500
    match_ratio: float = 0.75
    min_matches: int = 30
    ransac_thresh: float = 1.0
    fx: float = 0.0  # 0 = derive from frame width (rough pinhole guess)
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    crop: str = "left"  # "full" | "left" | "right"


class SlamBackend(Protocol):
    name: str

    def reset(self) -> None: ...
    def process_frame(self, frame: np.ndarray) -> Optional[Pose]: ...
    def trajectory_xyz(self) -> np.ndarray: ...


def _crop_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "left":
        w = frame.shape[1] // 2
        return frame[:, :w]
    if mode == "right":
        w = frame.shape[1] // 2
        return frame[:, w:]
    return frame


class VisualOdometry:
    """Frame-to-frame monocular VO. Cumulative pose is accumulated in world
    frame; scale is whatever the first valid recoverPose returns (drifts)."""

    name = "OpenCV ORB VO"

    def __init__(self, cfg: VOConfig | None = None):
        self.cfg = cfg or VOConfig()
        self.orb = cv2.ORB_create(nfeatures=self.cfg.n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.K: np.ndarray | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self._prev_kp = None
        self._prev_des = None
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.trajectory: list[np.ndarray] = [self.t.copy()]

    def reset(self) -> None:
        self.K = None
        self._reset_state()

    def _intrinsics(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        # Default to a wide-ish pinhole (fx ≈ width). It's a placeholder —
        # essential-matrix VO is fairly tolerant but absolute scale is bogus.
        fx = self.cfg.fx or float(w)
        fy = self.cfg.fy or float(w)
        cx = self.cfg.cx or w / 2.0
        cy = self.cfg.cy or h / 2.0
        return np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )

    def process_frame(self, frame: np.ndarray) -> Optional[Pose]:
        frame = _crop_frame(frame, self.cfg.crop)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < self.cfg.min_matches:
            self._prev_kp, self._prev_des = kp, des
            return None

        if self._prev_des is None or self.K is None:
            self.K = self._intrinsics(frame)
            self._prev_kp, self._prev_des = kp, des
            return None

        matches = self.matcher.knnMatch(self._prev_des, des, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.cfg.match_ratio * n.distance:
                good.append(m)
        if len(good) < self.cfg.min_matches:
            self._prev_kp, self._prev_des = kp, des
            return None

        pts_prev = np.float32([self._prev_kp[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(
            pts_curr, pts_prev, self.K,
            method=cv2.RANSAC, prob=0.999,
            threshold=self.cfg.ransac_thresh,
        )
        if E is None or E.shape != (3, 3):
            self._prev_kp, self._prev_des = kp, des
            return None

        _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, self.K, mask=mask)
        if np.linalg.norm(t) < 1e-6:
            self._prev_kp, self._prev_des = kp, des
            return None

        # Compose: the recovered (R, t) maps prev camera frame to curr.
        # Camera origin in world: t_world += R_world_prev @ t.
        self.t = self.t + self.R @ t
        self.R = R @ self.R
        self.trajectory.append(self.t.copy())
        self._prev_kp, self._prev_des = kp, des
        return Pose(R=self.R.copy(), t=self.t.copy(), timestamp=time.time())

    def trajectory_xyz(self) -> np.ndarray:
        if not self.trajectory:
            return np.zeros((0, 3))
        return np.hstack(self.trajectory).T


class OrbSlam3Backend:
    """Wrapper for an installed ORB-SLAM3 Python binding.

    Tries ``orbslam3`` first, then ``pyslam``. If neither imports we raise
    with the install hint from this module's docstring rather than silently
    falling back, so the user knows what's going on.
    """

    name = "ORB-SLAM3"

    def __init__(self, vocab_path: str, settings_path: str, crop: str = "left"):
        self.crop = crop
        self.trajectory: list[np.ndarray] = [np.zeros((3, 1))]
        try:
            import orbslam3  # type: ignore
            self._impl = orbslam3.System(
                vocab_path, settings_path, orbslam3.Sensor.MONOCULAR
            )
            self._impl.set_use_viewer(False)
            self._impl.initialize()
            self._kind = "orbslam3"
        except ImportError:
            try:
                import pyslam  # type: ignore  # noqa: F401
                raise NotImplementedError(
                    "pyslam detected but its API isn't auto-wired here yet — "
                    "see slam.OrbSlam3Backend for the integration point."
                )
            except ImportError as e:
                raise RuntimeError(
                    "ORB-SLAM3 backend not available: install a Python binding "
                    "(see slam.py docstring for build instructions)."
                ) from e

    def reset(self) -> None:
        if hasattr(self._impl, "reset"):
            self._impl.reset()
        self.trajectory = [np.zeros((3, 1))]

    def process_frame(self, frame: np.ndarray) -> Optional[Pose]:
        frame = _crop_frame(frame, self.crop)
        ts = time.time()
        # ORB-SLAM3 binding signature varies; the orbslam3 binding accepts
        # (image_bgr, timestamp_seconds) and returns a 4x4 pose matrix or None.
        try:
            T = self._impl.process_image_mono(frame, ts)
        except AttributeError:
            T = self._impl.process_mono(frame, ts)
        if T is None:
            return None
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            return None
        # ORB-SLAM3 returns world->camera; invert to get camera origin in world.
        R = T[:3, :3]
        t = T[:3, 3:4]
        Rw = R.T
        tw = -Rw @ t
        self.trajectory.append(tw.copy())
        return Pose(R=Rw, t=tw, timestamp=ts)

    def trajectory_xyz(self) -> np.ndarray:
        if not self.trajectory:
            return np.zeros((0, 3))
        return np.hstack(self.trajectory).T


class CameraWorker:
    """Background capture + SLAM thread. The main thread polls ``snapshot``
    to get the latest frame and pose without blocking the GUI loop."""

    def __init__(self, source: int | str, backend: SlamBackend):
        self.source = source
        self.backend = backend
        self.cap: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._pose: Pose | None = None
        self._fps = 0.0
        self._n = 0
        self.error: str | None = None
        # Pose subscribers (e.g. recorder) — called from the worker thread.
        self._subscribers: list = []

    def subscribe(self, fn) -> None:
        self._subscribers.append(fn)

    def unsubscribe(self, fn) -> None:
        if fn in self._subscribers:
            self._subscribers.remove(fn)

    def start(self) -> bool:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = f"failed to open camera {self.source!r}"
            return False
        self.cap = cap
        self.running = True
        self.error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _loop(self) -> None:
        last = time.time()
        smooth = 0.0
        while self.running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                self.error = "camera read failed"
                time.sleep(0.05)
                continue
            try:
                pose = self.backend.process_frame(frame)
            except Exception as e:
                self.error = f"slam: {e}"
                pose = None
            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                smooth = 0.9 * smooth + 0.1 * (1.0 / dt)
            with self.lock:
                self._frame = frame
                self._pose = pose
                self._fps = smooth
                self._n += 1
            if pose is not None:
                for fn in list(self._subscribers):
                    try:
                        fn(pose)
                    except Exception:
                        pass

    def snapshot(self):
        with self.lock:
            return self._frame, self._pose, self._fps, self._n
