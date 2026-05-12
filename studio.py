"""umi — gripper data-collection studio.

Tk GUI on top of ``Gripper`` (servo control), ``slam`` (monocular VO /
ORB-SLAM3 wrapper) and ``imu`` (orientation tracker + Insta360 file
loader). Adds:

* arrow-key jog with acceleration on hold (Left/Up = open, Right/Down = close)
* live matplotlib plots for servo position / current / temperature
* live camera feed + 3D camera-trajectory plot
* IMU panel (load IMU from a recorded .insv for replay-fusion)
* recorder (CSV of synchronized signals + MP4 of the camera feed)

Branding: title and header use ``Moonhouse`` if installed (download free
for personal use from dafont and install via Font Book), else fall back
to a stylish bold sans-serif.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from collections import deque
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # camera/SLAM features become unavailable
    cv2 = None  # type: ignore

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from gripper import (
    PROTOCOL_VERSION,
    SCAN_BAUDRATES,
    SCAN_IDS,
    TICK_TO_DEG,
    Gripper,
    PacketHandler,
    PortHandler,
    COMM_SUCCESS,
)
from slam import CameraWorker, OrbSlam3Backend, VisualOdometry, VOConfig
from imu import (
    Insta360FileLoader,
    ImuReplay,
    ImuSample,
    OrientationTracker,
)
from dataset import DatasetManager, Episode
from x3_control import X3Control, X3Device
import version as umi_version


# -----------------------------------------------------------------------------
# Branding
# -----------------------------------------------------------------------------

BRAND_NAME = "umi"
BRAND_TAGLINE = "gripper studio"
BRAND_FONT_FAMILIES = (
    "Moonhouse",
    "Futura",
    "Avenir Next Heavy",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
)

# Sober dark palette to make the live plots and camera feed read well.
BG = "#13151a"
PANEL = "#1c1f27"
INK = "#e8eaf0"
DIM = "#8c93a3"
ACCENT = "#5cc8ff"
WARN = "#ffb84d"
OK = "#4ade80"


def _pick_brand_family(root: tk.Tk) -> str:
    available = set(tkfont.families(root))
    for name in BRAND_FONT_FAMILIES:
        if name in available:
            return name
    return "TkDefaultFont"


# -----------------------------------------------------------------------------
# Held-key tracking with acceleration
# -----------------------------------------------------------------------------

@dataclass
class HoldJog:
    """Models held arrow keys as a velocity that accelerates while held and
    decays on release. Tk's autorepeat fires KeyRelease+KeyPress in quick
    succession on X11; we debounce releases by waiting one autorepeat
    interval before treating a key as truly released.
    """

    accel: float = 3000.0       # ticks / s^2
    max_v: float = 2400.0       # ticks / s
    decay: float = 8000.0       # ticks / s^2 toward zero on release
    autorepeat_grace_s: float = 0.06

    velocity: float = 0.0
    _direction: int = 0          # -1 open, +1 close, 0 none
    _held: dict = field(default_factory=dict)  # key -> True
    _release_pending: dict = field(default_factory=dict)  # key -> wall time

    def press(self, direction: int, key: str) -> None:
        self._held[key] = True
        self._release_pending.pop(key, None)
        self._direction = direction

    def release(self, key: str) -> None:
        # Defer the actual release to filter X11 autorepeat KeyRelease bursts.
        self._release_pending[key] = time.time()

    def reconcile(self, now: float) -> None:
        for key, t in list(self._release_pending.items()):
            if now - t >= self.autorepeat_grace_s:
                self._held.pop(key, None)
                self._release_pending.pop(key, None)
        if not self._held:
            self._direction = 0

    def step(self, dt: float) -> float:
        if self._direction != 0:
            self.velocity += self._direction * self.accel * dt
            self.velocity = max(-self.max_v, min(self.max_v, self.velocity))
        else:
            if self.velocity > 0:
                self.velocity = max(0.0, self.velocity - self.decay * dt)
            elif self.velocity < 0:
                self.velocity = min(0.0, self.velocity + self.decay * dt)
        return self.velocity * dt


# -----------------------------------------------------------------------------
# Recorder
# -----------------------------------------------------------------------------

CSV_COLUMNS = [
    "wall_time", "elapsed_s",
    "offset_ticks", "open_limit", "close_limit",
    "servo_a_pos", "servo_a_current_ma", "servo_a_temp_c",
    "servo_b_pos", "servo_b_current_ma", "servo_b_temp_c",
    "vo_x", "vo_y", "vo_z",
    "imu_qw", "imu_qx", "imu_qy", "imu_qz",
    "imu_roll_deg", "imu_pitch_deg", "imu_yaw_deg",
]


class EpisodeRecorder:
    """Writes one Episode: samples.csv + optional video.mp4. ``finalize``
    updates the episode's meta.json with duration / counts when stopped."""

    def __init__(self, episode: Episode, record_video: bool):
        self.episode = episode
        self.record_video = record_video
        self._csv_file = None
        self._csv_writer = None
        self._video_writer = None
        self._video_size: Optional[tuple[int, int]] = None
        self.t0 = time.time()
        self.lock = threading.Lock()

    @property
    def n_samples(self) -> int:
        return self.episode.n_samples

    @property
    def n_frames(self) -> int:
        return self.episode.n_frames

    def start(self) -> None:
        with self.lock:
            self._csv_file = open(self.episode.samples_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(CSV_COLUMNS)
            self.t0 = time.time()
            self.episode.started_at = self.t0
            self.episode.n_samples = 0
            self.episode.n_frames = 0

    def stop(self) -> None:
        with self.lock:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            self.episode.duration_s = max(0.0, time.time() - self.t0)
            try:
                self.episode.save_meta()
            except OSError:
                pass

    def write_sample(self, row: list) -> None:
        with self.lock:
            if self._csv_writer is None:
                return
            self._csv_writer.writerow(row)
            self.episode.n_samples += 1

    def write_frame(self, frame) -> None:
        if cv2 is None or not self.record_video or frame is None:
            return
        with self.lock:
            if self._video_writer is None:
                h, w = frame.shape[:2]
                self._video_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._video_writer = cv2.VideoWriter(
                    self.episode.video_path, fourcc, 30.0, (w, h)
                )
            elif self._video_size != (frame.shape[1], frame.shape[0]):
                return
            self._video_writer.write(frame)
            self.episode.n_frames += 1


# -----------------------------------------------------------------------------
# Time series buffer for plots
# -----------------------------------------------------------------------------

class TimeSeries:
    def __init__(self, maxlen: int = 600):
        self.t = deque(maxlen=maxlen)
        self.values: dict[str, deque] = {}
        self.maxlen = maxlen

    def push(self, t: float, **named_values: float) -> None:
        self.t.append(t)
        n_prev = len(self.t) - 1  # timestamps before this one
        for k, v in named_values.items():
            if k not in self.values:
                self.values[k] = deque([float("nan")] * n_prev, maxlen=self.maxlen)
            self.values[k].append(v)

    def array(self, key: str) -> np.ndarray:
        return np.array(self.values.get(key, []))

    def t_array(self) -> np.ndarray:
        return np.array(self.t)


# -----------------------------------------------------------------------------
# Studio
# -----------------------------------------------------------------------------

class Studio:
    POLL_MS = 50
    PLOT_MS = 100

    def __init__(self, default_port: str, dataset_root: str):
        self.default_port = default_port
        self.gripper: Optional[Gripper] = None
        self.camera: Optional[CameraWorker] = None
        self.orientation = OrientationTracker(beta=0.05)
        self.imu_replay: Optional[ImuReplay] = None
        self.recorder: Optional[EpisodeRecorder] = None
        self.dataset = DatasetManager(dataset_root)
        self.x3 = X3Control()
        self.x3.start()
        self._x3_devices: list[X3Device] = []
        self._selected_episode: Optional[Episode] = None

        self.series = TimeSeries(maxlen=600)
        self.jog = HoldJog()
        self._last_step = time.time()

        self.root = tk.Tk()
        self.root.title(f"{BRAND_NAME} · {BRAND_TAGLINE}")
        self.root.configure(bg=BG)
        self.root.geometry("1280x820")

        self.brand_family = _pick_brand_family(self.root)
        self._configure_styles()
        self._build_layout()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Schedule loops.
        self.root.after(self.POLL_MS, self._poll)
        self.root.after(self.PLOT_MS, self._redraw_plots)
        self.root.after(33, self._redraw_camera)

    # ---- styling ----------------------------------------------------------

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=INK, fieldbackground=PANEL)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Panel.TLabel", background=PANEL, foreground=INK)
        style.configure("Dim.TLabel", background=BG, foreground=DIM)
        style.configure("PanelDim.TLabel", background=PANEL, foreground=DIM)
        style.configure("Accent.TLabel", background=BG, foreground=ACCENT)
        style.configure("OK.TLabel", background=PANEL, foreground=OK)
        style.configure("Warn.TLabel", background=PANEL, foreground=WARN)
        style.configure(
            "TLabelframe", background=PANEL, foreground=DIM, borderwidth=0,
            relief="flat",
        )
        style.configure("TLabelframe.Label", background=PANEL, foreground=DIM)
        style.configure(
            "TButton", background=PANEL, foreground=INK, borderwidth=0,
            padding=6,
        )
        style.map("TButton", background=[("active", "#262a35")])
        style.configure("TEntry", fieldbackground=PANEL, foreground=INK,
                        insertcolor=INK, borderwidth=0)
        style.configure("TCheckbutton", background=PANEL, foreground=INK)
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG, foreground=DIM,
                        padding=(14, 6), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", PANEL)],
                  foreground=[("selected", INK)])
        style.configure("Horizontal.TScale", background=PANEL, troughcolor=BG,
                        sliderlength=32, sliderthickness=22)

    # ---- layout -----------------------------------------------------------

    def _build_layout(self) -> None:
        # Header
        header = ttk.Frame(self.root, style="TFrame")
        header.pack(fill="x", padx=18, pady=(14, 4))
        brand = tk.Label(
            header, text=BRAND_NAME, font=(self.brand_family, 44),
            fg=INK, bg=BG,
        )
        brand.pack(side="left")
        tag = tk.Label(
            header, text=BRAND_TAGLINE, font=(self.brand_family, 16),
            fg=DIM, bg=BG,
        )
        tag.pack(side="left", padx=(12, 0), pady=(20, 0))

        self._repo_dir = os.path.dirname(os.path.abspath(__file__))
        self._current_sha = umi_version.current_commit(self._repo_dir)
        version_text = f"v {umi_version.short(self._current_sha)}"
        ttk.Button(header, text="Update…", command=self._check_for_updates).pack(
            side="right", padx=(0, 0),
        )
        self.version_label = tk.Label(
            header, text=version_text, fg=DIM, bg=BG,
            font=(self.brand_family, 11),
        )
        self.version_label.pack(side="right", padx=(0, 12), pady=(22, 0))

        body = ttk.Frame(self.root, style="TFrame")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 14))
        body.columnconfigure(0, weight=0, minsize=380)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left_outer = ttk.Frame(body, style="TFrame")
        left_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left_canvas = tk.Canvas(
            left_outer, bg=BG, highlightthickness=0, width=360,
        )
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scroll = ttk.Scrollbar(
            left_outer, orient="vertical", command=left_canvas.yview,
        )
        left_scroll.pack(side="right", fill="y")
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left = ttk.Frame(left_canvas, style="TFrame")
        left_canvas.create_window((0, 0), window=left, anchor="nw", width=360)
        left.bind(
            "<Configure>",
            lambda _e: left_canvas.configure(scrollregion=left_canvas.bbox("all")),
        )

        self._build_connect(left)
        self._build_control(left)
        self._build_x3_panel(left)
        self._build_imu_panel(left)
        self._build_recording(left)
        self._build_failsafe(left)

        right = ttk.Notebook(body)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_servo_tab(right)
        self._build_slam_tab(right)
        self._build_dataset_tab(right)

        # Status bar
        self.status_var = tk.StringVar(value="ready.")
        status = tk.Label(
            self.root, textvariable=self.status_var, anchor="w",
            font=(self.brand_family, 11), fg=DIM, bg=BG, padx=18, pady=4,
        )
        status.pack(fill="x", side="bottom")

    def _build_connect(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Connection", padding=10)
        f.pack(fill="x", pady=(0, 10))
        self.port_var = tk.StringVar(value=self.default_port)
        self.baud_var = tk.IntVar(value=57600)
        self.id_a_var = tk.IntVar(value=1)
        self.id_b_var = tk.IntVar(value=2)
        self.home_a_var = tk.IntVar(value=2048)
        self.home_b_var = tk.IntVar(value=2048)
        self.mirror_var = tk.BooleanVar(value=True)

        # Always-visible: port + connect/scan
        ttk.Label(f, text="Port", style="PanelDim.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.port_var, width=28).grid(
            row=0, column=1, columnspan=3, sticky="ew", pady=2,
        )
        self.connect_btn = ttk.Button(f, text="Connect", command=self._do_connect)
        self.connect_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(f, text="Scan", command=self._do_scan).grid(
            row=1, column=2, columnspan=2, sticky="ew", pady=(6, 0), padx=(6, 0),
        )
        self.conn_status = ttk.Label(f, text="disconnected", style="PanelDim.TLabel")
        self.conn_status.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # Collapsible advanced settings (baud, IDs, home, mirror)
        adv_toggle = tk.Label(
            f, text="▸ Advanced", fg=DIM, bg=PANEL, cursor="hand2",
            font=(self.brand_family, 10),
        )
        adv_toggle.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))

        adv = ttk.Frame(f, style="TFrame")

        ttk.Label(adv, text="Baud", style="PanelDim.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(adv, textvariable=self.baud_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(adv, text="ID A/B", style="PanelDim.TLabel").grid(row=0, column=2, sticky="e")
        ttk.Entry(adv, textvariable=self.id_a_var, width=4).grid(row=0, column=3, sticky="w")
        ttk.Entry(adv, textvariable=self.id_b_var, width=4).grid(row=0, column=4, sticky="w")
        ttk.Label(adv, text="Home A/B", style="PanelDim.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(adv, textvariable=self.home_a_var, width=8).grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Entry(adv, textvariable=self.home_b_var, width=8).grid(row=1, column=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(adv, text="Mirror", variable=self.mirror_var).grid(
            row=1, column=3, columnspan=2, sticky="w", pady=(4, 0),
        )
        for c in range(5):
            adv.columnconfigure(c, weight=1)

        self._adv_conn_open = False

        def _toggle_adv(_e=None) -> None:
            self._adv_conn_open = not self._adv_conn_open
            if self._adv_conn_open:
                adv_toggle.configure(text="▾ Advanced")
                adv.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 0))
            else:
                adv_toggle.configure(text="▸ Advanced")
                adv.grid_remove()

        adv_toggle.bind("<Button-1>", _toggle_adv)

        for c in range(4):
            f.columnconfigure(c, weight=1)

    def _build_control(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Gripper", padding=10)
        f.pack(fill="x", pady=(0, 10))
        self.offset_var = tk.IntVar(value=0)
        self.offset_slider = ttk.Scale(
            f, from_=-600, to=600, orient="horizontal",
            variable=self.offset_var, command=self._on_slider,
        )
        self.offset_slider.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        self.offset_readout = ttk.Label(f, text="offset 0  (0.0°)", style="Panel.TLabel")
        self.offset_readout.grid(row=1, column=0, columnspan=4, sticky="w")

        ttk.Button(f, text="◀ Open", command=lambda: self._jog(-50)).grid(
            row=2, column=0, sticky="ew", pady=6, padx=(0, 4),
        )
        ttk.Button(f, text="Home", command=self._on_home).grid(
            row=2, column=1, sticky="ew", pady=6, padx=4,
        )
        ttk.Button(f, text="Close ▶", command=lambda: self._jog(50)).grid(
            row=2, column=2, sticky="ew", pady=6, padx=4,
        )
        self.torque_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Torque", variable=self.torque_var,
                        command=self._on_torque).grid(row=2, column=3, sticky="w", padx=(8, 0))

        ttk.Button(f, text="Zero (full open)", command=self._do_zero).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0), padx=(0, 4),
        )
        ttk.Label(f, text="Close°", style="PanelDim.TLabel").grid(row=3, column=2, sticky="e")
        self.close_angle_var = tk.DoubleVar(value=90.0)
        ttk.Entry(f, textvariable=self.close_angle_var, width=6).grid(
            row=3, column=3, sticky="w",
        )

        ttk.Label(
            f,
            text="Hold ←/→ (or ↑/↓) to jog, accelerates with hold time.",
            style="PanelDim.TLabel",
            wraplength=340,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))

        self.overload_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            f, text="Overload protection",
            variable=self.overload_var, command=self._on_overload,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(f, text="Reboot servos", command=self._do_reboot).grid(
            row=5, column=2, columnspan=2, sticky="ew", pady=(8, 0), padx=(4, 0),
        )
        for c in range(4):
            f.columnconfigure(c, weight=1)

    def _build_imu_panel(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="IMU", padding=10)
        f.pack(fill="x", pady=(0, 10))
        self.imu_status = ttk.Label(
            f, text="no IMU source", style="PanelDim.TLabel",
        )
        self.imu_status.grid(row=0, column=0, columnspan=2, sticky="w")
        self.imu_attitude = ttk.Label(
            f, text="roll —  pitch —  yaw —", style="Panel.TLabel",
        )
        self.imu_attitude.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 6))
        ttk.Button(f, text="Load .insv IMU…", command=self._do_load_imu_file).grid(
            row=2, column=0, sticky="ew", padx=(0, 4),
        )
        ttk.Button(f, text="Stop IMU replay", command=self._do_stop_imu).grid(
            row=2, column=1, sticky="ew", padx=(4, 0),
        )
        ttk.Label(
            f,
            text=("Live USB IMU isn't exposed by the X3 — use a recorded "
                  ".insv to replay gyro+accel, or wire an external IMU "
                  "via imu.LiveImuSource."),
            style="PanelDim.TLabel", wraplength=340,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

    def _build_x3_panel(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Insta360 X3  (Bluetooth)", padding=10)
        f.pack(fill="x", pady=(0, 10))
        self.x3_status = ttk.Label(f, text="not connected", style="PanelDim.TLabel")
        self.x3_status.grid(row=0, column=0, columnspan=2, sticky="w")
        self.x3_devices_var = tk.StringVar(value="")
        self.x3_devices_combo = ttk.Combobox(
            f, textvariable=self.x3_devices_var, state="readonly", values=[],
        )
        self.x3_devices_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(f, text="Scan", command=self._do_x3_scan).grid(
            row=2, column=0, sticky="ew", padx=(0, 4), pady=(6, 0),
        )
        self.x3_connect_btn = ttk.Button(f, text="Connect", command=self._do_x3_connect)
        self.x3_connect_btn.grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=(6, 0))
        self.x3_battery = ttk.Label(f, text="battery —", style="Panel.TLabel")
        self.x3_battery.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(f, text="● SD record", command=self._do_x3_start_record).grid(
            row=4, column=0, sticky="ew", padx=(0, 4), pady=(6, 0),
        )
        ttk.Button(f, text="■ SD stop", command=self._do_x3_stop_record).grid(
            row=4, column=1, sticky="ew", padx=(4, 0), pady=(6, 0),
        )
        ttk.Button(f, text="Inspect GATT…", command=self._do_x3_inspect).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0),
        )
        ttk.Label(
            f,
            text=("BLE runs alongside USB webcam mode. SD record/stop need "
                  "protocol bytes in x3_control.PROPRIETARY_COMMANDS — use "
                  "Inspect to discover service/characteristic UUIDs."),
            style="PanelDim.TLabel", wraplength=320,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

    def _build_recording(self, parent: ttk.Frame) -> None:
        f = ttk.LabelFrame(parent, text="Recording", padding=10)
        f.pack(fill="x", pady=(0, 10))
        ttk.Label(f, text="Dataset", style="PanelDim.TLabel").grid(row=0, column=0, sticky="w")
        self.rec_dir_var = tk.StringVar(value=self.dataset.root)
        ttk.Entry(f, textvariable=self.rec_dir_var).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(4, 0),
        )
        ttk.Button(f, text="…", width=2, command=self._pick_rec_dir).grid(
            row=0, column=3, sticky="w", padx=(4, 0),
        )
        ttk.Label(f, text="Task", style="PanelDim.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.rec_task_var = tk.StringVar(value="")
        ttk.Entry(f, textvariable=self.rec_task_var).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(4, 0), pady=(6, 0),
        )
        self.rec_video_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            f, text="Also save MP4 of camera feed", variable=self.rec_video_var,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.rec_btn = ttk.Button(f, text="● Start episode", command=self._toggle_record)
        self.rec_btn.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.rec_status = ttk.Label(f, text="not recording", style="PanelDim.TLabel")
        self.rec_status.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        for c in range(4):
            f.columnconfigure(c, weight=1)

    def _build_failsafe(self, parent: ttk.Frame) -> None:
        pass  # overload + reboot moved into _build_control

    def _build_servo_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, style="TFrame")
        nb.add(tab, text="  Servos  ")
        self.fig_servo = Figure(figsize=(7.6, 7.4), dpi=100, facecolor=BG)
        self.ax_pos = self.fig_servo.add_subplot(311)
        self.ax_cur = self.fig_servo.add_subplot(312, sharex=self.ax_pos)
        self.ax_tmp = self.fig_servo.add_subplot(313, sharex=self.ax_pos)
        for ax, title, ylabel in [
            (self.ax_pos, "position (deg)", "deg"),
            (self.ax_cur, "current (mA)", "mA"),
            (self.ax_tmp, "temperature (°C)", "°C"),
        ]:
            ax.set_facecolor(PANEL)
            ax.set_title(title, color=DIM, fontsize=10, loc="left")
            ax.set_ylabel(ylabel, color=DIM)
            ax.tick_params(colors=DIM)
            for spine in ax.spines.values():
                spine.set_color("#2a2f3a")
            ax.grid(True, color="#252934", linewidth=0.6)
        self.ax_tmp.set_xlabel("seconds", color=DIM)
        self.fig_servo.subplots_adjust(left=0.09, right=0.98, top=0.96,
                                       bottom=0.08, hspace=0.35)
        self.line_pos_a, = self.ax_pos.plot([], [], color=ACCENT, label="A")
        self.line_pos_b, = self.ax_pos.plot([], [], color="#ff7ab6", label="B")
        self.line_off, = self.ax_pos.plot([], [], color="#a3e635",
                                          linestyle="--", linewidth=0.8, label="offset")
        self.line_cur_a, = self.ax_cur.plot([], [], color=ACCENT)
        self.line_cur_b, = self.ax_cur.plot([], [], color="#ff7ab6")
        self.line_tmp_a, = self.ax_tmp.plot([], [], color=ACCENT)
        self.line_tmp_b, = self.ax_tmp.plot([], [], color="#ff7ab6")
        self.ax_pos.legend(loc="upper right", facecolor=PANEL,
                           edgecolor="#2a2f3a", labelcolor=INK, fontsize=8)
        self.canvas_servo = FigureCanvasTkAgg(self.fig_servo, master=tab)
        self.canvas_servo.get_tk_widget().pack(fill="both", expand=True)

    def _build_slam_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, style="TFrame")
        nb.add(tab, text="  Camera + SLAM  ")

        # Controls bar
        top = ttk.Frame(tab, style="TFrame")
        top.pack(fill="x", padx=8, pady=6)
        self.cam_index_var = tk.IntVar(value=0)
        self.crop_var = tk.StringVar(value="left")
        self.backend_var = tk.StringVar(value="OpenCV ORB VO")
        self.vocab_var = tk.StringVar(value="")
        self.settings_var = tk.StringVar(value="")
        ttk.Label(top, text="Cam index").pack(side="left")
        ttk.Entry(top, textvariable=self.cam_index_var, width=4).pack(
            side="left", padx=(4, 12),
        )
        ttk.Label(top, text="Lens").pack(side="left")
        ttk.Combobox(
            top, textvariable=self.crop_var,
            values=["left", "right", "full"], width=6, state="readonly",
        ).pack(side="left", padx=(4, 12))
        ttk.Label(top, text="Backend").pack(side="left")
        ttk.Combobox(
            top, textvariable=self.backend_var,
            values=["OpenCV ORB VO", "ORB-SLAM3"], width=18, state="readonly",
        ).pack(side="left", padx=(4, 12))
        self.cam_btn = ttk.Button(top, text="Start camera", command=self._toggle_camera)
        self.cam_btn.pack(side="left")
        self.cam_info = ttk.Label(tab, text="camera off", style="Dim.TLabel")
        self.cam_info.pack(fill="x", padx=8)

        # Vertical split: camera feed (top ~60%) + trajectory (bottom ~40%)
        paned = tk.PanedWindow(
            tab, orient="vertical", bg=BG,
            sashwidth=6, sashrelief="flat", sashpad=2,
        )
        paned.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        cam_pane = ttk.Frame(paned, style="TFrame")
        self.cam_canvas = tk.Label(cam_pane, bg=PANEL, text=" camera feed ",
                                   fg=DIM, font=(self.brand_family, 14))
        self.cam_canvas.pack(fill="both", expand=True)
        self._cam_imgtk = None  # keep ref so PhotoImage isn't GC'd
        paned.add(cam_pane, stretch="always", minsize=120)

        traj_pane = ttk.Frame(paned, style="TFrame")
        self.fig_traj = Figure(figsize=(7.4, 3.2), dpi=100, facecolor=BG)
        self.ax_traj = self.fig_traj.add_subplot(111, projection="3d")
        self.ax_traj.set_facecolor(PANEL)
        self.ax_traj.tick_params(colors=DIM)
        self.ax_traj.set_xlabel("x", color=DIM)
        self.ax_traj.set_ylabel("y", color=DIM)
        self.ax_traj.set_zlabel("z", color=DIM)
        self.ax_traj.set_title("camera trajectory  (monocular — relative scale)",
                               color=DIM, fontsize=10, loc="left")
        for axis in (self.ax_traj.xaxis, self.ax_traj.yaxis, self.ax_traj.zaxis):
            axis.set_pane_color((0.11, 0.12, 0.15, 1.0))
        self.line_traj, = self.ax_traj.plot([0], [0], [0], color=ACCENT, linewidth=1.4)
        self.scatter_curr = self.ax_traj.scatter([0], [0], [0], color=WARN, s=20)
        self.canvas_traj = FigureCanvasTkAgg(self.fig_traj, master=traj_pane)
        self.canvas_traj.get_tk_widget().pack(fill="both", expand=True)
        paned.add(traj_pane, stretch="always", minsize=80)

    def _build_trajectory_tab(self, nb: ttk.Notebook) -> None:
        pass  # merged into _build_slam_tab

    def _build_dataset_tab(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, style="TFrame")
        nb.add(tab, text="  Dataset  ")
        top = ttk.Frame(tab, style="TFrame")
        top.pack(fill="x", padx=8, pady=8)
        ttk.Button(top, text="Refresh", command=self._refresh_dataset).pack(side="left")
        ttk.Button(top, text="Reveal folder", command=self._reveal_dataset_root).pack(
            side="left", padx=(6, 0),
        )
        ttk.Button(top, text="Reveal episode", command=self._reveal_episode).pack(
            side="left", padx=(6, 0),
        )
        ttk.Button(top, text="Delete episode", command=self._delete_episode).pack(
            side="left", padx=(6, 0),
        )
        body = ttk.Frame(tab, style="TFrame")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)
        list_frame = ttk.Frame(body, style="TFrame")
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        cols = ("name", "task", "duration", "samples", "video")
        self.dataset_tree = ttk.Treeview(
            list_frame, columns=cols, show="headings", height=18,
        )
        for c, w in zip(cols, (180, 120, 70, 70, 50)):
            self.dataset_tree.heading(c, text=c.capitalize())
            self.dataset_tree.column(c, width=w, anchor="w")
        self.dataset_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical",
                           command=self.dataset_tree.yview)
        sb.pack(side="right", fill="y")
        self.dataset_tree.configure(yscrollcommand=sb.set)
        self.dataset_tree.bind("<<TreeviewSelect>>", self._on_episode_select)

        detail = ttk.Frame(body, style="TFrame")
        detail.grid(row=0, column=1, sticky="nsew")
        ttk.Label(detail, text="Episode", style="Dim.TLabel").pack(anchor="w")
        self.episode_meta = tk.Text(
            detail, height=8, bg=PANEL, fg=INK, insertbackground=INK,
            bd=0, relief="flat", font=("Menlo", 11),
        )
        self.episode_meta.pack(fill="x", pady=(4, 8))
        ttk.Label(detail, text="Task", style="Dim.TLabel").pack(anchor="w")
        self.episode_task_var = tk.StringVar(value="")
        ttk.Entry(detail, textvariable=self.episode_task_var).pack(fill="x", pady=(4, 8))
        ttk.Label(detail, text="Notes", style="Dim.TLabel").pack(anchor="w")
        self.episode_notes = tk.Text(
            detail, height=8, bg=PANEL, fg=INK, insertbackground=INK,
            bd=0, relief="flat", font=("Menlo", 11), wrap="word",
        )
        self.episode_notes.pack(fill="both", expand=True, pady=(4, 8))
        ttk.Button(detail, text="Save notes", command=self._save_episode_notes).pack(
            anchor="e",
        )
        self._refresh_dataset()

    # ---- key bindings -----------------------------------------------------

    def _bind_keys(self) -> None:
        for key, direction in (
            ("<KeyPress-Right>", +1), ("<KeyPress-Down>", +1),
            ("<KeyPress-Left>", -1), ("<KeyPress-Up>", -1),
        ):
            self.root.bind(key, lambda e, d=direction: self._on_key_press(e, d))
        for key in ("<KeyRelease-Right>", "<KeyRelease-Down>",
                    "<KeyRelease-Left>", "<KeyRelease-Up>"):
            self.root.bind(key, self._on_key_release)
        self.root.bind("<space>", lambda _e: self._on_home())

    def _on_key_press(self, event, direction: int) -> None:
        # Ignore key events that originate inside text-entry widgets
        # so users can type in the port/IDs without jogging the gripper.
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        self.jog.press(direction, event.keysym)

    def _on_key_release(self, event) -> None:
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        self.jog.release(event.keysym)

    # ---- connection -------------------------------------------------------

    def _do_connect(self) -> None:
        try:
            g = Gripper(
                port_name=self.port_var.get(),
                baud=self.baud_var.get(),
                id_a=self.id_a_var.get(),
                id_b=self.id_b_var.get(),
                home_a=self.home_a_var.get(),
                home_b=self.home_b_var.get(),
                mirror=self.mirror_var.get(),
            )
            g.connect()
            g.set_offset(0)
        except Exception as e:
            messagebox.showerror("Connect", str(e))
            return
        self.gripper = g
        self.conn_status.configure(text=f"connected  {g.id_a}/{g.id_b}",
                                   foreground=OK)
        self.connect_btn.configure(text="Disconnect", command=self._do_disconnect)
        self.offset_slider.configure(from_=g.open_limit, to=g.close_limit)
        self._set_status("servos connected.")

    def _do_disconnect(self) -> None:
        g = self.gripper
        self.gripper = None
        if g is not None:
            try:
                g.close()
            except Exception:
                pass
        self.conn_status.configure(text="disconnected", foreground=DIM)
        self.connect_btn.configure(text="Connect", command=self._do_connect)
        self._set_status("servos disconnected.")

    def _do_scan(self) -> None:
        packet = PacketHandler(PROTOCOL_VERSION)
        found = []
        for baud in SCAN_BAUDRATES:
            p = PortHandler(self.port_var.get())
            if not p.openPort():
                messagebox.showerror("Scan", f"cannot open {self.port_var.get()}")
                return
            if not p.setBaudRate(baud):
                p.closePort()
                continue
            for sid in SCAN_IDS:
                model, rc, err = packet.ping(p, sid)
                if rc == COMM_SUCCESS and err == 0:
                    found.append((baud, sid, model))
            p.closePort()
        if not found:
            messagebox.showwarning("Scan", "No servos found.")
            return
        if len(found) >= 2 and found[0][0] == found[1][0]:
            self.baud_var.set(found[0][0])
            self.id_a_var.set(found[0][1])
            self.id_b_var.set(found[1][1])
        msg = "\n".join(f"baud={b}  id={i}  model={m}" for b, i, m in found)
        messagebox.showinfo("Scan", msg)

    # ---- gripper actions --------------------------------------------------

    def _on_slider(self, _val) -> None:
        if self.gripper is None:
            return
        try:
            self.gripper.set_offset(int(self.offset_var.get()))
        except Exception as e:
            self._set_status(f"slider error: {e}")
        self._update_offset_readout()

    def _jog(self, delta: int) -> None:
        new_off = int(self.offset_var.get()) + delta
        if self.gripper is not None:
            new_off = max(self.gripper.open_limit,
                          min(self.gripper.close_limit, new_off))
        self.offset_var.set(new_off)
        self._on_slider(None)

    def _on_home(self) -> None:
        self.jog.velocity = 0.0
        self.offset_var.set(0)
        self._on_slider(None)

    def _on_torque(self) -> None:
        if self.gripper is None:
            return
        try:
            self.gripper.set_torque(self.torque_var.get())
        except Exception as e:
            messagebox.showerror("Torque", str(e))

    def _on_overload(self) -> None:
        if self.gripper is None:
            return
        enabled = self.overload_var.get()
        if not enabled and not messagebox.askokcancel(
            "Disable overload protection?",
            "Servos will no longer auto-shutdown on hard grips. Watch the\n"
            "live temperature — back off above ~70 °C.",
        ):
            self.overload_var.set(True)
            return
        try:
            self.gripper.set_overload_protection(enabled)
        except Exception as e:
            messagebox.showerror("Failsafe", str(e))

    def _do_zero(self) -> None:
        if self.gripper is None:
            messagebox.showinfo("Zero", "Connect first.")
            return
        try:
            angle = float(self.close_angle_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Zero", "Close angle must be a number.")
            return
        try:
            self.gripper.zero_at_current(close_angle_deg=angle)
        except Exception as e:
            messagebox.showerror("Zero", str(e))
            return
        self.home_a_var.set(self.gripper.home_a)
        self.home_b_var.set(self.gripper.home_b)
        self.offset_slider.configure(
            from_=self.gripper.open_limit, to=self.gripper.close_limit,
        )
        self.offset_var.set(0)
        self._on_slider(None)

    def _do_reboot(self) -> None:
        if self.gripper is None:
            messagebox.showinfo("Reboot", "Connect first.")
            return
        try:
            self.gripper.reboot()
        except Exception as e:
            messagebox.showerror("Reboot", str(e))
            return
        messagebox.showinfo("Reboot", "Servos rebooted. Reconnect or press Home.")

    # ---- camera / SLAM ----------------------------------------------------

    def _toggle_camera(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
            self.cam_btn.configure(text="Start camera")
            self.cam_info.configure(text="camera off")
            return
        if cv2 is None:
            messagebox.showerror(
                "Camera",
                "opencv-python isn't installed. Run:\n  pip install opencv-python",
            )
            return
        backend_name = self.backend_var.get()
        crop = self.crop_var.get()
        try:
            if backend_name == "ORB-SLAM3":
                if not self.vocab_var.get() or not self.settings_var.get():
                    messagebox.showinfo(
                        "ORB-SLAM3",
                        "Set vocab and settings paths via slam.OrbSlam3Backend.\n"
                        "Falling back to OpenCV VO.",
                    )
                    backend = VisualOdometry(VOConfig(crop=crop))
                else:
                    backend = OrbSlam3Backend(
                        vocab_path=self.vocab_var.get(),
                        settings_path=self.settings_var.get(),
                        crop=crop,
                    )
            else:
                backend = VisualOdometry(VOConfig(crop=crop))
        except Exception as e:
            messagebox.showerror("SLAM", str(e))
            return
        cam = CameraWorker(self.cam_index_var.get(), backend)
        if not cam.start():
            messagebox.showerror("Camera", cam.error or "failed to open camera")
            return
        cam.subscribe(self._on_pose)
        self.camera = cam
        self.cam_btn.configure(text="Stop camera")
        self._set_status(f"camera + {backend.name} running.")

    def _on_pose(self, pose) -> None:
        # called from camera worker thread — no Tk calls here.
        if self.recorder is not None and self.camera is not None:
            frame, _, _, _ = self.camera.snapshot()
            if frame is not None and self.rec_video_var.get():
                self.recorder.write_frame(frame)

    # ---- IMU --------------------------------------------------------------

    def _do_load_imu_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Insta360 recording",
            filetypes=[("Insta360 / video", "*.insv *.mp4 *.MP4 *.LRV"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            samples = Insta360FileLoader(path).load()
        except Exception as e:
            messagebox.showerror("IMU", str(e))
            return
        if not samples:
            messagebox.showwarning("IMU", "No IMU samples found in file.")
            return
        self._do_stop_imu()
        self.orientation.reset()
        self.imu_replay = ImuReplay(samples=samples, callback=self.orientation.update)
        self.imu_replay.start()
        self.imu_status.configure(
            text=f"replaying {len(samples)} IMU samples from {os.path.basename(path)}",
            foreground=ACCENT,
        )

    def _do_stop_imu(self) -> None:
        if self.imu_replay is not None:
            self.imu_replay.stop()
            self.imu_replay = None
            self.imu_status.configure(text="IMU replay stopped", foreground=DIM)

    # ---- recording --------------------------------------------------------

    def _pick_rec_dir(self) -> None:
        d = filedialog.askdirectory(initialdir=self.rec_dir_var.get() or os.path.expanduser("~"))
        if d:
            self.rec_dir_var.set(d)
            self.dataset = DatasetManager(d)
            self._refresh_dataset()

    # ---- X3 BLE -----------------------------------------------------------

    def _do_x3_scan(self) -> None:
        self.x3_status.configure(text="scanning…", foreground=DIM)
        self.root.update_idletasks()

        def work() -> None:
            devs = self.x3.scan(duration=4.0)
            self.root.after(0, lambda: self._x3_scan_done(devs))

        threading.Thread(target=work, daemon=True).start()

    def _x3_scan_done(self, devs: list[X3Device]) -> None:
        self._x3_devices = devs
        labels = [f"{d.name}  ({d.address})  RSSI {d.rssi}" for d in devs]
        self.x3_devices_combo.configure(values=labels)
        if labels:
            self.x3_devices_var.set(labels[0])
            self.x3_status.configure(
                text=f"found {len(devs)} camera(s)", foreground=ACCENT,
            )
        else:
            self.x3_status.configure(
                text=self.x3.last_error or "no Insta360 found", foreground=WARN,
            )

    def _do_x3_connect(self) -> None:
        if self.x3.connected:
            self.x3.disconnect()
            self.x3_status.configure(text="disconnected", foreground=DIM)
            self.x3_connect_btn.configure(text="Connect")
            return
        sel = self.x3_devices_var.get()
        if not sel or not self._x3_devices:
            messagebox.showinfo("X3", "Scan first and pick a camera.")
            return
        idx = self.x3_devices_combo.current()
        if idx < 0 or idx >= len(self._x3_devices):
            return
        dev = self._x3_devices[idx]
        self.x3_status.configure(text="connecting…", foreground=DIM)
        self.root.update_idletasks()

        def work() -> None:
            ok = self.x3.connect(dev.address, name=dev.name)
            self.root.after(0, lambda: self._x3_connect_done(ok, dev))

        threading.Thread(target=work, daemon=True).start()

    def _x3_connect_done(self, ok: bool, dev: X3Device) -> None:
        if ok:
            self.x3_status.configure(text=f"connected: {dev.name}", foreground=OK)
            self.x3_connect_btn.configure(text="Disconnect")
            self._poll_x3_battery()
        else:
            self.x3_status.configure(
                text=self.x3.last_error or "connect failed", foreground=WARN,
            )

    def _poll_x3_battery(self) -> None:
        if not self.x3.connected:
            return

        def work() -> None:
            pct = self.x3.get_battery()
            self.root.after(0, lambda: self._x3_battery_done(pct))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(15000, self._poll_x3_battery)

    def _x3_battery_done(self, pct: Optional[int]) -> None:
        if pct is None:
            self.x3_battery.configure(text="battery —", foreground=DIM)
        else:
            color = OK if pct > 30 else WARN
            self.x3_battery.configure(text=f"battery {pct}%", foreground=color)

    def _do_x3_start_record(self) -> None:
        if not self.x3.connected:
            messagebox.showinfo("X3", "Connect to the camera first.")
            return
        if not self.x3.start_sd_recording():
            messagebox.showwarning("X3", self.x3.last_error or "command failed")

    def _do_x3_stop_record(self) -> None:
        if not self.x3.connected:
            return
        if not self.x3.stop_sd_recording():
            messagebox.showwarning("X3", self.x3.last_error or "command failed")

    def _do_x3_inspect(self) -> None:
        if not self.x3.connected:
            messagebox.showinfo("X3", "Connect to the camera first.")
            return
        win = tk.Toplevel(self.root)
        win.title("X3 GATT inspector")
        win.configure(bg=BG)
        win.geometry("760x520")
        header = ttk.Frame(win, style="TFrame")
        header.pack(fill="x", padx=12, pady=(12, 6))
        tk.Label(
            header,
            text=f"GATT services for {self.x3.connected_to.name if self.x3.connected_to else 'camera'}",
            fg=INK, bg=BG, font=(self.brand_family, 14),
        ).pack(side="left")
        status = tk.Label(header, text="loading…", fg=DIM, bg=BG)
        status.pack(side="right")
        body = ttk.Frame(win, style="TFrame")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        text = tk.Text(
            body, bg=PANEL, fg=INK, insertbackground=INK, bd=0, relief="flat",
            font=("Menlo", 11), wrap="none",
        )
        text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(body, orient="vertical", command=text.yview)
        sb.pack(side="right", fill="y")
        text.configure(yscrollcommand=sb.set)
        footer = ttk.Frame(win, style="TFrame")
        footer.pack(fill="x", padx=12, pady=(0, 12))

        def copy_to_clipboard() -> None:
            self.root.clipboard_clear()
            self.root.clipboard_append(text.get("1.0", "end"))
            status.configure(text="copied to clipboard")

        ttk.Button(footer, text="Copy", command=copy_to_clipboard).pack(side="left")
        ttk.Button(footer, text="Close", command=win.destroy).pack(side="right")

        def work() -> None:
            chars = self.x3.list_characteristics()
            self.root.after(0, lambda: render(chars))

        def render(chars: list) -> None:
            if not chars:
                status.configure(text=self.x3.last_error or "no characteristics")
                return
            lines = []
            current_svc = None
            for svc, ch, props in chars:
                if svc != current_svc:
                    lines.append("")
                    lines.append(f"service  {svc}")
                    current_svc = svc
                props_str = ",".join(props)
                lines.append(f"  char  {ch}   [{props_str}]")
            text.insert("1.0", "\n".join(lines).lstrip())
            status.configure(text=f"{len(chars)} characteristic(s)")

        threading.Thread(target=work, daemon=True).start()

    # ---- updates ----------------------------------------------------------

    def _check_for_updates(self) -> None:
        self._set_status("checking for updates…")

        def work() -> None:
            latest = umi_version.latest_remote_commit()
            self.root.after(0, lambda: self._update_result(latest))

        threading.Thread(target=work, daemon=True).start()

    def _update_result(self, latest_sha: Optional[str]) -> None:
        current = self._current_sha
        if latest_sha is None:
            messagebox.showwarning(
                "Update",
                "Couldn't reach GitHub.\n"
                f"Repo: {umi_version.REPO_URL}",
            )
            self._set_status("update check failed.")
            return
        if current and latest_sha == current:
            messagebox.showinfo(
                "Update",
                f"Up to date.\n\nCurrent: {umi_version.short(current)}",
            )
            self._set_status("up to date.")
            return
        cur_s = umi_version.short(current)
        new_s = umi_version.short(latest_sha)
        if umi_version.is_dev_checkout(self._repo_dir):
            if not messagebox.askyesno(
                "Update available",
                f"New commit on main: {new_s}\n"
                f"You have:          {cur_s}\n\n"
                "Pull and relaunch now?",
            ):
                self._set_status("update declined.")
                return
            self._set_status("pulling update…")
            self.root.update_idletasks()

            def _do_pull() -> None:
                ok, msg = umi_version.pull(self._repo_dir)
                self.root.after(0, lambda: self._finish_pull(ok, msg))

            threading.Thread(target=_do_pull, daemon=True).start()
        else:
            if messagebox.askyesno(
                "Update available",
                f"New commit on main: {new_s}\n"
                f"This bundle:        {cur_s}\n\n"
                "Open the repo to pull the source and rerun build_app.sh?",
            ):
                import webbrowser
                webbrowser.open(umi_version.REPO_URL)
            self._set_status(f"newer version available ({new_s}).")

    def _finish_pull(self, ok: bool, msg: str) -> None:
        if not ok:
            messagebox.showerror("Update", msg)
            self._set_status("update failed.")
            return
        if "already up to date" in msg.lower():
            messagebox.showinfo("Update", "Already up to date.")
            self._set_status("up to date.")
            return
        import sys as _sys
        messagebox.showinfo("Updated", f"{msg}\n\nRelaunching now…")
        self.root.destroy()
        os.execv(_sys.executable, [_sys.executable, os.path.abspath(__file__)])

    # ---- dataset tab ------------------------------------------------------

    def _refresh_dataset(self) -> None:
        for iid in self.dataset_tree.get_children():
            self.dataset_tree.delete(iid)
        for ep in self.dataset.list_episodes():
            self.dataset_tree.insert(
                "", "end", iid=ep.path,
                values=(
                    ep.name, ep.task,
                    f"{ep.duration_s:.1f}s",
                    str(ep.n_samples),
                    "✓" if ep.has_video else "",
                ),
            )

    def _on_episode_select(self, _event=None) -> None:
        sel = self.dataset_tree.selection()
        if not sel:
            return
        path = sel[0]
        ep = Episode.load(path)
        self._selected_episode = ep
        self.episode_task_var.set(ep.task)
        self.episode_notes.delete("1.0", "end")
        self.episode_notes.insert("1.0", ep.notes)
        meta_lines = [
            f"path     : {ep.path}",
            f"started  : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ep.started_at or 0))}",
            f"duration : {ep.duration_s:.2f} s",
            f"samples  : {ep.n_samples}",
            f"frames   : {ep.n_frames}",
            f"slam     : {ep.slam_backend}",
            f"gripper  : {ep.gripper}",
        ]
        self.episode_meta.delete("1.0", "end")
        self.episode_meta.insert("1.0", "\n".join(meta_lines))

    def _save_episode_notes(self) -> None:
        ep = self._selected_episode
        if ep is None:
            return
        notes = self.episode_notes.get("1.0", "end").rstrip("\n")
        task = self.episode_task_var.get()
        try:
            self.dataset.update_notes(ep, notes=notes, task=task)
        except OSError as e:
            messagebox.showerror("Dataset", str(e))
            return
        self._set_status(f"saved notes for {ep.name}")
        self._refresh_dataset()

    def _reveal_dataset_root(self) -> None:
        self._open_in_finder(self.dataset.root)

    def _reveal_episode(self) -> None:
        ep = self._selected_episode
        if ep is None:
            return
        self._open_in_finder(ep.path)

    def _open_in_finder(self, path: str) -> None:
        import subprocess
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", path], check=False)
            elif sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
        except Exception as e:
            messagebox.showerror("Reveal", str(e))

    def _delete_episode(self) -> None:
        ep = self._selected_episode
        if ep is None:
            return
        if not messagebox.askyesno(
            "Delete episode",
            f"Delete {ep.name}? This removes the whole directory.",
        ):
            return
        try:
            self.dataset.delete(ep)
        except OSError as e:
            messagebox.showerror("Delete", str(e))
            return
        self._selected_episode = None
        self.episode_meta.delete("1.0", "end")
        self.episode_notes.delete("1.0", "end")
        self.episode_task_var.set("")
        self._refresh_dataset()

    def _toggle_record(self) -> None:
        if self.recorder is not None:
            self.recorder.stop()
            ep = self.recorder.episode
            ep.gripper = {
                "id_a": self.id_a_var.get(),
                "id_b": self.id_b_var.get(),
                "home_a": self.home_a_var.get(),
                "home_b": self.home_b_var.get(),
                "mirror": self.mirror_var.get(),
                "open_limit": self.gripper.open_limit if self.gripper else 0,
                "close_limit": self.gripper.close_limit if self.gripper else 0,
            }
            ep.slam_backend = self.backend_var.get() if self.camera else ""
            ep.save_meta()
            self.rec_status.configure(
                text=f"saved {ep.n_samples} samples, {ep.n_frames} frames → "
                     f"{ep.name}",
                foreground=OK,
            )
            self.recorder = None
            self.rec_btn.configure(text="● Start episode")
            self._refresh_dataset()
            return
        new_root = self.rec_dir_var.get()
        if new_root != self.dataset.root:
            self.dataset = DatasetManager(new_root)
        try:
            episode = self.dataset.new_episode(
                task=self.rec_task_var.get(),
                suffix=self.rec_task_var.get(),
            )
        except OSError as e:
            messagebox.showerror("Recording", str(e))
            return
        rec = EpisodeRecorder(
            episode=episode,
            record_video=self.rec_video_var.get() and cv2 is not None,
        )
        rec.start()
        self.recorder = rec
        self.rec_btn.configure(text="■ Stop episode")
        self.rec_status.configure(
            text=f"recording → {episode.name}", foreground=WARN,
        )

    # ---- main loops -------------------------------------------------------

    def _poll(self) -> None:
        now = time.time()
        dt = now - self._last_step
        self._last_step = now
        # Apply held-arrow jog.
        self.jog.reconcile(now)
        delta = self.jog.step(dt)
        if abs(delta) >= 1.0 and self.gripper is not None:
            new_off = int(self.offset_var.get()) + int(round(delta))
            new_off = max(self.gripper.open_limit,
                          min(self.gripper.close_limit, new_off))
            if new_off != int(self.offset_var.get()):
                self.offset_var.set(new_off)
                try:
                    self.gripper.set_offset(new_off)
                except Exception as e:
                    self._set_status(f"set_offset: {e}")

        self._update_offset_readout()

        # Read servo state.
        a_pos = a_cur = a_tmp = float("nan")
        b_pos = b_cur = b_tmp = float("nan")
        if self.gripper is not None:
            try:
                s = self.gripper.read_state()
                a, b = s[self.gripper.id_a], s[self.gripper.id_b]
                a_pos = a.position * TICK_TO_DEG
                b_pos = b.position * TICK_TO_DEG
                a_cur = a.current_ma
                b_cur = b.current_ma
                a_tmp = a.temperature_c
                b_tmp = b.temperature_c
            except Exception as e:
                self._set_status(f"read_state: {e}")

        # IMU readout.
        q = self.orientation.quaternion()
        roll, pitch, yaw = self.orientation.euler_deg()
        if self.imu_replay is not None:
            self.imu_attitude.configure(
                text=f"roll {roll:+6.1f}°  pitch {pitch:+6.1f}°  yaw {yaw:+6.1f}°"
            )

        # VO snapshot for the recorder.
        vx = vy = vz = float("nan")
        if self.camera is not None:
            _, pose, fps, n = self.camera.snapshot()
            if pose is not None:
                vx, vy, vz = float(pose.t[0]), float(pose.t[1]), float(pose.t[2])
            self.cam_info.configure(
                text=f"camera @ {fps:5.1f} fps · {n} frames · "
                     f"{self.backend_var.get()}  ·  err: {self.camera.error or '—'}"
            )

        elapsed = now - (self.recorder.t0 if self.recorder is not None else now)
        self.series.push(
            now,
            pos_a=a_pos, pos_b=b_pos,
            cur_a=a_cur, cur_b=b_cur,
            tmp_a=a_tmp, tmp_b=b_tmp,
            offset=int(self.offset_var.get()) * TICK_TO_DEG,
        )

        if self.recorder is not None:
            self.recorder.write_sample([
                now, elapsed,
                int(self.offset_var.get()),
                self.gripper.open_limit if self.gripper else 0,
                self.gripper.close_limit if self.gripper else 0,
                a_pos / TICK_TO_DEG if a_pos == a_pos else "",
                a_cur if a_cur == a_cur else "",
                a_tmp if a_tmp == a_tmp else "",
                b_pos / TICK_TO_DEG if b_pos == b_pos else "",
                b_cur if b_cur == b_cur else "",
                b_tmp if b_tmp == b_tmp else "",
                vx, vy, vz,
                q[0], q[1], q[2], q[3],
                roll, pitch, yaw,
            ])
            self.rec_status.configure(
                text=f"recording · {self.recorder.n_samples} samples · "
                     f"{self.recorder.n_frames} frames",
            )

        self.root.after(self.POLL_MS, self._poll)

    def _update_offset_readout(self) -> None:
        off = int(self.offset_var.get())
        self.offset_readout.configure(
            text=f"offset {off:+d}  ({off * TICK_TO_DEG:+6.1f}°)  "
                 f"v={self.jog.velocity:+6.0f} t/s",
        )

    def _redraw_plots(self) -> None:
        try:
            t = self.series.t_array()
            if len(t) > 1:
                t0 = t[0]
                xs = t - t0
                n = len(xs)
                self.line_pos_a.set_data(xs, self.series.array("pos_a")[:n])
                self.line_pos_b.set_data(xs, self.series.array("pos_b")[:n])
                self.line_off.set_data(xs, self.series.array("offset")[:n])
                self.line_cur_a.set_data(xs, self.series.array("cur_a")[:n])
                self.line_cur_b.set_data(xs, self.series.array("cur_b")[:n])
                self.line_tmp_a.set_data(xs, self.series.array("tmp_a")[:n])
                self.line_tmp_b.set_data(xs, self.series.array("tmp_b")[:n])
                for ax in (self.ax_pos, self.ax_cur, self.ax_tmp):
                    ax.relim()
                    ax.autoscale_view()
                self.canvas_servo.draw_idle()

            if self.camera is not None:
                traj = self.camera.backend.trajectory_xyz()
                if len(traj) > 1:
                    self.line_traj.set_data_3d(traj[:, 0], traj[:, 1], traj[:, 2])
                    self.scatter_curr._offsets3d = (
                        traj[-1:, 0], traj[-1:, 1], traj[-1:, 2],
                    )
                    lo = traj.min(axis=0)
                    hi = traj.max(axis=0)
                    pad = max(0.2, float((hi - lo).max()) * 0.1)
                    self.ax_traj.set_xlim(lo[0] - pad, hi[0] + pad)
                    self.ax_traj.set_ylim(lo[1] - pad, hi[1] + pad)
                    self.ax_traj.set_zlim(lo[2] - pad, hi[2] + pad)
                    self.canvas_traj.draw_idle()
        except Exception:
            pass
        finally:
            self.root.after(self.PLOT_MS, self._redraw_plots)

    def _redraw_camera(self) -> None:
        if (self.camera is not None and Image is not None
                and ImageTk is not None and cv2 is not None):
            frame, _, _, _ = self.camera.snapshot()
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                w = self.cam_canvas.winfo_width() or 640
                h = self.cam_canvas.winfo_height() or 360
                img = Image.fromarray(rgb)
                img.thumbnail((w, h))
                self._cam_imgtk = ImageTk.PhotoImage(img)
                self.cam_canvas.configure(image=self._cam_imgtk, text="")
        self.root.after(33, self._redraw_camera)

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    # ---- shutdown ---------------------------------------------------------

    def _on_close(self) -> None:
        if self.recorder is not None:
            self.recorder.stop()
        if self.camera is not None:
            self.camera.stop()
        if self.imu_replay is not None:
            self.imu_replay.stop()
        if self.gripper is not None:
            try:
                self.gripper.close()
            except Exception:
                pass
        try:
            self.x3.stop()
        except Exception:
            pass
        self.root.destroy()

    # ---- run --------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


def run_studio(default_port: str, dataset_root: str = "~/umi-data") -> int:
    studio = Studio(default_port=default_port, dataset_root=os.path.expanduser(dataset_root))
    studio.run()
    return 0
