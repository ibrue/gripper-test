"""Two-servo Dynamixel XL330 gripper controller.

Usage:
    python gripper.py scan [--port /dev/tty.usbserial-FTB8HK9X]
    python gripper.py run  [--port ...] [--baud 57600] ...
    python gripper.py gui  [--port ...]
"""

import argparse
import glob
import sys
import termios
import time
import tty
from dataclasses import dataclass

from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

PROTOCOL_VERSION = 2.0

# XL330-M288 / X-series Protocol 2.0 control table.
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4
LEN_STATE_BLOCK = 10  # current(2) + velocity(4) + position(4) starting at 126

OP_POSITION = 3

# XL330 reports current in 1 mA / LSB. XM430 ~2.69, XL430 ~2.69, XH540 ~2.69 —
# edit this if the hardware changes.
CURRENT_LSB_MA = 1.0

SCAN_BAUDRATES = [57600, 115200, 1000000, 2000000, 3000000, 4000000]
SCAN_IDS = range(1, 21)

FINE_STEP = 20
COARSE_STEP = 100
TICK_TO_DEG = 360.0 / 4096


def guess_default_port() -> str:
    for pat in ("/dev/tty.usbserial-*", "/dev/cu.usbserial-*", "/dev/ttyUSB*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return "/dev/tty.usbserial-FTB8HK9X"


def signed(value: int, bits: int) -> int:
    limit = 1 << bits
    return value - limit if value >= (limit >> 1) else value


@dataclass
class ServoState:
    position: int
    current_ma: float


class Gripper:
    def __init__(
        self,
        port_name: str,
        baud: int,
        id_a: int,
        id_b: int,
        home_a: int,
        home_b: int,
        mirror: bool,
    ):
        self.port_name = port_name
        self.baud = baud
        self.id_a = id_a
        self.id_b = id_b
        self.home_a = home_a
        self.home_b = home_b
        self.mirror = mirror
        self.offset = 0  # ticks from home; positive = close
        self.open_limit = -600
        self.close_limit = 600

        self.port = PortHandler(port_name)
        self.packet = PacketHandler(PROTOCOL_VERSION)
        self.sync_write = GroupSyncWrite(
            self.port, self.packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )
        self.sync_read = GroupSyncRead(
            self.port, self.packet, ADDR_PRESENT_CURRENT, LEN_STATE_BLOCK
        )

    def connect(self) -> None:
        if not self.port.openPort():
            raise RuntimeError(f"failed to open {self.port_name}")
        if not self.port.setBaudRate(self.baud):
            raise RuntimeError(f"failed to set baud {self.baud}")
        for sid in (self.id_a, self.id_b):
            model, rc, err = self.packet.ping(self.port, sid)
            if rc != COMM_SUCCESS or err != 0:
                raise RuntimeError(
                    f"ping ID {sid} failed: {self.packet.getTxRxResult(rc)} / {self.packet.getRxPacketError(err)}"
                )
            print(f"  ID {sid}: model {model}")
            # Torque must be OFF to change operating mode.
            self._write1(sid, ADDR_TORQUE_ENABLE, 0)
            self._write1(sid, ADDR_OPERATING_MODE, OP_POSITION)
            self._write1(sid, ADDR_TORQUE_ENABLE, 1)
            if not self.sync_read.addParam(sid):
                raise RuntimeError(f"sync_read addParam failed for ID {sid}")

    def close(self) -> None:
        for sid in (self.id_a, self.id_b):
            try:
                self._write1(sid, ADDR_TORQUE_ENABLE, 0)
            except Exception:
                pass
        self.port.closePort()

    def _write1(self, sid: int, addr: int, value: int) -> None:
        rc, err = self.packet.write1ByteTxRx(self.port, sid, addr, value)
        if rc != COMM_SUCCESS or err != 0:
            raise RuntimeError(
                f"write1 ID {sid} addr {addr}: {self.packet.getTxRxResult(rc)} / {self.packet.getRxPacketError(err)}"
            )

    def set_torque(self, on: bool) -> None:
        for sid in (self.id_a, self.id_b):
            self._write1(sid, ADDR_TORQUE_ENABLE, 1 if on else 0)

    def set_offset(self, offset: int) -> None:
        self.offset = max(self.open_limit, min(self.close_limit, offset))
        goal_a = self.home_a + self.offset
        goal_b = self.home_b + (-self.offset if self.mirror else self.offset)
        self.sync_write.clearParam()
        for sid, goal in ((self.id_a, goal_a), (self.id_b, goal_b)):
            goal &= 0xFFFFFFFF
            data = [
                goal & 0xFF,
                (goal >> 8) & 0xFF,
                (goal >> 16) & 0xFF,
                (goal >> 24) & 0xFF,
            ]
            if not self.sync_write.addParam(sid, bytes(data)):
                raise RuntimeError(f"sync_write addParam failed for ID {sid}")
        rc = self.sync_write.txPacket()
        if rc != COMM_SUCCESS:
            raise RuntimeError(f"sync_write tx: {self.packet.getTxRxResult(rc)}")

    def read_state(self) -> dict[int, ServoState]:
        rc = self.sync_read.txRxPacket()
        if rc != COMM_SUCCESS:
            raise RuntimeError(f"sync_read tx: {self.packet.getTxRxResult(rc)}")
        out: dict[int, ServoState] = {}
        for sid in (self.id_a, self.id_b):
            cur_raw = self.sync_read.getData(sid, ADDR_PRESENT_CURRENT, 2)
            pos_raw = self.sync_read.getData(sid, ADDR_PRESENT_POSITION, 4)
            out[sid] = ServoState(
                position=signed(pos_raw, 32),
                current_ma=signed(cur_raw, 16) * CURRENT_LSB_MA,
            )
        return out


def cmd_scan(args: argparse.Namespace) -> int:
    packet = PacketHandler(PROTOCOL_VERSION)
    found: list[tuple[int, int, int]] = []
    for baud in SCAN_BAUDRATES:
        port = PortHandler(args.port)
        if not port.openPort():
            print(f"cannot open {args.port}", file=sys.stderr)
            return 1
        if not port.setBaudRate(baud):
            port.closePort()
            continue
        print(f"scanning {baud} bps ...", flush=True)
        for sid in SCAN_IDS:
            model, rc, err = packet.ping(port, sid)
            if rc == COMM_SUCCESS and err == 0:
                found.append((baud, sid, model))
                print(f"  FOUND  baud={baud:<7} id={sid:<3} model={model}")
        port.closePort()

    print()
    if not found:
        print("no servos found.")
        return 1
    print(f"{len(found)} servo(s) found.")
    same_baud = {b for b, _, _ in found}
    if len(same_baud) == 1 and len(found) >= 2:
        b = found[0][0]
        ids = [sid for _, sid, _ in found]
        print(
            f"\nsuggested run command:\n"
            f"  python gripper.py run --port {args.port} --baud {b} "
            f"--id-a {ids[0]} --id-b {ids[1]}"
        )
    return 0


HELP_TEXT = """\
keys:
  o / O    open one fine / coarse step
  c / C    close one fine / coarse step
  space    go to open limit
  f        go to close limit
  h        return to home (offset 0)
  [        set current offset as new OPEN limit
  ]        set current offset as new CLOSE limit
  t        toggle torque (lets you backdrive by hand)
  r        reprint this help
  q        quit"""


def cbreak_loop(gripper: Gripper) -> None:
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    torque_on = True
    print(HELP_TEXT)
    print()
    try:
        tty.setcbreak(fd)
        gripper.set_offset(0)
        last_draw = 0.0
        while True:
            import select

            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            key = sys.stdin.read(1) if ready else ""

            if key == "q":
                break
            elif key == "o":
                gripper.set_offset(gripper.offset - FINE_STEP)
            elif key == "O":
                gripper.set_offset(gripper.offset - COARSE_STEP)
            elif key == "c":
                gripper.set_offset(gripper.offset + FINE_STEP)
            elif key == "C":
                gripper.set_offset(gripper.offset + COARSE_STEP)
            elif key == " ":
                gripper.set_offset(gripper.open_limit)
            elif key == "f":
                gripper.set_offset(gripper.close_limit)
            elif key == "h":
                gripper.set_offset(0)
            elif key == "[":
                gripper.open_limit = gripper.offset
                sys.stdout.write(f"\n[open limit set to {gripper.offset}]\n")
            elif key == "]":
                gripper.close_limit = gripper.offset
                sys.stdout.write(f"\n[close limit set to {gripper.offset}]\n")
            elif key == "t":
                torque_on = not torque_on
                gripper.set_torque(torque_on)
                sys.stdout.write(f"\n[torque {'on' if torque_on else 'off'}]\n")
            elif key == "r":
                sys.stdout.write("\n" + HELP_TEXT + "\n")

            now = time.time()
            if now - last_draw >= 0.05:
                last_draw = now
                state = gripper.read_state()
                a = state[gripper.id_a]
                b = state[gripper.id_b]
                sys.stdout.write(
                    f"\rpos A={a.position:>5} ({a.position * TICK_TO_DEG:6.1f}°)  "
                    f"B={b.position:>5} ({b.position * TICK_TO_DEG:6.1f}°)  "
                    f"I A={a.current_ma:+6.0f}mA B={b.current_ma:+6.0f}mA  "
                    f"off={gripper.offset:+4d} lim=[{gripper.open_limit}..{gripper.close_limit}]  "
                )
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        print()


def cmd_gui(args: argparse.Namespace) -> int:
    import os
    import subprocess
    import tkinter as tk
    from tkinter import messagebox, ttk

    repo_dir = os.path.dirname(os.path.abspath(__file__))

    root = tk.Tk()
    root.title("Dynamixel Gripper")
    root.resizable(False, False)

    state: dict = {"gripper": None, "poll_id": None}

    conn = ttk.LabelFrame(root, text="Connection", padding=10)
    conn.grid(row=0, column=0, sticky="ew", padx=10, pady=8)

    ttk.Label(conn, text="Port:").grid(row=0, column=0, sticky="w")
    port_var = tk.StringVar(value=args.port)
    ttk.Entry(conn, textvariable=port_var, width=34).grid(
        row=0, column=1, columnspan=5, sticky="ew", padx=4
    )

    ttk.Label(conn, text="Baud:").grid(row=1, column=0, sticky="w")
    baud_var = tk.IntVar(value=57600)
    ttk.Entry(conn, textvariable=baud_var, width=8).grid(row=1, column=1, sticky="w")
    ttk.Label(conn, text="ID A:").grid(row=1, column=2, sticky="e")
    id_a_var = tk.IntVar(value=1)
    ttk.Entry(conn, textvariable=id_a_var, width=4).grid(row=1, column=3, sticky="w")
    ttk.Label(conn, text="ID B:").grid(row=1, column=4, sticky="e")
    id_b_var = tk.IntVar(value=2)
    ttk.Entry(conn, textvariable=id_b_var, width=4).grid(row=1, column=5, sticky="w")

    ttk.Label(conn, text="Home A:").grid(row=2, column=0, sticky="w")
    home_a_var = tk.IntVar(value=2048)
    ttk.Entry(conn, textvariable=home_a_var, width=8).grid(row=2, column=1, sticky="w")
    ttk.Label(conn, text="Home B:").grid(row=2, column=2, sticky="e")
    home_b_var = tk.IntVar(value=2048)
    ttk.Entry(conn, textvariable=home_b_var, width=8).grid(row=2, column=3, sticky="w")
    mirror_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(conn, text="Mirror", variable=mirror_var).grid(
        row=2, column=4, columnspan=2, sticky="w"
    )

    connect_btn = ttk.Button(conn, text="Connect")
    connect_btn.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    scan_btn = ttk.Button(conn, text="Scan")
    scan_btn.grid(row=3, column=2, columnspan=2, sticky="ew", pady=(8, 0))
    update_btn = ttk.Button(conn, text="Check for updates")
    update_btn.grid(row=3, column=4, columnspan=2, sticky="ew", pady=(8, 0))
    conn_status = ttk.Label(conn, text="disconnected", foreground="gray")
    conn_status.grid(row=4, column=0, columnspan=6, sticky="w", pady=(4, 0))

    ctrl = ttk.LabelFrame(root, text="Control", padding=10)
    ctrl.grid(row=1, column=0, sticky="ew", padx=10, pady=8)

    ttk.Label(ctrl, text="Offset:").grid(row=0, column=0, sticky="w")
    offset_var = tk.IntVar(value=0)
    slider = ttk.Scale(
        ctrl, from_=-600, to=600, orient="horizontal", length=320, variable=offset_var
    )
    slider.grid(row=0, column=1, columnspan=4, sticky="ew", padx=6)
    offset_label = ttk.Label(ctrl, text="0", width=6)
    offset_label.grid(row=0, column=5, sticky="w")

    open_btn = ttk.Button(ctrl, text="◀ Open")
    open_btn.grid(row=1, column=0, padx=2, pady=6)
    home_btn = ttk.Button(ctrl, text="Home")
    home_btn.grid(row=1, column=1, padx=2, pady=6)
    close_btn = ttk.Button(ctrl, text="Close ▶")
    close_btn.grid(row=1, column=2, padx=2, pady=6)
    torque_var = tk.BooleanVar(value=True)
    torque_cb = ttk.Checkbutton(ctrl, text="Torque", variable=torque_var)
    torque_cb.grid(row=1, column=3, columnspan=2, padx=10)

    status = ttk.LabelFrame(root, text="Status", padding=10)
    status.grid(row=2, column=0, sticky="ew", padx=10, pady=(8, 12))
    mono = ("Menlo", 12)
    status_a = ttk.Label(status, text="Servo A:  —", font=mono)
    status_a.grid(row=0, column=0, sticky="w")
    status_b = ttk.Label(status, text="Servo B:  —", font=mono)
    status_b.grid(row=1, column=0, sticky="w")

    def stop_poll() -> None:
        if state["poll_id"] is not None:
            root.after_cancel(state["poll_id"])
            state["poll_id"] = None

    def poll() -> None:
        g = state["gripper"]
        if g is None:
            return
        try:
            s = g.read_state()
            a, b = s[g.id_a], s[g.id_b]
            status_a.config(
                text=f"Servo A:  pos={a.position:>5}  "
                f"({a.position * TICK_TO_DEG:6.1f}°)  I={a.current_ma:+6.0f} mA"
            )
            status_b.config(
                text=f"Servo B:  pos={b.position:>5}  "
                f"({b.position * TICK_TO_DEG:6.1f}°)  I={b.current_ma:+6.0f} mA"
            )
        except Exception as e:
            status_a.config(text=f"read error: {e}")
        state["poll_id"] = root.after(100, poll)

    def do_update() -> None:
        if not os.path.isdir(os.path.join(repo_dir, ".git")):
            messagebox.showerror(
                "Update", "Not a git checkout — can't pull updates here."
            )
            return
        try:
            res = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            messagebox.showerror("Update", "git pull timed out (network?).")
            return
        except FileNotFoundError:
            messagebox.showerror("Update", "git isn't installed.")
            return
        if res.returncode != 0:
            messagebox.showerror(
                "Update", (res.stderr or res.stdout or "git failed").strip()
            )
            return
        out = (res.stdout or "").strip()
        if "Already up to date" in out or not out:
            messagebox.showinfo("Update", "Already up to date.")
        else:
            messagebox.showinfo(
                "Update",
                f"{out}\n\nQuit and double-click Gripper.command again to run the new version.",
            )

    def do_scan() -> None:
        packet = PacketHandler(PROTOCOL_VERSION)
        found = []
        for baud in SCAN_BAUDRATES:
            p = PortHandler(port_var.get())
            if not p.openPort():
                messagebox.showerror("Scan", f"cannot open {port_var.get()}")
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
            baud_var.set(found[0][0])
            id_a_var.set(found[0][1])
            id_b_var.set(found[1][1])
        msg = "\n".join(f"baud={b}  id={i}  model={m}" for b, i, m in found)
        messagebox.showinfo("Scan", msg)

    def do_connect() -> None:
        try:
            g = Gripper(
                port_name=port_var.get(),
                baud=baud_var.get(),
                id_a=id_a_var.get(),
                id_b=id_b_var.get(),
                home_a=home_a_var.get(),
                home_b=home_b_var.get(),
                mirror=mirror_var.get(),
            )
            g.connect()
            g.set_offset(0)
        except Exception as e:
            messagebox.showerror("Connect", str(e))
            return
        state["gripper"] = g
        conn_status.config(
            text=f"connected  {id_a_var.get()}/{id_b_var.get()}", foreground="#2a8a2a"
        )
        connect_btn.config(text="Disconnect", command=do_disconnect)
        poll()

    def do_disconnect() -> None:
        stop_poll()
        g = state["gripper"]
        if g is not None:
            try:
                g.close()
            except Exception:
                pass
        state["gripper"] = None
        conn_status.config(text="disconnected", foreground="gray")
        connect_btn.config(text="Connect", command=do_connect)
        status_a.config(text="Servo A:  —")
        status_b.config(text="Servo B:  —")

    def apply_offset() -> None:
        g = state["gripper"]
        if g is None:
            return
        try:
            g.set_offset(int(offset_var.get()))
        except Exception as e:
            messagebox.showerror("Move", str(e))

    def on_slider(_: str) -> None:
        offset_label.config(text=str(int(offset_var.get())))
        apply_offset()

    def jog(delta: int) -> None:
        offset_var.set(max(-600, min(600, int(offset_var.get()) + delta)))
        on_slider("")

    def on_home() -> None:
        offset_var.set(0)
        on_slider("")

    def on_torque_toggle(*_) -> None:
        g = state["gripper"]
        if g is None:
            return
        try:
            g.set_torque(torque_var.get())
        except Exception as e:
            messagebox.showerror("Torque", str(e))

    def on_close_window() -> None:
        do_disconnect()
        root.destroy()

    connect_btn.config(command=do_connect)
    scan_btn.config(command=do_scan)
    update_btn.config(command=do_update)
    open_btn.config(command=lambda: jog(-FINE_STEP))
    home_btn.config(command=on_home)
    close_btn.config(command=lambda: jog(FINE_STEP))
    slider.config(command=on_slider)
    torque_var.trace_add("write", on_torque_toggle)
    root.bind("<Left>", lambda _e: jog(-FINE_STEP))
    root.bind("<Right>", lambda _e: jog(FINE_STEP))
    root.bind("<Up>", lambda _e: on_home())
    root.protocol("WM_DELETE_WINDOW", on_close_window)

    root.mainloop()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    gripper = Gripper(
        port_name=args.port,
        baud=args.baud,
        id_a=args.id_a,
        id_b=args.id_b,
        home_a=args.home_a,
        home_b=args.home_b,
        mirror=args.mirror,
    )
    print(f"connecting to {args.port} @ {args.baud} ...")
    gripper.connect()
    print("connected. torque enabled, operating mode = position.\n")
    try:
        cbreak_loop(gripper)
    except KeyboardInterrupt:
        pass
    finally:
        gripper.close()
        print("torque off, port closed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    default_port = guess_default_port()

    p_scan = sub.add_parser("scan", help="find connected servos across baudrates")
    p_scan.add_argument("--port", default=default_port)
    p_scan.set_defaults(func=cmd_scan)

    p_run = sub.add_parser("run", help="interactive keyboard gripper control")
    p_run.add_argument("--port", default=default_port)
    p_run.add_argument("--baud", type=int, default=57600)
    p_run.add_argument("--id-a", type=int, default=1)
    p_run.add_argument("--id-b", type=int, default=2)
    p_run.add_argument("--home-a", type=int, default=2048)
    p_run.add_argument("--home-b", type=int, default=2048)
    mir = p_run.add_mutually_exclusive_group()
    mir.add_argument("--mirror", dest="mirror", action="store_true", default=True)
    mir.add_argument("--no-mirror", dest="mirror", action="store_false")
    p_run.set_defaults(func=cmd_run)

    p_gui = sub.add_parser("gui", help="tkinter GUI with buttons and live readout")
    p_gui.add_argument("--port", default=default_port)
    p_gui.set_defaults(func=cmd_gui)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
