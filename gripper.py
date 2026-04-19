"""Two-servo Dynamixel XL330 gripper controller.

Usage:
    python gripper.py scan [--port /dev/tty.usbserial-FTB8HK9X]
    python gripper.py run  [--port ...] [--baud 57600]
                           [--id-a 1] [--id-b 2]
                           [--mirror / --no-mirror]
                           [--home-a 2048] [--home-b 2048]
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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
