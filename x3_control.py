"""Insta360 X3 BLE control.

The X3's Bluetooth radio works in parallel with USB webcam mode, so you can
stream video over USB for live SLAM and issue commands over BLE for things
USB doesn't expose (start/stop on-card recording, change ISO, switch lens,
read battery).

What this module gives you:

* Cross-platform async BLE scan + connect via ``bleak``. Connection lives
  on a dedicated asyncio worker thread; the Tk GUI talks to it via a
  thin sync API (no awaits in your event handler).
* Standard Bluetooth Battery Service read (UUID 0x180F / 0x2A19). Most
  cameras expose this and it Just Works.
* Hooks for proprietary commands (start/stop record, set ISO, ...).
  **The byte sequences are NOT included.** They have to be supplied by
  you in ``PROPRIETARY_COMMANDS``. Why: Insta360 doesn't publish a
  documented protocol, and the bytes vary across firmware revisions —
  hardcoding numbers I haven't verified would silently break or
  worse. To get them:

    - crib from a community library, e.g. ``insta360-bluetooth`` (JS),
      various forks tagged "insta360" on GitHub, or
    - sniff your camera with nRF Connect (Android/iOS) while
      controlling it from the Insta360 mobile app.

  Once you have the (characteristic_uuid, bytes) for each command,
  drop them into ``PROPRIETARY_COMMANDS`` and the studio buttons
  light up.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Optional


BATTERY_SERVICE = "0000180f-0000-1000-8000-00805f9b34fb"
BATTERY_LEVEL_CHAR = "00002a19-0000-1000-8000-00805f9b34fb"
INSTA360_NAME_PREFIX = "Insta360"


@dataclass
class X3Device:
    name: str
    address: str
    rssi: int = 0


# Fill this in with bytes you've verified for your firmware. Schema:
#   key:   command name used by the public API ("start_recording" etc.)
#   value: (characteristic_uuid, payload_bytes)
PROPRIETARY_COMMANDS: dict[str, tuple[str, bytes]] = {
    # "start_recording": ("0000fff1-0000-1000-8000-00805f9b34fb", b"..."),
    # "stop_recording":  ("0000fff1-0000-1000-8000-00805f9b34fb", b"..."),
}


class X3Control:
    """Synchronous façade over a background asyncio + bleak client."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client = None
        self.connected: bool = False
        self.connected_to: Optional[X3Device] = None
        self.last_error: Optional[str] = None

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()

        def runner() -> None:
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        ready.wait(timeout=2.0)

    def stop(self) -> None:
        if self._loop is None:
            return
        if self.connected:
            try:
                self._run_sync(self._async_disconnect(), timeout=5)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._loop = None
        self._thread = None

    def _run_sync(self, coro, timeout: float = 10.0):
        if self._loop is None:
            raise RuntimeError("X3Control not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ---- scan ------------------------------------------------------------

    def scan(self, duration: float = 4.0) -> list[X3Device]:
        try:
            return self._run_sync(self._async_scan(duration), timeout=duration + 5)
        except Exception as e:
            self.last_error = f"scan: {e}"
            return []

    async def _async_scan(self, duration: float) -> list[X3Device]:
        try:
            from bleak import BleakScanner
        except ImportError as e:
            raise RuntimeError("bleak not installed (pip install bleak)") from e
        devices = await BleakScanner.discover(timeout=duration)
        out: list[X3Device] = []
        for d in devices:
            name = d.name or ""
            if name.startswith(INSTA360_NAME_PREFIX):
                rssi = int(getattr(d, "rssi", 0) or 0)
                out.append(X3Device(name=name, address=d.address, rssi=rssi))
        return out

    # ---- connect / disconnect -------------------------------------------

    def connect(self, address: str, name: str = "") -> bool:
        try:
            self._run_sync(self._async_connect(address, name), timeout=15)
            return self.connected
        except Exception as e:
            self.last_error = f"connect: {e}"
            return False

    async def _async_connect(self, address: str, name: str) -> None:
        from bleak import BleakClient
        self._client = BleakClient(address)
        await self._client.connect()
        self.connected = bool(getattr(self._client, "is_connected", True))
        self.connected_to = X3Device(name=name, address=address) if self.connected else None

    def disconnect(self) -> None:
        try:
            self._run_sync(self._async_disconnect(), timeout=5)
        except Exception as e:
            self.last_error = f"disconnect: {e}"

    async def _async_disconnect(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
        self._client = None
        self.connected = False
        self.connected_to = None

    # ---- generic operations ---------------------------------------------

    def get_battery(self) -> Optional[int]:
        if not self.connected:
            return None
        try:
            return self._run_sync(self._async_battery(), timeout=5)
        except Exception as e:
            self.last_error = f"battery: {e}"
            return None

    async def _async_battery(self) -> Optional[int]:
        if self._client is None:
            return None
        data = await self._client.read_gatt_char(BATTERY_LEVEL_CHAR)
        return int(data[0]) if data else None

    def list_characteristics(self) -> list[tuple[str, str, list[str]]]:
        """Return (service_uuid, char_uuid, properties) for every GATT
        characteristic on the connected device. Handy for protocol RE."""
        if not self.connected:
            return []
        try:
            return self._run_sync(self._async_list_chars(), timeout=10)
        except Exception as e:
            self.last_error = f"list_chars: {e}"
            return []

    async def _async_list_chars(self) -> list[tuple[str, str, list[str]]]:
        out: list[tuple[str, str, list[str]]] = []
        if self._client is None:
            return out
        services = self._client.services
        for svc in services:
            for ch in svc.characteristics:
                out.append((str(svc.uuid), str(ch.uuid), list(ch.properties)))
        return out

    # ---- proprietary commands (require PROPRIETARY_COMMANDS entries) ----

    def start_sd_recording(self) -> bool:
        return self._send_proprietary("start_recording")

    def stop_sd_recording(self) -> bool:
        return self._send_proprietary("stop_recording")

    def set_iso(self, iso: int) -> bool:
        return self._send_proprietary(f"iso_{iso}")

    def _send_proprietary(self, name: str) -> bool:
        if not self.connected:
            self.last_error = "not connected"
            return False
        spec = PROPRIETARY_COMMANDS.get(name)
        if spec is None:
            self.last_error = (
                f"command '{name}' not configured — fill in "
                f"x3_control.PROPRIETARY_COMMANDS (see module docstring)."
            )
            return False
        try:
            self._run_sync(self._async_write(*spec), timeout=5)
            return True
        except Exception as e:
            self.last_error = f"{name}: {e}"
            return False

    async def _async_write(self, char_uuid: str, data: bytes) -> None:
        if self._client is None:
            return
        await self._client.write_gatt_char(char_uuid, data, response=True)
