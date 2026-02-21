"""
ble_connector.py -- BLE connection manager for BerryMed pulse oximeters.

Uses the `bleak` library to scan, connect, and stream data from Berry
devices.  Runs the async event loop in a background thread and pushes
decoded BerryPacket objects into a thread-safe queue for the GUI.
"""

import asyncio
import threading
import queue
import time
from typing import Callable

try:
    from bleak import BleakScanner, BleakClient
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False

import config as cfg
from berry_protocol import decode_packet, decode_version, BerryPacket


# ─── Connection States ──────────────────────────────────────────────
STATE_IDLE = "idle"
STATE_SCANNING = "scanning"
STATE_CONNECTING = "connecting"
STATE_CONNECTED = "connected"
STATE_DISCONNECTED = "disconnected"
STATE_ERROR = "error"


class BerryBLEConnector:
    """
    Manages BLE communication with a BerryMed pulse oximeter.

    The connector uses a background thread for the asyncio event loop.
    Decoded packets are placed into `self.packet_queue` for the GUI
    thread to consume.
    """

    def __init__(self):
        self.packet_queue: queue.Queue[BerryPacket] = queue.Queue(maxsize=2000)
        self.state = STATE_IDLE
        self.state_callback: Callable[[str], None] | None = None
        self.version_info: str = ""

        self._client: "BleakClient | None" = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._devices: list = []
        self._stop_event = threading.Event()
        self._buffer = bytearray()  # Reassembly buffer

        # Stats
        self.packets_received = 0
        self.packets_bad = 0

    # ─── Public API ──────────────────────────────────────────────────

    def scan(self, duration: float = 5.0, callback: Callable | None = None):
        """
        Scan for BerryMed devices in a background thread.

        Parameters
        ----------
        duration : float
            Scan duration in seconds.
        callback : callable
            Called with list of (name, address) tuples when scan completes.
        """
        if not BLEAK_AVAILABLE:
            self._set_state(STATE_ERROR)
            if callback:
                callback([])
            return

        self._set_state(STATE_SCANNING)

        def _scan_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                devices = loop.run_until_complete(self._async_scan(duration))
                self._devices = devices
                result = [(d.name or "Unknown", d.address) for d in devices]
                self._set_state(STATE_IDLE)
                if callback:
                    callback(result)
            except Exception as e:
                self._set_state(STATE_ERROR)
                if callback:
                    callback([])
            finally:
                loop.close()

        t = threading.Thread(target=_scan_thread, daemon=True)
        t.start()

    def connect(self, address: str):
        """
        Connect to a Berry device by address.
        Starts notification subscription in a background thread.
        """
        if not BLEAK_AVAILABLE:
            self._set_state(STATE_ERROR)
            return

        self._stop_event.clear()
        self._set_state(STATE_CONNECTING)

        self._thread = threading.Thread(
            target=self._run_loop, args=(address,), daemon=True
        )
        self._thread.start()

    def disconnect(self):
        """Disconnect from the device."""
        self._stop_event.set()
        self._set_state(STATE_DISCONNECTED)

    def send_command(self, cmd: int):
        """Send a 1-byte command to the device."""
        if self._client and self._loop and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._async_send(cmd), self._loop
                )
            except RuntimeError:
                pass

    def set_rate(self, hz: int):
        """Set the packet rate (50, 100, or 200 Hz)."""
        cmd_map = {
            1: cfg.BERRY_CMD_1HZ,
            50: cfg.BERRY_CMD_50HZ,
            100: cfg.BERRY_CMD_100HZ,
            200: cfg.BERRY_CMD_200HZ,
        }
        if hz in cmd_map:
            self.send_command(cmd_map[hz])

    # ─── Internal Async Methods ──────────────────────────────────────

    async def _async_scan(self, duration: float):
        """Scan for nearby BLE devices.

        Note: We do NOT filter by service_uuids because many BerryMed
        devices (especially on Windows) do not advertise their service
        UUIDs in the BLE advertisement packets.  Instead we return all
        discovered devices so the user can pick the correct one.
        """
        devices = await BleakScanner.discover(timeout=duration)
        # Optionally sort Berry-named devices to the top
        berry_first = sorted(
            devices,
            key=lambda d: (0 if d.name and "berry" in d.name.lower() else 1),
        )
        return berry_first

    async def _async_connect(self, address: str):
        """Connect and subscribe to notifications."""
        self._client = BleakClient(
            address,
            disconnected_callback=self._on_disconnect,
        )
        await self._client.connect()

        if self._client.is_connected:
            self._set_state(STATE_CONNECTED)

            # Set desired packet rate
            await self._async_send(cfg.BERRY_CMD_200HZ)
            await asyncio.sleep(0.1)

            # Subscribe to notifications
            await self._client.start_notify(
                cfg.BERRY_SEND_CHAR_UUID, self._notification_handler
            )

            # Request version info
            await self._async_send(cfg.BERRY_CMD_SW_VERSION)

            # Keep running until stop
            # We cycle commands to keep the device awake (bypass 40s sleep timer)
            last_ping = time.time()
            ping_tiggle = False
            
            while not self._stop_event.is_set() and self._client.is_connected:
                if time.time() - last_ping > 1.5:
                    # Send command to reset inactivity timer
                    cmd = cfg.BERRY_CMD_200HZ if ping_tiggle else cfg.BERRY_CMD_ADC_ORIGINAL
                    await self._async_send(cmd)
                    
                    ping_tiggle = not ping_tiggle
                    last_ping = time.time()
                    
                await asyncio.sleep(0.1)

            # Cleanup
            try:
                if self._client.is_connected:
                    await self._client.stop_notify(cfg.BERRY_SEND_CHAR_UUID)
                    await self._client.disconnect()
            except Exception:
                pass
        else:
            self._set_state(STATE_ERROR)

    async def _async_send(self, cmd: int):
        """Write a single-byte command to the device."""
        if self._client and self._client.is_connected:
            try:
                await self._client.write_gatt_char(
                    cfg.BERRY_RECV_CHAR_UUID, bytes([cmd])
                )
            except Exception:
                pass

    def _notification_handler(self, sender, data: bytearray):
        """Handle incoming BLE notifications."""
        # Add to reassembly buffer
        self._buffer.extend(data)

        # Process complete packets
        while len(self._buffer) >= cfg.BERRY_PACKET_SIZE:
            # Find header
            header_pos = -1
            for i in range(len(self._buffer) - 1):
                if self._buffer[i] == 0xFF and self._buffer[i + 1] == 0xAA:
                    header_pos = i
                    break

            if header_pos < 0:
                # No header found, discard
                self._buffer.clear()
                break

            # Discard bytes before header
            if header_pos > 0:
                del self._buffer[:header_pos]

            # Need at least 20 bytes from header
            if len(self._buffer) < cfg.BERRY_PACKET_SIZE:
                break

            # Extract packet
            raw = bytes(self._buffer[:cfg.BERRY_PACKET_SIZE])
            del self._buffer[:cfg.BERRY_PACKET_SIZE]

            # Check for version response
            ver = decode_version(raw)
            if ver:
                self.version_info = ver
                continue

            # Decode data packet
            pkt = decode_packet(raw)
            if pkt:
                self.packets_received += 1
                try:
                    self.packet_queue.put_nowait(pkt)
                except queue.Full:
                    # Drop oldest if queue is full
                    try:
                        self.packet_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.packet_queue.put_nowait(pkt)
            else:
                self.packets_bad += 1

    def _on_disconnect(self, client):
        """BLE disconnection callback."""
        # Note: Do not switch to DISCONNECTED here if we are auto-reconnecting.
        # The main loop will handle the state transition when it catches the disconnect.
        pass

    def _run_loop(self, address: str):
        """Background thread: run the asyncio event loop with auto-reconnect."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        try:
            # Auto-reconnect loop: keep trying until user explicitly disconnects
            while not self._stop_event.is_set():
                try:
                    # Try to communicate
                    loop.run_until_complete(self._async_connect(address))
                except Exception as e:
                    # If start failed, log it, but we retry
                    pass
                
                # If we are here, _async_connect returned (disconnect or failure)
                # If user did not ask to stop, wait and retry (auto-reconnect)
                if not self._stop_event.is_set():
                    self._set_state(STATE_CONNECTING)
                    time.sleep(1.0) # Wait before retry

        finally:
            # Safely shut down
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            finally:
                if not loop.is_closed():
                    loop.close()
                if self._loop is loop:
                    self._loop = None
            
            # Now we are truly done
            self._set_state(STATE_DISCONNECTED)

    def _set_state(self, new_state: str):
        """Update connection state and fire callback."""
        self.state = new_state
        if self.state_callback:
            self.state_callback(new_state)
