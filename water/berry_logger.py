"""
berry_logger.py — Standalone BerryMed raw packet debugger.

Connects to the watch, dumps every BLE notification as:
  1. A timestamped HEX dump of raw bytes (berry_raw_<timestamp>.log)
  2. A CSV of all decoded fields + ADC per packet (berry_packets_<timestamp>.csv)

Usage:
    python berry_logger.py                  # auto-scan & pick first Berry device
    python berry_logger.py --addr AA:BB:... # connect directly by address
    python berry_logger.py --scan-only      # just print discovered devices
    python berry_logger.py --duration 60    # log for 60 seconds then stop (default: until Ctrl+C)
"""

import asyncio
import argparse
import csv
import os
import struct
import sys
import time
from datetime import datetime

try:
    from bleak import BleakScanner, BleakClient
except ImportError:
    print("ERROR: bleak not installed. Run: pip install bleak")
    sys.exit(1)

# ─── BerryMed UUIDs (from config.py) ────────────────────────────────
SERVICE_UUID  = "49535343-FE7D-4AE5-8FA9-9FAFD205E455"
SEND_UUID     = "49535343-1E4D-4BD9-BA61-23C647249616"   # Notify (device → PC)
RECV_UUID     = "49535343-8841-43F4-A8D4-ECBE34729BB3"   # Write  (PC → device)

CMD_200HZ     = 0xF2
CMD_ADC_ORIG  = 0xF4
CMD_SW_VER    = 0xFF
PACKET_SIZE   = 20

# ─── Output Directory ────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

ts_str      = datetime.now().strftime("%Y%m%d_%H%M%S")
RAW_LOG     = os.path.join(LOG_DIR, f"berry_raw_{ts_str}.log")
CSV_LOG     = os.path.join(LOG_DIR, f"berry_packets_{ts_str}.csv")


# ─── CSV Header ─────────────────────────────────────────────────────
CSV_FIELDS = [
    "wall_time", "rx_timestamp", "pkt_num",
    "pkt_index",                        # byte 2: sequence 0-255
    "status_raw", "status_text",        # byte 3
    "spo2_avg", "spo2_real",            # bytes 4,5
    "hr_avg", "hr_real",                # bytes 6,7
    "rr_raw", "rr_ms",                  # bytes 8-9
    "pi_avg_permil", "pi_real_permil",  # bytes 10,11
    "pi_pct",
    "pleth",                            # byte 12
    "adc_sample",                       # bytes 13-16 (int32 LE)
    "battery",                          # byte 17
    "freq_hz",                          # byte 18
    "checksum_byte",                    # byte 19
    "raw_hex",                          # full 20-byte hex
]

STATUS_SENSOR_OFF = 0x01
STATUS_NO_FINGER  = 0x02
STATUS_NO_PULSE   = 0x04
STATUS_PULSE_BEAT = 0x08


# ─── State ───────────────────────────────────────────────────────────
pkt_count    = 0
raw_buf      = bytearray()
stop_after   = None   # set from --duration
start_time   = None

raw_log_fh   = None
csv_writer   = None
csv_fh       = None


def status_text(s):
    parts = []
    if s & STATUS_SENSOR_OFF: parts.append("SensorOff")
    if s & STATUS_NO_FINGER:  parts.append("NoFinger")
    if s & STATUS_NO_PULSE:   parts.append("NoPulse")
    if s & STATUS_PULSE_BEAT: parts.append("Beat")
    return "|".join(parts) if parts else "OK"


def decode_and_log(raw: bytes, rx_ts: float):
    global pkt_count

    if len(raw) != PACKET_SIZE:
        return
    if raw[0] != 0xFF or raw[1] != 0xAA:
        return

    pkt_count += 1
    wall = datetime.fromtimestamp(rx_ts).strftime("%H:%M:%S.%f")[:-3]

    # Skip version packets
    if raw[2] in (0x53, 0x48):
        prefix = "SW" if raw[2] == 0x53 else "HW"
        ver_str = raw[3:18].split(b'\x00')[0].decode("ascii", errors="replace")
        line = f"[{wall}] VERSION {prefix}: {ver_str}\n"
        print(line.rstrip())
        raw_log_fh.write(line)
        return

    rr_raw  = raw[8] | (raw[9] << 8)
    rr_ms   = rr_raw * 5.0 if rr_raw != 0xFFFF else 0.0
    adc     = struct.unpack_from("<i", raw, 13)[0]
    pi_avg  = raw[10]
    pi_pct  = pi_avg / 10.0 if pi_avg != 0 else 0.0
    status  = raw[3]
    hex_str = raw.hex(" ").upper()

    # Console summary (compact)
    print(
        f"[{wall}] #{pkt_count:5d} | "
        f"SpO2={raw[4]}% HR={raw[6]}bpm PI={pi_pct:.1f}% "
        f"Pleth={raw[12]:3d} ADC={adc:+10d} "
        f"Bat={raw[17] & 0x7F}% Freq={raw[18]}Hz "
        f"Status={status_text(status)}"
    )

    # Raw log
    raw_log_fh.write(
        f"[{wall}] PKT#{pkt_count} | {hex_str}\n"
        f"          idx={raw[2]} status=0x{status:02X}({status_text(status)}) "
        f"SpO2={raw[4]}/{raw[5]} HR={raw[6]}/{raw[7]} "
        f"RR_raw={rr_raw}({rr_ms:.0f}ms) PI={raw[10]}/{raw[11]} "
        f"Pleth={raw[12]} ADC={adc} Bat={raw[17]&0x7F}% Freq={raw[18]}Hz "
        f"CRC=0x{raw[19]:02X}\n"
    )
    raw_log_fh.flush()

    # CSV row
    csv_writer.writerow({
        "wall_time":       wall,
        "rx_timestamp":    f"{rx_ts:.4f}",
        "pkt_num":         pkt_count,
        "pkt_index":       raw[2],
        "status_raw":      f"0x{status:02X}",
        "status_text":     status_text(status),
        "spo2_avg":        raw[4],
        "spo2_real":       raw[5],
        "hr_avg":          raw[6],
        "hr_real":         raw[7],
        "rr_raw":          rr_raw,
        "rr_ms":           rr_ms,
        "pi_avg_permil":   raw[10],
        "pi_real_permil":  raw[11],
        "pi_pct":          pi_pct,
        "pleth":           raw[12],
        "adc_sample":      adc,
        "battery":         raw[17] & 0x7F,
        "freq_hz":         raw[18],
        "checksum_byte":   f"0x{raw[19]:02X}",
        "raw_hex":         hex_str,
    })
    csv_fh.flush()


def notification_handler(sender, data: bytearray):
    rx_ts = time.time()

    # Log the raw BLE chunk (may not be aligned to packet boundary)
    raw_buf.extend(data)

    # Process all complete packets
    while len(raw_buf) >= PACKET_SIZE:
        # Find header
        pos = -1
        for i in range(len(raw_buf) - 1):
            if raw_buf[i] == 0xFF and raw_buf[i + 1] == 0xAA:
                pos = i
                break
        if pos < 0:
            raw_buf.clear()
            break
        if pos > 0:
            del raw_buf[:pos]
        if len(raw_buf) < PACKET_SIZE:
            break

        raw = bytes(raw_buf[:PACKET_SIZE])
        del raw_buf[:PACKET_SIZE]
        decode_and_log(raw, rx_ts)

    # Check duration limit
    if stop_after and (time.time() - start_time) >= stop_after:
        raise KeyboardInterrupt("Duration elapsed")


async def run(address: str, duration=None):
    global start_time
    start_time = time.time()

    print(f"\nConnecting to {address} …")
    async with BleakClient(address) as client:
        if not client.is_connected:
            print("ERROR: Failed to connect.")
            return

        print(f"Connected! Sending 200Hz + ADC_ORIGINAL commands …")
        await client.write_gatt_char(RECV_UUID, bytes([CMD_200HZ]))
        await asyncio.sleep(0.05)
        await client.write_gatt_char(RECV_UUID, bytes([CMD_ADC_ORIG]))
        await asyncio.sleep(0.05)
        await client.write_gatt_char(RECV_UUID, bytes([CMD_SW_VER]))

        print(f"Logging to:\n  {RAW_LOG}\n  {CSV_LOG}\n")
        print(f"{'Time':12s} | {'#':>5s} | Fields …\n{'-'*80}")

        await client.start_notify(SEND_UUID, notification_handler)

        try:
            # Keep-alive loop
            last_ping = time.time()
            toggle = False
            while True:
                await asyncio.sleep(0.2)
                if time.time() - last_ping > 1.5:
                    cmd = CMD_200HZ if toggle else CMD_ADC_ORIG
                    await client.write_gatt_char(RECV_UUID, bytes([cmd]))
                    toggle = not toggle
                    last_ping = time.time()
                if duration and (time.time() - start_time) >= duration:
                    print(f"\nDuration ({duration}s) elapsed — stopping.")
                    break
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await client.stop_notify(SEND_UUID)

    print(f"\nDone. {pkt_count} packets logged.")
    print(f"RAW log : {RAW_LOG}")
    print(f"CSV log : {CSV_LOG}")


async def scan_only():
    print("Scanning for BLE devices (5s) …")
    devices = await BleakScanner.discover(timeout=5.0)
    if not devices:
        print("No devices found.")
        return None
    print(f"\n{'Name':30s}  Address")
    print("-" * 55)
    berry = []
    for d in devices:
        name = d.name or "(unknown)"
        marker = " ◀ Berry" if "berry" in name.lower() else ""
        print(f"{name:30s}  {d.address}{marker}")
        if "berry" in name.lower():
            berry.append(d)
    return berry[0].address if berry else None


def main():
    parser = argparse.ArgumentParser(description="BerryMed raw packet logger")
    parser.add_argument("--addr",      default=None, help="Device BLE address")
    parser.add_argument("--scan-only", action="store_true", help="Just scan, do not connect")
    parser.add_argument("--duration",  type=float, default=None,
                        help="Stop after N seconds (default: Ctrl+C)")
    args = parser.parse_args()

    if args.scan_only:
        asyncio.run(scan_only())
        return

    address = args.addr
    if not address:
        address = asyncio.run(scan_only())
        if not address:
            print("\nNo Berry device found. Specify one with --addr AA:BB:CC:DD:EE:FF")
            return
        print(f"\nAuto-selected: {address}\n")

    # Open log files
    global raw_log_fh, csv_writer, csv_fh
    raw_log_fh = open(RAW_LOG, "w", encoding="utf-8")
    csv_fh     = open(CSV_LOG, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()

    raw_log_fh.write(
        f"BerryMed Raw Packet Log — {datetime.now().isoformat()}\n"
        f"Device: {address}\n"
        f"{'='*80}\n\n"
    )

    try:
        asyncio.run(run(address, duration=args.duration))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        raw_log_fh.close()
        csv_fh.close()


if __name__ == "__main__":
    main()
