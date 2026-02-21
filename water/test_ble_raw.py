import asyncio
import time
from bleak import BleakScanner, BleakClient
import config as cfg
from berry_protocol import decode_packet, decode_version

# Reassembly buffer
buffer = bytearray()

def notification_handler(sender, data: bytearray):
    global buffer
    buffer.extend(data)

    while len(buffer) >= cfg.BERRY_PACKET_SIZE:
        # Find header 0xFF 0xAA
        header_pos = -1
        for i in range(len(buffer) - 1):
            if buffer[i] == 0xFF and buffer[i+1] == 0xAA:
                header_pos = i
                break
        
        if header_pos < 0:
            buffer.clear()
            break
        
        if header_pos > 0:
            del buffer[:header_pos]
        
        if len(buffer) < cfg.BERRY_PACKET_SIZE:
            break

        # Extract packet
        raw = bytes(buffer[:cfg.BERRY_PACKET_SIZE])
        del buffer[:cfg.BERRY_PACKET_SIZE]

        # Print RAW hex
        hex_str = " ".join(f"{b:02X}" for b in raw)
        
        # Decode
        ver = decode_version(raw)
        if ver:
            print(f"[RAW] {hex_str} | VERSION: {ver}")
            continue

        pkt = decode_packet(raw)
        if pkt:
            print(f"[RAW] {hex_str} | IDX:{pkt.packet_index:3d} SPO2:{pkt.spo2_avg}% HR:{pkt.hr_avg} PI:{pkt.pi_percent:.1f}% WAVE:{pkt.pleth_wave:3d} BATT:{pkt.battery}%")
        else:
            print(f"[RAW] {hex_str} | INVALID CHECKSUM")

async def main():
    print("Scanning for BerryMed devices...")
    # SCAN (No service UUID filter, consistent with fix)
    devices = await BleakScanner.discover(timeout=5.0)
    berry_devices = [d for d in devices if d.name and "berry" in d.name.lower()]
    
    if not berry_devices:
        print("No 'Berry' devices found via name filter.")
        print("All found devices:")
        for d in devices:
            print(f" - {d.name} ({d.address})")
        
        # If no Berry found, let's try connecting to the first device that has the service UUID?
        # Actually, let's just pick the first one if user wants, or exit.
        # But user likely has one nearby.
        if not devices:
            print("No BLE devices found at all.")
            return
        target = devices[0] # Fallback
    else:
        target = berry_devices[0]

    print(f"Connecting to {target.name} ({target.address})...")

    async with BleakClient(target.address) as client:
        print("Connected.")
        
        # Subscribe
        await client.start_notify(cfg.BERRY_SEND_CHAR_UUID, notification_handler)
        
        # Send 200Hz command
        print("Sending 200Hz command...")
        await client.write_gatt_char(cfg.BERRY_RECV_CHAR_UUID, bytes([cfg.BERRY_CMD_200HZ]))
        
        # Keep alive loop
        print("Listening for data... (Press Ctrl+C to stop)")
        last_ping = time.time()
        
        while True:
            await asyncio.sleep(0.1)
            if time.time() - last_ping > 2.0:
                # Keep-alive
                try:
                    await client.write_gatt_char(cfg.BERRY_RECV_CHAR_UUID, bytes([cfg.BERRY_CMD_SW_VERSION]))
                    last_ping = time.time()
                except Exception as e:
                    print(f"Ping failed: {e}")
                    break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
