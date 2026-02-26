import time
import queue
import argparse
import sys
import os
from ble_connector import BerryBLEConnector, STATE_CONNECTED, STATE_DISCONNECTED, STATE_ERROR

def main():
    parser = argparse.ArgumentParser(description="Collect raw data from BerryMed device.")
    parser.add_argument("--hz", type=int, choices=[50, 100, 200], default=100, help="Sampling rate (50, 100, or 200)")
    parser.add_argument("--duration", type=int, default=0, help="Duration to collect in seconds (0 = infinite)")
    args = parser.parse_args()

    connector = BerryBLEConnector()

    print("Scanning for BerryMed devices...")
    scan_results = []
    scan_done = False

    def on_scan(devices):
        nonlocal scan_done
        scan_results.extend(devices)
        scan_done = True

    connector.scan(duration=5.0, callback=on_scan)
    
    while not scan_done:
        time.sleep(0.1)

    if not scan_results:
        print("No devices found. Ensure device is turned on and in range.")
        sys.exit(1)

    print("Found devices:")
    target_addr = None
    for i, (name, addr) in enumerate(scan_results):
        print(f"[{i}] {name} ({addr})")
        if target_addr is None and "berry" in name.lower() or name.startswith("PC-"):
            target_addr = addr

    if target_addr is None:
        target_addr = scan_results[0][1] # Default to first

    print(f"Connecting to {target_addr}...")
    
    conn_state = ""
    def on_state(state):
        nonlocal conn_state
        conn_state = state
        print(f"BLE State: {state}")
        
    connector.state_callback = on_state
    connector.connect(target_addr)

    # Wait for connection
    timeout = 10
    start_wait = time.time()
    while conn_state != STATE_CONNECTED:
        time.sleep(0.5)
        if time.time() - start_wait > timeout:
            print("Failed to connect within timeout.")
            sys.exit(1)
        if conn_state in (STATE_DISCONNECTED, STATE_ERROR):
            print("Connection failed.")
            sys.exit(1)

    print(f"Connected! Setting rate to {args.hz}Hz...")
    import config as cfg
    time.sleep(1.0) # wait for connection to settle
    connector.set_rate(args.hz)
    time.sleep(0.5)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(cfg.OUTPUT_DIR, f"berry_raw_{int(time.time())}_{args.hz}hz.csv")
    
    print(f"Logging data to {filename}. Press Ctrl+C to stop.")
    start_time = time.time()
    packets_saved = 0
    
    try:
        with open(filename, "w") as f:
            f.write("Timestamp,Uptime,ADC_Sample,Pleth_Wave,SpO2,HR,PI,Battery,InvalidFlag,Raw_Hex\n")
            
            while True:
                if args.duration > 0 and (time.time() - start_time) > args.duration:
                    print(f"\nReached target duration of {args.duration}s.")
                    break
                    
                if conn_state != STATE_CONNECTED:
                    print("\nDevice disconnected unexpectedly.")
                    break
                    
                try:
                    pkt = connector.packet_queue.get(timeout=0.1)
                    uptime = time.time() - start_time
                    f.write(f"{time.time()},{uptime:.3f},{pkt.adc_sample},{pkt.pleth_wave},"
                            f"{pkt.spo2_avg},{pkt.hr_avg},{pkt.pi_percent:.2f},{pkt.battery},{int(not pkt.is_valid)},{pkt.raw_hex}\n")
                    packets_saved += 1
                    
                    if packets_saved % args.hz == 0:
                        sys.stdout.write(f"\rCollected {packets_saved} packets ({(packets_saved/uptime):.1f} Hz actual)... ")
                        sys.stdout.flush()
                except queue.Empty:
                    continue
                    
    except KeyboardInterrupt:
        print("\nStopping collection...")
    finally:
        connector.disconnect()
        print(f"\nDone! Saved {packets_saved} packets to {filename}.")

if __name__ == "__main__":
    main()
