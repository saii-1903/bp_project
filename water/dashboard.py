"""
dashboard.py -- CustomTkinter dashboard for Berry PPG hydration monitoring.

Provides a real-time GUI that:
  - Connects to a BerryMed pulse oximeter via BLE
  - Decodes Berry Protocol V1.5 packets
  - Displays live vitals (SpO2, HR, PI, battery)
  - Shows a rolling PPG waveform
  - Runs the hydration analysis pipeline and plots the trend
  - Supports simulation mode (--simulate) for testing without hardware

Usage:
    python dashboard.py              # Device mode
    python dashboard.py --simulate   # Simulation mode
"""

import sys
import os
import time
import queue
import threading
import argparse
from collections import deque

import numpy as np
import pandas as pd

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import config as cfg
from berry_protocol import BerryPacket, decode_packet
from ble_connector import (
    BerryBLEConnector, STATE_IDLE, STATE_SCANNING,
    STATE_CONNECTING, STATE_CONNECTED, STATE_DISCONNECTED, STATE_ERROR,
    BLEAK_AVAILABLE,
)
from signal_processing import preprocess
from feature_extraction import extract_features
from hydration_engine import analyse_hydration
from inference_engine import VitalInferenceEngine


# ─── Theme ───────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

C = cfg.COLORS  # shorthand


# =====================================================================
#  SIMULATION THREAD (for --simulate mode)
# =====================================================================

class SimulationThread(threading.Thread):
    """Generates fake BerryPackets from the PPG simulator."""

    def __init__(self, packet_queue: queue.Queue):
        super().__init__(daemon=True)
        self.packet_queue = packet_queue
        self._stop = threading.Event()

    def run(self):
        from ppg_simulator import generate_ppg

        sim = generate_ppg(duration_minutes=cfg.SIMULATION_DURATION_MINUTES, seed=42)
        ppg = sim["raw_ppg"]
        hydration = sim["hydration_factor"]
        fs = sim["fs"]
        n = len(ppg)

        idx = 0
        pkt_idx = 0
        interval = 1.0 / cfg.SIMULATION_SPEED_HZ

        while not self._stop.is_set() and idx < n:
            h = hydration[idx]

            # Simulate Berry packet fields from the synthetic signal
            pleth = int(np.clip((ppg[idx] / 2.5) * 100, 1, 100))
            hr = int(72 + (1.0 - h) * 16)
            spo2 = int(98 - (1.0 - h) * 3)
            pi_raw = int(np.clip(h * 150, 1, 200))

            pkt = BerryPacket(
                packet_index=pkt_idx % 256,
                status=0x08 if pleth > 0 else 0x00,
                spo2_avg=spo2, spo2_real=spo2,
                hr_avg=hr, hr_real=hr,
                rr_interval_raw=int(60000 / hr / 5) if hr > 0 else 0,
                pi_avg=pi_raw, pi_real=pi_raw,
                pleth_wave=pleth,
                adc_sample=int(ppg[idx] * 10000),
                battery=85,
                packet_freq=cfg.SIMULATION_SPEED_HZ,
            )

            try:
                self.packet_queue.put_nowait(pkt)
            except queue.Full:
                pass

            pkt_idx += 1
            idx += 1
            time.sleep(interval)

    def stop(self):
        self._stop.set()


# =====================================================================
#  VITAL CARD WIDGET
# =====================================================================

class VitalCard(ctk.CTkFrame):
    """A single vital-sign display card."""

    def __init__(self, parent, label: str, unit: str, **kwargs):
        super().__init__(parent, corner_radius=12, **kwargs)

        self.configure(fg_color=C["panel_bg"], border_color=C["grid"],
                       border_width=1)

        self._label = ctk.CTkLabel(self, text=label, font=("Segoe UI", 12),
                                   text_color=C["text"])
        self._label.pack(pady=(10, 0))

        self._value = ctk.CTkLabel(self, text="--", font=("Segoe UI Bold", 36),
                                   text_color=C["accent"])
        self._value.pack(pady=(2, 0))

        self._unit = ctk.CTkLabel(self, text=unit, font=("Segoe UI", 11),
                                  text_color="#8B949E")
        self._unit.pack(pady=(0, 10))

    def set_value(self, val: str, color: str | None = None):
        self._value.configure(text=val)
        if color:
            self._value.configure(text_color=color)


# =====================================================================
#  MAIN DASHBOARD
# =====================================================================

class HydrationDashboard(ctk.CTk):
    """Main application window."""

    def __init__(self, simulate: bool = False):
        super().__init__()

        self.title("PPG Hydration Trend Monitor")
        self.geometry(f"{cfg.DASHBOARD_WIDTH}x{cfg.DASHBOARD_HEIGHT}")
        self.configure(fg_color=C["background"])

        self._simulate = simulate
        self._ble = BerryBLEConnector()
        self._ble.state_callback = self._on_state_change
        self._engine = VitalInferenceEngine()

        # Data buffers
        self._pleth_buf = deque(maxlen=cfg.LIVE_WAVEFORM_MAX_POINTS)
        self._pi_history: list[float] = []
        self._hr_history: list[float] = []
        self._spo2_history: list[float] = []
        self._time_stamps: list[float] = []
        self._raw_pleth_for_analysis: list[float] = []
        self._analysis_results: list[dict] = []
        # Vital trend history (one point per analysis cycle)
        self._vital_times: list[float] = []
        self._sbp_history: list[float] = []
        self._dbp_history: list[float] = []
        self._hb_history: list[float] = []
        self._glucose_history: list[float] = []

        self._session_start: float | None = None
        self._total_packets = 0
        self._scan_results: list[tuple] = []
        self._sim_thread: SimulationThread | None = None
        self._analysis_running = False
        self._last_analysis_len = 0
        self._current_phase = "IDLE"  # IDLE, CONNECTING, MEASURING, ANALYZING
        self._phase_start_time = None

        self._build_ui()
        self._create_graphs()
        self._poll_queue()

    # ─── UI Construction ─────────────────────────────────────────────

    def _build_ui(self):
        # ── Top Bar ──────────────────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color=C["panel_bg"], corner_radius=0, height=60)
        top.pack(fill="x", padx=0, pady=0)
        top.pack_propagate(False)

        # Title
        ctk.CTkLabel(top, text="PPG Hydration Monitor",
                     font=("Segoe UI Bold", 16), text_color=C["accent"]
                     ).pack(side="left", padx=15)

        # Status LED
        self._status_dot = ctk.CTkLabel(top, text="\u2B24", font=("Segoe UI", 14),
                                        text_color="#8B949E")
        self._status_dot.pack(side="left", padx=(0, 5))

        self._status_label = ctk.CTkLabel(top, text="Idle", font=("Segoe UI", 12),
                                          text_color=C["text"])
        self._status_label.pack(side="left")

        # Right side controls
        if self._simulate:
            self._btn_connect = ctk.CTkButton(
                top, text="Start Simulation", width=140,
                command=self._toggle_simulation,
                fg_color=C["hydration_green"], hover_color="#2ea043",
            )
            self._btn_connect.pack(side="right", padx=10)
        else:
            # Device dropdown
            self._device_var = ctk.StringVar(value="Select device...")
            self._device_menu = ctk.CTkOptionMenu(
                top, variable=self._device_var, values=["Scan first..."],
                width=200, fg_color=C["panel_bg"],
            )
            self._device_menu.pack(side="right", padx=5)

            self._btn_connect = ctk.CTkButton(
                top, text="Connect", width=100,
                command=self._toggle_connect,
                fg_color=C["accent"],
            )
            self._btn_connect.pack(side="right", padx=5)

            self._btn_scan = ctk.CTkButton(
                top, text="Scan", width=80, command=self._start_scan,
                fg_color="#30363D", hover_color="#484F58",
            )
            self._btn_scan.pack(side="right", padx=5)

        # Mode label
        mode_text = "SIMULATION" if self._simulate else "DEVICE"
        ctk.CTkLabel(top, text=mode_text, font=("Segoe UI", 10),
                     text_color="#D29922",
                     corner_radius=4).pack(side="right", padx=10)

        # ── Measurement Progress Bar ─────────────────────────────────
        self._progress_frame = ctk.CTkFrame(self, fg_color=C["panel_bg"], height=35, corner_radius=0)
        self._progress_frame.pack(fill="x")
        self._progress_frame.pack_propagate(False)

        self._lbl_phase = ctk.CTkLabel(self._progress_frame, text="READY", font=("Segoe UI Bold", 11), text_color=C["text"])
        self._lbl_phase.pack(side="left", padx=15)

        self._progress_bar = ctk.CTkProgressBar(self._progress_frame, height=12, corner_radius=6)
        self._progress_bar.set(0)
        self._progress_bar.pack(side="left", padx=15, fill="x", expand=True)
        self._progress_bar.configure(progress_color=C["accent"], fg_color=C["grid"])

        # ── Main Content ─────────────────────────────────────────────
        content = ctk.CTkFrame(self, fg_color=C["background"])
        content.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel: vitals
        left = ctk.CTkFrame(content, fg_color=C["background"], width=200)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        self._card_spo2 = VitalCard(left, "SpO2", "%")
        self._card_spo2.pack(fill="x", pady=(0, 8))

        self._card_hr = VitalCard(left, "Heart Rate", "bpm")
        self._card_hr.pack(fill="x", pady=(0, 8))

        self._card_pi = VitalCard(left, "Perfusion Index", "%")
        self._card_pi.pack(fill="x", pady=(0, 8))

        self._card_battery = VitalCard(left, "Battery", "%")
        self._card_battery.pack(fill="x", pady=(0, 8))

        self._card_hydration = VitalCard(left, "Hydration", "Level")
        self._card_hydration.pack(fill="x", pady=(0, 8))

        self._card_bp = VitalCard(left, "BP (SBP/DBP)", "mmHg")
        self._card_bp.pack(fill="x", pady=(0, 8))

        self._card_hb = VitalCard(left, "Hemoglobin", "g/dL")
        self._card_hb.pack(fill="x", pady=(0, 8))

        self._card_glucose = VitalCard(left, "Glucose", "mg/dL")
        self._card_glucose.pack(fill="x", pady=(0, 8))

        # Center panel: graphs
        self._graph_frame = ctk.CTkFrame(content, fg_color=C["panel_bg"],
                                         corner_radius=12)
        self._graph_frame.pack(side="left", fill="both", expand=True)

        # ── Bottom Bar ───────────────────────────────────────────────
        bottom = ctk.CTkFrame(self, fg_color=C["panel_bg"], corner_radius=0,
                              height=40)
        bottom.pack(fill="x", padx=0, pady=0)
        bottom.pack_propagate(False)

        self._lbl_packets = ctk.CTkLabel(bottom, text="Packets: 0",
                                         font=("Segoe UI", 11),
                                         text_color="#8B949E")
        self._lbl_packets.pack(side="left", padx=15)

        self._lbl_timer = ctk.CTkLabel(bottom, text="Session: 00:00",
                                       font=("Segoe UI", 11),
                                       text_color="#8B949E")
        self._lbl_timer.pack(side="left", padx=15)

        self._lbl_rate = ctk.CTkLabel(bottom, text="Rate: -- Hz",
                                      font=("Segoe UI", 11),
                                      text_color=C["accent"])
        self._lbl_rate.pack(side="left", padx=15)

        self._lbl_trend = ctk.CTkLabel(bottom, text="Trend: --",
                                       font=("Segoe UI Bold", 11),
                                       text_color=C["accent"])
        self._lbl_trend.pack(side="right", padx=15)

        self._btn_export = ctk.CTkButton(
            bottom, text="Export CSV", width=100, height=28,
            command=self._export_csv,
            fg_color="#30363D", hover_color="#484F58",
        )
        self._btn_export.pack(side="right", padx=10)

    # ─── Graph Setup ─────────────────────────────────────────────────

    def _create_graphs(self):
        """Create embedded matplotlib figures for live data."""
        self._fig = Figure(figsize=(10, 8), dpi=90,
                           facecolor=C["background"])

        # Row 1 Left: PPG Waveform
        self._ax_ppg = self._fig.add_subplot(3, 2, (1, 2))  # spans full row
        self._ax_ppg.set_facecolor(C["panel_bg"])
        self._ax_ppg.set_title("Live PPG Waveform (200 Hz)", color=C["text"],
                               fontsize=10, fontweight="bold")
        self._ax_ppg.set_ylabel("Pleth", color=C["text"])
        self._ax_ppg.tick_params(colors=C["text"], labelsize=7)
        self._ax_ppg.set_ylim(0, 105)
        self._ax_ppg.grid(True, color=C["grid"], alpha=0.5)
        for spine in self._ax_ppg.spines.values(): spine.set_color(C["grid"])
        self._ppg_line, = self._ax_ppg.plot([], [], color=C["accent"], linewidth=1.2)

        # Row 2 Left: Blood Pressure Trend
        self._ax_bp = self._fig.add_subplot(3, 2, 3)
        self._ax_bp.set_facecolor(C["panel_bg"])
        self._ax_bp.set_title("Blood Pressure Trend", color=C["text"], fontsize=10, fontweight="bold")
        self._ax_bp.set_ylabel("mmHg", color=C["text"])
        self._ax_bp.tick_params(colors=C["text"], labelsize=7)
        self._ax_bp.grid(True, color=C["grid"], alpha=0.5)
        for spine in self._ax_bp.spines.values(): spine.set_color(C["grid"])
        self._sbp_line, = self._ax_bp.plot([], [], color=C["hydration_red"], linewidth=1.8, label="SBP")
        self._dbp_line, = self._ax_bp.plot([], [], color=C["hydration_orange"], linewidth=1.8, label="DBP")
        self._ax_bp.legend(loc="upper right", fontsize=7, facecolor=C["panel_bg"], labelcolor=C["text"])

        # Row 2 Right: Hydration Index Trend
        self._ax_hi = self._fig.add_subplot(3, 2, 4)
        self._ax_hi.set_facecolor(C["panel_bg"])
        self._ax_hi.set_title("Hydration Index Trend", color=C["text"], fontsize=10, fontweight="bold")
        self._ax_hi.set_ylabel("HI (%)", color=C["text"])
        self._ax_hi.tick_params(colors=C["text"], labelsize=7)
        self._ax_hi.grid(True, color=C["grid"], alpha=0.5)
        self._ax_hi.axhline(0, color=C["text"], alpha=0.3, linestyle="--", linewidth=0.8)
        for spine in self._ax_hi.spines.values(): spine.set_color(C["grid"])
        self._hi_line, = self._ax_hi.plot([], [], color=C["hydration_green"], linewidth=2.0)
        self._ax_hi.axhspan(0, 50, color=C["hydration_green"], alpha=0.06)
        self._ax_hi.axhspan(-10, 0, color=C["hydration_yellow"], alpha=0.06)
        self._ax_hi.axhspan(-25, -10, color=C["hydration_orange"], alpha=0.06)
        self._ax_hi.axhspan(-50, -25, color=C["hydration_red"], alpha=0.06)

        # Row 3 Left: Hemoglobin Trend
        self._ax_hb = self._fig.add_subplot(3, 2, 5)
        self._ax_hb.set_facecolor(C["panel_bg"])
        self._ax_hb.set_title("Hemoglobin Trend", color=C["text"], fontsize=10, fontweight="bold")
        self._ax_hb.set_ylabel("g/dL", color=C["text"])
        self._ax_hb.set_xlabel("Analysis #", color=C["text"])
        self._ax_hb.tick_params(colors=C["text"], labelsize=7)
        self._ax_hb.grid(True, color=C["grid"], alpha=0.5)
        for spine in self._ax_hb.spines.values(): spine.set_color(C["grid"])
        self._hb_line, = self._ax_hb.plot([], [], color=C["pi_high"], linewidth=1.8, marker="o", markersize=4)
        self._ax_hb.axhspan(12, 17.5, color=C["hydration_green"], alpha=0.06)  # Normal range band

        # Row 3 Right: Glucose Trend
        self._ax_glu = self._fig.add_subplot(3, 2, 6)
        self._ax_glu.set_facecolor(C["panel_bg"])
        self._ax_glu.set_title("Glucose Trend", color=C["text"], fontsize=10, fontweight="bold")
        self._ax_glu.set_ylabel("mg/dL", color=C["text"])
        self._ax_glu.set_xlabel("Analysis #", color=C["text"])
        self._ax_glu.tick_params(colors=C["text"], labelsize=7)
        self._ax_glu.grid(True, color=C["grid"], alpha=0.5)
        for spine in self._ax_glu.spines.values(): spine.set_color(C["grid"])
        self._glu_line, = self._ax_glu.plot([], [], color=C["pi_very_high"], linewidth=1.8, marker="o", markersize=4)
        self._ax_glu.axhspan(70, 140, color=C["hydration_green"], alpha=0.06)  # Normal range band

        self._fig.tight_layout(pad=1.8)

        self._canvas = FigureCanvasTkAgg(self._fig, self._graph_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    # ─── BLE Callbacks ───────────────────────────────────────────────

    def _on_state_change(self, state: str):
        """Called from the BLE thread when connection state changes."""
        # Schedule UI update on main thread
        self.after(0, lambda: self._update_status(state))

    def _update_status(self, state: str):
        color_map = {
            STATE_IDLE: ("#8B949E", "Idle"),
            STATE_SCANNING: (C["accent"], "Scanning..."),
            STATE_CONNECTING: (C["hydration_yellow"], "Connecting..."),
            STATE_CONNECTED: (C["hydration_green"], "Connected"),
            STATE_DISCONNECTED: (C["hydration_orange"], "Disconnected"),
            STATE_ERROR: (C["hydration_red"], "Error"),
        }
        color, text = color_map.get(state, ("#8B949E", state))
        self._status_dot.configure(text_color=color)
        self._status_label.configure(text=text)

        if state == STATE_CONNECTED:
            self._btn_connect.configure(text="Disconnect",
                                        fg_color=C["hydration_red"])
            if not self._session_start:          # Only set on very first connect
                self._session_start = time.time()
            self._current_phase = "CONNECTING"
            self._phase_start_time = time.time()
            # NOTE: Do NOT clear _raw_pleth_for_analysis here.
            # The device only transmits for ~30s each connection, so we
            # accumulate samples across multiple connections until we have
            # enough for a full analysis window.
        elif state in (STATE_DISCONNECTED, STATE_ERROR, STATE_IDLE):
            if not self._simulate:
                self._btn_connect.configure(text="Connect",
                                            fg_color=C["accent"])

    # ─── Scan / Connect ──────────────────────────────────────────────

    def _start_scan(self):
        self._ble.scan(duration=5.0, callback=self._on_scan_done)

    def _on_scan_done(self, devices: list):
        self._scan_results = devices
        if devices:
            names = [f"{name} ({addr})" for name, addr in devices]
            self.after(0, lambda: self._device_menu.configure(values=names))
            self.after(0, lambda: self._device_var.set(names[0]))
        else:
            self.after(0, lambda: self._device_var.set("No devices found"))

    def _toggle_connect(self):
        if self._ble.state == STATE_CONNECTED:
            self._ble.disconnect()
        else:
            sel = self._device_var.get()
            if "(" in sel and ")" in sel:
                addr = sel.split("(")[-1].rstrip(")")
                self._ble.connect(addr)

    def _toggle_simulation(self):
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.stop()
            self._sim_thread = None
            self._btn_connect.configure(text="Start Simulation",
                                        fg_color=C["hydration_green"])
            self._update_status(STATE_DISCONNECTED)
        else:
            self._sim_thread = SimulationThread(self._ble.packet_queue)
            self._sim_thread.start()
            self._session_start = time.time()
            self._btn_connect.configure(text="Stop Simulation",
                                        fg_color=C["hydration_red"])
            self._update_status(STATE_CONNECTED)

            self._update_status(STATE_CONNECTED)

    # ─── Data Processing Loop ────────────────────────────────────────

    def _poll_queue(self):
        """Poll the packet queue and update UI (runs every DASHBOARD_POLL_MS)."""
        # Process more packets per cycle to avoid lag (200Hz device -> ~10 pkts/50ms, but we buffer)
        batch_limit = 200 
        processed = 0

        while processed < batch_limit:
            try:
                pkt: BerryPacket = self._ble.packet_queue.get_nowait()
            except queue.Empty:
                break

            self._total_packets += 1
            processed += 1
            
            # Debug: print queue size occasionally
            if self._total_packets % 100 == 0:
                n_samples = len(self._raw_pleth_for_analysis)
                print(f"Queue Size: {self._ble.packet_queue.qsize()} | Speed: {processed}/poll | Last PI: {pkt.pi_percent:.2f} | Collected: {n_samples} samples")
                if n_samples == 0 and self._total_packets > 200:
                    print("WARNING: No pleth data collected — is a finger on the sensor? (pleth_wave=0 every packet)")
                if pkt.pi_percent > 15.0:
                    print(f"WARNING: High PI detected: {pkt.pi_percent:.2f}% (Limit is 20.0)")

            # Update vitals
            if pkt.spo2_valid:

                self._card_spo2.set_value(
                    str(pkt.spo2_avg),
                    C["hydration_green"] if pkt.spo2_avg >= 95
                    else C["hydration_yellow"] if pkt.spo2_avg >= 90
                    else C["hydration_red"]
                )
                self._spo2_history.append(pkt.spo2_avg)

            if pkt.hr_valid:
                self._card_hr.set_value(
                    str(pkt.hr_avg),
                    C["accent"] if 60 <= pkt.hr_avg <= 100
                    else C["hydration_yellow"]
                )
                self._hr_history.append(pkt.hr_avg)

            if pkt.pi_valid:
                # User-provided PI Ranges:
                # < 0.2: Very Low (Red), 0.2-0.5: Low (Orange), 0.5-2.0: Normal (Green)
                # 2.0-20.0: High (Blue), > 20.0: Very High (Purple)
                p = pkt.pi_percent
                color = (C["hydration_red"] if p < 0.2
                         else C["pi_low"] if p < 0.5
                         else C["hydration_green"] if p < 2.0
                         else C["pi_high"] if p <= 20.0
                         else C["pi_very_high"])
                
                self._card_pi.set_value(f"{p:.1f}", color)
                self._pi_history.append(p)

            self._card_battery.set_value(
                str(pkt.battery),
                C["hydration_green"] if pkt.battery > 30
                else C["hydration_yellow"] if pkt.battery > 10
                else C["hydration_red"]
            )

            # PPG waveform buffer — always collect if pleth_wave is valid
            # The watch sends pleth_wave=0 only when completely off.
            # Collect even with low/zero PI; quality filter happens in analysis.
            pleth_val = pkt.pleth_wave
            if pleth_val == 0 and pkt.adc_sample != 0:
                # Fallback: normalise ADC to 1-100 range for compatibility
                pleth_val = max(1, min(100, abs(pkt.adc_sample) % 100))
            if pleth_val > 0:
                self._pleth_buf.append(pleth_val)
                self._raw_pleth_for_analysis.append(pleth_val)

            # Timestamp
            if self._session_start:
                elapsed = time.time() - self._session_start
                self._time_stamps.append(elapsed)

        # Update graphs periodically (every 5th poll to save CPU)
        if self._total_packets % 10 == 0 and len(self._pleth_buf) > 2:
            self._update_ppg_graph()

        # --- Simple continuous collect → analyse loop ---
        # Always collecting. As soon as we have enough data, fire analysis.
        # While analysis runs in background we keep collecting.
        analysis_rate = cfg.SAMPLING_RATE_HZ
        window_samples = int(cfg.FEATURE_WINDOW_SECONDS * analysis_rate)  # 6000

        data_len = len(self._raw_pleth_for_analysis)

        if self._session_start:
            # Show collection progress until we have a full window
            progress = min(1.0, data_len / window_samples)
            self._progress_bar.set(progress)
            if data_len < window_samples:
                needed = window_samples - data_len
                self._lbl_phase.configure(
                    text=f"COLLECTING  {data_len}/{window_samples} samples  ({needed} to go…)",
                    text_color=C["hydration_yellow"]
                )
            else:
                self._lbl_phase.configure(
                    text=f"LIVE  |  {data_len} samples  — analysing every ~5s",
                    text_color=C["hydration_green"]
                )

        # --- Accumulate → analyse ---
        # Samples persist across connections. Fire analysis the INSTANT
        # we cross the 6000-sample threshold, and again every 6000 new samples
        # (full 30s window) — matching what the model was trained on.
        if data_len >= window_samples and not self._analysis_running:
            samples_since_last = data_len - self._last_analysis_len
            if samples_since_last >= window_samples or self._last_analysis_len == 0:
                self._run_analysis()
                self._last_analysis_len = data_len


        # Update footer
        self._lbl_packets.configure(text=f"Packets: {self._total_packets:,}")
        if self._session_start:
            elapsed = time.time() - self._session_start
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            self._lbl_timer.configure(text=f"Session: {mins:02d}:{secs:02d}")

        # Schedule next poll
        self.after(cfg.DASHBOARD_POLL_MS, self._poll_queue)

        # Update PPS rate every second
        curr_time = time.time()
        if not hasattr(self, "_last_rate_update"):
            self._last_rate_update = curr_time
            self._last_total_pkts = self._total_packets

        if curr_time - self._last_rate_update >= 1.0:
            pps = self._total_packets - self._last_total_pkts
            self._lbl_rate.configure(text=f"Rate: {pps} Hz", 
                                     text_color=C["hydration_green"] if pps > 100 else C["hydration_yellow"])
            self._last_rate_update = curr_time
            self._last_total_pkts = self._total_packets

    # ─── Graph Updates ───────────────────────────────────────────────

    def _update_ppg_graph(self):
        """Update the live PPG waveform graph."""
        data = list(self._pleth_buf)
        x = list(range(len(data)))
        self._ppg_line.set_data(x, data)
        self._ax_ppg.set_xlim(0, max(len(data), 10))
        self._canvas.draw_idle()

    def _update_hi_graph(self, hi_times: list, hi_values: list,
                         hi_colors: list):
        """Update the hydration index trend graph."""
        self._hi_line.set_data(hi_times, hi_values)
        if hi_values:
            self._ax_hi.set_xlim(0, max(hi_times) + 0.5)
            self._ax_hi.set_ylim(-5, 5)

            # Update line color based on latest value
            last_val = hi_values[-1]
            if last_val >= 0:
                self._hi_line.set_color(C["hydration_green"])
            elif last_val >= -10:
                self._hi_line.set_color(C["hydration_yellow"])
            elif last_val >= -25:
                self._hi_line.set_color(C["hydration_orange"])
            else:
                self._hi_line.set_color(C["hydration_red"])

        self._canvas.draw_idle()

    def _update_vital_graphs(self, times, sbp, dbp, hb, glu):
        """Update BP, Hb, and Glucose trend graphs."""
        import math
        # Filter out NaN/Inf values paired with their x index
        def clean(vals):
            return [(i+1, v) for i, v in enumerate(vals)
                    if v is not None and not math.isnan(v) and not math.isinf(v)]

        x = list(range(1, len(times) + 1))

        # --- Blood Pressure ---
        sbp_clean = [(i+1, v) for i, v in enumerate(sbp)
                     if v is not None and not math.isnan(v) and not math.isinf(v)]
        dbp_clean = [(i+1, v) for i, v in enumerate(dbp)
                     if v is not None and not math.isnan(v) and not math.isinf(v)]
        if sbp_clean and dbp_clean:
            sx, sv = zip(*sbp_clean); dx, dv = zip(*dbp_clean)
            self._sbp_line.set_data(sx, sv)
            self._dbp_line.set_data(dx, dv)
            all_bp = list(sv) + list(dv)
            self._ax_bp.set_xlim(0.5, max(max(sx), max(dx)) + 0.5)
            self._ax_bp.set_ylim(max(0, min(all_bp) - 10), max(all_bp) + 10)
            self._ax_bp.set_title(
                f"Blood Pressure  ▶  SBP: {sv[-1]:.0f}  /  DBP: {dv[-1]:.0f} mmHg",
                color=C["hydration_red"], fontsize=10, fontweight="bold"
            )

        # --- Hemoglobin ---
        hb_clean = clean(hb)
        if hb_clean:
            hx, hv = zip(*hb_clean)
            self._hb_line.set_data(hx, hv)
            self._ax_hb.set_xlim(0.5, max(hx) + 0.5)
            self._ax_hb.set_ylim(max(0, min(hv) - 2), max(hv) + 2)
            self._ax_hb.set_title(
                f"Hemoglobin  ▶  {hv[-1]:.1f} g/dL",
                color=C["pi_high"], fontsize=10, fontweight="bold"
            )

        # --- Glucose ---
        glu_clean = clean(glu)
        if glu_clean:
            gx, gv = zip(*glu_clean)
            self._glu_line.set_data(gx, gv)
            self._ax_glu.set_xlim(0.5, max(gx) + 0.5)
            self._ax_glu.set_ylim(max(0, min(gv) - 20), max(gv) + 20)
            self._ax_glu.set_title(
                f"Glucose  ▶  {gv[-1]:.0f} mg/dL",
                color=C["pi_very_high"], fontsize=10, fontweight="bold"
            )

        self._canvas.draw_idle()


    def _run_analysis(self):
        """Run the hydration pipeline on accumulated data."""
        self._analysis_running = True

        def _worker():
            try:
                raw = np.array(self._raw_pleth_for_analysis, dtype=float)

                # Normalize pleth (0-100) to ~0-2 volt-like range
                raw_norm = raw / 50.0

                fs = cfg.SIMULATION_SPEED_HZ if self._simulate else cfg.BERRY_DEFAULT_RATE_HZ
                analysis_rate = fs
                window_samples = int(cfg.FEATURE_WINDOW_SECONDS * analysis_rate)
                proc = preprocess(raw_norm, fs)

                if len(proc["peaks"]) < cfg.MIN_PEAKS_PER_WINDOW:
                    print(f"DEBUG: Analysis skipped: Too few peaks ({len(proc['peaks'])} < {cfg.MIN_PEAKS_PER_WINDOW})")
                    self._analysis_running = False
                    return

                print(f"DEBUG: Extracting features from {len(proc['corrected'])} samples...")
                features = extract_features(
                    corrected=proc["corrected"],
                    raw_ppg=raw_norm,
                    peaks=proc["peaks"],
                    valleys=proc["valleys"],
                    sqi=proc["sqi"],
                    fs=fs,
                )

                if features.empty:
                    print("DEBUG: Analysis skipped: Feature extraction returned empty (PI filter issue?)")
                    self._analysis_running = False
                    return

                print(f"DEBUG: Running hydration analysis on {len(features)} windows...")
                hydration = analyse_hydration(features)
                results = hydration["results"]

                # ── Multi-Vital Prediction (runs regardless of hydration result) ──
                print("DEBUG: Running multi-vital inference...")
                seg = raw_norm[-window_samples:]
                if self._session_start and len(self._raw_pleth_for_analysis) > 0:
                    elapsed_s = max(1.0, time.time() - self._session_start)
                    actual_rate = len(self._raw_pleth_for_analysis) / elapsed_s
                    print(f"DEBUG: Actual measured rate: {actual_rate:.1f} Hz")
                else:
                    actual_rate = fs
                vitals = self._engine.predict_vitals(seg, actual_rate_hz=actual_rate)
                if vitals:
                    print(f"DEBUG: Vitals predicted: {vitals}")
                    cycle_num = len(self._vital_times) + 1
                    self._vital_times.append(cycle_num)
                    self._sbp_history.append(vitals.get("sbp", float("nan")))
                    self._dbp_history.append(vitals.get("dbp", float("nan")))
                    self._hb_history.append(vitals.get("hb", float("nan")))
                    self._glucose_history.append(vitals.get("glucose", float("nan")))
                    sbp_v  = vitals.get("sbp", None)
                    dbp_v  = vitals.get("dbp", None)
                    bp_cat = vitals.get("bp_category", "normal")
                    hb_v   = vitals.get("hb", None)
                    glu_v  = vitals.get("glucose", None)
                    t_snap   = list(self._vital_times)
                    sbp_snap = list(self._sbp_history)
                    dbp_snap = list(self._dbp_history)
                    hb_snap  = list(self._hb_history)
                    glu_snap = list(self._glucose_history)
                    if sbp_v is not None and dbp_v is not None:
                        bp_color = (C["hydration_green"] if bp_cat == "normal"
                                    else C["hydration_yellow"] if bp_cat == "hypo"
                                    else C["hydration_red"])
                        self.after(0, lambda s=sbp_v, d=dbp_v, c=bp_color:
                                   self._card_bp.set_value(f"{s:.0f}/{d:.0f}", c))
                    if hb_v is not None:
                        self.after(0, lambda h=hb_v:
                                   self._card_hb.set_value(f"{h:.1f} g/dL"))
                    if glu_v is not None:
                        glu_col = (C["hydration_green"] if 70 <= glu_v <= 140
                                   else C["hydration_yellow"])
                        self.after(0, lambda g=glu_v, c=glu_col:
                                   self._card_glucose.set_value(f"{g:.0f} mg/dL", c))
                    self.after(0, lambda ts=t_snap, sb=sbp_snap, db=dbp_snap,
                               hb=hb_snap, gl=glu_snap:
                               self._update_vital_graphs(ts, sb, db, hb, gl))

                # ── Hydration UI (only if results available) ──────────────
                if results.empty:
                    self._analysis_running = False
                    return

                hi_times = (results["window_center_s"] / 60).tolist()
                hi_values = results["hi_smoothed"].tolist()
                last_hi = hi_values[-1]
                trend = hydration["overall_kendall"]["trend"]
                
                print(f"DEBUG: Analysis Result | HI: {last_hi:.2f} | Trend: {trend}")
                
                hi_colors = results["zone_color"].tolist()
                last_level = results["hydration_level"].iloc[-1]
                trend = hydration["overall_kendall"]["trend"]

                # Store result
                self._analysis_results.append({
                    "hydration_index": last_hi,
                    "trend": trend,
                    "timestamp": time.time()
                })

                # Schedule UI updates on main thread
                self.after(0, lambda: self._update_hi_graph(
                    hi_times, hi_values, hi_colors))
                self.after(0, lambda: self._card_hydration.set_value(
                    last_level.split()[0],  # First word
                    C["hydration_green"] if last_hi >= 0
                    else C["hydration_yellow"] if last_hi >= -10
                    else C["hydration_orange"] if last_hi >= -25
                    else C["hydration_red"]
                ))
                self.after(0, lambda: self._lbl_trend.configure(
                    text=f"Trend: {trend}",
                    text_color=C["hydration_green"] if "Increasing" in trend
                    else C["hydration_red"] if "Decreasing" in trend
                    else C["text"]
                ))


            except Exception as e:
                import traceback
                print(f"Analysis error: {e}")
                traceback.print_exc()
            finally:
                self._analysis_running = False

        threading.Thread(target=_worker, daemon=True).start()

    # ─── Export ──────────────────────────────────────────────────────

    def _export_csv(self):
        """Export collected data to CSV."""
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Export vitals history
        n = min(len(self._time_stamps),
                len(self._hr_history),
                len(self._pi_history))
        if n > 0:
            df = pd.DataFrame({
                "time_s": self._time_stamps[:n],
                "hr_bpm": self._hr_history[:n],
                "pi_percent": self._pi_history[:n],
            })
            path = os.path.join(cfg.OUTPUT_DIR, f"vitals_{timestamp}.csv")
            df.to_csv(path, index=False)

        # Export raw pleth
        if self._raw_pleth_for_analysis:
            pleth_df = pd.DataFrame({"pleth": self._raw_pleth_for_analysis})
            ppath = os.path.join(cfg.OUTPUT_DIR, f"pleth_{timestamp}.csv")
            pleth_df.to_csv(ppath, index=False)

        self._lbl_trend.configure(
            text=f"Exported to output/", text_color=C["hydration_green"]
        )


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PPG Hydration Trend Monitor Dashboard"
    )
    parser.add_argument("--simulate", action="store_true",
                        help="Run in simulation mode without a real device")
    args = parser.parse_args()

    if not args.simulate and not BLEAK_AVAILABLE:
        print("Warning: 'bleak' not installed. Running in simulation mode.")
        print("Install with: pip install bleak")
        args.simulate = True

    app = HydrationDashboard(simulate=args.simulate)
    app.mainloop()


if __name__ == "__main__":
    main()
