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
from inference_engine import VitalInferenceEngine, BPTrendTracker, GenericTrendTracker


# ─── JUMP FILTER LOGIC ───────────────────────────────────────────────
class SmoothValueFilter:
    """
    Prevents sudden spikes > threshold.
    Only allows a jump if it persists for 'persistence' consecutive readings.
    """
    def __init__(self, threshold=10.0, persistence=3):
        self.current_val = None
        self.threshold = threshold
        self.persistence_limit = persistence
        self.spike_count = 0
        self.pending_val = None

    def update(self, new_val):
        if new_val is None: return self.current_val
        
        # First reading ever? Accept it.
        if self.current_val is None:
            self.current_val = new_val
            return new_val

        delta = abs(new_val - self.current_val)

        if delta <= self.threshold:
            # Safe change -> Update immediately
            self.current_val = new_val
            self.spike_count = 0
            return new_val
        else:
            # Sudden Spike -> Suppress it initially
            self.spike_count += 1
            
            # If spike persists (real physiological shift or consistent sensor error), accept it
            if self.spike_count >= self.persistence_limit:
                self.current_val = new_val
                self.spike_count = 0
                return new_val
            
            # Return old value (suppress spike) and wait
            return self.current_val


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
        self.packets_lost = 0  # Dummy for UI compatibility
        self.packets_received = 0

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
            self.packets_received += 1
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
        self._pr_for_analysis: list[float] = []
        self._bp_trend_tracker = BPTrendTracker()
        self._hb_trend_tracker = GenericTrendTracker(threshold=0.1) # g/dL
        self._glu_trend_tracker = GenericTrendTracker(threshold=5.0) # mg/dL
        self._last_data_time = time.time()
        self._last_vitals_report_time = 0
        self._last_hydration_report_time = 0
        self._vitals_first_run = True
        self._calib_offsets = {"sbp": 0.0, "dbp": 0.0, "hb": 0.0, "glucose": 0.0}
        self._calib_targets = {"sbp": None, "dbp": None}
        self._vitals_calibrated = False
        self._last_raw_preds = {} # Baseline for relative calibration
        
        # --- Jump Filters (Prevents scary spikes) ---
        self._sbp_filter = SmoothValueFilter(threshold=10.0, persistence=3)
        self._dbp_filter = SmoothValueFilter(threshold=8.0, persistence=3)
        
        # --- Data Quality Logging ---
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, "logs"), exist_ok=True)
        self._loss_log_path = os.path.join(cfg.OUTPUT_DIR, "logs", f"data_loss_{int(time.time())}.log")
        self._bp_log_path = os.path.join(cfg.OUTPUT_DIR, "logs", f"bp_calculations_{int(time.time())}.log")
        self._last_logged_loss = 0
        self._total_loss_events = 0

        # Create BF log header
        with open(self._bp_log_path, "w") as f:
            f.write("Timestamp,Raw_SBP,Raw_DBP,Offset_SBP,Offset_DBP,Cal_SBP,Cal_DBP,Smooth_SBP,Smooth_DBP,Category,Jump_Supressed\n")

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

        # Left panel: vitals (Scrollable to handle many cards)
        left = ctk.CTkScrollableFrame(content, fg_color=C["background"], width=220, label_text="Vitals & Calib")
        left.pack(side="left", fill="y", padx=(0, 10))
        # left.pack_propagate(False) # Not needed/supported for scrollable frames in same way

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

        # --- Manual Calibration ---
        self._calib_frame = ctk.CTkFrame(left, fg_color=C["panel_bg"], corner_radius=12,
                                         border_color=C["grid"], border_width=1)
        self._calib_frame.pack(fill="x", pady=(0, 0))
        
        ctk.CTkLabel(self._calib_frame, text="Manual Calibration", font=("Segoe UI Bold", 11),
                     text_color=C["accent"]).pack(pady=(5, 2))
        
        input_row = ctk.CTkFrame(self._calib_frame, fg_color="transparent")
        input_row.pack(fill="x", padx=10, pady=2)
        
        ctk.CTkLabel(input_row, text="SBP:", font=("Segoe UI", 10), text_color=C["text"]).pack(side="left")
        self._ent_sbp = ctk.CTkEntry(input_row, width=40, font=("Segoe UI", 10), height=22, fg_color=C["background"])
        self._ent_sbp.pack(side="left", padx=2)
        
        ctk.CTkLabel(input_row, text="DBP:", font=("Segoe UI", 10), text_color=C["text"]).pack(side="left", padx=(5, 0))
        self._ent_dbp = ctk.CTkEntry(input_row, width=40, font=("Segoe UI", 10), height=22, fg_color=C["background"])
        self._ent_dbp.pack(side="left", padx=2)
        
        input_row2 = ctk.CTkFrame(self._calib_frame, fg_color="transparent")
        input_row2.pack(fill="x", padx=10, pady=2)
        
        ctk.CTkLabel(input_row2, text="Hb:", font=("Segoe UI", 10), text_color=C["text"]).pack(side="left")
        self._ent_hb = ctk.CTkEntry(input_row2, width=40, font=("Segoe UI", 10), height=22, fg_color=C["background"])
        self._ent_hb.pack(side="left", padx=2)
        
        ctk.CTkLabel(input_row2, text="Glu:", font=("Segoe UI", 10), text_color=C["text"]).pack(side="left", padx=(5, 0))
        self._ent_glu = ctk.CTkEntry(input_row2, width=40, font=("Segoe UI", 10), height=22, fg_color=C["background"])
        self._ent_glu.pack(side="left", padx=2)
        
        self._btn_calib = ctk.CTkButton(self._calib_frame, text="Set Baselines", height=24,
                                        font=("Segoe UI Bold", 10), command=self._calibrate_vitals,
                                        fg_color="#30363D", hover_color="#484F58")
        self._btn_calib.pack(pady=(2, 5), padx=10, fill="x")

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

        self._lbl_loss = ctk.CTkLabel(bottom, text="Loss: 0 (0.0%)",
                                     font=("Segoe UI", 11),
                                     text_color="#8B949E")
        self._lbl_loss.pack(side="left", padx=15)

        self._lbl_trend = ctk.CTkLabel(bottom, text="HI Trend: --",
                                       font=("Segoe UI Bold", 11),
                                       text_color=C["accent"])
        self._lbl_trend.pack(side="right", padx=15)

        self._lbl_bp_trend = ctk.CTkLabel(bottom, text="BP Trend: --",
                                          font=("Segoe UI Bold", 11),
                                          text_color=C["pi_very_high"])
        self._lbl_bp_trend.pack(side="right", padx=15)

        self._lbl_hb_trend = ctk.CTkLabel(bottom, text="Hb Trend: --",
                                          font=("Segoe UI Bold", 11),
                                          text_color=C["pi_high"])
        self._lbl_hb_trend.pack(side="right", padx=15)

        self._lbl_glu_trend = ctk.CTkLabel(bottom, text="Glu Trend: --",
                                           font=("Segoe UI Bold", 11),
                                           text_color=C["pi_very_high"])
        self._lbl_glu_trend.pack(side="right", padx=15)

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
            self._last_data_time = time.time()
            
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
                # Collect PR at ~1 Hz to match PRAllData format in training JSONs.
                # At 200Hz we'd otherwise accumulate 200 duplicate values per second.
                _now = time.time()
                if not hasattr(self, '_last_pr_t') or _now - self._last_pr_t >= 1.0:
                    self._pr_for_analysis.append(pkt.hr_avg)
                    self._last_pr_t = _now

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

            # PPG waveform buffer.
            # DISPLAY (live graph): use pleth_wave (1-100), already screen-scaled.
            # ML ANALYSIS: use adc_sample — matches the training JSON 'Pleth' field
            # exactly (same infrared ADC values, ~620k-630k range), giving 655x better
            # resolution than pleth_wave. Only collect when finger is present.
            pleth_val = pkt.pleth_wave
            if pleth_val > 0:
                self._pleth_buf.append(pleth_val)   # display only

            if pkt.is_valid and pkt.adc_sample != 0:
                self._raw_pleth_for_analysis.append(float(pkt.adc_sample))


            # Timestamp
            if self._session_start:
                elapsed = time.time() - self._session_start
                self._time_stamps.append(elapsed)

        # Update graphs periodically (every 5th poll to save CPU)
        if self._total_packets % 10 == 0 and len(self._pleth_buf) > 2:
            self._update_ppg_graph()

        # --- Simple continuous collect → analyse loop ---
        # Fix 5: Use the measured rate (not hardcoded 200Hz).
        # If device is at 100Hz, we need 3000 samples for 30s — not 6000.
        analysis_rate  = getattr(self, '_measured_hz', cfg.SAMPLING_RATE_HZ)
        window_samples = int(cfg.FEATURE_WINDOW_SECONDS * analysis_rate)

        data_len = len(self._raw_pleth_for_analysis)

        if self._session_start:
            # Show collection progress until we have a full window
            progress = min(1.0, data_len / window_samples)
            
            # Handle long disconnections (Stale data protection)
            if self._ble.state != STATE_CONNECTED and not self._simulate:
                if time.time() - self._last_data_time > 120:
                    if len(self._raw_pleth_for_analysis) > 0:
                        print("DISCONNECT: Buffer cleared (Stale data protection — >120s)")
                        self._raw_pleth_for_analysis.clear()
                        self._pr_for_analysis.clear()
                self._progress_bar.configure(progress_color=C["hydration_red"])
            else:
                self._progress_bar.configure(progress_color=C["accent"])

            self._progress_bar.set(progress)
            if data_len < window_samples:
                needed = window_samples - data_len
                pct = (data_len / window_samples) * 100
                self._lbl_phase.configure(
                    text=f"COLLECTING  {data_len}/{window_samples} samples  ({needed} to go…)",
                    text_color=C["hydration_yellow"]
                )
                # Update cards to show progress
                status_text = f"Wait...{pct:.0f}%"
                if self._total_packets % 20 == 0: # Throttle UI updates
                    self._card_bp.set_value(status_text, C["hydration_yellow"])
                    self._card_hb.set_value(status_text, C["hydration_yellow"])
                    self._card_glucose.set_value(status_text, C["hydration_yellow"])
                    self._card_hydration.set_value("Wait", C["hydration_yellow"])
            else:
                self._lbl_phase.configure(
                    text=f"LIVE  |  {data_len} samples  — analysing every ~5s",
                    text_color=C["hydration_green"]
                )

        # --- Accumulate → analyse ---
        # Fix 6: Re-analyse every ~5s of new samples once the initial window is collected.
        # Previously fired only every full 30s window — far too infrequent for a live dashboard.
        if data_len >= window_samples and not self._analysis_running:
            samples_since_last = data_len - self._last_analysis_len
            update_every = int(analysis_rate * 5)   # 5s worth of new samples
            if samples_since_last >= update_every or self._last_analysis_len == 0:
                self._run_analysis()
                self._last_analysis_len = data_len


        # Update footer
        self._lbl_packets.configure(text=f"Packets: {self._total_packets:,}")

        # --- Update Data Loss Stats ---
        connector = self._sim_thread if self._simulate else self._ble
        if connector:
            lost = getattr(connector, 'packets_lost', 0)
            received = getattr(connector, 'packets_received', 0)
            total = lost + received
            if total > 0:
                loss_pct = (lost / total) * 100
                color = "#8B949E" if loss_pct < 1.0 else (C["hydration_yellow"] if loss_pct < 5.0 else C["hydration_red"])
                self._lbl_loss.configure(text=f"Loss: {lost:,} ({loss_pct:.2f}%)", text_color=color)
                
                # Persistent Logging
                if lost > self._last_logged_loss:
                    new_gaps = lost - self._last_logged_loss
                    self._total_loss_events += 1
                    with open(self._loss_log_path, "a") as f:
                        f.write(f"[{time.strftime('%H:%M:%S')}] EVENT #{self._total_loss_events}: Gap of {new_gaps} packets. Total Lost: {lost}\n")
                    self._last_logged_loss = lost

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
            self._ax_hi.set_ylim(-50, 50)  # Expanded range for extreme dehydration

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

        # Use indices if times is not provided or empty
        x_indices = list(range(1, len(times) + 1))

        # --- Blood Pressure ---
        sbp_clean = [(times[i], v) for i, v in enumerate(sbp)
                     if v is not None and i < len(times) and not math.isnan(v) and not math.isinf(v)]
        dbp_clean = [(times[i], v) for i, v in enumerate(dbp)
                     if v is not None and i < len(times) and not math.isnan(v) and not math.isinf(v)]
        if sbp_clean and dbp_clean:
            sx, sv = zip(*sbp_clean); dx, dv = zip(*dbp_clean)
            self._sbp_line.set_data(sx, sv)
            self._dbp_line.set_data(dx, dv)
            all_bp = list(sv) + list(dv)
            self._ax_bp.set_xlim(min(sx + dx) - 0.5, max(sx + dx) + 0.5)
            self._ax_bp.set_ylim(max(0, min(all_bp) - 15), max(all_bp) + 15)
            self._ax_bp.set_title(
                f"Blood Pressure  ▶  SBP: {sv[-1]:.0f}  /  DBP: {dv[-1]:.0f} mmHg",
                color=C["hydration_red"], fontsize=10, fontweight="bold"
            )

        # --- Hemoglobin ---
        hb_clean = [(times[i], v) for i, v in enumerate(hb)
                    if v is not None and i < len(times) and not math.isnan(v) and not math.isinf(v)]
        if hb_clean:
            hx, hv = zip(*hb_clean)
            self._hb_line.set_data(hx, hv)
            self._ax_hb.set_xlim(min(hx) - 0.5, max(hx) + 0.5)
            self._ax_hb.set_ylim(max(0, min(hv) - 3), max(hv) + 3)
            self._ax_hb.set_title(
                f"Hemoglobin  ▶  {hv[-1]:.1f} g/dL",
                color=C["pi_high"], fontsize=10, fontweight="bold"
            )
            # Update trend indicator (tracker updated in analysis/calibration)
            hb_trend_res = self._hb_trend_tracker.get_trend()
            self._lbl_hb_trend.configure(text=f"Hb Trend: {hb_trend_res['trend']}")

        # --- Glucose ---
        glu_clean = [(times[i], v) for i, v in enumerate(glu)
                     if v is not None and i < len(times) and not math.isnan(v) and not math.isinf(v)]
        if glu_clean:
            gx, gv = zip(*glu_clean)
            self._glu_line.set_data(gx, gv)
            self._ax_glu.set_xlim(min(gx) - 0.5, max(gx) + 0.5)
            self._ax_glu.set_ylim(max(0, min(gv) - 25), max(gv) + 25)
            self._ax_glu.set_title(
                f"Glucose  ▶  {gv[-1]:.0f} mg/dL",
                color=C["pi_very_high"], fontsize=10, fontweight="bold"
            )
            # Update trend indicator (tracker updated in analysis/calibration)
            glu_trend_res = self._glu_trend_tracker.get_trend()
            self._lbl_glu_trend.configure(text=f"Glu Trend: {glu_trend_res['trend']}")

        # SBP/DBP Trend Label Update
        bp_trend_res = self._bp_trend_tracker.get_trend()
        self._lbl_bp_trend.configure(text=f"BP Trend: {bp_trend_res['trend']}")

        self._canvas.draw_idle()


    def _run_analysis(self):
        """Run the hydration pipeline on accumulated data."""
        self._analysis_running = True

        def _worker():
            try:
                raw = np.array(self._raw_pleth_for_analysis, dtype=float)

                from scipy.signal import resample as _resample
                
                fs_in = cfg.SIMULATION_SPEED_HZ if self._simulate else cfg.BERRY_DEFAULT_RATE_HZ
                duration = len(raw) / fs_in
                
                # Upsample to 120Hz for model consistency
                n_target = int(round(duration * cfg.MODEL_SAMPLING_RATE_HZ))
                if n_target != len(raw):
                    raw = _resample(raw, n_target)
                
                # New canonical analysis frequency
                fs = cfg.MODEL_SAMPLING_RATE_HZ
                analysis_rate = fs
                window_samples = int(cfg.FEATURE_WINDOW_SECONDS * analysis_rate)
                
                raw_norm = raw

                # --- NEW: Data Integrity Check ---
                # Check if the segment has enough samples (e.g., >85% of expected)
                if len(raw) < (window_samples * 0.85):
                    print(f"ANALYSIS REJECTED: Poor signal integrity. Only {len(raw)}/{window_samples} samples collected.")
                    self._analysis_running = False
                    self.after(0, lambda: self._lbl_phase.configure(text="SIGNAL UNRELIABLE (Data Loss)", text_color=C["hydration_red"]))
                    return

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
                    # User-friendly explanation: Usually NO pulses found in the 30s window
                    msg = "SIGNAL UNRELIABLE (No beats detected)"
                    print(f"DEBUG: Analysis skipped: {msg}. Is the sensor placed correctly?")
                    self.after(0, lambda m=msg: self._lbl_phase.configure(text=m, text_color=C["hydration_red"]))
                    self._analysis_running = False
                    return

                print(f"DEBUG: Running hydration analysis on {len(features)} windows...")
                hydration = analyse_hydration(features)
                results = hydration["results"]

                # Timing Gate:
                # 1. Skip first 10s entirely for stabilization
                # 2. Start Vitals at 40s (10s skip + 30s window) then every 30s
                # 3. Start Hydration at 40s too, then every 3 minutes (180s)
                
                elapsed_s = 0
                if self._session_start:
                    elapsed_s = time.time() - self._session_start
                
                if elapsed_s < 10:
                    print(f"DEBUG: Stabilization skip (Elapsed: {elapsed_s:.1f}s < 10s)")
                    self._analysis_running = False
                    return

                # Vitals Gate (BP, Hb, Glucose)
                should_run_vitals = False
                if elapsed_s >= 40:
                    if self._vitals_first_run or (elapsed_s - self._last_vitals_report_time >= 30):
                        should_run_vitals = True

                if should_run_vitals:
                    print(f"DEBUG: Running multi-vital inference (Elapsed: {elapsed_s:.1f}s)...")
                    seg = raw_norm[-window_samples:]
                    
                    if self._session_start and len(self._raw_pleth_for_analysis) > 0:
                        self._measured_hz = len(self._raw_pleth_for_analysis) / elapsed_s
                        print(f"DEBUG: Actual measured rate: {self._measured_hz:.1f} Hz")

                    # Predict RAW values first (no offsets)
                    vitals = self._engine.predict_vitals(
                        seg,
                        actual_rate_hz=cfg.MODEL_SAMPLING_RATE_HZ,
                        age=getattr(self, '_user_age', None),
                        gender=getattr(self, '_user_gender', None),
                        bmi=getattr(self, '_user_bmi', None),
                        pr_all_data=self._pr_for_analysis[-30:],
                        offsets=None # Get raw first for jump filter/baseline
                    )
                    
                    if vitals:
                        # Store raw for baseline math
                        raw_s = vitals.get("sbp", 0.0)
                        raw_d = vitals.get("dbp", 0.0)
                        raw_h = vitals.get("hb", 0.0)
                        raw_g = vitals.get("glucose", 0.0)
                        self._last_raw_preds = {"sbp": raw_s, "dbp": raw_d, "hb": raw_h, "glucose": raw_g}
                        
                        if self._vitals_calibrated:
                            # 1. Apply Offsets
                            cal_s = raw_s + self._calib_offsets["sbp"]
                            cal_d = raw_d + self._calib_offsets["dbp"]
                            cal_h = raw_h + self._calib_offsets["hb"]
                            cal_g = raw_g + self._calib_offsets["glucose"]
                            
                            # Ignore offset if calculated (raw) BP is within +/- 5 mmHg of calibrated target
                            if self._calib_targets.get("sbp") is not None and abs(raw_s - self._calib_targets["sbp"]) <= 5:
                                cal_s = raw_s
                            if self._calib_targets.get("dbp") is not None and abs(raw_d - self._calib_targets["dbp"]) <= 5:
                                cal_d = raw_d

                            
                            # 4. Update UI cards
                            bp_cat = vitals.get("bp_category", "normal")
                            
                            # 2. Apply Jump Filters
                            smooth_s = self._sbp_filter.update(cal_s)
                            smooth_d = self._dbp_filter.update(cal_d)
                            
                            was_jump_s = abs(smooth_s - cal_s) > 0.1
                            was_jump_d = abs(smooth_d - cal_d) > 0.1

                            # --- NEW: BP Detailed Logging ---
                            log_entry = (
                                f"{time.strftime('%H:%M:%S')},"
                                f"{raw_s:.2f},{raw_d:.2f},"
                                f"{self._calib_offsets['sbp']:.2f},{self._calib_offsets['dbp']:.2f},"
                                f"{cal_s:.2f},{cal_d:.2f},"
                                f"{smooth_s:.2f},{smooth_d:.2f},"
                                f"{bp_cat},"
                                f"{'YES' if (was_jump_s or was_jump_d) else 'NO'}\n"
                            )
                            with open(self._bp_log_path, "a") as f:
                                f.write(log_entry)
                            
                            # 3. Store in History (Smoothed + Calibrated)
                            cycle_num = len(self._vital_times) + 1
                            self._vital_times.append(cycle_num)
                            self._sbp_history.append(smooth_s)
                            self._dbp_history.append(smooth_d)
                            self._hb_history.append(cal_h)
                            self._glucose_history.append(cal_g)
                            
                            # Update UI cards
                            bp_color = (C["hydration_green"] if bp_cat == "normal"
                                        else C["hydration_yellow"] if bp_cat == "hypo"
                                        else C["hydration_red"])
                            
                            # Additional safety: High BP color check
                            if smooth_s > 140 or smooth_d > 90: bp_color = C["hydration_red"]
                            
                            self.after(0, lambda s=smooth_s, d=smooth_d, c=bp_color:
                                       self._card_bp.set_value(f"{s:.0f}/{d:.0f}", c))
                            
                            # Update Trend Tracker
                            self._bp_trend_tracker.update({"sbp": smooth_s, "dbp": smooth_d, "bp_category": bp_cat})
                            bp_trend = self._bp_trend_tracker.get_trend()
                            self.after(0, lambda t=f"BP Trend: {bp_trend['trend']}": self._lbl_bp_trend.configure(text=t))
                            
                            self._hb_trend_tracker.update(cal_h)
                            hb_trend = self._hb_trend_tracker.get_trend()
                            self.after(0, lambda h=cal_h, t=hb_trend['trend']: self._update_hb_ui(h, t))
                            
                            self._glu_trend_tracker.update(cal_g)
                            glu_trend = self._glu_trend_tracker.get_trend()
                            self.after(0, lambda g=cal_g, t=glu_trend['trend']: self._update_glu_ui(g, t))
                            
                            # Update Graphs
                            t_snap = list(self._vital_times)
                            s_snap = list(self._sbp_history)
                            d_snap = list(self._dbp_history)
                            h_snap = list(self._hb_history)
                            g_snap = list(self._glucose_history)
                            self.after(0, lambda t=t_snap, s=s_snap, d=d_snap, h=h_snap, g=g_snap: \
                                       self._update_vital_graphs(t, s, d, h, g))
                        else:
                            # Ghost Mode
                            self.after(0, lambda: self._card_bp.set_value("Calibrate...", C["hydration_yellow"]))
                            self.after(0, lambda: self._card_hb.set_value("Calibrate...", C["hydration_yellow"]))
                            self.after(0, lambda: self._card_glucose.set_value("Calibrate...", C["hydration_yellow"]))
                        
                    self._last_vitals_report_time = elapsed_s
                    self._vitals_first_run = False

                # ── Hydration UI ──
                # Timing Gate: Every 3 minutes (180s) after first 40s
                should_run_hydration = False
                if elapsed_s >= 40:
                    if (self._last_hydration_report_time == 0) or (elapsed_s - self._last_hydration_report_time >= 180):
                        should_run_hydration = True

                if not should_run_hydration:
                    # Show calibration status if still in first 2 mins
                    if elapsed_s < (cfg.BASELINE_WINDOW_MINUTES * 60):
                        remain = int((cfg.BASELINE_WINDOW_MINUTES * 60) - elapsed_s)
                        msg = f"Calibrating ({remain}s)"
                        self.after(0, lambda m=msg: self._card_hydration.set_value(m, C["hydration_yellow"]))
                    
                    self._analysis_running = False
                    return

                self._last_hydration_report_time = elapsed_s

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

    def _calibrate_vitals(self):
        """Handle manual vitals calibration (BP, Hb, Glucose) from UI input."""
        try:
            val_s = self._ent_sbp.get().strip()
            val_d = self._ent_dbp.get().strip()
            val_h = self._ent_hb.get().strip()
            val_g = self._ent_glu.get().strip()
            
            # Identify which tags were actually provided
            sbp = float(val_s) if val_s else None
            dbp = float(val_d) if val_d else None
            hb = float(val_h) if val_h else None
            glu = float(val_g) if val_g else None
            
            if sbp is None and dbp is None and hb is None and glu is None:
                return

            print(f"CALIBRATION: Manual values provided -> SBP:{sbp}, DBP:{dbp}, Hb:{hb}, Glu:{glu}")
            
            # --- CRITICAL FIX: Calculate Offsets ---
            # Formula: Offset = User_Input - Last_Raw_Model_Prediction
            if sbp is not None and dbp is not None:
                raw_s = self._last_raw_preds.get("sbp")
                raw_d = self._last_raw_preds.get("dbp")
                if raw_s is not None and raw_d is not None:
                    self._calib_offsets["sbp"] = sbp - raw_s
                    self._calib_offsets["dbp"] = dbp - raw_d
                    self._calib_targets["sbp"] = sbp
                    self._calib_targets["dbp"] = dbp
                    print(f"DEBUG: Calibrated BP. Offsets: SBP={self._calib_offsets['sbp']:.1f}, DBP={self._calib_offsets['dbp']:.1f}")
                else:
                    print("CALIBRATION WARNING: No raw BP prediction to baseline against yet.")

            if hb is not None:
                raw_h = self._last_raw_preds.get("hb")
                if raw_h is not None:
                    self._calib_offsets["hb"] = hb - raw_h
                    print(f"DEBUG: Calibrated Hb. Offset: {self._calib_offsets['hb']:.2f}")

            if glu is not None:
                raw_g = self._last_raw_preds.get("glucose")
                if raw_g is not None:
                    self._calib_offsets["glucose"] = glu - raw_g
                    print(f"DEBUG: Calibrated Glucose. Offset: {self._calib_offsets['glucose']:.1f}")
            
            self._vitals_calibrated = True # --- NEW: Unlock UI display ---
            
            # Seed jump filters with the manual baseline for a smooth transition
            if sbp is not None:
                self._sbp_filter.current_val = sbp
            if dbp is not None:
                self._dbp_filter.current_val = dbp
            # ---------------------------------------
            
            # Validate BP if provided
            if sbp is not None and dbp is not None:
                if not (cfg.BP_SBP_LIMITS[0] <= sbp <= cfg.BP_SBP_LIMITS[1]) or \
                   not (cfg.BP_DBP_LIMITS[0] <= dbp <= cfg.BP_DBP_LIMITS[1]):
                    print(f"CALIBRATION ERROR: BP {sbp}/{dbp} out of physiological limits.")
                    sbp, dbp = None, None
            
            # Update histories and trackers
            now = time.time()
            cycle_num = len(self._vital_times) + 1
            self._vital_times.append(cycle_num)
            
            self._sbp_history.append(sbp if sbp is not None else float("nan"))
            self._dbp_history.append(dbp if dbp is not None else float("nan"))
            self._hb_history.append(hb if hb is not None else float("nan"))
            self._glucose_history.append(glu if glu is not None else float("nan"))
            
            if sbp is not None and dbp is not None:
                self._bp_trend_tracker.update({"sbp": sbp, "dbp": dbp, "bp_category": "normal"})
                self._card_bp.set_value(f"{sbp:.0f}/{dbp:.0f}", C["hydration_green"])
                
            if hb is not None:
                self._hb_trend_tracker.update(hb)
                self._card_hb.set_value(f"{hb:.1f}", C["hydration_green"])
                
            if glu is not None:
                self._glu_trend_tracker.update(glu)
                self._card_glucose.set_value(f"{glu:.0f}", C["hydration_green"])
            
            # Refresh graphs
            self._update_vital_graphs(list(self._vital_times), list(self._sbp_history), 
                                      list(self._dbp_history), list(self._hb_history), 
                                      list(self._glucose_history))
            
            # Clear entries
            self._ent_sbp.delete(0, 'end')
            self._ent_dbp.delete(0, 'end')
            self._ent_hb.delete(0, 'end')
            self._ent_glu.delete(0, 'end')
            
        except ValueError:
            print("CALIBRATION ERROR: Invalid numeric input in calibration fields.")

    def _update_bp_ui(self, sbp, dbp, color, trend):
        self._card_bp.set_value(f"{sbp:.0f}/{dbp:.0f}", color)
        self._lbl_bp_trend.configure(text=f"BP Trend: {trend}")

    def _update_hb_ui(self, hb, trend):
        self._card_hb.set_value(f"{hb:.1f}", C["hydration_green"])
        self._lbl_hb_trend.configure(text=f"Hb Trend: {trend}")

    def _update_glu_ui(self, glu, trend):
        glu_col = (C["hydration_green"] if 70 <= glu <= 140 else C["hydration_yellow"])
        self._card_glucose.set_value(f"{glu:.0f}", glu_col)
        self._lbl_glu_trend.configure(text=f"Glu Trend: {trend}")

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
