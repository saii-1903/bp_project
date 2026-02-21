"""
config.py — Central configuration for the PPG Hydration Trend Discovery Pipeline.

All constants, filter specifications, smoothing parameters, and plot styling
are defined here for easy tuning and reproducibility.
"""

import os

# ─── Sampling & Timing ───────────────────────────────────────────────
SAMPLING_RATE_HZ = 200                # PPG sensor sampling rate
SIMULATION_DURATION_MINUTES = 60     # Total simulated duration
MEASUREMENT_DURATION_SECONDS = 30    # Active PPG collection phase
CONNECTION_DELAY_SECONDS = 10        # Expected initial link delay

# ─── Simulation Phases (minutes) ─────────────────────────────────────
# The simulator walks through these phases sequentially:
PHASE_BASELINE_MIN = 10              # Known-hydrated warm-up
PHASE_DEHYDRATION_MIN = 20           # Progressive dehydration ramp
PHASE_WATER_INTAKE_MIN = 5           # Acute water intake event
PHASE_REHYDRATION_MIN = 25           # Post-intake recovery

# ─── Cardiac Parameters ─────────────────────────────────────────────
HEART_RATE_BPM_BASELINE = 72         # Resting heart rate
HEART_RATE_BPM_DEHYDRATED = 88       # Elevated HR during dehydration
SYSTOLIC_AMPLITUDE_BASELINE = 1.0    # Normalised AC peak height
DICROTIC_NOTCH_RATIO = 0.35         # Notch depth relative to systole
DIASTOLIC_AMPLITUDE_RATIO = 0.25    # Diastolic peak relative to systole

# ─── Noise & Artefacts ──────────────────────────────────────────────
NOISE_STD = 0.02                     # Gaussian noise standard deviation
MOTION_ARTIFACT_PROBABILITY = 0.005  # Per-sample probability of burst
MOTION_ARTIFACT_AMPLITUDE = 0.3      # Motion burst peak amplitude

# ─── Respiratory Modulation ──────────────────────────────────────────
RESP_RATE_HZ = 0.25                  # ~15 breaths per minute
RIIV_AMPLITUDE = 0.04                # Baseline (DC) modulation depth
RIAV_AMPLITUDE = 0.06                # Pulse-amplitude modulation depth

# ─── Signal Processing ──────────────────────────────────────────────
BANDPASS_LOW_HZ = 0.5                # Highpass cut-off
BANDPASS_HIGH_HZ = 5.0               # Lowpass cut-off
BANDPASS_ORDER = 5                   # Elliptic filter order
BANDPASS_RIPPLE_DB = 0.1             # Passband ripple
BANDPASS_ATTENUATION_DB = 40         # Stopband attenuation

# ─── Feature Extraction ─────────────────────────────────────────────
FEATURE_WINDOW_SECONDS = 30          # Window size for aggregated features
FEATURE_WINDOW_OVERLAP = 0.5         # 50 % overlap between windows
MIN_PEAKS_PER_WINDOW = 5             # Minimum beats to accept a window
PEAK_DISTANCE_SAMPLES = 20           # Minimum distance between peaks

# ─── Hydration Engine ───────────────────────────────────────────────
BASELINE_WINDOW_MINUTES = 0.5          # Initial window for personal baseline (30s)
EWMA_SPAN_MINUTES = 5               # EWMA smoothing span
TREND_WINDOW_POINTS = 10             # Sliding window for slope estimation
KENDALL_SIGNIFICANCE = 0.05          # p-value threshold for Kendall test

# Hydration classification thresholds (HI percentages)
HI_FULLY_HYDRATED = 0               # HI >= 0
HI_MILD_DEHYDRATION = -10           # -10 <= HI < 0
HI_MODERATE_DEHYDRATION = -25       # -25 <= HI < -10
HI_EXTREME_DEHYDRATION = -40        # HI < -25

# ─── Visualisation ───────────────────────────────────────────────────
OUTPUT_DIR = "output"
FIGURE_DPI = 150
FIGURE_WIDTH = 18
FIGURE_HEIGHT = 22

# Dark-themed colour palette
COLORS = {
    "background":       "#0D1117",
    "panel_bg":         "#161B22",
    "text":             "#C9D1D9",
    "text_dim":         "#8B949E",
    "accent":           "#58A6FF",
    "raw_signal":       "#8B949E",
    "filtered_signal":  "#58A6FF",
    "pi_line":          "#79C0FF",
    "amplitude_line":   "#D2A8FF",
    "ba_ratio_line":    "#FFA657",
    "hydration_green":  "#3FB950",
    "hydration_yellow": "#D29922",
    "hydration_orange": "#DB6D28",
    "hydration_red":    "#F85149",
    "pi_low":           "#DB6D28", # Orange-ish
    "pi_high":          "#388BFD", # Blue
    "pi_very_high":     "#BC8CFF", # Purple
    "trend_arrow":      "#F0F6FC",
    "grid":             "#21262D",
}

# ─── Berry BLE Protocol (v1.5) ──────────────────────────────────────
BERRY_SERVICE_UUID = "49535343-FE7D-4AE5-8FA9-9FAFD205E455"
BERRY_SEND_CHAR_UUID = "49535343-1E4D-4BD9-BA61-23C647249616"  # Notify
BERRY_RECV_CHAR_UUID = "49535343-8841-43F4-A8D4-ECBE34729BB3"  # Write

BERRY_PACKET_SIZE = 20
BERRY_HEADER = bytes([0xFF, 0xAA])

# Host commands
BERRY_CMD_50HZ = 0xF0
BERRY_CMD_100HZ = 0xF1
BERRY_CMD_200HZ = 0xF2
BERRY_CMD_1HZ = 0xF3
BERRY_CMD_ADC_ORIGINAL = 0xF4
BERRY_CMD_ADC_FILTERED = 0xF5
BERRY_CMD_STOP = 0xF6
BERRY_CMD_SW_VERSION = 0xFF
BERRY_CMD_HW_VERSION = 0xFE

BERRY_DEFAULT_RATE_HZ = 200          # Device sampling rate

# Invalid-value sentinels
BERRY_SPO2_INVALID = 127
BERRY_HR_INVALID = 255
BERRY_PI_INVALID = 0
BERRY_RR_INVALID = 0

# ─── Model Paths ───────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
BP_MODEL_CONFIG = {
    "classifier": os.path.join(MODEL_DIR, "classifier.pkl"),
    "global_scaler": os.path.join(MODEL_DIR, "global_feature_scaler.pkl"),
    "hypo": os.path.join(MODEL_DIR, "hypo_models.pkl"),
    "normal": os.path.join(MODEL_DIR, "normal_models.pkl"),
    "hyper": os.path.join(MODEL_DIR, "hyper_models.pkl"),
}
HB_GLU_MODEL_CONFIG = {
    "hb_scaler": os.path.join(MODEL_DIR, "scaler_hb.pkl"),
    "hb_model": os.path.join(MODEL_DIR, "hb_regressor.pkl"),
    "glucose_scaler": os.path.join(MODEL_DIR, "scaler_glucose.pkl"),
    "glucose_model": os.path.join(MODEL_DIR, "glucose_regressor.pkl"),
}

# ─── Dashboard Settings ────────────────────────────────────────────
DASHBOARD_WIDTH = 1400
DASHBOARD_HEIGHT = 850
DASHBOARD_POLL_MS = 50               # GUI queue-poll interval
LIVE_WAVEFORM_SECONDS = 10           # Rolling PPG window
LIVE_WAVEFORM_MAX_POINTS = 500       # Max points in waveform graph
SIMULATION_SPEED_HZ = 200            # Simulation packet rate (match device)
