import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "water"))
import numpy as np

fs = 200
t = np.linspace(0, 30, 30 * fs)
ppg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)
ppg += 0.05 * np.random.randn(len(ppg))

from water.inference_engine import VitalInferenceEngine

print("Loading engine...")
engine = VitalInferenceEngine()
print("Models loaded:", list(engine.models.keys()))

print("Running predict_vitals...")
result = engine.predict_vitals(ppg, actual_rate_hz=200.0)
print("Result:", result)
