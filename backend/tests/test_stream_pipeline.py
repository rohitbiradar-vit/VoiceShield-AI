import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import numpy as np
from backend.ai.pipeline import AIPipeline
from backend.ai.config import DEFAULT_CONFIG

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import numpy as np
from backend.ai.pipeline import AIPipeline
from backend.ai.config import DEFAULT_CONFIG

results = []
pipeline = AIPipeline(room_id="test_room", on_result_callback=lambda r: results.append(r))

# Generate speech-like signal: 120Hz fundamental + formants at 700, 1220, 2600 Hz modulated by syllabic rhythm (4 Hz)
sr = 16000
duration = 3.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4 * t)) # 4Hz syllabic modulation
f0 = 130.0
speech = envelope * (
    0.4 * np.sin(2 * np.pi * f0 * t) +
    0.3 * np.sin(2 * np.pi * 700 * t) +
    0.2 * np.sin(2 * np.pi * 1220 * t) +
    0.1 * np.sin(2 * np.pi * 2600 * t) +
    0.05 * np.random.normal(0, 0.1, len(t))
).astype(np.float32)

pcm_bytes = (speech / np.max(np.abs(speech)) * 25000).astype(np.int16).tobytes()

print("Streaming audio into real-time pipeline...")
chunk_size = int(sr * 0.20 * 2) # 6400 bytes (200ms)
for i in range(0, len(pcm_bytes), chunk_size):
    chunk = pcm_bytes[i:i+chunk_size]
    pipeline.ingest_pcm(chunk)
    time.sleep(0.04)

time.sleep(2.0)
pipeline.stop()

print(f"\n--- Processed Windows: {len(results)} ---")
for idx, r in enumerate(results):
    print(
        f"[{idx+1}] Status: {r.get('status'):<10} | Risk: {r.get('risk_score')} | "
        f"Conf: {r.get('confidence')} | Class: {r.get('classification'):<12} | "
        f"AASIST: {r.get('aasist_risk')} | RawNet: {r.get('rawnet_risk')} | "
        f"Lat: {r.get('latency_ms')}ms | Timings: {r.get('timings')}"
    )

print("\nPipeline test complete!")
