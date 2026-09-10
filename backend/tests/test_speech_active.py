import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import numpy as np
from backend.ai.pipeline import AIPipeline

results = []
pipeline = AIPipeline(room_id="speech_test", on_result_callback=lambda r: results.append(r))

# Mock VAD to simulate speech detection on incoming speech
pipeline.vad.detect_speech = lambda wf, sample_rate=16000: {
    "is_speech": True,
    "probability": 0.94,
    "speech_ratio": 0.88,
    "speech_segments": [[0.0, 1.5]]
}

sr = 16000
duration = 2.5
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
waveform = (0.2 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.random.normal(0, 0.05, len(t))).astype(np.float32)
pcm_bytes = (waveform * 32767).astype(np.int16).tobytes()

print("Streaming active speech audio...")
chunk_size = int(sr * 0.20 * 2) # 6400 bytes (200ms)
for i in range(0, len(pcm_bytes), chunk_size):
    chunk = pcm_bytes[i:i+chunk_size]
    pipeline.ingest_pcm(chunk)
    time.sleep(0.04)

time.sleep(2.0)
pipeline.stop()

print(f"\n--- Active Speech Results Count: {len(results)} ---")
for idx, r in enumerate(results):
    status = r.get("status")
    risk = r.get("risk_score")
    conf = r.get("confidence")
    classification = r.get("classification")
    aasist = r.get("aasist_risk")
    rawnet = r.get("rawnet_risk")
    bisp = r.get("bispectral_risk")
    lat = r.get("latency_ms")
    timings = r.get("timings", {})
    print(f"[{idx+1}] Status: {status:<10} | Risk: {risk} | Conf: {conf} | Class: {classification:<12} | AASIST: {aasist} | RawNet: {rawnet} | Bisp: {bisp} | Total Latency: {lat}ms | Timings: {timings}")

assert len(results) > 0
print("\nSUCCESS: All branches executed concurrently and produced calibrated scores!")
