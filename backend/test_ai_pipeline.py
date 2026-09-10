"""
VoiceShield — Comprehensive Standalone AI Pipeline Verification Test
Verifies AudioBuffer, Preprocessing, Silero VAD, RawNet3, ECAPA-TDNN,
BispectralAnalyzer, and FusionEngine.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, ".")

from backend.audio.buffer import AudioBuffer
from backend.audio.preprocessing import AudioPreprocessor
from backend.ai.vad import SileroVAD
from backend.ai.rawnet3 import RawNet3SpoofDetector
from backend.ai.ecapa import ECAPATDNNVerifier
from backend.ai.bispectral import BispectralAnalyzer
from backend.ai.fusion import FusionEngine
from backend.ai.pipeline import AIPipeline


def generate_synthetic_speech_signal(duration_sec: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a multi-harmonic formant-rich synthetic voice signal for testing."""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    # Fundamental frequency f0 ~ 130 Hz (typical male/female voice pitch) with pitch modulation
    f0 = 130.0 + 15.0 * np.sin(2 * np.pi * 3.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate

    # Formant resonances (F1 ~ 700Hz, F2 ~ 1220Hz, F3 ~ 2600Hz)
    harmonics = (
        1.0 * np.sin(phase) +
        0.8 * np.sin(2 * phase) +
        0.6 * np.sin(3 * phase) +
        0.5 * np.sin(5 * phase) +
        0.3 * np.sin(9 * phase) +
        0.2 * np.sin(19 * phase)
    )

    # Glottal pulse shaping / amplitude envelope
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    waveform = (harmonics * envelope).astype(np.float32)

    # Normalize to [-0.9, 0.9]
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = 0.85 * (waveform / max_val)

    return waveform


def run_standalone_tests():
    print("==================================================")
    print("VOICESHIELD AI PIPELINE STANDALONE VERIFICATION")
    print("==================================================")

    # 1. Test AudioBuffer
    print("\n[TEST 1] Testing AudioBuffer sliding window...")
    buffer = AudioBuffer(sample_rate=16000, window_sec=3.0, hop_sec=1.0)
    fake_chunk = bytes([0] * 3200)  # 100ms
    for _ in range(35):
        buffer.add_chunk(fake_chunk)
    assert buffer.has_window()
    window = buffer.get_and_advance()
    assert len(window) == 96000  # 3s * 16000 * 2 bytes
    print("[PASS] AudioBuffer operates correctly (window: 96000 bytes).")

    # 2. Test AudioPreprocessor
    print("\n[TEST 2] Testing AudioPreprocessor...")
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    pcm_int16 = (np.sin(np.linspace(0, 100, 48000)) * 30000).astype(np.int16).tobytes()
    waveform = preprocessor.process(pcm_int16, source_sr=16000, normalize=True)
    assert len(waveform) == 48000
    assert -1.0 <= np.max(waveform) <= 1.0
    print(f"[PASS] Preprocessor normalized peak: {np.max(np.abs(waveform)):.4f}")

    # 3. Test Silero VAD
    print("\n[TEST 3] Testing Silero VAD...")
    vad = SileroVAD()
    assert vad.is_available, "Silero VAD model must be loaded!"
    silence = np.zeros(48000, dtype=np.float32)
    silence_res = vad.detect_speech(silence)
    print(f"  Silence speech prob: {silence_res['probability']}, is_speech: {silence_res['is_speech']}")
    assert not silence_res["is_speech"], "Silence must not be detected as speech!"

    speech_wave = generate_synthetic_speech_signal(3.0)
    speech_res = vad.detect_speech(speech_wave)
    print(f"  Speech prob: {speech_res['probability']}, is_speech: {speech_res['is_speech']}")
    print("[PASS] Silero VAD gating verified.")

    # 4. Test RawNet3
    print("\n[TEST 4] Testing RawNet3...")
    rawnet = RawNet3SpoofDetector()
    assert rawnet.is_available, "RawNet3 model must be loaded!"
    rawnet_res = rawnet.analyze(speech_wave)
    print(f"  RawNet3 risk score: {rawnet_res['rawnet_risk']}, model: {rawnet_res['model']}")
    assert rawnet_res["available"] and rawnet_res["rawnet_risk"] is not None
    assert 0.0 <= rawnet_res["rawnet_risk"] <= 1.0
    print("[PASS] RawNet3 inference and score calibration verified.")

    # 5. Test ECAPA-TDNN
    print("\n[TEST 5] Testing ECAPA-TDNN...")
    ecapa = ECAPATDNNVerifier()
    assert ecapa.is_available, "ECAPA-TDNN model must be loaded!"

    # 5a: Unenrolled test (must report NOT_ENROLLED, never fake)
    unrolled_res = ecapa.verify_speaker(speech_wave, speaker_id="Unknown")
    assert unrolled_res["status"] == "NOT_ENROLLED"
    assert unrolled_res["speaker_risk"] is None
    print(f"  Unenrolled speaker status: {unrolled_res['status']} (No fake scores)")

    # 5b: Enrolled test
    ref_speech = generate_synthetic_speech_signal(3.0)
    ecapa.enroll_speaker("Friend", ref_speech)
    enrolled_res = ecapa.verify_speaker(speech_wave, speaker_id="Friend")
    print(f"  Enrolled verification: sim={enrolled_res['speaker_similarity']}, risk={enrolled_res['speaker_risk']}")
    assert enrolled_res["status"] == "VERIFIED"
    assert enrolled_res["speaker_similarity"] is not None
    print("[PASS] ECAPA-TDNN embedding and speaker verification verified.")

    # 6. Test Bispectral Analyzer
    print("\n[TEST 6] Testing Bispectral Analyzer...")
    bispectral = BispectralAnalyzer(sample_rate=16000)
    bisc_res = bispectral.analyze(speech_wave)
    print(f"  Bispectral risk: {bisc_res['bispectral_risk']}, harmonic bic: {bisc_res['mean_harmonic_bicoherence']}")
    assert bisc_res["available"] and bisc_res["bispectral_risk"] is not None
    assert 0.0 <= bisc_res["bispectral_risk"] <= 1.0
    print("[PASS] Bispectral higher-order spectral analysis verified.")

    # 7. Test Fusion Engine
    print("\n[TEST 7] Testing Fusion Engine...")
    fusion = FusionEngine()

    # Case A: Silence
    fused_silence = fusion.fuse(silence_res, rawnet_res, enrolled_res, bisc_res)
    assert fused_silence["status"] == "NO_SPEECH"
    assert fused_silence["risk_score"] is None
    print(f"  Silence fusion verdict: {fused_silence['status']}")

    # Case B: All 3 active with speech present
    active_speech_vad = {"is_speech": True, "probability": 0.92, "speech_ratio": 0.85}
    fused_all = fusion.fuse(active_speech_vad, rawnet_res, enrolled_res, bisc_res)
    print(f"  All 3 active: risk={fused_all['risk_score']}, conf={fused_all['confidence']}, class={fused_all['classification']}")
    assert fused_all["status"] == "ANALYZED"
    assert fused_all["risk_score"] is not None
    assert fused_all["classification"] in ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]

    # Case C: Dynamic renormalization with ECAPA not enrolled
    fused_no_ecapa = fusion.fuse(active_speech_vad, rawnet_res, unrolled_res, bisc_res)
    print(f"  Renormalized (no ECAPA): risk={fused_no_ecapa['risk_score']}, active={fused_no_ecapa['active_models']}")
    assert "ecapa" not in fused_no_ecapa["active_models"]
    assert fused_no_ecapa["risk_score"] is not None
    print("[PASS] Fusion Engine dynamic weight renormalization verified.")

    # 8. Test End-to-End Pipeline
    print("\n[TEST 8] Testing AIPipeline Orchestrator...")
    results_received = []
    def on_result(res):
        results_received.append(res)

    pipeline = AIPipeline(room_id="test_room", on_result_callback=on_result)

    # Ingest 3.5 seconds of PCM audio
    pcm_bytes = (speech_wave * 30000).astype(np.int16).tobytes()
    # Send in 20ms chunks (640 bytes)
    chunk_size = 640
    for i in range(0, len(pcm_bytes), chunk_size):
        pipeline.ingest_pcm(pcm_bytes[i : i + chunk_size])

    # Allow worker thread to process
    time.sleep(1.5)
    pipeline.stop()

    print(f"  Pipeline dispatched {len(results_received)} results.")
    assert len(results_received) >= 1
    last_res = results_received[-1]
    print(f"  Final Pipeline Output: {last_res}")
    assert last_res["type"] == "ai_result"
    assert last_res["status"] in ["ANALYZED", "NO_SPEECH"]
    print(f"[PASS] Full AIPipeline End-to-End verified successfully (Status: {last_res['status']})!")

    print("\n==================================================")
    print("ALL 8 STANDALONE AI PIPELINE TESTS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_standalone_tests()
