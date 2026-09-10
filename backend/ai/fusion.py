"""
VoiceShield — Score Fusion & Classification Engine
Combines multi-branch AI risk signals with dynamic renormalization,
confidence estimation, and temporal smoothing.
Smart India Hackathon 2026 | PS ID: 26104
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional

from .config import DEFAULT_CONFIG, PipelineConfig
from .temporal import TemporalSmoother

logger = logging.getLogger("VoiceShieldFusion")


class FusionEngine:
    """
    Multi-model risk fusion and classification engine.
    Applies configurable weights, dynamic renormalization for absent/unavailable branches,
    confidence calculation, and temporal smoothing across audio hops.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config
        self.weights = dict(config.weights)
        self.low_threshold = config.low_threshold
        self.high_threshold = config.high_threshold

        # Initialize temporal smoother per session
        self.smoother = TemporalSmoother(
            alpha=config.ema_alpha,
            max_step_delta=config.max_step_delta,
            history_size=config.history_size,
            min_speech_duration=config.min_speech_duration,
            low_threshold=config.low_threshold,
            high_threshold=config.high_threshold
        )

    def reset(self):
        """Reset temporal state for new call."""
        self.smoother.reset()

    def fuse(
        self,
        vad_result: Dict[str, Any],
        rawnet_result: Dict[str, Any],
        aasist_result: Dict[str, Any],
        ecapa_result: Dict[str, Any],
        bispectral_result: Dict[str, Any],
        hop_duration_sec: float = 0.20,
        timings: Optional[Dict[str, float]] = None,
        total_pipeline_ms: int = 0
    ) -> Dict[str, Any]:
        """
        Combine all available branch scores into a calibrated, temporally smoothed risk score.
        """
        vad_prob = float(vad_result.get("probability", 0.0))
        is_speech = bool(vad_result.get("is_speech", False))
        speech_ratio = float(vad_result.get("speech_ratio", 0.0))

        # 1. Handle No-Speech / Silence via Temporal Smoother
        if not is_speech:
            final_risk, conf, classification, status = self.smoother.update(
                raw_risk=0.05,
                is_speech=False,
                hop_duration_sec=hop_duration_sec,
                vad_prob=vad_prob,
                active_models_count=0,
                total_models=4
            )
            return {
                "type": "ai_result",
                "status": status,
                "risk_score": final_risk,
                "confidence": conf,
                "classification": classification,
                "rawnet_risk": None,
                "aasist_risk": None,
                "speaker_similarity": None,
                "speaker_risk": None,
                "bispectral_risk": None,
                "vad_probability": round(vad_prob, 4),
                "active_models": [],
                "latency_ms": total_pipeline_ms,
                "timings": timings or {},
                "timestamp": int(time.time())
            }

        # 2. Collect available branch risks
        active_scores = {}
        active_weights = {}

        # Branch 1: RawNet3
        rawnet_risk = rawnet_result.get("rawnet_risk")
        if rawnet_result.get("available") and rawnet_risk is not None:
            active_scores["rawnet3"] = float(rawnet_risk)
            active_weights["rawnet3"] = self.weights.get("rawnet3", 0.35)

        # Branch 2: AASIST (Primary lightweight spectral anti-spoofing)
        aasist_risk = aasist_result.get("aasist_risk")
        if aasist_result.get("available") and aasist_risk is not None:
            active_scores["aasist"] = float(aasist_risk)
            active_weights["aasist"] = self.weights.get("aasist", 0.35)

        # Branch 3: ECAPA-TDNN (Speaker Verification)
        speaker_risk = ecapa_result.get("speaker_risk")
        speaker_sim = ecapa_result.get("speaker_similarity")
        if ecapa_result.get("available") and speaker_risk is not None:
            active_scores["ecapa"] = float(speaker_risk)
            active_weights["ecapa"] = self.weights.get("ecapa", 0.20)

        # Branch 4: Bispectral Analysis (Higher-Order Spectra)
        bispectral_risk = bispectral_result.get("bispectral_risk")
        if bispectral_result.get("available") and bispectral_risk is not None:
            active_scores["bispectral"] = float(bispectral_risk)
            active_weights["bispectral"] = self.weights.get("bispectral", 0.10)

        # 3. Dynamic Weight Renormalization
        if not active_scores:
            return {
                "type": "ai_result",
                "status": "MODELS_UNAVAILABLE",
                "risk_score": None,
                "confidence": 0.0,
                "classification": "UNAVAILABLE",
                "rawnet_risk": None,
                "aasist_risk": None,
                "speaker_similarity": speaker_sim,
                "speaker_risk": None,
                "bispectral_risk": None,
                "vad_probability": round(vad_prob, 4),
                "active_models": [],
                "latency_ms": total_pipeline_ms,
                "timings": timings or {},
                "timestamp": int(time.time())
            }

        total_active_weight = sum(active_weights.values())
        raw_weighted_risk = sum(
            active_scores[name] * (active_weights[name] / total_active_weight)
            for name in active_scores
        )
        raw_weighted_risk = float(np.clip(raw_weighted_risk, 0.0, 1.0))

        # 4. Temporal Smoothing & Duration Gating
        final_risk, confidence, classification, status = self.smoother.update(
            raw_risk=raw_weighted_risk,
            is_speech=True,
            hop_duration_sec=hop_duration_sec,
            vad_prob=vad_prob,
            active_models_count=len(active_scores),
            total_models=4
        )

        return {
            "type": "ai_result",
            "status": status,
            "risk_score": final_risk,
            "confidence": confidence,
            "classification": classification,
            "raw_risk": round(raw_weighted_risk, 4),
            "rawnet_risk": round(rawnet_risk, 4) if rawnet_risk is not None else None,
            "aasist_risk": round(aasist_risk, 4) if aasist_risk is not None else None,
            "speaker_similarity": round(speaker_sim, 4) if speaker_sim is not None else None,
            "speaker_risk": round(speaker_risk, 4) if speaker_risk is not None else None,
            "bispectral_risk": round(bispectral_risk, 4) if bispectral_risk is not None else None,
            "vad_probability": round(vad_prob, 4),
            "speech_ratio": round(speech_ratio, 4),
            "active_models": list(active_scores.keys()),
            "latency_ms": total_pipeline_ms,
            "timings": timings or {},
            "timestamp": int(time.time())
        }
