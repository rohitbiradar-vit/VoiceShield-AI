"""
VoiceShield — Temporal Scoring, Rate Limiting & Confidence Engine
Eliminates single-window score jumps, smooths risk transitions, and enforces multi-window confirmation.
Smart India Hackathon 2026 | PS ID: 26104
"""

import collections
import logging
import numpy as np
from typing import Tuple, Optional, List

logger = logging.getLogger("VoiceShieldTemporal")


class TemporalSmoother:
    """
    Temporal scoring engine implementing:
    1. Exponential Moving Average (EMA) smoothing
    2. Controlled rate limiting (prevents 20% -> 90% single-step jumps)
    3. Multi-window persistent spoof confirmation
    4. Audio duration gating (status: ANALYZING before sufficient speech)
    5. Confidence estimation based on coverage, duration, and temporal consistency
    """

    def __init__(
        self,
        alpha: float = 0.30,
        max_step_delta: float = 0.12,
        history_size: int = 5,
        min_speech_duration: float = 1.0,
        low_threshold: float = 0.40,
        high_threshold: float = 0.70
    ):
        self.alpha = alpha
        self.max_step_delta = max_step_delta
        self.history_size = history_size
        self.min_speech_duration = min_speech_duration
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self.history: collections.deque = collections.deque(maxlen=history_size)
        self.smoothed_score: Optional[float] = None
        self.accumulated_speech_sec: float = 0.0
        self.silence_sec: float = 0.0

    def reset(self):
        """Reset temporal state when call ends or re-initiates."""
        self.history.clear()
        self.smoothed_score = None
        self.accumulated_speech_sec = 0.0
        self.silence_sec = 0.0

    def update(
        self,
        raw_risk: float,
        is_speech: bool,
        hop_duration_sec: float,
        vad_prob: float,
        active_models_count: int,
        total_models: int = 4
    ) -> Tuple[float, float, str, str]:
        """
        Update temporal smoother with new inference window.
        Returns: (final_risk, confidence, classification, status)
        """
        # 1. Handle No-Speech / Silence state
        if not is_speech:
            self.silence_sec += hop_duration_sec
            if self.silence_sec > 2.5:
                # Decay speech accumulator during sustained silence (> 2.5s)
                self.accumulated_speech_sec = max(0.0, self.accumulated_speech_sec - hop_duration_sec * 2)
                if self.silence_sec > 5.0:
                    self.history.clear()

            if self.smoothed_score is not None:
                # Slowly decay towards baseline neutral during ambient silence
                self.smoothed_score = max(0.05, self.smoothed_score * 0.95)
            else:
                self.smoothed_score = 0.05

            return (
                round(self.smoothed_score, 4),
                0.0,
                "SILENCE",
                "NO_SPEECH"
            )

        # Reset silence tracker on active speech
        self.silence_sec = 0.0

        # 2. Accumulate speech time
        self.accumulated_speech_sec += hop_duration_sec
        self.history.append(raw_risk)

        # 3. Audio Duration Gating
        # Before enough speech context has accumulated, report ANALYZING
        if self.accumulated_speech_sec < self.min_speech_duration:
            if self.smoothed_score is None:
                self.smoothed_score = max(0.05, min(0.35, raw_risk))
            return (
                round(self.smoothed_score, 4),
                round(min(0.40, self.accumulated_speech_sec / self.min_speech_duration * 0.40), 4),
                "ANALYZING",
                "ANALYZING"
            )

        # 4. Multi-Window Confirmation & Rate Limiting
        hist_list = list(self.history)
        hist_mean = float(np.mean(hist_list))
        hist_std = float(np.std(hist_list)) if len(hist_list) > 1 else 0.0

        # Check for confirmed persistent spoof: >= 3 windows all high risk
        persistent_spoof = len(hist_list) >= 3 and all(s >= self.high_threshold for s in hist_list[-3:])

        if self.smoothed_score is None:
            # First valid window after gating: initialize with bounded initial value
            self.smoothed_score = min(0.40, raw_risk)
        else:
            if persistent_spoof:
                # Persistent spoof attack across consecutive windows: accelerate to high risk
                eff_alpha = min(0.60, self.alpha * 2.0)
                step_limit = self.max_step_delta * 2.5
                delta = float(np.clip((raw_risk - self.smoothed_score) * eff_alpha, -step_limit, step_limit))
            else:
                # Controlled rise/fall
                delta = float(np.clip((raw_risk - self.smoothed_score) * self.alpha, -self.max_step_delta, self.max_step_delta))

            self.smoothed_score += delta

        self.smoothed_score = float(np.clip(self.smoothed_score, 0.02, 0.98))

        # 5. Confidence Calculation
        # Combines duration factor, active model coverage, VAD probability, and temporal consistency
        duration_factor = min(1.0, self.accumulated_speech_sec / 3.0)
        coverage_factor = min(1.0, active_models_count / total_models)
        consistency_factor = max(0.0, 1.0 - min(1.0, hist_std / 0.25))

        raw_confidence = (
            0.30 * duration_factor +
            0.30 * coverage_factor +
            0.20 * consistency_factor +
            0.20 * vad_prob
        )
        confidence = float(np.clip(raw_confidence, 0.15, 0.99))

        # 6. Classification with Confidence Gating
        if self.smoothed_score < self.low_threshold:
            classification = "LOW_RISK"
        elif self.smoothed_score < self.high_threshold:
            classification = "MEDIUM_RISK"
        else:
            # Require at least 45% confidence before alarming with HIGH_RISK
            if confidence >= 0.45:
                classification = "HIGH_RISK"
            else:
                classification = "MEDIUM_RISK"

        return (
            round(self.smoothed_score, 4),
            round(confidence, 4),
            classification,
            "ANALYZED"
        )
