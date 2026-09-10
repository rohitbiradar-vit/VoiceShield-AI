"""
VoiceShield — AI Detection Pipeline Configuration
Centralized configuration for weights, thresholds, temporal smoothing, and windowing.
Smart India Hackathon 2026 | PS ID: 26104
"""

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PipelineConfig:
    # 1. Base Model Weights (Must sum to 1.0)
    # Initial prototype weights as defined in Hackathon architecture
    weights: Dict[str, float] = field(default_factory=lambda: {
        "rawnet3": 0.35,
        "aasist": 0.35,
        "ecapa": 0.20,
        "bispectral": 0.10
    })

    # 2. Risk Classification Thresholds
    low_threshold: float = 0.40      # Below 40%: LOW_RISK (Authentic)
    high_threshold: float = 0.70     # Above 70%: HIGH_RISK (Cloned / Spoof Alert)

    # 3. Sliding Audio Buffer & Update Frequency
    # Target: ~200ms update latency with sufficient context
    source_sample_rate: int = 16000
    channels: int = 1
    window_sec: float = 1.5          # Model evaluation window (1.5 seconds)
    hop_sec: float = 0.20            # 200 ms update interval
    min_speech_duration: float = 1.0 # Minimum speech required before hard decisions

    # 4. Silero VAD Settings
    vad_threshold: float = 0.45      # Speech presence threshold
    min_speech_ratio: float = 0.12   # Minimum active speech ratio in window

    # 5. Temporal Smoothing & Rate Limiting
    # Prevents isolated noisy windows from jumping to 90%
    ema_alpha: float = 0.30          # Smoothing factor for Exponential Moving Average
    max_step_delta: float = 0.12     # Maximum score change allowed per 200ms step
    history_size: int = 5            # Number of consecutive windows for consistency check

    # 6. Device & Hardware Execution
    # Detects CUDA if available, otherwise configures CPU multi-threading
    num_threads: int = 4
    device: str = "cpu"

    def __post_init__(self):
        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 1e-4:
            self.weights = {k: v / total_w for k, v in self.weights.items()}


# Global configuration singleton
DEFAULT_CONFIG = PipelineConfig()
