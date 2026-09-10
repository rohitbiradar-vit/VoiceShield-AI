"""
VoiceShield — Singleton AI Model Manager
Pre-loads all models once at server startup and keeps them resident in memory.
Eliminates multi-second re-loading latency when calls connect.
"""

import logging
import torch
from typing import Optional

from .config import DEFAULT_CONFIG, PipelineConfig
from .vad import SileroVAD
from .aasist import AASISTSpoofDetector
from .rawnet3 import RawNet3SpoofDetector
from .ecapa import ECAPATDNNVerifier
from .bispectral import BispectralAnalyzer

logger = logging.getLogger("VoiceShieldModelManager")


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        if ModelManager._instance is not None:
            raise RuntimeError("ModelManager is a singleton. Use ModelManager.get_instance().")

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            torch.set_num_threads(self.config.num_threads)
            logger.info(f"Configured PyTorch CPU threads: {self.config.num_threads}")

        logger.info(f"Loading AI Models into memory (Device: {self.device.upper()})...")

        # 1. Silero VAD (ONNX)
        self.vad = SileroVAD()

        # 2. AASIST Anti-Spoofing (PyTorch)
        self.aasist = AASISTSpoofDetector(device=self.device)

        # 3. RawNet3 Anti-Spoofing / Embeddings (PyTorch)
        self.rawnet = RawNet3SpoofDetector(device=self.device)

        # 4. ECAPA-TDNN Speaker Verification (ONNX)
        self.ecapa = ECAPATDNNVerifier()

        # 5. Bispectral Analyzer (Vectorized NumPy)
        self.bispectral = BispectralAnalyzer(sample_rate=self.config.source_sample_rate)

        logger.info(
            f"All AI models loaded successfully! "
            f"[VAD: {self.vad.is_available}, AASIST: {self.aasist.is_available}, "
            f"RawNet3: {self.rawnet.is_available}, ECAPA: {self.ecapa.is_available}, "
            f"Bispectral: True]"
        )

    @classmethod
    def get_instance(cls, config: PipelineConfig = DEFAULT_CONFIG) -> "ModelManager":
        if cls._instance is None:
            cls._instance = ModelManager(config)
        return cls._instance
