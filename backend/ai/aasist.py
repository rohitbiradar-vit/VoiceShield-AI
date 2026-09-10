"""
VoiceShield — AASIST Anti-Spoofing & Deepfake Detection Branch
Implements Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks (Interspeech 2021).
Checkpoint: backend/models/AASIST.pth (NAVER Corp / Clova AI official pretrained weights)
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional

from .aasist_model.AASIST import Model

logger = logging.getLogger("VoiceShieldAASIST")


class AASISTSpoofDetector:
    """
    AASIST Deepfake & Voice Cloning Detector.
    Processes 16kHz raw audio waveform through SincNet + ResNet + Dual GAT.
    Outputs calibrated spoof risk in [0.0, 1.0].
    """

    def __init__(
        self,
        model_path: str = "backend/models/AASIST.pth",
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model: Optional[Model] = None
        self.is_available = False

        self._init_model()

    def _init_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"AASIST checkpoint not found at {self.model_path}")
            self.is_available = False
            return

        try:
            logger.info("Initializing AASIST architecture...")
            model_config = {
                "architecture": "AASIST",
                "nb_samp": 64600,
                "first_conv": 128,
                "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                "gat_dims": [64, 32],
                "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                "temperatures": [2.0, 2.0, 100.0, 100.0]
            }
            self.model = Model(model_config).to(self.device)

            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.model.load_state_dict(sd)
            self.model.eval()
            self.is_available = True
            logger.info("AASIST model initialized and set to eval mode (300k parameters)")

        except Exception as e:
            logger.error(f"Failed to initialize AASIST: {e}", exc_info=True)
            self.model = None
            self.is_available = False

    def analyze(self, waveform: np.ndarray) -> Dict[str, Any]:
        """
        Analyze 16kHz audio waveform for AI voice cloning / synthetic speech artifacts.
        Calculates calibrated aasist_risk in [0.0, 1.0].
        """
        if not self.is_available or self.model is None:
            return {
                "available": False,
                "aasist_risk": None,
                "reason": "AASIST model not available"
            }

        # Minimum required length: at least 0.5s (8000 samples)
        if len(waveform) < 8000:
            return {
                "available": True,
                "aasist_risk": None,
                "reason": "Audio segment too short for AASIST"
            }

        try:
            # Tile or pad to at least 16000 samples (1.0s) if needed
            audio = waveform
            if len(audio) < 16000:
                repeats = int(np.ceil(16000 / len(audio)))
                audio = np.tile(audio, repeats)[:16000]

            tensor_x = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

            with torch.inference_mode():
                _, logits = self.model(tensor_x)
                # Output logits shape: [1, 2] -> [bona_fide_logit, spoof_logit]
                bona_logit = float(logits[0, 0].item())
                spoof_logit = float(logits[0, 1].item())

                # Log-likelihood ratio for spoof detection: spoof_logit - bona_logit
                # ASVspoof calibration: bona speech produces negative diff (-15 to -25)
                # Cloned/synthetic speech produces positive diff (+2 to +15)
                diff = spoof_logit - bona_logit

                # Calibrated sigmoid with temperature scaling T=4.0
                calibrated_risk = 1.0 / (1.0 + np.exp(-diff / 4.0))
                calibrated_risk = float(np.clip(calibrated_risk, 0.02, 0.98))

                return {
                    "available": True,
                    "aasist_risk": round(calibrated_risk, 4),
                    "bona_logit": round(bona_logit, 2),
                    "spoof_logit": round(spoof_logit, 2),
                    "model": "AASIST-ASVspoof"
                }

        except Exception as e:
            logger.error(f"Error in AASIST analysis: {e}", exc_info=True)
            return {
                "available": False,
                "aasist_risk": None,
                "error": str(e)
            }
