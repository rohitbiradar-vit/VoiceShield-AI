"""
VoiceShield — Silero VAD (Voice Activity Detector)
Gates expensive deep models when speech is absent.
"""

import os
import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, Optional

logger = logging.getLogger("VoiceShieldVAD")


class SileroVAD:
    """
    Speech Activity Detector using official Silero VAD (ONNX runtime).
    Operates on 16kHz float32 audio chunks of 512 samples (32ms).
    """

    def __init__(
        self,
        model_path: str = "backend/models/silero_vad.onnx",
        threshold: float = 0.45,
        min_speech_ratio: float = 0.12
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.min_speech_ratio = min_speech_ratio
        self.session: Optional[ort.InferenceSession] = None
        self.is_available = False

        self._init_model()

    def _init_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Silero VAD model file not found at {self.model_path}")
            self.is_available = False
            return

        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self.is_available = True
            logger.info("Silero VAD initialized successfully via ONNX Runtime")
        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD ONNX session: {e}", exc_info=True)
            self.is_available = False

    def detect_speech(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Analyze a full 16kHz audio window (e.g. 2-3 seconds) for speech presence.
        Uses 256-sample chunks (16ms @ 16kHz) matched to Silero VAD ONNX receptive field.
        """
        if not self.is_available or self.session is None:
            return {
                "is_speech": True,  # Fallback to allow inference if VAD unavailable
                "probability": 1.0,
                "speech_ratio": 1.0,
                "available": False,
                "error": "VAD model not loaded"
            }

        chunk_size = 256
        if len(waveform) < chunk_size:
            return {
                "is_speech": False,
                "probability": 0.0,
                "speech_ratio": 0.0,
                "available": True
            }

        # Initialize persistent hidden state: shape [2, 1, 128]
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(sample_rate, dtype=np.int64)

        num_chunks = len(waveform) // chunk_size
        probabilities = []

        for i in range(num_chunks):
            chunk = waveform[i * chunk_size : (i + 1) * chunk_size].astype(np.float32)
            chunk = np.expand_dims(chunk, axis=0)  # Shape [1, 256]

            try:
                inputs = {
                    "input": chunk,
                    "state": state,
                    "sr": sr
                }
                out, state = self.session.run(None, inputs)
                prob = float(out[0][0])
                probabilities.append(prob)
            except Exception as e:
                logger.warning(f"Silero VAD chunk inference error: {e}")
                probabilities.append(0.0)

        if not probabilities:
            return {
                "is_speech": False,
                "probability": 0.0,
                "speech_ratio": 0.0,
                "available": True
            }

        probs_arr = np.array(probabilities)
        speech_chunks = np.sum(probs_arr >= self.threshold)
        speech_ratio = float(speech_chunks / len(probabilities))
        mean_prob = float(np.mean(probs_arr))
        max_prob = float(np.max(probs_arr))

        # Overall speech is detected if enough chunks exceed threshold or high-confidence speech detected
        is_speech = (speech_ratio >= self.min_speech_ratio) or (max_prob >= 0.60 and speech_ratio >= 0.06)

        # Effective speech probability: average of the top 35% active speech chunks
        top_k = max(1, int(len(probabilities) * 0.35))
        top_probs = np.sort(probs_arr)[-top_k:]
        effective_prob = float(np.mean(top_probs))

        return {
            "is_speech": is_speech,
            "probability": round(effective_prob, 4),
            "speech_ratio": round(speech_ratio, 4),
            "mean_probability": round(mean_prob, 4),
            "max_probability": round(max_prob, 4),
            "available": True
        }
