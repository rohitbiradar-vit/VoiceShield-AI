"""
VoiceShield — ECAPA-TDNN Speaker Verification Branch
Extracts 192-dimensional speaker embeddings and calculates cosine similarity against enrolled reference.
"""

import os
import logging
import numpy as np
import scipy.signal
import onnxruntime as ort
from typing import Dict, Any, Optional

logger = logging.getLogger("VoiceShieldECAPA")


class LogMelFilterbankExtractor:
    """
    Computes 80-dimensional log Mel-filterbank features from 16kHz audio.
    Matches standard Kaldi / WeSpeaker acoustic front-end.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,  # 25ms
        hop_length: int = 160,  # 10ms
        f_min: float = 20.0,
        f_max: float = 7600.0
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = np.hamming(win_length).astype(np.float32)

        # Precompute Mel filterbank matrix
        self.mel_filters = self._build_mel_filters(f_min, f_max)

    def _hz_to_mel(self, hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filters(self, f_min: float, f_max: float) -> np.ndarray:
        min_mel = self._hz_to_mel(f_min)
        max_mel = self._hz_to_mel(f_max)
        mel_points = np.linspace(min_mel, max_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        filters = np.zeros((self.n_mels, self.n_fft // 2 + 1), dtype=np.float32)

        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return filters

    def compute_fbank(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract [T, 80] log Mel-filterbank features from 16kHz waveform.
        """
        if len(waveform) < self.win_length:
            waveform = np.pad(waveform, (0, self.win_length - len(waveform)), mode="reflect")

        # Frame audio
        num_frames = 1 + (len(waveform) - self.win_length) // self.hop_length
        frames = np.zeros((num_frames, self.win_length), dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_length
            frames[i] = waveform[start : start + self.win_length] * self.window

        # Real FFT: power spectrum
        spectra = np.abs(np.fft.rfft(frames, n=self.n_fft)) ** 2

        # Apply Mel filters: shape [num_frames, n_mels]
        mel_energies = np.dot(spectra, self.mel_filters.T)
        mel_energies = np.maximum(mel_energies, 1e-5)

        # Log mel
        log_mel = np.log(mel_energies)

        # Cepstral mean subtraction across time
        mean = np.mean(log_mel, axis=0, keepdims=True)
        log_mel = log_mel - mean

        return log_mel.astype(np.float32)


class ECAPATDNNVerifier:
    """
    Branch 2: ECAPA-TDNN Speaker Verification & Impersonation Detector.
    Source: WeSpeaker / SpeechBrain (Interspeech 2020)
    Checkpoint: backend/models/ecapa_tdnn.onnx (24MB, 192-dim embedding)
    Sample Rate: 16000 Hz Mono
    Input: [1, T, 80] log Mel-filterbank tensor
    Output: 192-dim L2-normalized speaker embedding vector
    """

    def __init__(
        self,
        model_path: str = "backend/models/ecapa_tdnn.onnx"
    ):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.fbank_extractor = LogMelFilterbankExtractor()
        self.is_available = False

        # In-memory enrolled speaker database: speaker_id -> np.ndarray [192]
        self.enrolled_profiles: Dict[str, np.ndarray] = {}

        self._init_model()

    def _init_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"ECAPA-TDNN ONNX model not found at {self.model_path}")
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
            logger.info("ECAPA-TDNN initialized successfully via ONNX Runtime")
        except Exception as e:
            logger.error(f"Failed to initialize ECAPA-TDNN ONNX session: {e}", exc_info=True)
            self.is_available = False

    def extract_embedding(self, waveform: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 192-dimensional speaker embedding from 16kHz audio.
        """
        if not self.is_available or self.session is None:
            return None

        try:
            # 1. Compute [T, 80] fbank
            fbank = self.fbank_extractor.compute_fbank(waveform)
            # Add batch dimension: [1, T, 80]
            fbank_batch = np.expand_dims(fbank, axis=0)

            # 2. Forward pass through ONNX model
            embs = self.session.run(None, {"feats": fbank_batch})[0]  # Shape [1, 192]
            emb = embs[0].astype(np.float32)

            # 3. L2 normalize embedding
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm

            return emb

        except Exception as e:
            logger.error(f"ECAPA-TDNN embedding extraction error: {e}", exc_info=True)
            return None

    def enroll_speaker(self, speaker_id: str, audio_waveform: np.ndarray) -> bool:
        """
        Enroll a legitimate reference speaker from audio.
        """
        emb = self.extract_embedding(audio_waveform)
        if emb is not None:
            self.enrolled_profiles[speaker_id] = emb
            logger.info(f"Speaker '{speaker_id}' enrolled successfully (192-dim embedding)")
            return True
        return False

    def verify_speaker(
        self,
        live_waveform: np.ndarray,
        speaker_id: Optional[str] = "Friend"
    ) -> Dict[str, Any]:
        """
        Compare live audio embedding against enrolled reference speaker.
        Strict Rule: If not enrolled, reports NOT_ENROLLED rather than inventing a score.
        """
        if not self.is_available:
            return {
                "available": False,
                "status": "MODEL_UNAVAILABLE",
                "speaker_similarity": None,
                "speaker_risk": None
            }

        # Check if reference speaker is enrolled
        if not speaker_id or speaker_id not in self.enrolled_profiles:
            return {
                "available": True,
                "status": "NOT_ENROLLED",
                "speaker_id": speaker_id,
                "speaker_similarity": None,
                "speaker_risk": None,
                "message": "Speaker verification: NOT AVAILABLE (No reference enrolled)"
            }

        live_emb = self.extract_embedding(live_waveform)
        if live_emb is None:
            return {
                "available": True,
                "status": "EXTRACTION_FAILED",
                "speaker_similarity": None,
                "speaker_risk": None
            }

        ref_emb = self.enrolled_profiles[speaker_id]

        # Cosine similarity between L2-normalized embeddings
        similarity = float(np.dot(ref_emb, live_emb))
        similarity = float(np.clip(similarity, -1.0, 1.0))

        # Convert similarity to mismatch risk contribution
        # High similarity (> 0.70) = low risk
        # Low similarity (< 0.40) = high risk (voice does not match trusted profile)
        if similarity >= 0.70:
            risk = max(0.05, (1.0 - similarity) / 0.60)
        elif similarity <= 0.30:
            risk = min(0.95, 0.70 + (0.30 - similarity) * 0.80)
        else:
            risk = 0.30 + (0.70 - similarity) * 1.0

        risk = float(np.clip(risk, 0.05, 0.95))

        return {
            "available": True,
            "status": "VERIFIED",
            "speaker_id": speaker_id,
            "speaker_similarity": round(similarity, 4),
            "speaker_risk": round(risk, 4),
            "embedding_dim": len(live_emb)
        }
