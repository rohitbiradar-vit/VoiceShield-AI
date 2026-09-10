"""
VoiceShield — Audio Preprocessor
Converts raw PCM byte buffers into normalized, resampled float32 waveforms.
"""

import numpy as np
import scipy.signal
import logging

logger = logging.getLogger("VoiceShieldPreprocessor")


class AudioPreprocessor:
    """
    Centralized audio preprocessor for converting raw WebRTC PCM frames
    into clean, normalized 16kHz mono waveforms for downstream AI inference.
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def pcm16_to_float32(self, pcm_bytes: bytes, channels: int = 1) -> np.ndarray:
        """
        Convert raw 16-bit signed little-endian PCM bytes to float32 in [-1.0, 1.0].
        If stereo (channels=2), downmix to mono.
        """
        if not pcm_bytes:
            return np.zeros(0, dtype=np.float32)

        # Truncate if odd number of bytes
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]

        int_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        # Normalize to [-1.0, 1.0]
        float_samples = int_samples.astype(np.float32) / 32768.0

        if channels == 2 and len(float_samples) >= 2:
            # Downmix stereo interleaved [L, R, L, R] to mono
            valid_len = len(float_samples) - (len(float_samples) % 2)
            float_samples = float_samples[:valid_len].reshape(-1, 2).mean(axis=1)

        return float_samples

    def resample(
        self,
        waveform: np.ndarray,
        source_sr: int,
        target_sr: int = 16000
    ) -> np.ndarray:
        """
        Resample waveform from source_sr to target_sr using polyphase filtering.
        """
        if source_sr == target_sr or len(waveform) == 0:
            return waveform

        gcd = np.gcd(source_sr, target_sr)
        up = target_sr // gcd
        down = source_sr // gcd

        resampled = scipy.signal.resample_poly(waveform, up, down).astype(np.float32)
        return resampled

    def remove_dc_offset(self, waveform: np.ndarray) -> np.ndarray:
        """
        Subtract the mean (DC component) from the signal.
        """
        if len(waveform) == 0:
            return waveform
        return waveform - np.mean(waveform)

    def normalize_peak(self, waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """
        Peak normalize audio to target_peak, avoiding division by zero on silent audio.
        """
        if len(waveform) == 0:
            return waveform

        max_val = np.max(np.abs(waveform))
        if max_val > 1e-4:
            scale = target_peak / max_val
            # Apply scale if not already close
            if scale < 1.0 or scale > 1.2:
                waveform = waveform * scale

        return np.clip(waveform, -1.0, 1.0)

    def process(
        self,
        pcm_bytes: bytes,
        source_sr: int = 16000,
        channels: int = 1,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Full end-to-end preprocessing pipeline:
        PCM16 bytes -> float32 mono -> Resample to 16kHz -> Remove DC -> Normalize.
        """
        waveform = self.pcm16_to_float32(pcm_bytes, channels=channels)

        if source_sr != self.target_sample_rate:
            waveform = self.resample(waveform, source_sr, self.target_sample_rate)

        waveform = self.remove_dc_offset(waveform)

        if normalize:
            waveform = self.normalize_peak(waveform)

        return waveform
