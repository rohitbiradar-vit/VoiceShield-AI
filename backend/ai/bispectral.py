"""
VoiceShield — Bispectral Analysis Branch (Higher-Order Spectra)
Computes bifrequency bicoherence and non-linear phase coupling anomalies
to detect synthetic speech / vocoder synthesis artifacts.
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger("VoiceShieldBispectral")


class BispectralAnalyzer:
    """
    Branch 3: Bispectral / Higher-Order Spectral Analysis.
    Evaluates cross-frequency quadratic phase coupling (QPC) in the bifrequency plane:
      B(f1, f2) = E[ X(f1) * X(f2) * X*(f1 + f2) ]
      b^2(f1, f2) = |B(f1, f2)|^2 / ( E[|X(f1)X(f2)|^2] * E[|X(f1+f2)|^2] )
    
    Configuration:
      - Window: 512 samples (32ms at 16kHz) Hanning window
      - Hop: 256 samples (16ms, 50% overlap)
      - FFT size: 256 bins (128 non-negative frequency points up to 8kHz)
      - Bifrequency grid: 64 x 64 principal domain (0 <= f2 <= f1, f1 + f2 <= Fs/2)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 256,
        win_size: int = 512,
        hop_size: int = 256
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_size
        self.window = np.hanning(win_size).astype(np.float32)

        # Number of unique FFT bins up to Nyquist
        self.n_bins = n_fft // 2
        # Principal non-redundant domain max index
        self.max_idx = self.n_bins // 2

    def compute_bispectrum_and_bicoherence(self, waveform: np.ndarray) -> Dict[str, Any]:
        """
        Compute average cross-frequency bispectrum and bicoherence across all frames.
        Fully vectorized in NumPy without nested Python loops.
        """
        if len(waveform) < self.win_size:
            waveform = np.pad(waveform, (0, self.win_size - len(waveform)), mode="reflect")

        num_frames = 1 + (len(waveform) - self.win_size) // self.hop_size
        if num_frames < 2:
            return {"valid": False, "reason": "Insufficient frames"}

        # Vectorized frame extraction: shape [num_frames, win_size]
        frames = np.lib.stride_tricks.sliding_window_view(waveform, self.win_size)[::self.hop_size] * self.window
        X = np.fft.rfft(frames, n=self.n_fft)  # Shape [num_frames, n_fft//2 + 1]

        K = self.max_idx
        f1_grid, f2_grid = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
        f_sum_grid = f1_grid + f2_grid
        valid_mask = (f2_grid <= f1_grid) & (f_sum_grid < self.n_bins) & (f1_grid > 0) & (f2_grid > 0)

        f1_valid = f1_grid[valid_mask]
        f2_valid = f2_grid[valid_mask]
        f_sum_valid = f_sum_grid[valid_mask]

        # Vectorized triple cross product across all frames
        x1 = X[:, f1_valid]
        x2 = X[:, f2_valid]
        x3 = np.conj(X[:, f_sum_valid])

        triples = x1 * x2 * x3
        B_mean_valid = np.mean(triples, axis=0)
        p12_mean_valid = np.mean(np.abs(x1 * x2) ** 2, axis=0)
        p3_mean_valid = np.mean(np.abs(x3) ** 2, axis=0)

        denom_valid = np.maximum(p12_mean_valid * p3_mean_valid, 1e-12)
        bic_valid = np.clip((np.abs(B_mean_valid) ** 2) / denom_valid, 0.0, 1.0)
        phase_valid = np.angle(B_mean_valid)

        # Populate 2D matrices
        bicoherence = np.zeros((K, K), dtype=np.float64)
        bicoherence[f1_valid, f2_valid] = bic_valid

        bispectrum_phase = np.zeros((K, K), dtype=np.float64)
        bispectrum_phase[f1_valid, f2_valid] = phase_valid

        return {
            "valid": True,
            "bicoherence": bicoherence,
            "bispectrum_mag": np.abs(B_mean_valid),
            "bispectrum_phase": bispectrum_phase,
            "num_frames": num_frames
        }

    def analyze(self, waveform: np.ndarray) -> Dict[str, Any]:
        """
        Perform bispectral feature extraction and return calibrated anomaly score.
        Outputs bispectral_risk in [0.0, 1.0].
        """
        if len(waveform) < self.win_size:
            return {
                "available": True,
                "bispectral_risk": None,
                "reason": "Audio segment too short for bispectral analysis"
            }

        try:
            res = self.compute_bispectrum_and_bicoherence(waveform)
            if not res["valid"]:
                return {
                    "available": True,
                    "bispectral_risk": None,
                    "reason": res.get("reason", "Computation failed")
                }

            bic = res["bicoherence"]
            phase = res["bispectrum_phase"]
            K = self.max_idx

            # 1. Harmonic Region Bicoherence (vocal tract non-linearity)
            bin_min = max(1, int(300.0 / (self.sample_rate / self.n_fft)))
            bin_max = min(K, int(3400.0 / (self.sample_rate / self.n_fft)))

            harmonic_mask = np.zeros_like(bic, dtype=bool)
            harmonic_mask[bin_min:bin_max, bin_min:bin_max] = True
            harmonic_bic_vals = bic[harmonic_mask]
            mean_harmonic_bic = float(np.mean(harmonic_bic_vals)) if len(harmonic_bic_vals) > 0 else 0.0

            # 2. Diagonal Slice Bicoherence: b^2(f, f)
            diag_vals = np.array([bic[f, f] for f in range(1, K)])
            diag_mean = float(np.mean(diag_vals))
            diag_var = float(np.var(diag_vals))

            # 3. Phase Entropy (phase coupling randomness)
            phase_vals = phase[bic > 0.05]
            if len(phase_vals) > 10:
                hist, _ = np.histogram(phase_vals, bins=16, range=(-np.pi, np.pi), density=True)
                hist = hist[hist > 0]
                phase_entropy = float(-np.sum(hist * np.log(hist + 1e-12)))
            else:
                phase_entropy = 2.40  # Natural glottal prior

            # 4. Calibrated Anomaly Detection
            # Normal speech exhibits:
            # - moderate harmonic bicoherence (~ 0.01 - 0.40)
            # - moderate diagonal variance (< 0.15)
            # - phase entropy near human vocal fold coupling (~ 2.0 - 2.6)
            anomaly = 0.0

            # Artificial excessive coupling penalty (robotic repeating waveforms)
            if mean_harmonic_bic > 0.60:
                anomaly += (mean_harmonic_bic - 0.60) * 2.5

            # Unnatural diagonal slice variance penalty (vocoder tiling)
            if diag_var > 0.10:
                anomaly += (diag_var - 0.10) * 3.0

            # Phase entropy deviation penalty
            if phase_entropy > 2.80:
                anomaly += (phase_entropy - 2.80) * 1.0
            elif phase_entropy < 1.60:
                anomaly += (1.60 - phase_entropy) * 1.0

            # Direct calibrated mapping: anomaly=0.0 -> risk=0.05 (5%)
            calibrated_risk = float(np.clip(0.05 + anomaly * 0.70, 0.02, 0.95))

            return {
                "available": True,
                "bispectral_risk": round(calibrated_risk, 4),
                "mean_harmonic_bicoherence": round(mean_harmonic_bic, 4),
                "diagonal_slice_variance": round(diag_var, 4),
                "phase_entropy": round(phase_entropy, 4),
                "frames_analyzed": res["num_frames"]
            }

        except Exception as e:
            logger.error(f"Error in bispectral analysis: {e}", exc_info=True)
            return {
                "available": False,
                "bispectral_risk": None,
                "error": str(e)
            }
