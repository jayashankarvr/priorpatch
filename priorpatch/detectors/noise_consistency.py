"""
Noise consistency detector - finds regions with unnatural noise patterns.

Real cameras have consistent sensor noise. AI-generated images often have
too-smooth regions or inconsistent noise levels.

GPU-accelerated via CuPy when available.
"""

import logging
import numpy as np
from scipy.ndimage import uniform_filter

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
from priorpatch.utils import rgb_to_luminance
from priorpatch.gpu_backend import fft2_shifted

logger = logging.getLogger(__name__)


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Compute correlation coefficient with NaN handling.

    Returns 0.0 if correlation cannot be computed (constant arrays, etc.)
    """
    try:
        corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except (ValueError, FloatingPointError):
        return 0.0


@register_detector
class NoiseConsistencyDetector(DetectorInterface):
    """Detect unnatural noise patterns.

    Score interpretation:
    - 0.0: Natural noise patterns (consistent, Gaussian-like)
    - 1.0: Unnatural noise (too clean, inconsistent, or periodic)
    """

    name = 'noise_consistency'

    def score(self, patch: np.ndarray) -> float:
        """Score based on noise consistency analysis."""
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        H, W = patch.shape[:2]
        if H < 16 or W < 16:
            return 0.0

        try:
            gray = rgb_to_luminance(patch)
            scores = []

            # 1. Extract noise using high-pass filter
            smoothed = uniform_filter(gray, size=3)
            noise = gray - smoothed

            # 2. Check noise variance
            noise_var = np.var(noise)

            if noise_var < 0.5:
                scores.append(0.6)  # Suspiciously clean
            elif noise_var < 2.0:
                scores.append(0.3)  # Very clean
            elif noise_var > 50:
                scores.append(0.2)  # Very noisy

            # 3. Check noise distribution (should be roughly Gaussian)
            noise_flat = noise.flatten()
            mean_n = np.mean(noise_flat)
            std_n = np.std(noise_flat)

            if std_n > 1e-6:
                normalized = (noise_flat - mean_n) / std_n
                skewness = np.mean(normalized ** 3)
                kurtosis = np.mean(normalized ** 4)
                excess_kurtosis = kurtosis - 3.0

                skew_score = min(abs(skewness) / 2.0, 1.0) * 0.3
                kurt_score = min(abs(excess_kurtosis) / 3.0, 1.0) * 0.3
                scores.append(skew_score + kurt_score)

            # 4. Check noise consistency across sub-regions
            h2, w2 = H // 2, W // 2
            quadrants = [
                noise[:h2, :w2],
                noise[:h2, w2:],
                noise[h2:, :w2],
                noise[h2:, w2:],
            ]

            quad_vars = [np.var(q) for q in quadrants if q.size > 0]

            if len(quad_vars) >= 4:
                var_of_vars = np.var(quad_vars)
                mean_var = np.mean(quad_vars)

                if mean_var > 1e-6:
                    cv = np.sqrt(var_of_vars) / mean_var
                    if cv > 0.5:
                        scores.append(0.5)
                    elif cv > 0.3:
                        scores.append(0.3)
                    elif cv > 0.1:
                        scores.append(0.1)

            # 5. Check for periodic noise patterns (GPU-accelerated)
            noise_fft_shifted = np.abs(fft2_shifted(noise))

            center_y, center_x = H // 2, W // 2
            noise_fft_shifted[center_y-1:center_y+2, center_x-1:center_x+2] = 0

            mean_fft = np.mean(noise_fft_shifted)
            max_fft = np.max(noise_fft_shifted)

            if mean_fft > 1e-6:
                peak_ratio = max_fft / mean_fft
                if peak_ratio > 10:
                    scores.append(0.4)
                elif peak_ratio > 5:
                    scores.append(0.2)

            # 6. Check correlation between color channels' noise
            r_noise = patch[:, :, 0].astype(np.float64) - uniform_filter(patch[:, :, 0].astype(np.float64), size=3)
            g_noise = patch[:, :, 1].astype(np.float64) - uniform_filter(patch[:, :, 1].astype(np.float64), size=3)
            b_noise = patch[:, :, 2].astype(np.float64) - uniform_filter(patch[:, :, 2].astype(np.float64), size=3)

            rg_corr = _safe_corrcoef(r_noise, g_noise)
            rb_corr = _safe_corrcoef(r_noise, b_noise)
            gb_corr = _safe_corrcoef(g_noise, b_noise)

            avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3

            if avg_corr < 0.1:
                scores.append(0.4)
            elif avg_corr > 0.95:
                scores.append(0.2)

            if not scores:
                return 0.0

            return float(np.clip(np.mean(scores), 0.0, 1.0))

        except (ValueError, TypeError, FloatingPointError) as e:
            logger.warning(f"NoiseConsistencyDetector failed: {e}")
            return 0.0
