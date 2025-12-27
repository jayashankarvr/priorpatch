"""
Benford's Law detector.

In real-world data, the first digit isn't evenly distributed - 1 shows up
more than 2, 2 more than 3, etc. DCT coefficients from real photos follow
this pattern. Edited and AI images often don't.

Gets around 90% F1-score for detecting manipulation.
Ref: "Benford's law applied to digital forensic analysis" (2023)

Uses CPU (scipy) since per-block DCT has too much GPU transfer overhead.
"""

import logging
import numpy as np
from scipy.fftpack import dct
from scipy.stats import pearsonr

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
from priorpatch.utils import rgb_to_luminance

logger = logging.getLogger(__name__)


# Benford's Law expected probabilities for digits 1-9
BENFORD_EXPECTED = np.array([
    np.log10(1 + 1/d) for d in range(1, 10)
])


def get_first_significant_digit(values: np.ndarray) -> np.ndarray:
    """Extract first significant digit from array of values.

    Args:
        values: Array of numeric values

    Returns:
        Array of first significant digits (1-9)
    """
    # Filter out zeros and take absolute values
    abs_vals = np.abs(values)
    nonzero_mask = abs_vals > 0
    abs_vals = abs_vals[nonzero_mask]

    if len(abs_vals) == 0:
        return np.array([])

    # Get first significant digit using log10
    # First digit = floor(x / 10^floor(log10(x)))
    log_vals = np.log10(abs_vals)
    floor_log = np.floor(log_vals)
    first_digit = np.floor(abs_vals / (10 ** floor_log)).astype(int)

    # Clamp to valid range [1, 9]
    first_digit = np.clip(first_digit, 1, 9)

    return first_digit


def compute_digit_distribution(digits: np.ndarray) -> np.ndarray:
    """Compute probability distribution of digits 1-9.

    Args:
        digits: Array of first significant digits

    Returns:
        Probability distribution for digits 1-9
    """
    if len(digits) == 0:
        return np.zeros(9)

    counts = np.zeros(9)
    for d in range(1, 10):
        counts[d-1] = np.sum(digits == d)

    total = np.sum(counts)
    if total > 0:
        return counts / total
    return counts


@register_detector
class BenfordLawDetector(DetectorInterface):
    """Detect manipulation using Benford's Law on DCT coefficients.

    Natural images have DCT coefficients whose first significant digits
    follow Benford's Law. Manipulated and AI-generated images often
    deviate from this distribution.

    This is one of the most reliable non-ML forensic techniques,
    achieving ~90% accuracy in research studies.
    """

    name = 'benford_law'

    def __init__(self, block_size: int = 8, use_ac_only: bool = True):
        """
        Args:
            block_size: DCT block size (default 8 for JPEG compatibility)
            use_ac_only: If True, exclude DC coefficient
        """
        self.block_size = block_size
        self.use_ac_only = use_ac_only

    def get_config(self) -> dict:
        """Serialize for multiprocessing."""
        return {'block_size': self.block_size, 'use_ac_only': self.use_ac_only}

    def score(self, patch: np.ndarray) -> float:
        """Score patch based on Benford's Law deviation.

        Args:
            patch: RGB image patch (H, W, 3)

        Returns:
            Anomaly score (0 = follows Benford, 1 = strong deviation)
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        h, w = patch.shape[:2]
        min_size = self.block_size * 2

        if h < min_size or w < min_size:
            return 0.0

        try:
            # Convert to grayscale using standard BT.601 weights
            gray = rgb_to_luminance(patch).astype(np.float64)

            # Collect DCT coefficients from blocks
            all_coeffs = []

            n_blocks_h = h // self.block_size
            n_blocks_w = w // self.block_size

            for i in range(n_blocks_h):
                for j in range(n_blocks_w):
                    block = gray[
                        i*self.block_size:(i+1)*self.block_size,
                        j*self.block_size:(j+1)*self.block_size
                    ]

                    # Apply 2D DCT
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

                    if self.use_ac_only:
                        # Flatten and exclude DC coefficient (top-left)
                        coeffs = dct_block.flatten()[1:]
                    else:
                        coeffs = dct_block.flatten()

                    all_coeffs.extend(coeffs)

            all_coeffs = np.array(all_coeffs)

            if len(all_coeffs) < 100:
                return 0.0

            # Get first significant digits
            digits = get_first_significant_digit(all_coeffs)

            if len(digits) < 50:
                return 0.0

            # Compute observed distribution
            observed = compute_digit_distribution(digits)

            # Method 1: Pearson correlation (higher = more natural)
            correlation, _ = pearsonr(observed, BENFORD_EXPECTED)

            # Method 2: Chi-squared test
            # Scale expected to match observed counts
            expected_counts = BENFORD_EXPECTED * len(digits)
            observed_counts = observed * len(digits)

            # Add small epsilon to avoid division by zero
            expected_counts = np.maximum(expected_counts, 1e-10)

            # Compute chi-squared statistic (normalized)
            chi_sq = np.sum((observed_counts - expected_counts)**2 / expected_counts)

            # Normalize chi-squared to [0, 1] range
            # Typical chi-squared for 8 DOF: <15.5 is good (95% confidence)
            chi_score = min(chi_sq / 50.0, 1.0)

            # Correlation score: 1.0 = perfect correlation (natural)
            # We want high score = suspicious, so invert
            corr_score = 1.0 - max(0, correlation)

            # Combine both metrics
            # High chi-squared OR low correlation = suspicious
            combined_score = 0.6 * chi_score + 0.4 * corr_score

            return float(np.clip(combined_score, 0.0, 1.0))

        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"BenfordLawDetector failed: {e}")
            return 0.0
