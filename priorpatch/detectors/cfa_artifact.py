"""
CFA (Color Filter Array) artifact detector.

Real cameras have Bayer sensors - each pixel only captures one color (R, G, or B).
The camera interpolates the missing colors (demosaicing), which creates specific
patterns between neighboring pixels.

AI images are made directly in RGB, so they don't have these patterns. This is
a physical difference that's really hard to fake.

GPU acceleration via CuPy when available.

Ref: "Digital image forensic approach based on the second-order statistical
     analysis of CFA artifacts" (2019)
"""

import logging
import numpy as np
from scipy import signal

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
from priorpatch.gpu_backend import fft2_shifted

logger = logging.getLogger(__name__)


def compute_interpolation_residual(channel: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """Compute residual between original and re-interpolated channel.

    For a channel that was demosaiced from a Bayer CFA, there should be
    a specific pattern of interpolated vs. original pixels.

    Args:
        channel: Single color channel (H, W)
        pattern: CFA pattern (RGGB, BGGR, GRBG, GBRG)

    Returns:
        Interpolation residual image
    """
    h, w = channel.shape

    # Simple bilinear interpolation kernel
    kernel = np.array([
        [0.25, 0.5, 0.25],
        [0.5,  0.0, 0.5],
        [0.25, 0.5, 0.25]
    ])

    # Compute what the value would be if interpolated from neighbors
    # Note: 'symm' is scipy's equivalent of 'reflect' boundary mode
    interpolated = signal.convolve2d(channel, kernel, mode='same', boundary='symm')

    # Residual shows where actual values differ from interpolation
    residual = channel - interpolated

    return residual


def detect_cfa_periodic_pattern(channel: np.ndarray) -> float:
    """Detect periodic patterns characteristic of CFA interpolation.

    Real demosaiced images have peaks at specific frequencies in the
    Fourier spectrum corresponding to the CFA pattern periodicity.

    Args:
        channel: Single color channel (H, W)

    Returns:
        CFA artifact strength (higher = more CFA artifacts present)
    """
    h, w = channel.shape

    # Compute interpolation residual
    residual = compute_interpolation_residual(channel)

    # Analyze frequency content of residual (GPU-accelerated if available)
    f_residual = np.abs(fft2_shifted(residual))

    # Normalize by DC component
    dc_component = f_residual[h//2, w//2]
    if dc_component > 0:
        f_residual = f_residual / dc_component

    # CFA patterns create peaks at Nyquist frequencies (edges of spectrum)
    # For RGGB Bayer, expect peaks at (0.5, 0), (0, 0.5), (0.5, 0.5) normalized

    # Sample at expected CFA peak locations
    # These correspond to the 2x2 periodicity of Bayer pattern
    peak_locations = [
        (h//4, w//2),      # Vertical Nyquist/2
        (h//2, w//4),      # Horizontal Nyquist/2
        (h//4, w//4),      # Diagonal
        (3*h//4, w//2),    # Vertical Nyquist/2 (symmetric)
        (h//2, 3*w//4),    # Horizontal Nyquist/2 (symmetric)
        (3*h//4, 3*w//4),  # Diagonal (symmetric)
    ]

    # Sum energy at CFA-related frequencies
    cfa_energy = 0.0
    for y, x in peak_locations:
        if 0 <= y < h and 0 <= x < w:
            # Sum 3x3 neighborhood around peak
            y_start, y_end = max(0, y-1), min(h, y+2)
            x_start, x_end = max(0, x-1), min(w, x+2)
            cfa_energy += np.sum(f_residual[y_start:y_end, x_start:x_end])

    # Compare to background energy
    # Exclude center (DC) and peaks
    background = np.median(f_residual)

    if background > 0:
        cfa_strength = cfa_energy / (background * len(peak_locations) * 9 + 1e-10)
    else:
        cfa_strength = 0.0

    return cfa_strength


def compute_variance_ratio(channel: np.ndarray) -> float:
    """Compute variance ratio between interpolated and sampled pixels.

    In a demosaiced image, pixels that were originally sampled have
    different variance characteristics than pixels that were interpolated.

    Based on Dirik & Memon's forensic feature.

    Args:
        channel: Single color channel

    Returns:
        Variance ratio (values close to 1 indicate CFA artifacts)
    """
    h, w = channel.shape

    # Create masks for "sampled" and "interpolated" positions
    # For Bayer RGGB: Red is at even rows, even cols
    # Green is at checkerboard, Blue is at odd rows, odd cols

    sampled_mask = np.zeros((h, w), dtype=bool)
    interpolated_mask = np.zeros((h, w), dtype=bool)

    # Assume even positions were sampled (this is a simplification)
    sampled_mask[0::2, 0::2] = True
    sampled_mask[1::2, 1::2] = True

    # Odd positions were interpolated
    interpolated_mask[0::2, 1::2] = True
    interpolated_mask[1::2, 0::2] = True

    # Compute local variance using a small window
    kernel = np.ones((3, 3)) / 9.0
    local_mean = signal.convolve2d(channel, kernel, mode='same', boundary='symm')
    local_var = signal.convolve2d((channel - local_mean)**2, kernel, mode='same', boundary='symm')

    # Get variance at sampled vs interpolated positions
    var_sampled = np.mean(local_var[sampled_mask])
    var_interpolated = np.mean(local_var[interpolated_mask])

    if var_interpolated > 0:
        ratio = var_sampled / var_interpolated
    else:
        ratio = 1.0

    return ratio


@register_detector
class CFAArtifactDetector(DetectorInterface):
    """Detect absence of CFA (Color Filter Array) demosaicing artifacts.

    Real camera images pass through Bayer demosaicing, which creates
    characteristic periodic correlations. AI-generated images are
    synthesized directly in RGB and lack these artifacts.

    This exploits a fundamental physical difference between cameras and AI.

    IMPORTANT: Score interpretation for this detector:
    - Score = 0.0 (LOW): CFA artifacts present → REAL camera image
    - Score = 1.0 (HIGH): CFA artifacts absent → SUSPICIOUS (likely AI-generated)

    This is opposite from what you might expect, but makes sense: we're
    detecting the ABSENCE of camera artifacts, not their presence.
    """

    name = 'cfa_artifact'

    def __init__(self, check_all_channels: bool = True):
        """Initialize detector.

        Args:
            check_all_channels: If True, analyze R, G, B separately
        """
        self.check_all_channels = check_all_channels

    def score(self, patch: np.ndarray) -> float:
        """Score patch based on CFA artifact presence.

        Args:
            patch: RGB image patch (H, W, 3)

        Returns:
            Anomaly score (0 = has CFA artifacts, 1 = no CFA artifacts = suspicious)
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        h, w = patch.shape[:2]

        # Need sufficient size for frequency analysis
        if h < 32 or w < 32:
            return 0.0

        try:
            # Convert to float64 for analysis
            img = patch.astype(np.float64)

            scores = []

            for c in range(3):
                channel = img[:, :, c]

                # Method 1: Periodic pattern detection
                cfa_strength = detect_cfa_periodic_pattern(channel)

                # Method 2: Variance ratio analysis
                var_ratio = compute_variance_ratio(channel)

                # CFA strength: higher = more CFA artifacts (natural)
                # We want: no artifacts = suspicious, so invert
                # Typical natural images have cfa_strength > 2.0
                cfa_score = max(0, 1.0 - cfa_strength / 5.0)

                # Variance ratio: close to 1.0 = no CFA pattern
                # Natural images typically have ratio ~0.85-0.95 for interpolated channels
                # Ratio close to 1.0 = suspicious
                var_score = 1.0 - abs(var_ratio - 1.0) * 2.0
                var_score = max(0, min(1, var_score))

                # Combine scores
                channel_score = 0.5 * cfa_score + 0.5 * var_score
                scores.append(channel_score)

            # Average across channels, weight green less (has more pixels in Bayer)
            # R, G, B weights
            weights = [0.35, 0.30, 0.35]
            final_score = sum(s * w for s, w in zip(scores, weights))

            return float(np.clip(final_score, 0.0, 1.0))

        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"CFAArtifactDetector failed: {e}")
            return 0.0
