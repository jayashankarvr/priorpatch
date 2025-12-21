"""
Looks at high-frequency noise. Real photos have sensor noise;
heavily edited areas often don't.
"""

import logging
import numpy as np
from scipy.ndimage import gaussian_filter

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
from priorpatch.utils import rgb_to_luminance

logger = logging.getLogger(__name__)


@register_detector
class ResidualEnergyDetector(DetectorInterface):
    """
    Subtracts a blurred version from the image to get high-freq residual.
    Low residual energy = probably smoothed/edited = suspicious.

    Score interpretation:
    - 0.0: High residual energy (natural noise present) = likely authentic
    - 1.0: Low residual energy (too smooth) = suspicious
    """

    name = 'residual_energy'

    def score(self, patch: np.ndarray) -> float:
        """Returns normalized score based on residual energy. Low noise = high score."""
        H, W = patch.shape[:2]
        if H < 8 or W < 8:
            return 0.0

        try:
            # Convert to luminance
            L = rgb_to_luminance(patch)

            # Compute smoothed version (low-frequency component)
            sm = gaussian_filter(L, sigma=1.0)

            # Compute residual (high-frequency component)
            residual = L - sm
            energy = float((residual ** 2).mean())

            # Convert energy to [0, 1] score using sigmoid-like transform
            # Low energy -> high score (suspicious)
            # High energy -> low score (natural)
            # Typical natural images have energy in range [10, 1000]
            # Use exponential decay: score = exp(-energy / scale)
            scale = 100.0  # Tuned for typical image energy ranges
            score = float(np.exp(-energy / scale))

            return np.clip(score, 0.0, 1.0)

        except (ValueError, TypeError) as e:
            logger.warning(f"ResidualEnergyDetector failed: {e}")
            return 0.0
