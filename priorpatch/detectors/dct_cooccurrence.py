"""
Looks at DCT coefficients in 8x8 blocks. Different compression
histories or spliced regions show up as inconsistent DC values.

Note: DCT operations use CPU (scipy). Block-based DCT has high overhead for
GPU transfer per block, so CPU is more efficient for this detector.
"""

import logging
import numpy as np
from scipy.fftpack import dct

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
from priorpatch.utils import rgb_to_luminance

logger = logging.getLogger(__name__)


@register_detector
class DCTCoocDetector(DetectorInterface):
    """
    Splits image into 8x8 blocks, computes DCT, looks at DC coefficient variance.
    Spliced images often have mismatched compression = high DC variance.

    Score interpretation:
    - 0.0: Consistent DC coefficients (uniform compression)
    - 1.0: Highly inconsistent DC coefficients (likely spliced or manipulated)
    """

    name = 'dct_cooccurrence'

    def score(self, patch: np.ndarray) -> float:
        """Variance of DC coefficients across 8x8 blocks."""
        H, W = patch.shape[:2]
        if H < 16 or W < 16:
            return 0.0

        try:
            # Convert to luminance
            L = rgb_to_luminance(patch)

            # Extract DC coefficients from 8x8 blocks
            bs = 8
            dc_vals = []

            for y in range(0, H - bs + 1, bs):
                for x in range(0, W - bs + 1, bs):
                    block = L[y:y+bs, x:x+bs]
                    # 2D DCT
                    B = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    dc_vals.append(B[0, 0])

            if len(dc_vals) < 4:
                return 0.0

            dc = np.array(dc_vals)

            # Compute normalized variance as anomaly score
            mean_abs_dc = np.mean(np.abs(dc))
            if mean_abs_dc < 1e-6:
                return 0.0

            variance_ratio = np.var(dc) / mean_abs_dc

            # Normalize to [0, 1] using sigmoid-like transform
            # Typical variance ratios range from 0 to ~10 for normal images
            score = float(1.0 - np.exp(-variance_ratio / 5.0))

            return np.clip(score, 0.0, 1.0)

        except (ValueError, TypeError) as e:
            logger.warning(f"DCTCoocDetector failed: {e}")
            return 0.0
