"""
JPEG Ghost detector - finds recompression artifacts.

When an image is saved as JPEG multiple times, or different parts
come from different JPEG sources, the compression artifacts don't
match up. This detector finds those inconsistencies.

Technique: Recompress the patch at various quality levels and find
where the difference is minimized. If a region was originally compressed
at quality Q, recompressing at Q will show minimal change.
"""

import logging
import numpy as np
from io import BytesIO
from PIL import Image

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector

logger = logging.getLogger(__name__)


@register_detector
class JPEGGhostDetector(DetectorInterface):
    """Detect JPEG recompression artifacts."""

    name = 'jpeg_ghost'

    def __init__(self, quality_levels=None):
        """
        Args:
            quality_levels: JPEG quality values to test (default: 50-95 step 5)
        """
        self.quality_levels = quality_levels or list(range(50, 96, 5))

    def score(self, patch: np.ndarray) -> float:
        """
        Score based on JPEG ghost analysis.

        Higher score = more suspicious (inconsistent compression).
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        H, W = patch.shape[:2]
        if H < 8 or W < 8:
            return 0.0

        try:
            # Convert to PIL Image
            pil_img = Image.fromarray(patch.astype(np.uint8))
            original = patch.astype(np.float32)

            differences = []

            for quality in self.quality_levels:
                # Recompress at this quality
                buffer = BytesIO()
                pil_img.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                recompressed = np.array(Image.open(buffer)).astype(np.float32)

                # Calculate difference
                diff = np.mean(np.abs(original - recompressed))
                differences.append(diff)

            differences = np.array(differences)

            if len(differences) == 0:
                return 0.0

            # The minimum difference tells us the likely original quality
            min_diff = differences.min()
            max_diff = differences.max()
            mean_diff = differences.mean()

            # Key insight: if the patch has been through JPEG before,
            # one quality level will show much lower difference than others.
            # If it's uncompressed or AI-generated, differences will be more uniform.

            # Calculate how "peaky" the difference curve is
            if max_diff - min_diff < 0.1:
                # Very flat curve - likely never JPEG compressed or synthetic
                # This is actually suspicious for photos (should have compression)
                return 0.3

            # Ratio of min to mean - lower means clearer JPEG signature
            ratio = min_diff / (mean_diff + 1e-6)

            # Calculate variance of differences
            variance = np.var(differences)

            # For spliced images: different regions have different "ghost" patterns
            # We detect this by looking at how clean the minimum is

            # Normalize: 0 = clear single compression, 1 = suspicious
            # A clear JPEG ghost (low ratio) is normal, unclear is suspicious
            score = ratio  # 0-1 range roughly

            # Also factor in absolute difference level
            # Very low differences everywhere might indicate heavy processing
            if mean_diff < 1.0:
                score += 0.2  # Suspiciously low differences

            return float(np.clip(score, 0.0, 1.0))

        except (ValueError, TypeError, OSError) as e:
            logger.warning(f"JPEGGhostDetector failed: {e}")
            return 0.0
