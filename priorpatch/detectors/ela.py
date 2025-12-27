"""
Error Level Analysis (ELA) detector.

Re-compress the image at a known quality and compare to original.
Edited regions compress differently and show up as anomalies.

One of the most popular forensic techniques - works well for JPEG edits.

Ref: Krawetz, "A Picture's Worth: Digital Image Analysis and Forensics" (2007)
"""

import logging
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector

logger = logging.getLogger(__name__)


def compute_ela(
    image: np.ndarray,
    quality: int = 90,
    scale: float = 10.0
) -> np.ndarray:
    """Compute Error Level Analysis image.

    Args:
        image: RGB image as numpy array (H, W, 3)
        quality: JPEG quality for recompression (default 90)
        scale: Scale factor for difference visualization

    Returns:
        ELA image showing compression differences
    """
    # Convert to PIL
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    pil_img = Image.fromarray(image)

    # Recompress at specified quality
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)

    recompressed = np.array(Image.open(buffer)).astype(np.float32)
    original = image.astype(np.float32)

    # Compute absolute difference
    ela = np.abs(original - recompressed)

    # Scale for better visualization
    ela = ela * scale

    return ela


def analyze_ela_statistics(ela: np.ndarray) -> dict:
    """Analyze ELA image statistics.

    Args:
        ela: ELA difference image

    Returns:
        Dictionary of statistical features
    """
    features = {}

    # Global statistics
    features['mean'] = float(np.mean(ela))
    features['std'] = float(np.std(ela))
    features['max'] = float(np.max(ela))

    # Per-channel statistics (if color)
    if ela.ndim == 3:
        for c, name in enumerate(['r', 'g', 'b']):
            features[f'{name}_mean'] = float(np.mean(ela[:, :, c]))
            features[f'{name}_std'] = float(np.std(ela[:, :, c]))

    # Block-based variance analysis
    h, w = ela.shape[:2]
    if h >= 16 and w >= 16:
        block_size = 16
        block_vars = []

        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = ela[i:i+block_size, j:j+block_size]
                block_vars.append(np.var(block))

        if block_vars:
            features['block_var_mean'] = float(np.mean(block_vars))
            features['block_var_std'] = float(np.std(block_vars))
            features['block_var_max'] = float(np.max(block_vars))

            # High variance ratio indicates manipulation
            high_var_threshold = np.percentile(block_vars, 90)
            features['high_var_ratio'] = float(
                np.sum(np.array(block_vars) > high_var_threshold) / len(block_vars)
            )

    return features


@register_detector
class ELADetector(DetectorInterface):
    """Error Level Analysis detector.

    Detects image manipulations by analyzing compression artifact
    inconsistencies. Works by recompressing the image and comparing
    to the original.

    Score interpretation:
    - 0.0: Uniform ELA (consistent compression) - likely authentic
    - 1.0: High ELA variance (inconsistent compression) - likely manipulated

    Best for:
    - JPEG images that have been edited
    - Copy-paste manipulations
    - Spliced images from different sources

    Limitations:
    - Less effective on heavily compressed images
    - May miss manipulations done before any JPEG compression
    - Can give false positives on high-detail areas
    """

    name = 'ela'

    def __init__(self, quality: int = 90, scale: float = 10.0):
        """
        Args:
            quality: JPEG quality for recompression (70-95 typical)
            scale: Scale factor for ELA visualization
        """
        self.quality = quality
        self.scale = scale

    def get_config(self) -> dict:
        """Serialize for multiprocessing."""
        return {'quality': self.quality, 'scale': self.scale}

    def score(self, patch: np.ndarray) -> float:
        """Score patch based on ELA analysis.

        Args:
            patch: RGB image patch (H, W, 3)

        Returns:
            Anomaly score (0 = consistent, 1 = inconsistent/suspicious)
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        h, w = patch.shape[:2]
        if h < 16 or w < 16:
            return 0.0

        try:
            # Compute ELA
            ela = compute_ela(patch, quality=self.quality, scale=self.scale)

            # Analyze statistics
            stats = analyze_ela_statistics(ela)

            scores = []

            # Score 1: Overall variance (high = suspicious)
            # Natural images have relatively uniform ELA
            mean_val = stats.get('mean', 0)
            std_val = stats.get('std', 0)

            if std_val > 0:
                cv = std_val / (mean_val + 1e-6)  # Coefficient of variation
                # High CV suggests non-uniform compression
                cv_score = min(cv / 2.0, 1.0)  # Normalize to 0-1
                scores.append(cv_score)

            # Score 2: Block variance inconsistency
            block_var_std = stats.get('block_var_std', 0)
            block_var_mean = stats.get('block_var_mean', 1)

            if block_var_mean > 0:
                block_cv = block_var_std / block_var_mean
                block_score = min(block_cv, 1.0)
                scores.append(block_score)

            # Score 3: High variance ratio
            high_var_ratio = stats.get('high_var_ratio', 0)
            # Some high variance blocks are normal, too many is suspicious
            ratio_score = min(high_var_ratio * 5, 1.0)  # 20% threshold
            scores.append(ratio_score)

            # Score 4: Channel consistency
            # Manipulations often affect channels differently
            if 'r_std' in stats and 'g_std' in stats and 'b_std' in stats:
                channel_stds = [stats['r_std'], stats['g_std'], stats['b_std']]
                channel_cv = np.std(channel_stds) / (np.mean(channel_stds) + 1e-6)
                channel_score = min(channel_cv, 1.0)
                scores.append(channel_score)

            if not scores:
                return 0.0

            # Combine scores
            final_score = np.mean(scores)

            return float(np.clip(final_score, 0.0, 1.0))

        except (ValueError, TypeError, OSError) as e:
            logger.warning(f"ELADetector failed: {e}")
            return 0.0
