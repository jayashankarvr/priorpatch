"""
Lighting consistency detector for image forensics.

Real photographs have consistent lighting direction across the scene.
Spliced or composited images often have lighting inconsistencies because
different source images were lit from different directions.

This detector estimates local lighting direction and checks for consistency.

Method:
1. Estimate surface normals from shading (shape-from-shading)
2. Estimate light direction in local regions
3. Check consistency of light direction across the image

Reference: Johnson & Farid, "Exposing Digital Forgeries Through
           Inconsistencies in Lighting" (2005)
"""

import logging
from typing import Tuple, Optional

import numpy as np
from scipy import ndimage

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
from priorpatch.utils import rgb_to_luminance

logger = logging.getLogger(__name__)


def estimate_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate image gradients using Sobel operators.

    Args:
        image: Grayscale image

    Returns:
        Tuple of (gradient_x, gradient_y)
    """
    # Sobel gradients
    gx = ndimage.sobel(image.astype(np.float64), axis=1)
    gy = ndimage.sobel(image.astype(np.float64), axis=0)

    return gx, gy


def estimate_light_direction_region(
    gx: np.ndarray,
    gy: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """Estimate dominant light direction from gradients.

    Uses the assumption that for Lambertian surfaces under directional
    lighting, the gradient direction indicates the light direction.

    Args:
        gx: X gradient
        gy: Y gradient
        mask: Optional mask for valid pixels

    Returns:
        Tuple of (light_x, light_y, confidence)
    """
    if mask is not None:
        gx = gx[mask]
        gy = gy[mask]

    gx = gx.flatten()
    gy = gy.flatten()

    # Filter out low gradient (flat) regions
    magnitude = np.sqrt(gx**2 + gy**2)
    threshold = np.percentile(magnitude, 50)
    valid = magnitude > threshold

    if np.sum(valid) < 10:
        return 0.0, 0.0, 0.0

    gx_valid = gx[valid]
    gy_valid = gy[valid]

    # Estimate light direction using least squares
    # The gradient points in the direction of increasing brightness
    # which is roughly the light direction for convex surfaces

    # Use circular statistics for direction estimation
    angles = np.arctan2(gy_valid, gx_valid)

    # Circular mean
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))

    mean_angle = np.arctan2(mean_sin, mean_cos)

    # Circular variance as confidence measure
    R = np.sqrt(mean_sin**2 + mean_cos**2)  # 0 to 1, higher = more consistent

    light_x = np.cos(mean_angle)
    light_y = np.sin(mean_angle)

    return float(light_x), float(light_y), float(R)


def analyze_lighting_consistency(
    image: np.ndarray,
    grid_size: int = 4
) -> dict:
    """Analyze lighting direction consistency across image regions.

    Args:
        image: Grayscale image
        grid_size: Number of regions per dimension

    Returns:
        Dictionary of lighting features
    """
    h, w = image.shape

    # Compute gradients
    gx, gy = estimate_gradient(image)

    # Divide into regions
    region_h = h // grid_size
    region_w = w // grid_size

    if region_h < 16 or region_w < 16:
        return {'consistency': 0.5, 'confidence': 0.0}

    light_directions = []
    confidences = []

    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * region_h
            y_end = (i + 1) * region_h
            x_start = j * region_w
            x_end = (j + 1) * region_w

            region_gx = gx[y_start:y_end, x_start:x_end]
            region_gy = gy[y_start:y_end, x_start:x_end]

            lx, ly, conf = estimate_light_direction_region(region_gx, region_gy)

            if conf > 0.1:  # Only consider high-confidence estimates
                light_directions.append((lx, ly))
                confidences.append(conf)

    if len(light_directions) < 2:
        return {'consistency': 0.5, 'confidence': 0.0}

    # Compute angular variance of light directions
    angles = [np.arctan2(ly, lx) for lx, ly in light_directions]

    # Circular statistics
    mean_sin = np.mean([np.sin(a) for a in angles])
    mean_cos = np.mean([np.cos(a) for a in angles])

    # R is the mean resultant length (0 = uniform, 1 = concentrated)
    R = np.sqrt(mean_sin**2 + mean_cos**2)

    # Circular variance
    circular_variance = 1 - R

    return {
        'consistency': float(R),
        'circular_variance': float(circular_variance),
        'confidence': float(np.mean(confidences)),
        'num_regions': len(light_directions),
        'mean_angle': float(np.arctan2(mean_sin, mean_cos))
    }


def analyze_shadow_consistency(image: np.ndarray, threshold: float = 0.2) -> dict:
    """Analyze shadow edge directions for consistency.

    Shadow edges should be roughly parallel if from a single light source.

    Args:
        image: Grayscale image
        threshold: Edge detection threshold

    Returns:
        Dictionary of shadow features
    """
    h, w = image.shape

    # Find dark regions (potential shadows)
    dark_threshold = np.percentile(image, 20)
    dark_mask = image < dark_threshold

    # Find edges of dark regions
    gx, gy = estimate_gradient(image.astype(np.float64))
    magnitude = np.sqrt(gx**2 + gy**2)

    # Edge mask
    edge_threshold = np.percentile(magnitude, 80)
    edge_mask = magnitude > edge_threshold

    # Shadow edges: edges near dark regions
    dilated_dark = ndimage.binary_dilation(dark_mask, iterations=3)
    shadow_edges = edge_mask & dilated_dark

    if np.sum(shadow_edges) < 20:
        return {'shadow_consistency': 0.5, 'shadow_confidence': 0.0}

    # Get edge orientations at shadow boundaries
    edge_gx = gx[shadow_edges]
    edge_gy = gy[shadow_edges]

    angles = np.arctan2(edge_gy, edge_gx)

    # Compute circular statistics
    mean_sin = np.mean(np.sin(2 * angles))  # Double angle for orientation
    mean_cos = np.mean(np.cos(2 * angles))

    R = np.sqrt(mean_sin**2 + mean_cos**2)

    return {
        'shadow_consistency': float(R),
        'shadow_confidence': float(np.sum(shadow_edges) / (h * w)),
        'num_shadow_edges': int(np.sum(shadow_edges))
    }


@register_detector
class LightingConsistencyDetector(DetectorInterface):
    """Detect lighting direction inconsistencies.

    Real photographs have consistent lighting across the scene.
    Composited or spliced images often show lighting from multiple
    directions, indicating manipulation.

    Score interpretation:
    - 0.0: Consistent lighting direction - likely authentic
    - 1.0: Inconsistent lighting - likely manipulated

    Best for:
    - Spliced images with subjects from different photos
    - Composites with mismatched lighting
    - Images with added/removed objects

    Limitations:
    - Requires sufficient texture for gradient estimation
    - Multiple light sources in legitimate scenes can trigger false positives
    - Low-light or flat-lit images may not provide enough signal
    """

    name = 'lighting_consistency'

    def __init__(self, grid_size: int = 4, use_shadows: bool = True):
        """
        Args:
            grid_size: Number of regions per dimension for analysis
            use_shadows: Whether to also analyze shadow consistency
        """
        self.grid_size = grid_size
        self.use_shadows = use_shadows

    def get_config(self) -> dict:
        """Serialize for multiprocessing."""
        return {'grid_size': self.grid_size, 'use_shadows': self.use_shadows}

    def score(self, patch: np.ndarray) -> float:
        """Score patch based on lighting consistency.

        Args:
            patch: RGB image patch (H, W, 3)

        Returns:
            Anomaly score (0 = consistent lighting, 1 = inconsistent)
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        h, w = patch.shape[:2]
        min_size = self.grid_size * 16

        if h < min_size or w < min_size:
            return 0.0

        try:
            # Convert to grayscale using standard BT.601 weights
            gray = rgb_to_luminance(patch).astype(np.float64)

            # Normalize to [0, 1] range if needed
            if gray.max() > 1.0:
                gray = gray / 255.0

            scores = []

            # Analyze lighting direction consistency
            lighting_stats = analyze_lighting_consistency(gray, self.grid_size)

            if lighting_stats['confidence'] > 0.1:
                # Lower consistency = higher anomaly score
                lighting_score = 1.0 - lighting_stats['consistency']
                scores.append(lighting_score)

            # Analyze shadow consistency
            if self.use_shadows:
                shadow_stats = analyze_shadow_consistency(gray)

                if shadow_stats['shadow_confidence'] > 0.01:
                    shadow_score = 1.0 - shadow_stats['shadow_consistency']
                    scores.append(shadow_score)

            if not scores:
                return 0.0

            final_score = np.mean(scores)

            return float(np.clip(final_score, 0.0, 1.0))

        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"LightingConsistencyDetector failed: {e}")
            return 0.0
