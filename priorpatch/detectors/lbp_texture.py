"""
Local Binary Pattern (LBP) texture detector.

LBP describes texture by comparing each pixel to its neighbors - if neighbor
is brighter, that's a 1, else 0. Creates an 8-bit pattern per pixel.

AI-generated images have subtly different textures that LBP can pick up.
Research shows 83-99% accuracy on deepfakes using this approach.

Ref: "LBPNet: Exploiting texture descriptor for deepfake detection" (2022)
"""

import logging
import numpy as np
from scipy import stats

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector

logger = logging.getLogger(__name__)

# Precomputed uniform LBP mapping for 8-point LBP (computed once at module load)
_UNIFORM_MAP_8 = None


def _build_uniform_map(n_points: int = 8) -> np.ndarray:
    """Build uniform LBP lookup table.

    Precomputes mapping from any pattern to its uniform code.
    Uniform patterns have <= 2 transitions between 0 and 1.
    """
    n_patterns = 2 ** n_points
    uniform_map = np.zeros(n_patterns, dtype=np.uint8)
    uniform_code = 0

    for pattern in range(n_patterns):
        # Count transitions using bitwise operations
        # XOR pattern with itself rotated by 1 bit
        rotated = ((pattern << 1) | (pattern >> (n_points - 1))) & ((1 << n_points) - 1)
        transitions = bin(pattern ^ rotated).count('1')

        if transitions <= 2:
            uniform_map[pattern] = uniform_code
            uniform_code += 1
        else:
            uniform_map[pattern] = n_points + 1  # Non-uniform bin

    return uniform_map


def _get_uniform_map_8() -> np.ndarray:
    """Get cached uniform map for 8-point LBP."""
    global _UNIFORM_MAP_8
    if _UNIFORM_MAP_8 is None:
        _UNIFORM_MAP_8 = _build_uniform_map(8)
    return _UNIFORM_MAP_8


def compute_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """Compute Local Binary Pattern for an image (vectorized).

    For each pixel, compare with neighbors on a circle.
    If neighbor >= center, set bit to 1, else 0.

    Args:
        image: Grayscale image (H, W)
        radius: Radius of the circle
        n_points: Number of points on the circle

    Returns:
        LBP image with same shape as input
    """
    h, w = image.shape

    # Precompute neighbor offsets
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    dy_offsets = np.round(radius * np.sin(angles)).astype(int)
    dx_offsets = np.round(radius * np.cos(angles)).astype(int)

    # Extract center region
    center = image[radius:h-radius, radius:w-radius]

    # Compute LBP using vectorized operations
    lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint16)

    for bit, (dy, dx) in enumerate(zip(dy_offsets, dx_offsets)):
        # Extract neighbor pixels at this offset
        neighbor = image[radius+dy:h-radius+dy, radius+dx:w-radius+dx]
        # Set bit where neighbor >= center
        lbp |= (neighbor >= center).astype(np.uint16) << bit

    # Pad back to original size with zeros
    result = np.zeros((h, w), dtype=np.uint8)
    result[radius:h-radius, radius:w-radius] = lbp.astype(np.uint8)

    return result


def compute_uniform_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """Compute uniform Local Binary Pattern (vectorized).

    Uniform patterns have at most 2 transitions between 0 and 1.
    Non-uniform patterns are grouped into a single bin.

    Args:
        image: Grayscale image
        radius: Circle radius
        n_points: Number of points

    Returns:
        Uniform LBP image
    """
    lbp = compute_lbp(image, radius, n_points)

    # Use precomputed lookup table for 8-point LBP
    if n_points == 8:
        uniform_map = _get_uniform_map_8()
    else:
        uniform_map = _build_uniform_map(n_points)

    # Vectorized lookup - much faster than nested loops
    uniform_lbp = uniform_map[lbp]

    return uniform_lbp


def compute_lbp_histogram(lbp_image: np.ndarray, n_bins: int = 59) -> np.ndarray:
    """Compute normalized histogram of LBP image.

    Args:
        lbp_image: LBP-encoded image
        n_bins: Number of histogram bins

    Returns:
        Normalized histogram
    """
    hist, _ = np.histogram(lbp_image.flatten(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)

    # Normalize
    total = np.sum(hist)
    if total > 0:
        hist = hist / total

    return hist


def compute_lbp_variance(lbp_image: np.ndarray, original: np.ndarray) -> float:
    """Compute variance of LBP values weighted by local contrast.

    Higher variance indicates more diverse textures (natural).
    AI images often have lower LBP variance (more uniform textures).

    Args:
        lbp_image: LBP-encoded image
        original: Original grayscale image

    Returns:
        Weighted LBP variance
    """
    # Compute local contrast
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(original.astype(np.float64), size=3)
    local_var = uniform_filter((original.astype(np.float64) - local_mean)**2, size=3)

    # Weight LBP values by local variance
    weights = np.sqrt(local_var + 1)

    # Compute weighted variance of LBP
    weighted_mean = np.average(lbp_image, weights=weights)
    weighted_var = np.average((lbp_image - weighted_mean)**2, weights=weights)

    return weighted_var


def analyze_lbp_uniformity(hist: np.ndarray) -> float:
    """Analyze how uniform the LBP histogram is.

    Natural images typically have specific LBP distributions.
    AI-generated images often have more uniform or unusual distributions.

    Args:
        hist: Normalized LBP histogram

    Returns:
        Uniformity score (higher = more uniform = suspicious)
    """
    # Compute entropy - higher entropy = more uniform distribution
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10))

    # Max entropy for uniform distribution
    max_entropy = np.log2(len(hist))

    # Normalized entropy (0 = peaked, 1 = uniform)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return norm_entropy


def analyze_lbp_pattern_concentration(hist: np.ndarray, top_k: int = 5) -> float:
    """Analyze concentration of top LBP patterns.

    Natural images typically have certain patterns dominating.
    AI images may have different concentration patterns.

    Args:
        hist: Normalized LBP histogram
        top_k: Number of top patterns to consider

    Returns:
        Concentration score
    """
    # Sort histogram
    sorted_hist = np.sort(hist)[::-1]

    # Concentration in top-k patterns
    top_k_sum = np.sum(sorted_hist[:top_k])

    # Expected natural concentration is around 0.3-0.5 for top 5
    # Too high (>0.7) or too low (<0.2) is suspicious
    if top_k_sum < 0.2:
        return 0.6  # Too spread out
    elif top_k_sum > 0.7:
        return 0.6  # Too concentrated
    else:
        return 0.0  # Normal range


@register_detector
class LBPTextureDetector(DetectorInterface):
    """Detect AI-generated images using Local Binary Pattern texture analysis.

    LBP captures local micro-texture patterns. AI-generated images often
    have subtly different texture characteristics:
    - More uniform LBP histograms
    - Different pattern concentrations
    - Unusual spatial distribution of patterns

    Research shows 83-99% accuracy on deepfakes using LBP features.
    """

    name = 'lbp_texture'

    def __init__(self, radius: int = 1, n_points: int = 8):
        """
        Args:
            radius: LBP circle radius
            n_points: Number of sampling points
        """
        self.radius = radius
        self.n_points = n_points

    def get_config(self) -> dict:
        """Serialize for multiprocessing."""
        return {'radius': self.radius, 'n_points': self.n_points}

    def score(self, patch: np.ndarray) -> float:
        """Score patch based on LBP texture analysis.

        Args:
            patch: RGB image patch (H, W, 3)

        Returns:
            Anomaly score (0 = natural texture, 1 = suspicious texture)
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        h, w = patch.shape[:2]
        min_size = (self.radius + 1) * 4

        if h < min_size or w < min_size:
            return 0.0

        try:
            # Convert to grayscale using standard luminance conversion
            from priorpatch.utils import rgb_to_luminance
            gray = rgb_to_luminance(patch).astype(np.float64)

            # Compute uniform LBP
            lbp = compute_uniform_lbp(gray, self.radius, self.n_points)

            # Crop to valid region (exclude border)
            valid_lbp = lbp[self.radius:-self.radius, self.radius:-self.radius]
            valid_gray = gray[self.radius:-self.radius, self.radius:-self.radius]

            if valid_lbp.size < 100:
                return 0.0

            # Feature 1: LBP histogram analysis
            # 59 bins for 8-point uniform LBP (58 uniform + 1 non-uniform)
            hist = compute_lbp_histogram(valid_lbp, n_bins=self.n_points + 2)

            # Feature 2: Histogram uniformity
            uniformity = analyze_lbp_uniformity(hist)

            # Feature 3: Pattern concentration
            concentration_score = analyze_lbp_pattern_concentration(hist)

            # Feature 4: LBP variance (texture diversity)
            lbp_var = compute_lbp_variance(valid_lbp, valid_gray)

            # Low variance indicates uniform/fake texture
            # Natural images typically have variance > 100
            var_score = max(0, 1.0 - lbp_var / 200.0)

            # Feature 5: Check for unusual kurtosis in histogram
            # Natural LBP histograms have specific shape characteristics
            if len(hist) > 4:
                kurtosis = stats.kurtosis(hist)
                # Extreme kurtosis (either very high or very low) is suspicious
                kurt_score = min(abs(kurtosis) / 10.0, 1.0)
            else:
                kurt_score = 0.0

            # Combine features
            # Higher uniformity = more suspicious (AI tends to be more uniform)
            # Pattern concentration issues = suspicious
            # Low variance = suspicious
            # Extreme kurtosis = suspicious
            final_score = (
                0.30 * uniformity +
                0.20 * concentration_score +
                0.30 * var_score +
                0.20 * kurt_score
            )

            return float(np.clip(final_score, 0.0, 1.0))

        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"LBPTextureDetector failed: {e}")
            return 0.0
