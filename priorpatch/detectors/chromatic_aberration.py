"""
Chromatic Aberration (CA) consistency detector.

Real lenses can't focus all colors to the same spot, so red and blue
channels shift slightly - especially toward the edges. AI images don't
go through real optics, so they either have no color fringing or it
looks wrong.

Ref: Johnson & Farid, "Exposing digital forgeries through chromatic
     aberration" (2006)
"""

import logging
import numpy as np
from scipy import ndimage
from scipy.ndimage import shift as ndshift

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector

logger = logging.getLogger(__name__)


def find_edges(channel: np.ndarray, threshold: float = 30.0) -> np.ndarray:
    """Find edge pixels using Sobel gradient.

    Args:
        channel: Single color channel
        threshold: Gradient magnitude threshold

    Returns:
        Binary edge mask
    """
    # Sobel gradients
    gx = ndimage.sobel(channel.astype(np.float64), axis=1)
    gy = ndimage.sobel(channel.astype(np.float64), axis=0)

    # Gradient magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # Threshold
    edges = magnitude > threshold

    return edges


def estimate_channel_shift(ref_channel: np.ndarray, target_channel: np.ndarray,
                           edge_mask: np.ndarray, max_shift: float = 3.0) -> tuple:
    """Estimate sub-pixel shift between two channels at edge locations.

    Uses cross-correlation to find the shift that best aligns channels.

    Args:
        ref_channel: Reference channel (typically green)
        target_channel: Channel to measure shift for (red or blue)
        edge_mask: Binary mask of edge locations
        max_shift: Maximum shift to search for

    Returns:
        Tuple of (shift_y, shift_x, correlation)
    """
    if np.sum(edge_mask) < 10:
        return 0.0, 0.0, 0.0

    # Extract edge regions
    ref = ref_channel.astype(np.float64)
    target = target_channel.astype(np.float64)

    # Apply edge mask
    ref_masked = ref * edge_mask
    target_masked = target * edge_mask

    best_corr = -1
    best_shift = (0.0, 0.0)

    # Search for best shift (sub-pixel)
    shifts = np.linspace(-max_shift, max_shift, 13)  # -3 to +3 in 0.5 steps

    for sy in shifts:
        for sx in shifts:
            # Shift target channel
            shifted = ndshift(target_masked, (sy, sx), order=1, mode='constant')

            # Compute correlation at edge locations
            valid = edge_mask & (shifted > 0) & (ref_masked > 0)
            if np.sum(valid) < 10:
                continue

            ref_vals = ref_masked[valid]
            shifted_vals = shifted[valid]

            # Pearson correlation
            ref_norm = ref_vals - np.mean(ref_vals)
            shifted_norm = shifted_vals - np.mean(shifted_vals)

            denom = np.sqrt(np.sum(ref_norm**2) * np.sum(shifted_norm**2))
            if denom > 0:
                corr = np.sum(ref_norm * shifted_norm) / denom
            else:
                corr = 0

            if corr > best_corr:
                best_corr = corr
                best_shift = (sy, sx)

    return best_shift[0], best_shift[1], best_corr


def compute_radial_ca_profile(r_shift: np.ndarray, b_shift: np.ndarray,
                              h: int, w: int, n_rings: int = 5) -> np.ndarray:
    """Compute how CA varies with distance from center.

    Real lateral CA increases approximately linearly with distance
    from the optical center.

    Args:
        r_shift: Red channel shift map
        b_shift: Blue channel shift map
        h, w: Image dimensions
        n_rings: Number of radial rings to analyze

    Returns:
        Array of CA magnitudes for each ring
    """
    # Create radial distance map
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    max_dist = np.sqrt(cy**2 + cx**2)

    # Compute CA magnitude
    ca_magnitude = np.sqrt(r_shift**2 + b_shift**2)

    # Average CA in radial rings
    ring_ca = []
    for i in range(n_rings):
        inner = i * max_dist / n_rings
        outer = (i + 1) * max_dist / n_rings
        ring_mask = (dist >= inner) & (dist < outer)

        if np.sum(ring_mask) > 0:
            ring_ca.append(np.mean(ca_magnitude[ring_mask]))
        else:
            ring_ca.append(0.0)

    return np.array(ring_ca)


def check_ca_linearity(profile: np.ndarray) -> float:
    """Check if CA profile follows expected linear increase with radius.

    Real CA should increase approximately linearly from center to edge.

    Args:
        profile: Radial CA profile

    Returns:
        Linearity score (0 = linear/natural, 1 = non-linear/suspicious)
    """
    if len(profile) < 3:
        return 0.5

    # Fit linear model
    x = np.arange(len(profile))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, profile, rcond=None)[0]

    # Compute residuals
    predicted = slope * x + intercept
    residuals = profile - predicted
    mse = np.mean(residuals**2)

    # Variance of profile
    profile_var = np.var(profile)

    if profile_var > 0:
        r_squared = 1 - mse / profile_var
        r_squared = max(0, r_squared)
    else:
        # No variance = no CA = suspicious
        return 0.8

    # High R^2 = linear = natural
    # Low R^2 = non-linear = suspicious
    return float(1.0 - r_squared)


@register_detector
class ChromaticAberrationDetector(DetectorInterface):
    """Detect inconsistent or absent chromatic aberration.

    Real camera lenses cause chromatic aberration (color fringing)
    that follows predictable patterns:
    - Increases radially from optical center
    - Red and blue shift in opposite directions
    - Magnitude depends on lens quality

    AI-generated images lack realistic CA, making this a strong
    indicator of synthetic origin.
    """

    name = 'chromatic_aberration'

    def __init__(self, edge_threshold: float = 30.0):
        """Initialize detector.

        Args:
            edge_threshold: Threshold for edge detection
        """
        self.edge_threshold = edge_threshold

    def score(self, patch: np.ndarray) -> float:
        """Score patch based on chromatic aberration analysis.

        Args:
            patch: RGB image patch (H, W, 3)

        Returns:
            Anomaly score (0 = realistic CA, 1 = suspicious CA/no CA)
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        h, w = patch.shape[:2]

        if h < 32 or w < 32:
            return 0.0

        try:
            # Extract channels
            if patch.dtype == np.uint8:
                r = patch[:, :, 0].astype(np.float64)
                g = patch[:, :, 1].astype(np.float64)
                b = patch[:, :, 2].astype(np.float64)
            else:
                r = patch[:, :, 0]
                g = patch[:, :, 1]
                b = patch[:, :, 2]

            # Find edges (using green channel as reference)
            edges = find_edges(g, self.edge_threshold)

            if np.sum(edges) < 50:
                # Not enough edges to analyze
                return 0.3  # Slightly suspicious (smooth regions)

            # Measure channel shifts at edges
            r_sy, r_sx, r_corr = estimate_channel_shift(g, r, edges)
            b_sy, b_sx, b_corr = estimate_channel_shift(g, b, edges)

            # Feature 1: Total CA magnitude
            ca_r = np.sqrt(r_sy**2 + r_sx**2)
            ca_b = np.sqrt(b_sy**2 + b_sx**2)
            total_ca = (ca_r + ca_b) / 2

            # Very low CA (<0.1 pixels) is suspicious for real photos
            # Very high CA (>2 pixels) is also suspicious
            if total_ca < 0.1:
                ca_score = 0.5  # Suspicious - no CA
            elif total_ca > 2.0:
                ca_score = 0.4  # Suspicious - excessive CA
            else:
                ca_score = 0.0  # Normal CA range

            # Feature 2: R and B should shift in opposite directions
            # (both radially outward, but R focuses longer, B shorter)
            r_dir = np.array([r_sy, r_sx])
            b_dir = np.array([b_sy, b_sx])

            if np.linalg.norm(r_dir) > 0.1 and np.linalg.norm(b_dir) > 0.1:
                # Check if R and B shift in roughly opposite directions
                r_dir_norm = r_dir / (np.linalg.norm(r_dir) + 1e-10)
                b_dir_norm = b_dir / (np.linalg.norm(b_dir) + 1e-10)
                direction_dot = np.dot(r_dir_norm, b_dir_norm)

                # Should be negative (opposite) for real CA
                if direction_dot > 0.5:
                    # Same direction = suspicious
                    direction_score = 0.5
                else:
                    direction_score = 0.0
            else:
                # Not enough CA to measure direction
                direction_score = 0.3

            # Feature 3: Correlation quality
            # Low correlation = noisy/unreliable measurement
            avg_corr = (r_corr + b_corr) / 2
            if avg_corr < 0.5:
                corr_score = 0.3  # Can't reliably measure CA
            else:
                corr_score = 0.0

            # Feature 4: Analyze spatial consistency
            # Divide into quadrants and check CA consistency
            quadrant_cas = []
            mid_h, mid_w = h // 2, w // 2

            for qy in [0, mid_h]:
                for qx in [0, mid_w]:
                    q_g = g[qy:qy+mid_h, qx:qx+mid_w]
                    q_r = r[qy:qy+mid_h, qx:qx+mid_w]
                    q_edges = edges[qy:qy+mid_h, qx:qx+mid_w]

                    if np.sum(q_edges) > 10:
                        _, _, q_corr = estimate_channel_shift(q_g, q_r, q_edges)
                        quadrant_cas.append(q_corr)

            # Check consistency across quadrants
            if len(quadrant_cas) >= 2:
                ca_variance = np.var(quadrant_cas)
                # High variance = inconsistent CA = suspicious
                consistency_score = min(ca_variance * 2, 1.0)
            else:
                consistency_score = 0.3

            # Combine features
            final_score = (
                0.25 * ca_score +
                0.25 * direction_score +
                0.20 * corr_score +
                0.30 * consistency_score
            )

            return float(np.clip(final_score, 0.0, 1.0))

        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"ChromaticAberrationDetector failed: {e}")
            return 0.0
