"""
GAN Fingerprint detector - finds artifacts from neural network upsampling.

GANs use transposed convolutions to upsample, which leaves checkerboard
patterns in the frequency domain. Shows up as peaks at certain frequencies
that real cameras don't produce.

GPU acceleration via CuPy when available.
"""

import logging
import numpy as np

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
from priorpatch.utils import rgb_to_luminance
from priorpatch.gpu_backend import fft2_shifted

logger = logging.getLogger(__name__)


@register_detector
class GANFingerprintDetector(DetectorInterface):
    """Detect GAN/AI upsampling artifacts."""

    name = 'gan_fingerprint'

    def score(self, patch: np.ndarray) -> float:
        """
        Score based on GAN fingerprint detection.

        Higher score = more likely AI-generated.
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        H, W = patch.shape[:2]
        if H < 16 or W < 16:
            return 0.0

        try:
            # Convert to grayscale
            gray = rgb_to_luminance(patch)

            # Compute 2D FFT (GPU-accelerated if available)
            fft_shifted = fft2_shifted(gray)
            magnitude = np.abs(fft_shifted)

            # Log magnitude for better visualization of patterns
            log_mag = np.log1p(magnitude)

            # Normalize
            log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-6)

            # GAN artifacts appear at specific frequencies due to upsampling
            # Typically at N/2, N/4, N/8 etc. from center

            center_y, center_x = H // 2, W // 2

            # Check for peaks at characteristic GAN frequencies
            # These correspond to 2x, 4x upsampling artifacts

            scores = []

            # Check horizontal and vertical mid-frequencies (N/2 pattern)
            # These are where checkerboard artifacts appear
            mid_h = magnitude[center_y, 0:W//4].mean()
            mid_v = magnitude[0:H//4, center_x].mean()
            center_region = magnitude[center_y-2:center_y+2, center_x-2:center_x+2].mean()

            # Ratio of edge frequencies to center
            # GANs often have unusual energy at specific frequencies
            if center_region > 0:
                h_ratio = mid_h / center_region
                v_ratio = mid_v / center_region
                scores.append(min(h_ratio, v_ratio) * 0.5)

            # Check for periodic peaks (checkerboard pattern)
            # Sample at quarter frequencies
            quarter_points = [
                (center_y, center_x + W//4),
                (center_y, center_x - W//4),
                (center_y + H//4, center_x),
                (center_y - H//4, center_x),
            ]

            quarter_vals = []
            for y, x in quarter_points:
                if 0 <= y < H and 0 <= x < W:
                    quarter_vals.append(magnitude[y, x])

            if quarter_vals and center_region > 0:
                quarter_mean = np.mean(quarter_vals)
                # Unusual peaks at quarter frequencies suggest upsampling
                quarter_ratio = quarter_mean / center_region
                scores.append(quarter_ratio * 0.3)

            # Check for overall spectral flatness
            # Real images have 1/f noise (power decreases with frequency)
            # AI images often have flatter or unusual spectra

            # Radial average of spectrum (vectorized)
            y_coords, x_coords = np.ogrid[:H, :W]
            r = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            r_int = r.astype(int)

            # Bin by radius using vectorized operations
            max_r = min(center_y, center_x)

            # Flatten arrays for bincount
            r_flat = r_int.ravel()
            mag_flat = magnitude.ravel()

            # Mask for valid radii
            valid_mask = r_flat < max_r
            r_valid = r_flat[valid_mask]
            mag_valid = mag_flat[valid_mask]

            # Use bincount for fast summation
            radial_profile = np.bincount(r_valid, weights=mag_valid, minlength=max_r).astype(np.float64)
            counts = np.bincount(r_valid, minlength=max_r).astype(np.float64)

            # Avoid division by zero
            counts[counts == 0] = 1
            radial_profile = radial_profile / counts

            # Check if spectrum follows expected 1/f pattern
            if len(radial_profile) > 10:
                # Fit log-log slope (should be around -1 for natural images)
                x = np.log(np.arange(1, len(radial_profile)) + 1)
                y = np.log(radial_profile[1:] + 1e-6)

                # Simple linear regression for slope
                slope = np.polyfit(x, y, 1)[0]

                # Natural images: slope around -1 to -2
                # AI images: often flatter (closer to 0) or steeper
                expected_slope = -1.5
                slope_deviation = abs(slope - expected_slope) / 2.0
                scores.append(np.clip(slope_deviation, 0, 1) * 0.4)

            # Check for unusual symmetry (GANs often produce symmetric artifacts)
            top_half = magnitude[:center_y, :]
            bottom_half = np.flipud(magnitude[center_y:, :])
            min_h = min(top_half.shape[0], bottom_half.shape[0])
            if min_h > 0:
                symmetry_diff = np.mean(np.abs(top_half[:min_h] - bottom_half[:min_h]))
                # Very high symmetry is suspicious
                symmetry_score = 1.0 - min(symmetry_diff / (magnitude.mean() + 1e-6), 1.0)
                if symmetry_score > 0.9:  # Unusually symmetric
                    scores.append(0.3)

            if not scores:
                return 0.0

            return float(np.clip(np.mean(scores), 0.0, 1.0))

        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"GANFingerprintDetector failed: {e}")
            return 0.0
