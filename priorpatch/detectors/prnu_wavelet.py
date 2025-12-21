"""
Proper PRNU (Photo Response Non-Uniformity) detector using wavelet denoising.

PRNU is a sensor fingerprint caused by manufacturing imperfections.
Every camera sensor has a unique noise pattern that appears in all images.

AI-generated images don't come from real sensors, so they lack proper
PRNU patterns. This is a fundamental physical difference.

Method:
1. Extract noise residual using wavelet denoising (Wiener filter in wavelet domain)
2. Analyze noise pattern characteristics
3. Check for sensor-like properties (spatial correlation, consistency)

Reference: Lukas et al., "Digital Camera Identification from Sensor
           Pattern Noise" (IEEE Trans. IFS, 2006)
"""

import logging
import numpy as np
from scipy import ndimage
from scipy.signal import wiener
from scipy.stats import kurtosis as scipy_kurtosis

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector

logger = logging.getLogger(__name__)


def haar_wavelet_decompose(image: np.ndarray, levels: int = 3) -> list:
    """Perform Haar wavelet decomposition.

    Args:
        image: Input image
        levels: Number of decomposition levels

    Returns:
        List of (LL, LH, HL, HH) tuples for each level
    """
    coeffs = []
    current = image.astype(np.float64)

    for _ in range(levels):
        h, w = current.shape

        # Make dimensions even
        if h % 2:
            current = current[:-1, :]
            h -= 1
        if w % 2:
            current = current[:, :-1]
            w -= 1

        if h < 4 or w < 4:
            break

        # Haar wavelet transform
        # Low-pass and high-pass filters
        even_rows = current[0::2, :]
        odd_rows = current[1::2, :]

        L = (even_rows + odd_rows) / 2  # Low-pass rows
        H = (even_rows - odd_rows) / 2  # High-pass rows

        even_cols_L = L[:, 0::2]
        odd_cols_L = L[:, 1::2]
        even_cols_H = H[:, 0::2]
        odd_cols_H = H[:, 1::2]

        LL = (even_cols_L + odd_cols_L) / 2  # Approximation
        LH = (even_cols_L - odd_cols_L) / 2  # Horizontal detail
        HL = (even_cols_H + odd_cols_H) / 2  # Vertical detail
        HH = (even_cols_H - odd_cols_H) / 2  # Diagonal detail

        coeffs.append((LL, LH, HL, HH))
        current = LL

    return coeffs


def haar_wavelet_reconstruct(coeffs: list) -> np.ndarray:
    """Reconstruct image from Haar wavelet coefficients.

    Args:
        coeffs: List of (LL, LH, HL, HH) tuples

    Returns:
        Reconstructed image
    """
    if not coeffs:
        return np.array([])

    # Start from the coarsest level
    current_LL = coeffs[-1][0]

    for i in range(len(coeffs) - 1, -1, -1):
        LL, LH, HL, HH = coeffs[i]

        if i < len(coeffs) - 1:
            # Use reconstructed LL from previous iteration
            LL = current_LL

        h, w = LL.shape

        # Inverse Haar transform
        even_cols_L = LL + LH
        odd_cols_L = LL - LH
        even_cols_H = HL + HH
        odd_cols_H = HL - HH

        # Interleave columns
        L = np.zeros((h, w * 2))
        L[:, 0::2] = even_cols_L
        L[:, 1::2] = odd_cols_L

        H = np.zeros((h, w * 2))
        H[:, 0::2] = even_cols_H
        H[:, 1::2] = odd_cols_H

        even_rows = L + H
        odd_rows = L - H

        # Interleave rows
        current_LL = np.zeros((h * 2, w * 2))
        current_LL[0::2, :] = even_rows
        current_LL[1::2, :] = odd_rows

    return current_LL


def wavelet_denoise(image: np.ndarray, sigma: float = None, levels: int = 3) -> np.ndarray:
    """Denoise image using wavelet soft thresholding.

    Args:
        image: Input image
        sigma: Noise standard deviation (estimated if None)
        levels: Number of wavelet levels

    Returns:
        Denoised image
    """
    # Decompose
    coeffs = haar_wavelet_decompose(image, levels)

    if not coeffs:
        return image

    # Estimate noise from finest level HH coefficients
    if sigma is None:
        hh = coeffs[0][3]
        sigma = np.median(np.abs(hh)) / 0.6745  # Robust estimator

    # Soft threshold detail coefficients
    denoised_coeffs = []
    for i, (LL, LH, HL, HH) in enumerate(coeffs):
        # Threshold increases with level
        threshold = sigma * np.sqrt(2 * np.log(LL.size)) * (2 ** i)

        # Soft thresholding
        LH_t = np.sign(LH) * np.maximum(np.abs(LH) - threshold, 0)
        HL_t = np.sign(HL) * np.maximum(np.abs(HL) - threshold, 0)
        HH_t = np.sign(HH) * np.maximum(np.abs(HH) - threshold, 0)

        denoised_coeffs.append((LL, LH_t, HL_t, HH_t))

    # Reconstruct
    return haar_wavelet_reconstruct(denoised_coeffs)


def extract_noise_residual(image: np.ndarray) -> np.ndarray:
    """Extract noise residual using wavelet denoising.

    The residual (original - denoised) contains the sensor noise pattern.

    Args:
        image: Input image (grayscale or color)

    Returns:
        Noise residual
    """
    if image.ndim == 3:
        # Process each channel
        residuals = []
        for c in range(image.shape[2]):
            channel = image[:, :, c].astype(np.float64)
            denoised = wavelet_denoise(channel)
            # Resize denoised to match original if needed
            if denoised.shape != channel.shape:
                denoised = ndimage.zoom(denoised, np.array(channel.shape) / np.array(denoised.shape), order=1)
            residual = channel - denoised
            residuals.append(residual)
        return np.mean(residuals, axis=0)
    else:
        image_f = image.astype(np.float64)
        denoised = wavelet_denoise(image_f)
        if denoised.shape != image_f.shape:
            denoised = ndimage.zoom(denoised, np.array(image_f.shape) / np.array(denoised.shape), order=1)
        return image_f - denoised


def analyze_noise_pattern(residual: np.ndarray) -> dict:
    """Analyze characteristics of noise residual.

    Real sensor noise has specific properties:
    - Near-Gaussian distribution
    - Consistent variance across regions
    - Specific spatial correlation structure

    Args:
        residual: Noise residual image

    Returns:
        Dictionary of noise features
    """
    features = {}

    # Feature 1: Variance (real sensors have consistent noise level)
    features['variance'] = np.var(residual)

    # Feature 2: Kurtosis (Gaussian has kurtosis ~0)
    flat_residual = residual.flatten()
    features['kurtosis'] = scipy_kurtosis(flat_residual)

    # Feature 3: Spatial correlation (check local consistency)
    # Real PRNU has spatial structure
    h, w = residual.shape
    if h >= 8 and w >= 8:
        # Divide into blocks and check variance consistency
        block_size = min(h, w) // 4
        block_vars = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = residual[i:i+block_size, j:j+block_size]
                block_vars.append(np.var(block))
        features['variance_consistency'] = np.std(block_vars) / (np.mean(block_vars) + 1e-10)
    else:
        features['variance_consistency'] = 1.0

    # Feature 4: Check for periodic structure (real PRNU shouldn't be periodic)
    from priorpatch.gpu_backend import fft2_shifted
    f_residual = np.abs(fft2_shifted(residual))
    dc = f_residual[h//2, w//2]
    if dc > 0:
        f_residual_norm = f_residual / dc
        # Check for peaks (excluding DC)
        f_residual_norm[h//2-2:h//2+3, w//2-2:w//2+3] = 0
        max_peak = np.max(f_residual_norm)
        features['periodicity'] = max_peak
    else:
        features['periodicity'] = 0.0

    return features


@register_detector
class PRNUWaveletDetector(DetectorInterface):
    """Proper PRNU detector using wavelet-based noise extraction.

    Extracts sensor noise pattern using wavelet denoising and analyzes
    its characteristics. Real camera images have specific PRNU patterns
    that AI-generated images lack.

    This is a fundamental physical fingerprint of real cameras.
    """

    name = 'prnu_wavelet'

    def __init__(self, wavelet_levels: int = 3):
        """Initialize detector.

        Args:
            wavelet_levels: Number of wavelet decomposition levels
        """
        self.wavelet_levels = wavelet_levels

    def score(self, patch: np.ndarray) -> float:
        """Score patch based on PRNU analysis.

        Args:
            patch: RGB image patch (H, W, 3)

        Returns:
            Anomaly score (0 = has PRNU, 1 = no PRNU = suspicious)
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        h, w = patch.shape[:2]
        min_size = 2 ** (self.wavelet_levels + 2)

        if h < min_size or w < min_size:
            return 0.0

        try:
            # Extract noise residual
            if patch.dtype == np.uint8:
                img = patch.astype(np.float64)
            else:
                img = patch

            residual = extract_noise_residual(img)

            if residual.size == 0:
                return 0.0

            # Analyze noise characteristics
            features = analyze_noise_pattern(residual)

            scores = []

            # Score 1: Variance check
            # Real PRNU has variance roughly proportional to image intensity
            # Typical values: 10-500 for 8-bit images
            var = features['variance']
            if var < 1:
                # Too clean = no sensor noise = suspicious
                scores.append(0.8)
            elif var > 1000:
                # Too noisy = might be artificial noise
                scores.append(0.5)
            else:
                scores.append(0.0)

            # Score 2: Kurtosis check
            # Gaussian noise has kurtosis ~0, real PRNU is near-Gaussian
            kurt = features['kurtosis']
            # Expect kurtosis between -1 and 3 for natural images
            if abs(kurt) > 5:
                scores.append(0.6)
            elif abs(kurt) > 3:
                scores.append(0.3)
            else:
                scores.append(0.0)

            # Score 3: Variance consistency
            # Real PRNU should have consistent variance across image
            var_cons = features['variance_consistency']
            # Lower consistency value = more consistent = more natural
            if var_cons > 1.0:
                scores.append(0.5)
            elif var_cons > 0.5:
                scores.append(0.2)
            else:
                scores.append(0.0)

            # Score 4: Periodicity
            # Real PRNU shouldn't have strong periodic components
            period = features['periodicity']
            if period > 0.3:
                # Strong periodicity = likely artificial
                scores.append(0.4)
            else:
                scores.append(0.0)

            # Combine scores with weights
            weights = [0.35, 0.25, 0.20, 0.20]
            final_score = sum(s * w for s, w in zip(scores, weights))

            return float(np.clip(final_score, 0.0, 1.0))

        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"PRNUWaveletDetector failed: {e}")
            return 0.0
