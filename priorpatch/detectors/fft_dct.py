"""
Frequency domain analysis - looks for weird patterns in the FFT.

Supports GPU acceleration via CuPy when available.
"""

import numpy as np
from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
from priorpatch.utils import rgb_to_luminance
from priorpatch.gpu_backend import fft2_shifted


@register_detector
class FFTDCTDetector(DetectorInterface):
    """
    Checks if frequency spectrum follows natural 1/f^2 power law.
    
    Real photos have ~1/f^2 falloff in frequency domain. Resizing, JPEG
    compression, or synthetic generation mess this up. Also catches weird
    periodic patterns that shouldn't be there.
    """
    
    name = 'fft_dct'
    
    def score(self, patch: np.ndarray) -> float:
        """Check if frequency spectrum looks natural (1/f^2 falloff)."""
        H, W = patch.shape[:2]
        if H < 16 or W < 16:
            return 0.0

        # Convert to luminance
        L = rgb_to_luminance(patch)
        
        # Apply Hanning window to reduce edge effects
        win = np.outer(np.hanning(H), np.hanning(W))
        Lw = L * win

        # Compute 2D FFT and power spectrum (GPU-accelerated if available)
        F = fft2_shifted(Lw)
        P = np.abs(F) ** 2 + 1e-12
        
        # Compute radial power spectrum
        cy, cx = H // 2, W // 2
        y, x = np.indices((H, W))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
        maxr = min(cx, cy)
        
        radial = np.zeros(maxr + 1, dtype=np.float32)
        counts = np.zeros_like(radial)
        
        for rr in range(maxr + 1):
            mask = (r == rr)
            counts[rr] = np.sum(mask)
            if counts[rr] > 0:
                radial[rr] = np.mean(P[mask])
        
        # Analyze power-law slope
        valid = np.where(counts[1:maxr//2] > 0)[0] + 1
        if valid.size < 3:
            return 0.0
        
        freqs = valid.astype(np.float32)
        vals = radial[valid]
        
        # Fit power-law: log(P) = b*log(f) + c
        b = np.polyfit(np.log(freqs), np.log(vals + 1e-12), 1)[0]
        slope = -b  # Natural images typically have slope ~2
        
        # Expected range for natural images
        low, high = 0.8, 2.5
        if slope < low:
            dev = (low - slope) / (high - low)
        elif slope > high:
            dev = (slope - high) / (high - low)
        else:
            dev = 0.0
        
        # Detect anomalous peaks (from resampling, compression artifacts)
        med = float(np.median(radial[1:]) + 1e-12)
        peaks = float((radial > 5.0 * med).sum())
        peak_penalty = peaks / (len(radial) + 1.0)
        
        return float(dev + peak_penalty)
