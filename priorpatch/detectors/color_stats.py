"""
Checks if RGB channels correlate like they should in natural images.
"""

import numpy as np
from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector


@register_detector
class ColorStatsDetector(DetectorInterface):
    """
    Checks RGB channel correlations against expected values.
    
    Natural photos have R-G ~0.9, G-B ~0.9, R-B ~0.85 correlation.
    Edited regions often break these relationships.
    """
    
    name = 'color_stats'
    
    def score(self, patch: np.ndarray) -> float:
        """How much RGB correlations deviate from natural image expectations."""
        # Validate input
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0
        
        # Reshape to (N_pixels, 3) for correlation computation
        C = patch.reshape(-1, 3).astype(np.float32)
        
        # Need sufficient pixels for meaningful statistics
        if C.shape[0] < 16:
            return 0.0
        
        # Compute covariance matrix
        cov = np.cov(C, rowvar=False)
        
        # Convert to correlation matrix
        diag = np.diag(cov).copy()  # Copy to make it writable
        diag[diag == 0] = 1e-9  # Avoid division by zero
        corr = cov / np.sqrt(np.outer(diag, diag))
        corr = np.clip(corr, -1.0, 1.0)
        
        # Expected correlation matrix for natural images
        target = np.array([
            [1.0, 0.9, 0.85],
            [0.9, 1.0, 0.9],
            [0.85, 0.9, 1.0]
        ], dtype=np.float32)
        
        # Compute deviation (ignore diagonal)
        mask = np.ones_like(corr) - np.eye(3)
        diff = (corr - target) * mask
        
        # Return normalized Frobenius norm of difference
        return float(np.linalg.norm(diff) / np.linalg.norm(mask))
