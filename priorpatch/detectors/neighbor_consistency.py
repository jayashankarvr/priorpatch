"""
Checks if pixels match what you'd expect from their neighbors.
Edited regions often have weird local patterns.
"""

import numpy as np
from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
from priorpatch.utils import rgb_to_luminance


@register_detector
class NeighborConsistencyDetector(DetectorInterface):
    """
    Compares each pixel to the average of its 8 neighbors.
    Big prediction errors = suspicious.
    """

    name = 'neighbor_consistency'

    def score(self, patch: np.ndarray) -> float:
        """Score based on how poorly neighbors predict center pixels."""
        H, W = patch.shape[:2]
        if H < 3 or W < 3:
            return 0.0

        # Convert to luminance
        L = rgb_to_luminance(patch)
        
        # Sum of 8 neighbors (excluding center)
        neigh_sum = (
            L[0:-2, 0:-2] + L[0:-2, 1:-1] + L[0:-2, 2:] +
            L[1:-1, 0:-2]                  + L[1:-1, 2:] +
            L[2:, 0:-2]   + L[2:, 1:-1]    + L[2:, 2:]
        )
        
        center = L[1:-1, 1:-1]
        pred = neigh_sum / 8.0
        
        # Compute prediction residual
        residual = center - pred
        mse = float((residual ** 2).mean())
        
        # Normalize by local variance
        var_local = float(center.var()) + 1e-9
        
        return mse / var_local
