"""
Base class that all detectors inherit from.
"""

from abc import ABC, abstractmethod
import numpy as np


class DetectorInterface(ABC):
    """
    Base class for detectors. Subclass this and implement score().
    """

    name = 'base_detector'

    @abstractmethod
    def score(self, patch: np.ndarray) -> float:
        """
        Score a patch. Higher = more suspicious.

        Args:
            patch: RGB image patch, shape (H, W, 3)

        Returns:
            Anomaly score (float). Will be normalized by ensemble.
        """
        raise NotImplementedError

    def get_config(self) -> dict:
        """Serialize detector params for multiprocessing."""
        return {}

    @classmethod
    def from_config(cls, config: dict):
        """Create detector from config dict."""
        return cls(**config)
