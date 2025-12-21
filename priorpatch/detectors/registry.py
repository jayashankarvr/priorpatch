"""
Simple registry so detectors can be looked up by name.
"""

from typing import Dict, Type, Optional
from priorpatch.detectors.base import DetectorInterface

# Global registry mapping detector names to classes
DETECTOR_REGISTRY: Dict[str, Type[DetectorInterface]] = {}


def register_detector(cls: Type[DetectorInterface]) -> Type[DetectorInterface]:
    """
    Decorator to add a detector to the registry.

    Usage:
        @register_detector
        class MyDetector(DetectorInterface):
            name = 'my_detector'
            ...
    """
    if cls.name in DETECTOR_REGISTRY:
        raise ValueError(f"Detector '{cls.name}' already registered")

    DETECTOR_REGISTRY[cls.name] = cls
    return cls


def get_detector_class(name: str) -> Optional[Type[DetectorInterface]]:
    """Get a detector class by name, or None if not found."""
    return DETECTOR_REGISTRY.get(name)
