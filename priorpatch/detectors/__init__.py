"""
Forensic detectors package.

Detectors are automatically discovered and registered when this package is imported.
Any Python file in this directory that contains a class decorated with @register_detector
will be automatically loaded.

To add a new detector:
1. Create a new .py file in this directory
2. Implement a class inheriting from DetectorInterface
3. Decorate it with @register_detector
4. The detector will be automatically discovered!

No need to manually edit this file.
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Files to skip during auto-discovery
_SKIP_MODULES = {'__init__', 'base', 'registry'}


def _discover_detectors() -> List[str]:
    """Auto-discover and import all detector modules.

    Returns:
        List of successfully imported module names
    """
    imported = []
    package_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        module_name = module_info.name

        if module_name in _SKIP_MODULES:
            continue

        if module_name.startswith('_'):
            continue

        try:
            full_name = f'priorpatch.detectors.{module_name}'
            importlib.import_module(full_name)
            imported.append(module_name)
            logger.debug(f"Loaded detector module: {module_name}")
        except ImportError as e:
            logger.warning(f"Failed to import detector module {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error loading detector module {module_name}: {e}")

    return imported


# Auto-discover all detectors on import
_discovered_modules = _discover_detectors()

# Export registry for convenience
from priorpatch.detectors.registry import DETECTOR_REGISTRY, register_detector, get_detector_class
from priorpatch.detectors.base import DetectorInterface

__all__ = [
    'DETECTOR_REGISTRY',
    'register_detector',
    'get_detector_class',
    'DetectorInterface',
] + _discovered_modules
