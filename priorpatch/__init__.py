"""
PriorPatch - Image forensics using math instead of machine learning.

Detects manipulated/AI-generated images using signal processing.

GPU Acceleration:
    PriorPatch supports optional GPU acceleration via CuPy. When CuPy is
    installed and a CUDA GPU is available, FFT operations will automatically
    use the GPU for improved performance.

    To check GPU status:
        >>> import priorpatch
        >>> priorpatch.get_gpu_info()
        {'available': True, 'enabled': True, 'active': True, ...}

    To disable GPU (force CPU):
        >>> priorpatch.disable_gpu()

    To re-enable GPU:
        >>> priorpatch.enable_gpu()

    Or set environment variable:
        $ PRIORPATCH_NO_GPU=1 priorpatch analyze --input image.jpg
"""

# Single source of truth: version comes from pyproject.toml via importlib.metadata
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version('priorpatch')
except Exception:
    # Fallback for development installs or older Python
    __version__ = '0.1.0'

from priorpatch.core import Ensemble, AnalysisResult, PatchResult
from priorpatch.utils import load_image, save_heatmap, validate_path, rgb_to_luminance
from priorpatch.gpu_backend import (
    use_gpu,
    enable_gpu,
    disable_gpu,
    get_gpu_info,
)

__all__ = [
    'Ensemble',
    'AnalysisResult',
    'PatchResult',
    'load_image',
    'save_heatmap',
    'validate_path',
    'rgb_to_luminance',
    '__version__',
    # GPU control
    'use_gpu',
    'enable_gpu',
    'disable_gpu',
    'get_gpu_info',
]
