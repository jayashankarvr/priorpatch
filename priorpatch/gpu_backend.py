"""
GPU backend - unified interface for CuPy (GPU) or NumPy (CPU) operations.

Usage:
    xp = get_array_module()  # cupy if available, else numpy
    arr = xp.fft.fft2(data)
    result = to_numpy(arr)  # convert back to numpy
"""

import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Global state for GPU availability
_GPU_AVAILABLE: Optional[bool] = None
_GPU_ENABLED: bool = True  # User can disable GPU even if available
_CUPY_MODULE: Any = None
_CUPYX_SCIPY_FFT: Any = None


def _check_gpu_availability() -> bool:
    """Check if CuPy is available and functional.

    Returns:
        True if CuPy is available and GPU can be used
    """
    global _GPU_AVAILABLE, _CUPY_MODULE, _CUPYX_SCIPY_FFT

    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    # Check environment variable to disable GPU
    if os.environ.get('PRIORPATCH_NO_GPU', '').lower() in ('1', 'true', 'yes'):
        logger.info("GPU disabled via PRIORPATCH_NO_GPU environment variable")
        _GPU_AVAILABLE = False
        return False

    try:
        import cupy as cp

        # Try a simple operation to verify GPU is functional
        test_arr = cp.array([1.0, 2.0, 3.0])
        _ = cp.fft.fft(test_arr)
        cp.cuda.Stream.null.synchronize()

        # Also check for cupyx.scipy.fft for DCT operations
        try:
            from cupyx.scipy import fft as cupyx_fft
            _CUPYX_SCIPY_FFT = cupyx_fft
        except ImportError:
            logger.info("cupyx.scipy.fft not available, DCT will use CPU")
            _CUPYX_SCIPY_FFT = None

        _CUPY_MODULE = cp
        _GPU_AVAILABLE = True

        # Log GPU info
        device = cp.cuda.Device()
        mem_info = device.mem_info
        total_mem_gb = mem_info[1] / (1024**3)
        try:
            # Get device name (may vary by CuPy version)
            device_name = cp.cuda.runtime.getDeviceProperties(device.id)['name']
            if isinstance(device_name, bytes):
                device_name = device_name.decode()
        except Exception:
            device_name = f"GPU {device.id}"
        logger.info(f"GPU available: {device_name} ({total_mem_gb:.1f}GB)")

        return True

    except ImportError:
        logger.info("CuPy not installed, using CPU-only mode")
        _GPU_AVAILABLE = False
        return False

    except Exception as e:
        logger.warning(f"GPU check failed: {e}. Using CPU-only mode")
        _GPU_AVAILABLE = False
        return False


def use_gpu() -> bool:
    """Check if GPU should be used.

    Returns:
        True if GPU is available and enabled
    """
    return _GPU_ENABLED and _check_gpu_availability()


def enable_gpu(enabled: bool = True) -> None:
    """Enable or disable GPU usage.

    Args:
        enabled: Whether to enable GPU usage
    """
    global _GPU_ENABLED
    _GPU_ENABLED = enabled
    status = "enabled" if enabled else "disabled"
    logger.info(f"GPU {status}")


def disable_gpu() -> None:
    """Disable GPU usage."""
    enable_gpu(False)


def get_array_module(arr: Optional[np.ndarray] = None):
    """Get the appropriate array module (cupy or numpy).

    Args:
        arr: Optional array to check. If provided, returns module that
             owns the array.

    Returns:
        cupy module if GPU is available and enabled, else numpy
    """
    if arr is not None and use_gpu() and _CUPY_MODULE is not None:
        # Check if array is already on GPU
        return _CUPY_MODULE.get_array_module(arr)

    if use_gpu() and _CUPY_MODULE is not None:
        return _CUPY_MODULE

    return np


def to_gpu(arr: np.ndarray) -> Any:
    """Transfer numpy array to GPU.

    Args:
        arr: NumPy array

    Returns:
        CuPy array if GPU available, else same numpy array
    """
    if use_gpu() and _CUPY_MODULE is not None:
        return _CUPY_MODULE.asarray(arr)
    return arr


def to_numpy(arr: Any) -> np.ndarray:
    """Transfer array to CPU as numpy array.

    Args:
        arr: Array (can be numpy or cupy)

    Returns:
        NumPy array
    """
    if use_gpu() and _CUPY_MODULE is not None:
        if hasattr(arr, 'get'):
            return arr.get()
    return np.asarray(arr)


def fft2(arr: np.ndarray, use_gpu_if_available: bool = True) -> Any:
    """2D FFT with GPU acceleration.

    Args:
        arr: Input array
        use_gpu_if_available: Whether to use GPU if available

    Returns:
        FFT result (on GPU if GPU used, else numpy)
    """
    if use_gpu_if_available and use_gpu() and _CUPY_MODULE is not None:
        cp = _CUPY_MODULE
        gpu_arr = cp.asarray(arr)
        return cp.fft.fft2(gpu_arr)
    return np.fft.fft2(arr)


def fftshift(arr: Any, use_gpu_if_available: bool = True) -> Any:
    """FFT shift with GPU acceleration.

    Args:
        arr: Input array
        use_gpu_if_available: Whether to use GPU if available

    Returns:
        Shifted array
    """
    if use_gpu_if_available and use_gpu() and _CUPY_MODULE is not None:
        cp = _CUPY_MODULE
        if not isinstance(arr, cp.ndarray):
            arr = cp.asarray(arr)
        return cp.fft.fftshift(arr)
    return np.fft.fftshift(arr)


def fft2_shifted(arr: np.ndarray, use_gpu_if_available: bool = True) -> np.ndarray:
    """Compute 2D FFT and shift, returning numpy result.

    This is a convenience function that handles GPU transfer internally
    and always returns a numpy array.

    Args:
        arr: Input array
        use_gpu_if_available: Whether to use GPU if available

    Returns:
        Shifted FFT as numpy array
    """
    if use_gpu_if_available and use_gpu() and _CUPY_MODULE is not None:
        cp = _CUPY_MODULE
        gpu_arr = cp.asarray(arr)
        result = cp.fft.fftshift(cp.fft.fft2(gpu_arr))
        return result.get()
    return np.fft.fftshift(np.fft.fft2(arr))


def dct2(arr: np.ndarray, norm: str = 'ortho', use_gpu_if_available: bool = True) -> np.ndarray:
    """2D DCT (Type II) with optional GPU acceleration.

    Note: CuPy's DCT support is limited, so this may fall back to CPU.

    Args:
        arr: Input 2D array
        norm: Normalization mode ('ortho' recommended)
        use_gpu_if_available: Whether to use GPU if available

    Returns:
        DCT result as numpy array
    """
    # Try GPU-accelerated DCT if available
    if use_gpu_if_available and use_gpu() and _CUPYX_SCIPY_FFT is not None:
        try:
            cp = _CUPY_MODULE
            gpu_arr = cp.asarray(arr)
            # cupyx.scipy.fft.dctn does N-dimensional DCT
            result = _CUPYX_SCIPY_FFT.dctn(gpu_arr, type=2, norm=norm)
            return result.get()
        except Exception as e:
            logger.info(f"GPU DCT failed, falling back to CPU: {e}")

    # CPU fallback using scipy
    from scipy.fftpack import dct
    return dct(dct(arr.T, norm=norm).T, norm=norm)


def batch_fft2(arrays: list, use_gpu_if_available: bool = True) -> list:
    """Batch 2D FFT for multiple arrays.

    More efficient than calling fft2 in a loop when using GPU
    due to reduced transfer overhead.

    Args:
        arrays: List of 2D arrays
        use_gpu_if_available: Whether to use GPU if available

    Returns:
        List of FFT results as numpy arrays
    """
    if not arrays:
        return []

    if use_gpu_if_available and use_gpu() and _CUPY_MODULE is not None:
        cp = _CUPY_MODULE

        # Stack arrays and transfer to GPU at once
        try:
            # Pad arrays to same size if needed
            shapes = [arr.shape for arr in arrays]
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)

            padded = []
            for arr in arrays:
                if arr.shape == (max_h, max_w):
                    padded.append(arr)
                else:
                    pad_h = max_h - arr.shape[0]
                    pad_w = max_w - arr.shape[1]
                    padded.append(np.pad(arr, ((0, pad_h), (0, pad_w))))

            # Transfer to GPU
            stacked = cp.asarray(np.stack(padded))

            # Batch FFT
            results = cp.fft.fft2(stacked, axes=(-2, -1))

            # Transfer back and unpack
            results_np = results.get()
            return [results_np[i, :shapes[i][0], :shapes[i][1]]
                    for i in range(len(arrays))]

        except Exception as e:
            logger.info(f"Batch GPU FFT failed, falling back to CPU: {e}")

    # CPU fallback
    return [np.fft.fft2(arr) for arr in arrays]


def batch_fft2_shifted(arrays: list, use_gpu_if_available: bool = True) -> list:
    """Batch 2D FFT with shift for multiple arrays.

    Performs FFT2 and fftshift in a single GPU transfer for efficiency.

    Args:
        arrays: List of 2D arrays
        use_gpu_if_available: Whether to use GPU if available

    Returns:
        List of shifted FFT results as numpy arrays
    """
    if not arrays:
        return []

    if use_gpu_if_available and use_gpu() and _CUPY_MODULE is not None:
        cp = _CUPY_MODULE

        try:
            shapes = [arr.shape for arr in arrays]
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)

            padded = []
            for arr in arrays:
                if arr.shape == (max_h, max_w):
                    padded.append(arr)
                else:
                    pad_h = max_h - arr.shape[0]
                    pad_w = max_w - arr.shape[1]
                    padded.append(np.pad(arr, ((0, pad_h), (0, pad_w))))

            stacked = cp.asarray(np.stack(padded))
            results = cp.fft.fftshift(cp.fft.fft2(stacked, axes=(-2, -1)), axes=(-2, -1))
            results_np = results.get()
            return [results_np[i, :shapes[i][0], :shapes[i][1]]
                    for i in range(len(arrays))]

        except Exception as e:
            logger.info(f"Batch GPU FFT shifted failed, falling back to CPU: {e}")

    return [np.fft.fftshift(np.fft.fft2(arr)) for arr in arrays]


def batch_dct2(arrays: list, norm: str = 'ortho', use_gpu_if_available: bool = True) -> list:
    """Batch 2D DCT for multiple arrays.

    Args:
        arrays: List of 2D arrays
        norm: Normalization mode ('ortho' recommended)
        use_gpu_if_available: Whether to use GPU if available

    Returns:
        List of DCT results as numpy arrays
    """
    if not arrays:
        return []

    if use_gpu_if_available and use_gpu() and _CUPYX_SCIPY_FFT is not None:
        cp = _CUPY_MODULE

        try:
            shapes = [arr.shape for arr in arrays]
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)

            padded = []
            for arr in arrays:
                if arr.shape == (max_h, max_w):
                    padded.append(arr)
                else:
                    pad_h = max_h - arr.shape[0]
                    pad_w = max_w - arr.shape[1]
                    padded.append(np.pad(arr, ((0, pad_h), (0, pad_w))))

            stacked = cp.asarray(np.stack(padded))
            results = _CUPYX_SCIPY_FFT.dctn(stacked, type=2, norm=norm, axes=(-2, -1))
            results_np = results.get()
            return [results_np[i, :shapes[i][0], :shapes[i][1]]
                    for i in range(len(arrays))]

        except Exception as e:
            logger.info(f"Batch GPU DCT failed, falling back to CPU: {e}")

    # CPU fallback
    from scipy.fftpack import dct
    return [dct(dct(arr.T, norm=norm).T, norm=norm) for arr in arrays]


def batch_process_patches(
    patches: list,
    operation: str = 'fft2',
    use_gpu_if_available: bool = True,
    **kwargs
) -> list:
    """Generic batch processor for patches.

    This is the main entry point for batch GPU operations on image patches.
    Minimizes GPU memory transfers by processing all patches at once.

    Args:
        patches: List of 2D arrays (patches)
        operation: Operation to perform ('fft2', 'fft2_shifted', 'dct2')
        use_gpu_if_available: Whether to use GPU if available
        **kwargs: Additional arguments for the operation

    Returns:
        List of results as numpy arrays
    """
    if operation == 'fft2':
        return batch_fft2(patches, use_gpu_if_available)
    elif operation == 'fft2_shifted':
        return batch_fft2_shifted(patches, use_gpu_if_available)
    elif operation == 'dct2':
        return batch_dct2(patches, use_gpu_if_available=use_gpu_if_available, **kwargs)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def get_gpu_info() -> dict:
    """Get information about GPU availability and status.

    Returns:
        Dictionary with GPU information
    """
    info = {
        'available': _check_gpu_availability(),
        'enabled': _GPU_ENABLED,
        'active': use_gpu(),
    }

    if info['available'] and _CUPY_MODULE is not None:
        try:
            cp = _CUPY_MODULE
            device = cp.cuda.Device()
            mem_info = device.mem_info

            # Get device name (may vary by CuPy version)
            try:
                device_name = cp.cuda.runtime.getDeviceProperties(device.id)['name']
                if isinstance(device_name, bytes):
                    device_name = device_name.decode()
            except Exception:
                device_name = f"GPU {device.id}"

            info['device_name'] = device_name
            info['device_id'] = device.id
            info['memory_free_gb'] = mem_info[0] / (1024**3)
            info['memory_total_gb'] = mem_info[1] / (1024**3)
            info['cupy_version'] = cp.__version__
        except Exception as e:
            info['error'] = str(e)

    return info


# Initialize on import (lazy - just checks availability)
_check_gpu_availability()
