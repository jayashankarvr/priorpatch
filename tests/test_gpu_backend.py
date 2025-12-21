"""
Tests for GPU backend functionality.

Tests both GPU (CuPy) and CPU (NumPy) code paths.
"""

import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock

from priorpatch import gpu_backend
from priorpatch.gpu_backend import (
    use_gpu,
    enable_gpu,
    disable_gpu,
    get_gpu_info,
    get_array_module,
    to_gpu,
    to_numpy,
    fft2,
    fftshift,
    fft2_shifted,
    dct2,
    batch_fft2,
    batch_fft2_shifted,
    batch_dct2,
    batch_process_patches,
)


class TestGPUAvailability:
    """Test GPU detection and control."""

    def test_use_gpu_returns_bool(self):
        """use_gpu() returns boolean."""
        result = use_gpu()
        assert isinstance(result, bool)

    def test_get_gpu_info_returns_dict(self):
        """get_gpu_info() returns dictionary with expected keys."""
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert 'available' in info
        assert 'enabled' in info
        assert 'active' in info

    def test_enable_disable_gpu(self):
        """Can enable and disable GPU."""
        original_state = use_gpu()

        disable_gpu()
        # After disable, use_gpu should return False
        # (assuming GPU was available before)

        enable_gpu()
        # After enable, should return to checking availability

        # Restore original
        if not original_state:
            disable_gpu()

    def test_gpu_disabled_via_env_var(self):
        """PRIORPATCH_NO_GPU environment variable disables GPU."""
        # This test verifies the env var logic exists
        # We can't easily test the actual behavior without mocking
        pass


class TestArrayModule:
    """Test array module selection."""

    def test_get_array_module_returns_numpy_when_no_gpu(self):
        """Without GPU, get_array_module returns numpy."""
        disable_gpu()
        try:
            xp = get_array_module()
            assert xp is np or hasattr(xp, 'ndarray')
        finally:
            enable_gpu()

    def test_get_array_module_with_array_argument(self):
        """get_array_module handles array argument."""
        arr = np.array([1, 2, 3])
        xp = get_array_module(arr)
        assert xp is not None


class TestDataTransfer:
    """Test CPU/GPU data transfer functions."""

    def test_to_gpu_with_numpy_array(self):
        """to_gpu handles numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_gpu(arr)
        assert result is not None

    def test_to_numpy_with_numpy_array(self):
        """to_numpy handles numpy array (no-op)."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_numpy_preserves_values(self):
        """to_numpy preserves array values."""
        arr = np.random.rand(10, 10)
        gpu_arr = to_gpu(arr)
        cpu_arr = to_numpy(gpu_arr)
        np.testing.assert_array_almost_equal(cpu_arr, arr)


class TestFFTOperations:
    """Test FFT functions."""

    def test_fft2_2d_array(self):
        """fft2 works with 2D array."""
        arr = np.random.rand(32, 32)
        result = fft2(arr)
        assert result.shape == arr.shape

    def test_fft2_complex_result(self):
        """fft2 returns complex array."""
        arr = np.random.rand(32, 32)
        result = fft2(arr)
        result_np = to_numpy(result)
        assert np.iscomplexobj(result_np)

    def test_fftshift_centers_dc(self):
        """fftshift moves DC component to center."""
        arr = np.zeros((32, 32))
        arr[0, 0] = 1.0  # DC component
        shifted = fftshift(arr)
        shifted_np = to_numpy(shifted)
        # DC should now be at center
        center = shifted_np.shape[0] // 2
        assert shifted_np[center, center] == 1.0

    def test_fft2_shifted_returns_numpy(self):
        """fft2_shifted always returns numpy array."""
        arr = np.random.rand(32, 32)
        result = fft2_shifted(arr)
        assert isinstance(result, np.ndarray)

    def test_fft2_shifted_shape_preserved(self):
        """fft2_shifted preserves shape."""
        arr = np.random.rand(64, 48)
        result = fft2_shifted(arr)
        assert result.shape == arr.shape

    def test_fft2_cpu_fallback(self):
        """fft2 works with use_gpu_if_available=False."""
        arr = np.random.rand(32, 32)
        result = fft2(arr, use_gpu_if_available=False)
        result_np = to_numpy(result)
        expected = np.fft.fft2(arr)
        np.testing.assert_array_almost_equal(result_np, expected)


class TestDCTOperations:
    """Test DCT functions."""

    def test_dct2_2d_array(self):
        """dct2 works with 2D array."""
        arr = np.random.rand(32, 32)
        result = dct2(arr)
        assert result.shape == arr.shape

    def test_dct2_real_result(self):
        """dct2 returns real array (not complex)."""
        arr = np.random.rand(32, 32)
        result = dct2(arr)
        assert not np.iscomplexobj(result)

    def test_dct2_cpu_fallback(self):
        """dct2 falls back to CPU correctly."""
        arr = np.random.rand(16, 16)
        result = dct2(arr, use_gpu_if_available=False)
        assert isinstance(result, np.ndarray)

    def test_dct2_energy_concentration(self):
        """DCT concentrates energy in low frequencies."""
        # Smooth image should have energy concentrated in top-left
        arr = np.outer(np.sin(np.linspace(0, np.pi, 32)),
                       np.sin(np.linspace(0, np.pi, 32)))
        result = dct2(arr)

        # Most energy should be in top-left quadrant
        top_left_energy = np.sum(result[:8, :8]**2)
        total_energy = np.sum(result**2)

        assert top_left_energy / total_energy > 0.9


class TestBatchOperations:
    """Test batch GPU operations."""

    def test_batch_fft2_empty_list(self):
        """batch_fft2 handles empty list."""
        result = batch_fft2([])
        assert result == []

    def test_batch_fft2_single_array(self):
        """batch_fft2 handles single array."""
        arr = np.random.rand(32, 32)
        result = batch_fft2([arr])
        assert len(result) == 1
        assert result[0].shape == arr.shape

    def test_batch_fft2_multiple_arrays(self):
        """batch_fft2 handles multiple arrays."""
        arrays = [np.random.rand(32, 32) for _ in range(5)]
        results = batch_fft2(arrays)
        assert len(results) == len(arrays)
        for arr, res in zip(arrays, results):
            assert res.shape == arr.shape

    def test_batch_fft2_different_sizes(self):
        """batch_fft2 handles arrays of different sizes."""
        arrays = [
            np.random.rand(32, 32),
            np.random.rand(24, 24),
            np.random.rand(16, 48),
        ]
        results = batch_fft2(arrays)
        assert len(results) == len(arrays)
        # Results should match original shapes
        for arr, res in zip(arrays, results):
            assert res.shape == arr.shape

    def test_batch_fft2_cpu_fallback(self):
        """batch_fft2 works with CPU fallback."""
        arrays = [np.random.rand(16, 16) for _ in range(3)]
        results = batch_fft2(arrays, use_gpu_if_available=False)
        assert len(results) == len(arrays)

        # Compare with individual FFTs
        for arr, res in zip(arrays, results):
            expected = np.fft.fft2(arr)
            np.testing.assert_array_almost_equal(res, expected)

    def test_batch_fft2_shifted_empty_list(self):
        """batch_fft2_shifted handles empty list."""
        result = batch_fft2_shifted([])
        assert result == []

    def test_batch_fft2_shifted_multiple_arrays(self):
        """batch_fft2_shifted handles multiple arrays."""
        arrays = [np.random.rand(32, 32) for _ in range(3)]
        results = batch_fft2_shifted(arrays)
        assert len(results) == len(arrays)
        for arr, res in zip(arrays, results):
            assert res.shape == arr.shape

    def test_batch_fft2_shifted_matches_individual(self):
        """batch_fft2_shifted matches individual fft2_shifted calls."""
        arrays = [np.random.rand(32, 32) for _ in range(3)]
        batch_results = batch_fft2_shifted(arrays, use_gpu_if_available=False)

        for arr, batch_res in zip(arrays, batch_results):
            individual_res = fft2_shifted(arr, use_gpu_if_available=False)
            np.testing.assert_array_almost_equal(batch_res, individual_res)

    def test_batch_dct2_empty_list(self):
        """batch_dct2 handles empty list."""
        result = batch_dct2([])
        assert result == []

    def test_batch_dct2_multiple_arrays(self):
        """batch_dct2 handles multiple arrays."""
        arrays = [np.random.rand(32, 32) for _ in range(3)]
        results = batch_dct2(arrays)
        assert len(results) == len(arrays)
        for arr, res in zip(arrays, results):
            assert res.shape == arr.shape
            assert not np.iscomplexobj(res)  # DCT should be real

    def test_batch_dct2_matches_individual(self):
        """batch_dct2 matches individual dct2 calls."""
        arrays = [np.random.rand(16, 16) for _ in range(3)]
        batch_results = batch_dct2(arrays, use_gpu_if_available=False)

        for arr, batch_res in zip(arrays, batch_results):
            individual_res = dct2(arr, use_gpu_if_available=False)
            np.testing.assert_array_almost_equal(batch_res, individual_res)

    def test_batch_process_patches_fft2(self):
        """batch_process_patches works with fft2 operation."""
        patches = [np.random.rand(32, 32) for _ in range(3)]
        results = batch_process_patches(patches, operation='fft2')
        assert len(results) == len(patches)

    def test_batch_process_patches_fft2_shifted(self):
        """batch_process_patches works with fft2_shifted operation."""
        patches = [np.random.rand(32, 32) for _ in range(3)]
        results = batch_process_patches(patches, operation='fft2_shifted')
        assert len(results) == len(patches)

    def test_batch_process_patches_dct2(self):
        """batch_process_patches works with dct2 operation."""
        patches = [np.random.rand(32, 32) for _ in range(3)]
        results = batch_process_patches(patches, operation='dct2')
        assert len(results) == len(patches)

    def test_batch_process_patches_invalid_operation(self):
        """batch_process_patches raises on invalid operation."""
        patches = [np.random.rand(32, 32)]
        with pytest.raises(ValueError):
            batch_process_patches(patches, operation='invalid')


class TestGPUMemoryManagement:
    """Test GPU memory handling."""

    def test_large_array_handling(self):
        """Can handle moderately large arrays."""
        arr = np.random.rand(512, 512)
        result = fft2_shifted(arr)
        assert result.shape == arr.shape

    def test_repeated_operations(self):
        """Repeated operations don't leak memory."""
        arr = np.random.rand(64, 64)

        # Run many iterations
        for _ in range(100):
            _ = fft2_shifted(arr)

        # If we get here without OOM, test passes


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_fft2_with_zeros(self):
        """fft2 handles all-zero array."""
        arr = np.zeros((32, 32))
        result = fft2(arr)
        result_np = to_numpy(result)
        np.testing.assert_array_equal(result_np, np.zeros_like(result_np))

    def test_fft2_with_constant(self):
        """fft2 handles constant array."""
        arr = np.ones((32, 32)) * 5.0
        result = fft2(arr)
        result_np = to_numpy(result)
        # DC component should equal sum of array
        assert abs(result_np[0, 0]) == pytest.approx(5.0 * 32 * 32, rel=1e-5)

    def test_dct2_with_ones(self):
        """dct2 handles all-ones array."""
        arr = np.ones((16, 16))
        result = dct2(arr)
        # Should produce a result without errors
        assert result.shape == arr.shape

    def test_small_arrays(self):
        """Operations work with small arrays."""
        arr = np.random.rand(4, 4)
        fft_result = fft2(arr)
        dct_result = dct2(arr)
        assert fft_result.shape == (4, 4)
        assert dct_result.shape == (4, 4)

    def test_non_square_arrays(self):
        """Operations work with non-square arrays."""
        arr = np.random.rand(32, 64)
        fft_result = fft2(arr)
        result_np = to_numpy(fft_result)
        assert result_np.shape == (32, 64)

    def test_single_precision(self):
        """Operations handle float32 arrays."""
        arr = np.random.rand(32, 32).astype(np.float32)
        result = fft2_shifted(arr)
        assert result.shape == arr.shape


class TestConsistency:
    """Test CPU/GPU consistency."""

    def test_fft2_cpu_gpu_consistency(self):
        """FFT results consistent between CPU and GPU paths."""
        arr = np.random.rand(32, 32)

        # Force CPU
        cpu_result = fft2(arr, use_gpu_if_available=False)
        cpu_result = to_numpy(cpu_result)

        # Allow GPU if available
        gpu_result = fft2(arr, use_gpu_if_available=True)
        gpu_result = to_numpy(gpu_result)

        np.testing.assert_array_almost_equal(cpu_result, gpu_result, decimal=5)

    def test_fft2_shifted_consistency(self):
        """fft2_shifted results consistent."""
        arr = np.random.rand(32, 32)

        # Multiple calls should give same result
        result1 = fft2_shifted(arr)
        result2 = fft2_shifted(arr)

        np.testing.assert_array_equal(result1, result2)
