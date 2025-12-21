"""
Edge case tests for PriorPatch.
"""

import pytest
import numpy as np
from priorpatch import Ensemble


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_tiny_image(self):
        """Test with very small image."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = e.score_image(img, patch_size=64, stride=32)
        assert result.shape == (1, 1)
    
    def test_single_pixel(self):
        """Test with 1x1 image."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
        result = e.score_image(img)
        assert result.shape == (1, 1)
    
    def test_narrow_image(self):
        """Test with very narrow image."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (500, 10, 3), dtype=np.uint8)
        result = e.score_image(img)
        assert result.ndim == 2
    
    def test_wide_image(self):
        """Test with very wide image."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (10, 500, 3), dtype=np.uint8)
        result = e.score_image(img)
        assert result.ndim == 2
    
    def test_grayscale_image(self):
        """Test with grayscale image."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = e.score_image(img)
        assert result.ndim == 2
    
    def test_minimum_patch_size(self):
        """Test with minimum allowed patch size."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result = e.score_image(img, patch_size=8, stride=4)
        assert result.shape[0] > 10  # Many patches
    
    def test_maximum_patch_size(self):
        """Test with maximum allowed patch size."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (1500, 1500, 3), dtype=np.uint8)
        result = e.score_image(img, patch_size=1024, stride=512)
        assert result.ndim == 2
    
    def test_stride_equals_patch_size(self):
        """Test with no overlap (stride == patch_size)."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = e.score_image(img, patch_size=64, stride=64)
        assert result.shape == (4, 4)  # 256/64 = 4
    
    def test_stride_larger_than_patch(self):
        """Test with stride > patch_size (gaps between patches)."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = e.score_image(img, patch_size=32, stride=64)
        assert result.ndim == 2
    
    def test_uniform_image(self):
        """Test with completely uniform image."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = e.score_image(img)
        # Should not crash, but scores will be uniform
        assert result.ndim == 2
    
    def test_black_image(self):
        """Test with all-black image."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = e.score_image(img)
        assert result.ndim == 2
    
    def test_white_image(self):
        """Test with all-white image."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        result = e.score_image(img)
        assert result.ndim == 2
    
    def test_single_detector(self):
        """Test with only one detector enabled."""
        from priorpatch.detectors.color_stats import ColorStatsDetector
        e = Ensemble([ColorStatsDetector()])
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = e.score_image(img)
        assert result.ndim == 2
    
    def test_many_detectors(self):
        """Test that all standard detectors work together."""
        e = Ensemble.from_config('config/detectors.json')
        assert len(e.detectors) == 16  # Should have 16 detectors
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = e.score_image(img)
        assert result.ndim == 2
    
    def test_multiprocessing_few_patches(self):
        """Test multiprocessing with fewer than threshold patches."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        # Will have < 10 patches, should fallback to sequential
        result = e.score_image(img, use_multiprocessing=True)
        assert result.ndim == 2
    
    def test_multiprocessing_exact_threshold(self):
        """Test multiprocessing at exactly the threshold."""
        e = Ensemble.from_config('config/detectors.json')
        # Calculate size for exactly 10 patches
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = e.score_image(img, patch_size=64, stride=64, use_multiprocessing=True)
        assert result.ndim == 2
    
    def test_individual_results_consistency(self):
        """Test that individual results sum sensibly."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = e.score_image(img, return_individual=True)
        
        # Check structure
        assert hasattr(result, 'combined')
        assert hasattr(result, 'individual')
        assert hasattr(result, 'detector_names')
        
        # Check shapes match
        assert result.combined.shape == list(result.individual.values())[0].shape
        
        # Check all detectors present
        assert len(result.detector_names) == len(result.individual)
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        e = Ensemble.from_config('config/detectors.json')
        np.random.seed(42)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result1 = e.score_image(img, patch_size=64, stride=32)
        result2 = e.score_image(img, patch_size=64, stride=32)
        
        assert np.allclose(result1, result2)
    
    def test_sequential_vs_parallel_consistency(self):
        """Test that sequential and parallel give same results."""
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        seq = e.score_image(img, use_multiprocessing=False)
        par = e.score_image(img, use_multiprocessing=True, n_jobs=2)
        
        assert seq.shape == par.shape
        assert np.allclose(seq, par, rtol=1e-5)


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_invalid_patch_size_negative(self):
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="patch_size must be"):
            e.score_image(img, patch_size=-10)
    
    def test_invalid_patch_size_zero(self):
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="patch_size must be"):
            e.score_image(img, patch_size=0)
    
    def test_invalid_patch_size_too_large(self):
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="patch_size must be"):
            e.score_image(img, patch_size=5000)
    
    def test_invalid_stride_negative(self):
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="stride must be"):
            e.score_image(img, stride=-5)
    
    def test_invalid_stride_zero(self):
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="stride must be"):
            e.score_image(img, stride=0)
    
    def test_invalid_n_jobs_zero(self):
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="n_jobs"):
            e.score_image(img, n_jobs=0)
    
    def test_invalid_n_jobs_less_than_minus_one(self):
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="n_jobs"):
            e.score_image(img, n_jobs=-5)
    
    def test_invalid_image_type(self):
        e = Ensemble.from_config('config/detectors.json')
        with pytest.raises(TypeError, match="must be numpy array"):
            e.score_image([[1, 2, 3]])
    
    def test_invalid_image_dimensions(self):
        e = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (100, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            e.score_image(img)
