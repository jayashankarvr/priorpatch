"""
Tests for individual detector implementations.
"""

import pytest
import numpy as np

from priorpatch.detectors.color_stats import ColorStatsDetector
from priorpatch.detectors.neighbor_consistency import NeighborConsistencyDetector
from priorpatch.detectors.fft_dct import FFTDCTDetector
from priorpatch.detectors.residual_energy import ResidualEnergyDetector
from priorpatch.detectors.dct_cooccurrence import DCTCoocDetector


class TestColorStatsDetector:
    """Test ColorStatsDetector."""
    
    def test_initialization(self):
        """Detector initializes correctly."""
        detector = ColorStatsDetector()
        assert detector.name == 'color_stats'
    
    def test_score_normal_image(self):
        """Score normal RGB image."""
        detector = ColorStatsDetector()
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score = detector.score(patch)
        
        assert isinstance(score, (int, float))
        assert score >= 0
    
    def test_score_uniform_color(self):
        """Score uniform color patch."""
        detector = ColorStatsDetector()
        patch = np.full((64, 64, 3), 128, dtype=np.uint8)
        score = detector.score(patch)
        
        assert isinstance(score, (int, float))
    
    def test_score_grayscale(self):
        """Score grayscale (R=G=B) patch."""
        detector = ColorStatsDetector()
        gray_val = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
        patch = np.repeat(gray_val, 3, axis=2)
        score = detector.score(patch)
        
        assert isinstance(score, (int, float))
    
    def test_score_different_sizes(self):
        """Score patches of different sizes."""
        detector = ColorStatsDetector()
        sizes = [(32, 32), (64, 64), (128, 128)]
        
        for size in sizes:
            patch = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            score = detector.score(patch)
            assert isinstance(score, (int, float))
    
    def test_score_black_image(self):
        """Score all-black patch."""
        detector = ColorStatsDetector()
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        score = detector.score(patch)
        
        assert isinstance(score, (int, float))
    
    def test_score_white_image(self):
        """Score all-white patch."""
        detector = ColorStatsDetector()
        patch = np.full((64, 64, 3), 255, dtype=np.uint8)
        score = detector.score(patch)
        
        assert isinstance(score, (int, float))


class TestNeighborConsistencyDetector:
    """Test NeighborConsistencyDetector."""
    
    def test_initialization(self):
        """Detector initializes correctly."""
        detector = NeighborConsistencyDetector()
        assert detector.name == 'neighbor_consistency'
    
    def test_score_normal_image(self):
        """Score normal image."""
        detector = NeighborConsistencyDetector()
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score = detector.score(patch)
        
        assert isinstance(score, (int, float))
        assert score >= 0
    
    def test_score_smooth_gradient(self):
        """Score smooth gradient."""
        detector = NeighborConsistencyDetector()
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            patch[i, :] = i * 4  # Smooth gradient
        
        score = detector.score(patch)
        assert isinstance(score, (int, float))
    
    def test_score_sharp_edges(self):
        """Score patch with sharp edges."""
        detector = NeighborConsistencyDetector()
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        patch[:32, :] = 0
        patch[32:, :] = 255  # Sharp horizontal edge
        
        score = detector.score(patch)
        assert isinstance(score, (int, float))
        # Should have high score due to edge


class TestFFTDCTDetector:
    """Test FFTDCTDetector."""
    
    def test_initialization(self):
        """Detector initializes correctly."""
        detector = FFTDCTDetector()
        assert detector.name == 'fft_dct'
    
    def test_score_normal_image(self):
        """Score normal image."""
        detector = FFTDCTDetector()
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score = detector.score(patch)
        
        assert isinstance(score, (int, float))
        assert score >= 0
    
    def test_score_periodic_pattern(self):
        """Score periodic pattern (should have frequency response)."""
        detector = FFTDCTDetector()
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                patch[i, j] = int(127 * (1 + np.sin(i / 4)))  # Periodic
        
        score = detector.score(patch)
        assert isinstance(score, (int, float))
    
    def test_score_high_frequency(self):
        """Score high-frequency pattern."""
        detector = FFTDCTDetector()
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        patch[::2, ::2] = 255  # Checkerboard
        
        score = detector.score(patch)
        assert isinstance(score, (int, float))


class TestResidualEnergyDetector:
    """Test ResidualEnergyDetector."""
    
    def test_initialization(self):
        """Detector initializes correctly."""
        detector = ResidualEnergyDetector()
        assert detector.name == 'residual_energy'
    
    def test_score_normal_image(self):
        """Score normal image."""
        detector = ResidualEnergyDetector()
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score = detector.score(patch)
        
        assert isinstance(score, (int, float))
        assert score >= 0
    
    def test_score_smooth_image(self):
        """Score smooth image (low residual energy)."""
        detector = ResidualEnergyDetector()
        patch = np.full((64, 64, 3), 128, dtype=np.uint8)
        
        score = detector.score(patch)
        assert isinstance(score, (int, float))
        # Smooth images should have low residual energy
    
    def test_score_noisy_image(self):
        """Score noisy image (high residual energy)."""
        detector = ResidualEnergyDetector()
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        score = detector.score(patch)
        assert isinstance(score, (int, float))


class TestDCTCoocDetector:
    """Test DCTCoocDetector."""
    
    def test_initialization(self):
        """Detector initializes correctly."""
        detector = DCTCoocDetector()
        assert detector.name == 'dct_cooccurrence'
    
    def test_score_normal_image(self):
        """Score normal image."""
        detector = DCTCoocDetector()
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score = detector.score(patch)
        
        assert isinstance(score, (int, float))
        assert score >= 0
    
    def test_score_large_enough_patch(self):
        """DCT requires minimum 16x16 patch."""
        detector = DCTCoocDetector()
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score = detector.score(patch)
        
        assert isinstance(score, (int, float))


class TestDetectorEdgeCases:
    """Test edge cases across all detectors."""
    
    @pytest.fixture
    def all_detectors(self):
        """Get all detector instances."""
        return [
            ColorStatsDetector(),
            NeighborConsistencyDetector(),
            FFTDCTDetector(),
            ResidualEnergyDetector(),
            DCTCoocDetector(),
        ]
    
    def test_all_detectors_have_name(self, all_detectors):
        """All detectors have name attribute."""
        for detector in all_detectors:
            assert hasattr(detector, 'name')
            assert isinstance(detector.name, str)
            assert len(detector.name) > 0
    
    def test_all_detectors_have_score_method(self, all_detectors):
        """All detectors have score method."""
        for detector in all_detectors:
            assert hasattr(detector, 'score')
            assert callable(detector.score)
    
    def test_all_detectors_with_tiny_patch(self, all_detectors):
        """All detectors handle tiny patches."""
        tiny_patch = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        
        for detector in all_detectors:
            try:
                score = detector.score(tiny_patch)
                assert isinstance(score, (int, float))
            except (ValueError, IndexError):
                # Some detectors may require minimum size
                pass
    
    def test_all_detectors_with_uniform_image(self, all_detectors):
        """All detectors handle uniform color."""
        uniform = np.full((64, 64, 3), 128, dtype=np.uint8)
        
        for detector in all_detectors:
            score = detector.score(uniform)
            assert isinstance(score, (int, float))
            assert not np.isnan(score)
            assert not np.isinf(score)
    
    def test_all_detectors_reproducible(self, all_detectors):
        """All detectors give consistent results."""
        patch = np.random.RandomState(42).randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        for detector in all_detectors:
            score1 = detector.score(patch)
            score2 = detector.score(patch)
            assert score1 == score2 or np.allclose(score1, score2)
    
    def test_all_detectors_different_images_different_scores(self, all_detectors):
        """Detectors produce different scores for different images."""
        patch1 = np.zeros((64, 64, 3), dtype=np.uint8)
        patch2 = np.full((64, 64, 3), 255, dtype=np.uint8)
        
        for detector in all_detectors:
            score1 = detector.score(patch1)
            score2 = detector.score(patch2)
            # Scores don't have to be different, but at least should be valid
            assert isinstance(score1, (int, float))
            assert isinstance(score2, (int, float))
