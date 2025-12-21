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
from priorpatch.detectors.jpeg_ghost import JPEGGhostDetector
from priorpatch.detectors.gan_fingerprint import GANFingerprintDetector
from priorpatch.detectors.noise_consistency import NoiseConsistencyDetector


class TestColorStatsDetector:
    def test_initialization(self):
        detector = ColorStatsDetector()
        assert detector.name == 'color_stats'
    
    def test_score_returns_float(self, sample_rgb_patch):
        detector = ColorStatsDetector()
        score = detector.score(sample_rgb_patch)
        assert isinstance(score, float)
        assert score >= 0.0
    
    def test_score_with_small_patch(self):
        detector = ColorStatsDetector()
        tiny_patch = np.random.randint(0, 256, (2, 2, 3), dtype=np.uint8)
        score = detector.score(tiny_patch)
        assert score == 0.0  # Should return 0 for patches with < 16 pixels
    
    def test_score_with_invalid_dimensions(self):
        detector = ColorStatsDetector()
        invalid_patch = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        score = detector.score(invalid_patch)
        assert score == 0.0
    
    def test_score_with_uniform_color(self):
        detector = ColorStatsDetector()
        uniform_patch = np.ones((64, 64, 3), dtype=np.uint8) * 128
        score = detector.score(uniform_patch)
        assert isinstance(score, float)


class TestNeighborConsistencyDetector:
    def test_initialization(self):
        detector = NeighborConsistencyDetector()
        assert detector.name == 'neighbor_consistency'
    
    def test_score_returns_float(self, sample_rgb_patch):
        detector = NeighborConsistencyDetector()
        score = detector.score(sample_rgb_patch)
        assert isinstance(score, float)
        assert score >= 0.0
    
    def test_score_with_small_patch(self):
        detector = NeighborConsistencyDetector()
        tiny_patch = np.random.randint(0, 256, (2, 2, 3), dtype=np.uint8)
        score = detector.score(tiny_patch)
        assert score == 0.0
    
    def test_score_with_smooth_patch(self):
        detector = NeighborConsistencyDetector()
        # Create smooth gradient
        smooth = np.linspace(0, 255, 64*64).reshape(64, 64)
        smooth_patch = np.stack([smooth, smooth, smooth], axis=-1).astype(np.uint8)
        score = detector.score(smooth_patch)
        assert isinstance(score, float)


class TestFFTDCTDetector:
    def test_initialization(self):
        detector = FFTDCTDetector()
        assert detector.name == 'fft_dct'
    
    def test_score_returns_float(self, sample_rgb_patch):
        detector = FFTDCTDetector()
        score = detector.score(sample_rgb_patch)
        assert isinstance(score, float)
        assert score >= 0.0
    
    def test_score_with_small_patch(self):
        detector = FFTDCTDetector()
        tiny_patch = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        score = detector.score(tiny_patch)
        assert score == 0.0
    
    def test_score_with_large_patch(self):
        detector = FFTDCTDetector()
        large_patch = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        score = detector.score(large_patch)
        assert isinstance(score, float)


class TestResidualEnergyDetector:
    def test_initialization(self):
        detector = ResidualEnergyDetector()
        assert detector.name == 'residual_energy'
    
    def test_score_returns_float(self, sample_rgb_patch):
        detector = ResidualEnergyDetector()
        score = detector.score(sample_rgb_patch)
        assert isinstance(score, float)
        assert score > 0.0
    
    def test_score_with_small_patch(self):
        detector = ResidualEnergyDetector()
        tiny_patch = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        score = detector.score(tiny_patch)
        assert score == 0.0


class TestDCTCoocDetector:
    def test_initialization(self):
        detector = DCTCoocDetector()
        assert detector.name == 'dct_cooccurrence'
    
    def test_score_returns_float(self, sample_rgb_patch):
        detector = DCTCoocDetector()
        score = detector.score(sample_rgb_patch)
        assert isinstance(score, float)
        assert score >= 0.0
    
    def test_score_with_small_patch(self):
        detector = DCTCoocDetector()
        tiny_patch = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        score = detector.score(tiny_patch)
        assert score == 0.0


class TestJPEGGhostDetector:
    def test_initialization(self):
        detector = JPEGGhostDetector()
        assert detector.name == 'jpeg_ghost'

    def test_initialization_with_custom_quality(self):
        detector = JPEGGhostDetector(quality_levels=[60, 70, 80, 90])
        assert detector.quality_levels == [60, 70, 80, 90]

    def test_score_returns_float(self, sample_rgb_patch):
        detector = JPEGGhostDetector()
        score = detector.score(sample_rgb_patch)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_with_small_patch(self):
        detector = JPEGGhostDetector()
        tiny_patch = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        score = detector.score(tiny_patch)
        assert score == 0.0

    def test_score_with_invalid_dimensions(self):
        detector = JPEGGhostDetector()
        invalid_patch = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        score = detector.score(invalid_patch)
        assert score == 0.0


class TestGANFingerprintDetector:
    def test_initialization(self):
        detector = GANFingerprintDetector()
        assert detector.name == 'gan_fingerprint'

    def test_score_returns_float(self, sample_rgb_patch):
        detector = GANFingerprintDetector()
        score = detector.score(sample_rgb_patch)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_with_small_patch(self):
        detector = GANFingerprintDetector()
        tiny_patch = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        score = detector.score(tiny_patch)
        assert score == 0.0

    def test_score_with_invalid_dimensions(self):
        detector = GANFingerprintDetector()
        invalid_patch = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        score = detector.score(invalid_patch)
        assert score == 0.0

    def test_score_with_large_patch(self):
        detector = GANFingerprintDetector()
        large_patch = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        score = detector.score(large_patch)
        assert isinstance(score, float)


class TestNoiseConsistencyDetector:
    def test_initialization(self):
        detector = NoiseConsistencyDetector()
        assert detector.name == 'noise_consistency'

    def test_score_returns_float(self, sample_rgb_patch):
        detector = NoiseConsistencyDetector()
        score = detector.score(sample_rgb_patch)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_with_small_patch(self):
        detector = NoiseConsistencyDetector()
        tiny_patch = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        score = detector.score(tiny_patch)
        assert score == 0.0

    def test_score_with_invalid_dimensions(self):
        detector = NoiseConsistencyDetector()
        invalid_patch = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        score = detector.score(invalid_patch)
        assert score == 0.0

    def test_score_with_uniform_patch(self):
        """Uniform patches should be flagged as suspicious (too clean)."""
        detector = NoiseConsistencyDetector()
        uniform_patch = np.ones((64, 64, 3), dtype=np.uint8) * 128
        score = detector.score(uniform_patch)
        assert isinstance(score, float)
        # Uniform patches have no noise - should be suspicious
        assert score > 0.0

    def test_score_with_noisy_patch(self):
        """Naturally noisy patches."""
        detector = NoiseConsistencyDetector()
        noisy_patch = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        score = detector.score(noisy_patch)
        assert isinstance(score, float)
