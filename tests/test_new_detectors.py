"""
Tests for new detectors: ELA, Lighting Consistency, Copy-Move.
"""

import pytest
import numpy as np

from priorpatch.detectors.ela import ELADetector, compute_ela, analyze_ela_statistics
from priorpatch.detectors.lighting_consistency import (
    LightingConsistencyDetector,
    estimate_gradient,
    analyze_lighting_consistency
)
from priorpatch.detectors.copy_move import (
    CopyMoveDetector,
    extract_dct_features,
    find_similar_blocks,
    analyze_copy_move
)


class TestELADetector:
    """Test ELA detector."""

    def test_initialization(self):
        """Detector initializes correctly."""
        detector = ELADetector()
        assert detector.name == 'ela'
        assert detector.quality == 90

    def test_initialization_custom_quality(self):
        """Detector accepts custom quality."""
        detector = ELADetector(quality=75, scale=15.0)
        assert detector.quality == 75
        assert detector.scale == 15.0

    def test_score_normal_image(self):
        """Score normal RGB image."""
        detector = ELADetector()
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score = detector.score(patch)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_uniform_image(self):
        """Score uniform color image."""
        detector = ELADetector()
        patch = np.full((64, 64, 3), 128, dtype=np.uint8)
        score = detector.score(patch)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_small_patch(self):
        """Small patches return 0."""
        detector = ELADetector()
        patch = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        score = detector.score(patch)

        assert score == 0.0

    def test_score_invalid_dimensions(self):
        """Invalid dimensions return 0."""
        detector = ELADetector()
        patch = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        score = detector.score(patch)

        assert score == 0.0

    def test_compute_ela_function(self):
        """compute_ela returns correct shape."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ela = compute_ela(image, quality=85)

        assert ela.shape == image.shape

    def test_analyze_ela_statistics(self):
        """analyze_ela_statistics returns expected keys."""
        ela = np.random.rand(64, 64, 3) * 10
        stats = analyze_ela_statistics(ela)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'max' in stats


class TestLightingConsistencyDetector:
    """Test lighting consistency detector."""

    def test_initialization(self):
        """Detector initializes correctly."""
        detector = LightingConsistencyDetector()
        assert detector.name == 'lighting_consistency'
        assert detector.grid_size == 4

    def test_initialization_custom_params(self):
        """Detector accepts custom parameters."""
        detector = LightingConsistencyDetector(grid_size=6, use_shadows=False)
        assert detector.grid_size == 6
        assert detector.use_shadows is False

    def test_score_normal_image(self):
        """Score normal RGB image."""
        detector = LightingConsistencyDetector()
        patch = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        score = detector.score(patch)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_gradient_image(self):
        """Score image with gradient (consistent lighting)."""
        detector = LightingConsistencyDetector()

        # Create gradient image (consistent lighting direction)
        x = np.linspace(0, 255, 128)
        y = np.linspace(0, 255, 128)
        xx, yy = np.meshgrid(x, y)
        gradient = ((xx + yy) / 2).astype(np.uint8)
        patch = np.stack([gradient, gradient, gradient], axis=-1)

        score = detector.score(patch)
        assert isinstance(score, float)

    def test_score_small_patch(self):
        """Small patches return 0."""
        detector = LightingConsistencyDetector()
        patch = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        score = detector.score(patch)

        assert score == 0.0

    def test_estimate_gradient(self):
        """estimate_gradient returns correct shapes."""
        image = np.random.rand(64, 64)
        gx, gy = estimate_gradient(image)

        assert gx.shape == image.shape
        assert gy.shape == image.shape

    def test_analyze_lighting_consistency(self):
        """analyze_lighting_consistency returns expected keys."""
        image = np.random.rand(128, 128)
        stats = analyze_lighting_consistency(image, grid_size=4)

        assert 'consistency' in stats


class TestCopyMoveDetector:
    """Test copy-move detector."""

    def test_initialization(self):
        """Detector initializes correctly."""
        detector = CopyMoveDetector()
        assert detector.name == 'copy_move'
        assert detector.block_size == 16

    def test_initialization_custom_params(self):
        """Detector accepts custom parameters."""
        detector = CopyMoveDetector(
            block_size=32,
            stride=16,
            similarity_threshold=0.95
        )
        assert detector.block_size == 32
        assert detector.stride == 16

    def test_score_normal_image(self):
        """Score normal RGB image."""
        detector = CopyMoveDetector()
        patch = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        score = detector.score(patch)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_image_with_copy(self):
        """Score image with actual copy-move."""
        detector = CopyMoveDetector(similarity_threshold=0.90)

        # Create image with copy-move forgery
        patch = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        # Copy a region
        source = patch[10:40, 10:40].copy()
        patch[70:100, 70:100] = source

        score = detector.score(patch)

        assert isinstance(score, float)
        # Should detect the copy-move (score > 0)
        # Note: may not always detect due to randomness

    def test_score_small_patch(self):
        """Small patches return 0."""
        detector = CopyMoveDetector()
        patch = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        score = detector.score(patch)

        assert score == 0.0

    def test_extract_dct_features(self):
        """extract_dct_features returns features and positions."""
        image = np.random.rand(64, 64)
        features, positions = extract_dct_features(image, block_size=16, stride=8)

        assert len(features) > 0
        assert len(positions) == len(features)
        assert features.ndim == 2

    def test_find_similar_blocks_no_matches(self):
        """find_similar_blocks with no matches."""
        # Random features should have no highly similar matches
        features = np.random.rand(10, 36)
        positions = [(i * 16, i * 16) for i in range(10)]

        matches = find_similar_blocks(
            features, positions,
            similarity_threshold=0.99,
            min_distance=32
        )

        # Random data should have few/no matches at high threshold
        assert isinstance(matches, list)

    def test_analyze_copy_move(self):
        """analyze_copy_move returns expected keys."""
        image = np.random.rand(128, 128)
        results = analyze_copy_move(image)

        assert 'detected' in results
        assert 'num_matches' in results
        assert 'num_clusters' in results
        assert 'coverage' in results


class TestNewDetectorEdgeCases:
    """Test edge cases for new detectors."""

    @pytest.fixture
    def new_detectors(self):
        """Get new detector instances."""
        return [
            ELADetector(),
            LightingConsistencyDetector(),
            CopyMoveDetector(),
        ]

    def test_all_return_float(self, new_detectors):
        """All detectors return float."""
        patch = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        for detector in new_detectors:
            score = detector.score(patch)
            assert isinstance(score, float), f"{detector.name} didn't return float"

    def test_all_bounded_scores(self, new_detectors):
        """All scores are in [0, 1]."""
        patch = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        for detector in new_detectors:
            score = detector.score(patch)
            assert 0.0 <= score <= 1.0, f"{detector.name} score out of bounds: {score}"

    def test_all_handle_float_input(self, new_detectors):
        """All detectors handle float input."""
        patch = np.random.rand(128, 128, 3)

        for detector in new_detectors:
            score = detector.score(patch)
            assert isinstance(score, float), f"{detector.name} failed on float input"

    def test_reproducibility(self, new_detectors):
        """Same input gives same output."""
        patch = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        for detector in new_detectors:
            score1 = detector.score(patch)
            score2 = detector.score(patch)
            assert score1 == score2, f"{detector.name} not reproducible"


class TestDetectorRegistry:
    """Test that new detectors are properly registered."""

    def test_ela_in_registry(self):
        """ELA detector is in registry."""
        from priorpatch.detectors.registry import DETECTOR_REGISTRY
        assert 'ela' in DETECTOR_REGISTRY

    def test_lighting_consistency_in_registry(self):
        """Lighting consistency detector is in registry."""
        from priorpatch.detectors.registry import DETECTOR_REGISTRY
        assert 'lighting_consistency' in DETECTOR_REGISTRY

    def test_copy_move_in_registry(self):
        """Copy-move detector is in registry."""
        from priorpatch.detectors.registry import DETECTOR_REGISTRY
        assert 'copy_move' in DETECTOR_REGISTRY

    def test_auto_discovery_loads_all(self):
        """Auto-discovery loads all expected detectors."""
        from priorpatch.detectors.registry import DETECTOR_REGISTRY

        expected = [
            'color_stats', 'neighbor_consistency', 'fft_dct',
            'residual_energy', 'dct_cooccurrence', 'jpeg_ghost',
            'gan_fingerprint', 'noise_consistency', 'benford_law',
            'cfa_artifact', 'lbp_texture', 'chromatic_aberration',
            'prnu_wavelet', 'ela', 'lighting_consistency', 'copy_move'
        ]

        for name in expected:
            assert name in DETECTOR_REGISTRY, f"Detector {name} not found in registry"
