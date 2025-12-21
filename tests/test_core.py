"""
Tests for core ensemble functionality.
"""

import pytest
import numpy as np
import json
from priorpatch.core import Ensemble
from priorpatch.detectors.color_stats import ColorStatsDetector
from priorpatch.detectors.neighbor_consistency import NeighborConsistencyDetector


class TestEnsemble:
    def test_initialization_with_detectors(self):
        detectors = [ColorStatsDetector(), NeighborConsistencyDetector()]
        ensemble = Ensemble(detectors)
        assert len(ensemble.detectors) == 2
        assert ensemble.weights == {}
    
    def test_initialization_with_weights(self):
        detectors = [ColorStatsDetector(), NeighborConsistencyDetector()]
        weights = {'color_stats': 2.0, 'neighbor_consistency': 0.5}
        ensemble = Ensemble(detectors, weights)
        assert ensemble.weights == weights
    
    def test_from_config(self, temp_config_file):
        ensemble = Ensemble.from_config(temp_config_file)
        assert len(ensemble.detectors) == 2
        assert 'color_stats' in ensemble.weights
    
    def test_from_config_missing_file(self):
        with pytest.raises(FileNotFoundError):
            Ensemble.from_config('nonexistent_config.json')
    
    def test_from_config_invalid_json(self, tmp_path):
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("{ invalid json }")
        with pytest.raises(json.JSONDecodeError):
            Ensemble.from_config(str(bad_config))
    
    def test_from_config_invalid_detector(self, tmp_path):
        config = {"enabled_detectors": ["nonexistent_detector"]}
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        with pytest.raises(ValueError, match="not found"):
            Ensemble.from_config(str(config_path))
    
    def test_score_patch(self, sample_rgb_patch):
        detectors = [ColorStatsDetector(), NeighborConsistencyDetector()]
        ensemble = Ensemble(detectors)
        result = ensemble.score_patch(sample_rgb_patch)

        assert isinstance(result.combined, float)
        assert isinstance(result.individual, list)
        assert len(result.individual) == 2
        assert all(isinstance(s, float) for s in result.individual)
    
    def test_score_patch_empty(self):
        detectors = [ColorStatsDetector()]
        ensemble = Ensemble(detectors)
        empty_patch = np.array([])
        result = ensemble.score_patch(empty_patch)
        assert result.combined == 0.0
        assert result.individual == []
    
    def test_score_patch_with_weights(self, sample_rgb_patch):
        detectors = [ColorStatsDetector(), NeighborConsistencyDetector()]
        weights = {'color_stats': 3.0, 'neighbor_consistency': 1.0}
        ensemble = Ensemble(detectors, weights)
        result = ensemble.score_patch(sample_rgb_patch)
        assert isinstance(result.combined, float)
    
    def test_score_image(self, sample_image):
        detectors = [ColorStatsDetector(), NeighborConsistencyDetector()]
        ensemble = Ensemble(detectors)
        heatmap = ensemble.score_image(sample_image, patch_size=32, stride=16)
        
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
    
    def test_score_image_small_image(self):
        detectors = [ColorStatsDetector()]
        ensemble = Ensemble(detectors)
        small_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        heatmap = ensemble.score_image(small_img, patch_size=64, stride=32)
        assert isinstance(heatmap, np.ndarray)
    
    def test_score_image_invalid_dimensions(self):
        detectors = [ColorStatsDetector()]
        ensemble = Ensemble(detectors)
        invalid_img = np.random.randint(0, 256, (256,), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            ensemble.score_image(invalid_img)
    
    def test_score_image_grayscale(self):
        detectors = [ColorStatsDetector()]
        ensemble = Ensemble(detectors)
        gray_img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        heatmap = ensemble.score_image(gray_img)
        assert isinstance(heatmap, np.ndarray)
    
    def test_score_image_different_strides(self, sample_image):
        detectors = [ColorStatsDetector()]
        ensemble = Ensemble(detectors)
        
        heatmap1 = ensemble.score_image(sample_image, patch_size=64, stride=32)
        heatmap2 = ensemble.score_image(sample_image, patch_size=64, stride=64)
        
        assert heatmap1.shape != heatmap2.shape
        assert heatmap2.size < heatmap1.size  # Larger stride = fewer patches
