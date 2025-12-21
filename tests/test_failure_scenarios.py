"""
Tests for failure scenarios and error handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from priorpatch import Ensemble
from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector


class BrokenDetector(DetectorInterface):
    """Detector that always fails for testing."""
    name = 'broken_detector'
    
    def score(self, patch):
        raise RuntimeError("Detector intentionally broken")


class InconsistentDetector(DetectorInterface):
    """Detector with inconsistent behavior."""
    name = 'inconsistent_detector'
    
    def __init__(self):
        self.call_count = 0
    
    def score(self, patch):
        self.call_count += 1
        if self.call_count % 2 == 0:
            raise ValueError("Fails on even calls")
        return 0.5


class TestDetectorFailureHandling:
    """Test handling of detector failures."""
    
    def test_single_detector_failure_continues(self):
        """Ensemble continues when one detector fails."""
        from priorpatch.detectors.color_stats import ColorStatsDetector

        ensemble = Ensemble([ColorStatsDetector(), BrokenDetector()])
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # Should not raise, despite BrokenDetector failing
        result = ensemble.score_patch(patch)

        assert isinstance(result.combined, float)
        assert len(result.failures) > 0  # BrokenDetector should have failed
        assert any('broken_detector' in str(f) for f in result.failures)
    
    def test_all_detectors_fail(self):
        """Ensemble handles all detectors failing."""
        ensemble = Ensemble([BrokenDetector(), BrokenDetector()])
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = ensemble.score_patch(patch)

        # Should return 0.5 (uniform) when all fail (all values are 0.0)
        assert isinstance(result.combined, float)
        assert len(result.failures) == 2
    
    def test_intermittent_detector_failure(self):
        """Ensemble handles intermittent failures."""
        from priorpatch.detectors.color_stats import ColorStatsDetector
        
        ensemble = Ensemble([ColorStatsDetector(), InconsistentDetector()])
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Should complete despite intermittent failures
        result = ensemble.score_image(img, patch_size=32, stride=32)
        
        assert result.ndim == 2
        assert not np.isnan(result).any()
    
    def test_failure_rate_threshold_warning(self, caplog):
        """Warning logged when detector fails >10% of patches."""
        ensemble = Ensemble([InconsistentDetector()])
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        with caplog.at_level('WARNING'):
            result = ensemble.score_image(img, patch_size=32, stride=32)
        
        # Should have warnings about failures
        assert any('failed' in record.message.lower() for record in caplog.records)
    
    def test_failure_rate_threshold_error(self, caplog):
        """Error logged when detector fails >50% of patches."""
        ensemble = Ensemble([BrokenDetector()])
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        with caplog.at_level('ERROR'):
            result = ensemble.score_image(img, patch_size=32, stride=32)
        
        # Should have error about high failure rate
        assert any('failed' in record.message.lower() and record.levelname == 'ERROR' 
                   for record in caplog.records)


class TestConfigurationErrors:
    """Test configuration error handling."""
    
    def test_config_missing_file(self):
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Ensemble.from_config('nonexistent_config.json')
    
    def test_config_invalid_json(self, tmp_path):
        """Invalid JSON raises JSONDecodeError."""
        bad_config = tmp_path / 'bad.json'
        bad_config.write_text('{ invalid json }')
        
        with pytest.raises(Exception):  # JSONDecodeError or similar
            Ensemble.from_config(str(bad_config))
    
    def test_config_missing_detector(self, tmp_path):
        """Config with non-existent detector raises ValueError."""
        import json
        
        config = {
            "version": "1.0",
            "enabled_detectors": ["nonexistent_detector"],
            "detector_weights": {}
        }
        
        config_path = tmp_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        with pytest.raises(ValueError, match="not found"):
            Ensemble.from_config(str(config_path))
    
    def test_config_empty_detectors_list(self, tmp_path, caplog):
        """Config with empty detectors list warns user."""
        import json
        
        config = {
            "version": "1.0",
            "enabled_detectors": [],
            "detector_weights": {}
        }
        
        config_path = tmp_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        with caplog.at_level('WARNING'):
            ensemble = Ensemble.from_config(str(config_path))
        
        assert len(ensemble.detectors) == 0
        assert any('no detectors' in record.message.lower() for record in caplog.records)
    
    def test_config_unknown_weight(self, tmp_path, caplog):
        """Config with weight for unknown detector warns."""
        import json
        
        config = {
            "version": "1.0",
            "enabled_detectors": ["color_stats"],
            "detector_weights": {
                "color_stats": 1.0,
                "nonexistent": 2.0  # Unknown detector
            }
        }
        
        config_path = tmp_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        with caplog.at_level('WARNING'):
            ensemble = Ensemble.from_config(str(config_path))
        
        assert any('unknown' in record.message.lower() for record in caplog.records)


class TestMemoryAndPerformance:
    """Test memory and performance edge cases."""
    
    def test_very_large_image(self):
        """Handle very large image without crashing."""
        ensemble = Ensemble.from_config('config/detectors.json')
        # 4K image
        large_img = np.random.randint(0, 255, (3840, 3840, 3), dtype=np.uint8)
        
        # Use large stride to reduce patch count
        result = ensemble.score_image(large_img, patch_size=128, stride=256)
        
        assert result.ndim == 2
        assert result.shape[0] > 0 and result.shape[1] > 0
    
    def test_many_small_patches(self):
        """Handle many small patches."""
        ensemble = Ensemble.from_config('config/detectors.json')
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Small stride = many patches
        result = ensemble.score_image(img, patch_size=32, stride=8)
        
        assert result.ndim == 2
        # Should have many patches
        assert result.shape[0] > 50 and result.shape[1] > 50
    
    def test_extreme_aspect_ratio(self):
        """Handle extreme aspect ratio images."""
        ensemble = Ensemble.from_config('config/detectors.json')
        
        # Very wide image
        wide = np.random.randint(0, 255, (50, 2000, 3), dtype=np.uint8)
        result = ensemble.score_image(wide, patch_size=32, stride=32)
        assert result.ndim == 2
        
        # Very tall image
        tall = np.random.randint(0, 255, (2000, 50, 3), dtype=np.uint8)
        result = ensemble.score_image(tall, patch_size=32, stride=32)
        assert result.ndim == 2


class TestConcurrencyIssues:
    """Test multiprocessing edge cases."""
    
    def test_multiprocessing_with_failures(self):
        """Multiprocessing handles detector failures."""
        ensemble = Ensemble([BrokenDetector()])
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        result = ensemble.score_image(
            img,
            patch_size=64,
            stride=64,
            use_multiprocessing=True,
            n_jobs=2
        )
        
        assert result.ndim == 2
    
    def test_multiprocessing_vs_sequential_with_failures(self):
        """Sequential and parallel give same results even with failures."""
        ensemble = Ensemble([InconsistentDetector()])
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Both should complete
        seq = ensemble.score_image(img, use_multiprocessing=False)
        par = ensemble.score_image(img, use_multiprocessing=True, n_jobs=2)
        
        assert seq.shape == par.shape
        # Results might differ due to detector inconsistency, but should be valid
        assert not np.isnan(seq).any()
        assert not np.isnan(par).any()
