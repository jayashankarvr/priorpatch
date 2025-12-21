"""
Tests for utility functions.
"""

import pytest
import numpy as np
from pathlib import Path
from priorpatch.utils import load_image, save_heatmap, validate_path


class TestLoadImage:
    def test_load_image_success(self, temp_image_file):
        img = load_image(temp_image_file)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
    
    def test_load_image_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_image('nonexistent_image.png')
    
    def test_load_image_invalid_format(self, tmp_path):
        bad_file = tmp_path / "not_an_image.txt"
        bad_file.write_text("This is not an image")
        with pytest.raises(IOError):
            load_image(str(bad_file))


class TestSaveHeatmap:
    def test_save_heatmap_success(self, tmp_path, sample_image):
        heatmap = np.random.rand(10, 10).astype(np.float32)
        outpath = tmp_path / "heatmap.png"
        
        save_heatmap(heatmap, sample_image, str(outpath))
        assert outpath.exists()
    
    def test_save_heatmap_invalid_heatmap_dimensions(self, sample_image, tmp_path):
        invalid_heatmap = np.random.rand(10, 10, 3)  # Should be 2D
        outpath = tmp_path / "heatmap.png"
        
        with pytest.raises(ValueError, match="must be 2D"):
            save_heatmap(invalid_heatmap, sample_image, str(outpath))
    
    def test_save_heatmap_invalid_image_dimensions(self, tmp_path):
        heatmap = np.random.rand(10, 10).astype(np.float32)
        invalid_image = np.random.randint(0, 256, (256,), dtype=np.uint8)
        outpath = tmp_path / "heatmap.png"
        
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            save_heatmap(heatmap, invalid_image, str(outpath))
    
    def test_save_heatmap_custom_params(self, tmp_path, sample_image):
        heatmap = np.random.rand(10, 10).astype(np.float32)
        outpath = tmp_path / "heatmap.png"
        
        save_heatmap(heatmap, sample_image, str(outpath), alpha=0.7, cmap='hot')
        assert outpath.exists()
    
    def test_save_heatmap_creates_directory(self, tmp_path, sample_image):
        heatmap = np.random.rand(10, 10).astype(np.float32)
        outpath = tmp_path / "subdir" / "heatmap.png"
        
        save_heatmap(heatmap, sample_image, str(outpath))
        assert outpath.exists()


class TestValidatePath:
    def test_validate_path_success(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        validated = validate_path(str(test_file), must_exist=True)
        assert isinstance(validated, Path)
        assert validated.exists()
    
    def test_validate_path_nonexistent_without_check(self):
        validated = validate_path('nonexistent.txt', must_exist=False)
        assert isinstance(validated, Path)
    
    def test_validate_path_nonexistent_with_check(self):
        with pytest.raises(FileNotFoundError):
            validate_path('nonexistent.txt', must_exist=True)
    
    def test_validate_path_traversal_warning(self):
        # Should not raise, but logs warning
        validated = validate_path('../etc/passwd', must_exist=False)
        assert isinstance(validated, Path)
