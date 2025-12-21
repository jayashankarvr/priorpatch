"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_rgb_patch():
    """Create a sample RGB image patch for testing."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    np.random.seed(42)
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def small_patch():
    """Create a small patch for edge case testing."""
    return np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)


@pytest.fixture
def grayscale_patch():
    """Create a grayscale patch."""
    return np.random.randint(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file."""
    import json
    config = {
        "enabled_detectors": ["color_stats", "neighbor_consistency"],
        "detector_weights": {
            "color_stats": 1.0,
            "neighbor_consistency": 1.5
        }
    }
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return str(config_path)


@pytest.fixture
def temp_image_file(tmp_path, sample_image):
    """Create a temporary image file."""
    from PIL import Image
    img_path = tmp_path / "test_image.png"
    Image.fromarray(sample_image).save(img_path)
    return str(img_path)
