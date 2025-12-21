"""
Tests with real (synthetic) images.
"""

import pytest
import os
from pathlib import Path
from priorpatch import Ensemble, load_image


class TestRealImages:
    """Test with fixture images."""
    
    @pytest.fixture
    def ensemble(self):
        return Ensemble.from_config('config/detectors.json')
    
    def test_authentic_image(self, ensemble):
        """Test with authentic-looking image."""
        img_path = 'tests/fixtures/authentic.png'
        if not os.path.exists(img_path):
            pytest.skip("Fixture image not found")
        
        img = load_image(img_path)
        result = ensemble.score_image(img)
        
        assert result.ndim == 2
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_manipulated_image(self, ensemble):
        """Test with manipulated-looking image."""
        img_path = 'tests/fixtures/manipulated.png'
        if not os.path.exists(img_path):
            pytest.skip("Fixture image not found")
        
        img = load_image(img_path)
        result = ensemble.score_image(img)
        
        assert result.ndim == 2
        # Manipulated image should have higher scores
        assert result.max() > 0.0
    
    def test_noisy_image(self, ensemble):
        """Test with noisy image."""
        img_path = 'tests/fixtures/noisy.png'
        if not os.path.exists(img_path):
            pytest.skip("Fixture image not found")
        
        img = load_image(img_path)
        result = ensemble.score_image(img)
        
        assert result.ndim == 2
    
    def test_smooth_image(self, ensemble):
        """Test with smooth image."""
        img_path = 'tests/fixtures/smooth.png'
        if not os.path.exists(img_path):
            pytest.skip("Fixture image not found")
        
        img = load_image(img_path)
        result = ensemble.score_image(img)
        
        assert result.ndim == 2
    
    def test_all_fixtures_exist(self):
        """Verify all fixture images exist."""
        fixtures = ['authentic.png', 'manipulated.png', 'noisy.png', 'smooth.png']
        fixtures_dir = Path('tests/fixtures')
        
        if not fixtures_dir.exists():
            pytest.skip("Fixtures directory not found")
        
        for fixture in fixtures:
            path = fixtures_dir / fixture
            assert path.exists(), f"Missing fixture: {fixture}"
    
    def test_individual_results_on_real_image(self, ensemble):
        """Test individual detector results on real image."""
        img_path = 'tests/fixtures/manipulated.png'
        if not os.path.exists(img_path):
            pytest.skip("Fixture image not found")
        
        img = load_image(img_path)
        result = ensemble.score_image(img, return_individual=True)
        
        assert hasattr(result, 'combined')
        assert hasattr(result, 'individual')
        assert len(result.individual) == len(ensemble.detectors)
