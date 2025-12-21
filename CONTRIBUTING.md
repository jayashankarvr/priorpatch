# Contributing

Thanks for considering contributing.! Here's how to get started.

## Code of conduct

Don't be a jerk. Be helpful. That's basically it.

## Setup

```bash
# Fork on GitHub, then:
git clone https://github.com/jayashankarvr/priorpatch.git
cd priorpatch
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest  # make sure tests pass
```

## What to contribute

### Easy wins

- Fix typos in docs
- Add examples
- Improve error messages
- Add tests

### Medium

- Add a new detector
- Improve existing detector
- Performance improvements
- Better visualizations

### Hard

- Proper PRNU implementation
- New analysis methods
- Web UI
- Video support

## Adding a detector

This is probably the most useful contribution. Here's how:

1. Create `priorpatch/detectors/your_detector.py`:

```python
import numpy as np
from .base import DetectorInterface
from .registry import register_detector

@register_detector
class YourDetector(DetectorInterface):
    """
    Brief explanation of what this detects and how.
    """
    name = 'your_detector'
    
    def score(self, patch: np.ndarray) -> float:
        """
        Args:
            patch: RGB image patch (H, W, 3)
        
        Returns:
            float: Higher = more suspicious
        """
        # Handle edge cases
        if patch.size == 0 or patch.shape[0] < 8:
            return 0.0
        
        # Your detection logic
        score = your_algorithm(patch)
        
        return float(score)
```

1. Add to `config/detectors.json`:

```json
{
  "enabled_detectors": [..., "your_detector"],
  "detector_weights": {"your_detector": 1.0}
}
```

1. Write tests in `tests/test_detectors.py`:

```python
class TestYourDetector:
    def test_returns_float(self, sample_rgb_patch):
        detector = YourDetector()
        score = detector.score(sample_rgb_patch)
        assert isinstance(score, float)
        assert score >= 0.0
    
    def test_handles_edge_cases(self):
        detector = YourDetector()
        assert detector.score(np.array([])) == 0.0
```

1. Document it in `docs/detectors.md`

That's it. No need to modify core code.

## Coding style

- Follow PEP 8 (ish - don't stress about line length)
- Use descriptive names
- Add docstrings for public functions
- Type hints are nice but not required
- Comments for non-obvious stuff

Run these before committing:

```bash
black priorpatch/ tests/        # auto-format
pytest                           # make sure tests pass
```

## Pull requests

1. Create a branch: `git checkout -b feature-name`
2. Make changes
3. Test locally
4. Commit with clear message
5. Push and open PR

PR template will guide you through the rest.

### Good commit messages

Good:

```
Add ELA detector for JPEG analysis

Implements error level analysis to detect different
compression levels in the same image.
```

Bad:

```
fixed stuff
```

```
Updated files
```

## Testing

We use pytest. Tests should be fast and independent.

```bash
pytest                    # run all
pytest tests/test_core.py # run specific file
pytest -v                 # verbose
pytest --cov=priorpatch   # with coverage
```

Write tests for:

- Normal cases
- Edge cases (empty input, tiny patches, etc.)
- Error handling

Don't need 100% coverage, but test the important paths.

## Documentation

- Update docs if you change behavior
- Add examples for new features
- Keep it simple and practical
- Code examples should actually work

## Questions?

- Open an issue
- Check existing issues first
- Be specific about what you're trying to do

## What we're NOT looking for

- Major architectural changes (discuss first)
- Dependencies on huge libraries
- Breaking changes to the API
- Machine learning models (defeats the purpose)
- Stuff that only works on one platform

## Future Ideas

Want to work on something bigger? Here are some ideas:

### Already Implemented

- [x] PRNU implementation (wavelet-based) - `prnu_wavelet.py`
- [x] ELA, lighting consistency, copy-move detectors
- [x] GPU acceleration via CuPy
- [x] Benchmarking infrastructure (`benchmarks/dataset_benchmark.py`)
- [x] Auto-discovery of detectors

### Short term

- Web UI (probably Gradio or Streamlit)
- Run benchmarks on CASIA/Columbia and tune weights
- Automatic threshold tuning

### Medium term

- Video support (frame-by-frame analysis)
- Export results to different formats
- Real-time analysis

### Long term

- Browser extension
- Mobile app
- Integration with fact-checking tools

These are just ideas. Feel free to propose your own or tackle these however you want.

## Release process

Maintainers handle releases. You don't need to worry about this.

## License

By contributing, you agree your code is licensed under Apache 2.0.

## Thanks

Every contribution helps, even small ones. Appreciate you taking the time.
