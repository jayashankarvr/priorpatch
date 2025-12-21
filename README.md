# PriorPatch

[![CI](https://github.com/jayashankarvr/priorpatch/actions/workflows/test.yml/badge.svg)](https://github.com/jayashankarvr/priorpatch/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/jayashankarvr/priorpatch/branch/main/graph/badge.svg)](https://codecov.io/gh/jayashankarvr/priorpatch)
[![PyPI version](https://badge.fury.io/py/priorpatch.svg)](https://badge.fury.io/py/priorpatch)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Detect fake/manipulated images using math instead of neural networks.

No ML models, just signal processing and stats. GPUs optional but not required. You can actually understand what it's doing.

## What does it do?

Takes an image, splits it into patches, runs forensic tests on each patch, and gives you a heatmap showing suspicious areas. Red = weird, blue = normal. Includes 16 detectors (all enabled by default).

The detectors check for:

- Unnatural color relationships
- Weird pixel-neighbor patterns
- Frequency domain anomalies (resampling, compression)
- Missing high-frequency noise
- JPEG compression inconsistencies
- JPEG recompression artifacts (spliced images)
- GAN/AI upsampling artifacts (checkerboard patterns)
- Unnatural noise patterns (AI-generated content)
- **Benford's Law violations** in DCT coefficients (~90% F1 in research)
- **Missing CFA/Bayer demosaicing artifacts** (AI doesn't use real cameras)
- **LBP texture anomalies** (83-99% accuracy on deepfakes)
- **Chromatic aberration inconsistencies** (AI doesn't simulate real lens physics)
- **Sensor noise (PRNU) absence** using wavelet denoising
- **Error Level Analysis (ELA)** for JPEG manipulation detection
- **Lighting consistency** analysis for composite detection
- **Copy-move forgery** detection using DCT block matching

You can see both individual detector results and the combined ensemble score.

## Install

```bash
git clone https://github.com/jayashankarvr/priorpatch.git
cd priorpatch
pip install -r requirements.txt
pip install -e .
```

### GPU Acceleration (Optional)

For faster processing on NVIDIA GPUs, install CuPy:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# Or for CUDA 11.x
pip install cupy-cuda11x
```

PriorPatch automatically uses GPU when available. No code changes needed.

## Quick start

Command line:

```bash
# Single image
priorpatch analyze --input photo.jpg --outdir results/

# Entire directory (recursive)
priorpatch analyze --input-dir photos/ --outdir results/

# Glob pattern
priorpatch analyze --input "photos/*.jpg" --outdir results/

# With individual detector heatmaps
priorpatch analyze --input photo.jpg --outdir results/ --save-individual

# Force CPU-only (disable GPU)
priorpatch analyze --input photo.jpg --outdir results/ --no-gpu
```

Python:

```python
from priorpatch import Ensemble, load_image, save_heatmap

img = load_image('photo.jpg')
ensemble = Ensemble.from_config('config/detectors.json')
heatmap = ensemble.score_image(img)
save_heatmap(heatmap, img, 'result.png')
```

Get individual detector results:

```python
results = ensemble.score_image(img, return_individual=True)

combined = results.combined      # overall score
individual = results.individual  # per-detector heatmaps

# see what each detector found
for name in results.detector_names:
    score = individual[name].max()
    print(f"{name}: {score:.3f}")
```

GPU control in Python:

```python
import priorpatch

# Check GPU status
print(priorpatch.get_gpu_info())
# {'available': True, 'active': True, 'device_name': 'NVIDIA RTX 3050', ...}

# Disable GPU (use CPU only)
priorpatch.disable_gpu()

# Re-enable GPU
priorpatch.enable_gpu()
```

Or via environment variable: `PRIORPATCH_NO_GPU=1`

## Configuration

Edit `config/detectors.json` to choose detectors and set weights:

```json
{
  "enabled_detectors": ["color_stats", "fft_dct", "neighbor_consistency"],
  "detector_weights": {
    "fft_dct": 2.0,
    "color_stats": 1.0,
    "neighbor_consistency": 0.5
  }
}
```

### Environment Variables

Override config without editing files:

```bash
# Custom config path
export PRIORPATCH_CONFIG_PATH=/path/to/custom/config.json

# Override enabled detectors (comma-separated)
export PRIORPATCH_ENABLED_DETECTORS=color_stats,fft_dct,benford_law

# Override weights (JSON format)
export PRIORPATCH_WEIGHTS='{"color_stats": 1.5, "fft_dct": 2.0}'

# Disable GPU acceleration
export PRIORPATCH_NO_GPU=1
```

## Examples

Check `examples/` folder:

- `basic_usage.py` - simple analysis
- `custom_detectors.py` - manual detector selection
- `batch_processing.py` - process multiple images
- `individual_detector_analysis.py` - compare detector outputs
- `performance_comparison.py` - benchmark speed

## Documentation

- `docs/installation.md` - setup help
- `docs/usage.md` - examples and tips
- `docs/detectors.md` - how each detector works
- `docs/architecture.md` - system design
- `docs/api.md` - function reference

Or: `mkdocs serve` then visit <http://localhost:8000>

## Tests

```bash
pytest                         # run tests
pytest --cov=priorpatch        # with coverage
```

## Adding a detector

Create a file in `priorpatch/detectors/`:

```python
from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
import numpy as np

@register_detector
class MyDetector(DetectorInterface):
    name = 'my_detector'

    def score(self, patch: np.ndarray) -> float:
        # your logic here
        # return higher score for suspicious patches
        return score
```

That's it. The registry picks it up automatically.

## How it works

1. Split image into overlapping patches
2. Run all detectors on each patch
3. Min-max normalize the results to [0,1]
4. Combine using weights
5. Generate heatmap
6. Overlay on original image

Standard ensemble approach, nothing fancy.

## What's good/bad

**Works well for:**

- Spliced images (copy-paste)
- Resized/resampled images
- Over-smoothed edits
- JPEG compression mismatches
- Some AI-generated content

**Struggles with:**

- High-quality fakes
- Images that are naturally weird
- Heavy post-processing on legitimate images
- Advanced adversarial examples

**Current limitations:**

- Detector weights are initial estimates, not tuned on benchmarks
- Not benchmarked on standard datasets yet (CASIA, Columbia, etc.)
- No deep learning features (by design, but means we miss some sophisticated AI)
- Scoring may produce false positives on heavily post-processed legitimate images

## Roadmap

Stuff that would be nice:

- [ ] Web UI (probably Gradio)
- [ ] Video support
- [ ] Benchmark on CASIA/Columbia datasets and tune weights (infrastructure ready in `benchmarks/`)

**Already implemented:**

- [x] GPU acceleration via CuPy (optional backend for FFT operations)
- [x] Auto-discovery of detectors (just drop a file in `priorpatch/detectors/`)
- [x] ELA, lighting consistency, copy-move detectors
- [x] Benchmarking infrastructure for standard datasets

## Why "math-only"?

Neural networks work great but:

- Black box - can't explain decisions
- Need training data
- Need GPUs
- Can be fooled by adversarial attacks
- Hard to debug

Math-based methods:

- Transparent - you can see what's being measured
- No training needed
- Run anywhere
- Harder to fool (if attacker doesn't know what you're checking)
- Easier to extend

Trade-off: probably lower accuracy than SOTA deep learning, but you actually know what you're getting.

## License

Apache 2.0 - do what you want with it.

## Contributing

PRs welcome. Check `CONTRIBUTING.md` for guidelines.

For bugs/features: use our [issue templates](.github/ISSUE_TEMPLATE/) on GitHub.

## Credits

Uses numpy, scipy, PIL, matplotlib. Detector algorithms based on various forensics papers. See `docs/detectors.md` for references.

## Disclaimer

This is a research tool. Don't use it as the only evidence for anything important. False positives happen. False negatives happen. Use multiple methods, use your brain, don't trust any single tool.

The GAN fingerprint and noise consistency detectors help catch AI-generated content, but the latest models (DALL-E 3, Midjourney v5+, etc.) are constantly improving. This tool works best for traditional edits, splices, and older AI generation techniques.
