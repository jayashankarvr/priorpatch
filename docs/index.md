# PriorPatch

Image forensics without the neural network nonsense.

## What is this?

Tool for detecting fake/edited images using signal processing instead of machine learning. Runs 6 different mathematical tests on image patches and shows you a heatmap of suspicious regions.

Why not ML? Because:

- You can actually understand what it's doing
- No GPU needed
- No training data needed
- Can't be easily fooled if attacker doesn't know what you're checking
- Easier to extend and debug

Trade-off: probably not as accurate as the latest deep learning models, but way more transparent.

## What It Does

PriorPatch analyzes images by:

1. Dividing them into overlapping patches
2. Scoring each patch with multiple forensic detectors
3. Combining scores using ensemble methods
4. Generating visual heatmaps showing suspicious regions

The toolkit includes 6 different detectors examining:

- Color channel correlations
- Spatial consistency
- Frequency domain characteristics
- High-frequency noise patterns
- JPEG compression artifacts
- Sensor noise (PRNU)

## Quick Example

**Command Line:**

```bash
priorpatch analyze --input photo.jpg --outdir results/
```

**Python API:**

```python
from priorpatch import Ensemble, load_image, save_heatmap

img = load_image('photo.jpg')
ensemble = Ensemble.from_config('config/detectors.json')
heatmap = ensemble.score_image(img)
save_heatmap(heatmap, img, 'result.png')
```

**Output:**

- Visual heatmap overlay (red = suspicious, blue = normal)
- JSON summary with scores and metadata

## Use cases

- Check if that viral image is actually real
- Learn how image forensics works
- Research detector techniques
- Build your own forensic tool
- Educational projects

Don't use this as your only verification method though. It's one tool, not a magic truth detector.

## Key Features

### Multiple Detection Methods

Each detector examines different forensic traces:

- **ColorStats**: RGB correlation analysis
- **NeighborConsistency**: Spatial prediction anomalies
- **FFT/DCT**: Frequency domain analysis
- **ResidualEnergy**: High-frequency noise patterns
- **DCTCooccurrence**: JPEG artifact analysis
- **PRNU**: Sensor noise detection

### Ensemble Approach

Combining multiple detectors:

- Reduces false positives
- Handles diverse manipulation types
- Provides confidence through redundancy
- Allows custom weighting

### Visual Heatmaps

Intuitive visualization:

- Color-coded suspicious regions
- Overlay on original image
- Customizable transparency and colormap
- Export as high-resolution PNG

### Configurable Pipeline

JSON-based configuration:

- Enable/disable detectors
- Set detector weights
- Adjust patch size and stride
- No code changes needed

## Getting Started

1. **[Installation](installation.md)** - Set up PriorPatch
2. **[Usage Guide](usage.md)** - Learn basic and advanced usage
3. **[Detector Details](detectors.md)** - Understand each detector
4. **[Architecture](architecture.md)** - Learn the system design
5. **[API Reference](api.md)** - Full API documentation

## Example Results

PriorPatch generates heatmaps like this:

```
Input: Potentially manipulated image
Output: Heatmap overlay showing:
  - Blue/green regions: Normal (low anomaly score)
  - Yellow/orange: Slightly suspicious
  - Red: Highly suspicious (high anomaly score)
```

The heatmap helps you visually identify:

- Spliced regions
- AI-generated content
- Retouched/airbrushed areas
- Copy-pasted objects

## Why I Made This

Mostly for learning. Wanted to understand how image forensics actually works without relying on black-box ML models. Also useful for research baselines and teaching.

## Limitations

PriorPatch is designed for research and education. Be aware:

- Not 100% accurate (no detector is)
- Can have false positives/negatives
- Performance varies by manipulation type
- Should be used with other verification methods
- Not a replacement for expert forensic analysis

## Links

- [Issues](https://github.com/jayashankarvr/priorpatch/issues) - Bug reports, feature requests
- [Contributing](https://github.com/jayashankarvr/priorpatch/blob/main/CONTRIBUTING.md) - Want to help?

## License

PriorPatch is licensed under the [Apache License 2.0](https://github.com/jayashankarvr/priorpatch/blob/main/LICENSE).

## Citation

If you use this in research:

```bibtex
@software{priorpatch2025,
  title={PriorPatch},
  author={Jayashankar VR},
  year={2025},
  url={https://github.com/jayashankarvr/priorpatch}
}
```
