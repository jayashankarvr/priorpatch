# Usage

## Command Line Interface

### Basic Analysis

```bash
priorpatch analyze --input examples/sample_input.png --outdir outputs
```

### With Individual Detector Results

```bash
priorpatch analyze --input image.jpg --outdir outputs --save-individual
```

This will create:

- `outputs/heatmap_combined.png` - Ensemble result
- `outputs/individual_detectors/color_stats_heatmap.png` - Color detector only
- `outputs/individual_detectors/fft_dct_heatmap.png` - Frequency detector only
- ... (one for each detector)
- `outputs/summary.json` - Scores from all detectors

### Batch Processing (Directory)

```bash
# Analyze all images in a directory (recursive)
priorpatch analyze --input-dir photos/ --outdir results/

# Using glob pattern
priorpatch analyze --input "photos/*.jpg" --outdir results/

# With individual detector heatmaps for each image
priorpatch analyze --input-dir photos/ --outdir results/ --save-individual
```

This will:

- Create a subfolder per image in the output directory
- Generate `batch_summary.json` with all results sorted by suspicion score
- Show the top 5 most suspicious images after processing

### Custom Parameters

```bash
priorpatch analyze \
  --input image.jpg \
  --outdir results/ \
  --patch_size 32 \
  --stride 16 \
  --config custom_config.json \
  --save-individual \
  --log-level DEBUG
```

**Available options:**

- `--input`: Single image file or glob pattern (e.g., `"photos/*.jpg"`)
- `--input-dir`: Directory containing images (recursive)
- `--patch_size`: Size of analysis patches (8-1024, default: 64)
- `--stride`: Stride for patch extraction (minimum 1, default: 32)
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## Python API

### Basic Usage

```python
from priorpatch.core import Ensemble
from priorpatch.utils import load_image, save_heatmap

# Load image
img = load_image('examples/sample_input.png')

# Create ensemble
ens = Ensemble.from_config('config/detectors.json')

# Analyze image
heatmap = ens.score_image(img, patch_size=64, stride=32)

# Save result
save_heatmap(heatmap, img, 'output.png')
```

### With Individual Detector Results

```python
from priorpatch.core import Ensemble
from priorpatch.utils import load_image, save_heatmap

img = load_image('image.jpg')
ens = Ensemble.from_config('config/detectors.json')

# Get both combined and individual results
results = ens.score_image(
    img, 
    patch_size=64, 
    stride=32,
    return_individual=True  # Enable individual tracking
)

# Access results
combined_heatmap = results.combined
individual_heatmaps = results.individual
detector_names = results.detector_names

# Save combined
save_heatmap(combined_heatmap, img, 'combined.png')

# Save each detector's result
for name in detector_names:
    detector_heatmap = individual_heatmaps[name]
    save_heatmap(detector_heatmap, img, f'{name}.png')
    print(f"{name}: max_score={detector_heatmap.max():.4f}")
```

### Analyzing Specific Regions

```python
# Extract a specific region (y:y+h, x:x+w)
region = img[100:200, 150:250]

# Score the region with all detectors
combined_score, individual_scores = ens.score_patch(region)

print(f"Combined score: {combined_score:.4f}")
print("\nIndividual detector scores:")
for detector, score in zip(ens.detectors, individual_scores):
    print(f"  {detector.name}: {score:.4f}")
```

### Comparing Detectors

```python
import numpy as np

# Get individual results
results = ens.score_image(img, return_individual=True)

# See which detector is most suspicious
for name in results.detector_names:
    heat = results.individual[name]
    print(f"{name}: max={heat.max():.4f}, mean={heat.mean():.4f}")

# Find detector with highest max score
max_scores = {name: results.individual[name].max() 
              for name in results.detector_names}
most_suspicious = max(max_scores, key=max_scores.get)
print(f"\nMost suspicious detector: {most_suspicious}")
```

## Understanding Results

### Combined Heatmap

- Shows ensemble decision (all detectors combined)
- **Red areas**: High anomaly score (suspicious)
- **Blue/Green areas**: Low anomaly score (normal)
- Score range: 0.0 (normal) to 1.0 (suspicious)

### Individual Detector Heatmaps

- Shows what each detector found
- Helps understand WHY something is flagged
- Different detectors detect different artifacts

**Example interpretation:**

- `fft_dct` high: Frequency anomalies (resampling, JPEG)
- `color_stats` high: Unnatural color relationships
- `neighbor_consistency` high: Splice boundaries
- `residual_energy` high: Too smooth (edited/AI-generated)

### Score Thresholds (Guidelines)

| Score Range | Interpretation |
|-------------|----------------|
| 0.0 - 0.3 | Likely authentic |
| 0.3 - 0.5 | Mild suspicion |
| 0.5 - 0.7 | Moderate suspicion |
| 0.7 - 0.9 | High suspicion |
| 0.9 - 1.0 | Very high suspicion |

**Note**: These are guidelines, not absolute thresholds. Always verify with multiple methods.

## Advanced Usage

### Custom Detector Selection

```python
from priorpatch.core import Ensemble
from priorpatch.detectors.registry import get_detector_class

# Manually select detectors
detectors = [
    get_detector_class('color_stats')(),
    get_detector_class('fft_dct')(),
]

# Custom weights
weights = {
    'color_stats': 2.0,  # Emphasize color analysis
    'fft_dct': 1.5,
}

ens = Ensemble(detectors, weights)
```

### Batch Processing

```python
import os
from pathlib import Path

# Process all images in a directory
input_dir = Path('images/')
output_dir = Path('results/')
output_dir.mkdir(exist_ok=True)

ens = Ensemble.from_config('config/detectors.json')

for img_path in input_dir.glob('*.jpg'):
    print(f"Processing: {img_path}")
    
    img = load_image(str(img_path))
    results = ens.score_image(img, return_individual=True)
    
    # Save combined result
    out_path = output_dir / f"{img_path.stem}_result.png"
    save_heatmap(results.combined, img, str(out_path))
    
    print(f"  Max score: {results.combined.max():.4f}")
```

## Tips

1. **Patch Size**:
   - Smaller (32-64): More detail, slower
   - Larger (128+): Faster, less detail

2. **Stride**:
   - Smaller: More overlap, better accuracy, slower
   - Larger: Faster, less overlap

3. **Individual Results**:
   - Use when you need to understand WHY something is flagged
   - Helps identify specific manipulation types
   - Good for debugging and analysis

4. **Multiprocessing**:
   - Automatically enabled for images with >10 patches
   - Significant speedup on multi-core systems
   - Use `use_multiprocessing=True` in Python API

5. **Detector Weights**:
   - Adjust based on your use case
   - Higher weight = more influence on final score
   - Start with equal weights (1.0), tune based on results

6. **Progress Bars**:
   - Install tqdm for progress indicators: `pip install tqdm`
   - Optional but recommended for large images

7. **Error Handling**:
   - Tool validates inputs and provides clear error messages
   - Detector failures are logged and tracked
   - If a detector fails on >50% of patches, an error is logged
