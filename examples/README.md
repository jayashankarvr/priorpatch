# PriorPatch Examples

This directory contains example scripts demonstrating various use cases for PriorPatch.

## Examples Overview

### 1. `basic_usage.py`

**Simplest usage example**

Shows the minimal code needed to analyze an image:

- Load image
- Create ensemble
- Generate heatmap
- Save visualization

**Run:**

```bash
cd examples/
python basic_usage.py
```

### 2. `custom_detectors.py`

**Custom detector selection and weights**

Demonstrates:

- Manual detector selection
- Custom weight configuration
- Analyzing specific image regions
- Interpreting individual detector scores

**Run:**

```bash
python custom_detectors.py
```

### 3. `batch_processing.py`

**Process multiple images efficiently**

Shows how to:

- Batch process multiple images
- Reuse ensemble instance
- Generate summary reports
- Handle errors gracefully

**Run:**

```bash
python batch_processing.py
```

**Customize:**
Edit the `image_paths` list to process your own images.

### 4. `performance_comparison.py`

**Benchmark sequential vs parallel processing**

Demonstrates:

- Performance with/without multiprocessing
- Processing large images efficiently
- Measuring speedup

**Run:**

```bash
python performance_comparison.py
```

**Note:** This creates a large synthetic image for testing.

### 5. `individual_detector_analysis.py`

**Analyze individual detector results**

Demonstrates:

- Viewing each detector's output separately
- Comparing detector results
- Understanding which detectors flag what
- Detector agreement analysis

**Run:**

```bash
python individual_detector_analysis.py
```

**Output:** Creates comparison grid showing all detectors side-by-side.

## Sample Input

The `sample_input.png` file is provided for testing. You can replace it with your own images.

## Common Patterns

### Loading Images

```python
from priorpatch.utils import load_image
img = load_image('path/to/image.jpg')
```

### Creating Ensemble

```python
from priorpatch import Ensemble

# From config file
ensemble = Ensemble.from_config('config/detectors.json')

# Manual creation
from priorpatch.detectors.registry import get_detector_class
detectors = [
    get_detector_class('color_stats')(),
    get_detector_class('fft_dct')()
]
weights = {'color_stats': 1.0, 'fft_dct': 1.5}
ensemble = Ensemble(detectors, weights)
```

### Analyzing Images

```python
# Full image analysis
heatmap = ensemble.score_image(img, patch_size=64, stride=32)

# Specific region
region = img[y:y+h, x:x+w]
score, individual_scores = ensemble.score_patch(region)
```

### Saving Results

```python
from priorpatch.utils import save_heatmap
save_heatmap(heatmap, img, 'output.png', alpha=0.5, cmap='hot')
```

## Tips

1. **Patch Size**: Smaller patches (32-64) provide finer detail but are slower
2. **Stride**: Smaller stride increases overlap and accuracy but increases computation
3. **Multiprocessing**: Enable for images with >100 patches
4. **Detector Selection**: Use fewer detectors for faster processing
5. **Weights**: Adjust based on which artifacts you want to detect

## Interpreting Results

**Heatmap Colors:**

- Blue/Green: Normal regions (low anomaly score)
- Yellow: Slightly suspicious
- Red: Highly suspicious (high anomaly score)

**Score Ranges:**

- 0.0-0.3: Likely authentic
- 0.3-0.7: Moderate suspicion
- 0.7-1.0: High suspicion

**Note:** These are not absolute thresholds. Always verify with multiple methods.

## Creating Your Own Examples

Feel free to create your own examples! Just follow these patterns:

```python
#!/usr/bin/env python
"""
Description of what this example demonstrates.
"""

from priorpatch import Ensemble, load_image, save_heatmap

def main():
    # Your code here
    pass

if __name__ == '__main__':
    main()
```

## Need Help?

- Check the [Documentation](../docs/)
- See the [API Reference](../docs/api.md)
- Open an [Issue](https://github.com/jayashankarvr/priorpatch/issues)
