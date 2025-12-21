# API Reference

## Core Classes

### `Ensemble`

Main class for managing and running multiple detectors.

```python
from priorpatch.core import Ensemble
```

#### Constructor

```python
Ensemble(detectors: List[DetectorInterface], weights: Optional[Dict[str, float]] = None)
```

**Parameters:**

- `detectors`: List of detector instances
- `weights`: Optional dictionary mapping detector names to weights (default: all weights = 1.0)

#### Class Methods

##### `from_config(path: str) -> Ensemble`

Create an ensemble from a JSON configuration file.

```python
ensemble = Ensemble.from_config('config/detectors.json')
```

**Parameters:**

- `path`: Path to JSON configuration file

**Returns:**

- Configured `Ensemble` instance

**Raises:**

- `FileNotFoundError`: Configuration file doesn't exist
- `ValueError`: Invalid detector name in configuration
- `json.JSONDecodeError`: Malformed JSON

#### Instance Methods

##### `score_patch(patch: np.ndarray) -> PatchResult`

Score a single image patch using all detectors.

```python
result = ensemble.score_patch(patch)
combined_score = result.combined
individual_scores = result.individual
failures = result.failures
```

**Parameters:**

- `patch`: Image patch as numpy array (H, W, 3)

**Returns:**

- `PatchResult` dataclass with fields:
  - `combined`: Weighted normalized combined score (float)
  - `individual`: List of raw detector scores
  - `failures`: List of (detector_name, error_message) tuples for failed detectors

##### `score_image(img: np.ndarray, patch_size: int = 64, stride: int = 32, return_individual: bool = False) -> Union[np.ndarray, AnalysisResult]`

Score an entire image by analyzing overlapping patches.

```python
# Combined result only
heatmap = ensemble.score_image(img, patch_size=64, stride=32)

# Combined + individual detector results
results = ensemble.score_image(img, patch_size=64, stride=32, return_individual=True)
combined = results.combined
individual = results.individual
detector_names = results.detector_names
```

**Parameters:**

- `img`: Input image as numpy array (H, W, 3)
- `patch_size`: Size of square patches (default: 64)
- `stride`: Stride for patch extraction (default: 32)
- `return_individual`: If True, return AnalysisResult with individual detector scores (default: False)

**Returns:**

- If `return_individual=False`: Heatmap of anomaly scores, normalized to [0, 1], shape (H', W')
- If `return_individual=True`: `AnalysisResult` object with fields:
  - `combined`: Combined heatmap (np.ndarray)
  - `individual`: Dict mapping detector names to their heatmaps
  - `detector_names`: List of detector names

**Raises:**

- `ValueError`: Invalid image dimensions

---

## Utility Functions

### `load_image(path: str) -> np.ndarray`

Load an image from file.

```python
from priorpatch.utils import load_image

img = load_image('path/to/image.jpg')
```

**Parameters:**

- `path`: Path to image file

**Returns:**

- Image as numpy array (H, W, 3) in RGB format

**Raises:**

- `FileNotFoundError`: Image file doesn't exist
- `IOError`: Cannot load image

### `save_heatmap(heatmap: np.ndarray, image: np.ndarray, outpath: str, alpha: float = 0.45, cmap: str = 'jet') -> None`

Save a heatmap overlay visualization.

```python
from priorpatch.utils import save_heatmap

save_heatmap(heatmap, img, 'output.png', alpha=0.5, cmap='hot')
```

**Parameters:**

- `heatmap`: 2D array of anomaly scores (H_heat, W_heat)
- `image`: Original image (H, W, 3)
- `outpath`: Output file path
- `alpha`: Overlay transparency, 0=transparent, 1=opaque (default: 0.45)
- `cmap`: Matplotlib colormap name (default: 'jet')

**Raises:**

- `ValueError`: Invalid input dimensions
- `IOError`: Cannot write file

### `validate_path(path: str, must_exist: bool = False) -> Path`

Validate and sanitize a file path.

```python
from priorpatch.utils import validate_path

safe_path = validate_path('user/input/path.jpg', must_exist=True)
```

**Parameters:**

- `path`: Path to validate
- `must_exist`: Require path to exist (default: False)

**Returns:**

- Validated `Path` object

**Raises:**

- `ValueError`: Invalid path
- `FileNotFoundError`: Path doesn't exist (if `must_exist=True`)

---

## Detector Interface

### `DetectorInterface`

Abstract base class for all detectors.

```python
from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
import numpy as np

@register_detector
class MyDetector(DetectorInterface):
    name = 'my_detector'
    
    def score(self, patch: np.ndarray) -> float:
        # Your detection logic here
        return anomaly_score
```

#### Abstract Methods

##### `score(patch: np.ndarray) -> float`

Compute anomaly score for an image patch.

**Parameters:**

- `patch`: Image patch as numpy array (H, W, 3)

**Returns:**

- Anomaly score as float (higher = more suspicious)

**Note:** The score range is detector-specific and will be normalized by the ensemble.

---

## Registry Functions

### `register_detector(cls: Type[DetectorInterface]) -> Type[DetectorInterface]`

Decorator to register a detector class.

```python
from priorpatch.detectors.registry import register_detector

@register_detector
class MyDetector(DetectorInterface):
    name = 'my_detector'
    # ...
```

**Parameters:**

- `cls`: Detector class to register

**Returns:**

- Same class (allows use as decorator)

**Raises:**

- `ValueError`: Detector name already registered

### `get_detector_class(name: str) -> Optional[Type[DetectorInterface]]`

Retrieve a detector class by name.

```python
from priorpatch.detectors.registry import get_detector_class

DetectorClass = get_detector_class('color_stats')
detector = DetectorClass()
```

**Parameters:**

- `name`: Detector name

**Returns:**

- Detector class if found, `None` otherwise

---

## Built-in Detectors

### `ColorStatsDetector`

Analyzes RGB channel correlations.

- **Name**: `'color_stats'`
- **Algorithm**: Computes deviation from expected RGB correlation matrix
- **Best for**: Detecting unnatural color relationships

### `NeighborConsistencyDetector`

Analyzes spatial prediction consistency.

- **Name**: `'neighbor_consistency'`
- **Algorithm**: Compares pixels to 8-neighbor average
- **Best for**: Detecting splice boundaries, local inconsistencies

### `FFTDCTDetector`

Examines frequency domain characteristics.

- **Name**: `'fft_dct'`
- **Algorithm**: Analyzes radial power spectrum and power-law slope
- **Best for**: Detecting resampling, compression artifacts

### `ResidualEnergyDetector`

Measures high-frequency residual energy.

- **Name**: `'residual_energy'`
- **Algorithm**: Computes energy after Gaussian smoothing
- **Best for**: Detecting smoothed/blurred regions

### `DCTCoocDetector`

Analyzes DCT coefficient patterns.

- **Name**: `'dct_cooccurrence'`
- **Algorithm**: Examines variance in 8x8 block DC coefficients
- **Best for**: Detecting JPEG compression inconsistencies

### `PRNUWaveletDetector`

Sensor noise pattern detector using wavelet denoising.

- **Name**: `'prnu_wavelet'`
- **Algorithm**: Extracts PRNU via wavelet-based noise extraction
- **Best for**: AI-generated image detection, source camera identification

### `ELADetector`

Error Level Analysis detector.

- **Name**: `'ela'`
- **Algorithm**: JPEG recompression analysis
- **Best for**: Detecting JPEG manipulation, spliced images

### `BenfordLawDetector`

Statistical distribution detector.

- **Name**: `'benford_law'`
- **Algorithm**: First-digit distribution of DCT coefficients
- **Best for**: General manipulation detection (~90% F1-score)

### `CFAArtifactDetector`

Camera sensor artifact detector.

- **Name**: `'cfa_artifact'`
- **Algorithm**: Detects Bayer demosaicing patterns
- **Best for**: AI detection (highest discriminative power)

---

## Configuration Format

### JSON Configuration Schema

```json
{
  "enabled_detectors": [
    "color_stats",
    "neighbor_consistency",
    "fft_dct",
    "residual_energy",
    "dct_cooccurrence",
    "benford_law",
    "cfa_artifact",
    "ela"
  ],
  "detector_weights": {
    "color_stats": 1.0,
    "neighbor_consistency": 1.2,
    "fft_dct": 1.5,
    "residual_energy": 1.0,
    "dct_cooccurrence": 1.1,
    "benford_law": 1.8,
    "cfa_artifact": 2.0,
    "ela": 1.2
  }
}
```

**Fields:**

- `enabled_detectors`: List of detector names to use
- `detector_weights`: Optional weights for each detector (default: 1.0)

---

## Examples

### Basic Usage

```python
from priorpatch import Ensemble, load_image, save_heatmap

# Load image
img = load_image('image.jpg')

# Create ensemble from config
ensemble = Ensemble.from_config('config/detectors.json')

# Analyze image
heatmap = ensemble.score_image(img, patch_size=64, stride=32)

# Save visualization
save_heatmap(heatmap, img, 'result.png')
```

### Custom Detector Weights

```python
from priorpatch.core import Ensemble
from priorpatch.detectors.registry import get_detector_class

# Manually create detectors with custom weights
detectors = [
    get_detector_class('color_stats')(),
    get_detector_class('fft_dct')(),
]

weights = {
    'color_stats': 2.0,  # Double weight
    'fft_dct': 0.5,      # Half weight
}

ensemble = Ensemble(detectors, weights)
```

### Analyzing Specific Regions

```python
# Extract a specific region
region = img[100:200, 150:250]  # y:y+h, x:x+w

# Score the region
result = ensemble.score_patch(region)

print(f"Combined score: {result.combined:.4f}")
for i, s in enumerate(result.individual):
    detector_name = ensemble.detectors[i].name
    print(f"  {detector_name}: {s:.4f}")

# Check for any failures
if result.failures:
    print("Failures:", result.failures)
```
