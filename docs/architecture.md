# Architecture

## Overview

PriorPatch uses a modular, plugin-based architecture that separates concerns and makes the system easily extensible.

## Core Components

### 1. Detector Interface (`DetectorInterface`)

The foundation of the system is the abstract `DetectorInterface` class:

```python
class DetectorInterface(ABC):
    name = 'base_detector'
    
    @abstractmethod
    def score(self, patch: np.ndarray) -> float:
        raise NotImplementedError
```

All forensic detectors must:

- Inherit from `DetectorInterface`
- Implement the `score()` method
- Define a unique `name` attribute

### 2. Registry Pattern

The registry system (`registry.py`) provides automatic detector discovery:

```python
@register_detector
class MyDetector(DetectorInterface):
    name = 'my_detector'
    # ...
```

Benefits:

- **Automatic registration**: Detectors self-register on import
- **Decoupling**: Core code doesn't need to know about specific detectors
- **Extensibility**: Add new detectors without modifying core code

### 3. Ensemble Manager

The `Ensemble` class orchestrates multiple detectors:

**Key Responsibilities:**

- Load detector configuration from JSON
- Instantiate detector instances
- Score patches using all detectors
- Normalize and combine scores
- Generate spatial heatmaps

**Scoring Pipeline:**

```
Image -> Patches -> [Detector 1, Detector 2, ..., Detector N]
                           |
                           v
                    Individual Scores
                           |
                           v
                    Min-Max Normalization
                           |
                           v
                    Weighted Average
                           |
                           v
                    Combined Score -> Heatmap
```

### 4. Configuration System

JSON-based configuration (`config/detectors.json`):

```json
{
  "enabled_detectors": ["color_stats", "neighbor_consistency", ...],
  "detector_weights": {
    "color_stats": 1.0,
    "neighbor_consistency": 1.2,
    ...
  }
}
```

## Design Patterns

### Plugin Architecture

- **Purpose**: Enable runtime extensibility
- **Implementation**: Registry + decorator pattern
- **Benefit**: Add detectors without modifying core code

### Strategy Pattern

- **Purpose**: Interchangeable detection algorithms
- **Implementation**: `DetectorInterface` abstraction
- **Benefit**: Swap/combine detection strategies

### Template Method

- **Purpose**: Standardize analysis workflow
- **Implementation**: `Ensemble.score_image()` pipeline
- **Benefit**: Consistent processing across detectors

## Data Flow

```
1. User Input (CLI or API)
        |
        v
2. Load Configuration
        |
        v
3. Initialize Ensemble
        |
        v
4. Load Image
        |
        v
5. Extract Patches (sliding window)
        |
        v
6. For Each Patch:
   - Score with all detectors
   - Normalize scores (z-score)
   - Combine with weights
        |
        v
7. Generate Heatmap
        |
        v
8. Output Results (visualization + JSON)
```

## Module Structure

```bash
priorpatch/
|-- __init__.py          # Package initialization, exports
|-- cli.py               # Command-line interface
|-- core.py              # Ensemble logic
|-- utils.py             # I/O and visualization helpers
|-- gpu_backend.py       # GPU acceleration (CuPy) with CPU fallback
+-- detectors/           # Auto-discovered detector plugins
    |-- base.py          # Abstract interface
    |-- registry.py      # Plugin registry
    |-- color_stats.py   # RGB correlation
    |-- neighbor_consistency.py
    |-- fft_dct.py       # Frequency analysis (GPU-accelerated)
    |-- residual_energy.py
    |-- dct_cooccurrence.py
    |-- benford_law.py   # DCT coefficient analysis
    |-- cfa_artifact.py  # Camera sensor artifacts
    |-- chromatic_aberration.py
    |-- copy_move.py     # Duplicated region detection
    |-- ela.py           # Error Level Analysis
    |-- gan_fingerprint.py
    |-- jpeg_ghost.py
    |-- lbp_texture.py
    |-- lighting_consistency.py
    |-- noise_consistency.py
    +-- prnu_wavelet.py  # Sensor noise analysis
```

## Extensibility Points

### Adding a New Detector

1. Create file in `detectors/`
2. Inherit from `DetectorInterface`
3. Implement `score()` method
4. Add `@register_detector` decorator
5. Update configuration file

### Adding a New Score Combination Method

Modify `Ensemble.score_patch()` to implement:

- Different normalization schemes
- Alternative voting/fusion methods
- Confidence weighting
- Outlier handling

### Adding New Output Formats

Extend `utils.py` with:

- Different visualization styles
- Export to other formats (CSV, XML, etc.)
- Interactive visualizations

## Performance Considerations

### Current Approach

- **Patch extraction**: Sliding window with configurable stride
- **Processing**: Sequential (single-threaded)
- **Memory**: Loads entire image into memory

### Optimization Opportunities

- **Multiprocessing**: Parallelize patch scoring
- **Vectorization**: Process multiple patches simultaneously
- **Streaming**: Process image in chunks for large files
- **Caching**: Cache detector computations for overlapping regions

## Testing Strategy

- **Unit tests**: Individual detector logic
- **Integration tests**: Ensemble coordination
- **End-to-end tests**: Full pipeline with sample images
- **Regression tests**: Ensure consistent results across versions
