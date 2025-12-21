# Known Limitations

## Per-Patch Normalization

**Important**: Scores are normalized per-patch, which means:

### What This Means

Each patch's detector scores are independently normalized to [0,1] before combining. This loses absolute scale information.

### Example

```python
# Patch A: detectors return [0.1, 0.2, 0.3, 0.4, 0.5]
# Normalized to: [0.0, 0.25, 0.5, 0.75, 1.0]

# Patch B: detectors return [0.5, 0.6, 0.7, 0.8, 0.9]
# Normalized to: [0.0, 0.25, 0.5, 0.75, 1.0]

# Same normalized values, but Patch B is objectively more suspicious!
```

### Implications

- Heatmap shows **relative** suspiciousness within each patch
- **Cannot directly compare** scores between different patches
- Patch with highest score might not be the most suspicious in absolute terms
- Entire image normalization happens afterward, but per-patch info is already lost

### Why This Design?

- Combines detectors with different output ranges
- Prevents one high-magnitude detector from dominating
- Standard practice in ensemble methods

### Alternatives (Not Implemented)

1. **Global normalization**: Normalize across all patches (requires two passes)
2. **Calibration**: Pre-compute detector ranges on training data
3. **Quantile normalization**: Use percentile-based scaling

### Recommendations

- Use heatmap to identify **regions of interest**
- Don't rely on absolute score values
- Compare patterns, not numbers
- Always manually inspect flagged regions

## Patch Overlap Artifacts

**Default settings**: `patch_size=64, stride=32` = 50% overlap

### Effect

- **Corner pixels**: Analyzed 1×
- **Edge pixels**: Analyzed 2×
- **Center pixels**: Analyzed 4×

### Impact

- Heatmap values not directly comparable across image
- Edges may appear different from center (not due to manipulation)
- More overlap = more averaged/smoothed results

### Workarounds

- Use `stride == patch_size` for no overlap (less smooth heatmap)
- Account for overlap when interpreting results
- Focus on relative differences, not absolute values

## PRNU Detector

Current implementation is a stub using gradient magnitude, not proper sensor noise extraction.

**Not suitable for**:

- Device identification
- Adversarial detection
- Production use

**Real PRNU requires**:

- Wavelet-based denoising
- Multiple reference images
- Proper correlation analysis

## Memory Limitations

**Current implementation** stores all patches in memory for multiprocessing.

**Memory usage**:

- 2048×2048 image: ~50 MB of patches
- 4K (3840×3840): ~180 MB of patches

**Can cause issues** with:

- Very large images (>4K)
- Limited RAM systems
- Batch processing many images

## Detection Limitations

### Works Well For

- Copy-paste splicing
- Obvious resampling
- JPEG compression mismatches
- Over-smoothed regions

### Struggles With

- High-quality professional edits
- Latest AI generation models (2025+)
- Naturally unusual images
- Heavy legitimate post-processing
- Adversarial attacks

### Not Designed For

- Video analysis
- Real-time detection
- Specific manipulation type classification
- Exact location of manipulation boundaries

## Performance

- **Speed**: Depends on image size and CPU cores
- **No GPU acceleration**: CPU-only implementation
- **Large images**: Can take minutes on single core
- **Multiprocessing**: Helps but has overhead

Typical times (on modern CPU):

- 512×512: 2-5 seconds
- 1024×1024: 10-20 seconds
- 2048×2048: 45-90 seconds
- 4K: 3-5 minutes

## Thresholds

**No automatic thresholds** for "fake" vs "authentic"

- Scores are relative, not absolute
- Depends on image content
- No ground truth calibration
- Requires manual interpretation

**Guidelines** (not rules):

- 0.0-0.3: Likely normal
- 0.3-0.7: Ambiguous
- 0.7-1.0: Suspicious

But these vary by detector, image type, and content.

## Future Improvements

See roadmap for planned features that address these limitations.
