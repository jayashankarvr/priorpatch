# Known Limitations

## Score Normalization

Understanding how scores are combined and normalized:

### How It Actually Works

1. **Per-Patch Scoring**: Each detector scores the patch (raw values typically 0-1)
2. **Weighted Averaging**: Detector scores are combined via weighted average (NO per-patch normalization)
3. **Heatmap Normalization**: Final heatmap is normalized to [0,1] for visualization

### Example

```python
# Patch A: detectors return [0.1, 0.2, 0.3]
# Weighted average (weights all 1.0): 0.2
# Stored as 0.2 in heatmap

# Patch B: detectors return [0.7, 0.8, 0.9]
# Weighted average: 0.8
# Stored as 0.8 in heatmap

# After all patches scored, entire heatmap normalized:
# If min=0.2, max=0.8: Patch A becomes 0.0, Patch B becomes 1.0
```

### Implications

- Patch scores CAN be compared before final normalization
- Weighted average preserves relative detector importance
- Final visualization normalization is for display purposes only
- Use `return_individual=True` to see per-detector heatmaps

### Why This Design?

- Weighted averaging allows tuning detector importance
- Preserves more information than per-patch normalization
- Final normalization improves visualization contrast

### Recommendations

- Use heatmap to identify **regions of interest**
- Consider detector weights based on your use case
- Compare patterns across the image
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

Real PRNU requires:

- Wavelet-based denoising
- Multiple reference images
- Proper correlation analysis

## Memory Limitations

Current implementation stores all patches in memory for multiprocessing.

Memory usage:

- 2048×2048 image: ~50 MB of patches
- 4K (3840×3840): ~180 MB of patches

Can cause issues with:

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
- **GPU acceleration**: Optional via CuPy (install separately)
- **Large images**: Can take minutes on single core
- **Multiprocessing**: Use `--jobs/-j` flag or Python API
- **CLI**: Use `-j -1` for all cores or `-j 4` for specific worker count

Typical times (on modern CPU, single-threaded):

- 512×512: 2-5 seconds
- 1024×1024: 10-20 seconds
- 2048×2048: 45-90 seconds
- 4K: 3-5 minutes

## Thresholds

No automatic thresholds for "fake" vs "authentic"

- Scores are relative, not absolute
- Depends on image content
- No ground truth calibration
- Requires manual interpretation

Guidelines (not rules):

- 0.0-0.3: Likely normal
- 0.3-0.7: Ambiguous
- 0.7-1.0: Suspicious

But these vary by detector, image type, and content.

## Future Improvements

See roadmap for planned features that address these limitations.
