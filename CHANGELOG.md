# Changelog

All notable changes to PriorPatch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.2.0

- BT.709 luminance conversion (modern cameras)
- Optional gamma linearization for physics-correct analysis
- Improved score combination (robust/Bayesian methods)

---

## [0.1.0] - 2025-12-21

**Initial Release** - Math-only forensic toolkit for detecting AI-generated and manipulated images.

### Core Framework

- **Ensemble System**: Combine multiple detectors with configurable weights
- **Patch-based Analysis**: Sliding window analysis with configurable patch size and stride
- **Heatmap Visualization**: Overlay anomaly scores on original image
- **CLI Tool**: `priorpatch analyze` for command-line usage
- **JSON Configuration**: Flexible detector selection and weighting
- **Multiprocessing**: Parallel patch processing for faster analysis
- **GPU Acceleration**: Optional CuPy backend with automatic CPU fallback
- **Auto-discovery**: Drop-in detector plugins (just add a file)

### Detectors (16)

**Frequency & Statistical Analysis:**

| Detector | Description | Best For |
|----------|-------------|----------|
| `fft_dct` | Frequency domain analysis | Resampling, compression |
| `dct_cooccurrence` | JPEG 8x8 block patterns | JPEG inconsistencies |
| `benford_law` | DCT coefficient statistics | General manipulation |
| `residual_energy` | High-frequency residual | Smoothing, AI backgrounds |

**Color & Spatial Analysis:**

| Detector | Description | Best For |
|----------|-------------|----------|
| `color_stats` | RGB channel correlation | Color manipulation |
| `neighbor_consistency` | Pixel prediction | Splice boundaries |
| `lbp_texture` | Local Binary Patterns | Deepfakes, AI textures |

**Camera/Sensor Forensics:**

| Detector | Description | Best For |
|----------|-------------|----------|
| `cfa_artifact` | Bayer demosaicing patterns | AI detection (highest accuracy) |
| `prnu_wavelet` | Sensor noise via wavelets | AI detection, source ID |
| `chromatic_aberration` | Lens CA consistency | AI detection, splicing |
| `noise_consistency` | Sensor noise patterns | AI-generated regions |

**Manipulation Detection:**

| Detector | Description | Best For |
|----------|-------------|----------|
| `ela` | Error Level Analysis | JPEG manipulation |
| `jpeg_ghost` | Recompression artifacts | JPEG splicing |
| `gan_fingerprint` | Upsampling artifacts | AI-generated images |
| `lighting_consistency` | Light direction | Composites |
| `copy_move` | Duplicated regions | Copy-paste forgery |

### CLI Features

```bash
# Analyze single image
priorpatch analyze -i photo.jpg -o results/

# Batch processing
priorpatch analyze --input-dir photos/ -o results/

# Glob patterns
priorpatch analyze -i "photos/*.png" -o results/

# Custom settings
priorpatch analyze -i photo.jpg --patch-size 128 --stride 64

# Per-detector output
priorpatch analyze -i photo.jpg --save-individual
```

### Documentation

- Installation guide
- Quick start tutorial
- Detector documentation with accuracy notes
- Architecture overview
- API reference
- Contributing guidelines

### Infrastructure

- GitHub Actions CI/CD (Python 3.8-3.12)
- Test suite with pytest
- Type hints (PEP 561 compliant)
- Dependabot for dependency updates
- Issue and PR templates
- Apache 2.0 License

### Known Limitations

1. **Color Space**: Uses BT.601 luminance (SDTV). BT.709 (modern) planned for v0.2.0
2. **Gamma**: Operates on gamma-encoded values. Linearization option in v0.2.0
3. **CFA Pattern**: Assumes RGGB Bayer. Multi-pattern detection in v0.2.0
4. **PRNU Model**: Uses additive model. Multiplicative model in v0.2.0
5. **Benchmarks**: Not validated on standard datasets yet. Coming in v0.4.0

### References

Based on peer-reviewed research:

- Lukas et al. (2006) - PRNU camera identification
- Fu et al. (2007) - Benford's Law for JPEG
- Johnson & Farid (2006) - Chromatic aberration forensics
- Bonettini et al. (2020) - GAN detection
- Fridrich et al. (2003) - Copy-move detection

---

## Links

- [Repository](https://github.com/jayashankarvr/priorpatch)
- [Issues](https://github.com/jayashankarvr/priorpatch/issues)
- [Documentation](https://github.com/jayashankarvr/priorpatch#readme)
