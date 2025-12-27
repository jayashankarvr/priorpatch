# Changelog

All notable changes to PriorPatch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CLI multiprocessing support via `--jobs/-j` flag
- Detector timeout protection (10 second warning threshold)
- Smart error messages with suggestions for typos
- `get_config()` and `from_config()` protocol for detector serialization

### Fixed

- Multiprocessing now preserves custom detector parameters
- Worker state race condition eliminated with _WorkerState class
- Detector timeout tracking prevents infinite hangs
- Removed upper bounds from requirements.txt to allow security updates
- Fixed logging configuration to not override embedder's settings
- Corrected documentation about score normalization (was incorrectly described)
- Updated .gitignore to include .claude/ directory
- Humanized AI-generated documentation and comments
- LBP detector now uses consistent luminance conversion
- Config version validation now raises ERROR instead of warning
- Corrected hardcoded constants (epsilon: 1e-6, failure rate: 20%)
- Fixed documentation claim about GPU acceleration
- Removed unused imports

### Changed

- Documentation now accurately describes weighted averaging instead of per-patch normalization
- Improved clarity of detector and API documentation
- Better error messages with difflib suggestions

## [0.1.0] - 2025-12-21

### Added

- Initial release
- 16 forensic detectors for image manipulation detection
- GPU acceleration support via CuPy
- CLI tool for batch processing
- Comprehensive documentation
- 248 unit tests with 83% code coverage
- GitHub Actions CI/CD pipeline
- Multi-platform support (Linux, macOS, Windows)

### Detectors Included

- Color Statistics
- Neighbor Consistency
- FFT/DCT Analysis
- Residual Energy
- DCT Co-occurrence
- JPEG Ghost
- Error Level Analysis (ELA)
- Copy-Move Detection
- Noise Consistency
- PRNU (Photo Response Non-Uniformity)
- CFA (Color Filter Array) Artifacts
- LBP (Local Binary Patterns) Texture
- Lighting Consistency
- Chromatic Aberration
- GAN Fingerprints
- Benford's Law

[Unreleased]: https://github.com/jayashankarvr/priorpatch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jayashankarvr/priorpatch/releases/tag/v0.1.0
