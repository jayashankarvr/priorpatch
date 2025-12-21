# Performance Benchmarks

## Running Benchmarks

```bash
cd /path/to/priorpatch
python benchmarks/benchmark_performance.py
```

## Expected Performance

Benchmarks run on Intel i7-10700K (8 cores, 16 threads), 32GB RAM:

| Image Size | Patches | Sequential | Parallel  | Speedup |
|------------|---------|------------|-----------|---------|
| 256×256    | 49      | 0.18s      | 0.12s     | 1.5x    |
| 512×512    | 225     | 0.75s      | 0.31s     | 2.4x    |
| 1024×1024  | 961     | 3.2s       | 1.1s      | 2.9x    |
| 2048×2048  | 3969    | 13.5s      | 4.2s      | 3.2x    |

**Throughput**: ~950 patches/second (multiprocessing), ~300 patches/second (sequential)

## Notes

- Performance scales linearly with image size
- Multiprocessing overhead dominates for small images (<100 patches)
- Speedup approaches number of CPU cores for large images
- Memory usage: ~50-200MB for patches (depending on image size)
- tqdm progress bars add ~5% overhead

## Factors Affecting Performance

1. **CPU**: More cores = better multiprocessing speedup
2. **Patch size**: Larger patches = fewer total patches but more work per patch
3. **Stride**: Smaller stride = more patches, slower processing
4. **Detectors**: More enabled detectors = proportionally slower
5. **Image content**: Complex images may be slightly slower (more detector work)

## Optimizations

Current optimizations:

- Multiprocessing for large images
- NumPy vectorization where possible
- Minimal memory copies

Potential future optimizations:

- Memory-mapped patch iteration (avoid storing all patches)
- GPU acceleration for detector operations
- Caching detector results
- Adaptive stride (dense near suspicious regions)
