"""
Performance benchmarks for PriorPatch.

Run with: python benchmarks/benchmark_performance.py
"""

import time
import numpy as np
from priorpatch import Ensemble


def benchmark_image_size(size, patch_size=64, stride=32):
    """Benchmark processing time for a given image size."""
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    ensemble = Ensemble.from_config('config/detectors.json')
    
    # Sequential
    start = time.time()
    _ = ensemble.score_image(img, patch_size=patch_size, stride=stride, use_multiprocessing=False)
    seq_time = time.time() - start
    
    # Multiprocessing
    start = time.time()
    _ = ensemble.score_image(img, patch_size=patch_size, stride=stride, use_multiprocessing=True, n_jobs=-1)
    mp_time = time.time() - start
    
    num_patches = ((size - patch_size) // stride + 1) ** 2
    
    return {
        'size': size,
        'patches': num_patches,
        'seq_time': seq_time,
        'mp_time': mp_time,
        'speedup': seq_time / mp_time if mp_time > 0 else 0,
        'patches_per_sec_seq': num_patches / seq_time if seq_time > 0 else 0,
        'patches_per_sec_mp': num_patches / mp_time if mp_time > 0 else 0,
    }


def main():
    print("PriorPatch Performance Benchmarks")
    print("=" * 80)
    print()
    
    sizes = [256, 512, 1024, 2048]
    results = []
    
    for size in sizes:
        print(f"Benchmarking {size}x{size} image...")
        result = benchmark_image_size(size)
        results.append(result)
        print(f"  Patches: {result['patches']}")
        print(f"  Sequential: {result['seq_time']:.2f}s ({result['patches_per_sec_seq']:.1f} patches/sec)")
        print(f"  Parallel:   {result['mp_time']:.2f}s ({result['patches_per_sec_mp']:.1f} patches/sec)")
        print(f"  Speedup:    {result['speedup']:.2f}x")
        print()
    
    print("=" * 80)
    print("Summary:")
    print()
    print(f"{'Size':>8} | {'Patches':>8} | {'Sequential':>12} | {'Parallel':>12} | {'Speedup':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['size']:>8} | {r['patches']:>8} | {r['seq_time']:>10.2f}s | {r['mp_time']:>10.2f}s | {r['speedup']:>6.2f}x")
    
    print()
    print("Note: Results vary by CPU cores and system load.")
    print("Run multiple times for more accurate measurements.")


if __name__ == '__main__':
    main()
