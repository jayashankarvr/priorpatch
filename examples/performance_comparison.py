"""
Example: Performance comparison with and without multiprocessing.

This script demonstrates the performance benefits of parallel processing
for large images.
"""

import time
import numpy as np
from priorpatch import Ensemble, load_image

def benchmark_performance():
    """Compare sequential vs parallel processing."""
    
    print("Performance Benchmark")
    print("="*60)
    
    # Create a large synthetic image for testing
    print("\nGenerating large test image (2048x2048)...")
    large_img = np.random.randint(0, 256, (2048, 2048, 3), dtype=np.uint8)
    
    # Initialize ensemble
    ensemble = Ensemble.from_config('config/detectors.json')
    
    # Test 1: Sequential processing
    print("\nTest 1: Sequential processing")
    start = time.time()
    heatmap_seq = ensemble.score_image(
        large_img, 
        patch_size=64, 
        stride=32,
        use_multiprocessing=False
    )
    time_seq = time.time() - start
    print(f"  Time: {time_seq:.2f} seconds")
    print(f"  Patches analyzed: {heatmap_seq.size}")
    
    # Test 2: Parallel processing
    print("\nTest 2: Parallel processing (all CPUs)")
    start = time.time()
    heatmap_par = ensemble.score_image(
        large_img, 
        patch_size=64, 
        stride=32,
        use_multiprocessing=True,
        n_jobs=-1
    )
    time_par = time.time() - start
    print(f"  Time: {time_par:.2f} seconds")
    print(f"  Patches analyzed: {heatmap_par.size}")
    
    # Calculate speedup
    speedup = time_seq / time_par
    print("\n" + "="*60)
    print(f"Speedup: {speedup:.2f}x faster with multiprocessing")
    print(f"Time saved: {time_seq - time_par:.2f} seconds")
    
    # Verify results are similar
    difference = np.abs(heatmap_seq - heatmap_par).mean()
    print(f"Mean difference: {difference:.6f} (should be ~0)")

def main():
    try:
        benchmark_performance()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError during benchmark: {e}")

if __name__ == '__main__':
    main()
