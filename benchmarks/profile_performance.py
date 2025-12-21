#!/usr/bin/env python3
"""
Performance profiling script for PriorPatch.

Identifies bottlenecks in detector execution and provides optimization insights.

Usage:
    python -m benchmarks.profile_performance [--image IMAGE] [--output OUTPUT]
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from priorpatch import Ensemble, load_image
from priorpatch.detectors.registry import DETECTOR_REGISTRY


def profile_single_detector(
    detector_name: str,
    patch: np.ndarray,
    n_iterations: int = 100
) -> Dict:
    """Profile a single detector.

    Args:
        detector_name: Name of detector to profile
        patch: Image patch to use
        n_iterations: Number of iterations

    Returns:
        Dictionary with timing statistics
    """
    detector_cls = DETECTOR_REGISTRY.get(detector_name)
    if detector_cls is None:
        return {'error': f'Detector {detector_name} not found'}

    detector = detector_cls()

    # Warmup
    for _ in range(5):
        detector.score(patch)

    # Profile
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        detector.score(patch)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        'name': detector_name,
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'iterations': n_iterations
    }


def profile_all_detectors(
    patch: np.ndarray,
    n_iterations: int = 50
) -> List[Dict]:
    """Profile all registered detectors.

    Args:
        patch: Image patch to use
        n_iterations: Number of iterations per detector

    Returns:
        List of timing dictionaries, sorted by mean time
    """
    results = []

    for name in DETECTOR_REGISTRY:
        print(f"Profiling {name}...", end=' ', flush=True)
        result = profile_single_detector(name, patch, n_iterations)
        print(f"{result.get('mean_ms', 'N/A'):.2f}ms")
        results.append(result)

    # Sort by mean time (slowest first)
    results.sort(key=lambda x: x.get('mean_ms', 0), reverse=True)

    return results


def profile_ensemble(
    ensemble: Ensemble,
    image: np.ndarray,
    patch_size: int = 64,
    stride: int = 32
) -> Dict:
    """Profile full ensemble on an image.

    Args:
        ensemble: Ensemble instance
        image: Full image
        patch_size: Patch size
        stride: Stride

    Returns:
        Profiling results
    """
    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    result = ensemble.score_image(
        image,
        patch_size=patch_size,
        stride=stride,
        return_individual=True
    )
    total_time = time.perf_counter() - start

    profiler.disable()

    # Get stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)

    h, w = image.shape[:2]
    n_patches = len(range(0, h - patch_size + 1, stride)) * len(range(0, w - patch_size + 1, stride))

    return {
        'total_time_s': total_time,
        'image_size': (h, w),
        'patch_size': patch_size,
        'stride': stride,
        'n_patches': n_patches,
        'time_per_patch_ms': (total_time / n_patches) * 1000,
        'profile_stats': stream.getvalue()
    }


def profile_gpu_vs_cpu(
    patch: np.ndarray,
    n_iterations: int = 50
) -> Dict:
    """Compare GPU vs CPU performance for FFT-heavy detectors.

    Args:
        patch: Image patch
        n_iterations: Number of iterations

    Returns:
        Comparison results
    """
    from priorpatch import gpu_backend, disable_gpu, enable_gpu

    # Detectors that use FFT
    fft_detectors = ['fft_dct', 'cfa_artifact', 'gan_fingerprint']

    results = {'cpu': {}, 'gpu': {}}

    # CPU timing
    disable_gpu()
    for name in fft_detectors:
        if name in DETECTOR_REGISTRY:
            result = profile_single_detector(name, patch, n_iterations)
            results['cpu'][name] = result['mean_ms']

    # GPU timing (if available)
    enable_gpu()
    if gpu_backend.use_gpu():
        for name in fft_detectors:
            if name in DETECTOR_REGISTRY:
                result = profile_single_detector(name, patch, n_iterations)
                results['gpu'][name] = result['mean_ms']

        # Compute speedup
        results['speedup'] = {}
        for name in fft_detectors:
            if name in results['cpu'] and name in results['gpu']:
                speedup = results['cpu'][name] / results['gpu'][name]
                results['speedup'][name] = speedup
    else:
        results['gpu'] = 'GPU not available'
        results['speedup'] = 'N/A'

    return results


def print_results(detector_results: List[Dict], ensemble_results: Optional[Dict] = None):
    """Print formatted profiling results."""
    print("\n" + "=" * 70)
    print("DETECTOR PERFORMANCE PROFILE")
    print("=" * 70)

    print(f"\n{'Detector':<30} {'Mean (ms)':<12} {'Std (ms)':<10} {'P95 (ms)':<10}")
    print("-" * 70)

    total_mean = 0
    for result in detector_results:
        if 'error' in result:
            print(f"{result.get('name', 'unknown'):<30} ERROR: {result['error']}")
        else:
            name = result['name']
            mean = result['mean_ms']
            std = result['std_ms']
            p95 = result['p95_ms']
            total_mean += mean
            print(f"{name:<30} {mean:<12.2f} {std:<10.2f} {p95:<10.2f}")

    print("-" * 70)
    print(f"{'TOTAL (sequential)':<30} {total_mean:<12.2f}")

    if ensemble_results:
        print("\n" + "=" * 70)
        print("ENSEMBLE PERFORMANCE")
        print("=" * 70)
        print(f"Image size:        {ensemble_results['image_size']}")
        print(f"Patch size:        {ensemble_results['patch_size']}")
        print(f"Stride:            {ensemble_results['stride']}")
        print(f"Number of patches: {ensemble_results['n_patches']}")
        print(f"Total time:        {ensemble_results['total_time_s']:.2f}s")
        print(f"Time per patch:    {ensemble_results['time_per_patch_ms']:.2f}ms")

        print("\nTop 30 functions by cumulative time:")
        print(ensemble_results['profile_stats'])


def main():
    parser = argparse.ArgumentParser(
        description='Profile PriorPatch performance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to image for profiling (generates synthetic if not provided)'
    )

    parser.add_argument(
        '--patch-size', '-p',
        type=int,
        default=64,
        help='Patch size for detector profiling'
    )

    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=50,
        help='Number of iterations per detector'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results'
    )

    parser.add_argument(
        '--gpu-compare',
        action='store_true',
        help='Compare GPU vs CPU performance'
    )

    parser.add_argument(
        '--full-image',
        action='store_true',
        help='Profile full image analysis (slower)'
    )

    args = parser.parse_args()

    # Load or generate image
    if args.image:
        print(f"Loading image: {args.image}")
        image = load_image(args.image)
    else:
        print("Generating synthetic test image...")
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Extract patch for individual detector profiling
    patch = image[:args.patch_size, :args.patch_size].copy()

    print(f"\nPatch size: {patch.shape}")
    print(f"Iterations: {args.iterations}")
    print(f"Registered detectors: {len(DETECTOR_REGISTRY)}")

    # Profile individual detectors
    print("\nProfiling individual detectors...")
    detector_results = profile_all_detectors(patch, args.iterations)

    ensemble_results = None
    if args.full_image:
        print("\nProfiling full image analysis...")
        ensemble = Ensemble.from_config()
        ensemble_results = profile_ensemble(ensemble, image)

    gpu_results = None
    if args.gpu_compare:
        print("\nComparing GPU vs CPU performance...")
        gpu_results = profile_gpu_vs_cpu(patch, args.iterations)

    # Print results
    print_results(detector_results, ensemble_results)

    if gpu_results:
        print("\n" + "=" * 70)
        print("GPU vs CPU COMPARISON")
        print("=" * 70)
        if isinstance(gpu_results.get('gpu'), str):
            print(f"GPU: {gpu_results['gpu']}")
        else:
            print(f"\n{'Detector':<25} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
            print("-" * 60)
            for name in gpu_results['cpu']:
                cpu_time = gpu_results['cpu'].get(name, 0)
                gpu_time = gpu_results.get('gpu', {}).get(name, 0) if isinstance(gpu_results.get('gpu'), dict) else 0
                speedup = gpu_results.get('speedup', {}).get(name, 'N/A')
                if isinstance(speedup, float):
                    print(f"{name:<25} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.2f}x")
                else:
                    print(f"{name:<25} {cpu_time:<12.2f} {'N/A':<12} {speedup}")

    # Save results
    if args.output:
        output_data = {
            'detector_results': detector_results,
            'ensemble_results': {
                k: v for k, v in (ensemble_results or {}).items()
                if k != 'profile_stats'
            },
            'gpu_comparison': gpu_results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
