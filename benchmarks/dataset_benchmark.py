#!/usr/bin/env python3
"""
Benchmarking suite for forensic datasets.

Supports standard forensic datasets:
- CASIA v1.0 and v2.0 (tampered image detection)
- Columbia Uncompressed Image Splicing Detection
- COVERAGE (copy-move forgery detection)
- Custom datasets

Usage:
    python -m benchmarks.dataset_benchmark --dataset casia --data-dir /path/to/casia

Environment variables:
    PRIORPATCH_CASIA_PATH: Path to CASIA dataset
    PRIORPATCH_COLUMBIA_PATH: Path to Columbia dataset
    PRIORPATCH_COVERAGE_PATH: Path to COVERAGE dataset
"""

"""
Benchmarking suite for forensic datasets.

Run with: python -m benchmarks.dataset_benchmark --dataset casia --data-dir /path/to/casia
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np

from priorpatch import Ensemble, load_image
from priorpatch.detectors.registry import DETECTOR_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results for a single image."""
    image_path: str
    ground_truth: str  # 'authentic' or 'tampered'
    predicted_score: float
    per_detector_scores: Dict[str, float]
    processing_time: float


@dataclass
class DatasetMetrics:
    """Aggregated metrics for a dataset."""
    dataset_name: str
    total_images: int
    authentic_count: int
    tampered_count: int

    # Classification metrics (at optimal threshold)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # ROC metrics
    auc_roc: float = 0.0
    optimal_threshold: float = 0.5

    # Per-detector metrics
    detector_auc: Dict[str, float] = field(default_factory=dict)
    detector_weights_suggestion: Dict[str, float] = field(default_factory=dict)

    # Timing
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0


class DatasetLoader:
    """Base class for loading forensic datasets."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    def get_images(self) -> List[Tuple[str, str]]:
        """Return list of (image_path, label) tuples.

        Label is 'authentic' or 'tampered'.
        """
        raise NotImplementedError


class CASIALoader(DatasetLoader):
    """Loader for CASIA v1.0 and v2.0 datasets.

    Expected structure:
        data_dir/
            Au/  (or Authentic/)
                *.jpg, *.tif, etc.
            Tp/  (or Tampered/)
                *.jpg, *.tif, etc.
    """

    def get_images(self) -> List[Tuple[str, str]]:
        images = []

        # Try different naming conventions
        authentic_dirs = ['Au', 'Authentic', 'authentic', 'au']
        tampered_dirs = ['Tp', 'Tampered', 'tampered', 'tp', 'Sp', 'Spliced']

        auth_dir = None
        tamp_dir = None

        for name in authentic_dirs:
            if (self.data_dir / name).exists():
                auth_dir = self.data_dir / name
                break

        for name in tampered_dirs:
            if (self.data_dir / name).exists():
                tamp_dir = self.data_dir / name
                break

        if auth_dir is None:
            logger.warning(f"No authentic directory found in {self.data_dir}")
        else:
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.bmp']:
                for img_path in auth_dir.glob(ext):
                    images.append((str(img_path), 'authentic'))
                for img_path in auth_dir.glob(ext.upper()):
                    images.append((str(img_path), 'authentic'))

        if tamp_dir is None:
            logger.warning(f"No tampered directory found in {self.data_dir}")
        else:
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.bmp']:
                for img_path in tamp_dir.glob(ext):
                    images.append((str(img_path), 'tampered'))
                for img_path in tamp_dir.glob(ext.upper()):
                    images.append((str(img_path), 'tampered'))

        return images


class ColumbiaLoader(DatasetLoader):
    """Loader for Columbia Uncompressed Image Splicing Detection dataset.

    Expected structure:
        data_dir/
            4cam_auth/  (authentic images from 4 cameras)
            4cam_splc/  (spliced images)
    """

    def get_images(self) -> List[Tuple[str, str]]:
        images = []

        auth_dir = self.data_dir / '4cam_auth'
        splc_dir = self.data_dir / '4cam_splc'

        if auth_dir.exists():
            for ext in ['*.tif', '*.TIF', '*.bmp', '*.BMP']:
                for img_path in auth_dir.glob(ext):
                    images.append((str(img_path), 'authentic'))
        else:
            logger.warning(f"Authentic directory not found: {auth_dir}")

        if splc_dir.exists():
            for ext in ['*.tif', '*.TIF', '*.bmp', '*.BMP']:
                for img_path in splc_dir.glob(ext):
                    images.append((str(img_path), 'tampered'))
        else:
            logger.warning(f"Spliced directory not found: {splc_dir}")

        return images


class COVERAGELoader(DatasetLoader):
    """Loader for COVERAGE copy-move forgery dataset.

    Expected structure:
        data_dir/
            image/  (forged images)
            mask/   (ground truth masks)
    """

    def get_images(self) -> List[Tuple[str, str]]:
        images = []

        image_dir = self.data_dir / 'image'

        if image_dir.exists():
            for ext in ['*.tif', '*.TIF', '*.png', '*.PNG']:
                for img_path in image_dir.glob(ext):
                    # All images in COVERAGE are forged
                    images.append((str(img_path), 'tampered'))
        else:
            logger.warning(f"Image directory not found: {image_dir}")

        return images


class CustomLoader(DatasetLoader):
    """Loader for custom datasets.

    Expected structure:
        data_dir/
            authentic/
                *.jpg, *.png, etc.
            tampered/  (or fake/, manipulated/, forged/)
                *.jpg, *.png, etc.

    Or with a labels.json file:
        data_dir/
            labels.json  # {"image1.jpg": "authentic", "image2.jpg": "tampered"}
            *.jpg, *.png, etc.
    """

    def get_images(self) -> List[Tuple[str, str]]:
        images = []

        # Check for labels.json
        labels_file = self.data_dir / 'labels.json'
        if labels_file.exists():
            with open(labels_file) as f:
                labels = json.load(f)
            for img_name, label in labels.items():
                img_path = self.data_dir / img_name
                if img_path.exists():
                    images.append((str(img_path), label))
            return images

        # Otherwise, use directory structure
        auth_names = ['authentic', 'real', 'original', 'genuine']
        tamp_names = ['tampered', 'fake', 'manipulated', 'forged', 'synthetic', 'ai']

        for name in auth_names:
            auth_dir = self.data_dir / name
            if auth_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.bmp', '*.webp']:
                    for img_path in auth_dir.rglob(ext):
                        images.append((str(img_path), 'authentic'))
                    for img_path in auth_dir.rglob(ext.upper()):
                        images.append((str(img_path), 'authentic'))

        for name in tamp_names:
            tamp_dir = self.data_dir / name
            if tamp_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.bmp', '*.webp']:
                    for img_path in tamp_dir.rglob(ext):
                        images.append((str(img_path), 'tampered'))
                    for img_path in tamp_dir.rglob(ext.upper()):
                        images.append((str(img_path), 'tampered'))

        return images


def get_loader(dataset_name: str, data_dir: str) -> DatasetLoader:
    """Get appropriate loader for dataset."""
    loaders = {
        'casia': CASIALoader,
        'columbia': ColumbiaLoader,
        'coverage': COVERAGELoader,
        'custom': CustomLoader,
    }

    loader_cls = loaders.get(dataset_name.lower())
    if loader_cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")

    return loader_cls(data_dir)


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Compute ROC AUC and optimal threshold.

    Returns:
        Tuple of (auc, optimal_threshold)
    """
    # Sort by score
    sorted_indices = np.argsort(y_scores)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # Get unique thresholds
    thresholds = np.unique(y_scores_sorted)

    tpr_list = []
    fpr_list = []

    total_positive = np.sum(y_true)
    total_negative = len(y_true) - total_positive

    if total_positive == 0 or total_negative == 0:
        return 0.5, 0.5

    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        predictions = (y_scores >= threshold).astype(int)

        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))

        tpr = tp / total_positive if total_positive > 0 else 0
        fpr = fp / total_negative if total_negative > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

        # Compute F1 for optimal threshold selection
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Compute AUC using trapezoidal rule
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    # Sort by FPR
    sorted_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sorted_idx]
    tpr_sorted = tpr_arr[sorted_idx]

    auc = np.trapz(tpr_sorted, fpr_sorted)

    return float(auc), float(best_threshold)


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> Tuple[float, float, float, float]:
    """Compute accuracy, precision, recall, F1 at given threshold."""
    predictions = (y_scores >= threshold).astype(int)

    tp = np.sum((predictions == 1) & (y_true == 1))
    tn = np.sum((predictions == 0) & (y_true == 0))
    fp = np.sum((predictions == 1) & (y_true == 0))
    fn = np.sum((predictions == 0) & (y_true == 1))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1


def suggest_weights(
    detector_aucs: Dict[str, float],
    current_weights: Dict[str, float]
) -> Dict[str, float]:
    """Suggest detector weights based on AUC performance.

    Detectors with higher AUC get higher weights.
    """
    if not detector_aucs:
        return current_weights

    # Normalize AUCs to weights
    min_auc = min(detector_aucs.values())
    max_auc = max(detector_aucs.values())

    if max_auc - min_auc < 0.01:
        # All similar performance, use equal weights
        return {name: 1.0 for name in detector_aucs}

    suggested = {}
    for name, auc in detector_aucs.items():
        # Map AUC 0.5-1.0 to weight 0.5-2.0
        # AUC < 0.5 means detector is inverted, give it low weight
        if auc < 0.5:
            suggested[name] = 0.5
        else:
            normalized = (auc - 0.5) / 0.5  # 0 to 1
            suggested[name] = 0.5 + 1.5 * normalized  # 0.5 to 2.0

    return suggested


def run_benchmark(
    ensemble: Ensemble,
    images: List[Tuple[str, str]],
    dataset_name: str,
    max_images: Optional[int] = None,
    patch_size: int = 64,
    stride: int = 32
) -> DatasetMetrics:
    """Run benchmark on dataset.

    Args:
        ensemble: Ensemble instance
        images: List of (image_path, label) tuples
        dataset_name: Name of dataset
        max_images: Maximum images to process (None for all)
        patch_size: Patch size for analysis
        stride: Stride for patch extraction

    Returns:
        DatasetMetrics with results
    """
    if max_images:
        images = images[:max_images]

    results: List[BenchmarkResult] = []
    detector_names = [d.name for d in ensemble.detectors]

    # Collect per-detector scores for AUC computation
    detector_scores: Dict[str, List[float]] = defaultdict(list)

    logger.info(f"Processing {len(images)} images...")

    try:
        from tqdm import tqdm
        image_iter = tqdm(images, desc="Benchmarking")
    except ImportError:
        image_iter = images

    for img_path, label in image_iter:
        try:
            img = load_image(img_path)

            start_time = time.time()
            analysis = ensemble.score_image(
                img,
                patch_size=patch_size,
                stride=stride,
                return_individual=True
            )
            processing_time = time.time() - start_time

            # Get combined score (max or mean of heatmap)
            combined_score = float(np.mean(analysis.combined))

            # Get per-detector scores
            per_detector = {}
            for name in detector_names:
                if analysis.individual and name in analysis.individual:
                    per_detector[name] = float(np.mean(analysis.individual[name]))
                    detector_scores[name].append(per_detector[name])

            results.append(BenchmarkResult(
                image_path=img_path,
                ground_truth=label,
                predicted_score=combined_score,
                per_detector_scores=per_detector,
                processing_time=processing_time
            ))

        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
            continue

    if not results:
        logger.error("No images were successfully processed")
        return DatasetMetrics(
            dataset_name=dataset_name,
            total_images=len(images),
            authentic_count=sum(1 for _, l in images if l == 'authentic'),
            tampered_count=sum(1 for _, l in images if l == 'tampered')
        )

    # Compute metrics
    y_true = np.array([1 if r.ground_truth == 'tampered' else 0 for r in results])
    y_scores = np.array([r.predicted_score for r in results])

    auc, optimal_threshold = compute_roc_auc(y_true, y_scores)
    accuracy, precision, recall, f1 = compute_metrics_at_threshold(
        y_true, y_scores, optimal_threshold
    )

    # Per-detector AUC
    detector_auc = {}
    for name, scores in detector_scores.items():
        if len(scores) == len(y_true):
            det_auc, _ = compute_roc_auc(y_true, np.array(scores))
            detector_auc[name] = det_auc

    # Suggest weights
    current_weights = ensemble.weights
    suggested_weights = suggest_weights(detector_auc, current_weights)

    # Timing stats
    processing_times = [r.processing_time for r in results]

    return DatasetMetrics(
        dataset_name=dataset_name,
        total_images=len(results),
        authentic_count=sum(1 for r in results if r.ground_truth == 'authentic'),
        tampered_count=sum(1 for r in results if r.ground_truth == 'tampered'),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        auc_roc=auc,
        optimal_threshold=optimal_threshold,
        detector_auc=detector_auc,
        detector_weights_suggestion=suggested_weights,
        avg_processing_time=np.mean(processing_times),
        total_processing_time=sum(processing_times)
    )


def print_results(metrics: DatasetMetrics) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS: {metrics.dataset_name}")
    print("=" * 60)

    print(f"\nDataset Statistics:")
    print(f"  Total images:    {metrics.total_images}")
    print(f"  Authentic:       {metrics.authentic_count}")
    print(f"  Tampered:        {metrics.tampered_count}")

    print(f"\nClassification Metrics (threshold={metrics.optimal_threshold:.3f}):")
    print(f"  Accuracy:        {metrics.accuracy:.4f}")
    print(f"  Precision:       {metrics.precision:.4f}")
    print(f"  Recall:          {metrics.recall:.4f}")
    print(f"  F1 Score:        {metrics.f1_score:.4f}")
    print(f"  AUC-ROC:         {metrics.auc_roc:.4f}")

    print(f"\nPer-Detector AUC:")
    sorted_detectors = sorted(metrics.detector_auc.items(), key=lambda x: x[1], reverse=True)
    for name, auc in sorted_detectors:
        print(f"  {name:30s} {auc:.4f}")

    print(f"\nSuggested Weights (based on AUC):")
    sorted_weights = sorted(metrics.detector_weights_suggestion.items(), key=lambda x: x[1], reverse=True)
    for name, weight in sorted_weights:
        print(f"  {name:30s} {weight:.2f}")

    print(f"\nTiming:")
    print(f"  Avg per image:   {metrics.avg_processing_time:.2f}s")
    print(f"  Total time:      {metrics.total_processing_time:.1f}s")

    print("=" * 60 + "\n")


def save_results(metrics: DatasetMetrics, output_path: str) -> None:
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(asdict(metrics), f, indent=2)
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark PriorPatch on forensic datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset', '-d',
        choices=['casia', 'columbia', 'coverage', 'custom'],
        default='custom',
        help='Dataset type'
    )

    parser.add_argument(
        '--data-dir', '-i',
        type=str,
        help='Path to dataset directory. Can also use env vars: '
             'PRIORPATCH_CASIA_PATH, PRIORPATCH_COLUMBIA_PATH, PRIORPATCH_COVERAGE_PATH'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/detectors.json',
        help='Path to detector config'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results'
    )

    parser.add_argument(
        '--max-images', '-n',
        type=int,
        help='Maximum number of images to process'
    )

    parser.add_argument(
        '--patch-size', '-p',
        type=int,
        default=64,
        help='Patch size for analysis'
    )

    parser.add_argument(
        '--stride', '-s',
        type=int,
        default=32,
        help='Stride for patch extraction'
    )

    args = parser.parse_args()

    # Determine data directory
    data_dir = args.data_dir
    if data_dir is None:
        env_vars = {
            'casia': 'PRIORPATCH_CASIA_PATH',
            'columbia': 'PRIORPATCH_COLUMBIA_PATH',
            'coverage': 'PRIORPATCH_COVERAGE_PATH',
        }
        env_var = env_vars.get(args.dataset)
        if env_var:
            data_dir = os.environ.get(env_var)

        if data_dir is None:
            parser.error(
                f"--data-dir is required (or set {env_var} environment variable)"
            )

    # Load ensemble
    logger.info(f"Loading ensemble from {args.config}")
    ensemble = Ensemble.from_config(args.config)
    logger.info(f"Loaded {len(ensemble.detectors)} detectors")

    # Load dataset
    logger.info(f"Loading {args.dataset} dataset from {data_dir}")
    loader = get_loader(args.dataset, data_dir)
    images = loader.get_images()
    logger.info(f"Found {len(images)} images")

    if not images:
        logger.error("No images found in dataset")
        sys.exit(1)

    # Run benchmark
    metrics = run_benchmark(
        ensemble=ensemble,
        images=images,
        dataset_name=args.dataset,
        max_images=args.max_images,
        patch_size=args.patch_size,
        stride=args.stride
    )

    # Print results
    print_results(metrics)

    # Save results
    if args.output:
        save_results(metrics, args.output)
    else:
        # Default output path - ensure directory exists
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{args.dataset}_{int(time.time())}.json"
        save_results(metrics, str(output_path))


if __name__ == '__main__':
    main()
