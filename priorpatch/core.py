"""
Main ensemble class - runs detectors on patches and combines scores.
"""

import json
import logging
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from priorpatch.detectors.registry import DETECTOR_REGISTRY
from priorpatch.detectors.base import DetectorInterface

logger = logging.getLogger(__name__)

# Try to import tqdm for progress bars (optional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# Try to import importlib.resources for Python 3.9+
try:
    if sys.version_info >= (3, 9):
        from importlib.resources import files
    else:
        from importlib_resources import files
    HAS_IMPORTLIB_RESOURCES = True
except ImportError:
    HAS_IMPORTLIB_RESOURCES = False
    files = None

# Constants
MIN_PATCH_SIZE = 8
MAX_PATCH_SIZE = 1024
MIN_STRIDE = 1
NORMALIZATION_EPSILON = 1e-8  # Use smaller epsilon for float32 precision
MIN_PATCHES_FOR_MULTIPROCESSING = 10
MAX_DETECTOR_FAILURE_RATE = 0.5
SUPPORTED_CONFIG_VERSIONS = {'2.0', '2.1'}  # Supported config file versions


def _find_config_file(config_path: str = 'config/detectors.json') -> str:
    """
    Find config file, checking multiple locations in this order:
    1. Absolute path (if provided)
    2. Relative to current working directory
    3. Relative to package installation directory
    4. Inside installed package data

    Args:
        config_path: Path to config file (can be absolute or relative)

    Returns:
        Absolute path to config file

    Raises:
        FileNotFoundError: If config file cannot be found
    """
    # If absolute path provided, use it directly
    if os.path.isabs(config_path):
        if os.path.exists(config_path):
            return config_path
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Try relative to current working directory
    if os.path.exists(config_path):
        return os.path.abspath(config_path)

    # Try relative to package installation directory
    package_dir = Path(__file__).parent.parent
    package_config = package_dir / config_path
    if package_config.exists():
        return str(package_config)

    # Try using importlib.resources for installed package
    if HAS_IMPORTLIB_RESOURCES and files is not None:
        try:
            # For Python 3.9+, use importlib.resources.files
            config_file = files('priorpatch').parent / config_path
            if config_file.exists():
                return str(config_file)
        except Exception as e:
            logger.debug(f"Could not locate config via importlib.resources: {e}")

    # Last resort: check in site-packages
    try:
        import priorpatch
        priorpatch_path = Path(priorpatch.__file__).parent.parent
        fallback_config = priorpatch_path / config_path
        if fallback_config.exists():
            return str(fallback_config)
    except Exception:
        pass

    raise FileNotFoundError(
        f"Config file '{config_path}' not found. Searched in:\n"
        f"  - Current directory: {os.path.abspath(config_path)}\n"
        f"  - Package directory: {package_config}\n"
        f"Please ensure the config file exists or provide an absolute path."
    )


# Global variable for worker process (set by initializer)
_worker_detectors = None
_worker_detector_names = None
_worker_weights = None


def _init_worker(detector_classes: List, detector_names: List[str], weights: Dict[str, float]):
    """Initialize worker process with detector instances.

    This is called once per worker process to avoid re-creating detectors
    for every patch (major performance improvement).
    """
    global _worker_detectors, _worker_detector_names, _worker_weights
    _worker_detectors = [cls() for cls in detector_classes]
    _worker_detector_names = detector_names
    _worker_weights = weights


def _worker_score_patch(img_patch: np.ndarray) -> Tuple[float, List[float], List[Tuple[str, str]]]:
    """Multiprocessing worker. Has to be module-level to be picklable.

    Uses detector instances created by _init_worker (once per process)
    instead of creating new ones for each patch.
    """
    global _worker_detectors, _worker_detector_names, _worker_weights

    vals = []
    failures = []
    for i, d in enumerate(_worker_detectors):
        try:
            score = float(d.score(img_patch))
            vals.append(score)
        except Exception as e:
            vals.append(0.0)
            failures.append((_worker_detector_names[i], str(e)))

    if not vals:
        return 0.0, [], failures

    arr = np.array(vals, dtype=np.float32)
    weight_arr = np.array([_worker_weights.get(name, 1.0) for name in _worker_detector_names], dtype=np.float32)

    # Use weighted average directly - NO per-patch normalization
    # Normalization happens later at the heatmap level for visualization
    # Handle NaN values gracefully
    if np.any(np.isnan(arr)) or np.all(arr == 0.0):
        # If we have NaN or all zeros, return 0.0 as safe default
        combined = 0.0
    else:
        combined = float(np.average(arr, weights=weight_arr))

    return combined, vals, failures


@dataclass
class AnalysisResult:
    """Output from score_image() when return_individual=True."""
    combined: np.ndarray
    individual: Optional[Dict[str, np.ndarray]] = None
    detector_names: Optional[List[str]] = None


@dataclass
class PatchResult:
    """Output from score_patch() - combined score plus per-detector breakdown."""
    combined: float
    individual: List[float]
    failures: List[Tuple[str, str]]


class Ensemble:
    """Runs multiple detectors on image patches and combines the results."""
    
    def __init__(self, detectors: List[DetectorInterface], weights: Optional[Dict[str, float]] = None):
        self.detectors = detectors
        self.weights = weights or {}
        logger.debug(f"Loaded {len(detectors)} detectors")
    
    @classmethod
    def from_config(cls, path: str = None) -> 'Ensemble':
        """Load ensemble from JSON config.

        Args:
            path: Path to config file (can be absolute or relative).
                  Will search multiple locations if relative.
                  If None, uses PRIORPATCH_CONFIG_PATH env var or default.

        Returns:
            Ensemble instance with detectors loaded from config

        Raises:
            FileNotFoundError: If config file cannot be found
            json.JSONDecodeError: If config file is not valid JSON
            ValueError: If config specifies unknown detectors or invalid weights
        """
        # Check environment variable for config path
        if path is None:
            path = os.environ.get('PRIORPATCH_CONFIG_PATH', 'config/detectors.json')

        try:
            config_file = _find_config_file(path)
            logger.info(f"Loading config from: {config_file}")
            with open(config_file, 'r') as f:
                cfg = json.load(f)
        except FileNotFoundError:
            logger.error(f"Config not found: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Bad JSON in config: {e}")
            raise

        # Validate config version if present
        config_version = cfg.get('version')
        if config_version and config_version not in SUPPORTED_CONFIG_VERSIONS:
            logger.warning(
                f"Config version '{config_version}' may not be fully compatible. "
                f"Supported versions: {SUPPORTED_CONFIG_VERSIONS}"
            )

        enabled = cfg.get('enabled_detectors', [])
        weights = cfg.get('detector_weights', {})

        # Environment variable overrides for detectors (comma-separated)
        env_detectors = os.environ.get('PRIORPATCH_ENABLED_DETECTORS')
        if env_detectors:
            enabled = [d.strip() for d in env_detectors.split(',') if d.strip()]
            logger.info(f"Using detectors from environment: {enabled}")

        # Environment variable overrides for weights (JSON format)
        env_weights = os.environ.get('PRIORPATCH_WEIGHTS')
        if env_weights:
            try:
                weights = json.loads(env_weights)
                logger.info(f"Using weights from environment: {weights}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid PRIORPATCH_WEIGHTS JSON, using config file: {e}")

        # Validate weights
        for detector_name, weight in weights.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Weight for '{detector_name}' must be a number, got {type(weight).__name__}")
            if weight <= 0:
                raise ValueError(f"Weight for '{detector_name}' must be positive, got {weight}")
            if weight > 100:
                logger.warning(f"Weight for '{detector_name}' is unusually high ({weight}), consider values between 0.1-10")
        
        if not enabled:
            logger.warning("No detectors enabled")
        
        unknown_weights = set(weights.keys()) - set(enabled)
        if unknown_weights:
            logger.warning(f"Config has weights for unknown detectors: {unknown_weights}")
        
        dets = []
        for name in enabled:
            clsobj = DETECTOR_REGISTRY.get(name)
            if clsobj is None:
                available = list(DETECTOR_REGISTRY.keys())
                raise ValueError(f"Detector '{name}' not found. Available: {available}")
            dets.append(clsobj())
            logger.debug(f"Loaded: {name}")
        
        return cls(dets, weights)
    
    def score_patch(self, patch: np.ndarray) -> PatchResult:
        """Run all detectors on a single patch, return combined + individual scores."""
        if patch.size == 0:
            return PatchResult(0.0, [], [])
        
        vals = []
        failures = []
        for d in self.detectors:
            try:
                score = float(d.score(patch))
                vals.append(score)
            except Exception as e:
                logger.warning(f"Detector {d.name} failed on patch: {e}")
                vals.append(0.0)
                failures.append((d.name, str(e)))
        
        arr = np.array(vals, dtype=np.float32)
        if arr.size == 0:
            return PatchResult(0.0, [], failures)

        weight_arr = np.array([
            self.weights.get(d.name, 1.0) for d in self.detectors
        ], dtype=np.float32)

        # Use weighted average directly - NO per-patch normalization
        # Normalization happens later at the heatmap level for visualization
        # Handle NaN values gracefully
        if np.any(np.isnan(arr)) or np.all(arr == 0.0):
            # If we have NaN or all zeros, return 0.0 as safe default
            combined = 0.0
        else:
            combined = float(np.average(arr, weights=weight_arr))

        return PatchResult(combined, vals, failures)
    
    def score_image(self, img: np.ndarray, patch_size: int = 64, stride: int = 32,
                    use_multiprocessing: bool = False, n_jobs: int = -1,
                    return_individual: bool = False) -> Union[np.ndarray, AnalysisResult]:
        """
        Analyze image by scoring overlapping patches. Returns heatmap.
        Set return_individual=True to also get per-detector heatmaps.
        """
        
        # Input validation
        if not isinstance(img, np.ndarray):
            raise TypeError(f"img must be numpy array, got {type(img)}")
        
        if img.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D, got shape {img.shape}")
        
        if patch_size < MIN_PATCH_SIZE:
            raise ValueError(f"patch_size must be >= {MIN_PATCH_SIZE}, got {patch_size}")
        
        if patch_size > MAX_PATCH_SIZE:
            raise ValueError(f"patch_size must be <= {MAX_PATCH_SIZE}, got {patch_size}")
        
        if stride < MIN_STRIDE:
            raise ValueError(f"stride must be >= {MIN_STRIDE}, got {stride}")
        
        if n_jobs < -1 or n_jobs == 0:
            raise ValueError(f"n_jobs must be -1 or positive integer, got {n_jobs}")
        
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        
        H, W = img.shape[:2]
        
        if H < patch_size or W < patch_size:
            logger.warning(f"Image ({H}x{W}) smaller than patch size ({patch_size}), using full image")
            patch_size = min(H, W)
        
        ys = list(range(0, max(1, H - patch_size + 1), stride))
        xs = list(range(0, max(1, W - patch_size + 1), stride))
        
        total_patches = len(ys) * len(xs)
        logger.info(f"Analyzing {total_patches} patches")
        
        heat = np.zeros((len(ys), len(xs)), dtype=np.float32)
        
        # Track detector failures
        detector_failure_counts = {d.name: 0 for d in self.detectors}
        
        if return_individual:
            individual_heats = {d.name: np.zeros((len(ys), len(xs)), dtype=np.float32) 
                               for d in self.detectors}
        
        if use_multiprocessing and total_patches > MIN_PATCHES_FOR_MULTIPROCESSING:
            from multiprocessing import Pool, cpu_count

            if n_jobs == -1:
                n_jobs = cpu_count()

            logger.info(f"Using {n_jobs} parallel workers")

            detector_classes = [type(d) for d in self.detectors]
            detector_names = [d.name for d in self.detectors]

            # Lazy patch extraction generator - reduces peak memory usage
            def patch_generator():
                for y in ys:
                    for x in xs:
                        yield img[y:y+patch_size, x:x+patch_size].copy()

            # Use initializer to create detectors once per worker (not once per patch!)
            # Use imap with chunksize for better performance with lazy loading
            chunksize = max(1, total_patches // (n_jobs * 4))
            with Pool(
                processes=n_jobs,
                initializer=_init_worker,
                initargs=(detector_classes, detector_names, self.weights)
            ) as pool:
                if HAS_TQDM:
                    results = list(tqdm(
                        pool.imap(_worker_score_patch, patch_generator(), chunksize=chunksize),
                        total=total_patches,
                        desc="Processing patches",
                        unit="patch"
                    ))
                else:
                    results = list(pool.imap(_worker_score_patch, patch_generator(), chunksize=chunksize))
            
            idx = 0
            for i in range(len(ys)):
                for j in range(len(xs)):
                    combined, individual, failures = results[idx]
                    heat[i, j] = combined
                    
                    # Track failures
                    for detector_name, error in failures:
                        detector_failure_counts[detector_name] += 1
                    
                    if return_individual:
                        for det_idx, detector_name in enumerate(detector_names):
                            individual_heats[detector_name][i, j] = individual[det_idx]
                    idx += 1
        else:
            # Sequential processing with optional progress bar
            patches_iter = [(i, y, j, x) for i, y in enumerate(ys) for j, x in enumerate(xs)]
            if HAS_TQDM:
                patches_iter = tqdm(patches_iter, desc="Processing patches", unit="patch")
            
            for i, y, j, x in patches_iter:
                patch = img[y:y+patch_size, x:x+patch_size]
                patch_result = self.score_patch(patch)
                heat[i, j] = patch_result.combined
                
                # Track failures
                if patch_result.failures:
                    for detector_name, error in patch_result.failures:
                        detector_failure_counts[detector_name] += 1
                
                if return_individual:
                    for idx, detector_score in enumerate(patch_result.individual):
                        detector_name = self.detectors[idx].name
                        individual_heats[detector_name][i, j] = detector_score
        
        # Check for excessive detector failures
        for detector_name, failure_count in detector_failure_counts.items():
            if failure_count > 0:
                failure_rate = failure_count / total_patches
                if failure_rate > MAX_DETECTOR_FAILURE_RATE:
                    logger.error(
                        f"Detector '{detector_name}' failed on {failure_count}/{total_patches} "
                        f"patches ({failure_rate*100:.1f}%). Results may be unreliable."
                    )
                elif failure_rate > 0.1:  # Warn if >10%
                    logger.warning(
                        f"Detector '{detector_name}' failed on {failure_count}/{total_patches} "
                        f"patches ({failure_rate*100:.1f}%)"
                    )
        
        # Replace any NaN values with 0.0 before normalization
        heat = np.nan_to_num(heat, nan=0.0)

        mn, mx = heat.min(), heat.max()
        if mx - mn > NORMALIZATION_EPSILON:
            heat = (heat - mn) / (mx - mn)
        else:
            # All patches have uniform score - preserve actual value
            uniform_score = float(mn)  # or mx, they're the same
            logger.warning(f"All patches have uniform score: {uniform_score:.3f}")
            # Keep the original score as-is (it's already in 0-1 range from detectors)
            # No normalization needed - uniform detection result is meaningful
        
        if return_individual:
            for detector_name in individual_heats:
                det_heat = individual_heats[detector_name]
                mn, mx = det_heat.min(), det_heat.max()
                if mx - mn > NORMALIZATION_EPSILON:
                    individual_heats[detector_name] = (det_heat - mn) / (mx - mn)
            
            return AnalysisResult(
                combined=heat,
                individual=individual_heats,
                detector_names=[d.name for d in self.detectors]
            )
        
        return heat
