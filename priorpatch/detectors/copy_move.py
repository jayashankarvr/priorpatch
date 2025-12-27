"""
Copy-Move Forgery Detection.

Copy-move is when a region of an image is copied and pasted elsewhere
in the same image, usually to hide or duplicate objects.

This detector finds similar regions within the image using:
1. Block matching with DCT features
2. Memory-efficient chunked similarity computation

Reference: Fridrich et al., "Detection of Copy-Move Forgery in Digital Images" (2003)
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
from scipy.fftpack import dct

from priorpatch.detectors.base import DetectorInterface
from priorpatch.detectors.registry import register_detector
from priorpatch.utils import rgb_to_luminance

logger = logging.getLogger(__name__)


def extract_dct_features(
    image: np.ndarray,
    block_size: int = 16,
    stride: int = 4
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extract DCT-based features from overlapping blocks.

    Args:
        image: Grayscale image
        block_size: Size of blocks to extract
        stride: Step between blocks

    Returns:
        Tuple of (feature_matrix, block_positions)
    """
    h, w = image.shape

    features = []
    positions = []

    for y in range(0, h - block_size + 1, stride):
        for x in range(0, w - block_size + 1, stride):
            block = image[y:y+block_size, x:x+block_size]

            # Compute 2D DCT
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

            # Use low-frequency coefficients as features
            # Zigzag scan of top-left 8x8
            feature = []
            for i in range(8):
                for j in range(8 - i):
                    if i + j < 8:
                        feature.append(dct_block[i, j])

            features.append(feature)
            positions.append((y, x))

    return np.array(features), positions


def find_similar_blocks(
    features: np.ndarray,
    positions: List[Tuple[int, int]],
    similarity_threshold: float = 0.95,
    min_distance: int = 32,
    chunk_size: int = 500,
    max_matches: int = 1000
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """Find similar blocks that are spatially separated.

    Uses chunked computation for memory efficiency.
    Returns list of ((pos1), (pos2), similarity) tuples.
    """
    n_blocks = len(features)

    if n_blocks < 2:
        return []

    # Normalize features once
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features_norm = features / norms

    matches = []

    # Process in chunks to limit memory usage
    # Instead of N×N matrix, we compute chunk_size×N at a time
    for i_start in range(0, n_blocks, chunk_size):
        if len(matches) >= max_matches:
            break

        i_end = min(i_start + chunk_size, n_blocks)
        chunk = features_norm[i_start:i_end]

        # Only compare with blocks after this chunk (avoid duplicates)
        j_start = i_start
        remaining = features_norm[j_start:]

        # Compute similarities for this chunk: (chunk_size × remaining_blocks)
        similarities = np.dot(chunk, remaining.T)

        # Find matches above threshold
        for i_local in range(similarities.shape[0]):
            if len(matches) >= max_matches:
                break

            i_global = i_start + i_local

            for j_local in range(i_local + 1, similarities.shape[1]):
                j_global = j_start + j_local

                if similarities[i_local, j_local] >= similarity_threshold:
                    pos_i = positions[i_global]
                    pos_j = positions[j_global]

                    spatial_dist = np.sqrt(
                        (pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2
                    )

                    if spatial_dist >= min_distance:
                        matches.append((pos_i, pos_j, float(similarities[i_local, j_local])))

                        if len(matches) >= max_matches:
                            break

    return matches


def cluster_matches(
    matches: List[Tuple[Tuple[int, int], Tuple[int, int], float]],
    distance_threshold: int = 20
) -> List[List[Tuple[Tuple[int, int], Tuple[int, int], float]]]:
    """Cluster matches by spatial proximity.

    Copy-move regions have many nearby matches with consistent offset.

    Args:
        matches: List of matches
        distance_threshold: Max distance to consider same cluster

    Returns:
        List of match clusters
    """
    if not matches:
        return []

    # Compute offset for each match
    offsets = []
    for pos1, pos2, sim in matches:
        offset = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        offsets.append(offset)

    # Simple clustering: group by similar offset
    clusters = []
    used = [False] * len(matches)

    for i, (match, offset) in enumerate(zip(matches, offsets)):
        if used[i]:
            continue

        cluster = [match]
        used[i] = True

        for j in range(i + 1, len(matches)):
            if used[j]:
                continue

            other_offset = offsets[j]
            offset_dist = np.sqrt(
                (offset[0] - other_offset[0])**2 +
                (offset[1] - other_offset[1])**2
            )

            if offset_dist < distance_threshold:
                cluster.append(matches[j])
                used[j] = True

        clusters.append(cluster)

    return clusters


def analyze_copy_move(
    image: np.ndarray,
    block_size: int = 16,
    stride: int = 4,
    similarity_threshold: float = 0.95,
    min_cluster_size: int = 5
) -> dict:
    """Analyze image for copy-move forgery.

    Args:
        image: Grayscale image
        block_size: Block size for feature extraction
        stride: Step between blocks
        similarity_threshold: Threshold for block matching
        min_cluster_size: Minimum matches to consider a cluster valid

    Returns:
        Dictionary of copy-move detection results
    """
    h, w = image.shape

    # Extract features
    features, positions = extract_dct_features(image, block_size, stride)

    if len(features) < 10:
        return {
            'detected': False,
            'num_matches': 0,
            'num_clusters': 0,
            'max_cluster_size': 0,
            'coverage': 0.0
        }

    # Find similar blocks
    matches = find_similar_blocks(
        features, positions,
        similarity_threshold=similarity_threshold,
        min_distance=block_size * 2
    )

    # Cluster matches
    clusters = cluster_matches(matches)

    # Filter to significant clusters
    significant_clusters = [c for c in clusters if len(c) >= min_cluster_size]

    # Compute coverage (approximate area affected)
    affected_positions = set()
    for cluster in significant_clusters:
        for pos1, pos2, _ in cluster:
            affected_positions.add(pos1)
            affected_positions.add(pos2)

    coverage = len(affected_positions) * (block_size**2) / (h * w)

    return {
        'detected': len(significant_clusters) > 0,
        'num_matches': len(matches),
        'num_clusters': len(significant_clusters),
        'max_cluster_size': max(len(c) for c in significant_clusters) if significant_clusters else 0,
        'coverage': float(coverage),
        'total_affected_blocks': len(affected_positions)
    }


@register_detector
class CopyMoveDetector(DetectorInterface):
    """Copy-move forgery detector.

    Detects when a region of an image has been copied and pasted
    elsewhere in the same image. Uses DCT-based block matching
    to find similar regions.

    Score interpretation:
    - 0.0: No similar regions found - likely authentic
    - 1.0: Strong copy-move evidence - likely forged

    Best for:
    - Object duplication (copying objects to duplicate them)
    - Object removal (copying background to hide something)
    - Traditional image manipulation without AI

    Limitations:
    - Computationally intensive for large images
    - Can miss small copy-move regions
    - May false positive on legitimate repeated textures
    """

    name = 'copy_move'

    def __init__(
        self,
        block_size: int = 16,
        stride: int = 8,
        similarity_threshold: float = 0.92,
        min_cluster_size: int = 4
    ):
        """
        Args:
            block_size: Size of blocks for matching
            stride: Step between blocks
            similarity_threshold: Cosine similarity threshold (0.9-0.98)
            min_cluster_size: Minimum matches to consider significant
        """
        self.block_size = block_size
        self.stride = stride
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

    def get_config(self) -> dict:
        """Serialize for multiprocessing."""
        return {
            'block_size': self.block_size,
            'stride': self.stride,
            'similarity_threshold': self.similarity_threshold,
            'min_cluster_size': self.min_cluster_size
        }

    def score(self, patch: np.ndarray) -> float:
        """Score patch for copy-move forgery.

        Args:
            patch: RGB image patch (H, W, 3)

        Returns:
            Anomaly score (0 = no copy-move, 1 = strong evidence)
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            return 0.0

        h, w = patch.shape[:2]
        min_size = self.block_size * 4

        if h < min_size or w < min_size:
            return 0.0

        try:
            # Convert to grayscale using standard BT.601 weights
            gray = rgb_to_luminance(patch).astype(np.float64)

            # Analyze for copy-move
            results = analyze_copy_move(
                gray,
                block_size=self.block_size,
                stride=self.stride,
                similarity_threshold=self.similarity_threshold,
                min_cluster_size=self.min_cluster_size
            )

            if not results['detected']:
                return 0.0

            scores = []

            # Score 1: Based on number of clusters
            num_clusters = results['num_clusters']
            cluster_score = min(num_clusters / 3.0, 1.0)  # 3+ clusters = max score
            scores.append(cluster_score)

            # Score 2: Based on coverage
            coverage = results['coverage']
            coverage_score = min(coverage * 10, 1.0)  # 10% coverage = max score
            scores.append(coverage_score)

            # Score 3: Based on cluster size
            max_cluster = results['max_cluster_size']
            size_score = min(max_cluster / 20.0, 1.0)  # 20+ matches = max score
            scores.append(size_score)

            # Combine with emphasis on coverage (most reliable indicator)
            weights = [0.3, 0.4, 0.3]
            final_score = sum(s * w for s, w in zip(scores, weights))

            return float(np.clip(final_score, 0.0, 1.0))

        except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as e:
            logger.warning(f"CopyMoveDetector failed: {e}")
            return 0.0
