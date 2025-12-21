"""
Image loading, heatmap saving, and other helpers.
"""

import os
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    'rgb_to_luminance',
    'load_image',
    'save_heatmap',
    'validate_path',
    'LUMINANCE_R',
    'LUMINANCE_G',
    'LUMINANCE_B',
]

# ITU-R BT.601 luminance coefficients (standard for SDTV)
LUMINANCE_R = 0.299
LUMINANCE_G = 0.587
LUMINANCE_B = 0.114


def rgb_to_luminance(patch: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale using standard BT.601 weights."""
    R = patch[..., 0].astype(np.float32)
    G = patch[..., 1].astype(np.float32)
    B = patch[..., 2].astype(np.float32)
    return LUMINANCE_R * R + LUMINANCE_G * G + LUMINANCE_B * B


def load_image(path: str) -> np.ndarray:
    """Load image file, convert to RGB numpy array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    try:
        im = Image.open(path)
        
        # Convert to RGB (handles RGBA, grayscale, etc.)
        if im.mode != 'RGB':
            logger.debug(f"Converting image from {im.mode} to RGB")
            im = im.convert('RGB')
        
        arr = np.array(im)
        logger.debug(f"Loaded image: {arr.shape}, dtype={arr.dtype}")
        return arr
        
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise IOError(f"Cannot load image from {path}: {e}")


def save_heatmap(heatmap: np.ndarray, image: np.ndarray, outpath: str,
                 alpha: float = 0.45, cmap: str = 'jet') -> None:
    """Overlay heatmap on image and save as PNG."""
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, got shape {heatmap.shape}")
    
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")
    
    # Ensure output directory exists
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    
    H, W = image.shape[:2]
    hm_H, hm_W = heatmap.shape
    
    # Upsample heatmap to match image size
    scale_y = max(1, H // hm_H)
    scale_x = max(1, W // hm_W)
    
    # Use Kronecker product for nearest-neighbor upsampling
    up = np.kron(heatmap, np.ones((scale_y, scale_x)))
    up = up[:H, :W]  # Crop to exact image size
    
    # Create visualization
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image.astype('uint8'))
        im = ax.imshow(up, alpha=alpha, cmap=cmap, vmin=0, vmax=1)
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Anomaly Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Heatmap saved to: {outpath}")
        
    except Exception as e:
        logger.error(f"Failed to save heatmap: {e}")
        raise IOError(f"Cannot save heatmap to {outpath}: {e}")


def validate_path(path: str, must_exist: bool = False, base_dir: str = None) -> Path:
    """
    Check path is valid. If base_dir given, ensures path doesn't escape it.
    """
    if not path:
        raise ValueError("Path cannot be empty")

    try:
        p = Path(path).resolve()
    except Exception as e:
        raise ValueError(f"Invalid path: {path}") from e

    # If base_dir specified, ensure resolved path is within it
    if base_dir is not None:
        base = Path(base_dir).resolve()
        try:
            p.relative_to(base)
        except ValueError:
            raise ValueError(f"Path traversal detected: {path} escapes {base_dir}")

    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    return p
