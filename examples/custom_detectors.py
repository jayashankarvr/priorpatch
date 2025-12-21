"""
Example: Using custom detector selection and weights.

This script shows how to manually select detectors and configure
custom weights for the ensemble.
"""

from priorpatch.core import Ensemble
from priorpatch.detectors.registry import get_detector_class
from priorpatch.utils import load_image, save_heatmap

def main():
    # Load image
    img = load_image('sample_input.png')
    
    # Option 1: Manually select specific detectors
    print("Creating custom ensemble...")
    detector_names = ['color_stats', 'fft_dct', 'neighbor_consistency']
    detectors = [get_detector_class(name)() for name in detector_names]
    
    # Define custom weights (higher = more important)
    weights = {
        'color_stats': 1.5,          # Emphasize color analysis
        'fft_dct': 2.0,              # Strong weight on frequency analysis
        'neighbor_consistency': 1.0  # Standard weight
    }
    
    ensemble = Ensemble(detectors, weights)
    
    # Analyze
    print("\nAnalyzing with custom configuration...")
    heatmap = ensemble.score_image(img, patch_size=64, stride=32)
    
    # Analyze a specific region
    print("\nAnalyzing specific region (100:200, 150:250)...")
    region = img[100:200, 150:250]
    score, individual_scores = ensemble.score_patch(region)
    
    print(f"Combined score: {score:.4f}")
    for detector, score_val in zip(detectors, individual_scores):
        print(f"  {detector.name}: {score_val:.4f}")
    
    # Save results
    save_heatmap(heatmap, img, 'custom_heatmap.png')
    print("\nDone! Check custom_heatmap.png")

if __name__ == '__main__':
    main()
