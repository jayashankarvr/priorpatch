"""
Basic usage example for PriorPatch.

This script demonstrates the simplest way to use PriorPatch
to analyze an image for potential manipulation.
"""

import os
import sys
from priorpatch import Ensemble, load_image, save_heatmap

def main():
    # Detect if running from examples/ or project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Set up paths
    if os.path.exists('examples/sample_input.png'):
        # Running from project root
        img_path = 'examples/sample_input.png'
        config_path = 'config/detectors.json'
        output_path = 'examples/output_heatmap.png'
    elif os.path.exists('sample_input.png'):
        # Running from examples directory
        img_path = 'sample_input.png'
        config_path = '../config/detectors.json'
        output_path = 'output_heatmap.png'
    else:
        print("Error: Cannot find sample_input.png")
        print("Run this script from either:")
        print("  - Project root: python examples/basic_usage.py")
        print("  - Examples dir: cd examples && python basic_usage.py")
        sys.exit(1)
    
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        print("Make sure you're running from the project root or examples directory")
        sys.exit(1)
    
    # Step 1: Load the image
    print("Loading image...")
    img = load_image(img_path)
    print(f"Image shape: {img.shape}")
    
    # Step 2: Create ensemble from configuration
    print("\nInitializing ensemble...")
    ensemble = Ensemble.from_config(config_path)
    print(f"Loaded {len(ensemble.detectors)} detectors")
    
    # Step 3: Analyze the image
    print("\nAnalyzing image...")
    heatmap = ensemble.score_image(img, patch_size=64, stride=32)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Max anomaly score: {heatmap.max():.4f}")
    print(f"Min anomaly score: {heatmap.min():.4f}")
    print(f"Mean anomaly score: {heatmap.mean():.4f}")
    
    # Step 4: Save visualization
    print("\nSaving heatmap...")
    save_heatmap(heatmap, img, output_path)
    print(f"Done! Check {output_path}")

if __name__ == '__main__':
    main()
