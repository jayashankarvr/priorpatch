"""
Example: Analyzing individual detector results.

This script demonstrates how to view and compare results from each
detector individually, in addition to the combined ensemble score.
"""

from priorpatch import Ensemble, load_image, save_heatmap
import matplotlib.pyplot as plt
import numpy as np

def analyze_with_individual_detectors():
    """Analyze image and show both individual and combined results."""
    
    print("Loading image and ensemble...")
    img = load_image('sample_input.png')
    ensemble = Ensemble.from_config('../config/detectors.json')
    
    print(f"\nAnalyzing with {len(ensemble.detectors)} detectors:")
    for detector in ensemble.detectors:
        print(f"  - {detector.name}")
    
    # Get individual detector results
    print("\nGenerating heatmaps (this may take a moment)...")
    results = ensemble.score_image(
        img, 
        patch_size=64, 
        stride=32,
        return_individual=True
    )
    
    combined = results.combined
    individual = results.individual
    detector_names = results.detector_names
    
    # Print statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nCombined Score:")
    print(f"  Max:  {combined.max():.4f}")
    print(f"  Mean: {combined.mean():.4f}")
    print(f"  Min:  {combined.min():.4f}")
    
    print(f"\nIndividual Detector Scores:")
    for name in detector_names:
        heat = individual[name]
        print(f"\n  {name}:")
        print(f"    Max:  {heat.max():.4f}")
        print(f"    Mean: {heat.mean():.4f}")
        print(f"    Min:  {heat.min():.4f}")
    
    # Save all results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save combined
    print("\nSaving combined heatmap...")
    save_heatmap(combined, img, 'output_combined.png')
    
    # Save individual
    print("\nSaving individual detector heatmaps...")
    for name in detector_names:
        filename = f'output_{name}.png'
        save_heatmap(individual[name], img, filename)
        print(f"  Saved: {filename}")
    
    # Create comparison visualization
    print("\nCreating comparison grid...")
    create_comparison_grid(img, combined, individual, detector_names)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - output_combined.png (ensemble result)")
    for name in detector_names:
        print(f"  - output_{name}.png")
    print("  - output_comparison_grid.png (all in one view)")


def create_comparison_grid(img, combined, individual, detector_names):
    """Create a grid showing all detectors + combined result."""
    
    n_detectors = len(detector_names)
    n_cols = 3
    n_rows = (n_detectors + 2) // n_cols  # +1 for combined, +1 for original
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Combined result
    axes[1].imshow(img)
    axes[1].imshow(combined, alpha=0.5, cmap='jet')
    axes[1].set_title(f'Combined (max={combined.max():.3f})', 
                      fontsize=12, fontweight='bold', color='red')
    axes[1].axis('off')
    
    # Individual detectors
    for idx, name in enumerate(detector_names):
        ax = axes[idx + 2]
        ax.imshow(img)
        heat = individual[name]
        ax.imshow(heat, alpha=0.5, cmap='jet')
        ax.set_title(f'{name}\n(max={heat.max():.3f})', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(detector_names) + 2, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('output_comparison_grid.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyze_specific_region():
    """Analyze a specific region with each detector."""
    
    print("\n" + "="*60)
    print("ANALYZING SPECIFIC REGION (100:200, 150:250)")
    print("="*60)
    
    img = load_image('sample_input.png')
    ensemble = Ensemble.from_config('../config/detectors.json')
    
    # Extract region
    region = img[100:200, 150:250]
    
    # Score with ensemble
    combined_score, individual_scores = ensemble.score_patch(region)
    
    print(f"\nCombined Score: {combined_score:.4f}")
    print(f"\nIndividual Detector Scores:")
    for detector, score in zip(ensemble.detectors, individual_scores):
        print(f"  {detector.name:25s}: {score:.4f}")
    
    # Find most suspicious detector
    max_idx = np.argmax(individual_scores)
    max_detector = ensemble.detectors[max_idx].name
    max_score = individual_scores[max_idx]
    
    print(f"\n!  Most suspicious detector: {max_detector} (score: {max_score:.4f})")


def compare_detector_agreement():
    """Analyze how much detectors agree with each other."""
    
    print("\n" + "="*60)
    print("DETECTOR AGREEMENT ANALYSIS")
    print("="*60)
    
    img = load_image('sample_input.png')
    ensemble = Ensemble.from_config('../config/detectors.json')
    
    results = ensemble.score_image(
        img, 
        patch_size=64, 
        stride=64,  # Larger stride for faster analysis
        return_individual=True
    )
    
    individual = results.individual
    detector_names = results.detector_names
    
    # Compute correlation between detectors
    print("\nDetector Correlation Matrix:")
    print("(How much do detectors agree? 1.0 = perfect agreement)")
    print()
    
    # Header
    print("Detector".ljust(25), end="")
    for name in detector_names:
        print(name[:8].ljust(10), end="")
    print()
    print("-" * (25 + 10 * len(detector_names)))
    
    # Correlation matrix
    for name1 in detector_names:
        print(name1.ljust(25), end="")
        for name2 in detector_names:
            heat1 = individual[name1].flatten()
            heat2 = individual[name2].flatten()
            corr = np.corrcoef(heat1, heat2)[0, 1]
            print(f"{corr:.3f}".ljust(10), end="")
        print()
    
    print("\nInterpretation:")
    print("  > 0.7  : High agreement (detectors see similar patterns)")
    print("  0.3-0.7: Moderate agreement (some overlap)")
    print("  < 0.3  : Low agreement (detecting different things)")


def main():
    """Run all analysis examples."""
    
    print("="*60)
    print("INDIVIDUAL DETECTOR ANALYSIS EXAMPLES")
    print("="*60)
    
    # Main analysis
    analyze_with_individual_detectors()
    
    # Region analysis
    analyze_specific_region()
    
    # Agreement analysis
    compare_detector_agreement()
    
    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
