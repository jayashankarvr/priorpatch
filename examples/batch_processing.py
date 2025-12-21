"""
Example: Batch processing multiple images.

This script demonstrates how to process multiple images
efficiently and generate a report.
"""

import json
from pathlib import Path
from priorpatch import Ensemble, load_image, save_heatmap

def analyze_batch(image_paths, output_dir='batch_results'):
    """Analyze multiple images and generate reports."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize ensemble once (reuse for all images)
    print("Initializing ensemble...")
    ensemble = Ensemble.from_config('../config/detectors.json')
    
    results = []
    
    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")
        
        try:
            # Load and analyze
            img = load_image(img_path)
            heatmap = ensemble.score_image(img, patch_size=64, stride=32)
            
            # Calculate statistics
            stats = {
                'filename': Path(img_path).name,
                'image_shape': list(img.shape),
                'max_score': float(heatmap.max()),
                'mean_score': float(heatmap.mean()),
                'suspicious_patches': int((heatmap > 0.7).sum()),
                'total_patches': int(heatmap.size)
            }
            
            # Save heatmap
            output_name = f"{Path(img_path).stem}_heatmap.png"
            save_heatmap(heatmap, img, str(output_path / output_name))
            
            results.append(stats)
            print(f"  Max score: {stats['max_score']:.4f}")
            print(f"  Suspicious patches: {stats['suspicious_patches']}/{stats['total_patches']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'filename': Path(img_path).name, 'error': str(e)})
    
    # Save summary report
    report_path = output_path / 'batch_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Batch processing complete.")
    print(f"  Results saved to: {output_dir}/")
    print(f"  Summary report: {report_path}")
    
    return results

def main():
    # Example: Process all PNG images in a directory
    image_paths = ['sample_input.png']  # Add more image paths here
    
    results = analyze_batch(image_paths)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for result in results:
        if 'error' not in result:
            print(f"{result['filename']}: max_score={result['max_score']:.4f}")
        else:
            print(f"{result['filename']}: ERROR")

if __name__ == '__main__':
    main()
