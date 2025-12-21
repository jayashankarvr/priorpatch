"""
CLI for analyzing images.
"""

import argparse
import glob
import os
import json
import logging
from pathlib import Path
from priorpatch.core import Ensemble
from priorpatch.utils import load_image, save_heatmap
from priorpatch.gpu_backend import use_gpu, disable_gpu, get_gpu_info

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def find_images(path_or_pattern):
    """
    Find image files from a path, directory, or glob pattern.
    Returns list of image file paths.
    """
    images = []

    # Check if it's a glob pattern (contains * or ?)
    if '*' in path_or_pattern or '?' in path_or_pattern:
        for match in glob.glob(path_or_pattern, recursive=True):
            if Path(match).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(match)
    elif os.path.isdir(path_or_pattern):
        # It's a directory - find all images
        for root, _, files in os.walk(path_or_pattern):
            for f in files:
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(os.path.join(root, f))
    elif os.path.isfile(path_or_pattern):
        # Single file
        images.append(path_or_pattern)
    else:
        raise FileNotFoundError(f"Path not found: {path_or_pattern}")

    return sorted(images)


def analyze_single_image(img_path, ensemble, args, outdir):
    """
    Analyze a single image and save results.
    Returns dict with results or error info.
    """
    result = {
        'file': img_path,
        'filename': os.path.basename(img_path),
        'status': 'success'
    }

    try:
        img = load_image(img_path)
        result['image_shape'] = list(img.shape)

        if args.save_individual:
            analysis = ensemble.score_image(
                img,
                patch_size=args.patch_size,
                stride=args.stride,
                return_individual=True
            )
            heat = analysis.combined
            individual_heats = analysis.individual
            detector_names = analysis.detector_names
        else:
            heat = ensemble.score_image(
                img,
                patch_size=args.patch_size,
                stride=args.stride
            )
            individual_heats = None
            detector_names = None

        # Save combined heatmap
        heatmap_path = os.path.join(outdir, 'heatmap_combined.png')
        save_heatmap(heat, img, heatmap_path)

        # Save individual heatmaps if requested
        if args.save_individual and individual_heats:
            individual_dir = os.path.join(outdir, 'individual_detectors')
            os.makedirs(individual_dir, exist_ok=True)
            for name in detector_names:
                det_path = os.path.join(individual_dir, f'{name}_heatmap.png')
                save_heatmap(individual_heats[name], img, det_path)

        # Build result summary
        result['max_score'] = float(heat.max())
        result['mean_score'] = float(heat.mean())
        result['min_score'] = float(heat.min())
        result['heatmap_shape'] = list(heat.shape)

        if args.save_individual and individual_heats:
            result['individual_scores'] = {
                name: {
                    'max': float(individual_heats[name].max()),
                    'mean': float(individual_heats[name].mean()),
                    'min': float(individual_heats[name].min())
                }
                for name in detector_names
            }

        # Save per-image summary
        summary_path = os.path.join(outdir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"Failed to analyze {img_path}: {e}")

    return result


def build_parser():
    """Set up argparse with all the CLI options."""
    from priorpatch import __version__

    p = argparse.ArgumentParser(
        prog='priorpatch',
        description='PriorPatch - Detect AI-generated or manipulated images'
    )
    p.add_argument(
        '--version', '-v',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    sub = p.add_subparsers(dest='cmd', help='Available commands')

    # Analyze subcommand
    run = sub.add_parser('analyze', help='Analyze image(s) for manipulation')

    # Input options (mutually exclusive)
    input_group = run.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        help='Path to input image file or glob pattern (e.g., "photos/*.jpg")'
    )
    input_group.add_argument(
        '--input-dir', '-d',
        help='Directory containing images to analyze (recursive)'
    )

    run.add_argument(
        '--outdir', '-o',
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    run.add_argument(
        '--patch_size',
        type=int,
        default=64,
        help='Size of analysis patches in pixels (default: 64)'
    )
    run.add_argument(
        '--stride',
        type=int,
        default=32,
        help='Stride for patch extraction (default: 32)'
    )
    run.add_argument(
        '--config',
        default='config/detectors.json',
        help='Path to detector configuration file (default: config/detectors.json)'
    )
    run.add_argument(
        '--save-individual',
        action='store_true',
        help='Save individual detector heatmaps in addition to combined'
    )
    run.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    run.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration (use CPU only)'
    )

    return p


def main(argv=None):
    """Main entry point. Pass argv for testing, otherwise uses sys.argv."""
    p = build_parser()
    args = p.parse_args(argv)

    # Configure logging
    if args.cmd == 'analyze' and hasattr(args, 'log_level'):
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    if args.cmd == 'analyze':
        try:
            # Handle GPU settings
            if args.no_gpu:
                disable_gpu()
                logger.info("GPU disabled by --no-gpu flag")
            else:
                gpu_info = get_gpu_info()
                if gpu_info['active']:
                    logger.info(f"GPU acceleration enabled: {gpu_info.get('device_name', 'Unknown GPU')}")
                elif gpu_info['available']:
                    logger.info("GPU available but disabled")
                else:
                    logger.info("GPU not available, using CPU")

            # Determine input path/pattern
            input_path = args.input if args.input else args.input_dir

            # Find all images to process
            images = find_images(input_path)

            if not images:
                logger.error(f"No images found in: {input_path}")
                return 1

            is_batch = len(images) > 1

            if is_batch:
                logger.info(f"Found {len(images)} images to analyze")
            else:
                logger.info(f"Analyzing image: {images[0]}")

            # Load ensemble once (reuse for all images)
            ens = Ensemble.from_config(args.config)
            logger.info(f"Loaded {len(ens.detectors)} detectors: {[d.name for d in ens.detectors]}")

            os.makedirs(args.outdir, exist_ok=True)

            results = []
            success_count = 0
            error_count = 0

            for idx, img_path in enumerate(images, 1):
                if is_batch:
                    logger.info(f"[{idx}/{len(images)}] Processing: {img_path}")
                    # Create subfolder for each image in batch mode
                    img_name = Path(img_path).stem
                    img_outdir = os.path.join(args.outdir, img_name)
                else:
                    img_outdir = args.outdir

                os.makedirs(img_outdir, exist_ok=True)

                result = analyze_single_image(img_path, ens, args, img_outdir)
                results.append(result)

                if result['status'] == 'success':
                    success_count += 1
                    score = result['max_score']
                    if is_batch:
                        logger.info(f"  Max score: {score:.4f}")
                    else:
                        logger.info(f"Max anomaly score: {score:.4f}")
                else:
                    error_count += 1

            # Save batch summary if multiple images
            if is_batch:
                batch_summary = {
                    'total_images': len(images),
                    'successful': success_count,
                    'failed': error_count,
                    'patch_size': args.patch_size,
                    'stride': args.stride,
                    'detectors_used': [d.name for d in ens.detectors],
                    'results': results
                }

                # Sort by max_score descending (most suspicious first)
                batch_summary['results_by_score'] = sorted(
                    [r for r in results if r['status'] == 'success'],
                    key=lambda x: x['max_score'],
                    reverse=True
                )

                batch_summary_path = os.path.join(args.outdir, 'batch_summary.json')
                with open(batch_summary_path, 'w') as f:
                    json.dump(batch_summary, f, indent=2)

                logger.info(f"")
                logger.info(f"Batch complete: {success_count} succeeded, {error_count} failed")
                logger.info(f"Results saved to: {args.outdir}/")
                logger.info(f"Batch summary: {batch_summary_path}")

                # Show top suspicious images
                if batch_summary['results_by_score']:
                    logger.info(f"")
                    logger.info("Most suspicious images:")
                    for r in batch_summary['results_by_score'][:5]:
                        logger.info(f"  {r['filename']}: {r['max_score']:.4f}")
            else:
                logger.info(f"Analysis complete. Outputs in {args.outdir}")

            return 0 if error_count == 0 else 1

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return 1
    else:
        p.print_help()

    return 0


if __name__ == '__main__':
    main()
