"""
Tests for CLI functionality.
"""

import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from priorpatch.cli import build_parser, main, find_images, IMAGE_EXTENSIONS


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image."""
    img_path = tmp_path / "test_image.png"
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config file."""
    config = {
        "version": "1.0",
        "enabled_detectors": ["color_stats", "neighbor_consistency"],
        "detector_weights": {"color_stats": 1.0, "neighbor_consistency": 1.0}
    }
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return str(config_path)


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a directory with multiple test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(3):
        img_path = img_dir / f"image_{i}.png"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_path)

    # Add a non-image file to test filtering
    (img_dir / "readme.txt").write_text("not an image")

    return str(img_dir)


class TestFindImages:
    """Test find_images helper function."""

    def test_find_single_file(self, temp_image):
        """Single file path returns that file."""
        images = find_images(temp_image)
        assert len(images) == 1
        assert images[0] == temp_image

    def test_find_directory(self, temp_image_dir):
        """Directory returns all images inside."""
        images = find_images(temp_image_dir)
        assert len(images) == 3
        assert all(Path(img).suffix.lower() in IMAGE_EXTENSIONS for img in images)

    def test_find_glob_pattern(self, temp_image_dir):
        """Glob pattern returns matching images."""
        pattern = os.path.join(temp_image_dir, "*.png")
        images = find_images(pattern)
        assert len(images) == 3

    def test_find_nonexistent_raises(self):
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            find_images("/nonexistent/path/image.jpg")

    def test_filters_non_images(self, temp_image_dir):
        """Non-image files are filtered out."""
        images = find_images(temp_image_dir)
        assert not any(img.endswith('.txt') for img in images)


class TestCLIParser:
    """Test command-line argument parsing."""

    def test_build_parser_creates_parser(self):
        """Parser is created with correct program name."""
        parser = build_parser()
        assert parser.prog == 'priorpatch'
        assert parser.description is not None

    def test_analyze_subcommand_exists(self):
        """Analyze subcommand is available."""
        parser = build_parser()
        args = parser.parse_args(['analyze', '--input', 'test.jpg'])
        assert args.cmd == 'analyze'
        assert args.input == 'test.jpg'

    def test_input_dir_argument(self):
        """--input-dir argument is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(['analyze', '--input-dir', 'photos/'])
        assert args.input_dir == 'photos/'
        assert args.input is None

    def test_input_and_input_dir_mutually_exclusive(self):
        """--input and --input-dir cannot be used together."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['analyze', '--input', 'test.jpg', '--input-dir', 'photos/'])

    def test_all_analyze_arguments(self):
        """All analyze arguments are parsed correctly."""
        parser = build_parser()
        args = parser.parse_args([
            'analyze',
            '--input', 'input.jpg',
            '--outdir', 'output',
            '--patch_size', '128',
            '--stride', '64',
            '--config', 'custom.json',
            '--save-individual',
            '--log-level', 'DEBUG'
        ])

        assert args.input == 'input.jpg'
        assert args.outdir == 'output'
        assert args.patch_size == 128
        assert args.stride == 64
        assert args.config == 'custom.json'
        assert args.save_individual is True
        assert args.log_level == 'DEBUG'

    def test_default_arguments(self):
        """Default argument values are correct."""
        parser = build_parser()
        args = parser.parse_args(['analyze', '--input', 'test.jpg'])

        assert args.outdir == 'outputs'
        assert args.patch_size == 64
        assert args.stride == 32
        assert args.config == 'config/detectors.json'
        assert args.save_individual is False
        assert args.log_level == 'INFO'

    def test_log_level_choices(self):
        """Log level only accepts valid values."""
        parser = build_parser()

        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            args = parser.parse_args(['analyze', '--input', 'x.jpg', '--log-level', level])
            assert args.log_level == level

        with pytest.raises(SystemExit):
            parser.parse_args(['analyze', '--input', 'x.jpg', '--log-level', 'INVALID'])


class TestCLIExecution:
    """Test CLI command execution."""

    def test_analyze_success(self, temp_image, temp_config, tmp_path):
        """Successful analysis creates all output files."""
        outdir = tmp_path / "output"

        result = main([
            'analyze',
            '--input', temp_image,
            '--outdir', str(outdir),
            '--config', temp_config,
            '--patch_size', '64',
            '--stride', '32'
        ])

        assert result == 0
        assert outdir.exists()
        assert (outdir / 'heatmap_combined.png').exists()
        assert (outdir / 'summary.json').exists()

    def test_analyze_with_individual_detectors(self, temp_image, temp_config, tmp_path):
        """Analysis with --save-individual creates individual heatmaps."""
        outdir = tmp_path / "output"

        result = main([
            'analyze',
            '--input', temp_image,
            '--outdir', str(outdir),
            '--config', temp_config,
            '--save-individual'
        ])

        assert result == 0
        assert (outdir / 'heatmap_combined.png').exists()
        assert (outdir / 'individual_detectors').exists()

        individual_dir = outdir / 'individual_detectors'
        heatmaps = list(individual_dir.glob('*_heatmap.png'))
        assert len(heatmaps) > 0

    def test_analyze_summary_json_content(self, temp_image, temp_config, tmp_path):
        """Summary JSON contains expected fields."""
        outdir = tmp_path / "output"

        main([
            'analyze',
            '--input', temp_image,
            '--outdir', str(outdir),
            '--config', temp_config
        ])

        summary_path = outdir / 'summary.json'
        with open(summary_path) as f:
            summary = json.load(f)

        assert 'max_score' in summary
        assert 'mean_score' in summary
        assert 'heatmap_shape' in summary
        assert 'filename' in summary
        assert 'status' in summary
        assert summary['status'] == 'success'

    def test_analyze_custom_patch_size(self, temp_image, temp_config, tmp_path):
        """Custom patch size is applied."""
        outdir = tmp_path / "output"

        result = main([
            'analyze',
            '--input', temp_image,
            '--outdir', str(outdir),
            '--config', temp_config,
            '--patch_size', '32',
            '--stride', '16'
        ])

        assert result == 0


class TestBatchProcessing:
    """Test batch/directory processing."""

    def test_input_dir_processes_all_images(self, temp_image_dir, temp_config, tmp_path):
        """--input-dir processes all images in directory."""
        outdir = tmp_path / "output"

        result = main([
            'analyze',
            '--input-dir', temp_image_dir,
            '--outdir', str(outdir),
            '--config', temp_config
        ])

        assert result == 0

        # Check batch summary exists
        assert (outdir / 'batch_summary.json').exists()

        # Check subfolders created for each image
        subdirs = [d for d in outdir.iterdir() if d.is_dir()]
        assert len(subdirs) == 3

    def test_glob_pattern_input(self, temp_image_dir, temp_config, tmp_path):
        """Glob pattern in --input works."""
        outdir = tmp_path / "output"
        pattern = os.path.join(temp_image_dir, "*.png")

        result = main([
            'analyze',
            '--input', pattern,
            '--outdir', str(outdir),
            '--config', temp_config
        ])

        assert result == 0
        assert (outdir / 'batch_summary.json').exists()

    def test_batch_summary_content(self, temp_image_dir, temp_config, tmp_path):
        """Batch summary JSON has correct structure."""
        outdir = tmp_path / "output"

        main([
            'analyze',
            '--input-dir', temp_image_dir,
            '--outdir', str(outdir),
            '--config', temp_config
        ])

        with open(outdir / 'batch_summary.json') as f:
            summary = json.load(f)

        assert summary['total_images'] == 3
        assert summary['successful'] == 3
        assert summary['failed'] == 0
        assert 'results' in summary
        assert 'results_by_score' in summary
        assert len(summary['results']) == 3

    def test_batch_results_sorted_by_score(self, temp_image_dir, temp_config, tmp_path):
        """Batch results are sorted by max_score descending."""
        outdir = tmp_path / "output"

        main([
            'analyze',
            '--input-dir', temp_image_dir,
            '--outdir', str(outdir),
            '--config', temp_config
        ])

        with open(outdir / 'batch_summary.json') as f:
            summary = json.load(f)

        scores = [r['max_score'] for r in summary['results_by_score']]
        assert scores == sorted(scores, reverse=True)

    def test_single_image_no_batch_summary(self, temp_image, temp_config, tmp_path):
        """Single image does not create batch_summary.json."""
        outdir = tmp_path / "output"

        main([
            'analyze',
            '--input', temp_image,
            '--outdir', str(outdir),
            '--config', temp_config
        ])

        assert not (outdir / 'batch_summary.json').exists()
        assert (outdir / 'summary.json').exists()

    def test_batch_with_save_individual(self, temp_image_dir, temp_config, tmp_path):
        """Batch processing with --save-individual creates individual heatmaps."""
        outdir = tmp_path / "output"

        main([
            'analyze',
            '--input-dir', temp_image_dir,
            '--outdir', str(outdir),
            '--config', temp_config,
            '--save-individual'
        ])

        # Check each image subfolder has individual_detectors
        for subdir in outdir.iterdir():
            if subdir.is_dir() and subdir.name != 'individual_detectors':
                assert (subdir / 'individual_detectors').exists()


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_missing_input_file(self, temp_config, tmp_path):
        """Missing input file returns error code."""
        result = main([
            'analyze',
            '--input', 'nonexistent_file.jpg',
            '--outdir', str(tmp_path / 'output'),
            '--config', temp_config
        ])

        assert result == 1

    def test_missing_config_file(self, temp_image, tmp_path):
        """Missing config file returns error code."""
        result = main([
            'analyze',
            '--input', temp_image,
            '--outdir', str(tmp_path / 'output'),
            '--config', 'nonexistent_config.json'
        ])

        assert result == 1

    def test_invalid_config_json(self, temp_image, tmp_path):
        """Invalid JSON config returns error code."""
        bad_config = tmp_path / 'bad_config.json'
        bad_config.write_text('{ invalid json }')

        result = main([
            'analyze',
            '--input', temp_image,
            '--outdir', str(tmp_path / 'output'),
            '--config', str(bad_config)
        ])

        assert result == 1

    def test_invalid_image_file(self, temp_config, tmp_path):
        """Invalid image file returns error code."""
        bad_image = tmp_path / 'bad.jpg'
        bad_image.write_text('not an image')

        result = main([
            'analyze',
            '--input', str(bad_image),
            '--outdir', str(tmp_path / 'output'),
            '--config', temp_config
        ])

        assert result == 1

    def test_empty_directory(self, temp_config, tmp_path):
        """Empty directory returns error code."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = main([
            'analyze',
            '--input-dir', str(empty_dir),
            '--outdir', str(tmp_path / 'output'),
            '--config', temp_config
        ])

        assert result == 1

    def test_no_command_shows_help(self, capsys):
        """Running without command shows help."""
        result = main([])
        captured = capsys.readouterr()

        assert 'usage' in captured.out.lower() or 'priorpatch' in captured.out

    def test_batch_partial_failure(self, temp_config, tmp_path):
        """Batch with some bad images reports partial failure."""
        img_dir = tmp_path / "mixed"
        img_dir.mkdir()

        # Good image
        good_img = img_dir / "good.png"
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(good_img)

        # Bad image
        bad_img = img_dir / "bad.png"
        bad_img.write_text("not an image")

        outdir = tmp_path / "output"
        result = main([
            'analyze',
            '--input-dir', str(img_dir),
            '--outdir', str(outdir),
            '--config', temp_config
        ])

        # Should return 1 due to partial failure
        assert result == 1

        # But batch summary should still be created
        with open(outdir / 'batch_summary.json') as f:
            summary = json.load(f)

        assert summary['successful'] == 1
        assert summary['failed'] == 1


class TestCLILogging:
    """Test CLI logging configuration."""

    def test_log_level_debug(self, temp_image, temp_config, tmp_path):
        """DEBUG log level runs without error."""
        result = main([
            'analyze',
            '--input', temp_image,
            '--outdir', str(tmp_path / 'output'),
            '--config', temp_config,
            '--log-level', 'DEBUG'
        ])

        assert result == 0

    def test_log_level_error(self, temp_image, temp_config, tmp_path):
        """ERROR log level runs without error."""
        result = main([
            'analyze',
            '--input', temp_image,
            '--outdir', str(tmp_path / 'output'),
            '--config', temp_config,
            '--log-level', 'ERROR'
        ])

        assert result == 0


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    def test_end_to_end_workflow(self, tmp_path):
        """Complete workflow from image to results."""
        img_path = tmp_path / "original.png"
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(img_path)

        config_path = tmp_path / "config.json"
        config = {
            "version": "1.0",
            "enabled_detectors": ["color_stats"],
            "detector_weights": {"color_stats": 1.0}
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

        outdir = tmp_path / "results"
        result = main([
            'analyze',
            '--input', str(img_path),
            '--outdir', str(outdir),
            '--config', str(config_path),
            '--save-individual'
        ])

        assert result == 0
        assert (outdir / 'heatmap_combined.png').exists()
        assert (outdir / 'summary.json').exists()
        assert (outdir / 'individual_detectors').exists()

        with open(outdir / 'summary.json') as f:
            summary = json.load(f)
        assert 'individual_scores' in summary

    def test_batch_workflow(self, tmp_path):
        """Complete batch workflow."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for i in range(3):
            img_path = img_dir / f"image_{i}.png"
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(img_path)

        config_path = tmp_path / "config.json"
        config = {
            "version": "1.0",
            "enabled_detectors": ["color_stats"],
            "detector_weights": {"color_stats": 1.0}
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

        outdir = tmp_path / "results"
        result = main([
            'analyze',
            '--input-dir', str(img_dir),
            '--outdir', str(outdir),
            '--config', str(config_path)
        ])

        assert result == 0
        assert (outdir / 'batch_summary.json').exists()

        with open(outdir / 'batch_summary.json') as f:
            summary = json.load(f)

        assert summary['total_images'] == 3
        assert summary['successful'] == 3
