"""
Helper to set up paths for example scripts.
Makes examples work from both project root and examples/ directory.
"""

import os
import sys

def setup_paths():
    """
    Returns (img_path, config_path, output_dir) based on current directory.
    Exits with helpful message if paths not found.
    """
    # Check if running from project root
    if os.path.exists('examples/sample_input.png'):
        return {
            'img': 'examples/sample_input.png',
            'config': 'config/detectors.json',
            'output_dir': 'examples/',
            'root': '.'
        }
    # Check if running from examples directory
    elif os.path.exists('sample_input.png'):
        return {
            'img': 'sample_input.png',
            'config': '../config/detectors.json',
            'output_dir': './',
            'root': '..'
        }
    else:
        print("Error: Cannot find sample_input.png")
        print("Run this script from either:")
        print("  - Project root: python examples/script_name.py")
        print("  - Examples dir: cd examples && python script_name.py")
        sys.exit(1)

def check_config(config_path):
    """Check if config file exists."""
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        print("Make sure you're running from the correct directory")
        sys.exit(1)
