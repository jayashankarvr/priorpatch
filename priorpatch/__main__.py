"""
Allow running priorpatch as: python -m priorpatch
"""

from priorpatch.cli import main
import sys

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
