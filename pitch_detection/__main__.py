"""
Main entry point for running the CLI as a module.

Usage:
    python -m pitch_detection.cli train --help
    python -m pitch_detection.cli predict --help
    python -m pitch_detection.cli evaluate --help
"""

from .cli import main

if __name__ == "__main__":
    main()
