"""
Configuration constants for the clustering pipeline.

This module defines default paths and constants used throughout the project.
Most of these are overridden by command-line arguments in pipeline.py and dashboard.py.

Default paths are project-relative and point to:
- images/data/test: Test dataset directory
- output: Directory for clustering results

Note: When running via command-line, these defaults are typically not used
as arguments are passed explicitly.
"""

import os

# Get the project root directory (parent of src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default paths (used as fallbacks)
PATH_OUTPUT = os.path.join(ROOT_DIR, "output")
IMAGES_DIR = os.path.join(ROOT_DIR, "images/data/test")

# Model configuration
MODEL_CLUSTERING = "kmeans"  # Default clustering model for backward compatibility
