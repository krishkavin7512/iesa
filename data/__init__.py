"""
Dataset Module Initialization

This module provides utilities for:
- Downloading datasets (download_datasets.py)
- Curating and preprocessing (curate_datasets.py)
- Augmentation (augment.py)
"""

from pathlib import Path

# Data directories
DATA_DIR = Path(__file__).parent
PROJECT_ROOT = DATA_DIR.parent

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"

# Class names
CLASS_NAMES = [
    "clean",
    "scratches",
    "particles", 
    "pattern_defects",
    "edge_defects",
    "center_defects",
    "random_defects",
    "other"
]

NUM_CLASSES = len(CLASS_NAMES)

# Image configuration
IMAGE_SIZE = (224, 224)
