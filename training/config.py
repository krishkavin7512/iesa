"""
Training Configuration for Semiconductor Defect Detection

Contains all hyperparameters, paths, and settings for training.
"""

from pathlib import Path
import os

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "augmented"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Create directories
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOG_DIR, CHECKPOINT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
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
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)  # Models expect 3 channels

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

# Batch size (RTX 4070 has 12GB VRAM - can handle 32-64)
BATCH_SIZE = 32

# Stage 1: Transfer Learning (freeze base, train head)
STAGE1_EPOCHS = 20
STAGE1_LEARNING_RATE = 1e-3
STAGE1_FREEZE_PERCENT = 0.80  # Freeze 80% of base layers

# Stage 2: Fine-tuning (unfreeze some base layers)
STAGE2_EPOCHS = 15
STAGE2_LEARNING_RATE = 1e-5
STAGE2_UNFREEZE_PERCENT = 0.20  # Unfreeze last 20% of base layers

# Regularization
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2
L2_REGULARIZATION = 0.01
LABEL_SMOOTHING = 0.1

# Early stopping
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# EfficientNet configuration
EFFICIENTNET_CONFIG = {
    "name": "efficientnet",
    "base_model": "EfficientNetB0",
    "input_shape": INPUT_SHAPE,
    "dense_units": 128,
    "dropout_1": DROPOUT_RATE_1,
    "dropout_2": DROPOUT_RATE_2,
    "l2_reg": L2_REGULARIZATION,
}

# MobileNet configuration
MOBILENET_CONFIG = {
    "name": "mobilenet",
    "base_model": "MobileNetV3Small",
    "input_shape": INPUT_SHAPE,
    "dense_units": 96,  # Smaller for MobileNet
    "dropout_1": DROPOUT_RATE_1,
    "dropout_2": DROPOUT_RATE_2,
    "l2_reg": L2_REGULARIZATION,
}

# Ensemble configuration
ENSEMBLE_CONFIG = {
    "efficientnet_weight": 0.7,
    "mobilenet_weight": 0.3,
}

# ==============================================================================
# DATA AUGMENTATION (runtime)
# ==============================================================================
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "vertical_flip": True,
    "brightness_range": [0.85, 1.15],
    "fill_mode": "reflect",
}

# ==============================================================================
# CALLBACKS
# ==============================================================================
TENSORBOARD_ENABLED = False
SAVE_BEST_ONLY = True
MONITOR_METRIC = "val_accuracy"
MONITOR_MODE = "max"
