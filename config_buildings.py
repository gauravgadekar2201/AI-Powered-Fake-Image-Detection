"""
Configuration for Buildings Deepfake Detection
Training on current dataset: 900 real + 1000 fake buildings
"""

import os
from pathlib import Path
import torch

# ===============================
# PATHS
# ===============================
BASE_DIR = Path("/Users/anmol/Documents/Data")
DATA_DIR = BASE_DIR / "datasets" / "Buildings"
REAL_DIR = DATA_DIR / "Real"
FAKE_DIR = DATA_DIR / "fake"

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create output directories
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===============================
# MODEL CONFIGURATION
# ===============================
MODEL_NAME = "efficientnet_b4"  # Good balance of speed and accuracy
NUM_CLASSES = 1  # Binary classification (Real vs Fake)
PRETRAINED = True

# ===============================
# TRAINING HYPERPARAMETERS
# ===============================
IMG_SIZE = 224
BATCH_SIZE = 32  # Smaller batch for limited dataset
NUM_EPOCHS = 30 # More epochs since dataset is small
LEARNING_RATE = 1e-4  # Lower learning rate for fine-tuning
WEIGHT_DECAY = 1e-4

# Data split ratios (will split from combined dataset)
TRAIN_RATIO = 0.75  # 1,425 images
VAL_RATIO = 0.15    # 285 images
TEST_RATIO = 0.10   # 190 images

# Learning rate scheduler
SCHEDULER_TYPE = "cosine"
SCHEDULER_PARAMS = {
    "T_max": 40,
    "eta_min": 1e-6
}

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# ===============================
# DATA LOADING
# ===============================
NUM_WORKERS = 8
PIN_MEMORY = False
PERSISTENT_WORKERS = True

# ===============================
# AUGMENTATION (Heavy for small dataset)
# ===============================
AUGMENTATION_CONFIG = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.3,  # Buildings can be flipped vertically
    "rotation_degrees": 20,  # More rotation for buildings
    "color_jitter": {
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.2,
        "hue": 0.1
    },
    "random_affine": {
        "degrees": 15,
        "translate": (0.1, 0.1),
        "scale": (0.8, 1.2)
    },
    "gaussian_blur_prob": 0.2,
    "random_erasing_prob": 0.2,
    "random_perspective_prob": 0.3,  # Good for buildings
}

# ===============================
# MODEL ARCHITECTURE
# ===============================
DROPOUT_RATE = 0.7  # High dropout for small dataset
USE_BATCH_NORM = True

# ===============================
# TRAINING OPTIONS
# ===============================
USE_MIXED_PRECISION = False
GRADIENT_CLIPPING = 1.0
LABEL_SMOOTHING = 0.1

# Loss function
LOSS_TYPE = "bce"
CLASS_WEIGHTS = [1.11, 1.0]  # Balance 900 real vs 1000 fake

# ===============================
# DEVICE CONFIGURATION
# ===============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
USE_MULTI_GPU = torch.cuda.device_count() > 1

# ===============================
# LOGGING
# ===============================
LOG_INTERVAL = 10
SAVE_CHECKPOINT_EVERY = 5
SAVE_BEST_ONLY = True
VERBOSE = True

# ===============================
# REPRODUCIBILITY
# ===============================
RANDOM_SEED = 42

# ===============================
# CLASS MAPPING
# ===============================
CLASS_NAMES = ['Real', 'Fake']
CLASS_TO_IDX = {'real': 0, 'fake': 1}

# Print config
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("BUILDINGS DETECTION CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Dataset: Buildings (Real + Fake)")
    print(f"  Train/Val/Test: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    print(f"{'='*60}\n")
