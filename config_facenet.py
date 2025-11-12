"""
Configuration file for FaceNet-based Face Deepfake Detection
Optimized for real vs fake face classification
"""

import os
from pathlib import Path
import torch

# ===============================
# PATHS
# ===============================
BASE_DIR = Path("/Users/anmol/Documents/Data")
DATA_DIR = BASE_DIR / "datasets" / "Human"
TRAIN_DIR = DATA_DIR / "Train"
VAL_DIR = DATA_DIR / "Validation"
TEST_DIR = DATA_DIR / "Test"

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create output directories
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===============================
# MODEL CONFIGURATION - FACENET
# ===============================
MODEL_NAME = "facenet"
MODEL_TYPE = "facenet"  # 'facenet' or 'facenet_attention'
FACENET_PRETRAINED = "vggface2"  # 'vggface2', 'casia-webface', or None
NUM_CLASSES = 1  # Binary classification (Real vs Fake)

# FaceNet uses 160x160 input by default
IMG_SIZE = 160

# ===============================
# FACE DETECTION & PREPROCESSING
# ===============================
USE_FACE_DETECTION = True  # Detect and crop faces before training
FACE_DETECTION_MARGIN = 20  # Margin around detected face
FACE_DETECTION_THRESHOLD = 0.9  # MTCNN detection threshold
ALIGN_FACES = True  # Align faces using facial landmarks

# ===============================
# TRAINING HYPERPARAMETERS
# ===============================
BATCH_SIZE = 32  # FaceNet can handle larger batches
NUM_EPOCHS = 30
LEARNING_RATE = 2e-4  # Slightly higher for better convergence
WEIGHT_DECAY = 1e-4  # Increased for better regularization

# Learning rate scheduler
SCHEDULER_TYPE = "cosine_warmup"  # Options: onecycle, cosine, cosine_warmup, step, reduce_on_plateau
SCHEDULER_PARAMS = {
    "T_max": 30,  # For cosine
    "eta_min": 1e-6,
    "warmup_epochs": 3  # Warmup for first 3 epochs
}

# Early stopping
EARLY_STOPPING_PATIENCE = 10  # Increased patience for better convergence
EARLY_STOPPING_MIN_DELTA = 0.001  # More sensitive to improvements

# ===============================
# DATA LOADING
# ===============================
NUM_WORKERS = 8 # Set to 0 for MPS (multiprocessing issue with face detection on MPS)
PIN_MEMORY = False  # Disabled for MPS
PERSISTENT_WORKERS = True

# ===============================
# AUGMENTATION (Face-specific)
# ===============================
AUGMENTATION_CONFIG = {
    "horizontal_flip_prob": 0.5,
    "rotation_degrees": 10,  # Smaller rotation for faces
    "color_jitter": {
        "brightness": 0.15,
        "contrast": 0.15,
        "saturation": 0.15,
        "hue": 0.05
    },
    "random_affine": {
        "degrees": 5,
        "translate": (0.05, 0.05),
        "scale": (0.95, 1.05)
    },
    # Face-specific augmentations
    "gaussian_blur_prob": 0.1,
    "random_erasing_prob": 0.1,
}

# ===============================
# MODEL ARCHITECTURE
# ===============================
DROPOUT_RATE = 1  # Increased back to prevent overfitting
USE_BATCH_NORM = True
FREEZE_EARLY_LAYERS = False  # Whether to freeze FaceNet backbone initially

# ===============================
# TRAINING OPTIONS
# ===============================
USE_MIXED_PRECISION = False  # Disabled for MPS
GRADIENT_CLIPPING = 1.0
LABEL_SMOOTHING = 0.05  # Reduced for sharper predictions

# Loss function - Class weighting for better balance
LOSS_TYPE = "weighted_bce"  # 'bce', 'weighted_bce', 'focal'
CLASS_WEIGHTS = [1.06, 1.0]  # [Real weight, Fake weight] - balances 36.5k real vs 38.7k fake
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# Hard example mining
USE_HARD_EXAMPLE_MINING = True  # Focus on difficult samples
HARD_EXAMPLE_RATIO = 0.5  # Increased to 50% - use more samples to reduce overfitting

# ===============================
# DEVICE CONFIGURATION
# ===============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
USE_MULTI_GPU = torch.cuda.device_count() > 1

# ===============================
# LOGGING
# ===============================
LOG_INTERVAL = 10
SAVE_CHECKPOINT_EVERY = 1
SAVE_BEST_ONLY = True  # Only save best model
VERBOSE = True

# ===============================
# REPRODUCIBILITY
# ===============================
RANDOM_SEED = 42

# ===============================
# CLASS MAPPING
# ===============================
CLASS_NAMES = ['Real', 'Fake']
CLASS_TO_IDX = {'real': 0, 'fake': 1}  # lowercase for folder matching

# ===============================
# FACENET SPECIFIC SETTINGS
# ===============================
EMBEDDING_DIM = 512  # FaceNet embedding dimension
USE_TRIPLET_LOSS = False  # Use triplet loss for embedding learning
TRIPLET_MARGIN = 0.2

# Print config
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("FACENET CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE}")
    print(f"  Model: {MODEL_NAME} (pretrained on {FACENET_PRETRAINED})")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Face Detection: {USE_FACE_DETECTION}")
    print(f"  Face Alignment: {ALIGN_FACES}")
    print(f"  Dropout Rate: {DROPOUT_RATE}")
    print(f"  Label Smoothing: {LABEL_SMOOTHING}")
    print(f"{'='*60}\n")
