"""
config.py — Central configuration for EYES-DEFY-ANEMIA project
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
SRC_DIR         = os.path.dirname(os.path.abspath(__file__))
BASE_DIR        = os.path.dirname(SRC_DIR)
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "efficientnet_b0_anemia.h5")   # ← fixed filename
SCALER_PATH     = os.path.join(MODEL_DIR, "hgb_scaler.pkl")               # ← scaler saved here

INDIA_DIR       = os.path.join(DATASET_DIR, "India")
ITALY_DIR       = os.path.join(DATASET_DIR, "Italy")
INDIA_EXCEL     = os.path.join(INDIA_DIR, "India.xlsx")
ITALY_EXCEL     = os.path.join(ITALY_DIR, "Italy.xlsx")

# ─────────────────────────────────────────────
# EXCEL COLUMN NAMES
# ─────────────────────────────────────────────
COL_FOLDER      = "Number"
COL_AGE         = "Age"
COL_GENDER      = "Gender"
COL_HGB         = "Hgb"

# ─────────────────────────────────────────────
# IMAGE SETTINGS
# ─────────────────────────────────────────────
IMG_SIZE        = (224, 224)
NUM_CHANNELS    = 3

# Only the primary combined image is used (single-input model)
IMAGE_TYPES = [
    "forniceal_palpebral.png",
    "palpebral.png",
    "forniceal.png",
]
USE_MULTI_INPUT = False   # Always False — notebook uses single image input

# ─────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE      = 16
EPOCHS          = 50
LEARNING_RATE   = 1e-4
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

# ─────────────────────────────────────────────
# WHO ANEMIA THRESHOLDS  (g/dL)
# ─────────────────────────────────────────────

NORMAL_HEMOGLOBIN_RANGES = {
    ("M",  0, 10): (11.0, 13.5),
    ("F",  0, 10): (11.0, 13.5),

    ("M", 11, 17): (12.5, 16.5),
    ("F", 11, 17): (12.0, 15.5),

    ("M", 18, 200): (13.0, 17.0),
    ("F", 18, 200): (12.0, 15.0),
}


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE  (matches notebook exactly)
# ─────────────────────────────────────────────
BACKBONE        = "efficientnetb0"
DROPOUT_RATE    = 0.3    # matches notebook
DENSE_UNITS     = 128    # matches notebook: single Dense(128) layer
