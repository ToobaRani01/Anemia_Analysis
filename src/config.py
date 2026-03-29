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
# Updated based on clinical guidelines provided:
# - Children 6-59m & Pregnant: Normal >= 11.0
# - Children 5-11y: Normal >= 11.5
# - Children 12-14y & Women 15+: Normal >= 12.0
# - Men 15+: Normal >= 13.0

NORMAL_HEMOGLOBIN_RANGES = {
    # (Gender, Age_Min, Age_Max, Is_Pregnant): (Min_Normal, Max_Normal)
    ("M",  0,  4, False): (11.0, 14.0),
    ("F",  0,  4, False): (11.0, 14.0),
    
    ("M",  5, 11, False): (11.5, 14.5),
    ("F",  5, 11, False): (11.5, 14.5),
    
    ("M", 12, 14, False): (12.0, 15.0),
    ("F", 12, 14, False): (12.0, 15.0),
    
    ("M", 15, 200, False): (13.0, 17.0),
    ("F", 15, 200, False): (12.0, 15.5),
    
    ("F", 0, 200, True): (11.0, 14.5), # Pregnant women
}

# Specific thresholds for Anemia Severity (Mild, Moderate, Severe)
# Values are upper bounds for each category.
SEVERITY_THRESHOLDS = {
    "children_6_59m":    {"severe": 7.0, "moderate": 10.0, "mild": 11.0},
    "children_5_11y":    {"severe": 8.0, "moderate": 11.0, "mild": 11.5},
    "children_12_14y":   {"severe": 8.0, "moderate": 11.0, "mild": 12.0},
    "women_non_pregnant": {"severe": 8.0, "moderate": 11.0, "mild": 12.0},
    "women_pregnant":    {"severe": 7.0, "moderate": 10.0, "mild": 11.0},
    "men_15_plus":       {"severe": 8.0, "moderate": 11.0, "mild": 11.9},
}

# ─────────────────────────────────────────────
# SEVERITY LABEL MAPPING
# ─────────────────────────────────────────────
SEVERITY_LABELS = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}



# ─────────────────────────────────────────────
# MODEL ARCHITECTURE  (matches notebook exactly)
# ─────────────────────────────────────────────
BACKBONE        = "efficientnetb0"
DROPOUT_RATE    = 0.3    # matches notebook
DENSE_UNITS     = 128    # matches notebook: single Dense(128) layer
