"""
utils.py — Utility functions: anemia classification, inference, metrics
EYES-DEFY-ANEMIA project
"""

import os
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

from config import (
    IMG_SIZE, IMAGE_TYPES,
    NORMAL_HEMOGLOBIN_RANGES, SEVERITY_THRESHOLDS,
    MODEL_PATH, SCALER_PATH, SEVERITY_LABELS
)


# ──────────────────────────────────────────────────────────────
# ANEMIA CLASSIFICATION
# ──────────────────────────────────────────────────────────────

def get_normal_range(age: int, gender: str, is_pregnant: bool = False) -> tuple:
    """Return (min_hb, max_hb) for given age, gender & pregnancy status."""
    g = "M" if str(gender).strip().lower() in ("m", "male", "1") else "F"
    age = int(age)
    
    # Priority to pregnancy if applicable
    if g == "F" and is_pregnant:
        for (gen, min_a, max_a, preg), (min_h, max_h) in NORMAL_HEMOGLOBIN_RANGES.items():
            if preg:
                return min_h, max_h

    for (gen, min_a, max_a, preg), (min_h, max_h) in NORMAL_HEMOGLOBIN_RANGES.items():
        if gen == g and min_a <= age <= max_a and preg == False:
            return min_h, max_h
            
    return 12.0, 15.0  # safe default


def classify_anemia(hgb: float, age: int, gender: str, is_pregnant: bool = False) -> dict:
    """Returns a full classification dict using specific thresholds for each category."""
    min_hb, max_hb = get_normal_range(age, gender, is_pregnant)
    
    # Determine the category for thresholds
    cat = "men_15_plus" if gender.lower() == "male" and age >= 15 else "women_non_pregnant"
    if age < 5:
        cat = "children_6_59m"
    elif 5 <= age <= 11:
        cat = "children_5_11y"
    elif 12 <= age <= 14:
        cat = "children_12_14y"
    
    if gender.lower() == "female":
        if is_pregnant:
            cat = "women_pregnant"
        elif age >= 15:
            cat = "women_non_pregnant"
            
    t = SEVERITY_THRESHOLDS.get(cat, SEVERITY_THRESHOLDS["women_non_pregnant"])
    
    is_anemic = hgb < t["mild"]
    is_high   = hgb > max_hb

    if is_anemic:
        if hgb < t["severe"]:     severity = 3
        elif hgb < t["moderate"]: severity = 2
        else:                     severity = 1
    elif is_high:
        severity = 0
    else:
        severity = 0

    dist = min_hb - hgb
    k = 2.5
    anemia_probability = float(1 / (1 + np.exp(-k * dist)))

    return {
        "is_anemic":          is_anemic,
        "is_high":            is_high,
        "severity":           severity,
        "anemia_probability": round(anemia_probability * 100, 1),
        "min_hb":             min_hb,
        "max_hb":             max_hb,
        "category":           cat
    }


# ──────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ──────────────────────────────────────────────────────────────

def preprocess_image(img_source) -> np.ndarray:
    """Accept PIL Image or file path → (1, 224, 224, 3) float32 array in [0,1]."""
    if isinstance(img_source, str):
        img = Image.open(img_source).convert("RGB")
    else:
        img = img_source.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def prepare_inference_inputs(uploaded_images: dict, age: int, gender: str) -> list:
    """
    Build model inputs matching the notebook's simple model:
        inputs = [image_array (1,224,224,3),  meta_array (1,2)]
    Only IMAGE_TYPES[0] (forniceal_palpebral) is used.
    """
    primary_key = IMAGE_TYPES[0]
    img_pil = uploaded_images.get(primary_key)

    if img_pil is not None:
        img_array = preprocess_image(img_pil)
    else:
        img_array = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)

    g_val = 1.0 if str(gender).strip().lower() in ("m", "male") else 0.0
    meta  = np.array([[age / 100.0, g_val]], dtype=np.float32)

    # Return as a list: [image_input, meta_input]
    return [img_array, meta]


# ──────────────────────────────────────────────────────────────
# SCALER  (needed because model was trained on normalised hgb)
# ──────────────────────────────────────────────────────────────

def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Load the MinMaxScaler saved during training.
    Returns None if the scaler file does not exist (graceful fallback).
    """
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"  ✓ Scaler loaded from {scaler_path}")
        return scaler
    else:
        print(f"  ⚠ Scaler not found at {scaler_path}. "
              "Predictions will be in normalised [0,1] range — retrain or save scaler.")
        return None


def inverse_transform_hgb(pred_norm: float, scaler) -> float:
    """Convert a normalised [0,1] prediction back to g/dL using the saved scaler."""
    if scaler is None:
        # Fallback: assume typical dataset range (~5–18 g/dL)
        return float(pred_norm) * 13.0 + 5.0
    arr = np.array([[pred_norm]], dtype=np.float32)
    return float(scaler.inverse_transform(arr)[0][0])


# ──────────────────────────────────────────────────────────────
# MODEL INFERENCE
# ──────────────────────────────────────────────────────────────

def predict_hemoglobin(model, inputs: list, scaler=None) -> float:
    """
    Run model inference and return hemoglobin value in g/dL.
    If a scaler is provided the normalised output is inverse-transformed.
    """
    pred_norm = float(np.squeeze(model.predict(inputs, verbose=0)))
    return inverse_transform_hgb(pred_norm, scaler)


# ──────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH):
    """Load the saved .h5 model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first by running final_training.ipynb."
        )
    model = tf.keras.models.load_model(model_path, compile=False)
    # Recompile with basic settings just for inference
    model.compile(optimizer='adam', loss='mse')
    print(f"  ✓ Model loaded from {model_path}")
    return model


# ──────────────────────────────────────────────────────────────
# COLOUR HELPERS FOR UI
# ──────────────────────────────────────────────────────────────

def severity_color(severity) -> str:
    # Handle both string (legacy/direct) and numeric values
    mapping = {
        0:          "#2ECC71",
        "Normal":   "#2ECC71",
        1:          "#F39C12",
        "Mild":     "#F39C12",
        2:          "#E67E22",
        "Moderate": "#E67E22",
        3:          "#E74C3C",
        "Severe":   "#E74C3C",
    }
    return mapping.get(severity, "#95A5A6")


def severity_emoji(severity) -> str:
    # Handle both string (legacy/direct) and numeric values
    mapping = {
        0:          "✅",
        "Normal":   "✅",
        1:          "⚠️",
        "Mild":     "⚠️",
        2:          "🔶",
        "Moderate": "🔶",
        3:          "🔴",
        "Severe":   "🔴",
    }
    return mapping.get(severity, "❓")
