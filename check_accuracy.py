import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Setup paths
sys.path.append('src')
from config import RANDOM_SEED, IMG_SIZE, NORMAL_HEMOGLOBIN_RANGES
from data_loader import load_full_dataset, AnemiaDataset

def get_normal_threshold(row):
    # Match row to NORMAL_HEMOGLOBIN_RANGES
    # Note: row['gender'] is 1 for Male, 0 for Female. config.py uses 'M'/'F'
    gender_map = {1: 'M', 0: 'F'}
    gender = gender_map[row['gender']]
    age = float(row['age'])
    is_pregnant = False # Default unless we have the field
    
    # Simple threshold matching for accuracy report
    # Children 6-59m (under 5): 11.0
    # Children 5-11y: 11.5
    # Women/Older children: 12.0
    # Men: 13.0
    
    if age < 5: return 11.0
    if age < 12: return 11.5
    if gender == 'M' and age >= 15: return 13.0
    return 12.0 # general female/youth

def check_accuracy():
    # 1. Load Data
    print("Loading data...")
    df = load_full_dataset()
    
    # 2. Split (Same as notebook)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"Test size: {len(test_df)} samples")
    
    # 3. Load Model and Scaler
    print("Loading model and scaler...")
    model = tf.keras.models.load_model('models/efficientnet_b0_anemia.h5', compile=False)
    # Scaler was fit on all data in the notebook
    # min_val = 6.4, max_val = 15.4 (approximately from previous summary)
    # We use the actual scaler if exists
    if os.path.exists('models/hgb_scaler.pkl'):
        scaler = joblib.load('models/hgb_scaler.pkl')
        hgb_min = scaler.data_min_[0]
        hgb_max = scaler.data_max_[0]
    else:
        # Fallback if pickle is corrupted
        hgb_min, hgb_max = 6.4, 15.4
    
    # 4. Predict
    # Use AnemiaDataset generator
    test_gen = AnemiaDataset(test_df, batch_size=1, shuffle=False, augment=False)
    preds_norm = model.predict(test_gen, verbose=0)
    
    # 5. Denormalize
    preds_real = (preds_norm.flatten() * (hgb_max - hgb_min)) + hgb_min
    actual_real = test_df['hemoglobin'].values
    
    # 6. Apply Clinical Thresholds
    y_true_binary = []
    y_pred_binary = []
    
    for i, (_, row) in enumerate(test_df.iterrows()):
        thresh = get_normal_threshold(row)
        
        y_true_binary.append("ANEMIC" if actual_real[i] < thresh else "NON-ANEMIC")
        y_pred_binary.append("ANEMIC" if preds_real[i] < thresh else "NON-ANEMIC")
    
    # 7. Results
    print("\n" + "="*50)
    print(" 🩸 ANEMIA PREDICTION ACCURACY (TEST SET)")
    print("="*50)
    acc = accuracy_score(y_true_binary, y_pred_binary)
    print(f"🚀 Overall Accuracy: {acc*100:.2f}%")
    print("-" * 50)
    print(classification_report(y_true_binary, y_pred_binary))
    
    # Regression Error
    mae_real = np.mean(np.abs(actual_real - preds_real))
    print(f"📐 Mean Absolute Error (MAE): {mae_real:.2f} g/dL")
    print("="*50)

if __name__ == "__main__":
    check_accuracy()
