import pandas as pd
import sys
sys.path.append('src')
from data_loader import load_full_dataset

df = load_full_dataset()
print("\n" + "="*30)
print("DATASET ANALYSIS")
print("="*30)
print(f"Total samples: {len(df)}")
print(f"Mean Hgb: {df['hemoglobin'].mean():.2f}")
print(f"Std Dev: {df['hemoglobin'].std():.2f}")
print(f"Min Hgb: {df['hemoglobin'].min():.2f}")
print(f"Max Hgb: {df['hemoglobin'].max():.2f}")

# Categorize based on 11.0 (Anemic) and 12.0 (Threshold for non-pregnant women)
anemic_11 = len(df[df['hemoglobin'] < 11.0])
anemic_12 = len(df[df['hemoglobin'] < 12.0])

print(f"\nSeverely/Moderately Anemic (<11.0): {anemic_11} ({anemic_11/len(df)*100:.1f}%)")
print(f"Likely Anemic (<12.0): {anemic_12} ({anemic_12/len(df)*100:.1f}%)")
print(f"Normal (>=12.0): {len(df[df['hemoglobin'] >= 12.0])} ({len(df[df['hemoglobin'] >= 12.0])/len(df)*100:.1f}%)")
print("="*30)
