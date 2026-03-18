# """
# model.py — EfficientNetB0 Regression Model & Training Pipeline
# EYES-DEFY-ANEMIA project
# """

# import os
# import sys
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, Model
# from tensorflow.keras.applications import EfficientNetB0
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# # Import project config and data loader
# # Note: When running as a script from the root, ensure src is in path or use relative imports
# from config import IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH, SCALER_PATH, RANDOM_SEED
# from data_loader import load_full_dataset, AnemiaDataset as AnemiaSequence

# # =============================================================================
# # 1. MODEL ARCHITECTURE
# # =============================================================================

# def build_model(img_size=IMG_SIZE, dropout_rate=0.3):
#     """
#     Builds the EfficientNetB0 multi-input model.
#     Inputs:
#         image_input: (224, 224, 3)
#         meta_input: (2,) [normalized age, encoded gender]
#     """
#     # Image Branch
#     img_in = layers.Input(shape=(*img_size, 3), name='image_input')
#     base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=img_in)
#     base.trainable = False  # Freeze backbone for transfer learning
    
#     x = layers.GlobalAveragePooling2D()(base.output)
    
#     # Metadata Branch
#     meta_in = layers.Input(shape=(2,), name='meta_input')
    
#     # Fusion
#     merged = layers.Concatenate()([x, meta_in])
#     z = layers.Dense(128, activation='relu')(merged)
#     z = layers.Dropout(dropout_rate)(z)
#     output = layers.Dense(1, name='hemoglobin_output')(z)
    
#     model = Model(inputs=[img_in, meta_in], outputs=output, name="EfficientNetB0_Anemia")
    
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#         loss='mse',
#         metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
#     )
#     return model

# # =============================================================================
# # 2. UTILITIES & VISUALIZATION
# # =============================================================================

# def plot_training_curves(history):
#     fig, ax = plt.subplots(1, 2, figsize=(14, 5))
#     # Loss
#     ax[0].plot(history.history['loss'], label='Train Loss')
#     ax[0].plot(history.history['val_loss'], label='Val Loss')
#     ax[0].set_title('MSE Loss')
#     ax[0].legend()
#     # MAE
#     ax[1].plot(history.history['mae'], label='Train MAE')
#     ax[1].plot(history.history['val_mae'], label='Val MAE')
#     ax[1].set_title('Mean Absolute Error')
#     ax[1].legend()
#     plt.show()

# # =============================================================================
# # 3. TRAINING PIPELINE (EXECUTABLE SCRIPT)
# # =============================================================================

# if __name__ == "__main__":
#     print("🚀 Starting Automated Training Pipeline...")
    
#     # Ensure directories exist
#     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
#     # 1. Load and Preprocess Data
#     print("📊 Loading datasets...")
#     df = load_full_dataset()
    
#     # Normalize Hemoglobin [0, 1]
#     scaler = MinMaxScaler()
#     df['hgb_norm'] = scaler.fit_transform(df[['hemoglobin']])
    
#     # 2. Save Scaler (Crucial for Inference in app.py)
#     with open(SCALER_PATH, 'wb') as f:
#         pickle.dump(scaler, f)
#     print(f"✅ Scaler saved to {SCALER_PATH}")
    
#     # 3. Split Data
#     train_df, test_df = train_test_split(df, test_size=0.15, random_state=RANDOM_SEED)
#     train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=RANDOM_SEED)
    
#     # 4. Initialize Generators
#     train_gen = AnemiaSequence(train_df, batch_size=BATCH_SIZE, target_col='hgb_norm', augment=True)
#     val_gen = AnemiaSequence(val_df, batch_size=BATCH_SIZE, target_col='hgb_norm', shuffle=False)
    
#     # 5. Build and Train Model
#     model = build_model()
    
#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
#         tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_mae', save_best_only=True, verbose=1),
#         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6)
#     ]
    
#     print("🏋️ Starting fit...")
#     history = model.fit(
#         train_gen,
#         validation_data=val_gen,
#         epochs=EPOCHS,
#         callbacks=callbacks
#     )
    
#     # 6. Evaluation
#     print("\n🧐 Final Evaluation on Test Set...")
#     test_gen = AnemiaSequence(test_df, batch_size=1, target_col='hgb_norm', shuffle=False)
#     results = model.evaluate(test_gen, verbose=0)
#     metrics = dict(zip(model.metrics_names, results))
    
#     hgb_range = scaler.scale_[0] 
#     print(f"🎯 Test MAE (Normalized): {metrics['mae']:.4f}")
#     print(f"📐 Test MAE (g/dL):       {metrics['mae'] * hgb_range:.4f}")
    
#     # 7. Visualize Results
#     plot_training_curves(history)
#     print("✅ Training Complete.")
