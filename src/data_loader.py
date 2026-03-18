# """
# data_loader.py — Dataset loading, preprocessing & augmentation
# for EYES-DEFY-ANEMIA project
# """

# import os
# import numpy as np
# import pandas as pd
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import tensorflow as tf

# from config import (
#     INDIA_DIR, ITALY_DIR, INDIA_EXCEL, ITALY_EXCEL,
#     COL_FOLDER, COL_AGE, COL_GENDER, COL_HGB,
#     IMG_SIZE, IMAGE_TYPES, USE_MULTI_INPUT,
#     VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
# )


# # ──────────────────────────────────────────────────────────────
# # HELPERS
# # ──────────────────────────────────────────────────────────────

# def normalize_gender(val: str) -> int:
#     """Convert gender string → 0 (Female) / 1 (Male)."""
#     v = str(val).strip().lower()
#     return 1 if v in ("m", "male", "1") else 0


# def clean_to_float(val) -> float:
#     """Convert string/number to float, handling commas if necessary."""
#     if isinstance(val, str):
#         val = val.replace(",", ".").strip()
#     try:
#         return float(val)
#     except (ValueError, TypeError):
#         return np.nan


# def load_image(path: str) -> np.ndarray:
#     """Load & resize a single image to (224,224,3), normalised to [0,1]."""
#     try:
#         img = Image.open(path).convert("RGB").resize(IMG_SIZE)
#         return np.array(img, dtype=np.float32) / 255.0
#     except Exception as e:
#         # Handle corrupted or unidentified images by returning zeros
#         return np.zeros((*IMG_SIZE, 3), dtype=np.float32)


# def load_sample_images(folder_path: str) -> dict:
#     """
#     Given a sample folder, load all requested image types.
#     Returns dict  {image_type: np.ndarray}
#     """
#     images = {}
#     for img_type in IMAGE_TYPES:
#         img_path = os.path.join(folder_path, img_type)
#         if os.path.exists(img_path):
#             images[img_type] = load_image(img_path)
#         else:
#             # fill with zeros if a particular type is missing
#             images[img_type] = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
#     return images


# # ──────────────────────────────────────────────────────────────
# # LOAD ONE COUNTRY DATASET
# # ──────────────────────────────────────────────────────────────

# def load_country_data(country_dir: str, excel_path: str) -> pd.DataFrame:
#     """
#     Read the Excel label file and attach image paths for each sample.
#     Returns a DataFrame with columns:
#         folder, age, gender_raw, gender, hemoglobin, img_paths{...}
#     """
#     df = pd.read_excel(excel_path)
#     df.columns = df.columns.str.strip()

#     # Rename to standard names
#     rename_map = {}
#     for c in df.columns:
#         cl = c.lower()
#         if cl in ("id", "folder", "subject", "no", "number"):
#             rename_map[c] = "folder"
#         elif cl in ("age",):
#             rename_map[c] = "age"
#         elif cl in ("sex", "gender"):
#             rename_map[c] = "gender_raw"
#         elif cl in ("hb", "hemoglobin", "haemoglobin", "hgb"):
#             rename_map[c] = "hemoglobin"
#     df.rename(columns=rename_map, inplace=True)

#     required = {"folder", "age", "gender_raw", "hemoglobin"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"Excel columns not found: {missing}. "
#                          f"Available columns: {list(df.columns)}")

#     df["gender"] = df["gender_raw"].apply(normalize_gender)
#     df["age"] = df["age"].apply(clean_to_float)
#     df["hemoglobin"] = df["hemoglobin"].apply(clean_to_float)
#     df["folder"] = df["folder"].astype(str).str.strip()

#     # Build image paths
#     valid_rows = []
#     for _, row in df.iterrows():
#         folder_path = os.path.join(country_dir, str(row["folder"]))
#         if not os.path.isdir(folder_path):
#             continue  # skip if folder doesn't exist
        
#         row_dict = row.to_dict()
#         row_dict["folder_path"] = folder_path
        
#         # Search for files matching the types (handling prefixes)
#         files = os.listdir(folder_path)
#         for img_type in IMAGE_TYPES:
#             best_match = ""
#             for f in files:
#                 # Check for exact match or suffix match with underscore (e.g., '..._palpebral.png')
#                 if f == img_type or f.endswith("_" + img_type):
#                     best_match = os.path.join(folder_path, f)
#                     break
#             row_dict[f"img_{img_type}"] = best_match
            
#         valid_rows.append(row_dict)

#     return pd.DataFrame(valid_rows)


# # ──────────────────────────────────────────────────────────────
# # MERGE BOTH COUNTRIES
# # ──────────────────────────────────────────────────────────────

# def load_full_dataset() -> pd.DataFrame:
#     """Load and combine India + Italy datasets."""
#     dfs = []
#     for country_dir, excel_path, label in [
#         (INDIA_DIR, INDIA_EXCEL, "india"),
#         (ITALY_DIR, ITALY_EXCEL, "italy"),
#     ]:
#         if os.path.exists(excel_path) and os.path.isdir(country_dir):
#             df = load_country_data(country_dir, excel_path)
#             df["country"] = label
#             dfs.append(df)
#             print(f"  ✓ Loaded {len(df)} samples from {label}")
#         else:
#             print(f"  ⚠ Skipping {label} — path not found")

#     if not dfs:
#         raise RuntimeError("No dataset found. Check your dataset/ directory.")

#     full_df = pd.concat(dfs, ignore_index=True)
#     full_df.dropna(subset=["hemoglobin", "age", "gender"], inplace=True)
#     print(f"  Total samples: {len(full_df)}")
#     return full_df


# # ──────────────────────────────────────────────────────────────
# # DATA GENERATORS
# # ──────────────────────────────────────────────────────────────

# class AnemiaDataset(tf.keras.utils.Sequence):
#     """
#     Keras Sequence that yields:
#       inputs  → [img1, img2, img3, meta]  (multi-input mode)
#                OR [img_concat, meta]      (single-input mode)
#       output  → hemoglobin value
#     """

#     def __init__(self, df: pd.DataFrame, batch_size: int = 16,
#                  target_col: str = "hemoglobin",
#                  augment: bool = False, shuffle: bool = True, **kwargs):
#         super().__init__(**kwargs)
#         self.df         = df.reset_index(drop=True)
#         self.batch_size = batch_size
#         self.target_col = target_col
#         self.augment    = augment
#         self.shuffle    = shuffle
#         self.indices    = np.arange(len(self.df))
#         if self.shuffle:
#             np.random.shuffle(self.indices)

#         # Augmentation pipeline
#         self.aug = ImageDataGenerator(
#             rotation_range=15,
#             width_shift_range=0.1,
#             height_shift_range=0.1,
#             horizontal_flip=True,
#             brightness_range=[0.8, 1.2],
#             zoom_range=0.1,
#         ) if augment else None

#     def __len__(self):
#         return int(np.ceil(len(self.df) / self.batch_size))

#     def __getitem__(self, idx):
#         batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch     = self.df.iloc[batch_idx]

#         imgs      = {t: [] for t in IMAGE_TYPES}
#         metas     = []
#         hgbs      = []

#         for _, row in batch.iterrows():
#             # Robust image selection: try first type, then fall back to others if needed
#             p = ""
#             for t in IMAGE_TYPES:
#                 path = row.get(f"img_{t}", "")
#                 if path and os.path.exists(path):
#                     p = path
#                     break
            
#             img = load_image(p) if p else np.zeros((*IMG_SIZE, 3), dtype=np.float32)
#             if self.augment and self.aug:
#                 img = self.aug.random_transform(img)
#             imgs[IMAGE_TYPES[0]].append(img) 

#             for t in IMAGE_TYPES[1:]:
#                 path = row.get(f"img_{t}", "")
#                 img_t = load_image(path) if path and os.path.exists(path) else np.zeros((*IMG_SIZE, 3), dtype=np.float32)
#                 if self.augment and self.aug:
#                     img_t = self.aug.random_transform(img_t)
#                 imgs[t].append(img_t)

#             # Meta: age/100, gender 0/1
#             metas.append([float(row["age"]) / 100.0, float(row["gender"])])
#             hgbs.append(float(row[self.target_col]))

#         img_arrays = [np.array(imgs[t], dtype=np.float32) for t in IMAGE_TYPES]
#         meta_array = np.array(metas, dtype=np.float32)
#         hgb_array  = np.array(hgbs,  dtype=np.float32)

#         # Standard dictionary format for Keras multi-input models
#         if USE_MULTI_INPUT:
#             inputs = {f"img_{t}": img_arrays[i] for i, t in enumerate(IMAGE_TYPES)}
#             inputs["meta_input"] = meta_array
#             return inputs, hgb_array
#         else:
#             return {"image_input": img_arrays[0], "meta_input": meta_array}, hgb_array

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)


# # ──────────────────────────────────────────────────────────────
# # SPLIT DATASET
# # ──────────────────────────────────────────────────────────────

# def get_splits(df: pd.DataFrame):
#     """Returns (train_df, val_df, test_df)."""
#     train_df, test_df = train_test_split(
#         df, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
#     train_df, val_df  = train_test_split(
#         train_df, test_size=VAL_SPLIT / (1 - TEST_SPLIT), random_state=RANDOM_SEED)
#     print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
#     return train_df, val_df, test_df











"""
data_loader.py — Dataset loading, preprocessing & augmentation
for EYES-DEFY-ANEMIA project
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from config import (
    INDIA_DIR, ITALY_DIR, INDIA_EXCEL, ITALY_EXCEL,
    COL_FOLDER, COL_AGE, COL_GENDER, COL_HGB,
    IMG_SIZE, IMAGE_TYPES, USE_MULTI_INPUT,
    VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def normalize_gender(val: str) -> int:
    """Convert gender string → 0 (Female) / 1 (Male)."""
    v = str(val).strip().lower()
    return 1 if v in ("m", "male", "1") else 0


def clean_to_float(val) -> float:
    """Convert string/number to float, handling commas if necessary."""
    if isinstance(val, str):
        val = val.replace(",", ".").strip()
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def load_image(path: str) -> np.ndarray:
    """Load & resize a single image to (224,224,3), normalised to [0,1]."""
    try:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        # Don't print error message for every load failure, already pre-validated
        return np.zeros((*IMG_SIZE, 3), dtype=np.float32)


def is_valid_image(path: str) -> bool:
    """Return True only if the file exists AND PIL can open it without errors."""
    if not path or not os.path.exists(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()        # header-only check — fast
        return True
    except Exception:
        return False


def load_sample_images(folder_path: str) -> dict:
    """
    Given a sample folder, load all requested image types.
    Returns dict  {image_type: np.ndarray}
    """
    images = {}
    for img_type in IMAGE_TYPES:
        img_path = os.path.join(folder_path, img_type)
        if os.path.exists(img_path):
            images[img_type] = load_image(img_path)
        else:
            # fill with zeros if a particular type is missing
            images[img_type] = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
    return images


# ──────────────────────────────────────────────────────────────
# LOAD ONE COUNTRY DATASET
# ──────────────────────────────────────────────────────────────

def load_country_data(country_dir: str, excel_path: str) -> pd.DataFrame:
    """
    Read the Excel label file and attach image paths for each sample.
    Returns a DataFrame with columns:
        folder, age, gender_raw, gender, hemoglobin, img_paths{...}
    """
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()

    # Rename to standard names
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("id", "folder", "subject", "no", "number"):
            rename_map[c] = "folder"
        elif cl in ("age",):
            rename_map[c] = "age"
        elif cl in ("sex", "gender"):
            rename_map[c] = "gender_raw"
        elif cl in ("hb", "hemoglobin", "haemoglobin", "hgb"):
            rename_map[c] = "hemoglobin"
    df.rename(columns=rename_map, inplace=True)

    required = {"folder", "age", "gender_raw", "hemoglobin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel columns not found: {missing}. "
                         f"Available columns: {list(df.columns)}")

    df["gender"] = df["gender_raw"].apply(normalize_gender)
    df["age"] = df["age"].apply(clean_to_float)
    df["hemoglobin"] = df["hemoglobin"].apply(clean_to_float)
    df["folder"] = df["folder"].astype(str).str.strip()

    # Build image paths
    valid_rows = []
    for _, row in df.iterrows():
        folder_path = os.path.join(country_dir, str(row["folder"]))
        if not os.path.isdir(folder_path):
            continue  # skip if folder doesn't exist
        
        row_dict = row.to_dict()
        row_dict["folder_path"] = folder_path
        
        # Search for files matching the types (handling prefixes)
        files = os.listdir(folder_path)
        any_valid = False
        
        for img_type in IMAGE_TYPES:
            best_match = ""
            for f in files:
                # Check for exact match or suffix match with underscore (e.g., '..._palpebral.png')
                if f == img_type or f.endswith("_" + img_type):
                    potential_path = os.path.join(folder_path, f)
                    if is_valid_image(potential_path):
                        best_match = potential_path
                        any_valid = True
                        break
            
            row_dict[f"img_{img_type}"] = best_match
            
        if not any_valid:
            # Only remove if NO and any version of the images is present or valid
            continue

        valid_rows.append(row_dict)

    return pd.DataFrame(valid_rows)


# ──────────────────────────────────────────────────────────────
# MERGE BOTH COUNTRIES
# ──────────────────────────────────────────────────────────────

def load_full_dataset() -> pd.DataFrame:
    """Load and combine India + Italy datasets."""
    dfs = []
    for country_dir, excel_path, label in [
        (INDIA_DIR, INDIA_EXCEL, "india"),
        (ITALY_DIR, ITALY_EXCEL, "italy"),
    ]:
        if os.path.exists(excel_path) and os.path.isdir(country_dir):
            df = load_country_data(country_dir, excel_path)
            df["country"] = label
            dfs.append(df)
            print(f"  ✓ Loaded {len(df)} samples from {label}")
        else:
            print(f"  ⚠ Skipping {label} — path not found")

    if not dfs:
        raise RuntimeError("No dataset found. Check your dataset/ directory.")

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.dropna(subset=["hemoglobin", "age", "gender"], inplace=True)
    print(f"  Total samples: {len(full_df)}")
    return full_df


# ──────────────────────────────────────────────────────────────
# DATA GENERATORS
# ──────────────────────────────────────────────────────────────

class AnemiaDataset(tf.keras.utils.Sequence):
    """
    Keras Sequence that yields:
      inputs  → [img1, img2, img3, meta]  (multi-input mode)
               OR [img_concat, meta]      (single-input mode)
      output  → hemoglobin value
    """

    def __init__(self, df: pd.DataFrame, batch_size: int = 16,
                 target_col: str = "hemoglobin",
                 augment: bool = False, shuffle: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.df         = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.target_col = target_col
        self.augment    = augment
        self.shuffle    = shuffle
        self.indices    = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Augmentation pipeline
        self.aug = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            zoom_range=0.1,
        ) if augment else None

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch     = self.df.iloc[batch_idx]

        imgs      = {t: [] for t in IMAGE_TYPES}
        metas     = []
        hgbs      = []

        for _, row in batch.iterrows():
            # Robust image selection: try first type, then fall back to others if needed
            p = ""
            for t in IMAGE_TYPES:
                path = row.get(f"img_{t}", "")
                if path and os.path.exists(path):
                    p = path
                    break
            
            img = load_image(p) if p else np.zeros((*IMG_SIZE, 3), dtype=np.float32)
            if self.augment and self.aug:
                img = self.aug.random_transform(img)
            imgs[IMAGE_TYPES[0]].append(img) 

            for t in IMAGE_TYPES[1:]:
                path = row.get(f"img_{t}", "")
                img_t = load_image(path) if path and os.path.exists(path) else np.zeros((*IMG_SIZE, 3), dtype=np.float32)
                if self.augment and self.aug:
                    img_t = self.aug.random_transform(img_t)
                imgs[t].append(img_t)

            # Meta: age/100, gender 0/1
            metas.append([float(row["age"]) / 100.0, float(row["gender"])])
            hgbs.append(float(row[self.target_col]))

        img_arrays = [np.array(imgs[t], dtype=np.float32) for t in IMAGE_TYPES]
        meta_array = np.array(metas, dtype=np.float32)
        hgb_array  = np.array(hgbs,  dtype=np.float32)

        # Standard dictionary format for Keras multi-input models
        if USE_MULTI_INPUT:
            inputs = {f"img_{t}": img_arrays[i] for i, t in enumerate(IMAGE_TYPES)}
            inputs["meta_input"] = meta_array
            return inputs, hgb_array
        else:
            return {"image_input": img_arrays[0], "meta_input": meta_array}, hgb_array

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ──────────────────────────────────────────────────────────────
# SPLIT DATASET
# ──────────────────────────────────────────────────────────────

def get_splits(df: pd.DataFrame):
    """Returns (train_df, val_df, test_df)."""
    train_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
    train_df, val_df  = train_test_split(
        train_df, test_size=VAL_SPLIT / (1 - TEST_SPLIT), random_state=RANDOM_SEED)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df