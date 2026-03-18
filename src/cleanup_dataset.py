import os
import shutil
from PIL import Image
from tqdm import tqdm

DATASET_ROOT = r'd:\Anemia_prediction\dataset'
CORRUPTED_ROOT = r'd:\Anemia_prediction\dataset_corrupted'

if not os.path.exists(CORRUPTED_ROOT):
    os.makedirs(CORRUPTED_ROOT)

def is_corrupted(filepath):
    """Checks if an image file is corrupted or cannot be identified by PIL."""
    if not os.path.exists(filepath):
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()  # Very fast, just checks headers
        # Some corruptions are not caught by verify(), try to open fully
        with Image.open(filepath) as img:
            img.load()  # Actually loads pixels
        return False
    except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
        print(f"  [!] Identified corrupted image: {filepath} ({e})")
        return True
    except Exception as e:
        print(f"  [!] Error checking image {filepath}: {e}")
        return True

def cleanup_dataset():
    """Iterates through the dataset and moves any corrupted images to CORRUPTED_ROOT."""
    print(f"Scanning dataset in {DATASET_ROOT} for corrupted files...")
    corrupted_count = 0
    total_scanned = 0
    
    for root, dirs, files in os.walk(DATASET_ROOT):
        # Skip the corrupted folder if it's somehow inside the root
        if os.path.abspath(root).startswith(os.path.abspath(CORRUPTED_ROOT)):
            continue
            
        for name in files:
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                total_scanned += 1
                filepath = os.path.join(root, name)
                if is_corrupted(filepath):
                    # Maintain subdirectory structure in corrupted folder for traceablity
                    relative_path = os.path.relpath(root, DATASET_ROOT)
                    target_dir = os.path.join(CORRUPTED_ROOT, relative_path)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    
                    target_path = os.path.join(target_dir, name)
                    try:
                        shutil.move(filepath, target_path)
                        print(f"  [✓] Moved to: {target_path}")
                        corrupted_count += 1
                    except Exception as e:
                        print(f"  [X] Failed to move {filepath}: {e}")

    print("-" * 50)
    print(f"Cleanup complete.")
    print(f"Total images scanned: {total_scanned}")
    print(f"Corrupted images moved: {corrupted_count}")
    print(f"Corrupted images can be found in: {CORRUPTED_ROOT}")

if __name__ == "__main__":
    cleanup_dataset()
