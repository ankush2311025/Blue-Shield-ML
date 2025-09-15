import os
from PIL import Image, UnidentifiedImageError

dataset_path = r"C:\Users\ankus\Desktop\DistasterML\data\disaster-images-dataset"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()  # Just checks, doesn’t load fully
            except (UnidentifiedImageError, OSError):
                print(f"❌ Corrupted image found: {path}")
