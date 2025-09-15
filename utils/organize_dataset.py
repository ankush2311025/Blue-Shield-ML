import os
import shutil

# Source dataset (KaggleHub cache)
dataset_dir = r"C:\Users\ankus\.cache\kagglehub\datasets\varpit94\disaster-images-dataset\versions\1\Comprehensive Disaster Dataset(CDD)"

# Target folder where you want to see dataset on Desktop
target_dir = r"C:\Users\ankus\Desktop\disaster-images-dataset"
os.makedirs(target_dir, exist_ok=True)

# Walk recursively and copy images
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".png")):
            # Determine relative path to maintain folder structure
            rel_path = os.path.relpath(root, dataset_dir)
            dest_folder = os.path.join(target_dir, rel_path)
            os.makedirs(dest_folder, exist_ok=True)
            
            # Copy file
            shutil.copy2(os.path.join(root, file), os.path.join(dest_folder, file))

print("✅ Dataset copied and organized successfully!")
