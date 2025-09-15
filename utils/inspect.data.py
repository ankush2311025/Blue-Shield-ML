import os

dataset_dir = r"C:\Users\ankus\.cache\kagglehub\datasets\varpit94\disaster-images-dataset\versions\1\Comprehensive Disaster Dataset(CDD)"

for root, dirs, files in os.walk(dataset_dir):
    print("ROOT:", root)
    print("DIRS:", dirs)
    print("FILES:", files[:5])  # show first 5 files if any
    print("-"*50)
