import kagglehub

# Download latest version
path = kagglehub.dataset_download("varpit94/disaster-images-dataset")

print("Path to dataset files:", path)