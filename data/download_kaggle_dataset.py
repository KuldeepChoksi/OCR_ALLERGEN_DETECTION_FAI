import kagglehub

# Download latest version
path = kagglehub.dataset_download("harieh/ocr-dataset")

print("Path to dataset files:", path)
