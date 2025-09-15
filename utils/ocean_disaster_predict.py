# utils/ocean_disaster_predict.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os

# ----------------------------
# Define your model class
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):  # adjust classes
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),  # adjust based on training
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------------
# Load model (dynamic path)
# ----------------------------
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = SimpleCNN(num_classes=4)  # adjust to your training
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ----------------------------
# Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Predict Image
# ----------------------------
def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    return int(predicted_class.item()), float(confidence.item())

# ----------------------------
# Predict Video
# ----------------------------
def predict_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, frame_count // 10)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transform(frame).unsqueeze(0)

            with torch.no_grad():
                outputs = model(frame)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, 1)
                frame_predictions.append((int(predicted_class.item()), float(confidence.item())))
        count += 1
    
    cap.release()

    if frame_predictions:
        from collections import Counter
        classes = [p[0] for p in frame_predictions]
        most_common_class, _ = Counter(classes).most_common(1)[0]
        avg_confidence = sum([p[1] for p in frame_predictions if p[0]==most_common_class]) / classes.count(most_common_class)
    else:
        most_common_class, avg_confidence = -1, 0.0

    return most_common_class, avg_confidence
