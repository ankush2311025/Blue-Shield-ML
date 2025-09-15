# utils/ocean_disaster_predict.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import os


# Define your model class
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
# Load model (dynamic path)
def load_model(model_path):
    try:
        # Comprehensive list of safe globals for ResNet
        safe_globals_list = [
            # Basic torch modules
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.pooling.AdaptiveAvgPool2d,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.container.Sequential,
            
            # ResNet specific modules
            models.resnet.ResNet,
            models.resnet.BasicBlock,
            
            # PyTorch internal classes
            torch.Tensor,
            torch._C._TensorBase,
            torch.Size,
            torch.dtype,
            torch.device,
        ]
        
        # Try to load the complete model with weights_only=True
        with torch.serialization.safe_globals(safe_globals_list):
            model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        
        print("✅ Complete model loaded successfully with weights_only=True")
        
    except Exception as e:
        print(f"weights_only=True failed: {e}")
        
        try:
            # Fallback to weights_only=False
            model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
            print("✅ Complete model loaded successfully with weights_only=False")
            
        except Exception as e2:
            print(f"weights_only=False also failed: {e2}")
            print("⚠️  Creating new ResNet18 model with random weights as fallback")
            # Create a completely new model
            model = models.resnet18(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 4)  

    # Ensure the model is in evaluation mode
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Predict Image
def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    return int(predicted_class.item()), float(confidence.item())


# Predict Video
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
