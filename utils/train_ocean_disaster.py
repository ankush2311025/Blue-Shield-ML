import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError

# ----------------------------
# Config
# ----------------------------
DATASET_DIR = r"C:\Users\ankus\Desktop\DistasterML\data\disaster-images-dataset"
MODEL_SAVE_PATH = "models/ocean_disaster_model.pth"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
IMG_SIZE = 224
OCEAN_CLASSES = ['flood', 'cyclone', 'tsunami', 'storm', 'none']

# Mapping existing folders to ocean disaster classes
FOLDER_TO_OCEAN_CLASS = {
    'Damaged_Infrastructure': 'cyclone',
    'Fire_Disaster': 'storm',
    'Human_Damage': 'none',
    'Land_Disaster': 'storm',
    'Non_Damage': 'none',
    'Water_Disaster': 'flood'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# Custom Dataset
# ----------------------------
class OceanDisasterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            ocean_class = FOLDER_TO_OCEAN_CLASS.get(folder_name, 'none')
            label_idx = OCEAN_CLASSES.index(ocean_class)

            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(folder_path, img_file))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Warning: Unable to open image {img_path}. Skipping.")
            return torch.zeros(3, 224, 224), label  # Return a dummy tensor
        if self.transform:
            image = self.transform(image)
        return image, label

# ----------------------------
# Load dataset
# ----------------------------
dataset = OceanDisasterDataset(DATASET_DIR, transform=transform)

# Split train / validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# Model
# ----------------------------
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(OCEAN_CLASSES))
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

# ----------------------------
# Validation
# ----------------------------
model.eval()
val_corrects = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        val_corrects += torch.sum(preds == labels.data)

val_acc = val_corrects.double() / val_size
print(f"Validation Accuracy: {val_acc:.4f}")

# ----------------------------
# Save model
# ----------------------------
MODEL_SAVE_PATH = r"C:\Users\ankus\Desktop\DistasterML\models\ocean_disaster_model.pth"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model, MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
