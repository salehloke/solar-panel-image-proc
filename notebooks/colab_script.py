# ==============================================================================
# SOLAR PANEL DIRT DETECTION - COLAB SCRIPT (EXPERT NARRATIVE VERSION)
# ==============================================================================
# This script is organized into the 5 standard phases of a Machine Learning Project.
# Copy each block into a separate cell in Google Colab.

# ==============================================================================
# PHASE 1: DATA ACQUISITION & SETUP
# "Getting the raw material ready for the AI"
# ==============================================================================
import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Configure environment
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ PHASE 1 COMPLETE: Setup ready using {device}")

# SIMULATING DATA COLLECTION: In Colab, you would download your zip file here.
def create_demo_dataset(root_dir="data/raw"):
    root = Path(root_dir)
    if root.exists(): shutil.rmtree(root)
    (root / "clean").mkdir(parents=True)
    (root / "dirty").mkdir(parents=True)
    for i in range(20):
        # Generate random images to represent solar panels
        clean_img = Image.fromarray(np.random.randint(150, 255, (224, 224, 3), dtype=np.uint8)) # Lighter
        dirty_img = Image.fromarray(np.random.randint(0, 100, (224, 224, 3), dtype=np.uint8))   # Darker/Muddy
        clean_img.save(root / "clean" / f"clean_{i}.jpg")
        dirty_img.save(root / "dirty" / f"dirty_{i}.jpg")
    print(f"✅ Dataset simulated at {root}")

create_demo_dataset()


# ==============================================================================
# PHASE 2: PREPROCESSING & SPLITTING
# "Cleaning the data and dividing it into Study, Practice, and Final Exam buckets"
# ==============================================================================
def split_data(source_dir, dest_dir, ratios=(0.7, 0.15, 0.15)):
    source, dest = Path(source_dir), Path(dest_dir)
    if dest.exists(): shutil.rmtree(dest)
    
    for cls in ["clean", "dirty"]:
        imgs = list((source / cls).glob("*.jpg"))
        train, test = train_test_split(imgs, test_size=1-ratios[0], random_state=42)
        val, test = train_test_split(test, test_size=ratios[2]/(ratios[1]+ratios[2]), random_state=42)
        
        for subset, subset_imgs in zip(["train", "val", "test"], [train, val, test]):
            (dest / subset / cls).mkdir(parents=True, exist_ok=True)
            for img in subset_imgs: shutil.copy(img, dest / subset / cls / img.name)
    print("✅ PHASE 2 COMPLETE: Data split into Train(70%), Val(15%), and Test(15%)")

split_data("data/raw", "data/processed")

# CUSTOM DATASET HANDLER
class SolarPanelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        for cls_idx, cls_name in enumerate(self.classes):
            for img_path in (self.root / cls_name).glob("*.jpg"):
                self.samples.append((img_path, cls_idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

# DATA TRANSFORMS (Resize & Math Normalization)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

loaders = {x: DataLoader(SolarPanelDataset(f"data/processed/{x}", data_transforms[x]), batch_size=4, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(loaders[x].dataset) for x in ['train', 'val']}
class_names = ['clean', 'dirty']


# ==============================================================================
# PHASE 3: TRAINING
# "Teaching the AI to recognize dirt through repetition"
# ==============================================================================
# Using ResNet18: A proven architecture for image recognition
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters(): param.requires_grad = False # Freeze base
model.fc = nn.Linear(model.fc.in_features, 2) # Customize the "Brain" for our 2 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

def train_ai(num_epochs=5):
    history = {'loss': [], 'acc': []}
    for epoch in range(num_epochs):
        model.train()
        run_loss, run_corrects = 0.0, 0
        for inputs, labels in loaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * inputs.size(0)
            run_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = run_loss / dataset_sizes['train']
        epoch_acc = run_corrects.double() / dataset_sizes['train']
        history['loss'].append(epoch_loss); history['acc'].append(epoch_acc.item())
        print(f'Epoch {epoch+1}: Loss {epoch_loss:.4f} Acc {epoch_acc:.4f}')
    print("✅ PHASE 3 COMPLETE: AI Training Finished")
    return history

history = train_ai()


# ==============================================================================
# PHASE 4: EVALUATION
# "The Report Card - Visualizing how well the AI learned"
# ==============================================================================
def show_results(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history['loss']); plt.title('Training Loss (Wrongness)')
    plt.subplot(1, 2, 2); plt.plot(history['acc']); plt.title('Training Accuracy (%)')
    plt.show()

show_results(history)
print("✅ PHASE 4 COMPLETE: Learning curves visualized")


# ==============================================================================
# PHASE 5: INFERENCE
# "Applying the AI to real-world images"
# ==============================================================================
def predict_panel(image_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_t = data_transforms['val'](img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, 1)
        conf, pred = torch.max(probs, 1)
    
    result = class_names[pred.item()]
    plt.imshow(img); plt.title(f"AI STATUS: {result.upper()} ({conf.item():.2%})")
    plt.axis('off'); plt.show()

# Test the final system
sample_img = list(Path("data/processed/val/dirty").glob("*.jpg"))[0]
predict_panel(sample_img)
print("✅ PHASE 5 COMPLETE: Ready for real-world testing")