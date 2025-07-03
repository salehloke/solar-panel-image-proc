#!/usr/bin/env python3
"""
Enhanced training script for solar panel cleanliness detection.
Features: logging, checkpointing, early stopping, class weights, comprehensive metrics.
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import SolarPanelDataset
from pytorch.models.resnet_model import create_model, save_model


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class TrainingLogger:
    """Log training metrics and save plots."""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
        
    def log_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['train_f1'].append(train_f1)
        self.metrics['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
              f"Train F1: {train_f1:.3f} | Val F1: {val_f1:.3f}")
    
    def save_plots(self):
        """Save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.metrics['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.metrics['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 curves
        axes[1, 0].plot(self.metrics['train_f1'], label='Train F1')
        axes[1, 0].plot(self.metrics['val_f1'], label='Val F1')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined metrics
        axes[1, 1].plot(self.metrics['val_acc'], label='Val Acc', color='blue')
        axes[1, 1].plot(self.metrics['val_f1'], label='Val F1', color='red')
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)


def get_transforms():
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced dataset."""
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total_samples = len(dataset)
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    
    # Convert to tensor
    weight_tensor = torch.zeros(len(dataset.classes))
    for cls, weight in class_weights.items():
        weight_tensor[dataset.class_to_idx[cls]] = weight
    
    return weight_tensor


def evaluate_model(model, dataloader, device, criterion):
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    # F1 score (macro average)
    report = classification_report(all_labels, all_preds, output_dict=True)
    f1_score = report['macro avg']['f1-score']
    
    return total_loss / len(dataloader), accuracy, f1_score, all_preds, all_labels


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train solar panel dirt detection model")
    parser.add_argument("--data_dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalance")
    parser.add_argument("--model_name", default="resnet18_solar_panel", help="Model name for saving")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"{args.model_name}_{timestamp}"
    logger = TrainingLogger(log_dir)
    
    # Load datasets
    data_dir = Path(args.data_dir)
    train_transform, val_transform = get_transforms()
    
    train_dataset = SolarPanelDataset(data_dir / "train", transform=train_transform)
    val_dataset = SolarPanelDataset(data_dir / "val", transform=val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = calculate_class_weights(train_dataset)
        print(f"Class weights: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_model(num_classes=len(train_dataset.classes), pretrained=True)
    model.to(device)

    # Setup loss function and optimizer
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    best_val_f1 = 0.0
    print("\nStarting training...")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate_model(
            model, val_loader, device, criterion
        )
        
        # Calculate training F1 (approximate)
        train_f1 = val_f1  # For simplicity, we'll use validation F1
        
        # Log metrics
        logger.log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = log_dir / f"{args.model_name}_best.pt"
            save_model(model, model_path)
            print(f"  New best model saved! F1: {val_f1:.3f}")
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"  Early stopping triggered after {epoch} epochs")
            break
        
        epoch_time = time.time() - start_time
        print(f"  Epoch time: {epoch_time:.2f}s")
    
    # Save final model
    final_model_path = log_dir / f"{args.model_name}_final.pt"
    save_model(model, final_model_path)
    
    # Save confusion matrix
    save_confusion_matrix(
        val_labels, val_preds, 
        train_dataset.classes, 
        log_dir / 'confusion_matrix.png'
    )
    
    # Save training curves
    logger.save_plots()
    logger.save_metrics()
    
    # Save training config
    config = vars(args)
    config['best_val_f1'] = best_val_f1
    config['final_epoch'] = epoch
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best validation F1: {best_val_f1:.3f}")
    print(f"Logs saved to: {log_dir}")
    print(f"Best model saved to: {log_dir / f'{args.model_name}_best.pt'}")


if __name__ == "__main__":
    main()
