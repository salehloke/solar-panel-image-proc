#!/usr/bin/env python3
"""
Evaluation script for solar panel dirt detection model.
Evaluates model performance on test set and generates detailed metrics.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from src.data.dataset import SolarPanelDataset
from pytorch.models.resnet_model import load_model


def get_test_transforms():
    """Get test transforms (same as validation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions and probabilities."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
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
    
    return cm


def plot_roc_curve(y_true, y_probs, class_names, save_path):
    """Plot and save ROC curves."""
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        # For binary classification, we can use the positive class probabilities
        if len(class_names) == 2:
            if i == 1:  # Positive class (dirty)
                y_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        else:
            # For multi-class, use one-vs-rest
            y_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(y_true, y_probs, class_names, save_path):
    """Plot and save Precision-Recall curves."""
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        if len(class_names) == 2:
            if i == 1:  # Positive class (dirty)
                y_binary = (y_true == i).astype(int)
                precision, recall, _ = precision_recall_curve(y_binary, y_probs[:, i])
                avg_precision = average_precision_score(y_binary, y_probs[:, i])
                
                plt.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.3f})')
        else:
            y_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_binary, y_probs[:, i])
            avg_precision = average_precision_score(y_binary, y_probs[:, i])
            
            plt.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_metrics(y_true, y_pred, y_probs, class_names):
    """Calculate comprehensive metrics."""
    # Basic metrics
    accuracy = (y_true == y_pred).mean()
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC and PR metrics for binary classification
    roc_metrics = {}
    pr_metrics = {}
    
    if len(class_names) == 2:
        # For binary classification, focus on the positive class (dirty)
        y_binary = (y_true == 1).astype(int)  # Assuming dirty is class 1
        fpr, tpr, _ = roc_curve(y_binary, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(y_binary, y_probs[:, 1])
        
        roc_metrics = {
            'auc': roc_auc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        pr_metrics = {
            'average_precision': avg_precision
        }
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc_metrics': roc_metrics,
        'pr_metrics': pr_metrics
    }


def save_results(metrics, output_dir):
    """Save evaluation results to JSON file."""
    output_path = output_dir / 'evaluation_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    results = {
        'accuracy': float(metrics['accuracy']),
        'classification_report': metrics['classification_report'],
        'confusion_matrix': metrics['confusion_matrix'],
        'roc_metrics': metrics['roc_metrics'],
        'pr_metrics': metrics['pr_metrics']
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def print_summary(metrics, class_names):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"Overall Accuracy: {metrics['accuracy']:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(
        metrics['classification_report']['support'], 
        metrics['classification_report']['support'],  # This is a placeholder
        target_names=class_names
    ))
    
    if metrics['roc_metrics']:
        print(f"ROC AUC: {metrics['roc_metrics']['auc']:.3f}")
    
    if metrics['pr_metrics']:
        print(f"Average Precision: {metrics['pr_metrics']['average_precision']:.3f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(cm)
    
    # Calculate per-class metrics
    print(f"\nPer-class Metrics:")
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {class_name}:")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")
        print(f"    F1-Score: {f1:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate solar panel dirt detection model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--data_dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--output_dir", default="evaluation_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset
    test_transform = get_test_transforms()
    test_dataset = SolarPanelDataset(Path(args.data_dir) / "test", transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, num_classes=len(test_dataset.classes))
    model.to(device)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_probs, test_dataset.classes)
    
    # Generate plots
    print("Generating plots...")
    plot_confusion_matrix(
        y_true, y_pred, test_dataset.classes, 
        output_dir / 'confusion_matrix.png'
    )
    
    plot_roc_curve(
        y_true, y_probs, test_dataset.classes, 
        output_dir / 'roc_curves.png'
    )
    
    plot_precision_recall_curve(
        y_true, y_probs, test_dataset.classes, 
        output_dir / 'precision_recall_curves.png'
    )
    
    # Save results
    save_results(metrics, output_dir)
    
    # Print summary
    print_summary(metrics, test_dataset.classes)
    
    print(f"\nEvaluation completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 