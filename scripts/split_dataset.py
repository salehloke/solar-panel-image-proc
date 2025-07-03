#!/usr/bin/env python3
"""
Dataset splitting script for solar panel cleanliness detection.
Splits the dataset into train/validation/test sets with stratification.
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse

def split_dataset(data_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train/validation/test sets with stratification.
    
    Args:
        data_root: Root directory containing clean/ and dirty/ folders
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.15)
        test_ratio: Ratio for test set (default: 0.15)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    data_root = Path(data_root)
    clean_dir = data_root / "clean"
    dirty_dir = data_root / "dirty"
    
    # Get all image paths
    clean_images = list(clean_dir.glob("*.jpg"))
    dirty_images = list(dirty_dir.glob("*.jpg"))
    
    print(f"Found {len(clean_images)} clean images and {len(dirty_images)} dirty images")
    
    # Create labels for stratification
    clean_labels = [0] * len(clean_images)  # 0 for clean
    dirty_labels = [1] * len(dirty_images)  # 1 for dirty
    
    all_images = clean_images + dirty_images
    all_labels = clean_labels + dirty_labels
    
    # Split into train and temp (val + test)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images, all_labels, 
        test_size=(val_ratio + test_ratio), 
        stratify=all_labels, 
        random_state=seed
    )
    
    # Split temp into val and test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_labels,
        random_state=seed
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    # Create output directories
    output_dirs = {
        "train": data_root.parent / "processed" / "train",
        "val": data_root.parent / "processed" / "val", 
        "test": data_root.parent / "processed" / "test"
    }
    
    for split_name, output_dir in output_dirs.items():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "clean").mkdir(exist_ok=True)
        (output_dir / "dirty").mkdir(exist_ok=True)
    
    # Copy files to appropriate directories
    splits = {
        "train": (train_images, train_labels),
        "val": (val_images, val_labels),
        "test": (test_images, test_labels)
    }
    
    for split_name, (images, labels) in splits.items():
        output_dir = output_dirs[split_name]
        
        for img_path, label in zip(images, labels):
            class_name = "clean" if label == 0 else "dirty"
            dest_path = output_dir / class_name / img_path.name
            
            # Copy file
            shutil.copy2(img_path, dest_path)
        
        # Print class distribution for this split
        clean_count = sum(1 for label in labels if label == 0)
        dirty_count = sum(1 for label in labels if label == 1)
        print(f"  {split_name.capitalize()} - Clean: {clean_count}, Dirty: {dirty_count}")
    
    print(f"\nDataset split completed! Files saved to {data_root.parent / 'processed'}")

def main():
    parser = argparse.ArgumentParser(description="Split solar panel dataset into train/val/test sets")
    parser.add_argument("--data_root", default="data/train", help="Root directory with clean/ and dirty/ folders")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    split_dataset(args.data_root, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

if __name__ == "__main__":
    main()
