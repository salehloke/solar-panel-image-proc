#!/usr/bin/env python3
"""
Complete training pipeline for solar panel dirt detection.
Orchestrates the entire process from data preparation to model evaluation.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    package_mappings = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'numpy': 'numpy',
        'pillow': 'PIL',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn'
    }
    
    missing_packages = []
    for package_name, import_name in package_mappings.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_data_structure():
    """Check if data structure is correct."""
    data_dir = Path("data")
    train_dir = data_dir / "train"
    
    if not train_dir.exists():
        print("❌ data/train directory not found")
        return False
    
    clean_dir = train_dir / "clean"
    dirty_dir = train_dir / "dirty"
    
    if not clean_dir.exists() or not dirty_dir.exists():
        print("❌ data/train/clean or data/train/dirty directory not found")
        return False
    
    clean_count = len(list(clean_dir.glob("*.jpg")))
    dirty_count = len(list(dirty_dir.glob("*.jpg")))
    
    if clean_count == 0 or dirty_count == 0:
        print("❌ No images found in clean or dirty directories")
        return False
    
    print(f"✅ Data structure is correct")
    print(f"   Clean images: {clean_count}")
    print(f"   Dirty images: {dirty_count}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Complete training pipeline for solar panel dirt detection")
    parser.add_argument("--skip_data_split", action="store_true", help="Skip data splitting step")
    parser.add_argument("--skip_training", action="store_true", help="Skip training step")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation step")
    parser.add_argument("--model_name", default="resnet18_solar_panel", help="Model name")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalance")
    
    args = parser.parse_args()
    
    print("🚀 Starting Solar Panel Dirt Detection Training Pipeline")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check dependencies
    print("\n📋 Step 1: Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Step 2: Check data structure
    print("\n📋 Step 2: Checking data structure...")
    if not check_data_structure():
        sys.exit(1)
    
    # Step 3: Split dataset
    if not args.skip_data_split:
        print("\n📋 Step 3: Splitting dataset...")
        split_command = f"python3 scripts/split_dataset.py --data_root data/train"
        if not run_command(split_command, "Dataset splitting"):
            print("❌ Dataset splitting failed")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping dataset splitting...")
    
    # Step 4: Train model
    if not args.skip_training:
        print("\n📋 Step 4: Training model...")
        train_command = f"python3 src/train.py --data_dir data/processed --epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} --model_name {args.model_name}"
        
        if args.use_class_weights:
            train_command += " --use_class_weights"
        
        if not run_command(train_command, "Model training"):
            print("❌ Model training failed")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping model training...")
    
    # Step 5: Find best model
    print("\n📋 Step 5: Finding best model...")
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("❌ No logs directory found. Training may have failed.")
        sys.exit(1)
    
    # Find the most recent training run
    model_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and args.model_name in d.name]
    if not model_dirs:
        print("❌ No model directories found")
        sys.exit(1)
    
    latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    best_model_path = latest_model_dir / f"{args.model_name}_best.pt"
    
    if not best_model_path.exists():
        print(f"❌ Best model not found at {best_model_path}")
        sys.exit(1)
    
    print(f"✅ Found best model: {best_model_path}")
    
    # Step 6: Evaluate model
    if not args.skip_evaluation:
        print("\n📋 Step 6: Evaluating model...")
        eval_command = f"python3 src/evaluate.py --model_path {best_model_path} --data_dir data/processed --output_dir evaluation_results"
        
        if not run_command(eval_command, "Model evaluation"):
            print("❌ Model evaluation failed")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping model evaluation...")
    
    # Step 7: Copy best model to models directory
    print("\n📋 Step 7: Copying best model to models directory...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    final_model_path = models_dir / f"{args.model_name}.pt"
    import shutil
    shutil.copy2(best_model_path, final_model_path)
    print(f"✅ Best model copied to: {final_model_path}")
    
    # Step 8: Generate summary report
    print("\n📋 Step 8: Generating summary report...")
    summary_path = Path("training_summary.md")
    
    with open(summary_path, "w") as f:
        f.write("# Solar Panel Dirt Detection - Training Summary\n\n")
        f.write(f"**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model Name:** {args.model_name}\n\n")
        f.write(f"**Training Logs:** {latest_model_dir}\n\n")
        f.write(f"**Best Model:** {final_model_path}\n\n")
        
        if not args.skip_evaluation:
            f.write("**Evaluation Results:** evaluation_results/\n\n")
        
        f.write("## Training Configuration\n\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Learning Rate: {args.lr}\n")
        f.write(f"- Class Weights: {'Yes' if args.use_class_weights else 'No'}\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Test the API with the trained model\n")
        f.write("2. Deploy the model using Docker\n")
        f.write("3. Create a web interface for easy interaction\n")
        f.write("4. Set up continuous monitoring\n")
    
    print(f"✅ Summary report saved to: {summary_path}")
    
    print("\n🎉 Training pipeline completed successfully!")
    print(f"\n📁 Key files:")
    print(f"   Best model: {final_model_path}")
    print(f"   Training logs: {latest_model_dir}")
    if not args.skip_evaluation:
        print(f"   Evaluation results: evaluation_results/")
    print(f"   Summary report: {summary_path}")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Test the API: python backend/run.py")
    print(f"   2. Deploy with Docker: docker-compose up")
    print(f"   3. Create web interface for easy interaction")

if __name__ == "__main__":
    main() 