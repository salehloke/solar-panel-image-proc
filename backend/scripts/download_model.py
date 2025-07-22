#!/usr/bin/env python3
"""
Script to pre-download PyTorch models for faster container startup
"""

import torch
import torchvision
import os
from pathlib import Path

def download_resnet18():
    """Download ResNet18 model and cache it"""
    print("🔄 Downloading ResNet18 model...")
    
    try:
        # This will download and cache the model
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        print("✅ ResNet18 model downloaded successfully!")
        
        # Save the model to our models directory
        models_dir = Path("/app/models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / "resnet18_pretrained.pt"
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def download_common_models():
    """Download common models that might be needed"""
    models = [
        ('resnet18', 'pytorch/vision:v0.10.0'),
        ('resnet50', 'pytorch/vision:v0.10.0'),
        ('mobilenet_v2', 'pytorch/vision:v0.10.0'),
    ]
    
    for model_name, repo in models:
        print(f"🔄 Downloading {model_name}...")
        try:
            model = torch.hub.load(repo, model_name, pretrained=True)
            print(f"✅ {model_name} downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

if __name__ == "__main__":
    print("🚀 Starting model download process...")
    
    # Download ResNet18 for our specific use case
    success = download_resnet18()
    
    if success:
        print("🎉 Model download completed successfully!")
    else:
        print("⚠️  Model download failed, but container will still work")
    
    # Optionally download other common models
    # download_common_models() 