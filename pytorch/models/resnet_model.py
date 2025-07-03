import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class SolarPanelClassifier(nn.Module):
    """
    ResNet18-based classifier for solar panel dirt detection
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SolarPanelClassifier, self).__init__()
        # Load a pretrained ResNet18 model with newer API
        if pretrained:
            try:
                self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception as e:
                print(f"Warning: Could not load pretrained weights due to SSL issue: {e}")
                print("Falling back to random initialization...")
                self.model = models.resnet18(weights=None)
        else:
            self.model = models.resnet18(weights=None)
        
        # Get the number of features in the last layer
        num_features = self.model.fc.in_features
        
        # Replace the final fully connected layer for our classification task
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def create_model(num_classes=2, pretrained=True):
    """
    Factory function to create a new SolarPanelClassifier instance
    
    Args:
        num_classes: Number of output classes (default: 2 for clean/dirty)
        pretrained: Whether to use pretrained weights (default: True)
        
    Returns:
        A new SolarPanelClassifier model
    """
    return SolarPanelClassifier(num_classes=num_classes, pretrained=pretrained)

def save_model(model, path):
    """
    Save the trained model to the specified path
    
    Args:
        model: PyTorch model to save
        path: Path where the model will be saved
    """
    try:
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved successfully to {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise

def load_model(path, num_classes=2):
    """
    Load a trained model from the specified path
    
    Args:
        path: Path to the saved model weights
        num_classes: Number of output classes (default: 2 for clean/dirty)
        
    Returns:
        The loaded PyTorch model
    """
    try:
        model = create_model(num_classes=num_classes, pretrained=False)
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
