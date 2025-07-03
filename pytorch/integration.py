"""
Integration module for connecting PyTorch models with the FastAPI backend.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add the project root to the path to enable importing from pytorch module
sys.path.append(str(Path(__file__).parent.parent))

from pytorch.models.resnet_model import load_model

logger = logging.getLogger(__name__)

def load_model_for_inference(model_path=None):
    """
    Load a PyTorch model for inference in the FastAPI backend
    
    Args:
        model_path: Path to the trained model weights. If None, will use the default path.
        
    Returns:
        The loaded PyTorch model ready for inference
    """
    try:
        if model_path is None:
            # Default model path
            model_dir = os.environ.get("MODEL_DIR", "./models")
            model_path = os.environ.get("MODEL_PATH", f"{model_dir}/resnet18_solar_panel.pt")
        
        # Check if model exists
        if not Path(model_path).exists():
            logger.warning(f"Model file {model_path} not found.")
            return None
            
        # Load the model
        model = load_model(model_path)
        model.eval()  # Set to evaluation mode
        return model
        
    except Exception as e:
        logger.error(f"Error loading model for inference: {str(e)}")
        return None

def process_image_for_prediction(image, model):
    """
    Process an image and get a prediction from the model
    
    Args:
        image: Processed image tensor
        model: Loaded PyTorch model
        
    Returns:
        Dictionary with prediction results
    """
    try:
        with torch.no_grad():
            # Add batch dimension
            image = image.unsqueeze(0)
            
            # Get model predictions
            outputs = model(image)
            
            # Get the probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
            result = {
                "class_index": predicted_class.item(),
                "class_name": "dirty" if predicted_class.item() == 1 else "clean",
                "confidence": confidence.item()
            }
            
            return result
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
