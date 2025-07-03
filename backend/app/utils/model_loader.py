import sys
import os
from pathlib import Path
import logging

# Add the project root to the path to enable importing from pytorch module
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import from the pytorch module
from pytorch.models.resnet_model import load_model as load_pytorch_model
from pytorch.integration import load_model_for_inference, process_image_for_prediction

logger = logging.getLogger(__name__)

def load_model(model_path: str, num_classes: int = 2):
    """
    Load a pretrained ResNet18 model for solar panel dirt detection
    
    Args:
        model_path: Path to the saved model weights
        num_classes: Number of output classes (default: 2 for clean/dirty)
        
    Returns:
        The loaded PyTorch model
    """
    try:
        # Use the integration module to load the model
        model = load_model_for_inference(model_path)
        
        if model is None:
            logger.warning(f"Could not load model from {model_path}. Using fallback approach.")
            # Fallback to direct loading if the integration module failed
            model = load_pytorch_model(model_path, num_classes=num_classes)
            
        logger.info("Model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
        
def predict_image(model, image_tensor):
    """
    Make a prediction using the loaded model
    
    Args:
        model: Loaded PyTorch model
        image_tensor: Preprocessed image tensor
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Use the integration module to process the image
        return process_image_for_prediction(image_tensor, model)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise
