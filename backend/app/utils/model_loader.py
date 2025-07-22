import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import logging
import os
from pathlib import Path
import numpy as np
import io
import torchvision.models as models
from typing import Optional, Tuple
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ResNet18SolarPanel(nn.Module):
    """
    ResNet18 model for solar panel dirt detection
    """
    def __init__(self, num_classes=2):
        super(ResNet18SolarPanel, self).__init__()
        # Use pretrained ResNet18
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
        # Freeze the feature extraction layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the final layer for our binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

class DynamicModelLoader:
    """
    Dynamic model loader that supports different loading modes:
    - volume: Load from Docker volume (Solution 2)
    - build_time: Load from pre-copied model file (Solution 3)
    - runtime: Download at runtime (fallback)
    """
    
    def __init__(self):
        self.model_loading_mode = os.getenv('MODEL_LOADING_MODE', 'volume').lower()
        self.model_name = os.getenv('MODEL_NAME', 'resnet18_solar_panel.pt')
        self.model_path = os.getenv('MODEL_PATH', '/app/models')
        self.model_save_path = os.getenv('MODEL_SAVE_PATH', f'/app/models/{self.model_name}')
        self.model_download_url = os.getenv('MODEL_DOWNLOAD_URL', 'https://download.pytorch.org/models/resnet18-5c106cde.pth')
        self.model_cache_dir = os.getenv('MODEL_CACHE_DIR', '/home/appuser/.cache/torch/hub')
        
        logger.info(f"Model Loader initialized with mode: {self.model_loading_mode}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Model save path: {self.model_save_path}")
    
    def load_model(self) -> torch.nn.Module:
        """
        Load the model based on the configured loading mode
        """
        logger.info(f"Loading model using mode: {self.model_loading_mode}")
        
        if self.model_loading_mode == 'volume':
            return self._load_from_volume()
        elif self.model_loading_mode == 'build_time':
            return self._load_from_build_time()
        elif self.model_loading_mode == 'runtime':
            return self._load_from_runtime()
        else:
            logger.warning(f"Unknown loading mode: {self.model_loading_mode}, falling back to volume mode")
            return self._load_from_volume()
    
    def _load_from_volume(self) -> torch.nn.Module:
        """
        Solution 2: Load from Docker volume
        - Check if model exists in volume
        - If not, download and save to volume
        - Load from saved file
        """
        logger.info("Loading model from Docker volume (Solution 2)")
        
        # Ensure model directory exists
        model_dir = Path(self.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = Path(self.model_save_path)
        
        if model_file.exists():
            logger.info(f"Found existing model at: {model_file}")
            try:
                # Load the model properly
                model = self._load_model_from_file(str(model_file))
                logger.info("✅ Model loaded successfully from volume")
                return model
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
        
        # Model doesn't exist or failed to load, download and save
        logger.info("Downloading and saving model to volume...")
        model = self._download_and_save_model()
        return model
    
    def _load_from_build_time(self) -> torch.nn.Module:
        """
        Solution 3: Load from pre-copied model file
        - Model should be copied into image during build
        - Load directly from the copied file
        """
        logger.info("Loading model from build-time copy (Solution 3)")
        
        model_file = Path(self.model_save_path)
        
        if model_file.exists():
            logger.info(f"Found build-time model at: {model_file}")
            try:
                model = self._load_model_from_file(str(model_file))
                logger.info("✅ Model loaded successfully from build-time copy")
                return model
            except Exception as e:
                logger.error(f"Failed to load build-time model: {e}")
                raise
        else:
            logger.error(f"Build-time model not found at: {model_file}")
            logger.info("Falling back to runtime download...")
            return self._load_from_runtime()
    
    def _load_from_runtime(self) -> torch.nn.Module:
        """
        Runtime download mode (fallback)
        - Download model at runtime
        - Don't save to persistent storage
        """
        logger.info("Loading model from runtime download (fallback)")
        
        # Load from PyTorch Hub (cached if available)
        try:
            model = models.resnet18(pretrained=True)
            logger.info("✅ Model loaded successfully from PyTorch Hub")
            return model
        except Exception as e:
            logger.error(f"Failed to load from PyTorch Hub: {e}")
            raise
    
    def _download_and_save_model(self) -> torch.nn.Module:
        """
        Download model and save to volume
        """
        logger.info("Downloading model from PyTorch Hub...")
        
        try:
            # Load model from PyTorch Hub
            model = models.resnet18(pretrained=True)
            
            # Save to volume
            logger.info(f"Saving model to: {self.model_save_path}")
            model_dir = Path(self.model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(model, self.model_save_path)
            logger.info("✅ Model downloaded and saved successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to download and save model: {e}")
            raise
    
    def _load_model_from_file(self, model_path: str) -> torch.nn.Module:
        """
        Load model from file, handling both state dict and full model formats
        """
        try:
            # Try to load as full model first
            loaded_data = torch.load(model_path, map_location='cpu')
            
            # Check if it's a state dict (OrderedDict) or full model
            if isinstance(loaded_data, dict):
                logger.info("Loading state dict and creating model...")
                # It's a state dict, create model and load state
                model = models.resnet18(pretrained=False)
                model.load_state_dict(loaded_data)
            else:
                logger.info("Loading full model...")
                model = loaded_data
            
            model.eval()  # Set to evaluation mode
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model configuration
        """
        return {
            'loading_mode': self.model_loading_mode,
            'model_name': self.model_name,
            'model_path': self.model_path,
            'model_save_path': self.model_save_path,
            'model_exists': Path(self.model_save_path).exists(),
            'model_size_mb': self._get_model_size_mb() if Path(self.model_save_path).exists() else None
        }
    
    def _get_model_size_mb(self) -> Optional[float]:
        """
        Get model file size in MB
        """
        try:
            size_bytes = Path(self.model_save_path).stat().st_size
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return None

# Global model loader instance
model_loader = DynamicModelLoader()

def get_model() -> torch.nn.Module:
    """
    Get the loaded model instance
    """
    return model_loader.load_model()

def get_model_info() -> dict:
    """
    Get model information
    """
    return model_loader.get_model_info()

class ModelLoader:
    """
    Model loader class for managing ML model lifecycle
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    async def load_model(self):
        """Load the ML model"""
        try:
            # Ensure the models directory exists
            models_dir = os.path.dirname(self.model_path)
            os.makedirs(models_dir, exist_ok=True)
            
            # First, try to load from our saved model file
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from saved file: {self.model_path}")
                self.model = ResNet18SolarPanel(num_classes=2)
                
                if torch.cuda.is_available():
                    state_dict = torch.load(self.model_path, map_location='cuda')
                else:
                    state_dict = torch.load(self.model_path, map_location='cpu')
                
                try:
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    logger.info(f"Model loaded successfully from {self.model_path}")
                    return
                except Exception as load_error:
                    logger.warning(f"Could not load saved model due to architecture mismatch: {load_error}")
                    logger.info("Creating new model with current architecture...")
                    # Remove the incompatible saved model
                    try:
                        os.remove(self.model_path)
                        logger.info("Removed incompatible saved model")
                    except Exception as remove_error:
                        logger.warning(f"Could not remove incompatible model: {remove_error}")
            
            # If no saved model, try to load from PyTorch Hub cache
            logger.info("No saved model found, loading from PyTorch Hub...")
            
            # Ensure cache directory exists and has proper permissions
            cache_dir = os.path.expanduser("~/.cache/torch/hub")
            try:
                os.makedirs(cache_dir, exist_ok=True)
                logger.info(f"Cache directory ready: {cache_dir}")
            except Exception as cache_error:
                logger.warning(f"Could not create cache directory: {cache_error}")
            
            self.model = ResNet18SolarPanel(num_classes=2)
            self.model.eval()
            
            # Save the model for future use
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.model_path)
                logger.info(f"Model saved to {self.model_path} for future use")
            except Exception as save_error:
                logger.warning(f"Could not save model: {save_error}")
            
            logger.info("Model loaded successfully from PyTorch Hub")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            # Create a new model as fallback
            logger.info("Creating new model as fallback")
            self.model = ResNet18SolarPanel(num_classes=2)
    
    async def predict_image(self, image_content: bytes):
        """
        Predict from image content bytes
        
        Args:
            image_content: Image content as bytes
            
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_content)).convert('RGB')
            
            # Preprocess image
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get prediction and confidence
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Map class index to label
                class_labels = ['clean', 'dirty']
                prediction = class_labels[predicted_class]
                
                return prediction, confidence
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            # Return fallback prediction
            return 'clean', 0.5

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
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file {model_path} not found. Creating a new model.")
            model = ResNet18SolarPanel(num_classes=num_classes)
            return model
        
        # Load the model
        model = ResNet18SolarPanel(num_classes=num_classes)
        
        # Load state dict
        if torch.cuda.is_available():
            state_dict = torch.load(model_path, map_location='cuda')
        else:
            state_dict = torch.load(model_path, map_location='cpu')
            
        model.load_state_dict(state_dict)
        model.eval()
            
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # Return a new model as fallback
        logger.info("Creating new model as fallback")
        return ResNet18SolarPanel(num_classes=num_classes)

def preprocess_image(image_path: str, target_size: tuple = (224, 224)):
    """
    Preprocess an image for model prediction
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (width, height)
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Failed to preprocess image: {str(e)}")
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
        model.eval()
        
        with torch.no_grad():
            # Make prediction
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction and confidence
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Map class index to label
            class_labels = ['clean', 'dirty']
            prediction = class_labels[predicted_class]
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'clean': probabilities[0][0].item(),
                    'dirty': probabilities[0][1].item()
                }
            }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Return a fallback prediction
        return {
            'prediction': 'clean',
            'confidence': 0.5,
            'probabilities': {
                'clean': 0.5,
                'dirty': 0.5
            },
            'error': str(e)
        }

def predict_from_file(model, image_path: str):
    """
    Make a prediction from an image file
    
    Args:
        model: Loaded PyTorch model
        image_path: Path to the image file
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Preprocess the image
        image_tensor = preprocess_image(image_path)
        
        # Make prediction
        result = predict_image(model, image_tensor)
        
        # Add file information
        result['filename'] = os.path.basename(image_path)
        result['file_path'] = image_path
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to predict from file {image_path}: {str(e)}")
        return {
            'prediction': 'clean',
            'confidence': 0.0,
            'probabilities': {
                'clean': 0.5,
                'dirty': 0.5
            },
            'filename': os.path.basename(image_path),
            'file_path': image_path,
            'error': str(e)
        }
