from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging
from pathlib import Path

from .routers import prediction
from .utils.model_loader import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Solar Panel Dirt Detection API",
    description="API for detecting dirt on solar panels using deep learning",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Solar Panel Dirt Detection API"}

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

# Include routers
app.include_router(prediction.router)

@app.on_event("startup")
async def startup_event():
    """
    Load model during startup
    """
    try:
        # Define model path - this should be updated with the actual trained model path
        model_dir = os.environ.get("MODEL_DIR", "./models")
        model_path = os.environ.get("MODEL_PATH", f"{model_dir}/resnet18_solar_panel.pt")
        
        # Check if model exists
        if Path(model_path).exists():
            # Load model
            logger.info(f"Loading model from {model_path}...")
            prediction.model = load_model(model_path)
            logger.info("Model loaded successfully!")
        else:
            logger.warning(f"Model file {model_path} not found. API will respond with dummy predictions.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
