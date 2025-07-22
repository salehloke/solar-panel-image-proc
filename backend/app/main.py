import os
import time
from typing import Optional
from uuid import UUID
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

from .models.database import (
    Base, DatabaseManager, 
    AnalysisResultCreate, AnalysisResultResponse,
    UserResponse, SolarPanelResponse
)
from .utils.model_loader import get_model, get_model_info, model_loader

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://solarai:solarai123@postgres:5432/solarai")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Database setup
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Global model instance
ml_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting SolarAI Backend...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Load ML model
    global ml_model
    try:
        ml_model = get_model()
        print("✅ ML model loaded successfully")
        print(f"Model info: {get_model_info()}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load ML model: {e}")
    
    print("✅ SolarAI Backend started successfully!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down SolarAI Backend...")
    await engine.dispose()

# FastAPI app
app = FastAPI(
    title="SolarAI Backend",
    description="Solar Panel Dirt Detection API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Dependency to get database manager
async def get_db_manager(session: AsyncSession = Depends(get_db)) -> DatabaseManager:
    return DatabaseManager(session)

@app.get("/")
async def root():
    return {
        "message": "SolarAI Backend API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        async with AsyncSessionLocal() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        
        # Test model availability
        model_status = "loaded" if ml_model is not None else "not_loaded"
        
        return {
            "status": "healthy",
            "database": "connected",
            "model": model_status,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/predict", response_model=AnalysisResultResponse)
async def predict_solar_panel_dirt(
    file: UploadFile = File(...),
    user_id: Optional[str] = None,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Predict if a solar panel is clean or dirty from an uploaded image
    """
    print("[DEBUG] Received /predict request")
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    file_size = 0
    content = b""
    
    while chunk := await file.read(8192):
        content += chunk
        file_size += len(chunk)
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds 10MB limit"
            )
    
    try:
        # Process image with ML model
        if ml_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not loaded"
            )
        print("[DEBUG] Starting model inference")
        start_time = time.time()
        prediction, confidence = await run_in_threadpool(predict_image_with_model, ml_model, content)
        print("[DEBUG] Model inference complete")
        processing_time = time.time() - start_time
        
        # Save result to database
        result_data = {
            "user_id": UUID(user_id) if user_id else None,
            "image_path": f"uploads/{file.filename}",
            "prediction": prediction,
            "confidence": confidence,
            "model_version": "resnet18_v1",
            "processing_time": processing_time
        }
        
        analysis_result = await db_manager.create_analysis_result(result_data)
        
        return analysis_result
        
    except Exception as e:
        print(f"[ERROR] Exception in /predict: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/users/{user_id}/analysis", response_model=list[AnalysisResultResponse])
async def get_user_analysis_results(
    user_id: str,
    limit: int = 10,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Get analysis results for a specific user"""
    try:
        results = await db_manager.get_user_analysis_results(user_id, limit)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch analysis results: {str(e)}"
        )

@app.get("/users/{user_id}/panels", response_model=list[SolarPanelResponse])
async def get_user_solar_panels(
    user_id: str,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Get solar panels for a specific user"""
    try:
        panels = await db_manager.get_user_solar_panels(user_id)
        return panels
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch solar panels: {str(e)}"
        )

@app.get("/stats")
async def get_analysis_stats(
    user_id: Optional[str] = None,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Get analysis statistics"""
    try:
        stats = await db_manager.get_analysis_stats(user_id)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch statistics: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info_endpoint():
    """Get information about the loaded model"""
    try:
        model_info = get_model_info()
        return {
            "model_info": model_info,
            "model_loaded": ml_model is not None,
            "loading_mode": model_info.get('loading_mode', 'unknown')
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

async def predict_image_with_model(model, image_content: bytes) -> tuple[str, float]:
    """
    Predict if a solar panel is clean or dirty using the provided model
    """
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import io
    import numpy as np
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_content)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map class index to label (assuming binary classification: 0=clean, 1=dirty)
        prediction = "dirty" if predicted_class == 1 else "clean"
        
        return prediction, confidence
        
    except Exception as e:
        raise Exception(f"Image processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )
