import os
import io
import time
import joblib
import cv2
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta

from app.services.features import FeatureExtractor
from app.services.camera import CameraService
from app.database.db import get_db, init_db
from app.database.models import DetectionRecord

# Initialize FastAPI
app = FastAPI(
    title="Solar Panel Edge Detection API",
    description="SVM-based dirt classification for Raspberry Pi",
    version="1.2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for captured images
os.makedirs("data/captured", exist_ok=True)
app.mount("/images", StaticFiles(directory="data/captured"), name="images")

# Initialize Services
extractor = FeatureExtractor(target_size=(128, 128))
camera = CameraService()

# Global variable for the model
model = None
MODEL_PATH = os.getenv("MODEL_PATH", "app/models/solar_svm_model.pkl")

# Estimated efficiency loss mapping
EFFICIENCY_LOSS_MAP = {
    "Clean": 0.0,
    "Dust": 10.5,
    "Bird Droppings": 25.0,
    "Moss": 42.0
}

@app.on_event("startup")
async def startup_event():
    """Initialize DB and load models on startup."""
    await init_db()
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Pre-trained SVM model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

class DetectionResponse(BaseModel):
    class_name: str
    confidence: float
    inference_time: float
    efficiency_loss: float
    timestamp: datetime
    image_url: Optional[str] = None

class AnalyticsSummary(BaseModel):
    total_detections: int
    most_common_type: str
    avg_efficiency_loss: float
    class_distribution: List[Dict] # For Bar Chart
    efficiency_trend: List[Dict]   # For Line Chart
    recent_history: List[Dict]

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

async def gen_frames():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Lower resolution for smoother streaming on Pi
            frame = cv2.resize(frame, (320, 240))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.get("/stream")
async def video_stream():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/predict", response_model=DetectionResponse)
async def detect_dirt(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """
    Endpoint to receive an image, extract HOG/GLCM features, 
    predict the dirt type, and log to SQLite.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        start_time = time.time()
        contents = await file.read()
        
        # Save file to captured directory for serving
        filename = f"upload_{int(time.time())}_{file.filename}"
        save_path = os.path.join("data/captured", filename)
        with open(save_path, "wb") as f:
            f.write(contents)

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")

        # 1. Feature Extraction
        features = extractor.extract_features(img).reshape(1, -1)
        
        # 2. Model Inference
        if model is not None:
            prediction = model.predict(features)[0]
            try:
                probs = model.predict_proba(features)[0]
                confidence = float(np.max(probs))
            except:
                confidence = 1.0
                
            class_map = {0: "Clean", 1: "Dust", 2: "Bird Droppings", 3: "Moss"}
            class_name = class_map.get(prediction, "Unknown")
        else:
            class_name = "Clean" # Mock if model missing
            confidence = 0.5

        inference_time = round(time.time() - start_time, 4)
        efficiency_loss = EFFICIENCY_LOSS_MAP.get(class_name, 0.0)
        
        # 3. Log to Database
        record = DetectionRecord(
            class_name=class_name,
            confidence=confidence,
            inference_time=inference_time,
            efficiency_loss=efficiency_loss
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        
        # Add URL to response
        response_data = DetectionResponse(
            class_name=class_name,
            confidence=confidence,
            inference_time=inference_time,
            efficiency_loss=efficiency_loss,
            timestamp=record.timestamp,
            image_url=f"/images/{filename}"
        )
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics", response_model=AnalyticsSummary)
async def get_analytics(db: AsyncSession = Depends(get_db)):
    """
    Returns summarized metrics for the dashboard.
    """
    # 1. Get total count
    total_q = await db.execute(select(func.count(DetectionRecord.id)))
    total = total_q.scalar()
    
    if total == 0:
        return AnalyticsSummary(
            total_detections=0, most_common_type="None", 
            avg_efficiency_loss=0.0, recent_history=[]
        )

    # 2. Get class distribution
    dist_q = await db.execute(
        select(DetectionRecord.class_name, func.count(DetectionRecord.id))
        .group_by(DetectionRecord.class_name)
    )
    dist = [{"name": r[0], "value": r[1]} for r in dist_q.all()]

    # 3. Get most common type
    mode = "None"
    if dist:
        mode = max(dist, key=lambda x: x["value"])["name"]

    # 4. Get average loss
    avg_loss_q = await db.execute(select(func.avg(DetectionRecord.efficiency_loss)))
    avg_loss = round(avg_loss_q.scalar() or 0.0, 2)

    # 5. Get efficiency trend (Last 10 records)
    trend_q = await db.execute(
        select(DetectionRecord.timestamp, DetectionRecord.efficiency_loss)
        .order_by(DetectionRecord.timestamp.asc())
        .limit(20)
    )
    trend = [{"time": r[0].strftime("%H:%M"), "loss": r[1]} for r in trend_q.all()]

    # 6. Get recent history (Last 20 records)
    history_q = await db.execute(
        select(DetectionRecord).order_by(DetectionRecord.timestamp.desc()).limit(20)
    )
    history = history_q.scalars().all()
    
    formatted_history = [
        {"timestamp": r.timestamp, "class": r.class_name, "loss": r.efficiency_loss} 
        for r in history
    ]

    return AnalyticsSummary(
        total_detections=total,
        most_common_type=mode,
        avg_efficiency_loss=avg_loss,
        class_distribution=dist,
        efficiency_trend=trend,
        recent_history=formatted_history
    )

@app.post("/capture", response_model=DetectionResponse)
async def capture_and_detect(db: AsyncSession = Depends(get_db)):
    """
    Triggers the Raspberry Pi camera, captures an image, 
    and automatically runs detection.
    """
    try:
        start_time = time.time()
        
        # 1. Capture image from hardware
        frame, file_path = camera.capture_image()
        
        # 2. Extract Features
        features = extractor.extract_features(frame).reshape(1, -1)
        
        # 3. Model Inference
        if model is not None:
            prediction = model.predict(features)[0]
            try:
                probs = model.predict_proba(features)[0]
                confidence = float(np.max(probs))
            except:
                confidence = 1.0
                
            class_map = {0: "Clean", 1: "Dust", 2: "Bird Droppings", 3: "Moss"}
            class_name = class_map.get(prediction, "Unknown")
        else:
            class_name = "Clean"
            confidence = 0.0

        inference_time = round(time.time() - start_time, 4)
        efficiency_loss = EFFICIENCY_LOSS_MAP.get(class_name, 0.0)
        
        # 4. Log to Database
        record = DetectionRecord(
            class_name=class_name,
            confidence=confidence,
            inference_time=inference_time,
            efficiency_loss=efficiency_loss
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        
        return DetectionResponse(
            class_name=class_name,
            confidence=confidence,
            inference_time=inference_time,
            efficiency_loss=efficiency_loss,
            timestamp=record.timestamp,
            image_url=f"/images/{os.path.basename(file_path)}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera trigger failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)