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
    version="1.3.0"
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

# --- Model Management ---
class ModelState:
    def __init__(self):
        self.model = None
        self.current_model_name = "solar_svm_model" 
        self.required_features = ['hog', 'glcm']    
        self.models_dir = "app/models"
        self.class_map = {0: "Clean", 1: "Dust", 2: "Bird Droppings", 3: "Moss"}
        self.benchmarks = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "proc_time_ms": 0.0
        }
    
    def load_model(self, model_name: str):
        """Loads a model and determines feature requirements and benchmarks."""
        path = os.path.join(self.models_dir, f"{model_name}.pkl")
        metrics_path = os.path.join(self.models_dir, f"{model_name}_metrics.json")
        
        if not os.path.exists(path):
            if not path.endswith('.pkl'): path += '.pkl'
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

        print(f"🔄 Loading model: {model_name}...")
        try:
            self.model = joblib.load(path)
            self.current_model_name = model_name
            
            # Load benchmarks if available
            if os.path.exists(metrics_path):
                import json
                with open(metrics_path, 'r') as f:
                    self.benchmarks = json.load(f)
            else:
                self.benchmarks = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "proc_time_ms": 0}

            # Determine features based on naming convention
            if "_glcm" in model_name:
                self.required_features = ['glcm']
            elif "_hog" in model_name:
                self.required_features = ['hog']
            else:
                self.required_features = ['hog', 'glcm']
                
            print(f"✅ Model loaded: {model_name} (Accuracy: {self.benchmarks['accuracy']:.2f})")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def predict(self, image):
        """Runs prediction using the loaded model."""
        if self.model is None:
            return "Unknown", 0.0

        # Extract only required features
        features = extractor.extract_features(image, feature_types=self.required_features)
        features = features.reshape(1, -1)
        
        prediction = self.model.predict(features)[0]
        try:
            probs = self.model.predict_proba(features)[0]
            confidence = float(np.max(probs))
        except:
            confidence = 1.0 
            
        return self.class_map.get(prediction, "Unknown"), confidence

model_state = ModelState()

# Estimated efficiency loss mapping
EFFICIENCY_LOSS_MAP = {
    "Clean": 0.0,
    "Dust": 10.5,
    "Bird Droppings": 25.0,
    "Moss": 42.0
}

@app.on_event("startup")
async def startup_event():
    """Initialize DB and load default model on startup."""
    await init_db()
    default_model = os.getenv("MODEL_NAME", "solar_rf_glcm") 
    try:
        model_state.load_model(default_model)
    except:
        print("⚠️ Could not load default model. Waiting for configuration.")

# --- Config Schemas ---
class ModelConfigResponse(BaseModel):
    current_model: str
    available_models: List[str]
    features_used: List[str]
    benchmarks: Dict[str, float]

class ModelUpdateRequest(BaseModel):
    model_name: str

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_state.model is not None}

@app.get("/config/model", response_model=ModelConfigResponse)
async def get_model_config():
    """Returns current model configuration and available options."""
    models = []
    if os.path.exists(model_state.models_dir):
        models = [f.replace(".pkl", "") for f in os.listdir(model_state.models_dir) if f.endswith(".pkl")]
    
    return ModelConfigResponse(
        current_model=model_state.current_model_name,
        available_models=sorted(models),
        features_used=model_state.required_features,
        benchmarks=model_state.benchmarks
    )

@app.post("/config/model")
async def set_model(request: ModelUpdateRequest):
    """Switches the active model at runtime."""
    try:
        success = model_state.load_model(request.model_name)
        if not success:
             raise HTTPException(status_code=500, detail="Failed to load model")
        return {"status": "success", "message": f"Switched to {request.model_name}"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")

# --- Existing Endpoints (Refactored) ---

async def gen_frames():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (320, 240))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.get("/stream")
async def video_stream():
    """Video streaming route."""
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

class DetectionResponse(BaseModel):
    class_name: str
    confidence: float
    inference_time: float
    efficiency_loss: float
    timestamp: datetime
    image_url: Optional[str] = None
    # Benchmarks for the model used
    model_accuracy: float
    model_precision: float
    model_recall: float
    model_proc_time: float

@app.post("/predict", response_model=DetectionResponse)
async def detect_dirt(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Endpoint to receive an image, predict using current model, and log."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        start_time = time.time()
        contents = await file.read()
        
        filename = f"upload_{int(time.time())}_{file.filename}"
        save_path = os.path.join("data/captured", filename)
        with open(save_path, "wb") as f:
            f.write(contents)

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")

        class_name, confidence = model_state.predict(img)

        inference_time = round(time.time() - start_time, 4)
        efficiency_loss = EFFICIENCY_LOSS_MAP.get(class_name, 0.0)
        
        # Log to Database
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
            image_url=f"/images/{filename}",
            model_accuracy=model_state.benchmarks["accuracy"],
            model_precision=model_state.benchmarks["precision"],
            model_recall=model_state.benchmarks["recall"],
            model_proc_time=model_state.benchmarks["proc_time_ms"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AnalyticsSummary(BaseModel):
    total_detections: int
    most_common_type: str
    avg_efficiency_loss: float
    class_distribution: List[Dict] 
    efficiency_trend: List[Dict]   
    recent_history: List[Dict]

@app.get("/analytics", response_model=AnalyticsSummary)
async def get_analytics(db: AsyncSession = Depends(get_db)):
    """Returns summarized metrics for the dashboard."""
    total_q = await db.execute(select(func.count(DetectionRecord.id)))
    total = total_q.scalar()
    
    if total == 0:
        return AnalyticsSummary(
            total_detections=0, most_common_type="None", 
            avg_efficiency_loss=0.0, class_distribution=[], 
            efficiency_trend=[], recent_history=[]
        )

    dist_q = await db.execute(
        select(DetectionRecord.class_name, func.count(DetectionRecord.id))
        .group_by(DetectionRecord.class_name)
    )
    dist = [{"name": r[0], "value": r[1]} for r in dist_q.all()]

    mode = "None"
    if dist:
        mode = max(dist, key=lambda x: x["value"])["name"]

    avg_loss_q = await db.execute(select(func.avg(DetectionRecord.efficiency_loss)))
    avg_loss = round(avg_loss_q.scalar() or 0.0, 2)

    trend_q = await db.execute(
        select(DetectionRecord.timestamp, DetectionRecord.efficiency_loss)
        .order_by(DetectionRecord.timestamp.asc())
        .limit(20)
    )
    trend = [{"time": r[0].strftime("%H:%M"), "loss": r[1]} for r in trend_q.all()]

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
    """Triggers camera and runs detection."""
    try:
        start_time = time.time()
        frame, file_path = camera.capture_image()
        class_name, confidence = model_state.predict(frame)

        inference_time = round(time.time() - start_time, 4)
        efficiency_loss = EFFICIENCY_LOSS_MAP.get(class_name, 0.0)
        
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
            image_url=f"/images/{os.path.basename(file_path)}",
            model_accuracy=model_state.benchmarks["accuracy"],
            model_precision=model_state.benchmarks["precision"],
            model_recall=model_state.benchmarks["recall"],
            model_proc_time=model_state.benchmarks["proc_time_ms"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera trigger failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)