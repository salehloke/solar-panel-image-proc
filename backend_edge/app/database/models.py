from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .db import Base

class DetectionRecord(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String)
    confidence = Column(Float)
    inference_time = Column(Float)
    efficiency_loss = Column(Float) # Estimated loss: clean=0%, dust=10%, bird=25%, moss=40%
    timestamp = Column(DateTime, default=datetime.utcnow)
