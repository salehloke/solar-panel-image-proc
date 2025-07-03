from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    """
    Schema for prediction response
    """
    filename: str
    prediction: str
    confidence: float
    status: str

class HealthResponse(BaseModel):
    """
    Schema for health check response
    """
    status: str
