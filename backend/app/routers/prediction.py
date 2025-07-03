from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
from ..utils.model_loader import predict_image

router = APIRouter(
    prefix="/prediction",
    tags=["prediction"],
    responses={404: {"description": "Not found"}},
)

# Image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model instance - will be set during app startup
model = None

@router.post("/")
async def predict_solar_panel(file: UploadFile = File(...)):
    """
    Predict whether a solar panel is clean or dirty from an uploaded image
    """
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Apply transformations
        image_tensor = transform(image)
        
        # Check if model is loaded
        if model is None:
            return JSONResponse(
                status_code=503,
                content={"message": "Model not loaded. Server is not ready."}
            )
        
        # Get prediction
        result = predict_image(model, image_tensor)
        
        return {
            "filename": file.filename,
            "prediction": result["class_name"],
            "confidence": result["confidence"],
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
