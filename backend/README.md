# Solar Panel Dirt Detection API

A FastAPI application for detecting dirt on solar panels using a ResNet18-based deep learning model.

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application setup
│   ├── routers/             # API route handlers
│   │   ├── prediction.py    # Endpoints for model predictions
│   ├── models/              # Pydantic models for request/response validation
│   │   ├── prediction.py    # Schemas for prediction responses
│   └── utils/               # Utility functions
│       ├── model_loader.py  # Functions for loading and using the ML model
├── run.py                   # Entry point to run the application
└── README.md               # This documentation file
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- FastAPI
- Trained ResNet18 model for solar panel dirt detection

### Installation

1. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Place your trained model in the models directory:
   ```
   /models/resnet18_solar_panel.pt
   ```

### Running the API Server

Run the API server using:

```bash
python run.py
```

This starts the server at `http://localhost:8000` by default.

## API Endpoints

### GET /

Root endpoint with welcome message.

### GET /health

Health check endpoint that returns the server status.

### POST /prediction/

Upload an image to get a prediction on whether the solar panel is clean or dirty.

**Example cURL request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/prediction/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@solar_panel_image.jpg'
```

**Response:**
```json
{
  "filename": "solar_panel_image.jpg",
  "prediction": "dirty",
  "confidence": 0.94,
  "status": "success"
}
```

## Environment Variables

- `PORT`: Server port (default: 8000)
- `MODEL_PATH`: Path to the trained model file
- `MODEL_DIR`: Directory containing the model (default: "../models")

## Documentation

When the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API documentation: `http://localhost:8000/redoc`
