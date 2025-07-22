# Solar Panel Dirt Detection System - Project Summary

## Project Overview

This system is a comprehensive deep learning solution designed to detect dirt accumulation on solar panels using computer vision and PyTorch. It helps solar panel operators monitor panel cleanliness, schedule maintenance efficiently, and optimize energy production by identifying when panels need cleaning.

## 🚀 Getting Started

### Available Run Modes

#### 1. Docker Mode (Recommended for Production/Testing)
- **Description**: Complete containerized environment
- **Components**:
  - API Service
  - PostgreSQL Database
  - Redis Cache
  - pgAdmin Interface
- **Start Command**:
  ```bash
  ./scripts/start-backend-docker.sh
  ```
- **Access**:
  - API: http://localhost:8000
  - API Docs: http://localhost:8000/docs
  - pgAdmin: http://localhost:8081

#### 2. Hybrid Mode (Recommended for Development)
- **Description**: Local API with containerized databases
- **Components**:
  - Local API Server
  - Docker Containers for PostgreSQL and Redis
- **Start Commands**:
  ```bash
  # Start database services
  ./scripts/start-backend-hybrid.sh
  
  # In a new terminal
  cd backend
  uvicorn app.main:app --reload
  ```

#### 3. Local Development Mode
- **Description**: Everything runs locally (for advanced development)
- **Prerequisites**:
  - Python 3.9+
  - PostgreSQL and Redis installed locally
- **Setup**:
  ```bash
  # Install dependencies
  pip install -r requirements-dev.txt
  
  # Set up environment
  cp .env.example .env
  # Edit .env with your local configuration
  
  # Start the API
  cd backend
  uvicorn app.main:app --reload
  ```

### Next Steps
- Access the API documentation at http://localhost:8000/docs
- Try making a test prediction using the API
- Check the logs for any startup issues

## Key Features

- **Deep Learning Classification**: Uses a ResNet18-based model to classify solar panel images as either "clean" or "dirty"
- **Production-Ready API**: FastAPI backend provides real-time prediction endpoints with proper error handling
- **Database Integration**: Stores analysis results, user data, and solar panel information in a PostgreSQL database
- **Flexible Model Loading**: Supports multiple deployment strategies for model loading (volume, build-time, runtime)
- **User Management**: Tracks predictions by user ID and provides user-specific analytics
- **Performance Statistics**: API endpoints for retrieving analysis statistics and trends
- **Containerized Deployment**: Docker and Docker Compose support for easy deployment
- **Health Monitoring**: Health check endpoint to verify system status

## Technical Architecture

### Backend (FastAPI)

- **API Endpoints**: 
  - `/predict`: Upload images for dirt detection
  - `/users/{user_id}/analysis`: Get analysis results for a specific user
  - `/users/{user_id}/panels`: Get solar panels for a user
  - `/stats`: Get analysis statistics
  - `/model/info`: Get model information
  - `/health`: System health check

- **Database Models**:
  - Analysis results
  - User information
  - Solar panel metadata

- **ML Integration**:
  - Dynamic model loading from various sources
  - Image preprocessing pipeline
  - PyTorch inference with confidence scores

### ML Model

- **Architecture**: Modified ResNet18 with:
  - Pretrained weights
  - Custom fully connected layers
  - Dropout for regularization
  - Binary classification output (clean/dirty)

- **Preprocessing Pipeline**:
  - Image resizing to 224x224
  - Normalization
  - Data augmentation during training

### Project Structure

```
solar-image-processing/
├── backend/               # FastAPI application
│   ├── app/               # Main application code
│   │   ├── main.py        # API endpoints
│   │   ├── models/        # Database models
│   │   └── utils/         # Utilities (model loading)
│   ├── models/            # Saved ML models
│   └── run.py             # Server entry point
├── src/                   # ML model training code
│   ├── data/              # Dataset handling
│   ├── train.py           # Training script
│   └── evaluate.py        # Model evaluation
├── logs/                  # Training logs and saved models
├── notebooks/             # Development notebooks
├── services/              # Microservices
├── tests/                 # Unit and integration tests
└── docker-compose files   # Deployment configuration
```

## Technologies Used

- **Backend**: FastAPI, SQLAlchemy (async)
- **ML Framework**: PyTorch, torchvision
- **Database**: PostgreSQL with asyncpg
- **Image Processing**: PIL, OpenCV
- **Containerization**: Docker, Docker Compose
- **Development Tools**: Python 3.9+

## Workflow

1. Users upload solar panel images through the API
2. The system preprocesses images to match model input requirements
3. The ResNet18 model classifies the image as "clean" or "dirty"
4. Results are stored in the database with metadata
5. Users can retrieve their analysis history and statistics

## Deployment Options

The system supports multiple deployment scenarios:

1. **Volume-based**: Model is loaded from a persistent Docker volume
2. **Build-time**: Model is included in the Docker image during build
3. **Runtime**: Model is downloaded at runtime (fallback option)

## Future Improvements

- Multi-class classification (different types/levels of dirt)
- Integration with automated cleaning systems
- Time series analysis for degradation prediction
- Mobile application for field inspections
