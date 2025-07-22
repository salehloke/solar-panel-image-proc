# Solar Panel Dirt Detection System - Software Engineering Perspective

## System Architecture Overview

This project implements a robust, production-grade system for detecting dirt accumulation on solar panels using deep learning. The architecture follows modern software engineering best practices, with clear separation of concerns and scalability in mind.

### Core Components

1. **Machine Learning Pipeline**
   - **Model Architecture**: ResNet18-based classifier fine-tuned for binary classification (clean/dirty)
   - **Training Infrastructure**:
     - Data augmentation (random crops, flips, color jitter)
     - Class imbalance handling with weighted sampling
     - Comprehensive logging and model checkpointing
     - Early stopping and learning rate scheduling
   - **Evaluation Metrics**:
     - Accuracy, Precision, Recall, F1-score
     - Confusion matrix visualization
     - ROC curves for model assessment

2. **Backend Service**
   - **Framework**: FastAPI for high-performance API endpoints
   - **Key Endpoints**:
     - `POST /prediction/`: Image upload and classification
     - `GET /health`: System health monitoring
     - `GET /model/info`: Model metadata and versioning
   - **Features**:
     - Async request handling
     - Input validation with Pydantic models
     - Structured logging
     - Error handling middleware

3. **Data Management**
   - **Data Pipeline**:
     - Image preprocessing pipeline
     - Train/validation/test split utilities
     - Data versioning
   - **Augmentation Strategies**:
     - Geometric transformations
     - Photometric distortions
     - Random erasing for robustness

## Code Organization

```
solar-image-processing/
├── backend/                  # Production API service
│   ├── app/                  # Application core
│   │   ├── __init__.py       # Package initialization
│   │   ├── main.py           # FastAPI app factory
│   │   ├── models/           # Database and ML models
│   │   ├── routes/           # API endpoints
│   │   └── utils/            # Helper functions
│   ├── tests/                # API tests
│   └── run.py                # Service entry point
│
├── pytorch/                  # Model implementation
│   ├── models/               # Model architectures
│   │   ├── __init__.py
│   │   └── resnet_model.py   # Custom ResNet implementation
│   └── integration.py        # Model serving utilities
│
├── src/                      # Training pipeline
│   ├── data/                 # Dataset handling
│   │   └── dataset.py        # PyTorch dataset implementation
│   ├── train.py              # Training script
│   ├── evaluate.py           # Model evaluation
│   └── config.py             # Training configuration
│
├── scripts/                  # Utility scripts
│   ├── train_pipeline.py     # End-to-end training
│   ├── split_dataset.py      # Data preparation
│   └── deploy_model.py       # Model deployment
│
├── configs/                  # Configuration files
├── data/                     # Dataset storage
├── models/                   # Trained models
├── logs/                     # Training logs
├── tests/                    # Test suite
├── docker/                   # Docker configurations
└── requirements/             # Dependency specifications
```

## Key Software Engineering Practices

### 1. Code Quality
- Type hints throughout the codebase
- Comprehensive docstrings and API documentation
- Consistent code formatting with Black
- Linting with flake8 and mypy
- Unit test coverage > 80%

### 2. Model Development
- Experiment tracking with MLflow
- Model versioning
- Reproducible training pipelines
- Hyperparameter optimization support

### 3. API Design
- RESTful principles
- OpenAPI/Swagger documentation
- Input validation
- Rate limiting
- Authentication/Authorization

### 4. Testing Strategy
- Unit tests for core functionality
- Integration tests for API endpoints
- Model validation tests
- Load testing for performance

## Deployment Architecture

### Containerization
- Multi-stage Docker builds
- Lightweight production images
- Environment-based configuration
- Health checks

### Orchestration
- Docker Compose for local development
- Kubernetes manifests for production
- Horizontal pod autoscaling
- Resource limits and requests

### Monitoring
- Prometheus metrics
- Grafana dashboards
- Structured logging with ELK stack
- Error tracking with Sentry

## Development Workflow

1. **Local Development**
   - Docker-based development environment
   - Hot-reload for fast iteration
   - Pre-commit hooks

2. **Version Control**
   - Git flow branching strategy
   - Conventional commits
   - Pull request templates
   - Code review requirements

3. **CI/CD Pipeline**
   - Automated testing
   - Container scanning
   - Security scanning
   - Staging deployment

## Performance Considerations

1. **Model Inference**
   - ONNX runtime for production
   - Batch processing support
   - GPU acceleration
   - Model quantization

2. **API Performance**
   - Async request handling
   - Response caching
   - Connection pooling
   - Gunicorn workers

## Security Measures

1. **API Security**
   - JWT authentication
   - Rate limiting
   - CORS configuration
   - Request validation

2. **Model Security**
   - Input sanitization
   - Adversarial attack prevention
   - Model encryption
   - Secure model serving

## Future Roadmap

1. **Model Improvements**
   - Multi-task learning
   - Few-shot learning for rare cases
   - Uncertainty estimation
   - Explainable AI features

2. **Infrastructure**
   - Serverless deployment
   - Edge deployment
   - Model monitoring
   - A/B testing framework

3. **Features**
   - Time-series analysis
   - Automated reporting
   - Mobile application
   - Integration with cleaning systems

## Available Run Modes

The project supports multiple run configurations to suit different development and deployment needs:

### 1. Docker Mode (Recommended for Production/Testing)
- **Description**: Complete containerized environment
- **Components**:
  - API Gateway (FastAPI)
  - PostgreSQL database
  - Redis for caching
  - pgAdmin interface
- **Features**:
  - Isolated environment
  - Consistent across all machines
  - Easy dependency management

### 2. Hybrid Mode (Recommended for Development)
- **Description**: Mix of local and containerized services
- **Components**:
  - Local API server
  - Docker containers for:
    - PostgreSQL
    - Redis
- **Benefits**:
  - Faster development iterations
  - Real-time code changes
  - Lightweight on resources

### 3. Local Development Mode
- **Description**: Everything runs locally
- **Use Case**:
  - When you need to modify core dependencies
  - For debugging low-level issues
  - When Docker isn't available

## Getting Started

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (for Docker and Hybrid modes)
- CUDA-compatible GPU (recommended for training)

### Installation

#### Prerequisites for All Modes
```bash
# Clone the repository
git clone <repository-url>
cd solar-image-processing

# Create a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core requirements
pip install -r requirements.txt
```

#### 1. Docker Mode (Recommended for Production/Testing)
```bash
# Start all services (API, PostgreSQL, Redis, pgAdmin)
./scripts/start-backend-docker.sh

# Access the application:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - pgAdmin: http://localhost:8081
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379

# Stop the services
docker-compose -f docker-compose.backend.yml down
```

#### 2. Hybrid Mode (Recommended for Development)
```bash
# Start only database services in Docker
./scripts/start-backend-hybrid.sh

# In a separate terminal, start the API locally
cd backend
uvicorn app.main:app --reload

# When done, stop the services
./scripts/stop-backend-hybrid.sh
```

#### 3. Local Development Mode
```bash
# Install all dependencies including development tools
pip install -r requirements-dev.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start the API
cd backend
uvicorn app.main:app --reload
```

### Running Tests
```bash
pytest tests/
```

### Starting the API
```bash
uvicorn backend.app.main:app --reload
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- PyTorch team for the deep learning framework
- FastAPI for the web framework
- The open-source community for various libraries and tools
