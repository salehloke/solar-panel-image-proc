# Solar Panel Dirt Detection System

A comprehensive deep learning system for detecting dirt accumulation on solar panels using computer vision and PyTorch. This project provides an end-to-end solution from data processing to deployment.

## 🌟 Features

- **Hybrid AI Architecture**: Supports both Deep Learning (ResNet18/PyTorch) and Lightweight Edge (SVM/HOG/GLCM).
- **Multi-Class Detection**: Detects Clean, Dust, Bird Droppings, and Moss.
- **Production API**: FastAPI backend with real-time predictions and local SQLite analytics for edge.
- **Comprehensive Training**: Advanced pipelines for both CNN and classical ML models.
- **Docker First**: Full containerization for dashboard, API, and training modules.
- **Interactive Dashboard**: Next.js 15 dashboard with real-time status and historical trends.

## 📊 Project Overview

This system helps solar panel operators:

- **Detect dirt accumulation** and specific types (Moss, Bird Droppings) from images.
- **Estimate Efficiency Loss** based on the detected dirt type.
- **Trigger Remote Capture** via Raspberry Pi camera modules.
- **Monitor panel health** across multiple machines using a unified dashboard.

## 🏗️ Architecture

```
solar-image-processing/
├── backend/               # Cloud: PyTorch/DL Backend
├── backend_edge/          # Edge: SVM/Lite Backend (optimized for Pi)
├── src/                   # Cloud: DL training scripts
├── src_edge/              # Edge: Classical ML training (HOG/GLCM)
├── frontend/              # Unified Next.js Dashboard
├── scripts/               # Utility and pipeline scripts
├── data/                  # Shared dataset storage
└── models/                # Saved model weights
```

## 🚀 Quick Start

### 1. Run with Docker Compose (Recommended)

**Run Edge Stack:**
```bash
docker-compose -f docker-compose.edge.yml up --build
```

**Run Cloud Stack:**
```bash
docker-compose -f docker-compose.backend.yml up --build
```

### 2. Manual Installation

```bash
# Clone the repository
git clone <repository-url>
cd solar-image-processing

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Preparation

Your dataset should be organized as:

```
data/
├── train/
│   ├── clean/           # Clean solar panel images
│   ├── dust/            # Edge: Dust images
│   ├── bird_droppings/  # Edge: Bird droppings
│   └── moss/            # Edge: Moss images
└── test/
    ├── clean/
    └── ...
```

### 4. Complete Training Pipeline (Cloud)

Run the complete training pipeline with one command:

```bash
python scripts/train_pipeline.py --epochs 50 --use_class_weights
```

This will:

- ✅ Split your dataset into train/validation/test sets
- ✅ Train the model with advanced features
- ✅ Evaluate the model performance
- ✅ Generate comprehensive reports
- ✅ Save the best model for deployment

### 5. Start the API

**Option A: Edge Stack (Lite)**
```bash
docker-compose -f docker-compose.edge.yml up -d
```
API available at `http://localhost:8000`

**Option B: Cloud Stack (Deep Learning)**
```bash
# Start the FastAPI server locally or via compose
python backend/run.py
```
API available at `http://localhost:8000`

### 6. Make Predictions

Test the API with an image using `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/solar_panel_image.jpg"
```

## 📈 Training Features

### Advanced Training Options

```bash
# Basic training
python src/train.py --data_dir data/processed --epochs 50

# With class weights for imbalanced data
python src/train.py --use_class_weights --lr 1e-4

# Custom configuration
python src/train.py \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-5 \
    --patience 15 \
    --use_class_weights \
    --model_name my_solar_model
```

### Training Features

- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Class Weights**: Handles imbalanced datasets
- **Data Augmentation**: Random crops, flips, color jitter
- **Comprehensive Logging**: Training curves, metrics, confusion matrix
- **Model Checkpointing**: Saves best model automatically

## 🔍 Model Evaluation

### Run Evaluation

```bash
python src/evaluate.py \
    --model_path models/resnet18_solar_panel.pt \
    --data_dir data/processed \
    --output_dir evaluation_results
```

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Balanced measure of precision and recall
- **ROC Curves**: Model discrimination ability
- **Confusion Matrix**: Detailed error analysis

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t solar-panel-detection .

# Run the container
docker run -p 8000:8000 solar-panel-detection
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

## 📊 API Documentation

Once the server is running, visit:

- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### API Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check
- `POST /prediction/`: Upload image for prediction

### Example API Response

```json
{
  "filename": "solar_panel.jpg",
  "prediction": "dirty",
  "confidence": 0.89,
  "status": "success"
}
```

## 🧠 Agent Execution History

This project uses an "Agent OS" workflow to track architectural decisions and task execution. You can find the persistent memory of all AI-driven tasks in:
- `docs/agent-memory/history/`: Completed tasks and bug fixes.
- `docs/agent-memory/active/`: Currently in-progress tasks.
- `docs/agent-memory/PROJECT_CONTEXT.md`: High-level architecture and context for AI agents.

## 📁 Project Structure

### Key Directories

- **`data/`**: Raw and processed datasets
- **`models/`**: Trained model weights
- **`logs/`**: Training logs and visualizations
- **`evaluation_results/`**: Model evaluation outputs
- **`backend/`**: FastAPI application
- **`src/`**: Training and evaluation scripts
- **`scripts/`**: Utility and pipeline scripts

### Important Files

- **`scripts/train_pipeline.py`**: Complete training pipeline
- **`src/train.py`**: Enhanced training script
- **`src/evaluate.py`**: Model evaluation
- **`backend/run.py`**: API server
- **`requirements.txt`**: Python dependencies
- **`Dockerfile`**: Container configuration

## 🎯 End Goals & Roadmap

### ✅ Completed Features

- [x] ResNet18-based classifier
- [x] FastAPI backend
- [x] Comprehensive training pipeline
- [x] Data augmentation
- [x] Class imbalance handling
- [x] Model evaluation
- [x] Docker support
- [x] Production-ready API

### 🚀 Planned Enhancements

- [ ] **Web Interface**: User-friendly web UI
- [ ] **Real-time Monitoring**: Continuous panel monitoring
- [ ] **Maintenance Scheduling**: Automated cleaning recommendations
- [ ] **Performance Analytics**: Historical tracking
- [ ] **Multi-class Classification**: Different dirt types
- [ ] **Edge Deployment**: Raspberry Pi deployment
- [ ] **Cloud Integration**: AWS/GCP deployment

## 🔧 Configuration

### Environment Variables

```bash
# Model configuration
MODEL_DIR=./models
MODEL_PATH=./models/resnet18_solar_panel.pt

# API configuration
HOST=0.0.0.0
PORT=8000
```

### Training Configuration

Key parameters in `src/train.py`:

- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--patience`: Early stopping patience
- `--use_class_weights`: Enable class weights for imbalance

## 📊 Performance

### Model Performance

Based on typical solar panel datasets:

- **Accuracy**: 90-95%
- **F1-Score**: 0.85-0.92
- **Inference Time**: <100ms per image
- **Model Size**: ~45MB (ResNet18)

### Hardware Requirements

- **Training**: GPU with 8GB+ VRAM recommended
- **Inference**: CPU or GPU
- **Memory**: 4GB+ RAM
- **Storage**: 2GB+ for models and logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- FastAPI team for the web framework
- Solar panel datasets contributors
- Open source community

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the training logs for debugging

---

**Happy Solar Panel Monitoring! 🌞**
