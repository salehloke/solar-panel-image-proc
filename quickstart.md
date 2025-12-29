# Quickstart Guide: Solar Panel Dirt Detection System

This guide will help you quickly set up, train, and run the solar panel dirt detection system. The project supports two modes: **Edge (Lightweight)** and **Cloud (Deep Learning)**.

---

## 1. Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd solar-image-processing
   ```

2. **Install local dependencies (optional, for development):**
   ```bash
   pip3 install -r requirements.txt
   ```

---

## 2. Prepare Your Data

Organize your images in the `data/` directory:
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

---

## 🚀 3. Run the System (Docker Compose)

The easiest way to run the system is using Docker Compose.

### Option A: Edge-Optimized Stack (Recommended for Pi/Verification)
Uses SVM + HOG/GLCM features. Extremely lightweight and supports multi-class classification.
```bash
docker-compose -f docker-compose.edge.yml up --build
```
- **Dashboard**: http://localhost:3000
- **Edge API**: http://localhost:8000/docs

### Option B: Cloud/Deep Learning Stack
Uses ResNet18 (PyTorch) + PostgreSQL + Redis.
```bash
docker-compose -f docker-compose.backend.yml up --build
```
- **API Gateway**: http://localhost:8000/docs
- **pgAdmin**: http://localhost:8081

---

## 🛠️ 4. Manual Training (Edge)

To train the lightweight edge models locally:
```bash
# 1. Install edge requirements
pip install -r backend_edge/requirements-pi.txt

# 2. Run training
python3 src_edge/train.py --data_dir data/train --output_dir backend_edge/app/models

# 3. Run evaluation
python3 src_edge/evaluate.py --data_dir data/test --model_dir backend_edge/app/models
```

---

## ☁️ 5. Manual Training (Deep Learning)

Run the complete DL training pipeline:
```bash
python3 scripts/train_pipeline.py --epochs 10 --use_class_weights
```

---

## 🧪 6. Verification

Refer to `VERIFY.md` for a comprehensive multi-machine verification checklist.

---

**Happy Solar Panel Monitoring! 🌞**