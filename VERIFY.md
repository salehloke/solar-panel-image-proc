# Verification Guide: Solar Panel Dirt Detection System

This guide explains how to verify the current progress and run the system on any machine using Docker.

## 🏗️ System Overview
The project supports two deployment modes:
1.  **Cloud/Heavy**: ResNet18 (PyTorch) + PostgreSQL + Redis.
2.  **Edge/Lite**: SVM (HOG/GLCM) + SQLite (Optimized for Raspberry Pi).

---

## 🚀 1. Run the Edge-Optimized Stack (Recommended for Verification)
This mode is the most recent addition and is designed to be extremely lightweight.

### Build and Start
```bash
docker-compose -f docker-compose.edge.yml up --build -d
```

### Verify Components
- **Dashboard**: [http://localhost:3000](http://localhost:3000)
- **Edge API**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: `curl http://localhost:8000/health`

### Test Detection
1. Open the Dashboard.
2. Navigate to **Analysis History** to see local SQLite records.
3. Click **Remote Pi Capture** (Simulated on non-Pi hardware via local camera).
4. Upload an image from `data/train/` to see the multi-class classification (Dust, Moss, Bird Droppings).

---

## ☁️ 2. Run the Deep Learning Stack
This mode uses the heavier PyTorch models.

### Build and Start
```bash
docker-compose -f docker-compose.backend.yml up --build -d
```

### Verify Components
- **API Gateway**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **pgAdmin**: [http://localhost:8081](http://localhost:8081) (Login: `admin@solarai.com` / `admin123`)

---

## 🛠️ 3. Development & Training Verification
If you want to verify the training logic for the edge models:

1. **Install Local Environment**:
   ```bash
   pip install -r backend_edge/requirements-pi.txt
   ```

2. **Run Edge Training**:
   ```bash
   python3 src_edge/train.py --data_dir data/train --output_dir backend_edge/app/models
   ```

3. **Run Edge Evaluation**:
   ```bash
   python3 src_edge/evaluate.py --data_dir data/test --model_dir backend_edge/app/models
   ```

---

## 🧪 4. Cross-Machine Test Checklist
- [ ] Docker and Docker Compose installed.
- [ ] At least 4GB RAM available (for DL stack) or 1GB RAM (for Edge stack).
- [ ] Ports 3000, 8000, 5432, 6379 are free.
- [ ] `.env` file exists (can be copied from `.env.example`).

---

**Current Progress Status**:
- ✅ Unified Docker logic for both Edge and Cloud.
- ✅ Persistent local database (SQLite) for edge node.
- ✅ Multi-class visualization in frontend.
- ✅ End-to-end pipeline from capture to analytics.
