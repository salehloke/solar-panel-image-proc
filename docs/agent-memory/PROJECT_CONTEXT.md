# Solar Panel Dirt Detection Project

Persistent memory store for the Solar Panel Dirt Detection system. This project uses computer vision and PyTorch to monitor solar panel cleanliness and optimize energy production.

## 🏗️ Project Architecture

- **ML Pipeline**: ResNet18-based binary classifier (clean/dirty).
- **Backend**: FastAPI with PostgreSQL (Analysis results) and Redis (Caching).
- **Frontend**: Next.js (App Router) with Tailwind CSS.
- **Infrastructure**: Containerized using Docker and Docker Compose.

## 📂 Project Structure (Source)

- `backend/`: FastAPI application and database models.
- `src/`: Core training and evaluation scripts.
- `src_edge/`: Edge: Classical ML training (HOG/GLCM).
- `pytorch/`: Model architecture definitions and integration layers.
- `frontend/`: Next.js dashboard for real-time monitoring.
- `scripts/`: Utility scripts for pipeline orchestration and volume management.

## 🚀 Running the System

### Edge-Optimized Stack (Recommended)
```bash
docker-compose -f docker-compose.edge.yml up --build
```
- API: http://localhost:8000
- UI: http://localhost:3000

### Deep Learning Stack
```bash
docker-compose -f docker-compose.backend.yml up --build
```
- API: http://localhost:8000
- pgAdmin: http://localhost:8081

## 🧠 Memory Context

- **Task IDs**: Use `0001-NNNN` running numbers for project tasks.
- **Tech Stack**: Python 3.11, PyTorch 2.2, FastAPI, Next.js 15, PostgreSQL 15.
- **Workflow**: Follow the standards in `ai-task-memory/standards/AGENT_WORKFLOW.md`.

## 🛠️ Key Files & Entrypoints

- `backend/app/main.py`: Main API entrypoint.
- `src/train.py`: Enhanced training script with early stopping.
- `scripts/train_pipeline.py`: End-to-end pipeline orchestrator.
- `pytorch/models/resnet_model.py`: Custom ResNet implementation.
