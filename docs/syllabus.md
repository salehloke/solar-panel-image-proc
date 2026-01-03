# Solar Panel Dirt Detection — Relearning Syllabus (Hands‑on)

Goal: Rebuild end‑to‑end understanding (ML + API + Infra) and ship a maintainable, tested prototype you can iterate on.
Audience: You (developer re‑onboarding). Duration: 6–8 weeks (part‑time) or 3–4 weeks (full‑time).

## Structure
- Modules are sequenced from fundamentals → project specifics → deployment.
- Each module has: outcomes, prerequisites, topics, labs, deliverables.
- “Essentials” are marked. Optional advanced topics can be deferred.

## Module 0 — Setup & Baseline (Essentials)
Outcomes
- Working Python env, GPU optional; can run and debug code.
- Can run FastAPI locally and hit endpoints.

Prereqs: Basic Python + Git.

Topics
- Python 3.11+, venv, pip/uv: `pip install -r requirements.txt`
- VS Code + Python/Black extensions; pre‑commit (optional)
- Repo tour: `src/`, `backend/`, `pytorch/`, `docs/`, `scripts/`, `models/`, `data/`
- Environment variables: `.env`, `DATABASE_URL`

Labs
- Create venv; install deps; run: `uvicorn backend.app.main:app --reload`
- Hit `GET /health`; observe model status and DB connectivity

Deliverables
- Local run guide (one‑pager) and `.env.example` aligned with your machine

## Module 1 — ML Fundamentals Refresher (Essentials)
Outcomes
- Recall supervised classification pipeline & metrics.

Topics
- Supervised learning, bias/variance, train/val/test splits
- Metrics: accuracy, precision/recall, ROC AUC; confusion matrix
- Data leakage & augmentations

Labs
- On a tiny toy dataset, compute metrics manually in a Jupyter notebook

Deliverables
- Short notes + 1 notebook with metrics examples

## Module 2 — PyTorch Core (Essentials)
Outcomes
- Can read/write training loops and datasets used in this project.

Topics
- Tensors, autograd, nn.Module, optimizers, schedulers
- DataLoader, Dataset; transforms/augmentations (torchvision)
- Transfer learning with ResNet18

Labs
- Implement a minimal classifier with a small dataset
- Modify `src/data/dataset.py` & `src/train.py` to print shapes/overfit a small batch

Deliverables
- Script run showing 1–2 epochs and loss decreasing

## Module 3 — Computer Vision for This Project (Essentials)
Outcomes
- Understand choices in `pytorch/models/resnet_model.py` and training strategy.

Topics
- Pretrained ResNet18; freezing vs fine‑tuning
- Image preprocessing and augmentations relevant to solar panels
- Class imbalance handling (weights, sampling)

Labs
- Train ResNet18 on a subset; log metrics; save best weights to `models/`
- Reproduce `training_summary.md` with your new timestamp/log path

Deliverables
- Trained `models/resnet18_solar_panel.pt` + updated `training_summary.md`

## Module 4 — FastAPI Basics to Advanced (Essentials)
Outcomes
- Can build/extend the API, validate inputs, and return predictions.

Topics
- FastAPI app/lifespan; routers; Pydantic models
- File uploads, validation (size/type)
- Async/await, threadpool offload for CPU‑bound inference
- CORS, error handling

Labs
- Read `backend/app/main.py`; document current endpoints
- Add a new `GET /model/info` returning model metadata from `get_model_info()`

Deliverables
- API runs; Swagger shows your new endpoint

## Module 5 — Persistence with SQLAlchemy Async (Essentials)
Outcomes
- Can persist analysis results and query them safely.

Topics
- AsyncSession, models, CRUD patterns
- Migrations (Alembic) basics (optional to start)

Labs
- Point `DATABASE_URL` to local Postgres (or SQLite for dev)
- Exercise create/read of analysis results via the existing endpoints

Deliverables
- Verified DB connectivity in `/health`; saved result from a test call

## Module 6 — Integration: Model Loader + Predict Flow (Essentials)
Outcomes
- Understand how the model is loaded and used by the API.

Topics
- `pytorch/integration.py` and `backend/app/utils/model_loader.py`
- Threading vs async; CPU vs GPU inference
- Error paths: missing model weights; improper tensor shape

Labs
- Log model load time and device in startup
- Benchmark one image prediction; record latency in `processing_time`

Deliverables
- Screenshot/logs for a successful `/predict` with confidence score

## Module 7 — Packaging & Containers (Essentials)
Outcomes
- Can build and run the system via Docker Compose.

Topics
- Dockerfile basics; slim Python image; pinned base
- docker-compose services: API + Postgres (+ MinIO optional)
- Multi‑stage builds; caching; `.dockerignore`

Labs
- Build images and `docker compose up` the stack
- Verify `/health` and `/predict` in containerized env

Deliverables
- Working compose stack; short README snippet to run it

## Module 8 — Observability & Logging (Recommended)
Outcomes
- Basic logging strategy and health visibility.

Topics
- Structured logs; log levels; request IDs
- Prometheus/Grafana (outline); FastAPI metrics middleware (optional)

Labs
- Add minimal logging config; include model version and timing

Deliverables
- Logs showing prediction flow and timings

## Module 9 — Authentication & Security (Recommended)
Outcomes
- Protect endpoints for multi‑user contexts.

Topics
- JWT in FastAPI; dependency injection for auth
- Input validation, rate limiting (via reverse proxy later)

Labs
- Add a protected `/me` endpoint; mock user in dev

Deliverables
- Postman collection with tokens + example calls

## Module 10 — Frontend (Optional Initial Cut)
Outcomes
- Minimal UI to upload an image and view prediction.

Topics
- React (or Nx monorepo as per plan); simple upload form
- API client; error & loading states

Labs
- Build a minimal dashboard page; call `/predict`

Deliverables
- Screenshot of end‑to‑end interaction

## Module 11 — Testing & Quality (Essentials)
Outcomes
- Confidence in changes via tests.

Topics
- pytest for API (httpx/pytest‑asyncio)
- Unit tests for dataset, transforms, and model loader
- Smoke test for `/health` and `/predict`

Labs
- Add tests under `tests/` for API and ML utilities

Deliverables
- `pytest -q` green; simple coverage report

## Module 12 — Deployment & Next Steps (Recommended)
Outcomes
- Clear path to staging/production.

Topics
- Reverse proxy (Traefik), HTTPS, service routing
- Basic CI (lint/test/build images)
- Artifact registry + pinned images

Labs
- Write a CI pipeline draft (GitHub Actions) to run tests and build images

Deliverables
- CI YAML and a one‑pager “Deploy playbook”

---

## Reading & Resources
- PyTorch: https://pytorch.org/tutorials/
- FastAPI: https://fastapi.tiangolo.com/
- SQLAlchemy async: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- Docker: https://docs.docker.com/
- ML Ops (optional): DVC, Weights & Biases, TensorBoard

## Milestones
- Week 1–2: Modules 0–4 (local ML + API running)
- Week 3: Modules 5–7 (DB + containers)
- Week 4: Modules 8–9 (obs + auth) and/or Module 10 (UI)
- Week 5: Module 11 (tests) + 12 (deploy plan)

## Checklists
- [ ] Local env + API health OK
- [ ] Model trained + weights saved
- [ ] Predict endpoint functional end‑to‑end
- [ ] DB persistence verified
- [ ] Containerized run verified
- [ ] Basic logs in place
- [ ] Minimal auth (dev)
- [ ] Optional UI MVP
- [ ] Tests green; CI draft

