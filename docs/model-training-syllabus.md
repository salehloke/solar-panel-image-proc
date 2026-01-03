# Model Training & Prompt Crafting Syllabus

- **Prerequisites & Environment (Week 0)**
  - Refresh Python, linear algebra, and probability basics; review PyTorch tensors and autograd tutorials.
  - Clone `solar-panel-image-proc`, set up a virtual environment, and verify `pip install -r requirements.txt` completes.
  - Explore repository structure (`src/`, `pytorch/models/`, `scripts/`, `backend/`) and read `README.md`, `quickstart.md`, `PROJECT_SUMMARY.md`.
  - Document baseline hardware constraints (CPU/GPU, RAM, storage) and dataset availability to drive realistic experiment plans.

- **Module 1: Supervised Learning Fundamentals (Week 1)**
  - Revisit dataset splits, bias/variance, loss surfaces, and evaluation metrics (accuracy, precision/recall, F1, ROC).
  - Walk through `src/train.py` to map each concept to code (transforms, loaders, optimizer, scheduler, early stopping).
  - Implement a notebook or script that trains on a tiny synthetic dataset to visualize overfitting vs. regularization.
  - Summarize learnings in a journal entry; craft a short prompt explaining these fundamentals to validate understanding.

- **Module 2: Data Engineering & Augmentation (Week 2)**
  - Use `scripts/split_dataset.py` to organize raw images; inspect `SolarPanelDataset` to understand label inference.
  - Experiment with the augmentation pipeline in `get_transforms()`—toggle flips, rotations, color jitter—and measure effects on validation metrics.
  - Analyze class imbalance with histogram plots; compute class weights using `calculate_class_weights()` and compare training runs with/without weights.
  - Draft prompts that precisely describe dataset issues (e.g., imbalance, lighting variance) and request targeted augmentation strategies.

- **Module 3: Architecture & Optimization (Week 3)**
  - Dive into `pytorch/models/resnet_model.py`; replace ResNet18 with deeper or lightweight backbones and benchmark speed vs. accuracy.
  - Study optimizers (SGD, AdamW) and learning-rate schedulers (`ReduceLROnPlateau`) by running controlled experiments.
  - Perform ablation studies on hyperparameters (LR, batch size, weight decay, patience) and log findings in `logs/<run>/metrics.json`.
  - Practice prompts that ask for model/optimizer trade-offs, citing concrete metrics from your experiments.

- **Module 4: Training Pipeline Mastery (Week 4)**
  - Execute `scripts/train_pipeline.py` end-to-end; trace each stage (dependency check, splitting, training, evaluation, model export).
  - Inspect generated artifacts (`training_curves.png`, `confusion_matrix.png`, `config.json`) and correlate them with console logs.
  - Automate experiment tracking (e.g., spreadsheet or lightweight tracker) capturing hyperparameters, best F1, confusion-matrix insights.
  - Write prompts that request troubleshooting advice using specific log snippets and metric trends.

- **Module 5: Evaluation, Diagnostics & Explainability (Week 5)**
  - Use `src/evaluate.py` to run on held-out and stress-test datasets; interpret precision/recall per class.
  - Create custom confusion-matrix analyses (per-lighting condition, per-camera) to pinpoint failure modes.
  - Explore saliency/Grad-CAM libraries to visualize model focus areas on dirty vs. clean panels.
  - Formulate prompts that present diagnostic evidence (e.g., false positives in glare) and ask for mitigation strategies.

- **Module 6: Deployment & Inference (Week 6)**
  - Integrate trained weights into the FastAPI backend (`backend/run.py`); test inference through the `/prediction/` endpoint.
  - Containerize the service using `Dockerfile` and `docker-compose.yml`; measure latency under realistic workloads.
  - Implement lightweight monitoring hooks (latency, confidence thresholds) and log them for future prompt context.
  - Craft prompts describing deployment constraints (hardware limits, expected QPS) to gather optimization advice.

- **Module 7: Prompt Engineering for ML Ops (Week 7)**
  - Build a reusable prompt template covering context, objective, constraints, prior attempts, and desired output format.
  - Rehearse conversations with LLMs about debugging, experiment design, and documentation; score responses vs. expectations.
  - Iterate on prompt clarity by comparing vague vs. specific instructions and noting response deltas.
  - Compile a personal “prompt playbook” with best-performing examples and counterexamples.

- **Capstone Project (Week 8)**
  - Define a real-world scenario (e.g., detect soiling severity tiers) and run the full pipeline—from data prep to backend integration.
  - Maintain a detailed experiment log, including metrics and prompt interactions that influenced decisions.
  - Produce a short report summarizing approach, lessons learned, and a refined prompt template for future collaborators.
  - Present findings to a peer or mentor, using repository artifacts (logs, models, API demo) as evidence of mastery.
