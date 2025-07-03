# Solar Panel Cleanliness Detection — Project Guide

Detecting dirt accumulation on photovoltaic (PV) panels from images can help maintain optimal energy generation and schedule cleaning efficiently. This guide lays out everything you need to start and carry out an end-to-end image-based dirt detection project.

---

## 1. Project Objectives
1. Acquire images of solar panels under varying lighting and weather conditions.
2. Build a model (classical CV or deep learning) that classifies each image as **Clean** or **Dirty** (or outputs a dirt‐level score).
3. Evaluate the model’s reliability on unseen data.
4. (Optional) Deploy the model on an edge device or cloud service for real-time monitoring.

---

## 2. Prerequisites & Tooling

| Category | What you need | Purpose |
|----------|---------------|---------|
| Hardware | A camera (RGB or multispectral) mounted to view the panel surface. | Image acquisition |
|          | A workstation/GPU (NVIDIA 8 GB+ recommended) | Training deep models |
| Datasets | Labeled images (Clean/Dirty) of solar panels. Collect yourself or scrape public datasets. | Model training |
| Software | Python ≥3.9 | Primary language |
|          | OpenCV | Image preprocessing |
|          | NumPy / SciPy | Numeric ops |
|          | scikit-image | Classical CV filters |
|          | PyTorch **or** TensorFlow/Keras | CNN model implementation |
|          | scikit-learn | Metrics & baselines |
|          | Matplotlib / seaborn | Visualisation |
| DevOps   | Git + GitHub | Version control |
|          | Poetry / pipenv / venv | Dependency management |

Install dependencies (example with `venv`):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python torch torchvision scikit-image scikit-learn matplotlib seaborn
```

---

## 3. Recommended Repository Layout
```
solar-image-processing/
├── data/            # Raw & processed datasets (git-ignored)
│   ├── raw/
│   └── processed/
├── notebooks/       # Exploration & EDA
├── src/
│   ├── data/        # Data loading & augmentation
│   ├── models/      # CNN architectures or CV pipelines
│   ├── train.py     # Training entry point
│   └── eval.py      # Evaluation script
├── tests/           # Unit tests
├── README.md        # (this file)
└── requirements.txt # frozen dependencies
```

---

## 4. Step-by-Step Roadmap

### 4.1 Data Collection
1. Capture images every few hours for several days.
2. Label each image manually (`clean`, `dirty`, `partially_dirty`, etc.) using a tool like [LabelImg](https://github.com/tzutalin/labelImg) or a simple CSV.
3. Split into train/val/test (e.g., 70 / 15 / 15 %).

### 4.2 Data Pre-processing
- **Cropping/Masking**: Isolate the panel area; optionally apply a segmentation mask to ignore background.
- **Resize & Normalize**: Standardise resolution (e.g., 224×224) and scale pixel values to [0,1].
- **Augmentation**: Random brightness, rotation, flips to simulate lighting & orientation variations.

### 4.3 Baseline Model (Classical CV)
```python
# Example: simple threshold on HSV saturation to detect dust-induced dullness
blur = cv2.GaussianBlur(img, (5,5), 0)
hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
_, s, _ = cv2.split(hsv)
score = s.mean()  # lower saturation ⇒ dirtier
```
- Evaluate with scikit-learn metrics (accuracy, ROC-AUC).

### 4.4 Deep Learning Model
1. Start with a pretrained CNN (e.g., ResNet-18) via transfer learning.
2. Replace final layer with 2-class output.
3. Train for 10-20 epochs, monitor validation F1-score.
4. Perform hyper-parameter tuning (learning rate, augmentations).

### 4.5 Evaluation
- Confusion matrix on the held-out test set.
- Precision-Recall curve (important if dirt occurrences are rare).
- Grad-CAM visualisations to verify the model focuses on dirt spots.

### 4.6 Deployment (Optional)
- Export model to ONNX or TensorFlow Lite.
- For edge: Run inference on a Raspberry Pi with a camera.
- For cloud: Expose prediction via FastAPI endpoint (`src/api.py`) and host on AWS/GCP.

---

## 5. Getting Started Quickly
1. Clone this repo: `git clone <url>`.
2. Create and activate a virtual environment.
3. Place a handful of labelled images in `data/raw/` and update `src/data/dataset.py`.
4. Run `python src/train.py --config configs/baseline.yaml`.
5. View TensorBoard logs: `tensorboard --logdir runs/`.

---

## 6. Further Reading
- PV panel soiling datasets: "Solar Panel Surface Soiling Dataset (SPSD)", IEEE DataPort.
- Paper: Reyes, et al. *"An Image-Based Monitoring System for Photovoltaic Panels"* (2021).
- OpenCV Image Processing Tutorials: <https://docs.opencv.org/4.x/d9/df8/tutorial_root.html>

---

## 7. Contribution Guidelines
Pull requests are welcome. Please:
1. Fork and create a feature branch.
2. Add unit tests (`pytest`).
3. Run `pre-commit run --all-files` before pushing.

---

## 8. License
Specify an OSI-approved license (e.g., MIT) once you decide.

---

Happy hacking!
