# Step 1 – Prepare Data and Environment

Follow these practical actions to move the project forward.

---

## 1. Set-Up the Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Confirm that `python -c "import torch, cv2; print(torch.__version__, cv2.__version__)"` runs without errors.

---

## 2. Collect a Starter Dataset
1. Mount a camera (phone or webcam) above a panel.
2. Capture **≥ 100** photos each of *clean* and *dirty* panels.
3. Store raw images in `data/raw/<capture_session>/` (keep original resolution).  
   Example: `data/raw/2025-06-14_morning/IMG_0001.jpg`.

### Tips
- Vary lighting (morning/noon/evening) and weather.
- Include partially dirty cases for future multi-class support.

---

## 3. Curate & Label
Create two folders under `data/processed/train`: `clean/` and `dirty/`.
Move/copy images accordingly. Do the same for `data/processed/val` (15 %) and `data/processed/test` (15 %).

Shortcut: run the helper script below to auto-split after manual labelling:
```bash
python scripts/split_dataset.py --src data/processed/all --train 0.7 --val 0.15 --test 0.15
```
*(Create `scripts/split_dataset.py` later or split manually.)*

---

## 4. Sanity-Check the Data Loader
```python
python - <<'PY'
from pathlib import Path
from torchvision import transforms
from src.data.dataset import SolarPanelDataset

ds = SolarPanelDataset(Path('data/processed/train'), transform=transforms.ToTensor())
print('Samples:', len(ds))
img, label = ds[0]
print('Image tensor shape:', img.shape, 'label:', label)
PY
```
If this prints a tensor of shape `[3, H, W]` without exceptions, the loader works.

---

## 5. First Training Run
```bash
python src/train.py --data_dir data/processed --epochs 5 --batch_size 16
```
Expect the loss to decrease and accuracy to be above random (~0.5). Store the `models/resnet18.pt` weight.

---

## 6. Evaluate on Test Set
Add an `src/eval.py` script (next task) to compute accuracy/precision/recall on `data/processed/test`.

---

## 7. Track Results
Make a `notebooks/EDA.ipynb` or Markdown log to record:
- Dataset sizes per class
- Training/validation curves
- Test metrics

---

## 8. Commit Changes
```bash
git add .
git commit -m "Step 1: data collection & first training run"
```
Push to remote when ready.

Next up: build an evaluation script and add Grad-CAM visualisation to inspect what the model learns.
