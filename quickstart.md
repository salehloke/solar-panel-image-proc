# Quickstart Guide: Solar Panel Dirt Detection System

This guide will help you quickly set up, train, and run the solar panel dirt detection system using PyTorch and FastAPI.

---

## 1. Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd solar-image-processing
   ```

2. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

---

## 2. Prepare Your Data

Organize your images as follows:

```
data/
├── train/
│   ├── clean/   # Clean solar panel images
│   └── dirty/   # Dirty solar panel images
└── test/
    ├── clean/   # (Optional) Test clean images
    └── dirty/   # (Optional) Test dirty images
```

---

## 3. Train the Model

Run the complete training pipeline (from the project root):

```bash
python3 scripts/train_pipeline.py --epochs 10 --use_class_weights
```

- This will split your data, train the model, and save the best model in the `models/` directory.

---

## 4. Run the FastAPI Backend

Start the API server:

```bash
python3 backend/run.py
```

- The API will be available at: [http://localhost:8000](http://localhost:8000)
- Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 5. Make Predictions

Test the API with an image using `curl`:

```bash
curl -X POST "http://localhost:8000/prediction/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/solar_panel_image.jpg"
```

You will receive a response like:

```json
{
  "filename": "solar_panel.jpg",
  "prediction": "dirty",
  "confidence": 0.89,
  "status": "success"
}
```

---

## 6. (Optional) Run with Docker

To build and run everything in Docker:

```bash
docker-compose up --build
```

---

## Troubleshooting

- If you see missing package errors, run: `pip3 install -r requirements.txt`
- If the API cannot find the model, ensure you have trained the model and it is saved in the `models/` directory.
- For more details, see the main `README.md`.

---

**You're ready to detect solar panel dirt with deep learning!**
