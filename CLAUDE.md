# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Description

Facial expression recognition system (表情识别系统) using PyTorch. A `CNNWithAttention` model classifies 48x48 grayscale face images into 7 emotion classes: angry, disgust, fear, happy, sad, surprise, neutral.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train a new model (data in ./train/ and ./test/ organized by emotion subfolders)
python train.py --data_dir . --epochs 50 --batch_size 32

# Train with class balancing / weighted loss (quick_fix.py)
python quick_fix.py

# Single image inference (uses best_fixed_model.pth by default)
python simple_enhanced_inference.py --image <path> --model best_fixed_model.pth

# Batch processing
python simple_enhanced_inference.py --batch <folder> --model best_fixed_model.pth --visualize

# Webcam real-time recognition
python simple_enhanced_inference.py --webcam --model best_fixed_model.pth

# Evaluate model on test set
python evaluate_model.py --model best_model.pth --test_dir ./test

# Show evaluation visualization (confusion matrix, performance charts)
python show_results.py

# Interactive menu (single/batch/evaluate)
python simple_emotion_recognition.py

# Visualize data distribution (pie charts for train/test splits)
python data_loader.py --visualize --data_dir .
```

Base inference script (less features):
```bash
python inference.py --image test.jpg
python inference.py --webcam
```

## Full-Stack Web App

```bash
# Backend (use the conda env that has PyTorch installed)
cd backend
pip install -r requirements.txt
# From project root:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm install
npm run dev          # → http://localhost:5173
npm run build        # production build → frontend/dist/
```

Backend API:
- `GET /api/health` → `{"status": "ok", "model_loaded": true}`
- `POST /api/predict` (multipart form, field `file`) → `{"emotion", "confidence", "probabilities", "processing_time_ms"}`

Vite dev server proxies `/api/*` → `localhost:8000`, so no CORS issues during development.

## Project Architecture

### Code Layout

- **`model.py`** — Defines `CNNWithAttention` (main model), `SelfAttention` (spatial attention), `SEBlock` (channel attention). 4 conv blocks (1→64→128→256→512 channels) with attention after conv3 and conv4, global avg pooling, 3-layer classifier (512→256→128→7), ~4.2M params.

- **`data_loader.py`** — `FER2013Dataset` class loading images from `./train/<emotion>/` and `./test/<emotion>/` folder structure. Provides `get_data_loaders()` with train transforms (augmentation: flip, rotation, affine, crop) and val/test transforms (resize + normalize). Mean/std normalization: [0.5076], [0.2128]. Also has `visualize_data_distribution()` for pie charts.

- **`train.py`** — `Trainer` class wrapping training loop, validation, test evaluation. Uses AdamW (lr=0.001, wd=1e-4), ReduceLROnPlateau scheduler, CrossEntropyLoss. Saves best model to `best_model.pth`. Plots training curves and confusion matrix.

- **`quick_fix.py`** — Alternative training with `WeightedRandomSampler` for class balancing, stronger data augmentation, `WeightedCrossEntropyLoss`, pretrained weight loading, and early stopping. Saves to `best_fixed_model.pth`.

- **`inference.py`** — Base `EmotionRecognizer` class with single/batch prediction, visualization, and webcam support. Uses Haar Cascade face detection.

- **`simple_enhanced_inference.py`** — Enhanced `SimpleEnhancedEmotionRecognizer` with multiple cropping strategies (no_crop, smart_center, rule_based, center_region), image quality enhancement (histogram equalization, sharpening), Chinese-path support via `cv2.imdecode`, batch processing, and comparison mode. Main inference entrypoint.

- **`evaluate_model.py`** — `ModelEvaluator` class computing accuracy, per-class precision/recall/F1, confusion matrix, classification report. Outputs to `evaluation_results/`.

- **`show_results.py`** — Renders pre-computed evaluation metrics: confusion matrix heatmap, precision/recall/F1 bar charts, accuracy ranking, accuracy-vs-sample-count scatter, confidence analysis.

- **`simple_emotion_recognition.py`** — Interactive CLI menu wrapping the enhanced inference and evaluation scripts.

- **`backend/main.py`** — FastAPI app with `/api/health` and `/api/predict` endpoints. Loads model at startup, CORS enabled. Default model path resolves to `../best_model.pth` relative to this file.

- **`backend/model_handler.py`** — `ModelHandler` singleton: loads `CNNWithAttention`, preprocesses uploaded image bytes (decode → grayscale → resize 48×48 → normalize), runs inference, returns (emotion, confidence, probabilities).

- **`frontend/src/`** — React app: `ImageUpload` (drag-and-drop via react-dropzone), `ResultDisplay` (emotion + confidence bar + emoji), `ProbabilityChart` (Recharts bar chart of 7-class probabilities).

### Data Format

Images organized by emotion subfolders:
```
./
├── train/
│   ├── angry/    (images)
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    └── (same structure)
```

### Key Files

| File | Purpose |
|------|---------|
| `best_model.pth` | Standard training checkpoint (~64MB) |
| `best_fixed_model.pth` | Balanced training checkpoint (~64MB) |
| `haarcascade_frontalface_default.xml` | OpenCV face detector |
| `requirements.txt` | torch, torchvision, opencv-python, numpy, Pillow |
| `myPhoto/` | Sample images for testing |
| `inference_results/` | Visualization output directory |
| `evaluation_results/` | Evaluation reports and charts |

### Preprocessing Pipeline

1. Read image (supports Chinese paths via `cv2.imdecode`)
2. Optional: center/smart/rule-based cropping
3. Resize to 48x48
4. Convert to grayscale
5. Normalize: `(x - 0.5076) / 0.2128`

### Model Input/Output

- Input: `(batch, 1, 48, 48)` normalized grayscale tensor
- Output: `(batch, 7)` logits for [angry, disgust, fear, happy, sad, surprise, neutral]
