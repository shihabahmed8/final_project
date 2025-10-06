
# Produce Quality Dashboard

Real‑time produce quality / ripeness prediction using a fine‑tuned EfficientNet model. Supports upload, browser camera snapshot, and OpenCV capture with automatic database logging and lightweight analytics.

---
## 📂 Dataset (Summary)
- ~11,805 images (5 classes)
- Split concept: ~80% train / ~20% test
- Image size normalized to 224×224
- Classes stored in `models/class_names.json`

## ⚙️ Stack
Python, PyTorch, Streamlit, OpenCV, Pillow, Pandas, NumPy, Plotly, SQLite.

## ✨ Key Features
| Feature | Description |
|---------|-------------|
| Image upload | Predict class + probability + freshness index |
| Browser camera | One‑shot capture via `st.camera_input` |
| OpenCV fallback | Single frame capture if browser permission fails |
| Live loop script | `scripts/camera_predict.py` overlays predictions on video |
| DB logging | Saves sample, class, confidence, freshness index |
| Hash dedupe | Prevents duplicate storage of same image bytes |
| Analytics | Class distribution + average freshness + recent scans |
| Admin ops | Delete item set / reset entire database |

## 🧱 Project Structure

---
## 📌 Key Features
| Feature | Description |
|---------|-------------|
| Upload prediction | Upload a JPEG/PNG to get class + probability + freshness index |
| Browser camera capture | Use Streamlit's `camera_input` (Chrome/Edge recommended) |
| OpenCV single capture (in dashboard) | Fallback when browser permissions fail |
| Stand‑alone live video loop | `scripts/camera_predict.py` overlays top‑K predictions in real time |
| DB integration | Stores samples, predicted class, confidence, freshness index, derived storage hints |
| Model loading | EfficientNet-B0 fine‑tuned; configurable via `models/` artifacts |
| Heuristic freshness | Color + darkness metrics → simple freshness index & shelf estimation |

---
## 📂 Current Project Structure
```
final_project/
├── dashboard/
│   └── app.py                # Streamlit dashboard (upload + camera + DB views)
├── data/                     # Incoming & captured images (subset shown)
├── db/
│   ├── produce.db            # SQLite database
│   └── schema.sql            # (Schema reference)
├── models/
│   ├── best_efficientnet_b0.pth
│   └── class_names.json
├── scripts/
│   ├── camera_predict.py     # Live OpenCV loop (video window)
│   └── diagnose_python.ps1   # (Optional) environment diagnostic
├── src/
│   ├── train_banana.py       # Training script (MobileNetV2 example)
│   ├── model_pytorch.py      # Model helper(s)
│   ├── pipeline.py           # Processing + DB insertion helpers
│   ├── preprocessing.py      # Image preprocessing utilities
│   ├── db_utils.py           # SQLite helpers
│   ├── db_init.py            # (Optional) DB initialization
│   └── tools/                # Dataset splitting / labeling helpers
├── Preprocessing_training_testing.ipynb
├── requirements.txt
└── README.md
```

---
## 🧪 Classes
See `class_names.json`. Freshness index (simple heuristic) calculated in `pipeline.py`.

---
## 🛠 Setup (Windows PowerShell)
```powershell
cd path\to\final_project
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PyTorch install is slow or you need CPU-only wheel explicitly:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Then install remaining dependencies if needed:
```powershell
pip install -r requirements.txt --no-deps
```

---
## 🏋️ Training (Example)
Ensure your CSV / image layout matches expectations in `train_banana.py`, then:
```powershell
python src\train_banana.py
```
Model artifacts will be saved into `models/`.

---
## 🚀 Run the Dashboard
```powershell
streamlit run dashboard/app.py
```
Open: http://localhost:8501

### Inputs
1. Upload file (JPEG/PNG)
2. Browser camera snapshot
3. OpenCV single capture button
Prediction fires automatically when a new image + item name are present.

### Outputs
Prediction metrics, probability bar/table, class distribution pie, average freshness, recent scans, gallery, admin & debug info.

---
## ✅ Completed
Dataset prep, preprocessing, model fine‑tune, upload + camera prediction, OpenCV script, DB logging, analytics tabs, admin actions.

---
## � Possible Next
- GPU inference toggle
- Batch reprocess script
- Docker image

---
## 🙌 Acknowledgements
* PyTorch & TorchVision
* Streamlit
* OpenCV community
