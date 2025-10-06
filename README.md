
# 🥦 Produce Quality Dashboard

## 📌 Project Overview
This project focuses on building a **machine learning system** to classify and predict the quality/readiness of vegetables using **image datasets** and training models. The system supports both:
- **Manual image upload** for prediction.
- **Live image capture** from the camera for real-time prediction.

---
## About Dataset

Total number of images: 11805.
Training set size: 80% images.
Test set size: 20% images (one fruit or vegetable per image).
Number of classes: 5 
Image size: 224×224 pixels.

## ⚙️ Tools & Technologies
- **Python 3.12 / 3.13**
- **Jupyter Notebook** (exploration, preprocessing, model training & testing)
- **VS Code** (modular project development, packaging, integration)
- **PyTorch** (model building & training)
- **OpenCV / Pillow** (image preprocessing & capture)
- **Matplotlib & Seaborn** (visualization)
- **Pandas & NumPy** (data handling)
- **Streamlit** (for dashboard / web integration)
- **Git & GitHub** (version control & submission)

---

## 🧑‍💻 Workflow & Steps
# 🥦 Produce Quality Quality / Ripeness Prediction Dashboard

Real-time and batch prediction system for produce (e.g., bananas) quality / ripeness using a PyTorch EfficientNet model plus heuristic color analysis for derived freshness metrics. Includes:

* Streamlit dashboard (upload + camera + DB visualization)
* OpenCV live camera prediction script
* Training & preprocessing utilities
* SQLite persistence of scans and model outputs

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
## 🧪 Classes / Targets
Currently configured (example) classes (see `class_names.json`):
`["Unripe", "Perfectly Ripe", "Overripe", "Bruised/Damaged", "Healthy Leaf"]`

Freshness index & shelf-life heuristics are derived in `pipeline.py`.

---
## 📊 Dataset (Summary)
* ~11.8K images
* 80/20 (train/test) conceptually; validation derived in training script
* Normalized to 224×224 for EfficientNet/MobileNet

---
## 🛠 Environment Setup (Windows PowerShell)
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
## 🏋️ Training (Example MobileNet)
Ensure your CSV / image layout matches expectations in `train_banana.py`, then:
```powershell
python src\train_banana.py
```
Model artifacts will be saved into `models/`.

---
## 🚀 Running the Dashboard
```powershell
streamlit run dashboard/app.py
```
Open: http://localhost:8501

### Dashboard Inputs
1. Upload: Choose a file in sidebar.
2. Camera options (main page section "Capture / Camera"):
   * Toggle "Use browser camera" → take snapshot (requires browser permission).
   * Button "Capture via OpenCV" → grabs a single frame (fallback if browser capture fails).
3. Upon capture or upload + item name, prediction runs automatically and is inserted into DB.

### Output Panels
* Prediction (top‑K table + class metric)
* Distribution & Freshness metrics
* Recent scans table + Gallery
* Admin tools (delete items / reset database)

---
## 🎥 Stand‑Alone Live Camera (Continuous)
```powershell
python scripts\camera_predict.py            # default camera index 0
python scripts\camera_predict.py --camera 1 # alternate camera
python scripts\camera_predict.py --device cuda  # if GPU available
```
Controls in window:
* q / ESC → quit
* c → save frame only
* s → save + insert prediction into DB (requires working pipeline/model)

---
## � Database
SQLite file: `db/produce.db`
Tables (simplified):
* `Produce_Samples(sample_id, item_name, scan_date, image_path, ...)`
* `Quality_Results(result_id, sample_id, quality_class, confidence, freshness_index, ...)`
Insert flow handled in `pipeline.process_and_insert()`.

---
## 🔍 Troubleshooting
| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: numpy` | Activate venv; run `pip install -r requirements.txt` |
| Browser camera not prompting | Use Chrome/Edge; open external browser; check site permissions & OS privacy settings |
| Browser camera still blank | Use OpenCV button or run `scripts/camera_predict.py` |
| `best_efficientnet_b0.pth` not found | Place model weight file in `models/` or retrain |
| Slow predictions | Reduce image size or batch frequency (adjust camera script) |
| DB not updating | Check console for exceptions; confirm `process_and_insert` call |

---
## ✅ Completed Tasks
* Dataset preparation & cleaning  
* EDA & visualization  
* Preprocessing pipeline  
* CNN transfer learning & evaluation  
* Upload prediction feature  
* Browser camera + OpenCV fallback capture  
* Streamlit dashboard integration  
* Database storage integration  
* Stand‑alone live video prediction script  

---
## 📌 Next Possible Enhancements
* GPU inference toggle in dashboard
* Batch reprocessing / backfill script
* Export to CSV / PDF reports
* streamlit-webrtc continuous stream component
* Dockerfile for reproducible deployment

---
## 🪪 License
Add a LICENSE file if distributing publicly.

---
## 🙌 Acknowledgements
* PyTorch & TorchVision
* Streamlit
* OpenCV community

---
Feel free to open issues / requests for additional features.
