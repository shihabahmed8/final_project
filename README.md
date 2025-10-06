
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
### 1. Create Environment & Install Dependencies
python -m venv venv
venv\Scripts\activate        # On Windows

pip install -r requirements.txt
pip install streamlit

### 2. Data Preparation
- Organized dataset into `train`, `valid`, and `test` folders.
- Each folder contains **Excel files** with class labels and metadata.
- Created `_classes.csv` for mapping images to target classes.

### 3. Exploratory Data Analysis (Jupyter Notebook)
- Checked dataset balance.
- Visualized sample images for each class.
- Verified class distribution and labeling.

### 4. Preprocessing
- Applied **image resizing, normalization, and augmentation**.
- Converted categorical labels into numerical format.
- Split data into training, validation, and testing sets.

### 5. Model Development (PyTorch)
- Built a **CNN model** with convolutional, pooling, and fully connected layers.
- Defined **loss function (CrossEntropyLoss)** and **optimizers (Adam/SGD)**.
- Experimented with hyperparameters: learning rate, batch size, epochs.

### 6. Training & Evaluation
- Trained the model on the `train` dataset.
- Monitored **loss** and **accuracy** across epochs.
- Observed:  
  - **Loss decreasing** → Model learning correctly.  
  - **Accuracy increasing** → Predictions improving.
- Evaluated on `valid` and `test` sets using:
  - Accuracy, Precision, Recall, F1-score.
  - Confusion matrix & misclassified samples.

### 7. System Integration (VS Code)
- Created a **modular project structure**:
  - `src/train_banana.py` → training script.
  - `src/pipeline.py` → data pipeline & preprocessing.
  - `dashboard/app.py` → Flask app for UI & database insertion.
  - `db_utils.py` → helper for database operations.
- Added **image upload & live camera capture prediction**.
- Linked outputs to **database** (SQLite/PostgreSQL option).

### 8. Error Handling & Debugging
- Resolved `ModuleNotFoundError: torch` by installing PyTorch.
- Fixed `AttributeError` in `db_utils` by implementing missing functions.
- Adjusted dataset paths (`PROJECT/data/train/...`).
- Verified working predictions on test images.
- 
### 9. Run Dashboard (Streamlit)
Launch the interactive dashboard:
streamlit run dashboard/app.py

Open your browser at 👉 http://localhost:8501

The dashboard allows you to:

Upload fruit images

View predictions (class + confidence)

See shelf-life estimation

Get optimal storage suggestions


---

## 📊 Results
- Achieved good classification accuracy after tuning with

<img width="404" height="31" alt="Screenshot 2025-10-06 090545" src="https://github.com/user-attachments/assets/7358ef6f-d205-458a-81c0-aa4b0ac9f81d" />

- Model successfully predicts vegetable readiness/quality from images.
- Supports ** upload images prediction**.
- Dashboard integrated with prediction + storage pipeline.

---

## 📂 Project Structure
```
produce_quality_ready/
│
├── data/                 # Training, validation, and test data (CSV + images)
│   ├── train/
│   ├── valid/
│   └── test/
│
├── src/                  # Source code
│   ├── train_banana.py   # Training script
│   ├── predict.py        # Prediction script
│   └── shelf_life.py     # Shelf-life estimation function
│
├── dashboard/            
│   └── app.py            # Streamlit dashboard
│
├── weights/              # Saved model weights
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

```

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/shihabahmed8/final_project.git
   cd produce_quality_ready
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install streamlit
   ```
3. Run Predictions with Pretrained Model:
   Run:
   ```bash
  python src/predict.py --image sample.jpg --fruit banana --temp 28
   ```
4. Run the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
5. Access in browser:
   ```
   http://localhost:8501/
   ```

---

## ✅ Completed Tasks
- [x] Dataset preparation & cleaning  
- [x] EDA & visualization  
- [x] Preprocessing pipeline  
- [x] CNN model training & evaluation  
- [x] Upload prediction feature  
- [x] Live camera prediction feature  
- [x] Flask dashboard integration  
- [x] Database storage integration  
