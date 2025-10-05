
# 🥦 Vegetable Quality Prediction Project

## 📌 Project Overview
This project focuses on building a **machine learning system** to classify and predict the quality/readiness of vegetables using **image datasets** and training models. The system supports both:
- **Manual image upload** for prediction.
- **Live image capture** from the camera for real-time prediction.

---

## ⚙️ Tools & Technologies
- **Python 3.12 / 3.13**
- **Jupyter Notebook** (exploration, preprocessing, model training)
- **VS Code** (modular project development, packaging, integration)
- **PyTorch** (model building & training)
- **OpenCV / Pillow** (image preprocessing & capture)
- **Matplotlib & Seaborn** (visualization)
- **Pandas & NumPy** (data handling)
- **Flask** (for dashboard / web integration)
- **Git & GitHub** (version control & submission)

---

## 🧑‍💻 Workflow & Steps

### 1. Data Preparation
- Organized dataset into `train`, `valid`, and `test` folders.
- Each folder contains **Excel files** with class labels and metadata.
- Created `_classes.csv` for mapping images to target classes.

### 2. Exploratory Data Analysis (Jupyter Notebook)
- Checked dataset balance.
- Visualized sample images for each class.
- Verified class distribution and labeling.

### 3. Preprocessing
- Applied **image resizing, normalization, and augmentation**.
- Converted categorical labels into numerical format.
- Split data into training, validation, and testing sets.

### 4. Model Development (PyTorch)
- Built a **CNN model** with convolutional, pooling, and fully connected layers.
- Defined **loss function (CrossEntropyLoss)** and **optimizers (Adam/SGD)**.
- Experimented with hyperparameters: learning rate, batch size, epochs.

### 5. Training & Evaluation
- Trained the model on the `train` dataset.
- Monitored **loss** and **accuracy** across epochs.
- Observed:  
  - **Loss decreasing** → Model learning correctly.  
  - **Accuracy increasing** → Predictions improving.
- Evaluated on `valid` and `test` sets using:
  - Accuracy, Precision, Recall, F1-score.
  - Confusion matrix & misclassified samples.

### 6. System Integration (VS Code)
- Created a **modular project structure**:
  - `src/train_banana.py` → training script.
  - `src/pipeline.py` → data pipeline & preprocessing.
  - `dashboard/app.py` → Flask app for UI & database insertion.
  - `db_utils.py` → helper for database operations.
- Added **image upload & live camera capture prediction**.
- Linked outputs to **database** (SQLite/PostgreSQL option).

### 7. Error Handling & Debugging
- Resolved `ModuleNotFoundError: torch` by installing PyTorch.
- Fixed `AttributeError` in `db_utils` by implementing missing functions.
- Adjusted dataset paths (`PROJECT/data/train/...`).
- Verified working predictions on test images.

---

## 📊 Results
- Achieved good classification accuracy after tuning.
- Model successfully predicts vegetable readiness/quality from images.
- Supports **both upload & live camera prediction**.
- Dashboard integrated with prediction + storage pipeline.

---

## 📂 Repository Structure
```
vegetable-quality-prediction/
│
├── data/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── _classes.csv
│
├── src/
│   ├── train_banana.py
│   ├── pipeline.py
│   └── db_utils.py
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│   └── eda_training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vegetable-quality-prediction.git
   cd vegetable-quality-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python src/train_banana.py
   ```
4. Run the dashboard:
   ```bash
   python dashboard/app.py
   ```
5. Access in browser:
   ```
   http://127.0.0.1:5000
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
