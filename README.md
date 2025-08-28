# 🎓 Student Grade Prediction using Artificial Neural Networks

This project develops and optimizes an **Artificial Neural Network (ANN)** to predict student grades (`A`, `B`, `C`, `F`) using academic and engagement features.  
It integrates **Bayesian Optimization** for hyperparameter tuning, applies preprocessing (label encoding + normalization), and evaluates performance on training, validation, and test datasets.

---

## 📂 Project Structure
```
├── data/
│   ├── train_data (34).xlsx      # Training dataset
│   ├── val_data (31).xlsx        # Validation dataset
│   ├── test_data (31).xlsx       # Test dataset
│
├── notebooks/
│   ├── 8_21_morning_dataset_Model_training.ipynb   # Main Colab notebook
│
├── models/
│   ├── optimized_final_grade_model.h5              # Trained ANN model
│   ├── train_feature_means.npy                     # Saved normalization means
│   ├── train_feature_stds.npy                      # Saved normalization stds
│   ├── grade_label_classes.npy                     # Saved label encoder classes
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│
└── README.md
```

---

## ⚙️ Features Used
1. **QuizAverage** → Mean of Quiz 1 & Quiz 2  
2. **StudyEfficiency** → Study Hours ÷ Credits  
3. **ParticipationScore** → Attendance + Engagement × 20  
4. **AssignmentPenaltyScore** → Missed Deadlines × Assignment Impact  

---

## 🚀 Workflow

1. **Load Datasets** → Train (704 rows), Validation (151 rows), Test (31 rows)  
2. **Preprocessing**  
   - Feature selection (`QuizAverage`, `StudyEfficiency`, etc.)  
   - Encode labels with `LabelEncoder`  
   - Save normalization stats (mean/std)  
3. **Model Definition**  
   - ANN with up to **4 hidden layers**  
   - Batch Normalization + Dropout  
   - L2 regularization  
4. **Hyperparameter Tuning**  
   - **BayesSearchCV** explores learning rate, dropout, neurons/layer, batch size, epochs  
   - Best configuration achieved ~**0.85 CV accuracy**  
5. **Final Model Training**  
   - Trained with best hyperparameters  
   - Used **EarlyStopping** & **ReduceLROnPlateau**  
   - Training history saved (accuracy/loss curves)  
6. **Evaluation**  
   - Training & Validation performance  
   - Test set evaluation with Accuracy, Precision, Recall, F1  
   - Confusion Matrix visualization  
7. **Model Saving**  
   - ANN saved as `.h5`  
   - Preprocessing statistics & label classes saved as `.npy`  

---

## 📊 Evaluation Metrics

| Metric                  | Validation | Test |
|--------------------------|------------|------|
| Accuracy                | ~0.85      | ~0.84 |
| Macro Precision / Recall / F1 | ✅ | ✅ |
| Weighted Precision / Recall / F1 | ✅ | ✅ |

✔️ Confusion Matrix shows strong prediction accuracy across classes, with slight imbalance between higher/lower grades.

---

## 🧑‍💻 Usage

### 1. Install dependencies
```bash
pip install -U scikit-learn scikeras scikit-optimize tensorflow matplotlib pandas openpyxl
```

### 2. Train model (Colab)
Open `8_21_morning_dataset_Model_training.ipynb` in **Google Colab**, mount Google Drive, and run cells to:
- Preprocess data  
- Perform hyperparameter optimization  
- Train final ANN  
- Evaluate and save model  

### 3. Predict on new data
```python
import numpy as np
import tensorflow as tf
import pandas as pd

# Load model and preprocessing stats
model = tf.keras.models.load_model("optimized_final_grade_model.h5")
means = np.load("train_feature_means.npy")
stds = np.load("train_feature_stds.npy")
classes = np.load("grade_label_classes.npy", allow_pickle=True)

# Example new input
X_new = pd.DataFrame([{
    "QuizAverage": 75,
    "StudyEfficiency": 2.1,
    "ParticipationScore": 80,
    "AssignmentPenaltyScore": 5
}])

# Normalize
X_new_norm = (X_new - means) / stds

# Predict
pred = np.argmax(model.predict(X_new_norm), axis=1)
print("Predicted Grade:", classes[pred][0])
```

---

## 📈 Results
- **Best ANN model** tuned with Bayesian Optimization achieved **~85% accuracy** on test data.  
- Feature engineering (StudyEfficiency, AssignmentPenaltyScore) boosted performance.  
- Normalization was critical for stable training.  

---

## 🛠️ Technologies
- **Python** (NumPy, Pandas, Matplotlib)  
- **Scikit-learn, Scikit-optimize, SciKeras**  
- **TensorFlow / Keras**  
- **Google Colab (GPU runtime)**  

---

## 👨‍💻 Authors
**Tech Snatchers** – Rajarata University of Sri Lanka  

---

## 📜 License
MIT License  
