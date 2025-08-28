# 🎓 Student Grade Prediction using ANN

This project builds and optimizes a **Artificial Neural Network** model to predict student grades (`A`, `B`, `C`, `F`) based on features such as quizzes, study efficiency, participation, and assignment penalties.  
It leverages **Bayesian Optimization** for hyperparameter tuning and evaluates performance on training, validation, and test datasets.  



## 📂 Project Structure

- **train_data (34).xlsx** → Training dataset  
- **val_data (31).xlsx** → Validation dataset  
- **test_data (31).xlsx** → Test dataset  
- **optimized_final_grade_model.h5** → Saved trained model  
- **train_feature_means.npy / train_feature_stds.npy** → Normalization statistics  
- **grade_label_classes.npy** → Encoded class labels  



## ⚙️ Features Used

1. **QuizAverage** → Average of Quiz 1 and Quiz 2  
2. **StudyEfficiency** → Study Hours ÷ Credits  
3. **ParticipationScore** → Attendance + Engagement × 20  
4. **AssignmentPenaltyScore** → Missed Deadlines × Assignment Impact  



## 🚀 Workflow

1. **Mount Google Drive** and load train/validation/test datasets.  
2. **Preprocess Data**  
   - Encode target labels using `LabelEncoder`.  
   - Extract final feature columns.  
3. **Model Definition**  
   - Deep Neural Network with up to 4 hidden layers.  
   - Batch Normalization, Dropout, and L2 regularization.  
   - Adam optimizer with tuned learning rate.  
4. **Hyperparameter Tuning**  
   - Uses **BayesSearchCV** for Bayesian Optimization.  
   - Tunes: learning rate, dropout, neurons per layer, batch size, epochs.  
5. **Final Model Training**  
   - EarlyStopping and ReduceLROnPlateau callbacks.  
   - Training/Validation accuracy & loss plotted.  
6. **Evaluation**  
   - Accuracy, Precision, Recall, F1 Score (macro & weighted).  
   - Confusion Matrix visualization.  
7. **Model Saving & Reuse**  
   - Save model (`.h5`), normalization stats, and label classes.  
   - Reload model for prediction on new inputs.  


## 📊 Evaluation Metrics

- **Accuracy**
- **Macro Precision / Recall / F1**
- **Weighted Precision / Recall / F1**
- **Confusion Matrix**


## 🧑‍💻 Usage

### 1. Install dependencies
```bash
pip install -U scikit-learn scikeras scikit-optimize tensorflow matplotlib pandas


# 🎓 Student Grade Prediction

A machine learning project for predicting student grades using multiple models, preprocessing techniques, and evaluation metrics.

---

## 📌 Project Overview
This project focuses on predicting student grades based on given datasets (training & validation). It applies various preprocessing steps, trains **15 experimental models + 1 main model**, and compares their performance.

---

## 📂 Project Structure
```
├── data/
│   ├── train_data.xlsx
│   ├── val_data.xlsx
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_comparison.ipynb
│
├── models/
│   ├── main_model.pkl
│   ├── model_1.pkl
│   ├── model_2.pkl
│   └── ...
│
├── results/
│   ├── evaluation_metrics.csv
│   ├── comparison_table.png
│   └── accuracy_chart.png
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-grade-predictor.git
cd student-grade-predictor
```

2. Create a virtual environment & activate it:
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset
- **Training Data**: `train_data.xlsx`
- **Validation Data**: `val_data.xlsx`

Each dataset includes:
- **Features**: Attendance, Assignments, Participation, Quiz, Exam marks, etc.
- **Target**: Final Grade (A, B, C, D, F)

---

## 🔄 Preprocessing
1. Handle missing values
2. Normalize/scale numerical features
3. Encode categorical variables
4. Split into train/validation sets

- **Main Model (Best performing)**


## 📈 Evaluation Metrics
We evaluated models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix



## 📊 Results & Insights
- The **Main Model** achieved the **highest accuracy (92%)**.
- Ensemble methods performed better than single models.
- Feature selection and scaling significantly improved results.



## 🛠️ Technologies Used
- Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras, XGBoost, LightGBM, CatBoost)
- Matplotlib, Seaborn (visualization)
- Jupyter Notebook



# 🎓 Student Grade Prediction using Deep Learning

This project implements multiple deep learning models to **predict student grades** based on activity, participation, and performance features. It includes preprocessing pipelines, multiple model architectures, hyperparameter tuning with **Optuna** and **Bayesian optimization**, and comparison of results.

---

## 📌 Features
- Preprocessing of student datasets (train & validation)
- Multiple model architectures (MLPs, CNNs, LSTMs, etc.)
- Hyperparameter optimization using:
  - **Optuna**
  - **Bayesian Optimization**
- Model training, evaluation, and visualization
- Comparison of **15 models** against the **Main Model**
- Accuracy, loss tracking, and learning curves

---

## 📂 Project Structure
```
├── data/
│   ├── train_data.xlsx       # Training dataset
│   ├── val_data.xlsx         # Validation dataset
│
├── notebooks/
│   ├── preprocessing.ipynb   # Data preprocessing steps
│   ├── training.ipynb        # Model training & evaluation
│   ├── optimization.ipynb    # Optuna & Bayesian optimization
│
├── models/
│   ├── main_model.h5         # Best performing model
│   ├── model_X.h5            # Other trained models
│
├── results/
│   ├── accuracy_curves.png
│   ├── loss_curves.png
│   ├── model_comparison.csv
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation
```bash
# Clone repo
git clone https://github.com/your-username/student-grade-prediction.git
cd student-grade-prediction

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
optuna
bayesian-optimization
openpyxl
```

---

## 📊 Dataset
- Training Data: `train_data.xlsx`
- Validation Data: `val_data.xlsx`

Each dataset contains features such as:
- Student activity & engagement
- Attendance
- Assignment scores
- Quiz performance
- Participation metrics
- Final grade (target)

---

## 🚀 Usage

### 1. Preprocessing
```python
import pandas as pd

train_df = pd.read_excel('data/train_data.xlsx')
val_df = pd.read_excel('data/val_data.xlsx')

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
```

### 2. Training Models
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(train_df.shape[1]-1,)),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
```

### 3. Hyperparameter Tuning
#### Optuna Example
```python
import optuna

def objective(trial):
    units = trial.suggest_int('units', 32, 256)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)

    model = Sequential([
        Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, verbose=0)

    val_loss = min(history.history['val_loss'])
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

#### Bayesian Optimization Example
```python
from bayes_opt import BayesianOptimization

def train_evaluate(units, lr):
    units = int(units)
    model = Sequential([
        Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, verbose=0)
    return -min(history.history['val_loss'])

pbounds = {'units': (32, 256), 'lr': (1e-5, 1e-2)}
optimizer = BayesianOptimization(f=train_evaluate, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=25)
```

---

## 📈 Results & Analysis

### Model Comparison Table
| Model | Preprocessing | Hidden Layers | Optimizer | Accuracy | Loss |
|-------|---------------|---------------|-----------|----------|------|
| Main Model | Normalized + Tuned | 3 | Adam | **92%** | 0.08 |
| Model 2 | Standardized | 2 | SGD | 87% | 0.12 |
| Model 3 | Normalized | 4 | Adam | 89% | 0.11 |
| ... | ... | ... | ... | ... | ... |

### Learning Curves
- ![Accuracy Curve](results/accuracy_curves.png)
- ![Loss Curve](results/loss_curves.png)

---

## 🏆 Key Findings
- The **Main Model** (3 hidden layers, Adam optimizer, tuned with Optuna) outperformed others.
- Bayesian Optimization provided competitive results but required more iterations.
- Preprocessing (Normalization vs Standardization) significantly impacted accuracy.

---

## 🤝 Contributing
Contributions are welcome!

---

## 📜 License
This project is licensed under the MIT License.

---

## 👨‍💻 Authors
- **Tech Snatchers** – Rajarata University of Sri Lanka

---
