# 💧 Water Quality pH Prediction Using Machine Learning

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic-yellow.svg)]()

## 📊 Project Poster

<div align="center">
  <img src="Poster_ML.png" alt="Water Quality pH Prediction - Bayesian Ridge vs LSTM Comparison" width="900">
</div>

<p align="center">
  <em>Simple Beats Complex: Bayesian Ridge vs. LSTM for Water pH Prediction</em>
</p>

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Author](#-author)
- [Acknowledgments](#-acknowledgments)
- [Future Improvements](#-future-improvements)
- [References](#-references)

## 🎯 Overview

This project implements an **end-to-end machine learning pipeline** for predicting water quality pH levels across 37 monitoring stations in Georgia, USA. The system achieves **~83% variance explanation (R²=0.8328)** in next-day pH prediction using a comparative study of two distinct approaches: **Bayesian Ridge Regression** and **LSTM Neural Networks**.

The project demonstrates best practices in:
- **Data Quality**: Comprehensive preprocessing, standardization, and missing value handling
- **Machine Learning**: Systematic hyperparameter optimization with cross-validation
- **Model Comparison**: Rigorous evaluation across training, validation, and test sets
- **Interpretability**: Feature importance analysis and uncertainty quantification

**Key Achievement**: The Bayesian Ridge model outperforms the LSTM by **21% in prediction accuracy** while being **60x faster** and **810x simpler** (11 vs 8,913 parameters).

## ✨ Features

### 📊 **Data Analysis & Visualization**
- Comprehensive exploratory data analysis with 6+ visualization types
- Temporal autocorrelation analysis for time-series validation
- Spatial analysis across water systems and monitoring stations
- Feature correlation matrix and distribution analysis

### 🧠 **Dual Machine Learning Approaches**
- **Bayesian Ridge Regression**: Linear model with uncertainty quantification
- **LSTM Neural Networks**: Deep learning for temporal pattern recognition
- Systematic hyperparameter optimization (grid search + early stopping)
- 5-fold cross-validation for robust model selection

### 📈 **Comprehensive Evaluation**
- Multiple performance metrics (RMSE, MAE, R²)
- Overfitting analysis (train-test gap assessment)
- Residual distribution analysis
- Feature importance ranking

### 🔍 **Interpretable Results**
- Clear coefficient-based feature importance (Bayesian)
- Uncertainty quantification through posterior distributions
- Business perspective analysis for operational deployment
- Regulatory compliance considerations

## 📦 Dataset

**Source**: [UCI Machine Learning Repository - Water Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Water+Quality)

**Citation**: Zhao, L. (2019). Water Quality Prediction. UCI Machine Learning Repository.

### Structure

```
Water Quality Dataset (UCI ML Repository)
├── Temporal: 705 daily measurements
├── Spatial: 37 monitoring stations
├── Features: 11 water quality indices
└── Target: pH levels (continuous)

Data Split:
├── Training: 423 days (60%)
└── Test: 282 days (40%)
```

### Statistics

| Property | Value |
|----------|-------|
| **Total Samples** | 705 days |
| **Training Samples** | 423 days |
| **Test Samples** | 282 days |
| **Features** | 11 water quality indices |
| **Stations** | 37 monitoring locations |
| **Water Systems** | 2 major systems (Atlanta + East Coast) |
| **File Format** | MATLAB (.mat) |
| **File Size** | ~1 MB |

### Features Description

| Category | Parameters | Units |
|----------|-----------|-------|
| **Dissolved Oxygen** | Maximum, Mean, Minimum | mg/L |
| **Water Temperature** | Maximum, Mean, Minimum | °C |
| **Specific Conductance** | Maximum, Mean, Minimum | µS/cm at 25°C |
| **Additional Indices** | 2 more parameters | Various |

**Target Variable**: pH Level (Safe Range: 6.5 - 8.5 EPA)

### Access the Dataset

The dataset is included in this repository as `water_dataset.mat` in the Assessment 2 folder.

**External Link**: [Download from UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Water+Quality)

### Preprocessing Steps

1. Load MATLAB dataset using `scipy.io.loadmat()`
2. Extract nested arrays for train/test splits
3. Reshape data to appropriate dimensions (2D for Bayesian, 3D for LSTM)
4. Standardization using `StandardScaler` (zero mean, unit variance)
5. Train/validation split for hyperparameter tuning
6. Missing value analysis and handling

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager
- Jupyter Notebook
- 4GB RAM minimum

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/SalemAlnaqbi/Water-Quality-pH-Prediction-ML.git
cd Water-Quality-pH-Prediction-ML
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import numpy, pandas, sklearn, tensorflow; print('All packages installed successfully!')"
```

5. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

Expected installation time: **~5-10 minutes** depending on internet speed.

## 💻 Usage

### Workflow

1. **Clone the repository**
```bash
git clone https://github.com/SalemAlnaqbi/Water-Quality-pH-Prediction-ML.git
cd Water-Quality-pH-Prediction-ML
```

2. **Launch Jupyter Notebook**
```bash
jupyter notebook Water_Quality_pH_Prediction.ipynb
```

3. **Run the analysis** (Execute cells sequentially)
   - **Project Overview**: Introduction and objectives (~1 minute)
   - **Data Loading & Exploration**: Dataset analysis and visualizations (~2 minutes)
   - **Bayesian Ridge Regression**: Model training and evaluation (~3 minutes)
   - **LSTM Neural Network**: Deep learning approach (~10 minutes)
   - **Model Comparison**: Performance analysis and recommendations (~2 minutes)

4. **View results**
   - Performance metrics tables (RMSE, MAE, R²)
   - Prediction vs actual scatter plots
   - Feature importance rankings
   - Residual analysis plots
   - Model comparison summary

5. **Export results**
   - `File → Download as → PDF` for presentation
   - Save predictions to CSV: `predictions.to_csv('predictions.csv')`
   - Export visualizations for reports

**Total Runtime**: ~15-20 minutes for complete analysis

### Quick Start Code

**Load Dataset:**
```python
import scipy.io

# Load MATLAB dataset
data = scipy.io.loadmat('water_dataset.mat')
X_train = data['X_train']
Y_train = data['Y_train']
X_test = data['X_test']
Y_test = data['Y_test']
```

**Train Bayesian Ridge:**
```python
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = BayesianRidge(alpha_init=1.0, lambda_init=1e-6)
model.fit(X_train_scaled, Y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(Y_test, y_pred, squared=False)
r2 = r2_score(Y_test, y_pred)

print(f"Test RMSE: {rmse:.6f}")
print(f"Test R²: {r2:.4f}")
```

**Build LSTM Model:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define architecture
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(7, 11)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train
history = model.fit(
    X_train_seq, Y_train_seq,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
```

## 📁 Project Structure

```
Water-Quality-pH-Prediction-ML/
│
├── README.md                                  # Project documentation
├── requirements.txt                           # Python dependencies
├── Water_Quality_pH_Prediction.ipynb          # Main analysis notebook ⭐
├── water_dataset.mat                          # UCI ML Repository dataset
├── Poster_ML.png                              # Project poster
│
├── docs/
│   ├── PROJECT_DESCRIPTION.md                 # Detailed technical report
│   ├── CLAUDE.md                              # Development guide
│   └── NOTEBOOK_CLEANING_SUMMARY.md           # Notebook preparation notes
│
├── diagrams/
│   ├── Bayesian_Workflow.png                  # Bayesian Ridge workflow
│   └── LSTM_Neural_Network_Workflow.png       # LSTM architecture diagram
│
└── outputs/
    ├── predictions.csv                        # Model predictions (generated)
    ├── performance_metrics.csv                # Evaluation metrics (generated)
    └── visualizations/                        # Plots and charts (generated)
```

### Key Files

| File | Description | Size |
|------|-------------|------|
| `Water_Quality_pH_Prediction.ipynb` | Complete analysis notebook | ~1.2 MB |
| `water_dataset.mat` | UCI ML Repository dataset | ~1.0 MB |
| `Poster_ML.png` | Project overview poster | 243 KB |
| `requirements.txt` | Python dependencies | 1 KB |

## 🏗️ Model Architecture

### Bayesian Ridge Regression

```
Architecture Flow:
────────────────────

Input Features (11)
        ↓
Standardization (StandardScaler)
        ↓
Linear Model: y = X·w + ε
        ↓
Bayesian Inference:
  ├── Prior: p(w|α) ~ N(0, α⁻¹I)
  └── Likelihood: p(ε|λ) ~ N(0, λ⁻¹I)
        ↓
Posterior Distribution
        ↓
pH Prediction + Uncertainty
```

**Model Statistics:**
- **Parameters**: 11 (one per feature)
- **Hyperparameters**: α_init=1.0, λ_init=1e-6
- **Training Method**: 5-fold cross-validation
- **Regularization**: L2 (Ridge)
- **Training Time**: ~5 seconds
- **Inference Time**: <1 ms

**Design Rationale:**
- Provides uncertainty quantification through posterior distributions
- Natural regularization prevents overfitting with limited data
- Computationally efficient for real-time deployment
- Interpretable coefficients for regulatory compliance

### LSTM Neural Network

```
Architecture Flow:
────────────────────

Input Sequence (7 days × 11 features)
        ↓
Standardization (StandardScaler)
        ↓
LSTM Layer 1 (64 units, return_sequences=True)
        ↓
Dropout (0.2)
        ↓
LSTM Layer 2 (32 units, return_sequences=False)
        ↓
Dropout (0.2)
        ↓
Dense Layer (1 unit)
        ↓
pH Prediction
```

**Model Statistics:**
- **Parameters**: 8,913 trainable
- **Sequence Length**: 7 days
- **LSTM Units**: 64 → 32 (two layers)
- **Dropout Rate**: 0.2
- **Optimizer**: Adam (lr=0.001)
- **Training Time**: ~5-10 minutes
- **Inference Time**: 10-20 ms

**Design Rationale:**
- Captures temporal dependencies through memory cells
- Two-layer architecture balances complexity and generalization
- Dropout regularization prevents overfitting
- Early stopping monitors validation loss (patience=15)

## 📊 Results

### Performance Comparison

#### Test Set Performance (Final Evaluation)

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Parameters | Training Time |
|-------|--------|-------|------|------------|---------------|
| **Bayesian Ridge** ⭐ | **0.012036** | **0.007227** | **0.8328** | 11 | ~5 sec |
| LSTM | 0.014342 | 0.008572 | 0.7615 | 8,913 | ~5 min |
| **Improvement** | **21% better** | **16% better** | **9% better** | **810× simpler** | **60× faster** |

#### Training & Validation Performance

| Dataset | Model | RMSE | MAE | R² |
|---------|-------|------|-----|-----|
| **Training** | Bayesian Ridge | 0.012042 | 0.007385 | 0.8313 |
| | LSTM | 0.013123 | 0.008126 | 0.8007 |
| **Validation** | Bayesian Ridge | 0.010780 | 0.008865 | 0.8591 |
| | LSTM | 0.012995 | 0.008164 | 0.7997 |
| **Test** | Bayesian Ridge | **0.012036** | **0.007227** | **0.8328** |
| | LSTM | 0.014342 | 0.008572 | 0.7615 |

### Overfitting Analysis

| Model | Train RMSE | Test RMSE | Gap | Status |
|-------|-----------|-----------|-----|--------|
| **Bayesian Ridge** | 0.012042 | 0.012036 | **0.000006** | ✅ Excellent Generalization |
| LSTM | 0.013123 | 0.014342 | **0.001219** | ⚠️ Overfitting Detected |

**Key Finding**: Bayesian Ridge shows virtually no overfitting (gap: 0.000006), while LSTM degrades on unseen data by 203× more.

### Feature Importance (Bayesian Ridge)

| Rank | Feature | Coefficient | Impact on pH |
|------|---------|-------------|--------------|
| 1️⃣ | Dissolved Oxygen (Maximum) | +0.0286 | Strong positive correlation |
| 2️⃣ | Dissolved Oxygen (Mean) | -0.0274 | Inverse relationship |
| 3️⃣ | Specific Conductance (Mean) | +0.0242 | Moderate positive correlation |
| 4️⃣ | Temperature (Maximum) | +0.0218 | Warmer water → Higher pH |
| 5️⃣ | Conductance (Minimum) | -0.0196 | Inverse relationship |

**Interpretation**: Dissolved oxygen levels are the strongest predictors of pH, followed by conductance and temperature.

### Residual Statistics (Test Set)

| Metric | Bayesian Ridge | LSTM | Better Model |
|--------|---------------|------|--------------|
| **Mean** | 0.000272 | 0.001398 | Bayesian (closer to 0) |
| **Std Dev** | 0.012033 | 0.014274 | Bayesian (lower variance) |
| **Min Error** | -0.068677 | -0.072513 | Bayesian (smaller magnitude) |
| **Max Error** | 0.335277 | 0.355078 | Bayesian (smaller magnitude) |

### Training History

**Bayesian Ridge:**
- Grid search tested 16 hyperparameter combinations
- Best configuration: α=1.0, λ=1e-6
- 5-fold CV-RMSE: 0.012356
- Multiple configurations achieved similar performance (robust model)

**LSTM:**
- Tested 3 architectures (Small: 32-16, Medium: 64-32, Large: 96-48)
- Best configuration: Medium (64-32) with dropout=0.2
- Early stopping triggered around epoch 30-40
- Validation loss plateaued before overfitting

## 👨‍💻 Author

**Salem Alnaqbi**

- GitHub: [@SalemAlnaqbi](https://github.com/SalemAlnaqbi)
- LinkedIn: [salemalnaqbi](https://www.linkedin.com/in/salemalnaqbi/)
- Email: Student ID 201914118
- Program: MSc Artificial Intelligence, University of Leeds

## 🙏 Acknowledgments

### Dataset
- **Zhao, L. (2019)** - Water Quality Prediction Dataset, UCI Machine Learning Repository
- **UCI ML Repository** - Dataset hosting and documentation

### Frameworks & Libraries
- **scikit-learn** - Bayesian Ridge implementation and preprocessing tools
- **TensorFlow/Keras** - LSTM neural network framework
- **NumPy & Pandas** - Data manipulation and numerical computing
- **Matplotlib & Seaborn** - Visualization libraries

### Course & Institution
- **University of Leeds** - OCOM5200M Machine Learning course
- **School of Computing** - MSc Artificial Intelligence program
- **Teaching Staff** - Course instruction and guidance

### References
- **Bishop, C.M.** - Pattern Recognition and Machine Learning
- **Goodfellow et al.** - Deep Learning textbook
- **EPA** - Water quality standards and guidelines

## 🔮 Future Improvements

- [ ] **Spatial Modeling**: Incorporate station proximity and water system connectivity using Graph Neural Networks
- [ ] **Feature Engineering**: Add seasonal/cyclical features (day of year, month, day of week)
- [ ] **Ensemble Methods**: Combine Bayesian Ridge and LSTM predictions using weighted averaging or stacking
- [ ] **Online Learning**: Implement incremental learning for continuous model updates with new data
- [ ] **Multi-Output Prediction**: Forecast all 11 water parameters simultaneously
- [ ] **Bayesian Neural Networks**: Combine uncertainty quantification with deep learning
- [ ] **Anomaly Detection**: Develop outlier detection system for abnormal pH events
- [ ] **Causal Analysis**: Investigate treatment interventions and their effects on pH
- [ ] **Mobile Application**: Deploy model on mobile devices for field monitoring
- [ ] **Real-Time Dashboard**: Create Streamlit/Dash web interface for live predictions
- [ ] **Extended Dataset**: Gather 5+ years of data to support more complex temporal models
- [ ] **Meteorological Features**: Include rainfall, air temperature, and humidity data

## 📚 References

1. Zhao, L. (2019). Water Quality Prediction. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/Water+Quality](https://archive.ics.uci.edu/ml/datasets/Water+Quality)

2. University of Leeds (2025). OCOM5200M Machine Learning: Lecture Notes and Learning Materials. Leeds: University of Leeds, School of Computing.

3. EPA (2023). Water Quality Standards. U.S. Environmental Protection Agency.

4. Bishop, C.M. (2006). Pattern Recognition and Machine Learning. Springer.

5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

6. MacKay, D.J.C. (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press.

---

<div align="center">

**Built with ❤️ for water quality management and environmental protection**

⭐ Star this repository if you found it helpful!

</div>
