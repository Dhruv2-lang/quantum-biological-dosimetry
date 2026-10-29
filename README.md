# ⚛️ Quantum Biological Dosimetry: Classical ML vs. Variational Quantum Classifier (VQC)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.45.1-purple.svg)](https://pennylane.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47.1-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Quantum Computing framework designed to predict radiation-induced cellular and genetic damage directly from non-invasive **Nitrogen-Vacancy (NV) diamond** and **Quantum Dot (QD)** physical sensor measurements.

---

## 📌 Executive Summary & Key Results

- **Research Objective:** Evaluate whether a trainable **Variational Quantum Classifier (VQC)** offers an empirical advantage over Classical Machine Learning (Logistic Regression, Random Forest, XGBoost) when classifying radiation toxicity and predicting continuous biological damage scores from quantum sensor telemetry.
- **Dataset:** 5,000 real records (`Quantum_Biological_Dosimetry_Dataset_5000.csv`) with zero synthetic replacement.
- **Strict Target Leakage Safeguard:** Downstream destructive cellular assays (`ROS_Level`, `Mitochondrial_Damage`, `GammaH2AX_Foci`, `BP53_Foci`, `Cell_Viability`, `DNA_Damage_Score`) are strictly quarantined from input feature sets. Classification is performed **exclusively** on 7 physical sensor measurements.
- **Empirical Leaderboard (1,000 Unseen Test Samples):**

| Model | Type | Accuracy | Macro-F1 | Precision | Recall | Target Hardware / Architecture |
|---|---|---|---|---|---|---|
| **Logistic Regression** | Classical ML | **96.10%** | **0.9614** | **0.9627** | **0.9603** | CPU (Multinomial Linear Logits) |
| **XGBoost** | Classical Ensemble | **95.90%** | **0.9594** | **0.9597** | **0.9591** | CPU (100 Gradient Boosted Trees) |
| **Random Forest** | Classical Ensemble | **95.80%** | **0.9586** | **0.9590** | **0.9582** | CPU (100 Decision Trees) |
| **PennyLane 4-Qubit VQC** | Quantum Neural Net | **67.70%** | **0.5601** | **0.6430** | **0.6110** | Quantum Simulator (`default.qubit`, 24 params) |

- **Secondary Regression Task (`DNA_Damage_Score`):**
  - **Ridge Regression:** RMSE = 1.4472, $R^2 = 0.9940$
  - **Random Forest Regressor:** RMSE = 1.5251, $R^2 = 0.9933$

---

## 🔬 Scientific Integrity & Scope Clarifications

1. **Absence of Quantum Advantage:**
   Classical tree-based models and regularized linear classifiers dramatically outperform the 4-qubit parameterized quantum circuit on this low-dimensional tabular dataset (~96% vs 67.7%). This confirms established theoretical expectations: NISQ-era shallow variational circuits do not exhibit quantum supremacy for structured tabular biosensing data.
2. **Tabular Telemetry vs. Problem 7 ("Quantum-Centric Image Processing"):**
   This dataset consists exclusively of **scalar tabular spectroscopic and relaxometry readings**, containing no spatial pixel arrays, histological micrographs, or image frames. No image processing has been implemented. Any genuine image-processing dosimetry pipeline would require spatial fluorescing images processed through Quantum Convolutional Neural Networks (QCNNs).
3. **Preclinical Validation:**
   This project models nanophotonic laboratory sensor responses; no claims of human clinical validity or diagnostic clearance are made.

---

## 🗂️ Project Architecture

```
quantum_biodosimetry/
├── app/
│   └── app.py                     # Interactive 5-tab Streamlit dashboard
├── data/
│   └── Quantum_Biological_Dosimetry_Dataset_5000.csv # Verified 5,000-row dataset
├── models/
│   ├── preprocessor_bundle.joblib # Scaler, PCA-4, LabelEncoder
│   ├── data_splits.joblib         # Train/test split matrices
│   ├── logistic_regression.joblib # Trained Logistic Regression
│   ├── random_forest.joblib       # Trained Random Forest
│   ├── xgboost.joblib             # Trained XGBoost Classifier
│   ├── ridge_regression.joblib    # Trained Ridge Regressor
│   ├── random_forest_regressor.joblib # Trained Random Forest Regressor
│   ├── qnn_weights.pt             # PyTorch state dict for VQC
│   └── qnn_vqc.joblib             # Serializable PennyLane VQC checkpoint
├── reports/
│   ├── eda_summary.json           # Missing values, class balance, leakage analysis
│   ├── classical_metrics.json     # Full metrics for classical models
│   ├── quantum_metrics.json       # VQC training history & test performance
│   ├── benchmark_summary.json     # Consolidated head-to-head comparison
│   └── figures/                   # High-resolution evaluation plots (300 DPI)
│       ├── eda_correlation_matrix.png
│       ├── eda_leakage_audit.png
│       ├── eda_sensor_distributions.png
│       ├── classical_confusion_matrices.png
│       ├── regression_evaluation.png
│       ├── qnn_training_curve.png
│       ├── qnn_confusion_matrix.png
│       ├── model_comparison_benchmark.png
│       └── all_confusion_matrices_combined.png
├── src/
│   ├── eda.py                     # Data inspection & target leakage audit
│   ├── preprocessing.py           # Strict leakage safeguard, scaling, PCA-4
│   ├── models_classical.py        # Classical training & evaluation
│   ├── models_quantum.py          # PennyLane VQC circuit, training & inference
│   └── compare_models.py          # Consolidated benchmark generator
├── tests/
│   ├── test_pipeline.py           # Unit tests for data & preprocessing
│   └── test_models.py             # Unit tests for model inference & artifacts
├── PROJECT_STATUS.md              # Resumption checkpoint and milestone ledger
├── RESEARCH_REPORT.md             # Formal scientific manuscript
├── requirements.txt               # Pinned dependencies
└── quantum_biodosimetry.zip       # Full export archive
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup
```bash
# Clone or navigate to the workspace
cd /Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis & Leakage Audit
```bash
python3 src/eda.py
```

### 3. Run Preprocessing Pipeline
```bash
python3 src/preprocessing.py
```

### 4. Train Classical & Quantum Models
```bash
# Train classical classifiers and regressors
python3 src/models_classical.py

# Train PennyLane Variational Quantum Classifier (VQC)
python3 src/models_quantum.py

# Generate comparison plots & benchmark JSON
python3 src/compare_models.py
```

### 5. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app/app.py
```
The dashboard provides 5 comprehensive modules:
- **Dataset & Sensor EDA:** Interactive exploration of physical features, correlations, and leakage audit.
- **Single Prediction:** Interactive sliders with clinical preset profiles to query both classical and quantum models in real time.
- **Batch CSV Prediction:** Bulk inference on uploaded sensor files with downloadable CSV reports.
- **Benchmark Leaderboard:** Head-to-head metrics, confusion matrices, and VQC training convergence curves.
- **Sensor Physics & Disambiguation:** Detailed scientific background on ODMR, $T_1/T_2$ relaxometry, and Problem 7 disambiguation.

### 6. Run the Test Suite
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## ⚛️ Quantum Circuit Architecture

The PennyLane Variational Quantum Classifier is parameterized as follows:
- **Quantum Simulator:** PennyLane `default.qubit` (4 wires).
- **State Preparation:** $Y$-rotation Angle Embedding $\mathcal{U}_{\text{emb}}(\mathbf{x}) = \bigotimes_{i=0}^3 R_y(x_i)$, mapping the top 4 principal components of the standardized 7-dimensional sensor vector into single-qubit rotations.
- **Variational Ansatz:** 2 layers of `StronglyEntanglingLayers` featuring general $SU(2)$ rotations ($R_x, R_y, R_z$) on each qubit followed by cyclic entangling CNOT gates. Total trainable quantum parameters: $2 \times 4 \times 3 = 24$.
- **Hamiltonian Measurement:** Pauli-Z observables measured on wires 0, 1, and 2:
  $$\mathbf{z} = \begin{bmatrix} \langle \psi | Z_0 | \psi \rangle \\ \langle \psi | Z_1 | \psi \rangle \\ \langle \psi | Z_2 | \psi \rangle \end{bmatrix} \in [-1, 1]^3$$
- **Hybrid Logit Transformation:** $\mathbf{y}_{\text{logits}} = \mathbf{w}_{\text{scale}} \odot \mathbf{z} + \mathbf{b}_{\text{bias}}$.
- **Optimization:** PyTorch backend with exact analytic quantum backpropagation (`diff_method="backprop"`), optimized using Adam ($\text{lr}=0.03$, 45 epochs).

---

## 🛡️ Target Leakage Prevention Protocol

A critical methodological flaw in naive biodosimetry ML is the inclusion of downstream biological endpoint assays. As demonstrated in our audit:
$$\text{Radiotoxicity\_Class} = \begin{cases} \text{Low} & \text{if } \text{DNA\_Damage\_Score} < 25.0 \\ \text{Moderate} & \text{if } 25.0 \le \text{DNA\_Damage\_Score} < 50.0 \\ \text{High} & \text{if } \text{DNA\_Damage\_Score} \ge 50.0 \end{cases}$$

Including `DNA_Damage_Score`, `ROS_Level`, `Mitochondrial_Damage`, or foci counts (`GammaH2AX_Foci`, `BP53_Foci`) introduces artificial 100% target leakage. In this project, all input matrices are strictly restricted to the non-invasive physical sensor telemetry:
1. `Dose_Gy` (Absorbed ionizing radiation dose)
2. `ODMR_Shift_MHz` (Diamond NV center Zeeman splitting shift)
3. `T1_us` (Longitudinal spin-lattice relaxation time)
4. `T2_us` (Transverse spin-spin relaxation time)
5. `Fluorescence_Intensity` (Nanodiamond photoluminescence count)
6. `Fluorescence_Lifetime_ns` (Photoluminescence decay lifetime)
7. `QD_Emission_nm` (Colloidal quantum dot emission spectral peak)

---

## 📄 License
This project is open-source under the MIT License.
