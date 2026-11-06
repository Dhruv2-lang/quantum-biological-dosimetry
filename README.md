# ⚛️ Quantum Biological Dosimetry
### Classical Machine Learning, Variational Quantum Classifier (VQC), and Quantum Support Vector Machine (QSVM)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Quantum-Qiskit%202.4-blueviolet.svg)](https://qiskit.org/)
[![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-purple.svg)](https://pennylane.ai/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)

An experimental machine learning and quantum computing framework for studying radiation-associated biological damage using physical sensor measurements inspired by nitrogen-vacancy (NV) diamond sensing and quantum-dot spectroscopy.

The project benchmarks classical machine learning models (Logistic Regression, Random Forest, XGBoost, Classical RBF SVM) against two genuine quantum learning paradigms: a Variational Quantum Classifier (PennyLane VQC) and a Quantum Support Vector Machine (Qiskit QSVM with $ZZ$-feature map fidelity kernel). An interactive Streamlit dashboard is provided for real-time inference, batch CSV scoring, sensor exploratory data analysis, and benchmark evaluation.

> **Research disclaimer:** This is an experimental computational project, not a clinically validated diagnostic system. Dataset provenance, sensor assumptions, and model performance must be independently verified before drawing scientific or medical conclusions.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Dataset & Physics Features](#-dataset--physics-features)
- [Target Leakage Prevention](#-target-leakage-prevention)
- [Project Structure](#-project-structure)
- [Models and Methodology](#-models-and-methodology)
  - [1. Classical Classifiers (LogReg, Random Forest, XGBoost)](#1-classical-classifiers)
  - [2. Classical Support Vector Machine (RBF SVM)](#2-classical-support-vector-machine-rbf-svm)
  - [3. Variational Quantum Classifier (PennyLane VQC)](#3-variational-quantum-classifier-pennylane-vqc)
  - [4. Quantum Support Vector Machine (Qiskit QSVM)](#4-quantum-support-vector-machine-qiskit-qsvm)
  - [5. Damage Regression Models](#5-damage-regression-models)
- [Empirical Benchmark Results](#-empirical-benchmark-results)
- [Streamlit Interactive Dashboard](#-streamlit-interactive-dashboard)
- [Installation & Setup](#-installation--setup)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Limitations & Scientific Integrity](#-limitations--scientific-integrity)

---

## 🔬 Overview

Biological dosimetry investigates cellular and genetic responses associated with ionizing radiation exposure. This project demonstrates how non-invasive physical quantum sensor telemetry—specifically optically detected magnetic resonance (ODMR) frequency shifts and spin relaxometry ($T_1, T_2$) from nitrogen-vacancy diamond centers, coupled with quantum dot photoluminescence—can accurately classify radiation damage categories without destroying biological tissue.

Key architectural highlights:
- **Strict Leakage Quarantine:** Downstream biological assay endpoints (`ROS_Level`, `Mitochondrial_Damage`, `GammaH2AX_Foci`, `BP53_Foci`, `Cell_Viability`, `DNA_Damage_Score`) are strictly quarantined and never exposed to classification models.
- **Genuine Quantum Computing:** Implements both a variational quantum neural network (PennyLane VQC) and a quantum kernel machine (Qiskit QSVM) using statevector simulation.
- **Robust macOS ARM64 Hardening:** Single-threaded OpenMP pinning and headless Matplotlib Agg backends prevent runtime segmentation faults.

**Important distinction:** The dataset is tabular telemetry (7 continuous scalars). This implementation does not process microscopic images, pixel grids, or histological slides, and does not claim to solve Problem 7 ("Quantum-Centric Image Processing").

---

## 🎯 Objectives

1. Explore physical sensor telemetry correlations with radiation-induced damage.
2. Prevent direct target leakage between cellular assays and radiotoxicity classes.
3. Benchmark classical classifiers against a PennyLane Variational Quantum Classifier.
4. Integrate a genuine Qiskit Quantum Support Vector Machine (QSVM) using quantum state fidelity kernels.
5. Provide continuous estimation of DNA damage score using regression models ($R^2 > 0.99$).
6. Provide an interactive Streamlit application supporting single and batch inference.
7. Deliver automated unit tests and transparent, reproducible research documentation.

---

## 📊 Dataset & Physics Features

The project operates on `data/Quantum_Biological_Dosimetry_Dataset_5000.csv` containing 5,000 verified rows with zero missing values.

### Non-Invasive Sensor Telemetry Features (Model Inputs)

| Feature | Physical Principle & Unit | Role in Dosimetry |
|---|---|---|
| `Dose_Gy` | Absorbed radiation dose (Gray) | Primary exposure variable |
| `ODMR_Shift_MHz` | NV- center zero-field splitting shift ($\Delta D$ in MHz) | Probes local magnetic fluctuations and lattice strain |
| `T1_us` | Longitudinal spin-lattice relaxation time ($\mu\text{s}$) | Shortens sharply in presence of free paramagnetic radicals (ROS) |
| `T2_us` | Transverse spin-spin relaxation time ($\mu\text{s}$) | Measures spin dephasing driven by radical fluctuations |
| `Fluorescence_Intensity` | Emission intensity (arbitrary units) | Reflects quantum yield and optical quenching |
| `Fluorescence_Lifetime_ns` | Photoluminescence decay lifetime ($\text{ns}$) | Quenched by non-radiative energy transfer in acidified media |
| `QD_Emission_nm` | Quantum dot emission peak wavelength ($\text{nm}$) | Bathochromic shift caused by oxidative nanocrystal degradation |

### Quarantined Biological Assays (Excluded from Inputs)
- `DNA_Damage_Score` (Continuous ground-truth score, $0 - 68.26$)
- `Radiotoxicity_Class` (Primary target: Low $< 25$, Moderate $25 - 50$, High $\ge 50$)
- `ROS_Level`, `Mitochondrial_Damage`, `GammaH2AX_Foci`, `BP53_Foci`, `Cell_Viability`

---

## 🛡️ Target Leakage Prevention

`Radiotoxicity_Class` is a direct threshold partition of `DNA_Damage_Score`. Feeding `DNA_Damage_Score` or collinear downstream assays into a classifier would yield trivial 100% accuracy with zero real-world utility.

**Enforced Rule:** All classification models are strictly trained on the 7 non-invasive physical sensor features. The preprocessing pipeline enforces this quarantine in `src/preprocessing.py`, verified by automated tests in `tests/test_pipeline.py`.

---

## 📂 Project Structure

```text
quantum_biodosimetry/
├── app/
│   └── app.py                        # Hardened Streamlit interactive dashboard
├── data/
│   └── Quantum_Biological_Dosimetry_Dataset_5000.csv
├── models/
│   ├── preprocessor_bundle.joblib    # Fitted scaler, PCA (4 components), encoders
│   ├── data_splits.joblib            # Stratified 80/20 train/test data splits
│   ├── logistic_regression.joblib    # Trained Multinomial Logistic Regression
│   ├── random_forest.joblib          # Trained Random Forest Classifier
│   ├── xgboost.joblib                # Trained XGBoost Classifier
│   ├── svm_classifier.joblib         # Trained Classical RBF SVM (C=5.0, gamma=0.05)
│   ├── svm.joblib                    # Model alias
│   ├── qnn_vqc.joblib                # Trained PennyLane 4-Qubit VQC wrapper
│   ├── qnn_weights.pt                # PyTorch tensor weights for PennyLane VQC
│   ├── qsvm_classifier.joblib        # Trained Qiskit 4-Qubit QSVM (ZZFeatureMap)
│   ├── qsvm.joblib                   # Model alias
│   ├── qsvm_config.json              # Architecture metadata and parameters for QSVM
│   ├── ridge_regression.joblib       # Trained Ridge Regressor for DNA Damage
│   └── random_forest_regressor.joblib# Trained Random Forest Regressor
├── reports/
│   ├── eda_summary.json              # Statistical audit & correlation metrics
│   ├── classical_metrics.json        # Test evaluation for classical models
│   ├── quantum_metrics.json          # Test evaluation for PennyLane VQC
│   ├── svm_metrics.json              # Detailed test evaluation for Classical SVM
│   ├── qsvm_metrics.json             # Detailed test evaluation for Qiskit QSVM
│   ├── qsvm_config.json              # Quantum feature map & kernel configuration
│   ├── benchmark_summary.json        # Unified 6-model comparison leaderboard
│   └── figures/                      # 15 high-resolution diagnostic plots
├── src/
│   ├── eda.py                        # Dataset validation & sensor physics EDA
│   ├── preprocessing.py              # Preprocessing & leakage quarantine pipeline
│   ├── models_classical.py           # Training classical baselines & Classical SVM
│   ├── models_quantum.py             # Trainable PennyLane 4-qubit VQC
│   ├── models_qsvm.py                # Genuine Qiskit 4-qubit QSVM quantum kernel
│   └── compare_models.py             # Benchmark aggregation & plot generation
├── tests/
│   ├── test_pipeline.py              # Pipeline & leakage unit tests
│   └── test_models.py                # Model inference & quantum kernel unit tests
├── PROJECT_STATUS.md                 # Complete milestone tracker & diagnostics ledger
├── RESEARCH_REPORT.md                # Formal scientific paper & mathematical analysis
├── requirements.txt                  # Pinned dependencies
├── .gitignore                        # Git exclusion rules
└── README.md                         # Comprehensive documentation
```

---

## 🧠 Models and Methodology

### 1. Classical Classifiers
- **Logistic Regression:** Multinomial linear logit model optimized with L-BFGS.
- **Random Forest:** Ensemble of 100 decorrelated trees with maximum depth 12.
- **XGBoost:** Gradient-boosted tree model with learning rate $\eta = 0.1$, max depth 6.

### 2. Classical Support Vector Machine (RBF SVM)
- **Algorithm:** Scikit-learn `SVC` with Radial Basis Function kernel $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$.
- **Feature Pipeline:** 7-dimensional standardized features.
- **Hyperparameter Optimization:** 5-fold cross-validation grid search identified optimal parameters $C = 5.0$, $\gamma = 0.05$.
- **Class Balancing:** `class_weight='balanced'` to compensate for minority class proportions.

### 3. Variational Quantum Classifier (PennyLane VQC)
- **Qubit Allocation:** 4 simulated qubits (`default.qubit`).
- **Feature Encoding:** Angle embedding with $R_y(z_j)$ rotations using 4 PCA components (retaining 99.62% variance).
- **Variational Ansatz:** 2 layers of `StronglyEntanglingLayers` (24 trainable rotation parameters).
- **Readout:** Expectation values $\langle Z_0 \rangle, \langle Z_1 \rangle, \langle Z_2 \rangle$ mapped to class logits via linear bias layer.
- **Optimization:** Adam optimizer ($\text{lr} = 0.03$) over 45 epochs on $N=360$ stratified samples.

### 4. Quantum Support Vector Machine (Qiskit QSVM)
- **Quantum Feature Map:** Qiskit's second-order Pauli expansion `ZZFeatureMap` on 4 qubits with 1 repetition and linear entanglement:
  $$\mathcal{U}_{\Phi}(\mathbf{x}) = \exp\left(i \sum_{j=1}^4 \phi_j(\mathbf{x}) Z_j + i \sum_{j < k} \phi_{jk}(\mathbf{x}) Z_j Z_k\right) H^{\otimes 4}$$
  with phase scaling factor $s = 0.15$ to map PCA components smoothly into $[-\pi, \pi]$ without phase scrambling.
- **Quantum Kernel Evaluation:** Exact fidelity computation between quantum statevectors:
  $$K(\mathbf{x}_1, \mathbf{x}_2) = |\langle \Phi(\mathbf{x}_1) | \Phi(\mathbf{x}_2) \rangle|^2$$
  Mathematical verification confirms symmetry ($K = K^T$), unit diagonal ($K_{ii} = 1.0$), and positive semi-definiteness ($\lambda_i \ge 0$).
- **Dual Convex Classification:** Precomputed Gram matrix evaluated with Scikit-learn `SVC(kernel='precomputed', C=5.0, class_weight='balanced')` trained on $N = 360$ stratified samples.

### 5. Damage Regression Models
Continuous estimation of `DNA_Damage_Score` ($[0, 100]$):
- **Ridge Regression:** $L_2$-regularized linear model ($\alpha = 1.0$).
- **Random Forest Regressor:** 100 trees with max depth 12.

---

## 📈 Empirical Benchmark Results

Evaluated on the identical held-out test set ($N_{\text{test}} = 1,000$ samples, 20% stratified split):

### Classification Leaderboard

| Model | Architecture | Test Accuracy | Macro-F1 | Weighted-F1 | Multiclass ROC-AUC | Generalization Error | Train Time | Inference Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Classical SVM** | RBF Kernel ($C=5.0, \gamma=0.05$) | **96.30%** | **0.9634** | **0.9630** | **0.9971** | **0.0370** | 8.59 s | 0.0142 s |
| **Logistic Regression** | Multinomial Linear Logits | **96.10%** | **0.9614** | **0.9611** | **0.9971** | **0.0390** | 0.04 s | 0.0002 s |
| **Qiskit 4-Qubit QSVM** | Fidelity Kernel ($ZZ$-map, $s=0.15$) | **95.90%** | **0.9601** | **0.9589** | **0.9968** | **0.0410** | 0.10 s | 0.3627 s |
| **XGBoost** | 100 Boosted Trees ($\eta=0.1$) | **95.90%** | **0.9594** | **0.9590** | **0.9964** | **0.0410** | 0.09 s | 0.0032 s |
| **Random Forest** | 100 Decision Trees | **95.80%** | **0.9586** | **0.9581** | **0.9975** | **0.0420** | 0.24 s | 0.0056 s |
| **PennyLane 4-Qubit VQC** | Strongly Entangling Layers | **67.70%** | **0.5601** | **0.6199** | **0.8124** | **0.3230** | 2.90 s | 1.2500 s |

### Continuous Regression Performance (`DNA_Damage_Score`)

| Regressor | RMSE | MAE | $R^2$ Score |
|---|---:|---:|---:|
| **Ridge Regression** | 1.4472 | 1.1462 | **0.9940** |
| **Random Forest Regressor** | 1.5251 | 1.2068 | **0.9933** |

### Key Scientific Findings
1. **QSVM Drastically Outperforms VQC (95.90% vs 67.70%):** On tabular physical sensor data, quantum kernel methods solve a convex quadratic program in dual space, completely avoiding the barren plateau landscapes and local minima that restrict parameterized variational quantum circuits.
2. **Absence of Quantum Advantage:** Classical SVM achieves higher accuracy (**96.30%** vs 95.90%) and runs ~25x faster on inference (0.014s vs 0.363s). On low-dimensional tabular data, quantum kernels provide equivalent representation power but no computational advantage.

---

## 🖥️ Streamlit Interactive Dashboard

Launch the interactive dashboard:

```bash
streamlit run app/app.py
```

### Dashboard Features
1. **Dataset & Sensor EDA:** Interactive exploration of 5,000 samples, correlation heatmaps, leakage audit charts, and KDE sensor distributions.
2. **Single Prediction:** Interactive sliders with clinical exposure presets (Low, Moderate, Severe), supporting all 6 classification models and continuous regression.
3. **Batch CSV Prediction:** Upload custom sensor CSV files, process batch predictions with selected model, and download scored CSV results with confidence ratings.
4. **Benchmark Leaderboard:** Full 6-model comparison table and high-resolution diagnostic plots:
   - Overall Benchmark (Accuracy, Macro-F1, ROC-AUC)
   - Macro Precision, Recall, and F1 Breakdown
   - Generalization Error
   - Training & Inference Latency
   - Combined 2x3 Confusion Matrix Grid
   - Multiclass One-vs-Rest ROC Curves
   - QSVM Quantum Kernel Gram Matrix Heatmap
   - PennyLane VQC Optimization Trajectory
   - DNA Damage Score Regression Fit
5. **Sensor Physics & Disambiguation:** Detailed physical explanations of NV ODMR and relaxometry, problem 7 clarification, and scientific appraisal.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Dhruv2-lang/quantum-biological-dosimetry.git
cd quantum-biological-dosimetry
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Retrain or Evaluate Models
```bash
# Train classical models & Classical SVM
python3 src/models_classical.py

# Train PennyLane Variational Quantum Classifier
python3 src/models_quantum.py

# Train Qiskit Quantum Support Vector Machine
python3 src/models_qsvm.py

# Run comprehensive comparative benchmark & regenerate plots
python3 src/compare_models.py
```

---

## 🧪 Testing & Quality Assurance

Run the automated test suite covering preprocessing, target leakage quarantine, classical models, quantum models, and quantum kernel mathematical properties:

```bash
python3 -m unittest discover tests
```

Output:
```text
Ran 10 tests in 1.533s

OK
```

All 10 unit tests verify:
- Complete absence of target leakage in sensor feature sets.
- Correct scaling and 4-component PCA projection (>95% variance explained).
- Classical model prediction validity and probability calibration.
- Continuous regression output bounded within biological range $[0, 100]$.
- PennyLane VQC 4-qubit circuit inference and probability normalization.
- Qiskit QSVM quantum kernel inference and probability normalization.
- Quantum kernel Gram matrix Hermiticity ($K = K^T$), normalization ($K_{ii} = 1.0$), and positive semi-definiteness ($\lambda_i \ge -10^{-5}$).
- Complete existence of all 15 model and evaluation report artifacts.

---

## ⚠️ Limitations & Scientific Integrity

1. **Tabular Telemetry vs. Problem 7 Image Processing:** The dataset consists solely of 7 scalar physical measurements. It contains no pixel data, histology slides, or microscopy images. No image processing or computer vision methods have been implemented or claimed.
2. **Preclinical / Simulation Status:** Data represents laboratory nanophotonic physical simulations. The software is not approved for clinical diagnostic or in vivo human use.
3. **Quantum Advantage:** No quantum advantage is claimed. Classical RBF SVM outperforms the quantum models in accuracy and execution speed.
