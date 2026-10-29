
# ⚛️ Quantum Biological Dosimetry
### Classical Machine Learning vs. Variational Quantum Classifier (VQC)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-purple.svg)](https://pennylane.ai/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)


An experimental machine learning and quantum computing framework for studying radiation-associated biological damage using physical sensor measurements inspired by nitrogen-vacancy (NV) diamond sensing and quantum-dot spectroscopy.

The project compares classical machine learning models with a Variational Quantum Classifier (VQC) and provides an interactive Streamlit dashboard for data exploration, model evaluation, and prediction.

> **Research disclaimer:** This is an experimental computational project, not a clinically validated diagnostic system. Dataset provenance, sensor assumptions, and model performance must be independently verified before drawing scientific or medical conclusions.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Target Leakage Prevention](#-target-leakage-prevention)
- [Project Structure](#-project-structure)
- [Models and Methodology](#-models-and-methodology)
- [Reported Results](#-reported-results)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Testing](#-testing)
- [Limitations and Future Work](#-limitations-and-future-work)
- [Reproducibility](#-reproducibility)


---

## 🔬 Overview

Biological dosimetry investigates biological responses associated with exposure to ionizing radiation. This project explores whether machine learning can classify radiation-associated toxicity categories from a restricted set of physical sensor measurements.

It also investigates a hybrid quantum-classical approach using a Variational Quantum Classifier implemented with PennyLane.

The project includes:

- Exploratory data analysis and feature inspection.
- A preprocessing pipeline with leakage safeguards.
- Classical classification and regression models.
- A parameterized quantum circuit for classification.
- Model evaluation and comparison reports.
- A Streamlit dashboard for interactive exploration.
- Automated tests for selected data and model components.

**Important distinction:** The current dataset is tabular. This implementation does not constitute a quantum image-processing pipeline and does not process histological images or pixel-level biological data.

## 🎯 Objectives

1. Explore relationships between physical sensor features and radiation-associated target labels.
2. Compare classical classifiers with a simulator-based Variational Quantum Classifier.
3. Investigate regression approaches for estimating a continuous DNA damage score.
4. Prevent direct target leakage during classification.
5. Provide reproducible code, evaluation artifacts, and an interactive interface.
6. Document the limitations of the available data and computational methods.

---

## 📊 Dataset

The project uses the following dataset file:

`data/Quantum_Biological_Dosimetry_Dataset_5000.csv`

The dataset is described as containing 5,000 records of tabular measurements.

**Dataset provenance must be verified.** Do not describe these records as real experimental or clinical measurements unless the original data source, collection procedure, and supporting documentation establish that claim. If the dataset is generated or simulated, explicitly identify it as synthetic.

### Physical sensor features

The classification pipeline is designed to use the following seven features:

| Feature | Description |
|---|---|
| `Dose_Gy` | Absorbed radiation dose, when known at prediction time |
| `ODMR_Shift_MHz` | Optically detected magnetic resonance frequency shift |
| `T1_us` | Longitudinal spin-lattice relaxation time |
| `T2_us` | Transverse spin-spin relaxation time |
| `Fluorescence_Intensity` | Fluorescence intensity measurement |
| `Fluorescence_Lifetime_ns` | Fluorescence decay lifetime |
| `QD_Emission_nm` | Quantum-dot emission wavelength |

The feature list represents the intended classification inputs. The actual training and inference code should be checked to confirm that the same feature order and definitions are used consistently.

### Biological endpoint and target variables

The dataset also contains biological endpoint measurements, including:

- `DNA_Damage_Score`
- `ROS_Level`
- `Mitochondrial_Damage`
- `GammaH2AX_Foci`
- `BP53_Foci`
- `Cell_Viability`

These variables require careful handling because some may be measured after exposure or may be directly involved in defining a prediction target.

---

## 🛡️ Target Leakage Prevention

Target leakage occurs when a model receives information that directly reveals the answer it is intended to predict.

In this project, `Radiotoxicity_Class` is described using thresholds on `DNA_Damage_Score`:

| Radiotoxicity class | DNA damage score |
|---|---|
| Low | Less than 25.0 |
| Moderate | Greater than or equal to 25.0 and less than 50.0 |
| High | Greater than or equal to 50.0 |

Equivalently:

- **Low:** `DNA_Damage_Score < 25.0`
- **Moderate:** `25.0 <= DNA_Damage_Score < 50.0`
- **High:** `DNA_Damage_Score >= 50.0`

Because `DNA_Damage_Score` defines the target, using it as a classification input would reveal the label-generation rule and produce misleading evaluation results.

### Classification safeguards

The intended classifier uses only the seven physical sensor features listed above.

The following variables are excluded from its input matrix:

- `DNA_Damage_Score`
- `ROS_Level`
- `Mitochondrial_Damage`
- `GammaH2AX_Foci`
- `BP53_Foci`
- `Cell_Viability`

The exclusion of biological endpoint variables is appropriate for the intended sensor-only prediction setup, provided that their availability and role are consistent with the research question.

**Additional safeguard:** `Dose_Gy` must also be excluded if the actual objective is to estimate an unknown radiation dose. A model cannot meaningfully estimate a dose that it is already given as an input.

Leakage prevention must be enforced in the actual feature-selection and preprocessing code, not just documented in this README.

---

## 🗂️ Project Structure

```text
quantum_biodosimetry/
├── app/
│   └── app.py
├── data/
│   └── Quantum_Biological_Dosimetry_Dataset_5000.csv
├── models/
│   ├── preprocessor_bundle.joblib
│   ├── data_splits.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── ridge_regression.joblib
│   ├── random_forest_regressor.joblib
│   ├── qnn_weights.pt
│   └── qnn_vqc.joblib
├── reports/
│   ├── eda_summary.json
│   ├── classical_metrics.json
│   ├── quantum_metrics.json
│   ├── benchmark_summary.json
│   └── figures/
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── models_classical.py
│   ├── models_quantum.py
│   └── compare_models.py
├── tests/
│   ├── test_pipeline.py
│   └── test_models.py
├── PROJECT_STATUS.md
├── RESEARCH_REPORT.md
├── requirements.txt
├── .gitignore
└── README.md
```

This tree describes the intended organization. Model artifact names and optional files may differ in the current repository; check the actual `models/` and `reports/` directories before relying on any listed path.

---

## 🧠 Models and Methodology

### 1. Exploratory Data Analysis

The EDA stage examines:

- Dataset dimensions and data types.
- Missing values and potential invalid measurements.
- Feature distributions and correlations.
- Target class distribution.
- Potential target leakage.
- Descriptive statistics and visualizations.

### 2. Preprocessing

The preprocessing pipeline prepares the sensor features for machine learning.

Depending on the implementation, it may include:

- Feature selection.
- Train/test splitting.
- Feature scaling.
- Dimensionality reduction.
- Target-label encoding.

Any scaler, dimensionality reducer, or other learned transformation must be fitted using training data only. The fitted transformations must then be reused for validation, test, and inference data.

### 3. Classical machine learning

The project compares classical approaches, including:

- **Logistic Regression:** a linear classification baseline.
- **Random Forest:** an ensemble of decision trees.
- **XGBoost:** a gradient-boosted tree model.

For the regression task, the project includes:

- **Ridge Regression:** regularized linear regression.
- **Random Forest Regressor:** a nonlinear tree-ensemble approach.

### 4. Variational Quantum Classifier

The quantum classification experiment uses PennyLane and a simulator-based parameterized quantum circuit.

The reported design uses:

- Four simulated qubits.
- Angle embedding for reduced-dimensional input features.
- Trainable single-qubit rotations and entangling operations.
- Measurement of Pauli-Z observables.
- A classical transformation of the measurement outputs into classification logits.
- PyTorch-based optimization.

The circuit runs on a quantum simulator. This is not evidence of execution on quantum hardware or of a quantum computational advantage.

---

## 📈 Reported Results

The existing project README reports the following classification results. They should be treated as **previously reported results, not independently verified benchmarks**, until reproduced from the current code, saved models, and test split.

### Classification

| Model | Reported accuracy | Reported macro-F1 |
|---|---:|---:|
| Logistic Regression | 96.10% | 0.9614 |
| XGBoost | 95.90% | 0.9594 |
| Random Forest | 95.80% | 0.9586 |
| Four-qubit VQC | 67.70% | 0.5601 |

The existing report describes an evaluation set of 1,000 samples. Confirm the sample count and metrics in the current evaluation artifacts before presenting these values as final results.

### Regression

The existing README also reports the following results for predicting `DNA_Damage_Score`:

| Model | Reported RMSE | Reported R² |
|---|---:|---:|
| Ridge Regression | 1.4472 | 0.9940 |
| Random Forest Regressor | 1.5251 | 0.9933 |

These regression results also require independent reproduction and a leakage audit.

### Interpretation

In the reported classification comparison, classical models outperform the VQC on this dataset.

The results do not establish quantum advantage. They also do not establish generalization to real biological samples, other populations, different sensors, or clinical use.

---

## 🚀 Installation

### Prerequisites

Install:

- Git
- Python with a version compatible with the dependencies in `requirements.txt`
- pip
- A terminal or command prompt

The project was originally documented for Python 3.10 or later. Verify compatibility with the pinned dependencies before selecting a different Python version.

### 1. Clone the repository

```bash
git clone https://github.com/Dhruv2-lang/quantum-biological-dosimetry.git
cd quantum-biological-dosimetry
```

### 2. Create a virtual environment

**macOS or Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, check the Python version and the package versions specified in `requirements.txt`. Native scientific libraries and quantum-computing dependencies may require compatible versions.

---

## ▶️ Running the Project

Run commands from the repository root with the virtual environment activated.

### 1. Exploratory data analysis

```bash
python src/eda.py
```

This runs the EDA script if the dataset and its expected paths are available.

### 2. Preprocessing

```bash
python src/preprocessing.py
```

### 3. Train classical models

```bash
python src/models_classical.py
```

### 4. Train the quantum model

```bash
python src/models_quantum.py
```

### 5. Generate model comparisons

```bash
python src/compare_models.py
```

Training commands may regenerate model artifacts and reports. Training can take longer than running an already-trained model, especially for quantum circuit simulations.

If you only want to use the dashboard, you may not need to retrain the models, provided the required model artifacts are included and compatible with your installed libraries.

---

## 🖥️ Streamlit Dashboard

Launch the dashboard with:

```bash
python -m streamlit run app/app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that address in your browser.

The dashboard is intended to provide:

- Dataset exploration and sensor-feature visualizations.
- Interactive single-record predictions.
- Batch predictions from uploaded CSV files.
- Model benchmark comparisons.
- Evaluation metrics and confusion matrices.
- Information about sensor features and the quantum-classical methodology.

The exact functionality depends on the current implementation in `app/app.py`.

**Note:** The application must be tested on a clean environment before claiming that it runs reliably on every supported operating system.

---

## 🧪 Testing

Run the project's unit tests with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests should be used to check the data pipeline, feature handling, model loading, and inference behavior supported by the current test suite.

A passing test suite does not by itself prove scientific validity, eliminate all forms of data leakage, or establish that the model generalizes to real experimental data.

---

## ⚠️ Limitations and Future Work

### Dataset provenance

The origin of the dataset must be documented and verified. If the data are synthetic or simulated, the README and research report should say so explicitly.

### No established clinical validity

The project is an experimental machine learning study. Its outputs should not be used to diagnose radiation injury, guide medical treatment, or replace validated biological dosimetry methods.

### No demonstrated quantum advantage

The reported VQC performance is lower than that of the classical baselines. No quantum advantage is claimed.

### Tabular data rather than image processing

The current workflow uses structured numerical features. It does not implement quantum image processing, a quantum convolutional neural network, or pixel-level analysis.

If the intended assignment requires quantum-centric image processing, a separate image-based dataset and image-processing pipeline would be necessary.

### Generalization and reproducibility

Further validation should include:

- Reproducing results from a clean environment.
- Checking class-wise precision, recall, F1-score, and confusion matrices.
- Comparing models using consistent data splits.
- Auditing for direct and indirect target leakage.
- Evaluating robustness to noise and distribution shifts.
- Documenting dataset generation and feature definitions.
- Testing against independently collected data where available.

### Future improvements

Potential extensions include:

- Stronger validation and calibration.
- More comprehensive baseline comparisons.
- Reproducible experiment configurations and random seeds.
- Better model and dependency version management.
- Independent experimental validation.
- A separate quantum image-processing implementation if required by the research objective.

---

## 🔁 Reproducibility

For reproducible results:

1. Use a documented Python environment and install dependencies from `requirements.txt`.
2. Keep the dataset and target definitions documented.
3. Use consistent train, validation, and test splits.
4. Fit preprocessing transformations on training data only.
5. Exclude target-defining variables from classifier inputs.
6. Save evaluation metrics and model configurations.
7. Record random seeds and package versions where applicable.
8. Verify that saved model artifacts are compatible with the installed library versions.

Model files serialized with libraries such as scikit-learn may not be compatible across different versions. For reproducibility, use compatible dependency versions and retain the training configuration.

