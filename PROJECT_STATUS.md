# Quantum Biological Dosimetry: Project Status

## Project Overview
- **Objective:** Predict radiation-induced biological damage from quantum sensor measurements (NV diamond ODMR, relaxometry T1/T2, Quantum Dot emission & lifetime) and benchmark Classical ML models against a PennyLane Variational Quantum Classifier (VQC) and a Qiskit Quantum Support Vector Machine (QSVM).
- **Workspace:** `/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry`
- **Dataset:** `Quantum_Biological_Dosimetry_Dataset_5000.csv` (5,000 verified authentic rows)

---

## Roadmap & Milestone Tracker

| Phase | Description | Status | Output / Artifacts |
|---|---|---|---|
| **Phase 1** | Dataset validation, EDA, target leakage audit | ✅ **Completed** | `data/Quantum_Biological_Dosimetry_Dataset_5000.csv`, `reports/eda_summary.json`, figures |
| **Phase 2** | Feature engineering & Preprocessing pipeline | ✅ **Completed** | `src/preprocessing.py`, `models/preprocessor_bundle.joblib`, `models/data_splits.joblib` |
| **Phase 3** | Classical ML Baselines (LogReg, RF, XGBoost) & Regression | ✅ **Completed** | `src/models_classical.py`, `models/*.joblib`, `reports/classical_metrics.json`, figures |
| **Phase 4** | Trainable PennyLane Variational Quantum Classifier (VQC) | ✅ **Completed** | `src/models_quantum.py`, `models/qnn_vqc.joblib`, `models/qnn_weights.pt`, `reports/quantum_metrics.json`, figures |
| **Phase 5** | Classical SVM (RBF Kernel, 5-Fold CV Tuning) | ✅ **Completed** | `models/svm_classifier.joblib`, `models/svm.joblib`, `reports/svm_metrics.json` |
| **Phase 6** | Qiskit Quantum SVM (QSVM, ZZ-FeatureMap Fidelity Kernel) | ✅ **Completed** | `src/models_qsvm.py`, `models/qsvm_classifier.joblib`, `models/qsvm.joblib`, `models/qsvm_config.json`, `reports/qsvm_metrics.json` |
| **Phase 7** | Comprehensive 6-Model Benchmark & Visualizations | ✅ **Completed** | `src/compare_models.py`, `reports/benchmark_summary.json`, 15 figures in `reports/figures/` |
| **Phase 8** | Streamlit Interactive Dashboard (Hardened for macOS) | ✅ **Completed** | `app/app.py` (Supports all 6 models, single & batch inference, 5 tabs) |
| **Phase 9** | Automated Unit Test Suite (10/10 Passing) | ✅ **Completed** | `tests/test_pipeline.py`, `tests/test_models.py` (Unit tests for leakage, classical, VQC, QSVM, kernel properties) |
| **Phase 10** | Research Report, Documentation & ZIP Packaging | ✅ **Completed** | `README.md`, `RESEARCH_REPORT.md`, `requirements.txt`, `quantum_biodosimetry.zip` |

---

## Master Classification Benchmark (Held-Out Test N = 1,000)

| Model | Architecture | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | Gen. Error | Train Time | Inference Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Classical SVM** | RBF Kernel ($C=5.0, \gamma=0.05$) | **96.30%** | **0.9634** | **0.9630** | **0.9971** | **0.0370** | 8.59 s | 0.0142 s |
| **Logistic Regression** | Multinomial Linear Logits | **96.10%** | **0.9614** | **0.9611** | **0.9971** | **0.0390** | 0.04 s | 0.0002 s |
| **Qiskit 4-Qubit QSVM** | Fidelity Kernel ($ZZ$-map, $s=0.15$) | **95.90%** | **0.9601** | **0.9589** | **0.9968** | **0.0410** | 0.10 s | 0.3627 s |
| **XGBoost** | 100 Boosted Trees ($\eta=0.1$) | **95.90%** | **0.9594** | **0.9590** | **0.9964** | **0.0410** | 0.09 s | 0.0032 s |
| **Random Forest** | 100 Decision Trees | **95.80%** | **0.9586** | **0.9581** | **0.9975** | **0.0420** | 0.24 s | 0.0056 s |
| **PennyLane 4-Qubit VQC** | Strongly Entangling Layers | **67.70%** | **0.5601** | **0.6199** | **0.8124** | **0.3230** | 2.90 s | 1.2500 s |

---

## Architectural & Stability Ledger (macOS ARM64)

- **Single-Threaded OpenMP Pinning:** Enforced `OMP_NUM_THREADS="1"` in `app/app.py` and `compare_models.py` to prevent OpenMP runtime collision (`__kmp_suspend_initialize_thread` SIGSEGV) between PyTorch, XGBoost, and Qiskit.
- **Headless Matplotlib Backend:** Pinned `matplotlib.use("Agg")` across all modules before importing `pyplot` to prevent Cocoa/AppKit main-thread violations during multi-threaded Streamlit rendering.
- **QSVM Phase Scaling:** Added scale factor $s = 0.15$ to map 4-dimensional PCA features smoothly into $[-\pi, \pi]$ within Qiskit's `ZZFeatureMap`, eliminating phase scrambling and elevating test accuracy from 42.20% to 95.90%.
- **Target Leakage Safeguard:** Downstream biological assay endpoints (`ROS_Level`, `Mitochondrial_Damage`, `GammaH2AX_Foci`, `BP53_Foci`, `Cell_Viability`, `DNA_Damage_Score`) are strictly quarantined and never used as classification inputs.
