# Quantum Biological Dosimetry: Project Status

## Project Overview
- **Objective:** Predict radiation-induced biological damage from quantum sensor measurements (NV diamond ODMR, relaxometry T1/T2, Quantum Dot emission & lifetime) and benchmark Classical ML models against a genuine PennyLane Variational Quantum Classifier (VQC).
- **Workspace:** `/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry`
- **Dataset:** `Quantum_Biological_Dosimetry_Dataset_5000.csv` (5,000 verified authentic rows)

---

## Roadmap & Current Stage

| Phase | Description | Status | Output / Artifacts |
|---|---|---|---|
| **Phase 1** | Dataset validation, EDA, target leakage audit | ✅ **Completed** | `data/Quantum_Biological_Dosimetry_Dataset_5000.csv`, `reports/eda_summary.json`, figures |
| **Phase 2** | Feature engineering & Preprocessing pipeline | ✅ **Completed** | `src/preprocessing.py`, `models/preprocessor_bundle.joblib`, `models/data_splits.joblib` |
| **Phase 3** | Classical ML Baselines (LogReg, RF, XGBoost) & Regression | ✅ **Completed** | `src/models_classical.py`, `models/*.joblib`, `reports/classical_metrics.json`, figures |
| **Phase 4** | Trainable PennyLane Variational Quantum Classifier (VQC) | ✅ **Completed** | `src/models_quantum.py`, `models/qnn_vqc.joblib`, `models/qnn_weights.pt`, `reports/quantum_metrics.json`, figures |
| **Phase 5** | Model Comparison & Visualizations | ✅ **Completed** | `src/compare_models.py`, `reports/benchmark_summary.json`, figures |
| **Phase 6** | Streamlit Interactive Dashboard (Hardened) | ✅ **Completed** | `app/app.py` (Fixed macOS OpenMP thread conflict & Cocoa thread crash) |
| **Phase 7** | Automated Test Suite | ✅ **Completed** | `tests/test_pipeline.py`, `tests/test_models.py` (8/8 tests passing) |
| **Phase 8** | Research Report, Documentation & ZIP Archive | ✅ **Completed** | `README.md`, `RESEARCH_REPORT.md`, `requirements.txt`, `quantum_biodosimetry.zip` |

---

## Diagnostics & Root-Cause Resolution Ledger (macOS Segmentation Fault)

- **Root Cause 1 (Conflicting OpenMP Runtimes):**
  - *Symptom:* macOS Crash Log (`Python-*.ips`) confirmed `EXC_BAD_ACCESS (SIGSEGV)` at `0x580` inside `libomp.dylib :: __kmp_suspend_initialize_thread`.
  - *Mechanism:* Loading PyTorch (`torch/lib/libomp.dylib`) and XGBoost (`/opt/homebrew/*/libomp.dylib`) into the same process caused dyld symbol interposition and threadpool corruption when OpenMP attempted worker thread suspension (`KMP_BLOCKTIME`).
  - *Resolution:*
    1. Set `OMP_NUM_THREADS="1"` in `app/app.py` and set `n_jobs=1` on XGBoost model to prevent background OpenMP thread pooling.
    2. Implemented pure PennyLane NumPy statevector simulation in `TrainedQNNWrapper` during inference, completely removing `torch` and its bundled `libomp.dylib` from the Streamlit serving path while preserving 100% numerical identity (max diff $1.44 \times 10^{-15}$).
- **Root Cause 2 (Matplotlib Cocoa Main-Thread Violation):**
  - Matplotlib's default `macosx` backend initialized Cocoa on secondary Streamlit threads. Resolved by setting `matplotlib.use("Agg")` prior to importing `pyplot`.
- **Root Cause 3 (Warnings):**
  - Suppressed harmless `InconsistentVersionWarning` across scikit-learn 1.9.0/1.9.1.
  - Replaced/wrapped `use_container_width` to be compliant across Streamlit 1.47+ and future releases.
