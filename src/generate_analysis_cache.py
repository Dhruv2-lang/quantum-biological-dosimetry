"""
Generates and caches comprehensive evaluation artifacts for all 6 classification models:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. Classical SVM
5. PennyLane 4-Qubit VQC
6. Qiskit 4-Qubit QSVM

Computes exact ROC curves, PR curves, confusion matrices, classification reports,
feature importances, permutation importances, PCA 2D decision boundary,
kernel eigenvalue spectrum, and quantum circuit diagrams.
Saves to reports/model_analysis_cache.joblib for instant, reactive dashboard rendering.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize

from src.models_quantum import load_qnn_classifier
from src.models_qsvm import load_qsvm_classifier

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CACHE_FILE = os.path.join(REPORTS_DIR, "model_analysis_cache.joblib")


def generate_cache():
    print("Loading data splits and preprocessor...")
    splits = joblib.load(os.path.join(MODELS_DIR, "data_splits.joblib"))
    y_test = splits["y_test_class"]
    X_test_scaled = splits["X_test_scaled"]
    X_test_pca4 = splits["X_test_pca4"]
    X_train_scaled = splits["X_train_scaled"]
    X_train_pca4 = splits["X_train_pca4"]
    y_train = splits["y_train_class"]
    class_names = splits["class_names"]  # ['High', 'Low', 'Moderate']

    # Sensor feature names
    sensor_features = [
        "Dose_Gy",
        "ODMR_Shift_MHz",
        "T1_us",
        "T2_us",
        "Fluorescence_Intensity",
        "Fluorescence_Lifetime_ns",
        "QD_Emission_nm"
    ]

    # Load benchmark summary for historical execution metrics
    bench_file = os.path.join(REPORTS_DIR, "benchmark_summary.json")
    with open(bench_file) as f:
        bench_data = json.load(f)
    bench_map = {m["Model"]: m for m in bench_data["classification_comparison"]}

    # Load models
    print("Loading models...")
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib"))
    if hasattr(xgb_model, "set_params"):
        try:
            xgb_model.set_params(n_jobs=1)
        except Exception:
            pass

    models_dict = {
        "Logistic Regression": {
            "type": "Classical ML (Linear)",
            "input": "scaled",
            "model": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
        },
        "Random Forest": {
            "type": "Classical ML (Ensemble)",
            "input": "scaled",
            "model": joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
        },
        "XGBoost": {
            "type": "Classical ML (Gradient Boosting)",
            "input": "scaled",
            "model": xgb_model
        },
        "Classical SVM": {
            "type": "Classical Kernel Machine",
            "input": "scaled",
            "model": joblib.load(os.path.join(MODELS_DIR, "svm_classifier.joblib"))
        },
        "PennyLane 4-Qubit VQC": {
            "type": "Variational Quantum Circuit",
            "input": "pca4",
            "model": load_qnn_classifier(os.path.join(MODELS_DIR, "qnn_vqc.joblib"))
        },
        "Qiskit 4-Qubit QSVM": {
            "type": "Quantum Kernel Machine",
            "input": "pca4",
            "model": load_qsvm_classifier(os.path.join(MODELS_DIR, "qsvm_classifier.joblib"))
        }
    }

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = len(class_names)

    cache = {
        "class_names": class_names,
        "sensor_features": sensor_features,
        "models": {}
    }

    for name, info in models_dict.items():
        print(f"Evaluating {name}...")
        m = info["model"]
        inp = info["input"]
        X_test = X_test_pca4 if inp == "pca4" else X_test_scaled

        # Predict probabilities and labels
        probs = m.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)

        # Baseline metrics
        acc = float(accuracy_score(y_test, preds))
        macro_prec = float(precision_score(y_test, preds, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_test, preds, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        try:
            roc_auc = float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = 0.5

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        # Classification Report dict
        clr = classification_report(
            y_test, preds, target_names=class_names, output_dict=True, zero_division=0
        )

        # Per-Class Dataframe for charts
        class_perf = []
        for i, c_name in enumerate(class_names):
            class_perf.append({
                "Class": c_name,
                "Precision": float(clr[c_name]["precision"]),
                "Recall": float(clr[c_name]["recall"]),
                "F1-Score": float(clr[c_name]["f1-score"]),
                "Support": int(clr[c_name]["support"])
            })
        class_perf_df = pd.DataFrame(class_perf)

        # Multiclass ROC curves (One-vs-Rest)
        roc_data = {}
        for i in range(n_classes):
            c_name = class_names[i]
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
            roc_auc_val = float(auc(fpr, tpr))
            roc_data[c_name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": roc_auc_val
            }
        
        # Micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
        roc_data["Micro-Average"] = {
            "fpr": fpr_micro.tolist(),
            "tpr": tpr_micro.tolist(),
            "auc": float(auc(fpr_micro, tpr_micro))
        }

        # Multiclass Precision-Recall curves
        pr_data = {}
        for i in range(n_classes):
            c_name = class_names[i]
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], probs[:, i])
            ap = float(average_precision_score(y_test_bin[:, i], probs[:, i]))
            pr_data[c_name] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "ap": ap
            }

        # Benchmark reference data for latency & gen error
        bench_info = bench_map.get(name, {})
        train_time = bench_info.get("Training Time (s)", 0.0)
        infer_time = bench_info.get("Inference Time (s)", 0.0)
        gen_error = round(1.0 - acc, 4)

        model_entry = {
            "model_name": name,
            "category": info["type"],
            "metrics": {
                "accuracy": acc,
                "macro_precision": macro_prec,
                "macro_recall": macro_rec,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "roc_auc": roc_auc,
                "training_time": train_time,
                "inference_time": infer_time,
                "generalization_error": gen_error
            },
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_norm": cm_norm.tolist(),
            "classification_report": clr,
            "class_perf_df": class_perf_df,
            "roc_curves": roc_data,
            "pr_curves": pr_data,
        }

        # ----------------------------------------------------
        # MODEL SPECIFIC METRICS & ARTIFACTS
        # ----------------------------------------------------
        if name == "Logistic Regression":
            model_entry["coefficients"] = m.coef_.tolist()
            model_entry["intercept"] = m.intercept_.tolist()
            model_entry["odds_ratios"] = np.exp(m.coef_).tolist()

        elif name == "Random Forest":
            model_entry["feature_importances"] = m.feature_importances_.tolist()
            model_entry["n_estimators"] = m.n_estimators
            model_entry["max_depth"] = m.max_depth

        elif name == "XGBoost":
            model_entry["feature_importances"] = m.feature_importances_.tolist()
            model_entry["shap_status"] = (
                "SHAP analysis is currently omitted on this environment due to Numba runtime "
                "incompatibility with NumPy 2.5 on macOS Python 3.13. Displaying native XGBoost "
                "Gain and Weight importances instead."
            )

        elif name == "Classical SVM":
            model_entry["C"] = 5.0
            model_entry["gamma"] = 0.05
            model_entry["kernel"] = "rbf"
            model_entry["n_support"] = m.n_support_.tolist() if hasattr(m, "n_support_") else []
            model_entry["total_support_vectors"] = int(np.sum(m.n_support_)) if hasattr(m, "n_support_") else 0

            # Permutation importance on test set
            print("Calculating permutation importance for Classical SVM...")
            perm_res = permutation_importance(
                m, X_test_scaled, y_test, n_repeats=5, random_state=42, n_jobs=1
            )
            model_entry["permutation_importance"] = {
                "importances_mean": perm_res.importances_mean.tolist(),
                "importances_std": perm_res.importances_std.tolist()
            }

            # 2D PCA Decision Boundary Visualization
            print("Computing 2D PCA Decision Boundary for Classical SVM...")
            pca_2d_svm = SVC(kernel="rbf", C=5.0, gamma=0.05, random_state=42)
            pca_2d_svm.fit(X_train_pca4[:, :2], y_train)
            
            x_min, x_max = X_test_pca4[:, 0].min() - 0.5, X_test_pca4[:, 0].max() + 0.5
            y_min, y_max = X_test_pca4[:, 1].min() - 0.5, X_test_pca4[:, 1].max() + 0.5
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100)
            )
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = pca_2d_svm.predict(grid_points)
            Z = Z.reshape(xx.shape)

            model_entry["pca_2d_boundary"] = {
                "xx": xx.tolist(),
                "yy": yy.tolist(),
                "Z": Z.tolist(),
                "pca1_test": X_test_pca4[:300, 0].tolist(),
                "pca2_test": X_test_pca4[:300, 1].tolist(),
                "y_test_subset": y_test[:300].tolist()
            }

        elif name == "PennyLane 4-Qubit VQC":
            q_metrics_file = os.path.join(REPORTS_DIR, "quantum_metrics.json")
            with open(q_metrics_file) as f:
                q_meta = json.load(f)
            model_entry["loss_history"] = q_meta.get("loss_history", [])
            model_entry["accuracy_history"] = q_meta.get("accuracy_history", [])
            model_entry["circuit_summary"] = {
                "qubits": 4,
                "layers": 2,
                "trainable_parameters": 24,
                "embedding": "AngleEmbedding (Pauli-Y)",
                "ansatz": "StronglyEntanglingLayers",
                "readout": "Pauli-Z expectation values on wires 0, 1, 2"
            }

        elif name == "Qiskit 4-Qubit QSVM":
            qsvm_meta_file = os.path.join(REPORTS_DIR, "qsvm_metrics.json")
            with open(qsvm_meta_file) as f:
                qsvm_meta = json.load(f)
            
            # Extract kernel sample (50 x 50) and eigenvalues
            subset_X = m.X_train_[:50]
            K_50 = m.evaluate_kernel(subset_X)
            eigenvalues = np.linalg.eigvalsh(m.evaluate_kernel(m.X_train_[:100]))
            eigenvalues_sorted = np.sort(eigenvalues)[::-1]

            # Text circuit diagram
            fm = m.feature_map
            try:
                circuit_text = str(fm.draw("text"))
            except Exception:
                circuit_text = "4-Qubit ZZFeatureMap(feature_dimension=4, reps=1, entanglement='linear')"

            # Intra-class vs Inter-class fidelity distribution
            y_sub = y_train[:50]
            same_class_fidelities = []
            diff_class_fidelities = []
            for i in range(50):
                for j in range(i + 1, 50):
                    fid = float(K_50[i, j])
                    if y_sub[i] == y_sub[j]:
                        same_class_fidelities.append(fid)
                    else:
                        diff_class_fidelities.append(fid)

            model_entry["qsvm_details"] = {
                "num_qubits": 4,
                "reps": 1,
                "entanglement": "linear",
                "scale_factor": 0.15,
                "kernel_matrix_50": K_50.tolist(),
                "eigenvalues": eigenvalues_sorted.tolist(),
                "circuit_text": circuit_text,
                "same_class_fidelities": same_class_fidelities,
                "diff_class_fidelities": diff_class_fidelities
            }

        cache["models"][name] = model_entry

    # ----------------------------------------------------
    # HEAD-TO-HEAD SVM VS QSVM COMPARISON ARTIFACTS
    # ----------------------------------------------------
    print("Computing SVM vs QSVM Kernel Comparison...")
    svm_clf = models_dict["Classical SVM"]["model"]
    qsvm_clf = models_dict["Qiskit 4-Qubit QSVM"]["model"]

    # Compute RBF kernel vs Quantum kernel on same 40 test samples
    sub_7d = X_test_scaled[:40]
    sub_pca4 = X_test_pca4[:40]

    # RBF kernel: exp(-gamma * ||x1 - x2||^2)
    gamma = 0.05
    dists = np.sum((sub_7d[:, np.newaxis, :] - sub_7d[np.newaxis, :, :]) ** 2, axis=-1)
    K_rbf = np.exp(-gamma * dists)
    K_quantum = qsvm_clf.evaluate_kernel(sub_pca4)

    rbf_vals = K_rbf[np.triu_indices(40, k=1)].tolist()
    quantum_vals = K_quantum[np.triu_indices(40, k=1)].tolist()

    cache["svm_vs_qsvm"] = {
        "rbf_vals": rbf_vals,
        "quantum_vals": quantum_vals
    }

    print(f"Saving analysis cache to {CACHE_FILE}...")
    joblib.dump(cache, CACHE_FILE)
    print("Cache generation complete! Size:", os.path.getsize(CACHE_FILE), "bytes")


if __name__ == "__main__":
    generate_cache()
