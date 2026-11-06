"""
Classical Machine Learning Models for Quantum Biological Dosimetry.
Trains and benchmarks:
- Logistic Regression (Multinomial)
- Random Forest Classifier
- XGBoost Classifier
- Classical Support Vector Machine (SVM with RBF Kernel & Hyperparameter Tuning)
- Secondary Regression (Ridge & Random Forest) for DNA Damage Score
Saves all model artifacts, metric reports, and evaluation plots.
"""

import os
import json
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

MODELS_DIR = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/models"
REPORTS_DIR = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

def train_and_evaluate_classical():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    splits_path = os.path.join(MODELS_DIR, "data_splits.joblib")
    if not os.path.exists(splits_path):
        raise FileNotFoundError("data_splits.joblib not found. Run preprocessing.py first.")

    splits = joblib.load(splits_path)
    X_train = splits["X_train_scaled"]
    X_test = splits["X_test_scaled"]
    y_train_cls = splits["y_train_class"]
    y_test_cls = splits["y_test_class"]
    y_train_reg = splits["y_train_reg"]
    y_test_reg = splits["y_test_reg"]
    class_names = splits["class_names"]

    print("--- Training Classical Classification Models ---")

    classification_results = {}
    fitted_models = {}

    # 1. Logistic Regression
    print("Training Logistic Regression...")
    t0 = time.time()
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train_cls)
    t_train_logreg = time.time() - t0

    t0 = time.time()
    y_pred_logreg = logreg.predict(X_test)
    t_inf_logreg = time.time() - t0
    y_prob_logreg = logreg.predict_proba(X_test)

    fitted_models["Logistic Regression"] = logreg
    classification_results["Logistic Regression"] = {
        "accuracy": round(float(accuracy_score(y_test_cls, y_pred_logreg)), 4),
        "macro_f1": round(float(f1_score(y_test_cls, y_pred_logreg, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test_cls, y_pred_logreg, average="weighted")), 4),
        "macro_precision": round(float(precision_score(y_test_cls, y_pred_logreg, average="macro")), 4),
        "weighted_precision": round(float(precision_score(y_test_cls, y_pred_logreg, average="weighted")), 4),
        "macro_recall": round(float(recall_score(y_test_cls, y_pred_logreg, average="macro")), 4),
        "weighted_recall": round(float(recall_score(y_test_cls, y_pred_logreg, average="weighted")), 4),
        "roc_auc": round(float(roc_auc_score(y_test_cls, y_prob_logreg, multi_class="ovr")), 4),
        "generalization_error": round(float(1.0 - accuracy_score(y_test_cls, y_pred_logreg)), 4),
        "training_time_seconds": round(t_train_logreg, 4),
        "inference_time_seconds": round(t_inf_logreg, 4),
        "confusion_matrix": confusion_matrix(y_test_cls, y_pred_logreg).tolist(),
        "report": classification_report(y_test_cls, y_pred_logreg, target_names=class_names, output_dict=True)
    }
    joblib.dump(logreg, os.path.join(MODELS_DIR, "logistic_regression.joblib"))

    # 2. Random Forest
    print("Training Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    rf.fit(X_train, y_train_cls)
    t_train_rf = time.time() - t0

    t0 = time.time()
    y_pred_rf = rf.predict(X_test)
    t_inf_rf = time.time() - t0
    y_prob_rf = rf.predict_proba(X_test)

    fitted_models["Random Forest"] = rf
    classification_results["Random Forest"] = {
        "accuracy": round(float(accuracy_score(y_test_cls, y_pred_rf)), 4),
        "macro_f1": round(float(f1_score(y_test_cls, y_pred_rf, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test_cls, y_pred_rf, average="weighted")), 4),
        "macro_precision": round(float(precision_score(y_test_cls, y_pred_rf, average="macro")), 4),
        "weighted_precision": round(float(precision_score(y_test_cls, y_pred_rf, average="weighted")), 4),
        "macro_recall": round(float(recall_score(y_test_cls, y_pred_rf, average="macro")), 4),
        "weighted_recall": round(float(recall_score(y_test_cls, y_pred_rf, average="weighted")), 4),
        "roc_auc": round(float(roc_auc_score(y_test_cls, y_prob_rf, multi_class="ovr")), 4),
        "generalization_error": round(float(1.0 - accuracy_score(y_test_cls, y_pred_rf)), 4),
        "training_time_seconds": round(t_train_rf, 4),
        "inference_time_seconds": round(t_inf_rf, 4),
        "confusion_matrix": confusion_matrix(y_test_cls, y_pred_rf).tolist(),
        "report": classification_report(y_test_cls, y_pred_rf, target_names=class_names, output_dict=True)
    }
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))

    # 3. XGBoost
    print("Training XGBoost...")
    t0 = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=1
    )
    xgb_clf.fit(X_train, y_train_cls)
    t_train_xgb = time.time() - t0

    t0 = time.time()
    y_pred_xgb = xgb_clf.predict(X_test)
    t_inf_xgb = time.time() - t0
    y_prob_xgb = xgb_clf.predict_proba(X_test)

    fitted_models["XGBoost"] = xgb_clf
    classification_results["XGBoost"] = {
        "accuracy": round(float(accuracy_score(y_test_cls, y_pred_xgb)), 4),
        "macro_f1": round(float(f1_score(y_test_cls, y_pred_xgb, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test_cls, y_pred_xgb, average="weighted")), 4),
        "macro_precision": round(float(precision_score(y_test_cls, y_pred_xgb, average="macro")), 4),
        "weighted_precision": round(float(precision_score(y_test_cls, y_pred_xgb, average="weighted")), 4),
        "macro_recall": round(float(recall_score(y_test_cls, y_pred_xgb, average="macro")), 4),
        "weighted_recall": round(float(recall_score(y_test_cls, y_pred_xgb, average="weighted")), 4),
        "roc_auc": round(float(roc_auc_score(y_test_cls, y_prob_xgb, multi_class="ovr")), 4),
        "generalization_error": round(float(1.0 - accuracy_score(y_test_cls, y_pred_xgb)), 4),
        "training_time_seconds": round(t_train_xgb, 4),
        "inference_time_seconds": round(t_inf_xgb, 4),
        "confusion_matrix": confusion_matrix(y_test_cls, y_pred_xgb).tolist(),
        "report": classification_report(y_test_cls, y_pred_xgb, target_names=class_names, output_dict=True)
    }
    joblib.dump(xgb_clf, os.path.join(MODELS_DIR, "xgboost.joblib"))

    # 4. Classical Support Vector Machine (SVM with RBF Kernel & Hyperparameter Tuning)
    print("Training Classical Support Vector Machine (RBF kernel with CV tuning)...")
    param_grid_svm = {
        "C": [0.5, 1.0, 5.0, 10.0],
        "gamma": ["scale", "auto", 0.05, 0.1]
    }
    base_svc = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)

    t0 = time.time()
    grid_svm = GridSearchCV(base_svc, param_grid_svm, cv=5, scoring="f1_macro", n_jobs=1)
    grid_svm.fit(X_train, y_train_cls)
    t_train_svm = time.time() - t0

    best_svm = grid_svm.best_estimator_
    print(f"Best Classical SVM Parameters: {grid_svm.best_params_} (CV Macro-F1: {grid_svm.best_score_:.4f})")

    t0 = time.time()
    y_pred_svm = best_svm.predict(X_test)
    t_inf_svm = time.time() - t0
    y_prob_svm = best_svm.predict_proba(X_test)

    svm_metrics = {
        "model_name": "Classical Support Vector Machine (SVM)",
        "kernel": "RBF",
        "best_parameters": grid_svm.best_params_,
        "cv_best_macro_f1": round(float(grid_svm.best_score_), 4),
        "accuracy": round(float(accuracy_score(y_test_cls, y_pred_svm)), 4),
        "macro_f1": round(float(f1_score(y_test_cls, y_pred_svm, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test_cls, y_pred_svm, average="weighted")), 4),
        "macro_precision": round(float(precision_score(y_test_cls, y_pred_svm, average="macro")), 4),
        "weighted_precision": round(float(precision_score(y_test_cls, y_pred_svm, average="weighted")), 4),
        "macro_recall": round(float(recall_score(y_test_cls, y_pred_svm, average="macro")), 4),
        "weighted_recall": round(float(recall_score(y_test_cls, y_pred_svm, average="weighted")), 4),
        "roc_auc": round(float(roc_auc_score(y_test_cls, y_prob_svm, multi_class="ovr")), 4),
        "generalization_error": round(float(1.0 - accuracy_score(y_test_cls, y_pred_svm)), 4),
        "training_time_seconds": round(t_train_svm, 4),
        "inference_time_seconds": round(t_inf_svm, 4),
        "confusion_matrix": confusion_matrix(y_test_cls, y_pred_svm).tolist(),
        "report": classification_report(y_test_cls, y_pred_svm, target_names=class_names, output_dict=True)
    }

    fitted_models["Classical SVM"] = best_svm
    classification_results["Classical SVM"] = {
        "accuracy": svm_metrics["accuracy"],
        "macro_f1": svm_metrics["macro_f1"],
        "weighted_f1": svm_metrics["weighted_f1"],
        "macro_precision": svm_metrics["macro_precision"],
        "weighted_precision": svm_metrics["weighted_precision"],
        "macro_recall": svm_metrics["macro_recall"],
        "weighted_recall": svm_metrics["weighted_recall"],
        "roc_auc": svm_metrics["roc_auc"],
        "generalization_error": svm_metrics["generalization_error"],
        "training_time_seconds": svm_metrics["training_time_seconds"],
        "inference_time_seconds": svm_metrics["inference_time_seconds"],
        "confusion_matrix": svm_metrics["confusion_matrix"],
        "report": svm_metrics["report"]
    }

    # Save SVM model with both standard filenames for compatibility
    svm_path1 = os.path.join(MODELS_DIR, "svm_classifier.joblib")
    svm_path2 = os.path.join(MODELS_DIR, "svm.joblib")
    joblib.dump(best_svm, svm_path1)
    joblib.dump(best_svm, svm_path2)
    print(f"Saved Classical SVM to {svm_path1}")

    svm_metrics_path = os.path.join(REPORTS_DIR, "svm_metrics.json")
    with open(svm_metrics_path, "w") as f:
        json.dump(svm_metrics, f, indent=2)
    print(f"Saved SVM metrics to {svm_metrics_path}")

    # Plot Confusion Matrices for all 4 classical classifiers
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    cm_items = list(classification_results.items())
    for idx, (name, res) in enumerate(cm_items):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
        axes[idx].set_title(f"{name}\nAcc: {res['accuracy']*100:.2f}%, F1: {res['macro_f1']:.4f}, AUC: {res['roc_auc']:.4f}",
                            fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    plt.tight_layout()
    cm_plot_path = os.path.join(FIGURES_DIR, "classical_confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved classical confusion matrices plot to {cm_plot_path}")

    print("\n--- Training Secondary Regression Models (DNA Damage Score) ---")
    regressors = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    }

    regression_results = {}
    for name, reg_model in regressors.items():
        print(f"Training {name}...")
        t0 = time.time()
        reg_model.fit(X_train, y_train_reg)
        t_reg_train = time.time() - t0

        t0 = time.time()
        y_reg_pred = reg_model.predict(X_test)
        t_reg_inf = time.time() - t0

        mse = float(mean_squared_error(y_test_reg, y_reg_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test_reg, y_reg_pred))
        r2 = float(r2_score(y_test_reg, y_reg_pred))

        regression_results[name] = {
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "training_time_seconds": round(t_reg_train, 4),
            "inference_time_seconds": round(t_reg_inf, 4)
        }

        slug = name.lower().replace(" ", "_")
        model_path = os.path.join(MODELS_DIR, f"{slug}.joblib")
        joblib.dump(reg_model, model_path)
        print(f"Saved {name} to {model_path} | RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Plot Regression Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (name, reg_model) in enumerate(regressors.items()):
        y_reg_pred = reg_model.predict(X_test)
        r2 = regression_results[name]["r2"]
        rmse = regression_results[name]["rmse"]
        axes[idx].scatter(y_test_reg, y_reg_pred, alpha=0.3, color="#2b5c8f", s=15)
        min_val = min(y_test_reg.min(), y_reg_pred.min())
        max_val = max(y_test_reg.max(), y_reg_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Identity Line (y=x)")
        axes[idx].set_title(f"{name}\nRMSE: {rmse:.3f}, R²: {r2:.4f}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Actual DNA Damage Score")
        axes[idx].set_ylabel("Predicted DNA Damage Score")
        axes[idx].legend()
    plt.tight_layout()
    reg_plot_path = os.path.join(FIGURES_DIR, "regression_evaluation.png")
    plt.savefig(reg_plot_path, dpi=300)
    plt.close()
    print(f"Saved regression plot to {reg_plot_path}")

    # Save all classical metrics
    classical_summary = {
        "classification": classification_results,
        "regression": regression_results
    }
    summary_path = os.path.join(REPORTS_DIR, "classical_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(classical_summary, f, indent=2)
    print(f"Saved classical metrics to {summary_path}")

    return classical_summary

if __name__ == "__main__":
    train_and_evaluate_classical()
