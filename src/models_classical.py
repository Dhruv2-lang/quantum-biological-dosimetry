"""
Classical Machine Learning Models for Quantum Biological Dosimetry.
Trains and benchmarks:
- Logistic Regression (Multinomial)
- Random Forest Classifier
- XGBoost Classifier
- Secondary Regression (Ridge & Random Forest) for DNA Damage Score
Saves all model artifacts, metric reports, and evaluation plots.
"""

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
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
    
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            random_state=42, 
            eval_metric="mlogloss"
        )
    }

    classification_results = {}
    fitted_models = {}

    for name, model in classifiers.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train_cls)
        y_pred = model.predict(X_test)
        
        acc = float(accuracy_score(y_test_cls, y_pred))
        macro_f1 = float(f1_score(y_test_cls, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_test_cls, y_pred, average="weighted"))
        macro_prec = float(precision_score(y_test_cls, y_pred, average="macro"))
        macro_rec = float(recall_score(y_test_cls, y_pred, average="macro"))
        cm = confusion_matrix(y_test_cls, y_pred).tolist()
        report = classification_report(y_test_cls, y_pred, target_names=class_names, output_dict=True)

        classification_results[name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "confusion_matrix": cm,
            "report": report
        }
        fitted_models[name] = model

        # Save individual model
        slug = name.lower().replace(" ", "_")
        model_path = os.path.join(MODELS_DIR, f"{slug}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path} | Accuracy: {acc*100:.2f}%, Macro-F1: {macro_f1:.4f}")

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, res) in enumerate(classification_results.items()):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
        axes[idx].set_title(f"{name}\nAcc: {res['accuracy']*100:.2f}%, F1: {res['macro_f1']:.4f}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    plt.tight_layout()
    cm_plot_path = os.path.join(FIGURES_DIR, "classical_confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices plot to {cm_plot_path}")

    print("\n--- Training Secondary Regression Models (DNA Damage Score) ---")
    regressors = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    }

    regression_results = {}
    for name, reg_model in regressors.items():
        print(f"Training {name}...")
        reg_model.fit(X_train, y_train_reg)
        y_reg_pred = reg_model.predict(X_test)

        mse = float(mean_squared_error(y_test_reg, y_reg_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test_reg, y_reg_pred))
        r2 = float(r2_score(y_test_reg, y_reg_pred))

        regression_results[name] = {
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4)
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

    # Save all metrics
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
