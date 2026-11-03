
"""
Classical Machine Learning Models for Quantum Biological Dosimetry.

Trains and benchmarks:
- Logistic Regression (Multinomial)
- Random Forest Classifier
- XGBoost Classifier
- Secondary Regression (Ridge & Random Forest) for DNA Damage Score

Saves all model artifacts, metric reports, and evaluation plots.
"""

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

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


# ==========================================
# 1. CROSS-PLATFORM PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# ==========================================
# 2. MAIN TRAINING FUNCTION
# ==========================================

def train_and_evaluate_classical():

    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # LOAD PREPROCESSED DATA
    # ==========================================

    splits_path = MODELS_DIR / "data_splits.joblib"

    if not splits_path.exists():
        raise FileNotFoundError(
            "data_splits.joblib not found. "
            "Run preprocessing.py first."
        )

    splits = joblib.load(splits_path)

    X_train = splits["X_train_scaled"]
    X_test = splits["X_test_scaled"]

    y_train_cls = splits["y_train_class"]
    y_test_cls = splits["y_test_class"]

    y_train_reg = splits["y_train_reg"]
    y_test_reg = splits["y_test_reg"]

    class_names = splits["class_names"]

    # Ensure class names match encoded labels
    labels = np.arange(len(class_names))

    print("\n--- Classical Model Training ---")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Class encoding: {class_names}")

    # ==========================================
    # CLASSIFICATION MODELS
    # ==========================================

    classifiers = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42
        ),

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

    print("\n--- Training Classical Classification Models ---")

    for name, model in classifiers.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train_cls)

        y_pred = model.predict(X_test)

        # Classification metrics
        acc = float(
            accuracy_score(y_test_cls, y_pred)
        )

        macro_f1 = float(
            f1_score(
                y_test_cls,
                y_pred,
                average="macro",
                zero_division=0
            )
        )

        weighted_f1 = float(
            f1_score(
                y_test_cls,
                y_pred,
                average="weighted",
                zero_division=0
            )
        )

        macro_prec = float(
            precision_score(
                y_test_cls,
                y_pred,
                average="macro",
                zero_division=0
            )
        )

        macro_rec = float(
            recall_score(
                y_test_cls,
                y_pred,
                average="macro",
                zero_division=0
            )
        )

        cm = confusion_matrix(
            y_test_cls,
            y_pred,
            labels=labels
        ).tolist()

        report = classification_report(
            y_test_cls,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        classification_results[name] = {

            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),

            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),

            "confusion_matrix": cm,
            "class_names": class_names,
            "report": report
        }

        fitted_models[name] = model

        # Save individual model
        slug = name.lower().replace(" ", "_")

        model_path = MODELS_DIR / f"{slug}.joblib"

        joblib.dump(model, model_path)

        print(
            f"Saved {name} | "
            f"Accuracy: {acc * 100:.2f}% | "
            f"Macro-F1: {macro_f1:.4f}"
        )

    # ==========================================
    # PLOT CLASSIFICATION CONFUSION MATRICES
    # ==========================================

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, res) in enumerate(
        classification_results.items()
    ):

        cm = np.array(res["confusion_matrix"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[idx]
        )

        axes[idx].set_title(
            f"{name}\n"
            f"Acc: {res['accuracy'] * 100:.2f}%, "
            f"F1: {res['macro_f1']:.4f}",
            fontsize=12,
            fontweight="bold"
        )

        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")

    plt.tight_layout()

    cm_plot_path = (
        FIGURES_DIR / "classical_confusion_matrices.png"
    )

    plt.savefig(
        cm_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved confusion matrices: {cm_plot_path}")

    # ==========================================
    # REGRESSION MODELS
    # ==========================================

    print(
        "\n--- Training Secondary Regression Models "
        "(DNA Damage Score) ---"
    )

    regressors = {

        "Ridge Regression": Ridge(
            alpha=1.0
        ),

        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42
        )
    }

    regression_results = {}
    fitted_regressors = {}

    for name, reg_model in regressors.items():

        print(f"\nTraining {name}...")

        reg_model.fit(X_train, y_train_reg)

        y_reg_pred = reg_model.predict(X_test)

        mse = float(
            mean_squared_error(y_test_reg, y_reg_pred)
        )

        rmse = float(np.sqrt(mse))

        mae = float(
            mean_absolute_error(y_test_reg, y_reg_pred)
        )

        r2 = float(
            r2_score(y_test_reg, y_reg_pred)
        )

        regression_results[name] = {

            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4)
        }

        fitted_regressors[name] = reg_model

        # Save regression model
        slug = name.lower().replace(" ", "_")

        model_path = MODELS_DIR / f"{slug}.joblib"

        joblib.dump(reg_model, model_path)

        print(
            f"Saved {name} | "
            f"RMSE: {rmse:.4f} | "
            f"R2: {r2:.4f}"
        )

    # ==========================================
    # PLOT REGRESSION ACTUAL VS PREDICTED
    # ==========================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, reg_model) in enumerate(
        fitted_regressors.items()
    ):

        y_reg_pred = reg_model.predict(X_test)

        r2 = regression_results[name]["r2"]
        rmse = regression_results[name]["rmse"]

        axes[idx].scatter(
            y_test_reg,
            y_reg_pred,
            alpha=0.3,
            color="#2b5c8f",
            s=15
        )

        min_val = min(
            y_test_reg.min(),
            y_reg_pred.min()
        )

        max_val = max(
            y_test_reg.max(),
            y_reg_pred.max()
        )

        axes[idx].plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Identity Line (y=x)"
        )

        axes[idx].set_title(
            f"{name}\nRMSE: {rmse:.3f}, R2: {r2:.4f}",
            fontsize=12,
            fontweight="bold"
        )

        axes[idx].set_xlabel("Actual DNA Damage Score")
        axes[idx].set_ylabel("Predicted DNA Damage Score")

        axes[idx].legend()

    plt.tight_layout()

    reg_plot_path = (
        FIGURES_DIR / "regression_evaluation.png"
    )

    plt.savefig(
        reg_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved regression plot: {reg_plot_path}")

    # ==========================================
    # SAVE METRICS REPORT
    # ==========================================

    classical_summary = {

        "classification": classification_results,

        "regression": regression_results,

        "class_names": class_names,

        "input_features": X_train.shape[1],

        "training_samples": len(X_train),

        "testing_samples": len(X_test)
    }

    summary_path = REPORTS_DIR / "classical_metrics.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            classical_summary,
            f,
            indent=2
        )

    print(f"\nSaved classical metrics: {summary_path}")

    print("\n========================================")
    print("CLASSICAL MODEL TRAINING COMPLETED")
    print("========================================")

    return classical_summary


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    train_and_evaluate_classical()
