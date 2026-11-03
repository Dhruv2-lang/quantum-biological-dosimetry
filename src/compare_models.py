
"""
Model Comparison and Benchmark Analysis for Quantum Biological Dosimetry.

Aggregates classical and quantum model metrics into unified comparison tables,
generates side-by-side performance plots, combined confusion matrices,
and saves consolidated benchmark artifacts.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

CLASSICAL_METRICS_PATH = REPORTS_DIR / "classical_metrics.json"
QUANTUM_METRICS_PATH = REPORTS_DIR / "quantum_metrics.json"
SPLITS_PATH = MODELS_DIR / "data_splits.joblib"


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the corresponding model training script first."
        )

    with open(path, "r") as file:
        return json.load(file)


def get_class_names():
    """
    Recover the original class encoding used during preprocessing.
    LabelEncoder produces alphabetical order:
    High=0, Low=1, Moderate=2.
    """

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {SPLITS_PATH}\n"
            "Run src/preprocessing.py first."
        )

    splits = joblib.load(SPLITS_PATH)

    if "class_names" in splits:
        return list(splits["class_names"])

    # Fallback to the confirmed LabelEncoder ordering.
    return ["High", "Low", "Moderate"]


# --------------------------------------------------
# 3. GENERATE COMPARISONS
# --------------------------------------------------

def generate_comparisons():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cls_data = load_json(CLASSICAL_METRICS_PATH)
    q_data = load_json(QUANTUM_METRICS_PATH)

    class_names = get_class_names()

    if len(class_names) != 3:
        raise ValueError(
            f"Expected 3 classes, but found {len(class_names)}."
        )

    # --------------------------------------------------
    # CLASSIFICATION COMPARISON TABLE
    # --------------------------------------------------

    records = []

    for name, metrics in cls_data["classification"].items():

        if name == "Logistic Regression":
            architecture = "Linear Classifier"
        elif name == "Random Forest":
            architecture = "Random Forest Ensemble"
        elif name == "XGBoost":
            architecture = "Gradient Boosted Trees"
        else:
            architecture = "Classical ML Model"

        records.append({
            "Model": name,
            "Type": "Classical ML",
            "Accuracy (%)": metrics["accuracy"] * 100,
            "Macro-F1": metrics["macro_f1"],
            "Macro-Precision": metrics["macro_precision"],
            "Macro-Recall": metrics["macro_recall"],
            "Parameters / Architecture": architecture,
            "Hardware / Target": "Classical CPU"
        })

    # Quantum model

    records.append({
        "Model": "PennyLane 4-Qubit VQC",
        "Type": "Variational Quantum Circuit",
        "Accuracy (%)": q_data["accuracy"] * 100,
        "Macro-F1": q_data["macro_f1"],
        "Macro-Precision": q_data["macro_precision"],
        "Macro-Recall": q_data["macro_recall"],
        "Parameters / Architecture": (
            f"4 Qubits, {q_data['num_layers']} Layers "
            f"({q_data['variational_parameters']} variational parameters)"
        ),
        "Hardware / Target": "Quantum Simulator (default.qubit)"
    })

    df_comp = pd.DataFrame(records)

    print("\n" + "=" * 65)
    print("CLASSIFICATION BENCHMARK SUMMARY")
    print("=" * 65)
    print(df_comp.round(4).to_string(index=False))

    # --------------------------------------------------
    # REGRESSION COMPARISON
    # --------------------------------------------------

    regression_records = []

    for name, metrics in cls_data["regression"].items():
        regression_records.append({
            "Model": name,
            **metrics
        })

    df_regression = pd.DataFrame(regression_records)

    print("\n" + "=" * 65)
    print("REGRESSION BENCHMARK SUMMARY")
    print("=" * 65)
    print(df_regression.to_string(index=False))

    # --------------------------------------------------
    # KEY FINDINGS
    # --------------------------------------------------

    best_model = max(
        records,
        key=lambda item: item["Accuracy (%)"]
    )

    classical_best = max(
        [r for r in records if r["Type"] == "Classical ML"],
        key=lambda item: item["Accuracy (%)"]
    )

    quantum_record = next(
        r for r in records
        if r["Type"] == "Variational Quantum Circuit"
    )

    accuracy_difference = (
        classical_best["Accuracy (%)"]
        - quantum_record["Accuracy (%)"]
    )

    findings = [
        (
            f"{best_model['Model']} achieved the highest classification "
            f"accuracy at {best_model['Accuracy (%)']:.2f}%."
        ),
        (
            f"The PennyLane 4-Qubit VQC achieved "
            f"{quantum_record['Accuracy (%)']:.2f}% accuracy "
            f"and a Macro-F1 score of {quantum_record['Macro-F1']:.4f}."
        ),
        (
            f"The best classical model achieved "
            f"{classical_best['Accuracy (%)']:.2f}% accuracy."
        ),
        (
            f"The accuracy difference between the best classical model "
            f"and the quantum VQC is {accuracy_difference:.2f} "
            f"percentage points."
        ),
        (
            "These results do not demonstrate quantum advantage. "
            "The quantum model was evaluated using a classical "
            "quantum-circuit simulator."
        ),
        (
            "Biological assay columns and DNA damage score must remain "
            "excluded from classifier inputs to avoid target leakage."
        )
    ]

    # --------------------------------------------------
    # SAVE CONSOLIDATED JSON
    # --------------------------------------------------

    benchmark_artifact = {
        "class_names": class_names,
        "classification_comparison": records,
        "regression_comparison": cls_data["regression"],
        "key_findings": findings
    }

    bench_json_path = REPORTS_DIR / "benchmark_summary.json"

    with open(bench_json_path, "w") as file:
        json.dump(benchmark_artifact, file, indent=2)

    print(f"\nSaved benchmark summary to {bench_json_path}")

    # --------------------------------------------------
    # PLOT 1: ACCURACY AND MACRO-F1 COMPARISON
    # --------------------------------------------------

    model_names = df_comp["Model"].tolist()
    positions = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = plt.cm.tab10(
        np.linspace(0, 1, len(model_names))
    )

    # Accuracy

    bars1 = axes[0].bar(
        positions,
        df_comp["Accuracy (%)"],
        color=colors,
        width=0.6
    )

    axes[0].set_title(
        "Classification Accuracy Comparison",
        fontsize=13,
        fontweight="bold"
    )
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 110)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(model_names, rotation=20, ha="right")

    for bar in bars1:
        value = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontweight="bold"
        )

    # Macro-F1

    bars2 = axes[1].bar(
        positions,
        df_comp["Macro-F1"],
        color=colors,
        width=0.6
    )

    axes[1].set_title(
        "Macro-F1 Score Comparison",
        fontsize=13,
        fontweight="bold"
    )
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(model_names, rotation=20, ha="right")

    for bar in bars2:
        value = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            fontweight="bold"
        )

    plt.tight_layout()

    bench_plot_path = FIGURES_DIR / "model_comparison_benchmark.png"

    plt.savefig(bench_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved benchmark comparison figure to {bench_plot_path}")

    # --------------------------------------------------
    # PLOT 2: COMBINED CONFUSION MATRICES
    # --------------------------------------------------

    model_cms = []

    for name, metrics in cls_data["classification"].items():
        model_cms.append((
            name,
            metrics["confusion_matrix"]
        ))

    model_cms.append((
        "PennyLane 4-Qubit VQC",
        q_data["confusion_matrix"]
    ))

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 11)
    )

    axes = axes.flatten()

    for idx, (name, cm) in enumerate(model_cms):

        if idx >= len(axes):
            break

        cm = np.asarray(cm)

        if cm.shape != (len(class_names), len(class_names)):
            raise ValueError(
                f"Invalid confusion matrix shape for {name}: {cm.shape}"
            )

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
            name,
            fontsize=12,
            fontweight="bold"
        )
        axes[idx].set_xlabel("Predicted Class")
        axes[idx].set_ylabel("True Class")

    for idx in range(len(model_cms), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "Confusion Matrix Benchmark: Classical ML vs Quantum VQC",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()

    all_cm_path = FIGURES_DIR / "all_confusion_matrices_combined.png"

    plt.savefig(all_cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined confusion matrices to {all_cm_path}")

    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nGenerated files:")
    print(f"1. {bench_json_path}")
    print(f"2. {bench_plot_path}")
    print(f"3. {all_cm_path}")


if __name__ == "__main__":
    generate_comparisons()
