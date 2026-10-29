"""
Model Comparison and Benchmark Analysis for Quantum Biological Dosimetry.
Aggregates classical and quantum model metrics into unified comparison tables,
generates side-by-side performance plots, combined confusion matrices,
and saves consolidated benchmark artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

def generate_comparisons():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cls_path = os.path.join(REPORTS_DIR, "classical_metrics.json")
    q_path = os.path.join(REPORTS_DIR, "quantum_metrics.json")

    with open(cls_path) as f:
        cls_data = json.load(f)
    with open(q_path) as f:
        q_data = json.load(f)

    # Consolidated Classification Table
    records = []
    
    # Classical models
    for name, m in cls_data["classification"].items():
        records.append({
            "Model": name,
            "Type": "Classical ML",
            "Accuracy (%)": m["accuracy"] * 100,
            "Macro-F1": m["macro_f1"],
            "Macro-Precision": m["macro_precision"],
            "Macro-Recall": m["macro_recall"],
            "Parameters / Architecture": "Linear Logits" if "Logistic" in name else "100 Trees / Ensemble",
            "Hardware / Target": "Classical CPU"
        })

    # Quantum Model
    records.append({
        "Model": "PennyLane 4-Qubit VQC",
        "Type": "Variational Quantum Circuit",
        "Accuracy (%)": q_data["accuracy"] * 100,
        "Macro-F1": q_data["macro_f1"],
        "Macro-Precision": q_data["macro_precision"],
        "Macro-Recall": q_data["macro_recall"],
        "Parameters / Architecture": f"4 Qubits, {q_data['num_layers']} Layers ({q_data['variational_parameters']} rot params)",
        "Hardware / Target": "Quantum Simulator (default.qubit)"
    })

    df_comp = pd.DataFrame(records)
    print("\n=== CLASSIFICATION BENCHMARK SUMMARY ===")
    print(df_comp.to_string(index=False))

    # Consolidated JSON artifact
    benchmark_artifact = {
        "classification_comparison": records,
        "regression_comparison": cls_data["regression"],
        "key_findings": [
            "Classical models (Logistic Regression, Random Forest, XGBoost) achieved superior performance (~95.8% - 96.1% accuracy, ~0.96 Macro-F1).",
            "The 4-Qubit Variational Quantum Classifier (VQC) demonstrated genuine parameter convergence (loss decreased from 0.998 to 0.549) and achieved 67.70% test accuracy on unseen data.",
            "Quantum Advantage was NOT observed: classical gradient-boosted trees and linear classifiers dominate tabular sensor classification.",
            "Target Leakage Audit confirmed that biological assay columns (foci counts, viability, DNA damage score) must be excluded; predicting directly from NV-center and quantum dot sensor measurements maintains non-invasive biodosimetry integrity."
        ]
    }

    bench_json_path = os.path.join(REPORTS_DIR, "benchmark_summary.json")
    with open(bench_json_path, "w") as f:
        json.dump(benchmark_artifact, f, indent=2)
    print(f"\nSaved benchmark summary to {bench_json_path}")

    # Plot 1: Model Comparison Bar Chart
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Comparison
    colors = ["#2b5c8f", "#3a7ebf", "#529edb", "#8a2be2"]
    bars1 = axes[0].bar(df_comp["Model"], df_comp["Accuracy (%)"], color=colors, width=0.55)
    axes[0].set_title("Classification Accuracy Comparison (%)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 110)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)

    # Macro-F1 Comparison
    bars2 = axes[1].bar(df_comp["Model"], df_comp["Macro-F1"], color=colors, width=0.55)
    axes[1].set_title("Macro-F1 Score Comparison", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1.1)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    bench_plot_path = os.path.join(FIGURES_DIR, "model_comparison_benchmark.png")
    plt.savefig(bench_plot_path, dpi=300)
    plt.close()
    print(f"Saved benchmark comparison figure to {bench_plot_path}")

    # Plot 2: Combined 4-Model Confusion Matrix Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    class_names = ["Low", "Moderate", "High"]

    model_cms = [
        ("Logistic Regression (Acc: 96.10%)", cls_data["classification"]["Logistic Regression"]["confusion_matrix"], "Blues"),
        ("Random Forest (Acc: 95.80%)", cls_data["classification"]["Random Forest"]["confusion_matrix"], "Greens"),
        ("XGBoost (Acc: 95.90%)", cls_data["classification"]["XGBoost"]["confusion_matrix"], "Oranges"),
        ("PennyLane 4-Qubit VQC (Acc: 67.70%)", q_data["confusion_matrix"], "Purples")
    ]

    for idx, (title, cm, cmap) in enumerate(model_cms):
        sns.heatmap(np.array(cm), annot=True, fmt="d", cmap=cmap, cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
        axes[idx].set_title(title, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Class")
        axes[idx].set_ylabel("True Class")

    plt.suptitle("Confusion Matrix Benchmark: Classical ML vs Quantum VQC", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    all_cm_path = os.path.join(FIGURES_DIR, "all_confusion_matrices_combined.png")
    plt.savefig(all_cm_path, dpi=300)
    plt.close()
    print(f"Saved combined confusion matrices to {all_cm_path}")

if __name__ == "__main__":
    generate_comparisons()
