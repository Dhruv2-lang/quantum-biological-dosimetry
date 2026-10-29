"""
Exploratory Data Analysis and Target Leakage Audit for Quantum Biological Dosimetry.
Validates the 5,000-row tabular dataset, inspects sensor vs biological features,
analyzes target leakage, and generates audit reports and figures.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/data/Quantum_Biological_Dosimetry_Dataset_5000.csv"
REPORTS_DIR = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

SENSOR_FEATURES = [
    "Dose_Gy",
    "ODMR_Shift_MHz",
    "T1_us",
    "T2_us",
    "Fluorescence_Intensity",
    "Fluorescence_Lifetime_ns",
    "QD_Emission_nm"
]

BIOLOGICAL_MEASUREMENTS = [
    "ROS_Level",
    "Mitochondrial_Damage",
    "GammaH2AX_Foci",
    "BP53_Foci",
    "Cell_Viability"
]

REGRESSION_TARGET = "DNA_Damage_Score"
CLASSIFICATION_TARGET = "Radiotoxicity_Class"

def run_eda():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Basic validation
    missing = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated().sum())
    dtypes = {k: str(v) for k, v in df.dtypes.items()}

    # 2. Target distributions
    class_dist = df[CLASSIFICATION_TARGET].value_counts().to_dict()
    class_pct = df[CLASSIFICATION_TARGET].value_counts(normalize=True).mul(100).round(2).to_dict()

    dna_damage_stats = df[REGRESSION_TARGET].describe().to_dict()

    # 3. Target Leakage Analysis
    # Check DNA damage score by class
    dna_by_class = {}
    for cls in df[CLASSIFICATION_TARGET].unique():
        sub = df[df[CLASSIFICATION_TARGET] == cls][REGRESSION_TARGET]
        dna_by_class[cls] = {
            "min": float(sub.min()),
            "max": float(sub.max()),
            "mean": float(sub.mean()),
            "std": float(sub.std())
        }

    # Check threshold separation between classes
    class_order = ["Low", "Moderate", "High"]
    available_classes = [c for c in class_order if c in dna_by_class]
    
    # 4. Correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().round(4)

    sensor_corr_with_targets = {}
    for sf in SENSOR_FEATURES:
        sensor_corr_with_targets[sf] = {
            "corr_with_DNA_Damage": float(corr_matrix.loc[sf, REGRESSION_TARGET]),
            "corr_with_Dose": float(corr_matrix.loc[sf, "Dose_Gy"])
        }

    # Biological assays correlation with DNA Damage Score
    bio_corr_with_targets = {}
    for bf in BIOLOGICAL_MEASUREMENTS:
        bio_corr_with_targets[bf] = float(corr_matrix.loc[bf, REGRESSION_TARGET])

    summary = {
        "dataset_rows": len(df),
        "dataset_cols": df.shape[1],
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "classification_target": CLASSIFICATION_TARGET,
        "class_distribution": class_dist,
        "class_percentage": class_pct,
        "regression_target": REGRESSION_TARGET,
        "dna_damage_statistics": {k: round(v, 4) for k, v in dna_damage_stats.items()},
        "dna_damage_by_radiotoxicity_class": dna_by_class,
        "sensor_features": SENSOR_FEATURES,
        "biological_measurements": BIOLOGICAL_MEASUREMENTS,
        "sensor_target_correlations": sensor_corr_with_targets,
        "biological_target_correlations": bio_corr_with_targets,
        "target_leakage_warning": (
            "Biological measurements (ROS, Mitochondrial_Damage, GammaH2AX_Foci, BP53_Foci, "
            "Cell_Viability, DNA_Damage_Score) reflect cellular end-state assays and are nearly "
            "deterministic transformations or direct definition metrics of Radiotoxicity_Class. "
            "Using them to predict Radiotoxicity_Class or DNA_Damage_Score causes severe Target Leakage. "
            "Valid clinical/nanosensing models MUST predict damage strictly from quantum sensor features."
        )
    }

    summary_file = os.path.join(REPORTS_DIR, "eda_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_file}")

    # --- Plotting ---
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Target Distributions & Class Separation (Leakage Visualizer)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Class count
    sns.countplot(data=df, x=CLASSIFICATION_TARGET, order=["Low", "Moderate", "High"], palette="viridis", ax=axes[0])
    axes[0].set_title("Radiotoxicity Class Distribution (N=5,000)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f"{int(height)} ({height/len(df)*100:.1f}%)",
                         (p.get_x() + p.get_width() / 2., height / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    # DNA Damage Score vs Radiotoxicity Class
    sns.boxplot(data=df, x=CLASSIFICATION_TARGET, y=REGRESSION_TARGET, order=["Low", "Moderate", "High"], palette="Set2", ax=axes[1])
    axes[1].set_title("Target Leakage Audit: DNA Damage Score by Class", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("DNA Damage Score (0-100)")
    plt.tight_layout()
    leakage_fig_path = os.path.join(FIGURES_DIR, "eda_leakage_audit.png")
    plt.savefig(leakage_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {leakage_fig_path}")

    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, mask=mask, linewidths=0.5)
    plt.title("Correlation Matrix: Quantum Sensors vs Biological Damage Assays", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    corr_fig_path = os.path.join(FIGURES_DIR, "eda_correlation_matrix.png")
    plt.savefig(corr_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_fig_path}")

    # Plot 3: Quantum Sensor Distributions across Radiotoxicity Classes
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for idx, feature in enumerate(SENSOR_FEATURES):
        sns.kdeplot(data=df, x=feature, hue=CLASSIFICATION_TARGET, hue_order=["Low", "Moderate", "High"], 
                    common_norm=False, fill=True, alpha=0.3, ax=axes[idx], palette="viridis")
        axes[idx].set_title(f"{feature} Distribution", fontsize=11, fontweight="bold")
    
    # Hide 8th subplot
    fig.delaxes(axes[7])
    plt.suptitle("Quantum Sensor Feature Distributions by Radiotoxicity Class", fontsize=15, fontweight="bold")
    plt.tight_layout()
    sensor_fig_path = os.path.join(FIGURES_DIR, "eda_sensor_distributions.png")
    plt.savefig(sensor_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {sensor_fig_path}")

    return summary

if __name__ == "__main__":
    run_eda()
