"""
Plotly visualization components and styling utilities for the
Quantum Biological Dosimetry Streamlit Dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Theme Colors
DARK_BG = "#0E1117"
CARD_BG = "#161B22"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
TEXT_COLOR = "#E6EDF3"
MUTED_TEXT = "#8B949E"

CLASS_COLORS = {
    "High": "#EF5350",     # Red
    "Low": "#26A69A",      # Emerald Green
    "Moderate": "#FFA726", # Amber / Orange
    "Micro-Average": "#42A5F5" # Sky Blue
}


def apply_dark_layout(fig, title="", height=400, width=None):
    """Applies a consistent dark scientific aesthetic to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=15)),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color=TEXT_COLOR, family="sans-serif", size=12),
        height=height,
        width=width,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=MUTED_TEXT),
            title=dict(font=dict(color=TEXT_COLOR))
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=MUTED_TEXT),
            title=dict(font=dict(color=TEXT_COLOR))
        ),
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=11)
        )
    )
    return fig


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Interactive normalized and count confusion matrix heatmap."""
    cm = np.array(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    text_labels = [
        [f"<b>{int(cm[i, j])}</b><br>({cm_norm[i, j]*100:.1f}%)" for j in range(len(class_names))]
        for i in range(len(class_names))
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=class_names,
        y=class_names,
        text=text_labels,
        texttemplate="%{text}",
        textfont=dict(size=13, color=TEXT_COLOR),
        colorscale="Blues",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Accuracy Rate", tickformat=".0%"),
        hoverongaps=False
    ))

    fig.update_layout(
        xaxis=dict(title="Predicted Radiotoxicity Class"),
        yaxis=dict(title="True Radiotoxicity Class", autorange="reversed")
    )
    return apply_dark_layout(fig, title=title, height=380)


def plot_roc_curves(roc_data, title="Multiclass ROC Curves (One-vs-Rest)"):
    """Interactive One-vs-Rest ROC Curves."""
    fig = go.Figure()

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=MUTED_TEXT, dash="dash", width=1.5),
        name="Chance (AUC = 0.50)",
        hoverinfo="skip"
    ))

    for c_name, data in roc_data.items():
        color = CLASS_COLORS.get(c_name, "#AB47BC")
        dash = "dot" if c_name == "Micro-Average" else "solid"
        width = 2.5 if c_name == "Micro-Average" else 2.0
        
        fig.add_trace(go.Scatter(
            x=data["fpr"],
            y=data["tpr"],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=f"{c_name} (AUC = {data['auc']:.4f})",
            hovertemplate=f"<b>{c_name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>"
        ))

    fig.update_layout(
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate (Sensitivity)", range=[-0.02, 1.05])
    )
    return apply_dark_layout(fig, title=title, height=400)


def plot_pr_curves(pr_data, title="Multiclass Precision-Recall Curves"):
    """Interactive Precision-Recall Curves."""
    fig = go.Figure()

    for c_name, data in pr_data.items():
        color = CLASS_COLORS.get(c_name, "#42A5F5")
        fig.add_trace(go.Scatter(
            x=data["recall"],
            y=data["precision"],
            mode="lines",
            line=dict(color=color, width=2.0),
            name=f"{c_name} (AP = {data['ap']:.4f})",
            hovertemplate=f"<b>{c_name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>"
        ))

    fig.update_layout(
        xaxis=dict(title="Recall (Sensitivity)", range=[-0.02, 1.02]),
        yaxis=dict(title="Precision (PPV)", range=[-0.02, 1.05])
    )
    return apply_dark_layout(fig, title=title, height=400)


def plot_per_class_metrics(df_class, title="Per-Class Performance Comparison"):
    """Grouped bar chart for Precision, Recall, and F1-Score per class."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_class["Class"], y=df_class["Precision"],
        name="Precision", marker_color="#42A5F5"
    ))
    fig.add_trace(go.Bar(
        x=df_class["Class"], y=df_class["Recall"],
        name="Recall", marker_color="#66BB6A"
    ))
    fig.add_trace(go.Bar(
        x=df_class["Class"], y=df_class["F1-Score"],
        name="F1-Score", marker_color="#FFA726"
    ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.05], tickformat=".0%"),
        xaxis=dict(title="Target Class")
    )
    return apply_dark_layout(fig, title=title, height=380)


def plot_coefficients(coefs, feature_names, class_names, title="Multinomial Logistic Regression Coefficients"):
    """Bar chart of model coefficients across the 3 radiotoxicity classes."""
    coefs = np.array(coefs)
    fig = go.Figure()

    for i, c_name in enumerate(class_names):
        color = CLASS_COLORS.get(c_name, "#AB47BC")
        fig.add_trace(go.Bar(
            x=feature_names,
            y=coefs[i],
            name=f"Class: {c_name}",
            marker_color=color
        ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Physical Sensor Feature", tickangle=-25),
        yaxis=dict(title="Standardized Log-Odds Weight (β)")
    )
    return apply_dark_layout(fig, title=title, height=400)


def plot_feature_importance(importances, feature_names, title="Feature Importance"):
    """Horizontal sorted bar chart of feature importances."""
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation="h",
        marker=dict(
            color=imp_df["Importance"],
            colorscale="Viridis",
            showscale=False
        ),
        text=imp_df["Importance"].apply(lambda v: f"{v*100:.2f}%"),
        textposition="outside"
    ))

    fig.update_layout(
        xaxis=dict(title="Relative Importance Score", tickformat=".0%"),
        yaxis=dict(title="")
    )
    return apply_dark_layout(fig, title=title, height=380)


def plot_permutation_importance(perm_dict, feature_names, title="Permutation Feature Importance (Test Set)"):
    """Bar chart with error bars showing permutation importance mean and std."""
    means = np.array(perm_dict["importances_mean"])
    stds = np.array(perm_dict["importances_std"])
    
    p_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean": means,
        "Std": stds
    }).sort_values(by="Mean", ascending=True)

    fig = go.Figure(go.Bar(
        x=p_df["Mean"],
        y=p_df["Feature"],
        orientation="h",
        error_x=dict(type="data", array=p_df["Std"].tolist(), color="#E6EDF3"),
        marker_color="#29B6F6",
        text=p_df["Mean"].apply(lambda v: f"{v:.4f}"),
        textposition="outside"
    ))

    fig.update_layout(
        xaxis=dict(title="Mean Decrease in Accuracy"),
        yaxis=dict(title="")
    )
    return apply_dark_layout(fig, title=title, height=380)


def plot_pca_decision_boundary(pca_data, class_names, title="Classical SVM: 2D PCA Decision Boundary Projection"):
    """Contour plot showing SVM decision boundaries projected onto top 2 PCA components."""
    xx = np.array(pca_data["xx"])
    yy = np.array(pca_data["yy"])
    Z = np.array(pca_data["Z"])
    x_test = np.array(pca_data["pca1_test"])
    y_test = np.array(pca_data["pca2_test"])
    labels = np.array(pca_data["y_test_subset"])

    fig = go.Figure()

    # Decision regions
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        colorscale=[[0, "rgba(239, 83, 80, 0.3)"], [0.5, "rgba(38, 166, 154, 0.3)"], [1, "rgba(255, 167, 38, 0.3)"]],
        showscale=False,
        contours=dict(showlines=False),
        hoverinfo="skip"
    ))

    # Scatter of test samples
    for i, c_name in enumerate(class_names):
        mask = (labels == i)
        fig.add_trace(go.Scatter(
            x=x_test[mask],
            y=y_test[mask],
            mode="markers",
            name=f"Sample: {c_name}",
            marker=dict(size=6, color=CLASS_COLORS.get(c_name, "#AB47BC"), opacity=0.85),
            hovertemplate=f"<b>{c_name}</b><br>PCA 1: %{{x:.2f}}<br>PCA 2: %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        xaxis=dict(title="Principal Component 1 (Primary Variance Axis)"),
        yaxis=dict(title="Principal Component 2 (Secondary Variance Axis)")
    )
    return apply_dark_layout(fig, title=title, height=450)


def plot_vqc_training_curves(loss_hist, acc_hist, title="PennyLane VQC: 45-Epoch Optimization Trajectory"):
    """Dual-axis line plot showing loss and training accuracy over 45 epochs."""
    epochs = list(range(1, len(loss_hist) + 1))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=loss_hist,
        mode="lines+markers",
        name="Cross-Entropy Loss",
        line=dict(color="#EF5350", width=2.5),
        marker=dict(size=4)
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=acc_hist,
        mode="lines+markers",
        name="Training Accuracy",
        line=dict(color="#26A69A", width=2.5),
        marker=dict(size=4),
        yaxis="y2"
    ))

    fig.update_layout(
        xaxis=dict(title="Training Epoch", dtick=5),
        yaxis=dict(title=dict(text="Loss", font=dict(color="#EF5350")), tickfont=dict(color="#EF5350")),
        yaxis2=dict(
            title=dict(text="Training Accuracy", font=dict(color="#26A69A")),
            tickfont=dict(color="#26A69A"),
            anchor="x",
            overlaying="y",
            side="right",
            tickformat=".0%",
            range=[0.4, 0.8]
        ),
        legend=dict(x=0.02, y=0.98)
    )
    return apply_dark_layout(fig, title=title, height=400)


def plot_qsvm_kernel_heatmap(K_sample, title="Quantum Kernel Gram Matrix (Fidelity K_ij on 50x50 Subspace)"):
    """Interactive heatmap of quantum fidelity values."""
    K_arr = np.array(K_sample)
    fig = go.Figure(data=go.Heatmap(
        z=K_arr,
        colorscale="Viridis",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Fidelity |<φ_i|φ_j>|²", tickformat=".2f")
    ))
    fig.update_layout(
        xaxis=dict(title="Support Vector Index j"),
        yaxis=dict(title="Support Vector Index i", autorange="reversed")
    )
    return apply_dark_layout(fig, title=title, height=450)


def plot_qsvm_eigenvalues(eigenvalues, title="Quantum Kernel Eigenvalue Spectrum (Positive Semi-Definiteness)"):
    """Bar/line plot of eigenvalues verifying positive semi-definiteness."""
    eigs = np.array(eigenvalues)[:40]
    indices = list(range(1, len(eigs) + 1))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=indices,
        y=eigs,
        marker_color="#AB47BC",
        name="Eigenvalue λ_i"
    ))

    fig.update_layout(
        xaxis=dict(title="Eigenvalue Rank Index i"),
        yaxis=dict(title="Eigenvalue Magnitude (λ_i)", range=[0, max(eigs) * 1.1])
    )
    return apply_dark_layout(fig, title=title, height=380)


def plot_fidelity_distributions(same_class, diff_class, title="Quantum State Fidelity Distribution"):
    """Histogram comparing state fidelities between identical vs different classes."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=same_class,
        name="Same Class Pair",
        marker_color="#26A69A",
        opacity=0.75,
        nbinsx=30
    ))
    fig.add_trace(go.Histogram(
        x=diff_class,
        name="Different Class Pair",
        marker_color="#EF5350",
        opacity=0.75,
        nbinsx=30
    ))

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Quantum Fidelity |<φ_1|φ_2>|²", range=[0.0, 1.05]),
        yaxis=dict(title="Sample Pair Count")
    )
    return apply_dark_layout(fig, title=title, height=380)


def plot_svm_vs_qsvm_roc(svm_roc, qsvm_roc, class_names, title="Head-to-Head: Classical SVM vs Qiskit QSVM ROC Curves"):
    """ROC comparison comparing Classical SVM (solid) and QSVM (dashed) on the same plot."""
    fig = go.Figure()

    # Chance diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=MUTED_TEXT, dash="dot"),
        name="Chance",
        hoverinfo="skip"
    ))

    for c_name in class_names:
        color = CLASS_COLORS.get(c_name, "#42A5F5")
        
        # Classical SVM
        if c_name in svm_roc:
            fig.add_trace(go.Scatter(
                x=svm_roc[c_name]["fpr"],
                y=svm_roc[c_name]["tpr"],
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"Classical SVM: {c_name} (AUC={svm_roc[c_name]['auc']:.4f})"
            ))

        # QSVM
        if c_name in qsvm_roc:
            fig.add_trace(go.Scatter(
                x=qsvm_roc[c_name]["fpr"],
                y=qsvm_roc[c_name]["tpr"],
                mode="lines",
                line=dict(color=color, width=2.0, dash="dash"),
                name=f"Qiskit QSVM: {c_name} (AUC={qsvm_roc[c_name]['auc']:.4f})"
            ))

    fig.update_layout(
        xaxis=dict(title="False Positive Rate", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate", range=[-0.02, 1.05])
    )
    return apply_dark_layout(fig, title=title, height=450)


def plot_kernel_comparison_distributions(rbf_vals, quantum_vals, title="Kernel Similarity Distribution: Classical RBF vs Quantum Fidelity"):
    """Compares distribution of kernel similarity values between classical RBF and QSVM."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rbf_vals,
        name="Classical RBF Kernel: exp(-γ||x1-x2||²)",
        marker_color="#42A5F5",
        opacity=0.7,
        nbinsx=35
    ))
    fig.add_trace(go.Histogram(
        x=quantum_vals,
        name="Quantum Fidelity Kernel: |<φ(x1)|φ(x2)>|²",
        marker_color="#AB47BC",
        opacity=0.7,
        nbinsx=35
    ))

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Kernel Value K(x1, x2)", range=[0.0, 1.05]),
        yaxis=dict(title="Pairwise Frequency")
    )
    return apply_dark_layout(fig, title=title, height=400)
