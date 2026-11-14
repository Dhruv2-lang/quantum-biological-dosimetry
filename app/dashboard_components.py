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
GRID_COLOR = "rgba(128, 128, 128, 0.22)"
TEXT_COLOR = "#E6EDF3"
MUTED_TEXT = "#8B949E"

CLASS_COLORS = {
    "High": "#EF5350",     # Red
    "Low": "#26A69A",      # Emerald Green
    "Moderate": "#FFA726", # Amber / Orange
    "Micro-Average": "#42A5F5" # Sky Blue
}


def apply_dark_layout(fig, title="", height=400, width=None):
    """Applies a consistent scientific look to any Plotly figure.

    Despite the historical name this is THEME-NEUTRAL: text colour is deliberately left unset so
    Streamlit's plotly theme (st.plotly_chart default) paints it in the active light/dark theme.
    Grid lines and muted ticks use mid-grey/alpha values that read well on both.
    """
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family="sans-serif", size=12),
        height=height,
        width=width,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=MUTED_TEXT)
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=MUTED_TEXT)
        ),
        legend=dict(
            bgcolor="rgba(128, 128, 128, 0.10)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(size=11)
        )
    )
    return fig


def reorder_square(matrix, class_names, order):
    """Reorders rows AND columns of a square matrix from `class_names` order into `order`."""
    idx = [list(class_names).index(c) for c in order]
    m = np.array(matrix)
    return m[np.ix_(idx, idx)]


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", height=380, show_colorbar=True):
    """Interactive confusion matrix. Colour = row-normalised recall, label = count + recall %."""
    cm = np.array(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    text_labels = [
        [f"<b>{int(cm[i, j])}</b><br>{cm_norm[i, j]*100:.1f}%" for j in range(len(class_names))]
        for i in range(len(class_names))
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=list(class_names),
        y=list(class_names),
        text=text_labels,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#FFFFFF"),
        # Starts at the card colour (not white) so white labels stay readable on every cell.
        colorscale=[[0.0, "#1B2330"], [0.15, "#1F3B5C"], [1.0, "#2F80ED"]],
        zmin=0.0,
        zmax=1.0,
        showscale=show_colorbar,
        colorbar=dict(title="Row share", tickformat=".0%"),
        xgap=3, ygap=3,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Share of true class: %{z:.1%}<extra></extra>"
    ))

    fig.update_layout(
        xaxis=dict(title="Predicted class", side="bottom"),
        yaxis=dict(title="True class", autorange="reversed")
    )
    return apply_dark_layout(fig, title=title, height=height)


def plot_confusion_grid(models, class_names_by_model, order, ncols=3, title="Confusion Matrices - all models"):
    """Small-multiples grid of row-normalised confusion matrices (one per model)."""
    from plotly.subplots import make_subplots

    names = list(models.keys())
    nrows = int(np.ceil(len(names) / ncols))
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[
            f"{n}<br><sup>acc {models[n]['metrics']['accuracy']*100:.1f}%</sup>" for n in names
        ],
        horizontal_spacing=0.07, vertical_spacing=0.16
    )
    for k, n in enumerate(names):
        r, c = divmod(k, ncols)
        cm = reorder_square(models[n]["confusion_matrix"], class_names_by_model, order)
        norm = cm / cm.sum(axis=1)[:, None]
        text = [[f"<b>{int(cm[i, j])}</b>" for j in range(len(order))] for i in range(len(order))]
        fig.add_trace(go.Heatmap(
            z=norm, x=list(order), y=list(order), text=text, texttemplate="%{text}",
            textfont=dict(size=12, color="#FFFFFF"),
            colorscale=[[0.0, "#1B2330"], [0.15, "#1F3B5C"], [1.0, "#2F80ED"]],
            zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Share: %{z:.1%}<extra>" + n + "</extra>"
        ), row=r + 1, col=c + 1)
        fig.update_yaxes(autorange="reversed", row=r + 1, col=c + 1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12), height=330 * nrows + 40,
        margin=dict(l=40, r=20, t=70, b=30),
        title=dict(text=title, font=dict(size=15))
    )
    fig.update_annotations(font=dict(size=12))
    return fig


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
        error_x=dict(type="data", array=p_df["Std"].tolist(), color="#8B949E"),
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
    """Crisp decision regions of a 2-D PCA *proxy* RBF-SVM with test samples overlaid."""
    xx = np.array(pca_data["xx"])
    yy = np.array(pca_data["yy"])
    Z = np.array(pca_data["Z"])
    x_test = np.array(pca_data["pca1_test"])
    y_test = np.array(pca_data["pca2_test"])
    labels = np.array(pca_data["y_test_subset"])

    cols = [CLASS_COLORS.get(c, "#AB47BC") for c in class_names]
    fills = ["rgba(%d,%d,%d,0.28)" % tuple(int(col[i:i + 2], 16) for i in (1, 3, 5)) for col in cols]
    third = 1.0 / len(class_names)
    scale = []
    for k, f in enumerate(fills):
        scale += [[k * third, f], [(k + 1) * third, f]]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xx[0], y=yy[:, 0], z=Z, zmin=-0.5, zmax=len(class_names) - 0.5,
        colorscale=scale, showscale=False, hoverinfo="skip", name="Decision region"
    ))
    for i, c_name in enumerate(class_names):
        mask = (labels == i)
        fig.add_trace(go.Scatter(
            x=x_test[mask], y=y_test[mask], mode="markers", name=f"Test sample: {c_name}",
            marker=dict(size=6, color=cols[i], opacity=0.9, line=dict(width=0.5, color="#0E1117")),
            hovertemplate=f"<b>{c_name}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        xaxis=dict(title="Principal component 1"),
        yaxis=dict(title="Principal component 2")
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
    )
    fig = apply_dark_layout(fig, title=title, height=420)
    fig.update_layout(legend=dict(orientation="h", x=0, y=1.12))
    return fig


def plot_qsvm_kernel_heatmap(K_sample, title="Quantum Kernel Gram Matrix (fidelity K_ij, first 50 training samples)"):
    """Interactive heatmap of quantum fidelity values."""
    K_arr = np.array(K_sample)
    fig = go.Figure(data=go.Heatmap(
        z=K_arr,
        colorscale="Viridis",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title=dict(text="Fidelity", side="right"), tickformat=".2f", thickness=14)
    ))
    fig.update_layout(
        xaxis=dict(title="Training sample j"),
        yaxis=dict(title="Training sample i", autorange="reversed")
    )
    return apply_dark_layout(fig, title=title, height=450)


def plot_qsvm_eigenvalues(eigenvalues, title="Quantum kernel eigenvalue spectrum (100x100 Gram matrix)"):
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


# ---------------------------------------------------------------------------
# Per-model & cross-model charts
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "Logistic Regression": "#5C9DFF",
    "Random Forest": "#7BC47F",
    "XGBoost": "#E8B04B",
    "Classical SVM": "#9B8AFB",
    "PennyLane 4-Qubit VQC": "#F0719B",
    "Qiskit 4-Qubit QSVM": "#2DD4BF",
}


def plot_metric_scorecard(metrics, color="#42A5F5", title="Metric scorecard"):
    """Horizontal bar scorecard of the six headline quality metrics (0-1 scale)."""
    labels = ["Accuracy", "Macro precision", "Macro recall", "Macro F1", "Weighted F1", "ROC-AUC (OvR)"]
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "roc_auc"]
    vals = [float(metrics[k]) for k in keys]
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=labels[::-1], orientation="h", marker_color=color,
        text=[f"{v:.4f}" for v in vals[::-1]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(xaxis=dict(title="Score", range=[0, 1.12], tickformat=".1f"), yaxis=dict(title=""))
    return apply_dark_layout(fig, title=title, height=380)


def plot_support_vectors(n_support, class_names, n_train, title="Support vectors per class"):
    vals = list(n_support)
    fig = go.Figure(go.Bar(
        x=list(class_names), y=vals,
        marker_color=[CLASS_COLORS.get(c, "#AB47BC") for c in class_names],
        text=[f"{v}<br>({v / n_train * 100:.1f}% of train)" for v in vals], textposition="outside",
        cliponaxis=False
    ))
    fig.update_layout(xaxis=dict(title="Class"), yaxis=dict(title="Support vectors", range=[0, max(vals) * 1.25]))
    return apply_dark_layout(fig, title=title, height=380)


def plot_metric_comparison(models, metric_keys, metric_labels, title="Metric comparison"):
    """Grouped bars: one group per metric, one bar per model."""
    fig = go.Figure()
    for name, m in models.items():
        fig.add_trace(go.Bar(
            x=metric_labels, y=[float(m["metrics"][k]) for k in metric_keys], name=name,
            marker_color=MODEL_COLORS.get(name, "#AB47BC"),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>"
        ))
    fig.update_layout(barmode="group", yaxis=dict(title="Score (axis starts at 0.5)", range=[0.5, 1.02]), xaxis=dict(title=""))
    return apply_dark_layout(fig, title=title, height=430)


def plot_class_metric_heatmap(models, order, metric="F1-Score", title="Per-class F1 by model"):
    """Models x classes heatmap of a per-class metric."""
    names = list(models.keys())
    z, text = [], []
    for n in names:
        df = models[n]["class_perf_df"].set_index("Class")
        row = [float(df.loc[c, metric]) for c in order]
        z.append(row)
        text.append([f"{v:.3f}" for v in row])
    fig = go.Figure(go.Heatmap(
        z=z, x=list(order), y=names, text=text, texttemplate="%{text}",
        textfont=dict(size=13, color="#FFFFFF"), zmin=0, zmax=1, xgap=3, ygap=3,
        colorscale=[[0.0, "#7A1F2B"], [0.5, "#4A3B1F"], [0.8, "#1F3B5C"], [1.0, "#2F80ED"]],
        colorbar=dict(thickness=12, tickformat=".1f"),
        hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(xaxis=dict(title="Class"), yaxis=dict(title="", autorange="reversed"))
    return apply_dark_layout(fig, title=title, height=380)


def plot_latency(models, title="Training vs inference time (log scale)"):
    names = list(models.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=[max(models[n]["metrics"]["training_time"], 1e-4) for n in names],
        name="Training time (s)", marker_color="#5C9DFF",
        text=[f"{models[n]['metrics']['training_time']:.3f}s" for n in names], textposition="outside"
    ))
    fig.add_trace(go.Bar(
        x=names, y=[max(models[n]["metrics"]["inference_time"], 1e-4) for n in names],
        name="Inference on 1,000 samples (s)", marker_color="#E8B04B",
        text=[f"{models[n]['metrics']['inference_time']:.4f}s" for n in names], textposition="outside"
    ))
    fig.update_layout(barmode="group", yaxis=dict(title="Seconds (log)", type="log"), xaxis=dict(title="", tickangle=-15))
    return apply_dark_layout(fig, title=title, height=420)


def plot_macro_roc_overlay(models, title="Micro-average ROC - all models"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED_TEXT, dash="dash"),
                             name="Chance", hoverinfo="skip"))
    for n, m in models.items():
        d = m["roc_curves"]["Micro-Average"]
        fig.add_trace(go.Scatter(
            x=d["fpr"], y=d["tpr"], mode="lines", line=dict(color=MODEL_COLORS.get(n, "#AB47BC"), width=2.2),
            name=f"{n} (AUC {d['auc']:.4f})",
            hovertemplate=f"<b>{n}</b><br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}<extra></extra>"
        ))
    fig.update_layout(xaxis=dict(title="False positive rate", range=[-0.01, 1.01]),
                      yaxis=dict(title="True positive rate", range=[0, 1.03]))
    return apply_dark_layout(fig, title=title, height=430)


def plot_radar(models, title="Quality profile"):
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig = go.Figure()
    for n, m in models.items():
        vals = [float(m["metrics"][k]) for k in keys]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=labels + labels[:1], name=n, fill="none",
            line=dict(color=MODEL_COLORS.get(n, "#AB47BC"), width=2.5),
            marker=dict(size=5)
        ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0.5, 1.0], gridcolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT)),
                   angularaxis=dict(gridcolor=GRID_COLOR)),
        paper_bgcolor="rgba(0,0,0,0)", height=430,
        margin=dict(l=40, r=40, t=50, b=30),
        title=dict(text=title, font=dict(size=15)),
        legend=dict(bgcolor="rgba(128,128,128,0.10)", font=dict(size=11))
    )
    return fig
