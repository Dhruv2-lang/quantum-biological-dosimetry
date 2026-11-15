"""
Model Explorer page for the Quantum Biological Dosimetry dashboard.

Gives every classification model its own page with:
  - headline metrics (accuracy, precision, recall, F1, ROC-AUC, generalization error, latency)
  - confusion matrix, per-class precision/recall/F1, ROC and precision-recall curves
  - model-specific internals (coefficients, importances, support vectors, quantum kernel, VQC training ...)
  - a "reported vs re-evaluated" audit and CSV/JSON export
plus a "Compare all models" view.

All numbers come from reports/model_analysis_cache.joblib (built by src/generate_analysis_cache.py),
so this page needs no model loading and no quantum libraries.
"""

import os
import json
import inspect

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import dashboard_components as dc

# Natural severity order used for DISPLAY only. Cached arrays follow the encoder order
# (cache["class_names"] == ["High", "Low", "Moderate"]).
DISPLAY_ORDER = ["Low", "Moderate", "High"]
N_TRAIN = 4000  # 5,000 samples, 80/20 stratified split -> 4,000 train / 1,000 test
COMPARE_VIEW = "⚖️ Compare all models"

MODEL_BLURBS = {
    "Logistic Regression": dict(
        icon="📏",
        summary="Linear multinomial model: one weight per sensor feature per class, turned into probabilities with a softmax.",
        inputs="7 standardized sensor features",
    ),
    "Random Forest": dict(
        icon="🌲",
        summary="Ensemble of 100 decision trees (max depth 12) voting on the class; each tree sees a bootstrap sample.",
        inputs="7 standardized sensor features",
    ),
    "XGBoost": dict(
        icon="🚀",
        summary="Gradient-boosted trees: 100 shallow trees added one at a time, each correcting the previous errors.",
        inputs="7 standardized sensor features",
    ),
    "Classical SVM": dict(
        icon="🎯",
        summary="RBF-kernel support vector machine (C=5, gamma=0.05) that separates classes with a maximum-margin boundary.",
        inputs="7 standardized sensor features",
    ),
    "PennyLane 4-Qubit VQC": dict(
        icon="⚛️",
        summary="Variational quantum circuit: 4 PCA features are angle-encoded onto 4 qubits, passed through trainable "
                "StronglyEntanglingLayers, and read out as Pauli-Z expectation values.",
        inputs="4 PCA components (of the 7 scaled features)",
    ),
    "Qiskit 4-Qubit QSVM": dict(
        icon="🔮",
        summary="SVM whose kernel is the quantum state fidelity between ZZFeatureMap-encoded samples; the SVM "
                "optimisation itself is classical.",
        inputs="4 PCA components (of the 7 scaled features)",
    ),
}


# --------------------------------------------------------------------------- helpers
# Newer Streamlit replaced `use_container_width=True` with `width="stretch"`; older versions only know the former.
_NEW_WIDTH_API = "width" in inspect.signature(st.plotly_chart).parameters


def show_fig(fig, key):
    """Renders a Plotly figure across Streamlit versions."""
    cfg = {"displaylogo": False}
    if _NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key, config=cfg)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config=cfg)


def show_df(df, **kwargs):
    if _NEW_WIDTH_API:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


@st.cache_data(show_spinner=False)
def load_analysis_cache(path, mtime):
    """`mtime` is only part of the cache key so regenerating the cache file refreshes the page."""
    return joblib.load(path)


def _slug(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower())


def _reorder_class_df(df):
    return df.set_index("Class").loc[DISPLAY_ORDER].reset_index()


def _reorder_curves(curves):
    keys = [k for k in DISPLAY_ORDER if k in curves] + [k for k in curves if k not in DISPLAY_ORDER]
    return {k: curves[k] for k in keys}


def _top_confusions(cm, class_names, k=2):
    """Largest off-diagonal confusion cells as (true, predicted, count, share_of_true)."""
    cm = np.array(cm)
    rows = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                rows.append((class_names[i], class_names[j], int(cm[i, j]), cm[i, j] / cm[i].sum()))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows[:k]


def _best_values(models):
    """Best value of each metric across all models (max for quality, min for error/time)."""
    out = {}
    for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "roc_auc"]:
        out[key] = max(m["metrics"][key] for m in models.values())
    for key in ["generalization_error", "training_time", "inference_time"]:
        out[key] = min(m["metrics"][key] for m in models.values())
    return out


def _rank_by_accuracy(models, name):
    order = sorted(models, key=lambda n: (models[n]["metrics"]["accuracy"], models[n]["metrics"]["macro_f1"]),
                   reverse=True)
    return order.index(name) + 1


def _bench_lookup(bench_summary):
    return {m["Model"]: m for m in bench_summary.get("classification_comparison", [])}


# --------------------------------------------------------------------------- per-model sections
def _render_header(name, entry, bench_row):
    blurb = MODEL_BLURBS.get(name, {})
    st.markdown(f"## {blurb.get('icon', '🧪')} {name}")
    st.caption(f"{entry['category']}  ·  Input: {blurb.get('inputs', '-')}")
    if blurb.get("summary"):
        st.write(blurb["summary"])
    if bench_row:
        c1, c2 = st.columns(2)
        c1.markdown(f"**Architecture:** {bench_row.get('Parameters / Architecture', '-')}")
        c2.markdown(f"**Execution engine:** {bench_row.get('Execution Engine', '-')}")


def _kpi(col, label, value, delta=None):
    """st.metric with a plain-text delta (no misleading up/down arrow) on any Streamlit version."""
    try:
        col.metric(label, value, delta, delta_color="off", delta_arrow="off")
    except TypeError:
        col.metric(label, value, delta, delta_color="off")


def _render_kpis(name, models):
    m = models[name]["metrics"]
    best = _best_values(models)

    def vs_best(key, scale=100.0, unit=" pt", lower_is_better=False):
        diff = (m[key] - best[key]) * scale
        if abs(diff) < 1e-9:
            return "★ best of 6"
        return f"{diff:+.2f}{unit} vs best"

    st.caption("Held-out test set, N = 1,000 samples. Deltas compare against the best of the 6 models.")
    row1 = st.columns(6)
    _kpi(row1[0], "Accuracy", f"{m['accuracy'] * 100:.1f}%", vs_best("accuracy"))
    _kpi(row1[1], "Macro precision", f"{m['macro_precision']:.4f}", vs_best("macro_precision", 100))
    _kpi(row1[2], "Macro recall", f"{m['macro_recall']:.4f}", vs_best("macro_recall", 100))
    _kpi(row1[3], "Macro F1", f"{m['macro_f1']:.4f}", vs_best("macro_f1", 100))
    _kpi(row1[4], "Weighted F1", f"{m['weighted_f1']:.4f}", vs_best("weighted_f1", 100))
    _kpi(row1[5], "ROC-AUC (OvR)", f"{m['roc_auc']:.4f}", vs_best("roc_auc", 100))

    errors = int(round(m["generalization_error"] * 1000))
    row2 = st.columns(6)
    _kpi(row2[0], "Generalization error", f"{m['generalization_error'] * 100:.1f}%", vs_best("generalization_error", 100))
    _kpi(row2[1], "Training time", f"{m['training_time']:.3f} s")
    _kpi(row2[2], "Inference (1k samples)", f"{m['inference_time']:.4f} s")
    _kpi(row2[3], "Accuracy rank", f"#{_rank_by_accuracy(models, name)} of {len(models)}")
    _kpi(row2[4], "Correct (of 1,000)", f"{1000 - errors}")
    _kpi(row2[5], "Misclassified", f"{errors}")


def _render_drift_note(name, entry, bench_row):
    """Flags when the live re-evaluation differs from what benchmark_summary.json reported."""
    if not bench_row:
        return
    m = entry["metrics"]
    diff_acc = abs(m["accuracy"] * 100 - bench_row["Accuracy (%)"])
    diff_auc = abs(m["roc_auc"] - bench_row["ROC-AUC"])
    if diff_acc >= 0.05 or diff_auc >= 0.0005:
        st.warning(
            f"**Numbers differ from the benchmark report.** `benchmark_summary.json` lists "
            f"{bench_row['Accuracy (%)']:.1f}% accuracy / {bench_row['ROC-AUC']:.4f} ROC-AUC, while re-evaluating the saved "
            f"model file on the test split gives {m['accuracy'] * 100:.1f}% / {m['roc_auc']:.4f}. This page shows the "
            f"re-evaluated values so they agree with the plotted confusion matrix and curves. "
            f"See the *Report & export* tab for the full comparison."
        )


def _tab_performance(name, entry, key):
    metrics = entry["metrics"]
    class_names = entry["class_names"]
    color = dc.MODEL_COLORS.get(name, "#42A5F5")

    left, right = st.columns(2)
    with left:
        show_fig(dc.plot_metric_scorecard(metrics, color=color, title="Headline metrics"), f"{key}_score")
    with right:
        cm = dc.reorder_square(entry["confusion_matrix"], class_names, DISPLAY_ORDER)
        show_fig(dc.plot_confusion_matrix(cm, DISPLAY_ORDER, title="Confusion matrix (rows = true class)"),
                 f"{key}_cm")

    left, right = st.columns(2)
    class_df = _reorder_class_df(entry["class_perf_df"])
    with left:
        show_fig(dc.plot_per_class_metrics(class_df, title="Precision / recall / F1 by class"), f"{key}_pc")
    with right:
        st.markdown("**Per-class breakdown**")
        show_df(
            class_df,
            column_config={
                "Precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "Recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                "F1-Score": st.column_config.NumberColumn("F1", format="%.3f"),
                "Support": st.column_config.NumberColumn("Support", format="%d"),
            },
        )

        st.markdown("**What the errors look like**")
        weakest = class_df.sort_values("Recall").iloc[0]
        st.markdown(f"- Weakest class: **{weakest['Class']}** (recall {weakest['Recall']:.1%}, "
                    f"precision {weakest['Precision']:.1%}).")
        for t, p, n, share in _top_confusions(entry["confusion_matrix"], class_names, k=2):
            st.markdown(f"- **{t} → {p}**: {n} samples ({share:.1%} of true {t}).")
        cm_arr = np.array(entry["confusion_matrix"])
        hi, lo = class_names.index("High"), class_names.index("Low")
        extreme = int(cm_arr[hi, lo] + cm_arr[lo, hi])
        st.markdown(f"- **Extreme misses (High ↔ Low, skipping Moderate):** {extreme} of 1,000 samples"
                    + (" - the most serious kind of error." if extreme else "."))
        if weakest["Recall"] < 0.5:
            st.error(f"This model misses most **{weakest['Class']}** samples - a class-level failure that the "
                     f"overall accuracy hides.")


def _tab_curves(name, entry, key):
    left, right = st.columns(2)
    with left:
        show_fig(dc.plot_roc_curves(_reorder_curves(entry["roc_curves"]),
                                    title="ROC - one-vs-rest"), f"{key}_roc")
    with right:
        show_fig(dc.plot_pr_curves(_reorder_curves(entry["pr_curves"]),
                                   title="Precision-recall - one-vs-rest"), f"{key}_pr")

    rows = []
    for c in DISPLAY_ORDER:
        rows.append({"Class": c, "ROC-AUC": f"{entry['roc_curves'][c]['auc']:.4f}",
                     "Average precision (AP)": f"{entry['pr_curves'][c]['ap']:.4f}"})
    rows.append({"Class": "Micro-average", "ROC-AUC": f"{entry['roc_curves']['Micro-Average']['auc']:.4f}",
                 "Average precision (AP)": "-"})
    show_df(pd.DataFrame(rows))


def _tab_internals(name, entry, key, quantum_metrics, qsvm_metrics, cache):
    features = cache["sensor_features"]
    class_names = entry["class_names"]

    if name == "Logistic Regression":
        coefs = np.array(entry["coefficients"])
        show_fig(dc.plot_coefficients(coefs, features, class_names,
                                      title="Standardized coefficients (log-odds per feature)"), f"{key}_coef")
        coef_df = pd.DataFrame(coefs.T, columns=[f"β {c}" for c in class_names])
        coef_df.insert(0, "Feature", features)
        odds_df = pd.DataFrame(np.array(entry["odds_ratios"]).T, columns=[f"Odds ratio {c}" for c in class_names])
        odds_df.insert(0, "Feature", features)
        left, right = st.columns(2)
        with left:
            st.markdown("**Coefficients**")
            show_df(coef_df, column_config={c: st.column_config.NumberColumn(format="%+.3f") for c in coef_df.columns[1:]})
        with right:
            st.markdown("**Odds ratios (per +1 SD)**")
            show_df(odds_df, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in odds_df.columns[1:]})
        st.caption("Intercepts: " + ", ".join(f"{c} = {b:+.3f}" for c, b in zip(class_names, entry["intercept"])))
        st.info("The sensor features are strongly inter-correlated (|r| ≈ 0.97-0.99, see the EDA heatmap), so the "
                "weights split credit between near-duplicate signals. Read the **direction** and rough size of each "
                "coefficient, not an isolated per-sensor effect.")

    elif name in ("Random Forest", "XGBoost"):
        imps = np.array(entry["feature_importances"])
        left, right = st.columns([3, 2])
        with left:
            show_fig(dc.plot_feature_importance(imps, features, title="Feature importance"), f"{key}_imp")
        with right:
            st.markdown("**Configuration**")
            if name == "Random Forest":
                st.markdown(f"- Trees: **{entry['n_estimators']}**\n- Max depth: **{entry['max_depth']}**\n"
                            f"- Importance type: mean decrease in impurity")
            else:
                st.markdown("- Boosting rounds: **100**\n- Learning rate (eta): **0.1**\n- Max depth: **6**\n"
                            "- Importance type: native XGBoost importance")
            top = features[int(np.argmax(imps))]
            st.markdown(f"Most used feature: **{top}** ({imps.max():.1%}).")
        st.info("Tree importances are split between correlated sensors, so a low score does not mean a feature is "
                "uninformative - another sensor may simply carry the same signal.")
        if name == "XGBoost":
            st.caption(entry.get("shap_status", ""))

    elif name == "Classical SVM":
        c = st.columns(5)
        c[0].metric("Kernel", entry["kernel"].upper())
        c[1].metric("C", entry["C"])
        c[2].metric("gamma", entry["gamma"])
        c[3].metric("Support vectors", f"{entry['total_support_vectors']}")
        c[4].metric("Share of training set", f"{entry['total_support_vectors'] / N_TRAIN:.1%}")
        left, right = st.columns(2)
        with left:
            show_fig(dc.plot_support_vectors(entry["n_support"], class_names, N_TRAIN), f"{key}_sv")
        with right:
            show_fig(dc.plot_permutation_importance(entry["permutation_importance"], features,
                                                    title="Permutation importance (test set)"), f"{key}_perm")
        show_fig(dc.plot_pca_decision_boundary(entry["pca_2d_boundary"], class_names,
                                               title="Decision regions in 2-D PCA space"), f"{key}_boundary")
        st.caption("The shaded regions come from an RBF-SVM (same C and gamma) refit on only the first two principal "
                   "components so the boundary can be drawn in 2-D. It illustrates the geometry; it is not the exact "
                   "7-D model that produced the metrics above. Points are the first 300 test samples.")

    elif name == "PennyLane 4-Qubit VQC":
        circ = entry["circuit_summary"]
        c = st.columns(5)
        c[0].metric("Qubits", circ["qubits"])
        c[1].metric("Layers", circ["layers"])
        c[2].metric("Trainable parameters", circ["trainable_parameters"])
        c[3].metric("Epochs", quantum_metrics.get("epochs", len(entry["loss_history"])))
        c[4].metric("Training subset", f"{quantum_metrics.get('training_subset_size', '-')} samples")
        show_fig(dc.plot_vqc_training_curves(entry["loss_history"], entry["accuracy_history"],
                                             title=f"VQC optimisation trajectory ({len(entry['loss_history'])} epochs)"),
                 f"{key}_train")
        left, right = st.columns(2)
        with left:
            st.markdown("**Circuit**")
            st.markdown(f"- Embedding: {circ['embedding']}\n- Ansatz: {circ['ansatz']}\n- Readout: {circ['readout']}")
            st.markdown(f"- Optimizer: Adam, lr = {quantum_metrics.get('learning_rate', '-')}, "
                        f"batch size = {quantum_metrics.get('batch_size', '-')}")
        with right:
            st.markdown("**Training outcome**")
            lh, ah = entry["loss_history"], entry["accuracy_history"]
            st.markdown(f"- Loss: {lh[0]:.3f} → {lh[-1]:.3f}\n- Training accuracy: {ah[0]:.1%} → {ah[-1]:.1%} "
                        f"(best {max(ah):.1%})")
        st.info(f"Training used a {quantum_metrics.get('training_subset_size', 360)}-sample subset of the "
                f"{N_TRAIN:,}-sample training set. Training accuracy levels off within the first few epochs while the "
                f"loss keeps falling slowly. Check the *Performance* tab for where it fails: errors between the two "
                f"extreme classes (High ↔ Low) are the clearest sign this circuit is not separating the physics well.")

    elif name == "Qiskit 4-Qubit QSVM":
        qd = entry["qsvm_details"]
        props = qsvm_metrics.get("kernel_properties", {})
        eigs = np.array(qd["eigenvalues"])
        c = st.columns(5)
        c[0].metric("Qubits", qd["num_qubits"])
        c[1].metric("Feature-map reps", qd["reps"])
        c[2].metric("Entanglement", qd["entanglement"])
        c[3].metric("Phase scale factor", qd["scale_factor"])
        _kpi(c[4], "Min kernel eigenvalue", f"{eigs.min():.1e}",
             "PSD ✓" if eigs.min() > -1e-8 else "not PSD ✗")

        left, right = st.columns(2)
        with left:
            show_fig(dc.plot_qsvm_kernel_heatmap(qd["kernel_matrix_50"]), f"{key}_gram")
        with right:
            show_fig(dc.plot_qsvm_eigenvalues(qd["eigenvalues"]), f"{key}_eig")

        left, right = st.columns(2)
        with left:
            show_fig(dc.plot_fidelity_distributions(qd["same_class_fidelities"], qd["diff_class_fidelities"],
                                                    title="Fidelity: same-class vs different-class pairs"),
                     f"{key}_fid")
        with right:
            show_fig(dc.plot_kernel_comparison_distributions(cache["svm_vs_qsvm"]["rbf_vals"],
                                                             cache["svm_vs_qsvm"]["quantum_vals"],
                                                             title="Kernel values: classical RBF vs quantum fidelity"),
                     f"{key}_kcmp")

        same, diff = np.mean(qd["same_class_fidelities"]), np.mean(qd["diff_class_fidelities"])
        st.markdown(f"Mean fidelity: **{same:.3f}** for same-class pairs vs **{diff:.3f}** for different-class pairs "
                    f"(first 50 training samples). Kernel symmetric: **{props.get('is_symmetric', '-')}**, "
                    f"mean diagonal: **{props.get('mean_diagonal', '-')}**.")

        svm_entry = cache["models"].get("Classical SVM")
        if svm_entry:
            st.markdown("#### Head-to-head: Classical SVM vs QSVM")
            show_fig(dc.plot_svm_vs_qsvm_roc(svm_entry["roc_curves"], entry["roc_curves"], DISPLAY_ORDER),
                     f"{key}_h2h")
            ms, mq = svm_entry["metrics"], entry["metrics"]
            st.markdown(
                f"Accuracy **{ms['accuracy']:.1%}** (SVM) vs **{mq['accuracy']:.1%}** (QSVM); "
                f"inference **{ms['inference_time']:.4f}s** vs **{mq['inference_time']:.4f}s** "
                f"(≈{mq['inference_time'] / max(ms['inference_time'], 1e-9):.0f}x slower for the simulated quantum kernel). "
                f"On this tabular data the quantum kernel matches the classical one but shows no advantage."
            )

        with st.expander("ZZFeatureMap circuit (text drawing)"):
            st.code(qd["circuit_text"], language=None)


def _tab_report(name, entry, bench_row, key):
    m = entry["metrics"]
    st.markdown("**Reported vs re-evaluated**")
    st.caption("*Reported* = `reports/benchmark_summary.json` (written by the training/benchmark scripts). "
               "*Re-evaluated* = the shipped model file scored on the test split by `src/generate_analysis_cache.py`. "
               "Training/inference times are carried over from the benchmark run, not re-measured.")
    if bench_row:
        pairs = [
            ("Accuracy (%)", bench_row["Accuracy (%)"], m["accuracy"] * 100),
            ("Macro precision", bench_row["Macro-Precision"], m["macro_precision"]),
            ("Macro recall", bench_row["Macro-Recall"], m["macro_recall"]),
            ("Macro F1", bench_row["Macro-F1"], m["macro_f1"]),
            ("Weighted F1", bench_row["Weighted-F1"], m["weighted_f1"]),
            ("ROC-AUC", bench_row["ROC-AUC"], m["roc_auc"]),
            ("Generalization error", bench_row["Generalization Error"], m["generalization_error"]),
        ]
        df = pd.DataFrame(pairs, columns=["Metric", "Reported", "Re-evaluated"])
        df["Difference"] = df["Re-evaluated"] - df["Reported"]
        df["Status"] = np.where(df["Difference"].abs() < 5e-4 * np.where(df["Metric"] == "Accuracy (%)", 100, 1),
                                "✓ matches", "≠ differs")
        show_df(df, column_config={
            "Reported": st.column_config.NumberColumn(format="%.4f"),
            "Re-evaluated": st.column_config.NumberColumn(format="%.4f"),
            "Difference": st.column_config.NumberColumn(format="%+.4f"),
        })
    else:
        st.info("No benchmark entry found for this model.")

    st.markdown("**Downloads**")
    class_df = _reorder_class_df(entry["class_perf_df"])
    cm = pd.DataFrame(dc.reorder_square(entry["confusion_matrix"], entry["class_names"], DISPLAY_ORDER),
                      index=[f"true_{c}" for c in DISPLAY_ORDER], columns=[f"pred_{c}" for c in DISPLAY_ORDER])
    payload = {"model": name, "category": entry["category"], "test_samples": 1000, "metrics": entry["metrics"],
               "per_class": class_df.to_dict(orient="records"), "class_order_in_confusion_matrix": DISPLAY_ORDER,
               "confusion_matrix": cm.values.tolist()}
    c1, c2, c3 = st.columns(3)
    c1.download_button("📥 Metrics (JSON)", json.dumps(payload, indent=2), f"{_slug(name)}_metrics.json",
                       "application/json", key=f"{key}_dl_json")
    c2.download_button("📥 Per-class metrics (CSV)", class_df.to_csv(index=False), f"{_slug(name)}_per_class.csv",
                       "text/csv", key=f"{key}_dl_class")
    c3.download_button("📥 Confusion matrix (CSV)", cm.to_csv(), f"{_slug(name)}_confusion_matrix.csv",
                       "text/csv", key=f"{key}_dl_cm")


# --------------------------------------------------------------------------- compare view
def _render_compare(models, bench_lookup):
    st.markdown("## ⚖️ Compare all models")
    st.caption("All six classifiers on the same 1,000 held-out samples.")

    rows = []
    for n, e in models.items():
        m = e["metrics"]
        rows.append({
            "Model": n, "Type": e["category"], "Accuracy": m["accuracy"], "Macro precision": m["macro_precision"],
            "Macro recall": m["macro_recall"], "Macro F1": m["macro_f1"], "Weighted F1": m["weighted_f1"],
            "ROC-AUC": m["roc_auc"], "Gen. error": m["generalization_error"],
            "Train (s)": m["training_time"], "Inference (s)": m["inference_time"],
        })
    table = pd.DataFrame(rows).sort_values(["Accuracy", "Macro F1"], ascending=False).reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    num = lambda label: st.column_config.NumberColumn(label, format="%.4f")
    show_df(table, column_config={
        "Accuracy": num("Accuracy"), "Macro precision": num("Macro precision"), "Macro recall": num("Macro recall"),
        "Macro F1": num("Macro F1"), "Weighted F1": num("Weighted F1"), "ROC-AUC": num("ROC-AUC"),
        "Gen. error": st.column_config.NumberColumn(format="%.3f"),
        "Train (s)": st.column_config.NumberColumn(format="%.4f"),
        "Inference (s)": st.column_config.NumberColumn(format="%.4f"),
    })
    st.download_button("📥 Download comparison (CSV)", table.to_csv(index=False), "model_comparison.csv", "text/csv",
                       key="cmp_dl")

    show_fig(dc.plot_metric_comparison(
        models, ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "roc_auc"],
        ["Accuracy", "Macro precision", "Macro recall", "Macro F1", "Weighted F1", "ROC-AUC"],
        title="Quality metrics by model"), "cmp_bars")

    left, right = st.columns(2)
    with left:
        show_fig(dc.plot_radar(models, title="Quality profile (radial axis starts at 0.5)"), "cmp_radar")
    with right:
        show_fig(dc.plot_macro_roc_overlay(models), "cmp_roc")

    left, right = st.columns(2)
    with left:
        show_fig(dc.plot_class_metric_heatmap(models, DISPLAY_ORDER, "F1-Score", "Per-class F1 by model"), "cmp_f1")
    with right:
        show_fig(dc.plot_class_metric_heatmap(models, DISPLAY_ORDER, "Recall", "Per-class recall by model"), "cmp_rec")

    show_fig(dc.plot_latency(models), "cmp_latency")
    _confusion_grid(models, "cmp_grid")


def _confusion_grid(models, key):
    some = next(iter(models.values()))
    show_fig(dc.plot_confusion_grid(models, some["class_names"], DISPLAY_ORDER,
                                    title="Confusion matrices - all models (row-normalised colour, counts in cells)"),
             key)


def render_confusion_grid_section(reports_dir):
    """Interactive, correctly labelled replacement for the static 2x3 confusion-matrix PNG."""
    path = os.path.join(reports_dir, "model_analysis_cache.joblib")
    if not os.path.exists(path):
        return False
    cache = load_analysis_cache(path, os.path.getmtime(path))
    models = {}
    for n, e in cache["models"].items():
        models[n] = dict(e, class_names=cache["class_names"])
    _confusion_grid(models, "lb_grid")
    return True


# --------------------------------------------------------------------------- entry point
def render_model_explorer(reports_dir, bench_summary, quantum_metrics, qsvm_metrics):
    st.header("🧪 Model Explorer")
    st.markdown("Pick a model to see **its own** metrics, curves and internals, or compare all of them side by side.")

    path = os.path.join(reports_dir, "model_analysis_cache.joblib")
    if not os.path.exists(path):
        st.error("`reports/model_analysis_cache.joblib` not found. Generate it with "
                 "`python src/generate_analysis_cache.py`.")
        return
    cache = load_analysis_cache(path, os.path.getmtime(path))
    models = {}
    for n, e in cache["models"].items():
        models[n] = dict(e, class_names=cache["class_names"])
    bench_lookup = _bench_lookup(bench_summary)

    choice = st.radio("Model", list(models.keys()) + [COMPARE_VIEW], horizontal=True, key="explorer_model",
                      label_visibility="collapsed")
    st.markdown("---")

    if choice == COMPARE_VIEW:
        _render_compare(models, bench_lookup)
        return

    entry = models[choice]
    key = _slug(choice)
    bench_row = bench_lookup.get(choice)

    _render_header(choice, entry, bench_row)
    _render_kpis(choice, models)
    _render_drift_note(choice, entry, bench_row)

    t_perf, t_curves, t_int, t_rep = st.tabs(["📊 Performance", "📈 Curves", "🔬 Model internals", "🧾 Report & export"])
    with t_perf:
        _tab_performance(choice, entry, key)
    with t_curves:
        _tab_curves(choice, entry, key)
    with t_int:
        _tab_internals(choice, entry, key, quantum_metrics, qsvm_metrics, cache)
    with t_rep:
        _tab_report(choice, entry, bench_row, key)
