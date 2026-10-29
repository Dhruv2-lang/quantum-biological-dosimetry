"""
Streamlit Dashboard for Quantum Biological Dosimetry.
Supports:
1. Dataset Exploration & Sensor Physics EDA
2. Single-Sample Real-Time Inference (Classical & Quantum VQC)
3. Batch CSV Inference & Export
4. Benchmark Comparison Leaderboard & Confusion Matrices
5. Research Context, Sensor Principles & Problem 7 Disambiguation

Architectural Hardening on macOS ARM64:
- Pins OpenMP execution to single-threaded mode to prevent multiple libomp collision.
- Uses headless Agg Matplotlib backend to prevent Cocoa/AppKit main-thread violations.
- Evaluates PennyLane VQC with pure NumPy statevector simulation at serving time.
- Handles scikit-learn cross-version and Streamlit use_container_width deprecation warnings.
"""

import os
import sys
import warnings

# 1. Enforce single-threaded OpenMP to prevent conflicting libomp runtimes on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Enforce headless Matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 3. Suppress scikit-learn version mismatch and Streamlit deprecation warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import PreprocessingPipeline
from src.models_quantum import load_qnn_classifier

DATA_PATH = os.path.join(BASE_DIR, "data", "Quantum_Biological_Dosimetry_Dataset_5000.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
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

CLASS_NAMES = ["Low", "Moderate", "High"]

# Page configuration
st.set_page_config(
    page_title="Quantum Biological Dosimetry",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_dataframe(df_to_render):
    """Renders DataFrame safely across all Streamlit versions."""
    try:
        st.dataframe(df_to_render, use_container_width=True)
    except TypeError:
        st.dataframe(df_to_render, width="stretch")

def render_image(image_path, caption=None):
    """Renders Image safely across all Streamlit versions."""
    if os.path.exists(image_path):
        try:
            st.image(image_path, caption=caption, use_container_width=True)
        except TypeError:
            st.image(image_path, caption=caption, width="stretch")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_resource
def load_models_and_preprocessor():
    preprocessor = PreprocessingPipeline.load(os.path.join(MODELS_DIR, "preprocessor_bundle.joblib"))
    
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib"))
    if hasattr(xgb_model, "set_params"):
        try:
            xgb_model.set_params(n_jobs=1)
        except Exception:
            pass

    models = {
        "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib")),
        "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib")),
        "XGBoost": xgb_model,
        "PennyLane 4-Qubit VQC": load_qnn_classifier(os.path.join(MODELS_DIR, "qnn_vqc.joblib")),
        "Ridge Regressor (DNA Damage)": joblib.load(os.path.join(MODELS_DIR, "ridge_regression.joblib")),
        "Random Forest Regressor (DNA Damage)": joblib.load(os.path.join(MODELS_DIR, "random_forest_regressor.joblib")),
    }
    return preprocessor, models

@st.cache_data
def load_reports():
    eda_path = os.path.join(REPORTS_DIR, "eda_summary.json")
    bench_path = os.path.join(REPORTS_DIR, "benchmark_summary.json")
    q_path = os.path.join(REPORTS_DIR, "quantum_metrics.json")
    
    eda_data = json.load(open(eda_path)) if os.path.exists(eda_path) else {}
    bench_data = json.load(open(bench_path)) if os.path.exists(bench_path) else {}
    q_data = json.load(open(q_path)) if os.path.exists(q_path) else {}
    return eda_data, bench_data, q_data

df = load_data()
eda_summary, bench_summary, quantum_metrics = load_reports()

# Sidebar
st.sidebar.title("⚛️ Quantum Biodosimetry")
st.sidebar.markdown(
    "Predicting radiation-induced biological damage from **NV-center Diamond** "
    "and **Quantum Dot** sensor telemetry."
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dataset & Sensor EDA",
        "Single Prediction",
        "Batch CSV Prediction",
        "Benchmark Leaderboard",
        "Sensor Physics & Disambiguation"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Integrity Note:** Only 7 quantum sensor features are used. "
    "Downstream biological assays are excluded to prevent target leakage."
)

# ----------------- TAB 1: DATASET EXPLORATION & EDA -----------------
if page == "Dataset & Sensor EDA":
    st.header("📊 Dataset Exploration & Quantum Sensor Audit")
    st.markdown(
        "Exploratory analysis of `Quantum_Biological_Dosimetry_Dataset_5000.csv` (5,000 verified rows). "
        "The dataset captures non-invasive quantum sensor observables alongside cellular damage markers."
    )

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Sensor Features", len(SENSOR_FEATURES))
        c3.metric("Missing Values", df.isnull().sum().sum())
        c4.metric("Radiotoxicity Classes", df["Radiotoxicity_Class"].nunique())

        st.subheader("Dataset Preview")
        render_dataframe(df.head(10))

        st.subheader("Target Leakage Audit")
        st.warning(
            "⚠️ **Strict Target Leakage Safeguard:** "
            "Downstream assay measurements (`ROS_Level`, `Mitochondrial_Damage`, `GammaH2AX_Foci`, "
            "`BP53_Foci`, `Cell_Viability`, `DNA_Damage_Score`) are cellular destruction metrics. "
            "Specifically, `Radiotoxicity_Class` is directly thresholded on `DNA_Damage_Score` "
            "(Low < 25.0, Moderate 25.0-50.0, High >= 50.0). "
            "To build an authentic non-invasive dosimeter, ALL models are trained strictly on the 7 quantum sensor features."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            render_image(os.path.join(FIGURES_DIR, "eda_leakage_audit.png"), caption="DNA Damage Score Separation Across Radiotoxicity Classes")
        with col_b:
            render_image(os.path.join(FIGURES_DIR, "eda_correlation_matrix.png"), caption="Feature Correlation Heatmap")

        st.subheader("Sensor Distributions Across Damage Levels")
        render_image(os.path.join(FIGURES_DIR, "eda_sensor_distributions.png"), caption="KDE Distributions of Quantum Sensor Features")

# ----------------- TAB 2: SINGLE PREDICTION -----------------
elif page == "Single Prediction":
    st.header("🔮 Real-Time Single-Sample Inference")
    st.markdown(
        "Simulate quantum sensor readings from an irradiated biosample. Select a classical model "
        "or the 4-qubit Variational Quantum Classifier (VQC) to predict radiotoxicity and DNA damage score."
    )

    preprocessor, models = load_models_and_preprocessor()

    # Preset selector
    st.subheader("Preset Profiles")
    preset = st.selectbox(
        "Choose a representative dosimetry preset or adjust sliders below:",
        ["Custom", "Low Exposure (Dose ~1.5 Gy)", "Moderate Exposure (Dose ~5.5 Gy)", "Severe Radiotoxicity (Dose ~9.0 Gy)"]
    )

    defaults = {
        "Dose_Gy": 4.5,
        "ODMR_Shift_MHz": 12.0,
        "T1_us": 280.0,
        "T2_us": 95.0,
        "Fluorescence_Intensity": 650.0,
        "Fluorescence_Lifetime_ns": 8.5,
        "QD_Emission_nm": 552.0
    }

    if preset == "Low Exposure (Dose ~1.5 Gy)":
        defaults = {"Dose_Gy": 1.5, "ODMR_Shift_MHz": 3.8, "T1_us": 420.0, "T2_us": 145.0, "Fluorescence_Intensity": 880.0, "Fluorescence_Lifetime_ns": 13.5, "QD_Emission_nm": 541.0}
    elif preset == "Moderate Exposure (Dose ~5.5 Gy)":
        defaults = {"Dose_Gy": 5.5, "ODMR_Shift_MHz": 14.2, "T1_us": 265.0, "T2_us": 88.0, "Fluorescence_Intensity": 620.0, "Fluorescence_Lifetime_ns": 7.9, "QD_Emission_nm": 554.0}
    elif preset == "Severe Radiotoxicity (Dose ~9.0 Gy)":
        defaults = {"Dose_Gy": 9.2, "ODMR_Shift_MHz": 22.4, "T1_us": 135.0, "T2_us": 42.0, "Fluorescence_Intensity": 350.0, "Fluorescence_Lifetime_ns": 4.1, "QD_Emission_nm": 567.0}

    col1, col2, col3 = st.columns(3)
    with col1:
        dose = st.slider("Dose (Gy)", 0.0, 10.0, float(defaults["Dose_Gy"]), 0.1)
        odmr = st.slider("NV ODMR Shift (MHz)", 0.0, 26.0, float(defaults["ODMR_Shift_MHz"]), 0.1)
        t1 = st.slider("Spin Lattice Relaxation T1 (µs)", 100.0, 500.0, float(defaults["T1_us"]), 1.0)
    with col2:
        t2 = st.slider("Spin-Spin Relaxation T2 (µs)", 20.0, 180.0, float(defaults["T2_us"]), 1.0)
        fluor = st.slider("Fluorescence Intensity (a.u.)", 200.0, 1050.0, float(defaults["Fluorescence_Intensity"]), 5.0)
        lifetime = st.slider("Fluorescence Lifetime (ns)", 2.0, 16.0, float(defaults["Fluorescence_Lifetime_ns"]), 0.1)
    with col3:
        qd = st.slider("QD Emission Peak (nm)", 530.0, 575.0, float(defaults["QD_Emission_nm"]), 0.1)
        model_choice = st.selectbox(
            "Classification Model",
            ["Logistic Regression", "Random Forest", "XGBoost", "PennyLane 4-Qubit VQC"]
        )
        reg_model_choice = st.selectbox(
            "Regression Model (DNA Damage)",
            ["Ridge Regressor (DNA Damage)", "Random Forest Regressor (DNA Damage)"]
        )

    input_dict = {
        "Dose_Gy": dose,
        "ODMR_Shift_MHz": odmr,
        "T1_us": t1,
        "T2_us": t2,
        "Fluorescence_Intensity": fluor,
        "Fluorescence_Lifetime_ns": lifetime,
        "QD_Emission_nm": qd
    }

    if st.button("🚀 Run Prediction", type="primary"):
        transformed = preprocessor.transform_single(input_dict)
        clf = models[model_choice]
        reg = models[reg_model_choice]

        # Classification prediction
        if model_choice == "PennyLane 4-Qubit VQC":
            probs = clf.predict_proba(transformed["pca4"])[0]
            pred_idx = np.argmax(probs)
        else:
            probs = clf.predict_proba(transformed["scaled"])[0]
            pred_idx = np.argmax(probs)
        
        pred_class = CLASS_NAMES[pred_idx]

        # Regression prediction
        dna_score = float(reg.predict(transformed["scaled"])[0])
        dna_score = max(0.0, min(100.0, dna_score))

        st.markdown("---")
        st.subheader("Prediction Results")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            color_map = {"Low": "green", "Moderate": "orange", "High": "red"}
            st.markdown(f"### Predicted Class: :{color_map[pred_class]}[**{pred_class}**]")
            st.metric("Model Used", model_choice)
            
            # Probability distribution chart
            prob_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Class"))

        with res_col2:
            st.markdown("### Secondary Regression Target")
            st.metric("Predicted DNA Damage Score", f"{dna_score:.2f} / 100")
            st.progress(dna_score / 100.0)
            st.caption(f"Calculated via {reg_model_choice}")

            if model_choice == "PennyLane 4-Qubit VQC":
                st.info(
                    "⚛️ **Quantum Execution Detail:**\n"
                    "- 4 PCA-encoded features embedded via `AngleEmbedding(Y)`.\n"
                    "- 2 layers of `StronglyEntanglingLayers` rotated state vector.\n"
                    "- Measured expectation values: $\\langle Z_0 \\rangle, \\langle Z_1 \\rangle, \\langle Z_2 \\rangle$."
                )

# ----------------- TAB 3: BATCH CSV PREDICTION -----------------
elif page == "Batch CSV Prediction":
    st.header("📁 Batch CSV Inference & Export")
    st.markdown(
        "Upload a CSV file containing the 7 quantum sensor columns to run batch predictions "
        "and download the resulting radiotoxicity and DNA damage assessments."
    )

    preprocessor, models = load_models_and_preprocessor()

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if df is not None:
        sample_batch = df[SENSOR_FEATURES].head(25)
        st.download_button(
            label="📥 Download Sample Input CSV (25 rows)",
            data=sample_batch.to_csv(index=False),
            file_name="quantum_sensor_sample_input.csv",
            mime="text/csv"
        )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows.")
            
            missing_cols = [c for c in SENSOR_FEATURES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required sensor columns: {missing_cols}")
            else:
                batch_model = st.selectbox(
                    "Select Inference Model for Batch:",
                    ["Logistic Regression", "Random Forest", "XGBoost", "PennyLane 4-Qubit VQC"]
                )

                if st.button("Run Batch Inference"):
                    with st.spinner("Processing batch predictions..."):
                        t_data = preprocessor.transform_batch(batch_df)
                        clf = models[batch_model]
                        reg = models["Ridge Regressor (DNA Damage)"]

                        if batch_model == "PennyLane 4-Qubit VQC":
                            probs = clf.predict_proba(t_data["pca4"])
                            preds = clf.predict(t_data["pca4"])
                        else:
                            probs = clf.predict_proba(t_data["scaled"])
                            preds = clf.predict(t_data["scaled"])

                        reg_preds = reg.predict(t_data["scaled"])

                        out_df = batch_df.copy()
                        out_df["Predicted_Radiotoxicity_Class"] = [CLASS_NAMES[p] for p in preds]
                        out_df["Confidence"] = np.max(probs, axis=1).round(4)
                        out_df["Predicted_DNA_Damage_Score"] = reg_preds.round(2)

                        st.success(f"Inference complete on {len(out_df)} samples!")
                        render_dataframe(out_df.head(15))

                        csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Predictions CSV",
                            data=csv_bytes,
                            file_name=f"biodosimetry_predictions_{batch_model.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# ----------------- TAB 4: BENCHMARK LEADERBOARD -----------------
elif page == "Benchmark Leaderboard":
    st.header("📈 Model Benchmark Leaderboard")
    st.markdown(
        "Empirical head-to-head comparison between Classical ML baselines and the "
        "PennyLane 4-Qubit Variational Quantum Classifier (VQC) on 1,000 unseen test samples."
    )

    if bench_summary and "classification_comparison" in bench_summary:
        bench_df = pd.DataFrame(bench_summary["classification_comparison"])
        render_dataframe(bench_df)

        st.subheader("Key Empirical Insights")
        for finding in bench_summary.get("key_findings", []):
            st.markdown(f"- {finding}")

        st.subheader("Performance Comparison Plots")
        render_image(os.path.join(FIGURES_DIR, "model_comparison_benchmark.png"), caption="Accuracy & Macro-F1 Benchmark Across Models")

        st.subheader("Combined Confusion Matrices")
        render_image(os.path.join(FIGURES_DIR, "all_confusion_matrices_combined.png"), caption="Confusion Matrices across all 4 Evaluated Architectures")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("PennyLane VQC Convergence Curve")
            render_image(os.path.join(FIGURES_DIR, "qnn_training_curve.png"), caption="Loss and Accuracy Trajectories during Quantum Circuit Optimization")
        with col2:
            st.subheader("DNA Damage Score Regression Fit")
            render_image(os.path.join(FIGURES_DIR, "regression_evaluation.png"), caption="Actual vs Predicted DNA Damage Score (R² > 0.99)")

# ----------------- TAB 5: SENSOR PHYSICS & DISAMBIGUATION -----------------
elif page == "Sensor Physics & Disambiguation":
    st.header("🔬 Quantum Sensor Physics & Problem 7 Disambiguation")
    
    st.subheader("1. Quantum Sensor Telemetry Principles")
    st.markdown("""
    In quantum biological dosimetry, cellular damage is probed non-destructively using nanoscale solid-state quantum defects:
    - **Optically Detected Magnetic Resonance (ODMR Shift in MHz):** Negatively charged Nitrogen-Vacancy ($NV^-$) centers in diamond nanoparticles exhibit a zero-field splitting ($D \\approx 2.87\\text{ GHz}$) sensitive to local temperature, strain, and magnetic fluctuations induced by free radicals (ROS).
    - **Relaxometry ($T_1$ and $T_2$ in µs):** Spin-lattice ($T_1$) and spin-spin ($T_2$) relaxation times of NV centers shorten sharply in the presence of paramagnetic species (e.g. hydroxyl and superoxide radicals).
    - **Quantum Dot (QD) Spectral Shift & Lifetime:** Colloidal semiconducting quantum dots undergo photoluminescence quenching and lifetime reduction in acidified, oxidizing microenvironments.
    """)

    st.subheader("2. Disambiguation: Tabular Sensor Data vs Problem 7 Image Processing")
    st.info("""
    **Clarification on Problem 7 ("Quantum-Centric Image Processing"):**
    - The dataset examined in this project is **100% tabular sensor telemetry**, containing 7 scalar physical measurements per irradiation event.
    - **No image processing or quantum computer vision has been implemented**, as the data contains no spatial pixel grids, histological slides, or microscopic image frames.
    - Treating scalar spectroscopic parameters as image inputs would be scientifically unsound. Future quantum image processing extensions would require raw fluorescent microscopy micrographs processed via Quantum Convolutional Neural Networks (QCNNs).
    """)

    st.subheader("3. Scientific Integrity & Absence of Quantum Advantage")
    st.warning("""
    **Honest Scientific Appraisal:**
    - **No Quantum Advantage:** Classical models (Logistic Regression, Random Forest, XGBoost) achieved superior classification accuracy (~96%) and near-instant inference compared to the simulated 4-qubit VQC (~67.7%). On low-dimensional tabular data, classical gradient boosting remains state-of-the-art.
    - **No Clinical Validity Claim:** This simulator and dataset represent preclinical nanophotonic experimental models. No claims of immediate clinical diagnostic validity or human in vivo efficacy are made.
    """)
