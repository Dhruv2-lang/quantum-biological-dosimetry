"""Regression tests for class-label ordering (encoder order is High, Low, Moderate)."""
import os
import sys

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.preprocessing import PreprocessingPipeline

DATA = os.path.join(BASE_DIR, "data", "Quantum_Biological_Dosimetry_Dataset_5000.csv")


def test_split_class_names_follow_encoder_order():
    splits = PreprocessingPipeline().fit_and_split(pd.read_csv(DATA))
    assert splits["class_names"] == ["High", "Low", "Moderate"]


def test_dashboard_class_names_match_saved_preprocessor():
    pre = PreprocessingPipeline.load(os.path.join(BASE_DIR, "models", "preprocessor_bundle.joblib"))
    assert [str(c) for c in pre.label_encoder.classes_] == ["High", "Low", "Moderate"]


def test_logistic_regression_labels_match_truth():
    pre = PreprocessingPipeline.load(os.path.join(BASE_DIR, "models", "preprocessor_bundle.joblib"))
    lr = joblib.load(os.path.join(BASE_DIR, "models", "logistic_regression.joblib"))
    df = pd.read_csv(DATA).sample(500, random_state=1)
    pred = lr.predict(pre.transform_batch(df)["scaled"])
    names = np.array([str(c) for c in pre.label_encoder.classes_])[pred]
    assert (names == df["Radiotoxicity_Class"].values).mean() > 0.9
