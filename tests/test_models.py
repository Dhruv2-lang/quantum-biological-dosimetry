"""
Automated unit tests for Classical and Quantum ML models.
Compatible with both pytest and python3 -m unittest.
"""

import os
import unittest
import joblib
import numpy as np

import sys
sys.path.insert(0, "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry")

from src.preprocessing import PreprocessingPipeline
from src.models_quantum import load_qnn_classifier, TrainedQNNWrapper

MODELS_DIR = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/models"
REPORTS_DIR = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/reports"

class TestModelInference(unittest.TestCase):
    def setUp(self):
        self.preprocessor = PreprocessingPipeline.load(os.path.join(MODELS_DIR, "preprocessor_bundle.joblib"))
        self.logreg = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
        self.rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
        self.xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib"))
        self.qnn = load_qnn_classifier(os.path.join(MODELS_DIR, "qnn_vqc.joblib"))
        self.ridge = joblib.load(os.path.join(MODELS_DIR, "ridge_regression.joblib"))
        self.rf_reg = joblib.load(os.path.join(MODELS_DIR, "random_forest_regressor.joblib"))

        # Synthetic 7-dimensional sensor input
        self.sample_7d = np.array([[4.5, 12.0, 280.0, 95.0, 650.0, 8.5, 552.0]])
        self.sample_scaled = self.preprocessor.scaler.transform(self.sample_7d)
        self.sample_pca4 = self.preprocessor.pca_4.transform(self.sample_scaled)

    def test_classical_classifiers_prediction(self):
        """Test prediction shape, valid classes, and probability normalization for classical models."""
        for name, clf in [("LogReg", self.logreg), ("RandomForest", self.rf), ("XGBoost", self.xgb)]:
            pred = clf.predict(self.sample_scaled)
            self.assertEqual(pred.shape, (1,), f"{name} prediction shape mismatch")
            self.assertIn(int(pred[0]), [0, 1, 2], f"{name} predicted invalid class: {pred[0]}")

            probs = clf.predict_proba(self.sample_scaled)
            self.assertEqual(probs.shape, (1, 3))
            self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4, msg=f"{name} probabilities do not sum to 1")

    def test_regression_prediction(self):
        """Test secondary regression target predictions within biological bounds [0, 100]."""
        for name, reg in [("Ridge", self.ridge), ("RF_Reg", self.rf_reg)]:
            dna_pred = reg.predict(self.sample_scaled)
            self.assertEqual(dna_pred.shape, (1,))
            self.assertTrue(0.0 <= dna_pred[0] <= 100.0, f"{name} score {dna_pred[0]} out of [0, 100]")

    def test_quantum_vqc_prediction(self):
        """Test genuine 4-qubit PennyLane VQC inference."""
        self.assertIsInstance(self.qnn, TrainedQNNWrapper)
        q_pred = self.qnn.predict(self.sample_pca4)
        self.assertEqual(q_pred.shape, (1,))
        self.assertIn(int(q_pred[0]), [0, 1, 2], f"VQC predicted invalid class: {q_pred[0]}")

        q_probs = self.qnn.predict_proba(self.sample_pca4)
        self.assertEqual(q_probs.shape, (1, 3))
        self.assertAlmostEqual(float(np.sum(q_probs)), 1.0, places=4, msg="VQC probabilities do not sum to 1")

    def test_evaluation_artifacts_exist(self):
        """Verify report files and plots have been generated and are non-empty."""
        expected_files = [
            os.path.join(REPORTS_DIR, "eda_summary.json"),
            os.path.join(REPORTS_DIR, "classical_metrics.json"),
            os.path.join(REPORTS_DIR, "quantum_metrics.json"),
            os.path.join(REPORTS_DIR, "benchmark_summary.json"),
            os.path.join(REPORTS_DIR, "figures", "eda_correlation_matrix.png"),
            os.path.join(REPORTS_DIR, "figures", "eda_leakage_audit.png"),
            os.path.join(REPORTS_DIR, "figures", "classical_confusion_matrices.png"),
            os.path.join(REPORTS_DIR, "figures", "qnn_training_curve.png"),
            os.path.join(REPORTS_DIR, "figures", "model_comparison_benchmark.png"),
            os.path.join(REPORTS_DIR, "figures", "all_confusion_matrices_combined.png")
        ]
        for f in expected_files:
            self.assertTrue(os.path.exists(f), f"Expected artifact missing: {f}")
            self.assertGreater(os.path.getsize(f), 0, f"Artifact is empty: {f}")

if __name__ == "__main__":
    unittest.main()
