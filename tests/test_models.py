"""
Automated unit tests for Classical and Quantum ML models.
Tests cover:
1. Classical Classifiers (Logistic Regression, Random Forest, XGBoost, Classical SVM)
2. Classical Regressors (Ridge, Random Forest Regressor)
3. Variational Quantum Classifier (PennyLane 4-Qubit VQC)
4. Quantum Support Vector Machine (Qiskit 4-Qubit QSVM)
5. Quantum Kernel Properties (Hermiticity/Symmetry, Normalization, Positive Semi-Definiteness)
6. Model Artifact Integrity & Diagnostic Figures

Compatible with both pytest and python3 -m unittest.
"""

import os
import unittest
import joblib
import numpy as np

import sys
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.preprocessing import PreprocessingPipeline
from src.models_quantum import load_qnn_classifier, TrainedQNNWrapper
from src.models_qsvm import load_qsvm_classifier, QSVMClassifier

MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
REPORTS_DIR = os.path.join(WORKSPACE_DIR, "reports")

class TestModelInference(unittest.TestCase):
    def setUp(self):
        self.preprocessor = PreprocessingPipeline.load(os.path.join(MODELS_DIR, "preprocessor_bundle.joblib"))
        self.logreg = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
        self.rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
        self.xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib"))
        self.svm = joblib.load(os.path.join(MODELS_DIR, "svm_classifier.joblib"))
        self.qnn = load_qnn_classifier(os.path.join(MODELS_DIR, "qnn_vqc.joblib"))
        self.qsvm = load_qsvm_classifier(os.path.join(MODELS_DIR, "qsvm_classifier.joblib"))
        self.ridge = joblib.load(os.path.join(MODELS_DIR, "ridge_regression.joblib"))
        self.rf_reg = joblib.load(os.path.join(MODELS_DIR, "random_forest_regressor.joblib"))

        # Synthetic 7-dimensional sensor input
        self.sample_7d = np.array([[4.5, 12.0, 280.0, 95.0, 650.0, 8.5, 552.0]])
        self.sample_scaled = self.preprocessor.scaler.transform(self.sample_7d)
        self.sample_pca4 = self.preprocessor.pca_4.transform(self.sample_scaled)

    def test_classical_classifiers_prediction(self):
        """Test prediction shape, valid classes, and probability normalization for classical models."""
        classifiers = [
            ("LogReg", self.logreg),
            ("RandomForest", self.rf),
            ("XGBoost", self.xgb),
            ("ClassicalSVM", self.svm)
        ]
        for name, clf in classifiers:
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

    def test_quantum_qsvm_prediction(self):
        """Test genuine 4-qubit Qiskit QSVM quantum kernel inference."""
        self.assertIsInstance(self.qsvm, QSVMClassifier)
        qsvm_pred = self.qsvm.predict(self.sample_pca4)
        self.assertEqual(qsvm_pred.shape, (1,))
        self.assertIn(int(qsvm_pred[0]), [0, 1, 2], f"QSVM predicted invalid class: {qsvm_pred[0]}")

        qsvm_probs = self.qsvm.predict_proba(self.sample_pca4)
        self.assertEqual(qsvm_probs.shape, (1, 3))
        self.assertAlmostEqual(float(np.sum(qsvm_probs)), 1.0, places=4, msg="QSVM probabilities do not sum to 1")

    def test_quantum_kernel_properties(self):
        """Verify quantum kernel Gram matrix symmetry, normalization, and positive semi-definiteness."""
        subset_X = self.qsvm.X_train_[:60]
        K_sub = self.qsvm.evaluate_kernel(subset_X)
        self.assertEqual(K_sub.shape, (60, 60))

        # 1. Symmetry / Hermiticity: K == K.T
        np.testing.assert_allclose(K_sub, K_sub.T, atol=1e-6, err_msg="Quantum kernel matrix is not symmetric")

        # 2. Normalization: K_ii == 1.0 (state fidelity with itself is unity)
        diag = np.diag(K_sub)
        np.testing.assert_allclose(diag, np.ones_like(diag), atol=1e-5, err_msg="Quantum kernel diagonal is not ~1.0")

        # 3. Positive semi-definiteness: all eigenvalues >= -1e-5
        eigenvalues = np.linalg.eigvalsh(K_sub)
        min_eig = np.min(eigenvalues)
        self.assertGreaterEqual(min_eig, -1e-5, f"Quantum kernel is not positive semi-definite (min eigenvalue = {min_eig})")

    def test_evaluation_artifacts_exist(self):
        """Verify report files and plots have been generated and are non-empty."""
        expected_files = [
            os.path.join(REPORTS_DIR, "eda_summary.json"),
            os.path.join(REPORTS_DIR, "classical_metrics.json"),
            os.path.join(REPORTS_DIR, "quantum_metrics.json"),
            os.path.join(REPORTS_DIR, "svm_metrics.json"),
            os.path.join(REPORTS_DIR, "qsvm_metrics.json"),
            os.path.join(REPORTS_DIR, "qsvm_config.json"),
            os.path.join(REPORTS_DIR, "benchmark_summary.json"),
            os.path.join(REPORTS_DIR, "figures", "eda_correlation_matrix.png"),
            os.path.join(REPORTS_DIR, "figures", "eda_leakage_audit.png"),
            os.path.join(REPORTS_DIR, "figures", "classical_confusion_matrices.png"),
            os.path.join(REPORTS_DIR, "figures", "qnn_training_curve.png"),
            os.path.join(REPORTS_DIR, "figures", "qsvm_confusion_matrix.png"),
            os.path.join(REPORTS_DIR, "figures", "qsvm_kernel_matrix.png"),
            os.path.join(REPORTS_DIR, "figures", "model_comparison_benchmark.png"),
            os.path.join(REPORTS_DIR, "figures", "precision_recall_f1_comparison.png"),
            os.path.join(REPORTS_DIR, "figures", "generalization_error_comparison.png"),
            os.path.join(REPORTS_DIR, "figures", "training_inference_time_comparison.png"),
            os.path.join(REPORTS_DIR, "figures", "multiclass_roc_curves.png"),
            os.path.join(REPORTS_DIR, "figures", "all_confusion_matrices_combined.png")
        ]
        for f in expected_files:
            self.assertTrue(os.path.exists(f), f"Expected artifact missing: {f}")
            self.assertGreater(os.path.getsize(f), 0, f"Artifact is empty: {f}")

if __name__ == "__main__":
    unittest.main()
