"""
Quantum Support Vector Machine (QSVM) for Quantum Biological Dosimetry.
Implements a genuine Quantum Kernel Support Vector Machine using Qiskit and Qiskit Machine Learning:
- Quantum Feature Map: ZZFeatureMap (4 qubits, reps=1, linear entanglement)
- Quantum Kernel: FidelityQuantumKernel computing K(x1, x2) = |<phi(x1)|phi(x2)>|^2
- Dimensionality Reduction: 4-component PCA (retaining 99.62% sensor variance, fitted strictly on train data)
- Classification: sklearn.svm.SVC with precomputed quantum kernel
- Thread-safe, cached statevector fidelity evaluation for fast, crash-free serving
- Saves trained model artifacts, quantum configuration, and evaluation reports.
"""

import os
import json
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

import qiskit
from qiskit.circuit.library import zz_feature_map, ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels import FidelityQuantumKernel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

NUM_QUBITS = 4
REPS = 1
ENTANGLEMENT = "linear"
SCALE_FACTOR = 0.15
C_PARAM = 5.0
DEFAULT_TRAIN_SUBSET = 360  # Stratified representative subset, aligned with VQC training size


class QSVMClassifier:
    """
    Genuine Quantum Kernel Support Vector Machine.
    Computes fidelity kernel matrix K(x1, x2) = |<phi(x1)|phi(x2)>|^2
    using Qiskit's ZZFeatureMap and fits a precomputed SVC.
    """
    def __init__(
        self,
        feature_dimension=NUM_QUBITS,
        reps=REPS,
        entanglement=ENTANGLEMENT,
        scale=SCALE_FACTOR,
        C=C_PARAM,
        random_state=42
    ):
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.entanglement = entanglement
        self.scale = scale
        self.C = C
        self.random_state = random_state

        self.svc = SVC(
            kernel="precomputed",
            C=C,
            probability=True,
            class_weight="balanced",
            random_state=random_state
        )
        self.X_train_ = None
        self.sv_train_ = None
        self.classes_ = None
        self._feature_map = None
        self._qiskit_kernel = None

    def __getstate__(self):
        """
        Serialize only stable training state.

        Qiskit circuits and FidelityQuantumKernel instances are deliberately
        excluded because their internal ParameterExpression objects can be
        incompatible across Qiskit releases. They are recreated lazily.
        """
        state = self.__dict__.copy()
        state["_feature_map"] = None
        state["_qiskit_kernel"] = None
        return state

    def __setstate__(self, state):
        """Restore the fitted QSVM and recreate Qiskit caches lazily."""
        self.__dict__.update(state)
        self._feature_map = None
        self._qiskit_kernel = None

        # Keep compatibility with artifacts created before these attributes
        # were introduced.
        if not hasattr(self, "X_train_"):
            self.X_train_ = None
        if not hasattr(self, "sv_train_"):
            self.sv_train_ = None
        if not hasattr(self, "classes_"):
            self.classes_ = getattr(getattr(self, "svc", None), "classes_", None)

    @property
    def feature_map(self):
        """Constructs or returns the Qiskit QuantumCircuit for the ZZFeatureMap."""
        if self._feature_map is None:
            try:
                self._feature_map = zz_feature_map(
                    feature_dimension=self.feature_dimension,
                    reps=self.reps,
                    entanglement=self.entanglement
                )
            except Exception:
                self._feature_map = ZZFeatureMap(
                    feature_dimension=self.feature_dimension,
                    reps=self.reps,
                    entanglement=self.entanglement
                )
        return self._feature_map

    @property
    def qiskit_kernel(self):
        """Direct access to Qiskit Machine Learning FidelityQuantumKernel."""
        if self._qiskit_kernel is None:
            self._qiskit_kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        return self._qiskit_kernel

    def _compute_statevectors(self, X):
        """
        Computes quantum statevectors |phi(x)> = U_ZZ(scale * x)|0> for each classical sample.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        scaled_X = self.scale * X_arr
        fm = self.feature_map
        svs = [Statevector.from_instruction(fm.assign_parameters(x)).data for x in scaled_X]
        return np.array(svs)

    def evaluate_kernel(self, X1, X2=None):
        """
        Calculates the quantum fidelity kernel matrix:
        K_ij = |<phi(x_1i)|phi(x_2j)>|^2
        """
        sv1 = self._compute_statevectors(X1)
        if X2 is None or X2 is X1:
            overlaps = sv1 @ sv1.conj().T
            K = np.abs(overlaps) ** 2
            np.fill_diagonal(K, 1.0)
            return K

        sv2 = self._compute_statevectors(X2)
        overlaps = sv1 @ sv2.conj().T
        return np.abs(overlaps) ** 2

    def fit(self, X, y):
        """
        Fits the precomputed SVC using the quantum kernel matrix evaluated on X.
        """
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.sv_train_ = self._compute_statevectors(self.X_train_)

        overlaps = self.sv_train_ @ self.sv_train_.conj().T
        K_train = np.abs(overlaps) ** 2
        np.fill_diagonal(K_train, 1.0)

        self.svc.fit(K_train, y)
        self.classes_ = self.svc.classes_
        return self

    def _kernel_against_training(self, X):
        """Compute the quantum fidelity kernel between X and fitted training states."""
        if self.sv_train_ is None:
            raise ValueError("QSVMClassifier must be fitted before prediction.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Expected input with {self.feature_dimension} features; "
                f"received shape {X_arr.shape}."
            )

        sv_test = self._compute_statevectors(X_arr)
        overlaps = sv_test @ self.sv_train_.conj().T
        return np.abs(overlaps) ** 2

    def predict_proba(self, X):
        """Calculate test-to-training quantum fidelity kernels and class probabilities."""
        K_test = self._kernel_against_training(X)
        return self.svc.predict_proba(K_test)

    def predict(self, X):
        """
        Predict original class labels using the fitted precomputed-kernel SVC.

        Calling SVC.predict preserves the classifier's actual class labels and
        avoids assuming that labels are zero-based probability-column indices.
        """
        K_test = self._kernel_against_training(X)
        return self.svc.predict(K_test)


def train_and_evaluate_qsvm(subset_size=DEFAULT_TRAIN_SUBSET):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    splits_path = os.path.join(MODELS_DIR, "data_splits.joblib")
    if not os.path.exists(splits_path):
        raise FileNotFoundError("data_splits.joblib not found. Run preprocessing.py first.")

    splits = joblib.load(splits_path)
    X_train_pca = splits["X_train_pca4"]
    y_train = splits["y_train_class"]
    X_test_pca = splits["X_test_pca4"]
    y_test = splits["y_test_class"]
    class_names = splits["class_names"]

    # Stratified training subset for quantum kernel computation
    sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=42)
    sub_idx, _ = next(sss.split(X_train_pca, y_train))
    X_sub = X_train_pca[sub_idx]
    y_sub = y_train[sub_idx]

    print(f"--- Training Quantum Support Vector Machine (QSVM) ---")
    print(f"Quantum Architecture: {NUM_QUBITS} Qubits, ZZFeatureMap(reps={REPS}, entanglement='{ENTANGLEMENT}')")
    print(f"Training on {len(X_sub)} stratified samples across classes: {np.bincount(y_sub)}")

    qsvm = QSVMClassifier(
        feature_dimension=NUM_QUBITS,
        reps=REPS,
        entanglement=ENTANGLEMENT,
        scale=SCALE_FACTOR,
        C=C_PARAM,
        random_state=42
    )

    # 1. Fit QSVM (computes K_train and trains SVC)
    t0 = time.time()
    qsvm.fit(X_sub, y_sub)
    train_time = time.time() - t0
    print(f"QSVM fit completed in {train_time:.3f} seconds.")

    # Verify kernel properties
    K_train = qsvm.evaluate_kernel(X_sub)
    is_symmetric = bool(np.allclose(K_train, K_train.T))
    min_eigval = float(np.min(np.linalg.eigvalsh(K_train)))
    diag_mean = float(np.mean(np.diag(K_train)))
    print(f"Kernel Verification -> Symmetric: {is_symmetric}, Min Eigenvalue: {min_eigval:.6f}, Mean Diagonal: {diag_mean:.4f}")

    # 2. Evaluate on full test set (1,000 samples)
    t0 = time.time()
    y_pred = qsvm.predict(X_test_pca)
    inf_time = time.time() - t0
    y_prob = qsvm.predict_proba(X_test_pca)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    macro_prec = float(precision_score(y_test, y_pred, average="macro"))
    weighted_prec = float(precision_score(y_test, y_pred, average="weighted"))
    macro_rec = float(recall_score(y_test, y_pred, average="macro"))
    weighted_rec = float(recall_score(y_test, y_pred, average="weighted"))
    roc_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))
    gen_error = float(1.0 - acc)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print(f"QSVM Test Evaluation (N={len(y_test)}):")
    print(f"  Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(f"  Inference time (1,000 test samples): {inf_time:.3f} s")

    # 3. Save model artifacts as a plain state dictionary.
    # This avoids pickling the class object while this module is executing
    # as __main__ via `python -m src.models_qsvm`.
    qsvm_path1 = os.path.join(MODELS_DIR, "qsvm_classifier.joblib")
    qsvm_path2 = os.path.join(MODELS_DIR, "qsvm.joblib")
    _save_qsvm_classifier(qsvm, qsvm_path1)
    _save_qsvm_classifier(qsvm, qsvm_path2)
    print(f"Saved QSVM model to {qsvm_path1}")

    # 4. Save QSVM config
    config = {
        "model_name": "Quantum Support Vector Machine (QSVM)",
        "framework": "Qiskit Machine Learning",
        "quantum_feature_map": "ZZFeatureMap",
        "feature_dimension": NUM_QUBITS,
        "reps": REPS,
        "entanglement": ENTANGLEMENT,
        "scale_factor": SCALE_FACTOR,
        "kernel_type": "FidelityQuantumKernel (| <phi(x1)|phi(x2)> |^2)",
        "classical_classifier": "sklearn.svm.SVC(kernel='precomputed')",
        "C_parameter": C_PARAM,
        "class_weight": "balanced",
        "random_state": 42,
        "training_subset_size": subset_size,
        "test_sample_count": len(y_test),
        "qiskit_version": qiskit.__version__
    }
    config_path1 = os.path.join(REPORTS_DIR, "qsvm_config.json")
    config_path2 = os.path.join(MODELS_DIR, "qsvm_config.json")
    with open(config_path1, "w") as f:
        json.dump(config, f, indent=2)
    with open(config_path2, "w") as f:
        json.dump(config, f, indent=2)

    # 5. Save QSVM metrics
    metrics = {
        "model_name": "Quantum Support Vector Machine (QSVM)",
        "type": "Quantum Kernel Machine",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "weighted_precision": round(weighted_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "roc_auc": round(roc_auc, 4),
        "generalization_error": round(gen_error, 4),
        "training_time_seconds": round(train_time, 4),
        "inference_time_seconds": round(inf_time, 4),
        "kernel_properties": {
            "is_symmetric": is_symmetric,
            "min_eigenvalue": round(min_eigval, 6),
            "mean_diagonal": round(diag_mean, 4)
        },
        "confusion_matrix": cm,
        "report": report
    }
    metrics_path = os.path.join(REPORTS_DIR, "qsvm_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved QSVM metrics to {metrics_path}")

    # 6. Generate Figures
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Qiskit 4-Qubit QSVM (ZZ Kernel)\nTest Acc: {acc*100:.2f}%, F1: {macro_f1:.4f}, AUC: {roc_auc:.4f}",
              fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "qsvm_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved QSVM confusion matrix to {cm_path}")

    # Plot Training Quantum Kernel Heatmap (first 50x50 block for clear visualization)
    plt.figure(figsize=(7, 6))
    sns.heatmap(K_train[:50, :50], cmap="viridis", cbar=True)
    plt.title("Qiskit ZZ Quantum Kernel Gram Matrix (50x50 Subspace)", fontsize=12, fontweight="bold")
    plt.xlabel("Training Sample Index")
    plt.ylabel("Training Sample Index")
    plt.tight_layout()
    k_path = os.path.join(FIGURES_DIR, "qsvm_kernel_matrix.png")
    plt.savefig(k_path, dpi=300)
    plt.close()
    print(f"Saved quantum kernel heatmap to {k_path}")

    return metrics


def _save_qsvm_classifier(model, model_path):
    """Save fitted QSVM state without pickling the Python class object."""
    if not isinstance(model, QSVMClassifier):
        raise TypeError("model must be a fitted QSVMClassifier")
    if model.sv_train_ is None or not hasattr(model.svc, "support_"):
        raise ValueError("Cannot save an unfitted QSVMClassifier")

    payload = {
        "artifact_type": "quantum_biological_dosimetry.qsvm",
        "artifact_version": 1,
        "state": model.__getstate__(),
    }
    joblib.dump(payload, model_path, compress=3)


def load_qsvm_classifier(model_path=None):
    """Load a state-dictionary QSVM artifact and reconstruct its Python wrapper."""
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "qsvm_classifier.joblib")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"QSVM model not found at '{model_path}'. "
            "Run `python -m src.models_qsvm` to train and save it."
        )

    try:
        payload = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not load this QSVM artifact. It may be an older artifact "
            "containing an incompatible Qiskit object. Back it up and retrain "
            "using `python -m src.models_qsvm`."
        ) from exc

    if not isinstance(payload, dict) or payload.get("artifact_type") != \
            "quantum_biological_dosimetry.qsvm":
        raise ValueError(
            "Unsupported or legacy QSVM artifact format. Back up the old "
            "artifact and retrain using `python -m src.models_qsvm`."
        )

    model = QSVMClassifier()
    model.__setstate__(payload["state"])

    if model.sv_train_ is None or not hasattr(model.svc, "support_"):
        raise ValueError("The QSVM artifact does not contain a fitted classifier.")
    return model


if __name__ == "__main__":
    train_and_evaluate_qsvm()
