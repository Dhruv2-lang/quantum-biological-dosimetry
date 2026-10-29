"""
PennyLane Variational Quantum Classifier (VQC) for Quantum Biological Dosimetry.
Implements a genuine 4-qubit parameterized quantum circuit with:
- AngleEmbedding for 4-dimensional quantum sensor representation
- StronglyEntanglingLayers variational ansatz
- Pauli-Z expectation measurements mapped to 3-class radiotoxicity logits
- PyTorch backpropagation optimization for training
- Pure PennyLane NumPy statevector simulation for thread-safe inference on macOS ARM64
- Eliminates conflicting OpenMP runtimes between PyTorch and XGBoost during serving.
"""

import os
import json
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

import pennylane as qml

MODELS_DIR = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/models"
REPORTS_DIR = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

NUM_QUBITS = 4
NUM_LAYERS = 2
NUM_CLASSES = 3


class TrainedQNNWrapper:
    """
    Lightweight, thread-safe inference wrapper for deployment and Streamlit dashboard.
    Uses pure PennyLane NumPy statevector simulation to avoid conflicting native
    OpenMP runtimes (libomp) between PyTorch and XGBoost on macOS ARM64.
    """
    def __init__(self, model_state, num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Extract weights as numpy arrays (handles torch tensors or numpy arrays)
        q_w = model_state["q_weights"]
        sc = model_state["scale"]
        b = model_state["bias"]
        self.q_weights = q_w.detach().cpu().numpy() if hasattr(q_w, "detach") else np.array(q_w)
        self.scale = sc.detach().cpu().numpy() if hasattr(sc, "detach") else np.array(sc)
        self.bias = b.detach().cpu().numpy() if hasattr(b, "detach") else np.array(b)

        self._dev = None
        self._circuit = None

    def _get_circuit(self):
        if self._circuit is None:
            dev = qml.device("default.qubit", wires=self.num_qubits)

            @qml.qnode(dev)
            def _qnode(w, x):
                qml.AngleEmbedding(x, wires=range(self.num_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(w, wires=range(self.num_qubits))
                return [
                    qml.expval(qml.PauliZ(0)),
                    qml.expval(qml.PauliZ(1)),
                    qml.expval(qml.PauliZ(2))
                ]
            self._circuit = _qnode
        return self._circuit

    def predict_proba(self, x_np):
        circuit = self._get_circuit()
        x_arr = np.asarray(x_np, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        probs = []
        for row in x_arr:
            z = np.array(circuit(self.q_weights, row))
            logits = z * self.scale + self.bias
            # Numerically stable softmax
            shift = logits - np.max(logits)
            exp_l = np.exp(shift)
            probs.append(exp_l / np.sum(exp_l))
        return np.array(probs)

    def predict(self, x_np):
        probs = self.predict_proba(x_np)
        return np.argmax(probs, axis=1)


def load_qnn_classifier(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODELS_DIR, "qnn_vqc.joblib")
    ckpt = joblib.load(checkpoint_path)
    wrapper = TrainedQNNWrapper(
        ckpt["model_state"],
        ckpt["num_qubits"],
        ckpt["num_layers"],
        ckpt["num_classes"]
    )
    return wrapper


def _get_torch_training_module():
    """Lazily imports torch only when retraining the VQC model."""
    import torch
    import torch.nn as nn

    dev = qml.device("default.qubit", wires=NUM_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_circuit_torch(weights, features):
        qml.AngleEmbedding(features, wires=range(NUM_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2))
        ]

    class VariationalQuantumClassifier(nn.Module):
        def __init__(self, num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
            super().__init__()
            self.num_qubits = num_qubits
            self.num_layers = num_layers
            self.num_classes = num_classes
            
            weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)
            self.q_weights = nn.Parameter(torch.randn(weight_shape, dtype=torch.float64) * 0.1)
            self.scale = nn.Parameter(torch.ones(num_classes, dtype=torch.float64) * 2.0)
            self.bias = nn.Parameter(torch.zeros(num_classes, dtype=torch.float64))

        def forward(self, x):
            expvals = quantum_circuit_torch(self.q_weights, x)
            z_stack = torch.stack(expvals, dim=1)
            logits = z_stack * self.scale + self.bias
            return logits

        def predict_proba(self, x_np):
            self.eval()
            with torch.no_grad():
                x_t = torch.tensor(x_np, dtype=torch.float64)
                logits = self.forward(x_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        def predict(self, x_np):
            probs = self.predict_proba(x_np)
            return np.argmax(probs, axis=1)

    return torch, nn, VariationalQuantumClassifier


def train_and_evaluate_quantum(subset_size=360, epochs=45, batch_size=32, lr=0.03):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    torch, nn, VariationalQuantumClassifier = _get_torch_training_module()
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    splits_path = os.path.join(MODELS_DIR, "data_splits.joblib")
    splits = joblib.load(splits_path)

    X_train_pca = splits["X_train_pca4"]
    y_train = splits["y_train_class"]
    X_test_pca = splits["X_test_pca4"]
    y_test = splits["y_test_class"]
    class_names = splits["class_names"]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=42)
    sub_idx, _ = next(sss.split(X_train_pca, y_train))
    X_sub = X_train_pca[sub_idx]
    y_sub = y_train[sub_idx]

    print(f"Training PennyLane VQC on {len(X_sub)} stratified samples across {NUM_CLASSES} classes...")

    train_dataset = TensorDataset(
        torch.tensor(X_sub, dtype=torch.float64),
        torch.tensor(y_sub, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = VariationalQuantumClassifier(NUM_QUBITS, NUM_LAYERS, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    acc_history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        avg_loss = epoch_loss / total
        train_acc = correct / total
        loss_history.append(avg_loss)
        acc_history.append(train_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {avg_loss:.4f} - Train Acc: {train_acc*100:.2f}%")

    train_time = time.time() - t_start
    print(f"VQC Training complete in {train_time:.2f} seconds.")

    # Save state dict and checkpoint
    model_state = model.state_dict()
    weights_path = os.path.join(MODELS_DIR, "qnn_weights.pt")
    torch.save(model_state, weights_path)

    qnn_checkpoint = {
        "model_state": {k: v.cpu().numpy() for k, v in model_state.items()},
        "num_qubits": NUM_QUBITS,
        "num_layers": NUM_LAYERS,
        "num_classes": NUM_CLASSES
    }
    wrapper_path = os.path.join(MODELS_DIR, "qnn_vqc.joblib")
    joblib.dump(qnn_checkpoint, wrapper_path)
    print(f"Saved QNN checkpoint to {wrapper_path}")

    # Evaluate via thread-safe wrapper
    wrapper = TrainedQNNWrapper(qnn_checkpoint["model_state"], NUM_QUBITS, NUM_LAYERS, NUM_CLASSES)
    y_test_pred = wrapper.predict(X_test_pca)
    acc = float(accuracy_score(y_test, y_test_pred))
    macro_f1 = float(f1_score(y_test, y_test_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))
    macro_prec = float(precision_score(y_test, y_test_pred, average="macro"))
    macro_rec = float(recall_score(y_test, y_test_pred, average="macro"))
    cm = confusion_matrix(y_test, y_test_pred).tolist()
    report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)

    print(f"VQC Test Evaluation (N={len(y_test)}):")
    print(f"  Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}")

    # Plot Training Convergence Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(1, epochs + 1), loss_history, marker='o', color='#8a2be2', lw=2)
    axes[0].set_title("VQC Training Loss Curve (Adam Optimizer)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")

    axes[1].plot(range(1, epochs + 1), [a * 100 for a in acc_history], marker='s', color='#20b2aa', lw=2)
    axes[1].set_title("VQC Training Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    plt.tight_layout()
    curve_plot_path = os.path.join(FIGURES_DIR, "qnn_training_curve.png")
    plt.savefig(curve_plot_path, dpi=300)
    plt.close()

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"PennyLane 4-Qubit VQC\nTest Acc: {acc*100:.2f}%, Macro-F1: {macro_f1:.4f}", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    cm_plot_path = os.path.join(FIGURES_DIR, "qnn_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # Save quantum metrics JSON
    quantum_summary = {
        "model_name": "PennyLane Variational Quantum Classifier (VQC)",
        "num_qubits": NUM_QUBITS,
        "ansatz": "AngleEmbedding(Y) + StronglyEntanglingLayers",
        "num_layers": NUM_LAYERS,
        "variational_parameters": int(np.prod(qml.StronglyEntanglingLayers.shape(n_layers=NUM_LAYERS, n_wires=NUM_QUBITS))),
        "training_subset_size": subset_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_time_seconds": round(train_time, 2),
        "test_sample_count": len(y_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "loss_history": [round(l, 4) for l in loss_history],
        "accuracy_history": [round(a, 4) for a in acc_history]
    }
    q_metrics_path = os.path.join(REPORTS_DIR, "quantum_metrics.json")
    with open(q_metrics_path, "w") as f:
        json.dump(quantum_summary, f, indent=2)

    return quantum_summary

if __name__ == "__main__":
    train_and_evaluate_quantum()
