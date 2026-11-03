
"""
Preprocessing and Data Pipeline for Quantum Biological Dosimetry.

Enforces strict target leakage prevention using only the 7 quantum sensor
features. Splits data into stratified Train/Test sets, fits StandardScaler,
LabelEncoder, and 4-component PCA for the PennyLane Variational Quantum Classifier.
"""

import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


# ==========================================
# 1. CROSS-PLATFORM PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Quantum_Biological_Dosimetry_Dataset_5000.csv"

MODELS_DIR = BASE_DIR / "models"


# ==========================================
# 2. FEATURE DEFINITIONS
# ==========================================

SENSOR_FEATURES = [
    "Dose_Gy",
    "ODMR_Shift_MHz",
    "T1_us",
    "T2_us",
    "Fluorescence_Intensity",
    "Fluorescence_Lifetime_ns",
    "QD_Emission_nm"
]

TARGET_CLASS = "Radiotoxicity_Class"
TARGET_REG = "DNA_Damage_Score"

CLASS_ORDER = ["Low", "Moderate", "High"]


# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================

class PreprocessingPipeline:

    def __init__(self, random_state=42):

        self.random_state = random_state

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        self.pca_4 = PCA(
            n_components=4,
            random_state=random_state
        )

        self.feature_names = SENSOR_FEATURES.copy()
        self.class_order = CLASS_ORDER.copy()

        self.is_fitted = False
        self.pca_explained_variance_ratio_ = None

    # ==========================================
    # FIT AND SPLIT
    # ==========================================

    def fit_and_split(self, df: pd.DataFrame, test_size=0.2):

        """
        Validates columns, prevents target leakage,
        and splits the dataset into train/test sets.
        """

        # Check required columns
        required_columns = (
            self.feature_names
            + [TARGET_CLASS, TARGET_REG]
        )

        for col in required_columns:

            if col not in df.columns:
                raise ValueError(
                    f"Missing required column: {col}"
                )

        # Explicit target leakage protection
        leakage_cols = [
            "ROS_Level",
            "Mitochondrial_Damage",
            "GammaH2AX_Foci",
            "BP53_Foci",
            "Cell_Viability",
            TARGET_REG,
            TARGET_CLASS
        ]

        for col in self.feature_names:

            if col in leakage_cols:

                raise ValueError(
                    f"CRITICAL LEAKAGE ERROR: {col} "
                    "cannot be used as an input feature!"
                )

        # Extract sensor features and targets
        X = df[self.feature_names].values

        y_class_raw = df[TARGET_CLASS].values
        y_reg = df[TARGET_REG].values

        # Validate classification labels
        unknown_classes = set(y_class_raw) - set(CLASS_ORDER)

        if unknown_classes:
            raise ValueError(
                f"Unexpected classification labels: {unknown_classes}"
            )

        # Encode classification labels
        self.label_encoder.fit(CLASS_ORDER)

        y_class = self.label_encoder.transform(y_class_raw)

        # Keep class names consistent with LabelEncoder's encoding
        self.class_order = self.label_encoder.classes_.tolist()

        # Stratified train/test split
        (
            X_train,
            X_test,
            y_train_class,
            y_test_class,
            y_train_reg,
            y_test_reg
        ) = train_test_split(
            X,
            y_class,
            y_reg,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_class
        )

        # Standardization
        X_train_scaled = self.scaler.fit_transform(X_train)

        X_test_scaled = self.scaler.transform(X_test)

        # Four-component PCA
        X_train_pca4 = self.pca_4.fit_transform(X_train_scaled)

        X_test_pca4 = self.pca_4.transform(X_test_scaled)

        # Update pipeline state
        self.is_fitted = True

        self.pca_explained_variance_ratio_ = (
            self.pca_4.explained_variance_ratio_
        )

        # Store all processed splits
        data_splits = {

            "X_train_raw": X_train,
            "X_test_raw": X_test,

            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,

            "X_train_pca4": X_train_pca4,
            "X_test_pca4": X_test_pca4,

            "y_train_class": y_train_class,
            "y_test_class": y_test_class,

            "y_train_reg": y_train_reg,
            "y_test_reg": y_test_reg,

            "class_names": self.class_order.copy()
        }

        print("\nPreprocessing completed successfully!")

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        print(f"Input features: {len(self.feature_names)}")
        print("PCA components: 4")

        print(f"Class encoding: {self.class_order}")

        return data_splits

    # ==========================================
    # TRANSFORM SINGLE INPUT
    # ==========================================

    def transform_single(self, input_dict: dict):

        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline must be fitted before transformation."
            )

        missing = [
            col for col in self.feature_names
            if col not in input_dict
        ]

        if missing:
            raise ValueError(
                f"Missing input features: {missing}"
            )

        row = np.array([
            [float(input_dict[col]) for col in self.feature_names]
        ])

        row_scaled = self.scaler.transform(row)

        row_pca4 = self.pca_4.transform(row_scaled)

        return {
            "raw": row,
            "scaled": row_scaled,
            "pca4": row_pca4
        }

    # ==========================================
    # TRANSFORM BATCH INPUT
    # ==========================================

    def transform_batch(self, df_input: pd.DataFrame):

        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline must be fitted before transformation."
            )

        missing = [
            col for col in self.feature_names
            if col not in df_input.columns
        ]

        if missing:
            raise ValueError(
                f"Missing input features: {missing}"
            )

        X = df_input[self.feature_names].values

        X_scaled = self.scaler.transform(X)

        X_pca4 = self.pca_4.transform(X_scaled)

        return {
            "scaled": X_scaled,
            "pca4": X_pca4
        }

    # ==========================================
    # SAVE PIPELINE
    # ==========================================

    def save(self, filepath=None):

        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted preprocessing pipeline."
            )

        if filepath is None:
            filepath = MODELS_DIR / "preprocessor_bundle.joblib"

        filepath = Path(filepath)

        # Automatically create parent directory
        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        bundle = {

            "scaler": self.scaler,
            "pca_4": self.pca_4,
            "label_encoder": self.label_encoder,

            "feature_names": self.feature_names,
            "class_order": self.class_order,

            "pca_explained_variance_ratio_":
                self.pca_explained_variance_ratio_,

            "is_fitted": self.is_fitted,
            "random_state": self.random_state
        }

        joblib.dump(bundle, filepath)

        print(f"Saved preprocessor bundle to {filepath}")

    # ==========================================
    # LOAD PIPELINE
    # ==========================================

    @classmethod
    def load(cls, filepath=None):

        if filepath is None:
            filepath = MODELS_DIR / "preprocessor_bundle.joblib"

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(
                f"Preprocessor bundle not found: {filepath}"
            )

        bundle = joblib.load(filepath)

        instance = cls(
            random_state=bundle.get("random_state", 42)
        )

        instance.scaler = bundle["scaler"]
        instance.pca_4 = bundle["pca_4"]

        instance.label_encoder = bundle["label_encoder"]

        instance.feature_names = bundle["feature_names"]
        instance.class_order = bundle["class_order"]

        instance.pca_explained_variance_ratio_ = (
            bundle["pca_explained_variance_ratio_"]
        )

        instance.is_fitted = bundle["is_fitted"]

        return instance


# ==========================================
# 4. BUILD AND SAVE PIPELINE
# ==========================================

def build_and_save_pipeline():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Ensure the CSV file exists inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Loaded dataset: {df.shape[0]} rows, "
        f"{df.shape[1]} columns"
    )

    pipeline = PreprocessingPipeline(random_state=42)

    splits = pipeline.fit_and_split(df)

    # Save pipeline
    pipeline.save()

    # Ensure models directory exists
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed data splits
    splits_path = MODELS_DIR / "data_splits.joblib"

    joblib.dump(splits, splits_path)

    print(f"Saved processed data splits to {splits_path}")

    print(
        "PCA 4-component explained variance: "
        f"{np.sum(pipeline.pca_explained_variance_ratio_):.4f}"
    )

    return pipeline, splits


# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":

    build_and_save_pipeline()
