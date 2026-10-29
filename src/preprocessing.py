"""
Preprocessing and Data Pipeline for Quantum Biological Dosimetry.
Enforces strict target leakage prevention: uses only the 7 quantum sensor features.
Splits data into Stratified Train/Test sets, fits StandardScaler, LabelEncoder,
and 4-component PCA for the PennyLane Variational Quantum Classifier.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

DATA_PATH = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/data/Quantum_Biological_Dosimetry_Dataset_5000.csv"
MODELS_DIR = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/models"

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

class PreprocessingPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca_4 = PCA(n_components=4, random_state=random_state)
        self.feature_names = SENSOR_FEATURES
        self.class_order = CLASS_ORDER
        self.is_fitted = False
        self.pca_explained_variance_ratio_ = None

    def fit_and_split(self, df: pd.DataFrame, test_size=0.2):
        """
        Validates columns, checks for target leakage, and splits into train/test.
        """
        for col in self.feature_names:
            if col not in df.columns:
                raise ValueError(f"Missing required sensor feature: {col}")
        
        leakage_cols = ["ROS_Level", "Mitochondrial_Damage", "GammaH2AX_Foci", 
                        "BP53_Foci", "Cell_Viability", TARGET_REG, TARGET_CLASS]
        for col in self.feature_names:
            if col in leakage_cols:
                raise ValueError(f"CRITICAL LEAKAGE ERROR: {col} is an assay target and cannot be an input feature!")

        X = df[self.feature_names].values
        y_class_raw = df[TARGET_CLASS].values
        y_reg = df[TARGET_REG].values

        self.label_encoder.fit(self.class_order)
        y_class = self.label_encoder.transform(y_class_raw)

        X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
            X, y_class, y_reg, test_size=test_size, random_state=self.random_state, stratify=y_class
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_pca4 = self.pca_4.fit_transform(X_train_scaled)
        X_test_pca4 = self.pca_4.transform(X_test_scaled)

        self.is_fitted = True
        self.pca_explained_variance_ratio_ = self.pca_4.explained_variance_ratio_

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
            "class_names": self.class_order
        }

        return data_splits

    def transform_single(self, input_dict: dict):
        row = np.array([[float(input_dict[col]) for col in self.feature_names]])
        row_scaled = self.scaler.transform(row)
        row_pca4 = self.pca_4.transform(row_scaled)
        return {
            "raw": row,
            "scaled": row_scaled,
            "pca4": row_pca4
        }

    def transform_batch(self, df_input: pd.DataFrame):
        X = df_input[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        X_pca4 = self.pca_4.transform(X_scaled)
        return {
            "scaled": X_scaled,
            "pca4": X_pca4
        }

    def save(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "preprocessor_bundle.joblib")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save state dictionary of standard objects to avoid __main__ class pickling issues
        bundle = {
            "scaler": self.scaler,
            "pca_4": self.pca_4,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "class_order": self.class_order,
            "pca_explained_variance_ratio_": self.pca_explained_variance_ratio_,
            "is_fitted": self.is_fitted,
            "random_state": self.random_state
        }
        joblib.dump(bundle, filepath)
        print(f"Saved preprocessor bundle to {filepath}")

    @classmethod
    def load(cls, filepath=None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "preprocessor_bundle.joblib")
        bundle = joblib.load(filepath)
        instance = cls(random_state=bundle.get("random_state", 42))
        instance.scaler = bundle["scaler"]
        instance.pca_4 = bundle["pca_4"]
        instance.label_encoder = bundle["label_encoder"]
        instance.feature_names = bundle["feature_names"]
        instance.class_order = bundle["class_order"]
        instance.pca_explained_variance_ratio_ = bundle["pca_explained_variance_ratio_"]
        instance.is_fitted = bundle["is_fitted"]
        return instance


def build_and_save_pipeline():
    df = pd.read_csv(DATA_PATH)
    pipeline = PreprocessingPipeline(random_state=42)
    splits = pipeline.fit_and_split(df)
    pipeline.save()

    splits_path = os.path.join(MODELS_DIR, "data_splits.joblib")
    joblib.dump(splits, splits_path)
    print(f"Saved processed data splits to {splits_path}")
    print(f"PCA 4-component explained variance: {np.sum(pipeline.pca_explained_variance_ratio_):.4f}")
    return pipeline, splits

if __name__ == "__main__":
    build_and_save_pipeline()
