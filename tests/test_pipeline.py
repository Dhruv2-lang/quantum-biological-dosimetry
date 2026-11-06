"""
Automated unit tests for data loading, preprocessing, and target leakage prevention.
Compatible with both pytest and python -m unittest.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve the project root from this test file, not from a user-specific path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "Quantum_Biological_Dosimetry_Dataset_5000.csv"
)

# Import the actual feature definitions and preprocessing implementation.
from src.preprocessing import (  # noqa: E402
    PreprocessingPipeline,
    SENSOR_FEATURES,
    TARGET_CLASS,
    TARGET_REG,
)


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            DATA_PATH.is_file(),
            f"Dataset file missing at {DATA_PATH}",
        )
        self.df = pd.read_csv(DATA_PATH)

    def test_dataset_dimensions_and_columns(self):
        """Validate row count and essential sensor columns."""
        self.assertEqual(
            len(self.df),
            5000,
            f"Expected 5,000 rows, found {len(self.df)}",
        )

        for feature in SENSOR_FEATURES:
            self.assertIn(
                feature,
                self.df.columns,
                f"Feature {feature} missing from dataset",
            )

        self.assertIn(TARGET_CLASS, self.df.columns)
        self.assertIn(TARGET_REG, self.df.columns)

    def test_target_leakage_absence(self):
        """Verify downstream assay variables are excluded from sensor inputs."""
        forbidden_targets = {
            "ROS_Level",
            "Mitochondrial_Damage",
            "GammaH2AX_Foci",
            "BP53_Foci",
            "Cell_Viability",
            TARGET_REG,
            TARGET_CLASS,
        }

        self.assertTrue(
            forbidden_targets.isdisjoint(SENSOR_FEATURES),
            "CRITICAL LEAKAGE: a target/assay variable appears in SENSOR_FEATURES",
        )

    def test_preprocessing_pipeline_fit_and_split(self):
        """Verify train/test split shapes, scaling, and PCA dimensions."""
        pipeline = PreprocessingPipeline(random_state=42)
        splits = pipeline.fit_and_split(self.df, test_size=0.2)

        # 80/20 split of 5,000 samples = 4,000 train and 1,000 test.
        self.assertEqual(splits["X_train_scaled"].shape, (4000, 7))
        self.assertEqual(splits["X_test_scaled"].shape, (1000, 7))
        self.assertEqual(splits["X_train_pca4"].shape, (4000, 4))
        self.assertEqual(splits["X_test_pca4"].shape, (1000, 4))
        self.assertEqual(len(splits["y_train_class"]), 4000)
        self.assertEqual(len(splits["y_test_class"]), 1000)

        total_variance = np.sum(pipeline.pca_explained_variance_ratio_)
        self.assertGreater(
            total_variance,
            0.95,
            f"PCA explained variance {total_variance:.4f} is below 95%",
        )

    def test_single_sample_transformation(self):
        """Verify inference transformations on a single sample."""
        pipeline = PreprocessingPipeline.load()

        sample = {
            "Dose_Gy": 5.0,
            "ODMR_Shift_MHz": 14.0,
            "T1_us": 250.0,
            "T2_us": 80.0,
            "Fluorescence_Intensity": 600.0,
            "Fluorescence_Lifetime_ns": 7.5,
            "QD_Emission_nm": 555.0,
        }

        result = pipeline.transform_single(sample)
        self.assertEqual(result["scaled"].shape, (1, 7))
        self.assertEqual(result["pca4"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
