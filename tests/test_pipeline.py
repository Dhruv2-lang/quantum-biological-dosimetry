"""
Automated unit tests for data loading, preprocessing, and target leakage prevention.
Compatible with both pytest and python3 -m unittest.
"""

import os
import unittest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry")

from src.preprocessing import PreprocessingPipeline, SENSOR_FEATURES, TARGET_CLASS, TARGET_REG

DATA_PATH = "/Users/dhruvpratapsingh/.gemini/antigravity/scratch/quantum_biodosimetry/data/Quantum_Biological_Dosimetry_Dataset_5000.csv"

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.path.exists(DATA_PATH), f"Dataset file missing at {DATA_PATH}")
        self.df = pd.read_csv(DATA_PATH)

    def test_dataset_dimensions_and_columns(self):
        """Validate row count and essential sensor columns."""
        self.assertEqual(len(self.df), 5000, f"Expected 5,000 rows, found {len(self.df)}")
        for feat in SENSOR_FEATURES:
            self.assertIn(feat, self.df.columns, f"Feature {feat} missing from dataset")
        self.assertIn(TARGET_CLASS, self.df.columns)
        self.assertIn(TARGET_REG, self.df.columns)

    def test_target_leakage_absence(self):
        """Verify downstream assay variables are strictly excluded from sensor features."""
        forbidden_targets = [
            "ROS_Level", "Mitochondrial_Damage", "GammaH2AX_Foci",
            "BP53_Foci", "Cell_Viability", TARGET_REG, TARGET_CLASS
        ]
        for f in SENSOR_FEATURES:
            self.assertNotIn(f, forbidden_targets, f"CRITICAL LEAKAGE: {f} is an assay target!")

    def test_preprocessing_pipeline_fit_and_split(self):
        """Verify train/test split shapes, scaling, and PCA dimension reduction."""
        pipeline = PreprocessingPipeline(random_state=42)
        splits = pipeline.fit_and_split(self.df, test_size=0.2)

        # 80/20 split of 5000 = 4000 train, 1000 test
        self.assertEqual(splits["X_train_scaled"].shape, (4000, 7))
        self.assertEqual(splits["X_test_scaled"].shape, (1000, 7))
        self.assertEqual(splits["X_train_pca4"].shape, (4000, 4))
        self.assertEqual(splits["X_test_pca4"].shape, (1000, 4))
        self.assertEqual(len(splits["y_train_class"]), 4000)
        self.assertEqual(len(splits["y_test_class"]), 1000)

        # Check PCA explained variance > 95%
        total_var = np.sum(pipeline.pca_explained_variance_ratio_)
        self.assertGreater(total_var, 0.95, f"PCA explained variance {total_var} is below 95%")

    def test_single_sample_transformation(self):
        """Verify inference transformations on single sample inputs."""
        pipeline = PreprocessingPipeline.load()
        sample = {
            "Dose_Gy": 5.0,
            "ODMR_Shift_MHz": 14.0,
            "T1_us": 250.0,
            "T2_us": 80.0,
            "Fluorescence_Intensity": 600.0,
            "Fluorescence_Lifetime_ns": 7.5,
            "QD_Emission_nm": 555.0
        }
        res = pipeline.transform_single(sample)
        self.assertEqual(res["scaled"].shape, (1, 7))
        self.assertEqual(res["pca4"].shape, (1, 4))

if __name__ == "__main__":
    unittest.main()
