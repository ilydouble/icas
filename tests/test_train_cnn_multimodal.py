"""Unit tests for multimodal CNN training helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_cnn_multimodal import (
    ASR_TOP9_FEATURES,
    CLINICAL_TOP3_FEATURES,
    ClinicalResidualFusionModel,
    ThermalStructuredFusionModel,
    broadcast_patient_features_to_samples,
    build_patient_sample_weight_lookup,
    choose_structured_feature_names,
    load_initial_thermal_weights,
    load_structured_feature_table,
)


class StructuredFeatureTableTests(unittest.TestCase):
    def test_choose_structured_feature_names_supports_branch_ablation(self):
        asr_only, clinical_only = choose_structured_feature_names(disable_asr=False, disable_clinical=False)
        self.assertEqual(asr_only, ASR_TOP9_FEATURES)
        self.assertEqual(clinical_only, CLINICAL_TOP3_FEATURES)

        asr_only, clinical_only = choose_structured_feature_names(disable_asr=True, disable_clinical=False)
        self.assertEqual(asr_only, [])
        self.assertEqual(clinical_only, CLINICAL_TOP3_FEATURES)

        asr_only, clinical_only = choose_structured_feature_names(disable_asr=False, disable_clinical=True)
        self.assertEqual(asr_only, ASR_TOP9_FEATURES)
        self.assertEqual(clinical_only, [])

        with self.assertRaises(ValueError):
            choose_structured_feature_names(disable_asr=True, disable_clinical=True)

    def test_load_structured_feature_table_merges_filtered_asr_and_clinical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            asr_csv = root / "asr.csv"
            clinical_csv = root / "clinical.csv"

            pd.DataFrame(
                {
                    "canonical_patient_id": ["P1", "P2", "P3"],
                    "clinical_match_status": ["matched", "matched", "missing"],
                    "label": [1, 0, 1],
                    "stenosis_multiclass": [2, 0, 3],
                    **{name: [idx + 0.1, idx + 1.1, idx + 2.1] for idx, name in enumerate(ASR_TOP9_FEATURES)},
                }
            ).to_csv(asr_csv, index=False)

            pd.DataFrame(
                {
                    "canonical_patient_id": ["P1", "P2", "P4"],
                    "label": [1, 0, 1],
                    "stenosis_multiclass": [2, 0, 1],
                    "waist_hip_ratio": [0.90, 0.80, 0.88],
                    "gender_encoded": [1, 0, 1],
                    "height": [170.0, 160.0, 168.0],
                    "bmi": [24.0, 22.5, 23.1],
                }
            ).to_csv(clinical_csv, index=False)

            merged = load_structured_feature_table(asr_csv, clinical_csv)

            self.assertEqual(merged["canonical_patient_id"].tolist(), ["P1", "P2"])
            self.assertEqual(
                merged.columns.tolist(),
                ["canonical_patient_id", "label", "stenosis_multiclass", *ASR_TOP9_FEATURES, *CLINICAL_TOP3_FEATURES],
            )

    def test_load_structured_feature_table_can_disable_one_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            asr_csv = root / "asr.csv"
            clinical_csv = root / "clinical.csv"

            pd.DataFrame(
                {
                    "canonical_patient_id": ["P1", "P2"],
                    "clinical_match_status": ["matched", "matched"],
                    "label": [1, 0],
                    "stenosis_multiclass": [2, 0],
                    **{name: [idx + 0.1, idx + 1.1] for idx, name in enumerate(ASR_TOP9_FEATURES)},
                }
            ).to_csv(asr_csv, index=False)

            pd.DataFrame(
                {
                    "canonical_patient_id": ["P1", "P2"],
                    "label": [1, 0],
                    "stenosis_multiclass": [2, 0],
                    "waist_hip_ratio": [0.90, 0.80],
                    "gender_encoded": [1, 0],
                    "height": [170.0, 160.0],
                }
            ).to_csv(clinical_csv, index=False)

            clinical_only = load_structured_feature_table(
                asr_csv,
                clinical_csv,
                asr_features=[],
                clinical_features=CLINICAL_TOP3_FEATURES,
            )
            self.assertEqual(
                clinical_only.columns.tolist(),
                ["canonical_patient_id", "label", "stenosis_multiclass", *CLINICAL_TOP3_FEATURES],
            )

            asr_only = load_structured_feature_table(
                asr_csv,
                clinical_csv,
                asr_features=ASR_TOP9_FEATURES,
                clinical_features=[],
            )
            self.assertEqual(
                asr_only.columns.tolist(),
                ["canonical_patient_id", "label", "stenosis_multiclass", *ASR_TOP9_FEATURES],
            )


class PatientBalancingTests(unittest.TestCase):
    def test_build_patient_sample_weight_lookup_balances_multi_image_patients(self):
        samples = [
            {"sample_id": "S1", "canonical_patient_id": "P1"},
            {"sample_id": "S2", "canonical_patient_id": "P1"},
            {"sample_id": "S3", "canonical_patient_id": "P1"},
            {"sample_id": "S4", "canonical_patient_id": "P2"},
        ]

        weights = build_patient_sample_weight_lookup(samples)

        self.assertAlmostEqual(weights["S1"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(weights["S2"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(weights["S3"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(weights["S4"], 1.0, places=6)


class PatientFeatureBroadcastTests(unittest.TestCase):
    def test_broadcast_patient_features_to_samples_repeats_patient_vector(self):
        samples = [
            {"sample_id": "S1", "canonical_patient_id": "P1"},
            {"sample_id": "S2", "canonical_patient_id": "P1"},
            {"sample_id": "S3", "canonical_patient_id": "P2"},
        ]
        patient_features = {
            "P1": np.array([1.0, 2.0], dtype=np.float32),
            "P2": np.array([3.0, 4.0], dtype=np.float32),
        }

        sample_features = broadcast_patient_features_to_samples(samples, patient_features)

        self.assertTrue(np.allclose(sample_features["S1"], [1.0, 2.0]))
        self.assertTrue(np.allclose(sample_features["S2"], [1.0, 2.0]))
        self.assertTrue(np.allclose(sample_features["S3"], [3.0, 4.0]))


class FusionModelTests(unittest.TestCase):
    def test_fusion_model_forward_supports_multitask(self):
        model = ThermalStructuredFusionModel(
            model_name="deeper",
            structured_dim=len(ASR_TOP9_FEATURES) + len(CLINICAL_TOP3_FEATURES),
            dropout=0.3,
            in_channels=1,
            img_size=64,
            multi_task=True,
            pretrained=False,
        )
        x_img = torch.randn(2, 1, 64, 64)
        x_struct = torch.randn(2, len(ASR_TOP9_FEATURES) + len(CLINICAL_TOP3_FEATURES))

        logits_cls, logits_sev = model(x_img, x_struct)

        self.assertEqual(tuple(logits_cls.shape), (2, 2))
        self.assertEqual(tuple(logits_sev.shape), (2, 1))

    def test_load_initial_thermal_weights_copies_matching_backbone_params(self):
        model = ThermalStructuredFusionModel(
            model_name="deeper",
            structured_dim=len(ASR_TOP9_FEATURES) + len(CLINICAL_TOP3_FEATURES),
            dropout=0.3,
            in_channels=1,
            img_size=64,
            multi_task=False,
            pretrained=False,
        )
        state_dict = model.thermal_backbone.state_dict()
        first_key = next(iter(state_dict))
        source_weight = torch.full_like(state_dict[first_key], 0.25)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "thermal.pt"
            torch.save({"model_state_dict": {first_key: source_weight}}, ckpt)
            loaded = load_initial_thermal_weights(model, ckpt, torch.device("cpu"))

        self.assertGreaterEqual(loaded, 1)
        self.assertTrue(torch.allclose(model.thermal_backbone.state_dict()[first_key], source_weight))

    def test_clinical_residual_fusion_model_returns_logits_and_multitask_head(self):
        model = ClinicalResidualFusionModel(
            model_name="mobilenet",
            clinical_dim=len(CLINICAL_TOP3_FEATURES),
            dropout=0.3,
            in_channels=1,
            img_size=64,
            multi_task=True,
            pretrained=False,
        )
        x_img = torch.randn(2, 1, 64, 64)
        x_clinical = torch.randn(2, len(CLINICAL_TOP3_FEATURES))

        logits_cls, logits_sev = model(x_img, x_clinical)

        self.assertEqual(tuple(logits_cls.shape), (2, 2))
        self.assertEqual(tuple(logits_sev.shape), (2, 1))

    def test_clinical_residual_fusion_requires_clinical_branch(self):
        with self.assertRaises(ValueError):
            ClinicalResidualFusionModel(
                model_name="deeper",
                clinical_dim=0,
                dropout=0.3,
                in_channels=1,
                img_size=64,
                multi_task=False,
                pretrained=False,
            )


if __name__ == "__main__":
    unittest.main()
