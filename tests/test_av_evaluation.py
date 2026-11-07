import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from evaluation.add_av_metrics import METRICS, evaluate
from evaluation.av_metrics import compute_av_metrics, test_av_metrics
from evaluation.label_geometry import align_prediction, read_geometry, read_label


def image(array):
    return sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8))


class MetricTests(unittest.TestCase):
    def test_requested_minimum_cases(self):
        test_av_metrics()

    def test_invalid_shape_and_label_configuration(self):
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            compute_av_metrics(np.zeros(2), np.zeros(3))
        with self.assertRaisesRegex(ValueError, "must differ"):
            compute_av_metrics(np.zeros(2), np.zeros(2), 1, 1)

    def test_false_positive_vessels_are_in_mcs_denominator(self):
        m = compute_av_metrics(np.array([1, 2, 0, 3]), np.array([2, 1, 1, 2]))
        self.assertEqual(m["av_mcs"], 2 / 6)
        self.assertEqual(m["merged_vessel_precision"], 0.5)
        self.assertEqual(m["other_to_a_voxels"], 1)
        self.assertEqual(m["other_to_v_voxels"], 1)


class GeometryTests(unittest.TestCase):
    def test_same_shape_shifted_origin_requires_physical_alignment(self):
        labels = np.zeros((3, 3, 4), dtype=np.uint8)
        labels[1, 1, 1:3] = [1, 2]
        gt = image(labels)
        shifted = np.zeros_like(labels)
        shifted[1, 1, 0:2] = [1, 2]
        pred = image(shifted)
        pred.SetOrigin((1.0, 0, 0))
        restored, report = align_prediction(pred, gt)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(restored), labels)
        self.assertEqual(report["action"], "physical_nearest_neighbor_to_gt")

    def test_changed_direction_is_not_an_array_resize(self):
        labels = np.array([[[0, 1, 2, 3], [0, 0, 0, 0]]], dtype=np.uint8)
        gt = image(labels)
        pred = image(labels[:, :, ::-1].copy())
        pred.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, 1))
        pred.SetOrigin((3, 0, 0))
        restored, _ = align_prediction(pred, gt)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(restored), labels)

    def test_spacing_and_shape_use_nearest_neighbor_physical_grid(self):
        labels = np.array([[[1, 2, 0], [0, 3, 0]]], dtype=np.uint8)
        gt = image(labels)
        pred = image(np.repeat(labels, 2, axis=2))
        pred.SetSpacing((0.5, 1, 1))
        restored, _ = align_prediction(pred, gt)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(restored), labels)

    def test_disjoint_physical_fields_of_view_are_rejected(self):
        gt, pred = image(np.ones((2, 2, 2))), image(np.ones((2, 2, 2)))
        pred.SetOrigin((1000, 0, 0))
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            align_prediction(pred, gt)

    def test_original_airrc_mapping_and_header_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.nii.gz"
            raw = image(np.array([[[0, 1, 2, 3, 4]]]))
            raw.SetSpacing((0.7, 0.8, 1.1))
            raw.SetOrigin((2, 3, 4))
            sitk.WriteImage(raw, str(path))
            mapped = read_label(path, "airrc_original")
            np.testing.assert_array_equal(sitk.GetArrayFromImage(mapped), [[[0, 3, 0, 1, 2]]])
            metadata = read_geometry(path)
            self.assertEqual(metadata["shape_zyx"], [1, 1, 5])
            self.assertEqual(metadata["origin_lps_mm"], [2, 3, 4])

    def test_float_label_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "float.nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.float32)), str(path))
            with self.assertRaisesRegex(ValueError, "integer label"):
                read_label(path, "unified")

    def test_unicode_nifti_path_preserves_geometry_and_label_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            ascii_path = Path(directory) / "source.nii.gz"
            path = Path(directory) / "人工标注.nii.gz"
            raw = image(np.array([[[0, 1, 2, 3, 4]]]))
            raw.SetOrigin((-10, 20, 30))
            raw.SetSpacing((0.8, 1, 1))
            sitk.WriteImage(raw, str(ascii_path))
            ascii_path.replace(path)
            mapped = read_label(path, "airrc_original")
            np.testing.assert_array_equal(sitk.GetArrayFromImage(mapped), [[[0, 3, 0, 1, 2]]])
            self.assertEqual(read_geometry(path)["origin_lps_mm"], [-10, 20, 30])


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gt_root, self.pred_root = self.root / "gt", self.root / "pred"
        self.gt_root.mkdir()
        for split in ("val", "test"):
            (self.pred_root / split).mkdir(parents=True)
        definitions = [
            ("val", "one", [1, 2, 0, 3], [1, 2, 0, 3]),
            ("val", "two", [1, 0, 0, 3], [0, 0, 0, 0]),
            ("test", "three", [1, 2, 0, 3], [2, 1, 0, 3]),
        ]
        cases = []
        self.original_rows = []
        for split, case_id, gt, pred in definitions:
            sitk.WriteImage(image(np.array(gt)[None, None, :]), str(self.gt_root / f"{case_id}.nii.gz"))
            sitk.WriteImage(image(np.array(pred)[None, None, :]), str(self.pred_root / split / f"{case_id}.nii.gz"))
            cases.append({"split": split, "case_id": case_id, "gt_path": f"{case_id}.nii.gz"})
            for cls in ("pulmonary_artery", "pulmonary_vein", "airway_lumen"):
                self.original_rows.append({"split": split, "patient": case_id, "class": cls,
                                           "dice": "0.77000", "precision": "0.8000"})
        self.cohort = {
            "dataset": "AirRC", "gt_av_complete": True, "gt_label_encoding": "unified",
            "prediction_label_encoding": "unified", "prediction_space": "physical_label_image", "cases": cases,
        }
        self.cohort_path = self.root / "cohort.json"
        self.save_cohort()
        self.source = self.root / "per_case.csv"
        with self.source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.original_rows[0]))
            writer.writeheader()
            writer.writerows(self.original_rows)
        self.original_bytes = self.source.read_bytes()
        self.output = self.root / "results"

    def save_cohort(self):
        self.cohort_path.write_text(json.dumps(self.cohort), encoding="utf-8")

    def run_batch(self):
        return evaluate(self.cohort_path, self.gt_root, self.pred_root, self.source, self.output, "synthetic")

    def test_patient_aggregation_nan_and_original_csv_preservation(self):
        result = self.run_batch()
        val, test = result["split_results"]["val"], result["split_results"]["test"]
        self.assertEqual(val["scores"]["av_mcs"]["mean"], 0)
        self.assertEqual(test["scores"]["av_mcs"]["mean"], 0.5)
        self.assertEqual(val["scores"]["merged_vessel_dice"]["mean"], 0.5)
        self.assertAlmostEqual(val["scores"]["merged_vessel_dice"]["sample_std"], np.sqrt(0.5))
        self.assertEqual(val["scores"]["merged_vessel_dice"]["median"], 0.5)
        self.assertEqual(val["scores"]["merged_vessel_dice"]["n_valid"], 2)
        self.assertEqual(val["scores"]["merged_vessel_precision"]["n_valid"], 1)
        self.assertEqual(val["scores"]["vein_to_artery_rate"]["invalid_cases"],
                         [{"case_id": "two", "reason": "zero_gt_vein_voxels"}])
        self.assertIsNone(test["scores"]["av_mcs"]["sample_std"])
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        with (self.output / "per_case_with_av_metrics.csv").open(newline="", encoding="utf-8") as stream:
            augmented = list(csv.DictReader(stream))
        self.assertEqual([{field: row[field] for field in self.original_rows[0]} for row in augmented],
                         self.original_rows)
        self.assertEqual(len(augmented), 9)
        with (self.output / "av_confusion_counts.csv").open(newline="", encoding="utf-8") as stream:
            counts = list(csv.DictReader(stream))
        self.assertEqual(len(counts), 3)
        self.assertEqual(counts[2]["cross_error_voxels"], "2")
        self.assertEqual(counts[2]["av_mcs_denominator_voxels"], "4")

    def test_missing_prediction_has_no_fabricated_score(self):
        (self.pred_root / "val/two.nii.gz").unlink()
        result = self.run_batch()
        self.assertEqual(result["state"], "incomplete")
        val = result["split_results"]["val"]
        self.assertEqual(val["n_missing"], 1)
        self.assertEqual(val["scores"]["av_mcs"]["n_valid"], 1)
        self.assertEqual(result["missing_files"][0]["prediction_path"], str(self.pred_root / "val/two.nii.gz"))
        geometry = json.loads((self.output / "av_geometry_checks.json").read_text(encoding="utf-8"))
        self.assertEqual(geometry[1]["comparison"], "not_performed_prediction_missing")
        self.assertIn("gt", geometry[1])

    def test_parse22_has_no_complete_av_evaluation(self):
        self.cohort["dataset"] = "PARSE22"
        self.save_cohort()
        result = self.run_batch()
        for part in result["split_results"].values():
            self.assertEqual(part["n_not_applicable"], part["n_expected"])
            for metric in METRICS:
                self.assertEqual(part["scores"][metric]["n_valid"], 0)
                self.assertIsNone(part["scores"][metric]["mean"])

    def test_copd_requires_explicit_complete_av_gt(self):
        self.cohort["dataset"] = "COPD"
        self.cohort.pop("gt_av_complete")
        self.cohort["cases"][0]["gt_av_complete"] = True
        self.save_cohort()
        result = self.run_batch()
        self.assertEqual(result["split_results"]["val"]["n_computed"], 1)
        self.assertEqual(result["split_results"]["val"]["n_not_applicable"], 1)
        self.assertEqual(result["split_results"]["test"]["n_not_applicable"], 1)

    def test_network_array_without_physical_restoration_is_rejected(self):
        self.cohort["prediction_space"] = "network_array"
        self.save_cohort()
        with self.assertRaisesRegex(ValueError, "restore nnU-Net"):
            self.run_batch()


if __name__ == "__main__":
    unittest.main()
