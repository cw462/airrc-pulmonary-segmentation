import csv
from pathlib import Path
import tempfile
import unittest

from evaluation.evaluate_hipas_skeleton_branches import fixed_cohort, legacy_scores
from evaluation.evaluate_skeleton_branches import selected_classes


class HiPaSBranchScopeTests(unittest.TestCase):
    def manifest(self):
        return {'cases': [{'case_id': f'hipas_{index:03d}'} for index in range(1, 251)]}

    def test_fixed_external_cohort_and_flat_prediction_paths(self):
        cohort = fixed_cohort(self.manifest())
        self.assertEqual(len(cohort['cases']), 250)
        self.assertEqual({row['split'] for row in cohort['cases']}, {'external'})
        self.assertEqual(cohort['cases'][0]['prediction_path'], 'hipas_001.nii.gz')

    def test_missing_or_duplicate_cohort_is_rejected(self):
        manifest = self.manifest()
        manifest['cases'][-1] = manifest['cases'][0]
        with self.assertRaises(ValueError):
            fixed_cohort(manifest)

    def test_airrc_default_stays_three_classes_and_hipas_is_two(self):
        self.assertEqual(len(selected_classes({})), 3)
        actual = selected_classes({'evaluated_classes': ['pulmonary_artery', 'pulmonary_vein']})
        self.assertEqual(actual, {1: 'pulmonary_artery', 2: 'pulmonary_vein'})
        with self.assertRaises(ValueError):
            selected_classes({'evaluated_classes': ['unknown']})

    def test_legacy_scores_require_every_patient_and_both_classes(self):
        cohort = fixed_cohort(self.manifest())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scores.csv'
            rows = [{'case_id': row['case_id'], 'structure': name, 'cldice': 0.8}
                    for row in cohort['cases'] for name in ['pulmonary_artery', 'pulmonary_vein']]
            def save(items):
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['case_id', 'structure', 'cldice'])
                    writer.writeheader()
                    writer.writerows(items)
            save(rows)
            self.assertEqual(len(legacy_scores(path, cohort)), 500)
            save(rows[:-1])
            with self.assertRaises(ValueError):
                legacy_scores(path, cohort)
            save(rows + rows[:1])
            with self.assertRaises(ValueError):
                legacy_scores(path, cohort)


if __name__ == '__main__':
    unittest.main()
