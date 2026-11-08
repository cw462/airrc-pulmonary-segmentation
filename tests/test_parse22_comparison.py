import unittest

from parse22_supervised.compare import paired_bootstrap


class PairedBootstrapTests(unittest.TestCase):
    def test_pairs_by_id_after_reordering_and_constant_difference(self):
        a = [{'case_id': f'PA{i:06d}', 'dice': .3 + i * .01} for i in range(20)]
        b = [dict(row, dice=row['dice'] + .02) for row in a[::-1]]
        result = paired_bootstrap(a, b, 'dice')
        self.assertEqual(result['n_pairs'], 20)
        for key in ['mean_difference', 'ci95_low', 'ci95_high']:
            self.assertAlmostEqual(result[key], .02, places=12)

    def test_undefined_values_are_not_imputed(self):
        a = [{'case_id': str(i), 'recall': .5} for i in range(20)]
        b = [dict(row, recall=.6) for row in a]
        b[3]['recall'] = float('nan')
        result = paired_bootstrap(a, b, 'recall')
        self.assertEqual(result['n_pairs'], 19)
        self.assertEqual(result['undefined_case_ids'], ['3'])
        self.assertAlmostEqual(result['mean_difference'], .1)

    def test_missing_patient_or_duplicate_is_rejected(self):
        a = [{'case_id': str(i), 'dice': .5} for i in range(20)]
        for b in [a[:-1], a[:-1] + a[:1]]:
            with self.assertRaises(ValueError):
                paired_bootstrap(a, b, 'dice')


if __name__ == '__main__':
    unittest.main()
