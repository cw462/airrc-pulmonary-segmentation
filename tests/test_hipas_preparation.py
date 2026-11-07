from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import SimpleITK as sitk

from evaluation.hipas_external import prepare_case, read_npz_data


class HiPaSPreparationTests(unittest.TestCase):
    def test_transient_archive_read_reopens_original_without_changing_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ct.npz'
            expected = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
            np.savez_compressed(path, data=expected)
            original_load = np.load
            attempts = []

            def flaky_load(*args, **kwargs):
                attempts.append(1)
                if len(attempts) == 1:
                    raise zipfile.BadZipFile('simulated transient archive read')
                return original_load(*args, **kwargs)

            with patch('evaluation.hipas_external.np.load', side_effect=flaky_load), \
                    patch('evaluation.hipas_external.time.sleep'):
                np.testing.assert_array_equal(read_npz_data(path), expected)
            self.assertEqual(len(attempts), 2)

    def test_persistently_invalid_archive_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ct.npz'
            path.write_bytes(b'invalid ZIP archive')
            with patch('evaluation.hipas_external.np.load',
                       side_effect=zipfile.BadZipFile('invalid archive')) as loader, \
                    patch('evaluation.hipas_external.time.sleep'):
                with self.assertRaises(zipfile.BadZipFile):
                    read_npz_data(path)
            self.assertEqual(loader.call_count, 3)

    def test_raw_hu_axis_spacing_and_author_overlap_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ct = np.arange(24, dtype=np.int16).reshape(2, 3, 4) * 100 - 1000
            artery = np.zeros_like(ct)
            vein = np.zeros_like(ct)
            artery[0, 1, 2] = 1
            vein[0, 1, 2] = 1
            vein[1, 2, 3] = 1
            for name, data in [('ct', ct), ('artery', artery), ('vein', vein)]:
                np.savez_compressed(root / (name + '.npz'), data=data)
            ct_path, gt, provenance = prepare_case({'spacing_xyz_mm': [0.7, 0.8, 1.2]}, root, root, root / 'gt.nii.gz')
            np.testing.assert_array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))), ct.transpose(2, 0, 1))
            expected = np.zeros_like(ct, dtype=np.uint8)
            expected[0, 1, 2], expected[1, 2, 3] = 1, 2
            np.testing.assert_array_equal(sitk.GetArrayFromImage(gt), expected.transpose(2, 0, 1))
            self.assertEqual(gt.GetSpacing(), (0.7, 0.8, 1.2))
            self.assertEqual(gt.GetSize(), (3, 2, 4))
            self.assertEqual(provenance['overlap_voxels_assigned_artery'], 1)
            self.assertEqual(provenance['ct_intensity_conversion'], 'none; actual source is raw HU')

    def test_normalized_values_are_not_mistaken_for_raw_hu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('ct', 'artery', 'vein'):
                np.savez_compressed(root / (name + '.npz'), data=np.zeros((2, 3, 4), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'Unexpected CT intensity'):
                prepare_case({'spacing_xyz_mm': [1, 1, 1]}, root, root, root / 'gt.nii.gz')


if __name__ == '__main__':
    unittest.main()
