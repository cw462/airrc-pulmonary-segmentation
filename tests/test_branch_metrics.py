import math
import unittest

import numpy as np

from evaluation.branch_metrics import extract_reference_branches, branch_coverages
from evaluation.skeleton_metrics import skeleton_coordinates, score_skeletons


def score(gt, pred, spacing=(1, 1, 1), direct_skeleton=False):
    coords = np.argwhere(gt) if direct_skeleton else skeleton_coordinates(gt)
    graph = extract_reference_branches(coords, spacing)
    metrics, details = branch_coverages(graph, pred)
    skel, _ = score_skeletons(gt, pred, coords)
    return {**metrics, **skel}, graph, details


class BranchMetricTests(unittest.TestCase):
    def line(self):
        gt = np.zeros((5, 7, 32), dtype=bool)
        gt[2, 3, 3:29] = True
        return gt

    def test_perfect_prediction(self):
        gt = self.line()
        metrics, _, _ = score(gt, gt.copy())
        for key in ('skeleton_recall', 'skeleton_precision', 'cldice_recomputed',
                    'branch_detection_rate_80', 'branch_detection_rate_90', 'branch_coverage_mean'):
            self.assertEqual(metrics[key], 1)
        self.assertEqual(metrics['gt_branch_count'], 1)

    def test_complete_miss(self):
        gt = self.line()
        metrics, _, _ = score(gt, np.zeros_like(gt))
        self.assertEqual(metrics['skeleton_recall'], 0)
        self.assertEqual(metrics['branch_detection_rate_80'], 0)
        self.assertEqual(metrics['cldice_recomputed'], 0)
        self.assertTrue(math.isnan(metrics['skeleton_precision']))

    def test_y_branch_count_and_one_missing_arm(self):
        gt = np.zeros((3, 27, 27), dtype=bool)
        gt[1, 13:24, 13] = True
        for offset in range(1, 10):
            gt[1, 13-offset, 13-offset] = True
            gt[1, 13-offset, 13+offset] = True
        pred = gt.copy()
        for offset in range(1, 10):
            pred[1, 13-offset, 13+offset] = False
        metrics, _, details = score(gt, pred)
        self.assertEqual(metrics['gt_branch_count'], 3)
        self.assertEqual(metrics['detected_branch_count_80'], 2)
        self.assertEqual(sum(item['detected80'] is False for item in details if not item['exclusion_reason']), 1)
        self.assertAlmostEqual(metrics['branch_detection_rate_80'], 2/3)

    def test_short_gap_can_pass_bdr_without_continuity(self):
        gt = self.line()
        pred = gt.copy()
        pred[2, 3, 15] = False
        metrics, _, _ = score(gt, pred)
        self.assertLess(metrics['skeleton_recall'], 1)
        self.assertEqual(metrics['branch_detection_rate_80'], 1)
        self.assertEqual(metrics['gt_branch_count'], 1)

    def test_false_prediction_branch_reduces_precision_not_gt_denominator(self):
        gt = self.line()
        extra = gt.copy()
        extra[2, 0:3, 15] = True
        base, _, _ = score(gt, gt.copy())
        metrics, _, _ = score(gt, extra)
        self.assertEqual(metrics['gt_branch_count'], base['gt_branch_count'])
        self.assertEqual(metrics['branch_detection_rate_80'], base['branch_detection_rate_80'])
        self.assertLess(metrics['skeleton_precision'], base['skeleton_precision'])

    def test_physical_length_anisotropic_and_rotated_direction(self):
        coords = np.array([[2, 3, x] for x in range(7)], dtype=np.int32)
        graph = extract_reference_branches(coords, spacing_xyz=(2, 3, 5),
                                          direction=np.array([[0,-1,0],[1,0,0],[0,0,1]]))
        self.assertEqual(len(graph['branches']), 1)
        self.assertAlmostEqual(graph['branches'][0]['length_mm'], 12)
        diagonal = extract_reference_branches(np.array([[i,i,i] for i in range(8)]), (2,3,5))
        self.assertAlmostEqual(diagonal['branches'][0]['length_mm'], 7 * math.sqrt(38))

    def test_cycle_isolated_short_components_are_recorded(self):
        gt = np.zeros((3, 35, 40), dtype=bool)
        for y in range(3, 14):
            for x in range(3, 14):
                if abs(y-8)+abs(x-8) == 5:
                    gt[1,y,x] = True
        gt[1,25,3] = True
        gt[1,25,20:22] = True
        metrics, _, details = score(gt, gt.copy(), direct_skeleton=True)
        self.assertEqual(metrics['excluded_cycle_component_count'], 1)
        self.assertEqual(metrics['excluded_short_branch_count'], 1)
        self.assertEqual(metrics['degenerate_structure_count'], 1)
        self.assertEqual(metrics['gt_branch_count'], 0)
        self.assertTrue(math.isnan(metrics['branch_detection_rate_80']))
        self.assertEqual({item['exclusion_reason'] for item in details},
                         {'cycle_component', 'short_branch', 'degenerate_isolated_voxel'})

    def test_all_connected_components_are_used(self):
        gt = self.line()
        gt[2, 0, 3:12] = True
        metrics, _, _ = score(gt, gt.copy())
        self.assertEqual(metrics['gt_branch_count'], 2)
        self.assertEqual(metrics['detected_branch_count_80'], 2)

    def test_straight_diagonal_and_bend_have_no_false_short_branches(self):
        for coords in (np.array([[i,i,i] for i in range(12)]),
                       np.array([[1,3,x] for x in range(3,12)] + [[1,y,11] for y in range(4,12)])):
            graph = extract_reference_branches(coords)
            shape = tuple(coords.max(axis=0)+2)
            pred = np.ones(shape, dtype=bool)
            metrics, _ = branch_coverages(graph, pred)
            self.assertEqual(metrics['gt_branch_count'], 1)
            self.assertEqual(metrics['excluded_short_branch_count'], 0)
            self.assertEqual(metrics['excluded_cycle_component_count'], 0)

    def test_gt_empty_and_legacy_cldice_policy(self):
        empty = np.zeros((3,3,3), dtype=bool)
        metrics, _, _ = score(empty, empty)
        self.assertTrue(math.isnan(metrics['skeleton_recall']))
        self.assertTrue(math.isnan(metrics['cldice_recomputed']))
        self.assertTrue(math.isnan(metrics['branch_detection_rate_80']))
        self.assertEqual(metrics['gt_branch_count'], 0)

    def test_edge_half_weight_coverage_and_90_sensitivity(self):
        coords = np.array([[1,1,x] for x in range(11)])
        graph = extract_reference_branches(coords)
        pred = np.ones((3,3,12), dtype=bool)
        pred[1,1,5] = False
        metrics, details = branch_coverages(graph, pred)
        self.assertAlmostEqual(details[0]['coverage'], 0.9)
        self.assertEqual(metrics['branch_detection_rate_80'], 1)
        self.assertEqual(metrics['branch_detection_rate_90'], 1)
        # Endpoint contributes half one edge, not a full voxel-length unit.
        pred[1,1,0] = False
        metrics, details = branch_coverages(graph, pred)
        self.assertAlmostEqual(details[0]['coverage'], 0.85)
        self.assertEqual(metrics['branch_detection_rate_80'], 1)
        self.assertEqual(metrics['branch_detection_rate_90'], 0)

    def test_perfect_mixed_diagonal_lengths_stay_exactly_in_unit_interval(self):
        coords = np.array([[1,1,1],[1,2,2],[1,2,3],[2,3,4],[2,4,5],[2,4,6],[3,5,7]])
        graph = extract_reference_branches(coords)
        _, details = branch_coverages(graph, np.ones((6,8,10), dtype=bool))
        for row in details:
            if row['length_mm'] > 0:
                self.assertEqual(row['coverage'], 1.0)


if __name__ == '__main__':
    unittest.main()
