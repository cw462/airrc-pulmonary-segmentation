"""Expose the existing Lee skeletonization without changing legacy metrics."""

import math
import numpy as np

from evaluate_airrc import skeletonize


def skeleton_coordinates(mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.int32)
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(mask.any(axis=tuple(other for other in range(3) if other != axis)))
        bounds.append(slice(int(occupied[0]), int(occupied[-1]) + 1))
    cropped = np.ascontiguousarray(mask[tuple(bounds)])
    skel = skeletonize(cropped, method='lee').astype(bool)
    coords = np.column_stack(np.nonzero(skel)).astype(np.int32)
    return coords + np.asarray([bound.start for bound in bounds], dtype=np.int32)


def score_skeletons(gt, pred, gt_coords=None, pred_coords=None):
    gt, pred = np.asarray(gt, dtype=bool), np.asarray(pred, dtype=bool)
    if gt.shape != pred.shape:
        raise ValueError('Skeleton scoring requires already aligned masks.')
    gt_coords = skeleton_coordinates(gt) if gt_coords is None else np.asarray(gt_coords)
    pred_coords = skeleton_coordinates(pred) if pred_coords is None else np.asarray(pred_coords)
    ga, pa = int(np.count_nonzero(gt)), int(np.count_nonzero(pred))
    gt_hits = int(np.count_nonzero(pred[tuple(gt_coords.T)])) if len(gt_coords) else 0
    pred_hits = int(np.count_nonzero(gt[tuple(pred_coords.T)])) if len(pred_coords) else 0
    recall = gt_hits / len(gt_coords) if ga and len(gt_coords) else float('nan')
    precision = pred_hits / len(pred_coords) if len(pred_coords) else float('nan')
    # Preserve score_class's shortcuts and its max(1, skeleton_count) policy.
    if ga == pa == 0:
        cldice, policy = float('nan'), 'both_empty_legacy_nan'
    elif ga == 0 or pa == 0:
        cldice, policy = 0.0, 'one_empty_legacy_zero'
    elif ga == pa and np.array_equal(gt, pred):
        cldice, policy = 1.0, 'perfect_mask_legacy_one'
    else:
        old_recall, old_precision = gt_hits / max(1, len(gt_coords)), pred_hits / max(1, len(pred_coords))
        cldice = 2 * old_recall * old_precision / (old_recall + old_precision) if old_recall + old_precision else 0.0
        policy = 'nonempty_legacy_skeleton_harmonic'
    issues = []
    if not ga:
        issues.append(('skeleton_recall', 'gt_mask_empty'))
    elif not len(gt_coords):
        issues.append(('skeleton_recall', 'gt_mask_nonempty_but_lee_skeleton_empty'))
    if not len(pred_coords):
        issues.append(('skeleton_precision', 'prediction_mask_empty' if not pa else 'prediction_lee_skeleton_empty'))
    if not math.isfinite(cldice):
        issues.append(('cldice_recomputed', 'both_empty_legacy_nan'))
    return {
        'skeleton_recall': recall, 'skeleton_precision': precision, 'cldice_recomputed': cldice,
        'gt_skeleton_voxel_count': len(gt_coords), 'pred_skeleton_voxel_count': len(pred_coords),
        'legacy_cldice_policy': policy,
    }, issues
