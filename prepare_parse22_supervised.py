"""Check the public100 PARSE22 pairs and freeze a patient-ID split.

Pixel fingerprints are computed during the required data read to identify
duplicate scans, not as a separate server read-back checksum procedure.
No holdout statistics are fitted or used for preprocessing choices here.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f'Refusing to change an already frozen file: {path}')
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def check_pair(task):
    case_id, ct_path, gt_path = task
    row = {'case_id': case_id, 'patient_id': case_id, 'image_path': str(ct_path), 'label_path': str(gt_path), 'issues': []}
    try:
        ct_image, gt_image = nib.load(str(ct_path)), nib.load(str(gt_path))
        ct, gt = np.asanyarray(ct_image.dataobj), np.asanyarray(gt_image.dataobj)
        affine = ct_image.affine
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        direction = affine[:3, :3] / spacing
        row.update(shape_xyz=list(ct.shape), spacing_xyz_mm=spacing.tolist(), affine_ras_mm=affine.tolist(),
                   origin_ras_mm=affine[:3, 3].tolist(), direction_ras=direction.ravel().tolist(),
                   ct_dtype=str(ct.dtype), gt_dtype=str(gt.dtype), ct_spatial_units=ct_image.header.get_xyzt_units()[0])
        if ct.ndim != 3 or ct.shape != gt.shape or not np.allclose(affine, gt_image.affine, atol=1e-5, rtol=0):
            row['issues'].append('ct_gt_grid_mismatch')
        if np.any(spacing <= 0) or not np.isfinite(affine).all() or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
            row['issues'].append('invalid_physical_geometry')
        if ct_image.header.get_xyzt_units()[0] not in ('mm', 'unknown'):
            row['issues'].append('unexpected_physical_units')
        if not np.isfinite(ct).all():
            row['issues'].append('nonfinite_ct')
        unique = np.unique(gt).tolist()
        row['label_values'] = unique
        if not np.issubdtype(gt.dtype, np.integer) or not set(unique).issubset({0, 1}):
            row['issues'].append('unexpected_binary_label_values')
        foreground = gt == 1
        count = int(foreground.sum())
        row.update(foreground_voxels=count, foreground_volume_mm3=count * float(np.prod(spacing)),
                   ct_hu_min=float(ct.min()), ct_hu_max=float(ct.max()))
        if not count:
            row['issues'].append('empty_gt')
        if row['ct_hu_min'] >= -200 or row['ct_hu_max'] <= 10:
            row['issues'].append('unexpected_hu_scale')
        if count and ct.shape == gt.shape:
            values = ct[foreground]
            row.update(foreground_hu_median=float(np.median(values)),
                       foreground_below_minus700_fraction=float(np.mean(values < -700)))
            if row['foreground_below_minus700_fraction'] > 0.05:
                row['issues'].append('foreground_mostly_outside_soft_tissue_review_required')
        canonical = np.asarray(nib.as_closest_canonical(ct_image).dataobj, dtype=np.float32, order='C')
        row['canonical_hu_content_sha256'] = hashlib.sha256(memoryview(canonical)).hexdigest()
        row['status'] = 'failed' if row['issues'] else 'ok'
    except Exception as error:
        row['status'] = 'failed'
        row['issues'].append(f'{type(error).__name__}: {error}')
    return row


def fixed_split(ids):
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError('Exactly100 unique public patient-case identifiers required.')
    shuffled = np.random.default_rng(2026).permutation(sorted(ids)).tolist()
    return {'dataset': 'PARSE22_public_labeled100', 'experiment_role': 'within_dataset_supervised',
            'seed': 2026, 'generator': 'numpy.random.default_rng(2026),PCG64; sorted identifiers before permutation',
            'official_vesmamba_split_available': False,
            'official_split_lookup': {'repository': 'https://github.com/Lzpbright/VesMamba',
                                      'commit': 'd491834149b0aa399deb4bf7ee9f5ca3a2d13475',
                                      'finding': 'Only README published on main at the pinned commit; no case split or implementation'},
            'train': sorted(shuffled[:64]), 'val': sorted(shuffled[64:80]), 'test': sorted(shuffled[80:]),
            'patient_group_basis': 'Published deidentified patient-case ID; one scan per public case; no cross-ID patient linkage provided',
            'scope_note': '64/16/20 is this project split of the100 labeled public cases, not the full challenge official split',
            'prior_exposure': 'The AirRC M0 was previously evaluated externally on all100 PARSE22 cases. The20 held-out cases are not study-wide never-viewed blind data.',
            'freeze_policy': 'All P0/P1/P2 share this exact split; no test-based resampling or exclusion; from-scratch training only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--reuse-checks', action='store_true', help='Reuse the completed immutable check records, not source images.')
    parser.add_argument('--review-decisions', type=Path)
    args = parser.parse_args()
    directories = sorted(p for p in args.source.iterdir() if p.is_dir() and p.name.startswith('PA'))
    if len(directories) != 100:
        raise ValueError('Expected exactly100 public labeled PARSE22 directories.')
    tasks = []
    for directory in directories:
        pair = []
        for kind in ('image', 'label'):
            files = [p for p in (directory / kind).iterdir() if p.name.endswith(('.nii', '.nii.gz'))]
            if len(files) != 1:
                raise ValueError(f'Expected exactly one{kind} volume in {directory}')
            pair.append(files[0])
        tasks.append((directory.name, *pair))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reuse_checks:
        rows = json.loads((args.output / 'data_pair_checks.json').read_text(encoding='utf-8'))['cases']
        if {row['case_id'] for row in rows} != {task[0] for task in tasks}:
            raise ValueError('Saved check records cover a different cohort.')
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(check_pair, task) for task in tasks]):
                row = future.result()
                rows.append(row)
                print(f"CHECK {len(rows)}/100 {row['case_id']} {row['status']} {row['issues']}", flush=True)
    rows.sort(key=lambda row: row['case_id'])
    groups = defaultdict(list)
    for row in rows:
        if row.get('canonical_hu_content_sha256'):
            groups[row['canonical_hu_content_sha256']].append(row['case_id'])
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    if args.review_decisions:
        decisions = json.loads(args.review_decisions.read_text(encoding='utf-8'))
        for row in rows:
            if row['case_id'] not in decisions:
                continue
            decision = decisions[row['case_id']]
            # Only this conservative HU warning can be reviewed. Never bypass
            # broken geometry, missing/empty labels, corrupt data or duplicates.
            if row['issues'] != ['foreground_mostly_outside_soft_tissue_review_required']:
                raise ValueError('Review decisions can only resolve the low-HU heuristic warning.')
            if decision.get('decision') != 'retain_original' or not decision.get('reason'):
                raise ValueError('A documented retain-original review decision is required.')
            row['original_issues'], row['issues'] = row['issues'], []
            row['review'] = decision
            row['status'] = 'reviewed_ok'
    failures = [row for row in rows if row['status'] not in ('ok', 'reviewed_ok')]
    result = {'state': 'failed' if failures or duplicate_groups else 'checked', 'patients': len(rows),
              'failed_cases': [row['case_id'] for row in failures], 'duplicate_scan_groups': duplicate_groups,
              'patient_identity_limit': 'No cross-ID patient linkage available; content duplicate check supplements published case IDs.',
              'statistics_use': 'Integrity/geometry checks only; no normalization or planning fitted here; fit using train64 only', 'cases': rows}
    save_json(args.output / ('data_pair_checks_reviewed.json' if args.reuse_checks else 'data_pair_checks.json'), result)
    if failures or duplicate_groups:
        raise RuntimeError('Data issues require review; no split saved and no case automatically excluded.')
    split = fixed_split([row['case_id'] for row in rows])
    digest = save_json(args.split_output, split)
    (args.output / 'split_sha256.txt').write_text(digest + '\n', encoding='ascii')
    manifest = {'dataset': split['dataset'], 'experiment_role': split['experiment_role'], 'split_sha256': digest,
                'source': 'https://parse2022.grand-challenge.org/Dataset/', 'cases': []}
    for row in rows:
        partition = next(key for key in ('train', 'val', 'test') if row['case_id'] in split[key])
        manifest['cases'].append({**row, 'split': partition})
    save_json(args.output / 'split_manifest.json', manifest)
    print('FROZEN_SPLIT', digest, {key: len(split[key]) for key in ('train', 'val', 'test')}, flush=True)


if __name__ == '__main__':
    main()
