"""Apply the frozen AirRC skeleton/GT-branch protocol to saved HiPaS labels.

Only cohort paths, the two evaluated classes and the legacy CSV schema differ.
Skeleton extraction, branch construction, coverage, aggregation and alignment
all call the existing implementation. No training or inference is available.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import math
import multiprocessing
from pathlib import Path

import skimage

from . import evaluate_skeleton_branches as base
from .add_av_metrics import read_json, write_json


NAMES = ['pulmonary_artery', 'pulmonary_vein']


def fixed_cohort(manifest):
    source = manifest['cases']
    ids = [row['case_id'] for row in source]
    expected = {f'hipas_{index:03d}' for index in range(1, 251)}
    if len(ids) != 250 or set(ids) != expected:
        raise ValueError('The frozen public HiPaS cohort must contain exactly cases001–250 once.')
    cases = [{'case_id': row['case_id'], 'split': 'external',
              'gt_path': row['case_id'] + '.nii.gz', 'prediction_path': row['case_id'] + '.nii.gz'}
             for row in source]
    return {'dataset': 'HiPaS_public250', 'gt_label_encoding': 'unified',
            'prediction_label_encoding': 'unified', 'prediction_space': 'physical_label_image', 'cases': cases}


def legacy_scores(path, cohort):
    old = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = ('external', row['case_id'], row['structure'])
            if key in old:
                raise ValueError('Duplicate legacy HiPaS case/class score.')
            value = float(row['cldice']) if row['cldice'] else None
            old[key] = value if value is not None and math.isfinite(value) else None
    expected = {('external', row['case_id'], name) for row in cohort['cases'] for name in NAMES}
    if set(old) != expected:
        raise ValueError('Legacy scores do not exactly cover the frozen250 patients and two classes.')
    return old


def build_protocol(cohort, manifest, inherited):
    core = base.protocol(cohort)
    # The three numerical implementations must match the previously delivered
    # AirRC protocol byte-for-byte (normalizing source line endings only).
    for name in ['branch_metrics.py', 'skeleton_metrics.py', 'evaluate_airrc.py']:
        if core['source_code_sha256'][name] != inherited['source_code_sha256'][name]:
            raise ValueError(f'Frozen AirRC numerical implementation changed: {name}')
    core.pop('raw_airrc_mapping')
    core.update(dataset='HiPaS_public250', evaluated_classes=NAMES, gt_allowed_labels=[0, 1, 2],
                source_record=manifest['source_record'],
                inherited_algorithm_protocol='airrc-skeleton-branch-v1',
                same_numerical_sources_as_airrc=True,
                gt_encoding='existing unified HiPaS GT:0 background,1 artery,2 vein; original binary NPZs unchanged; overlap artery priority per author issue10',
                geometry_basis=manifest['geometry_basis'],
                airway_gt_available=False,
                labels={'0': 'background', '1': 'pulmonary_artery', '2': 'pulmonary_vein', '3': 'airway_lumen_prediction_only_not_evaluated'})
    core['source_code_sha256'][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'airrc-protocol', 'gt-root', 'prediction-root', 'previous-per-case', 'output', 'cache-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pilot-count', type=int, default=0)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    if skimage.__version__ != '0.21.0':
        raise RuntimeError('Use the original scikit-image0.21.0 evaluation runtime.')
    manifest = read_json(args.manifest)
    cohort = fixed_cohort(manifest)
    old = legacy_scores(args.previous_per_case, cohort)
    core = build_protocol(cohort, manifest, read_json(args.airrc_protocol))
    cases = cohort['cases'][:args.pilot_count] if args.pilot_count else cohort['cases']
    output = args.output.resolve()
    for name in ['cases', 'previews']:
        (output / name).mkdir(parents=True, exist_ok=True)
    core.update(gt_root=str(args.gt_root), prediction_root=str(args.prediction_root),
                previous_per_case=str(args.previous_per_case), cache_root=str(args.cache_root),
                selected_cases=[{'case_id': row['case_id'], 'split': row['split']} for row in cases])
    protocol_path = output / 'evaluation_protocol.json'
    if protocol_path.exists() and read_json(protocol_path) != core:
        raise ValueError('Existing output has a different frozen protocol or input path.')
    write_json(protocol_path, core)
    write_json(output / 'cohort.json', cohort)
    tasks = [(case, args.gt_root, args.prediction_root, output, args.cache_root, core, old, args.plots)
             for case in cases if not (output / 'cases' / f"external_{case['case_id']}.json").exists()]
    completed = len(cases) - len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(base.evaluate_one, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            failed = sum(row['status'] != 'computed' for row in result['rows'])
            mismatch = sum(row.get('cldice_matches_legacy') is False for row in result['rows'])
            print(f"HIPAS_BRANCH {completed}/{len(cases)} {result['case_id']} {result['seconds']:.1f}s "
                  f'failed_rows={failed} cldice_mismatches={mismatch}', flush=True)
            if completed % 8 == 0 or completed == len(cases):
                base.aggregate(output, cases, core)
    summary = base.aggregate(output, cases, core)
    if summary['failed_structure_rows'] or summary['legacy_cldice_mismatches']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
