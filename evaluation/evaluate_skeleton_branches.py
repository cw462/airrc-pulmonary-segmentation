"""Evaluate saved AirRC labels; never run a network or change original scores."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import scipy
import SimpleITK as sitk
import skimage

from .add_av_metrics import read_json, write_json, validate_cohort
from .branch_metrics import ALGORITHM_VERSION, extract_reference_branches, branch_coverages
from .label_geometry import align_prediction, geometry, read_label
from .skeleton_metrics import skeleton_coordinates, score_skeletons


from evaluate_airrc import LABELS as CLASSES
RATIOS = ('skeleton_recall', 'skeleton_precision', 'cldice_recomputed',
          'branch_detection_rate_80', 'branch_detection_rate_90', 'branch_coverage_mean')
COUNTS = ('gt_branch_count', 'detected_branch_count_80', 'detected_branch_count_90',
          'excluded_short_branch_count', 'excluded_cycle_component_count')
METRICS = RATIOS + COUNTS
EXTRA = ('excluded_short_branch_total_length_mm', 'degenerate_structure_count',
         'junction_cluster_count', 'suppressed_degree2_cluster_count',
         'gt_skeleton_voxel_count', 'pred_skeleton_voxel_count')


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return clean(value.item())
    return value


def code_versions():
    from evaluate_airrc import __file__ as legacy_file
    files = [Path(__file__), Path(__file__).with_name('branch_metrics.py'),
             Path(__file__).with_name('skeleton_metrics.py'), Path(legacy_file)]
    return {path.name: hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for path in files}


def protocol(cohort):
    return {
        'protocol_id': 'airrc-skeleton-branch-v1', 'source_code_sha256': code_versions(),
        'python': platform.python_version(), 'skimage': skimage.__version__, 'scipy': scipy.__version__,
        'skeleton_extractor': 'evaluate_airrc.skeletonize(mask, method=lee).astype(bool)',
        'skeleton_count_weighting': 'voxel counts; no dilation or spatial tolerance',
        'gt_cache': 'GT-only tight bounding-box skeleton, stored in full reference ZYX indices; prediction independent',
        'connectivity': 26,
        'junction_cluster_rule': '26-connected degree>=3 voxel clusters; absorb degree2 local returns with both neighbours in same cluster',
        'corner_rule': 'clusters with exactly two external paths are nonbranching bends; join paths with shortest internal 26-neighbour connection',
        'junction_internal_length': 'excluded for true junction clusters; nonbranching bend connectors belong to the degree2 path',
        'all_components': True, 'pure_cycle_components': 'record and exclude, including closed degree2 quotient components',
        'minimum_branch_length_mm': 3.0, 'main_coverage_threshold': 0.80, 'sensitivity_threshold': 0.90,
        'coverage_weighting': 'half physical edge length to each endpoint; branch-macro average within patient',
        'coverage_numerics': 'math.fsum numerator/denominator, roundoff-only clamp to [0,1]; thresholds unchanged, no epsilon added',
        'physical_length': 'norm(direction @ diag(spacing_xyz) @ delta_index_xyz); convert ZYX indices to XYZ',
        'spatial_tolerance_mm': 0.0,
        'labels': {'0': 'background', '1': 'pulmonary_artery', '2': 'pulmonary_vein', '3': 'airway_lumen'},
        'raw_airrc_mapping': [0, 3, 0, 1, 2], 'gt_encoding': 'already unified prepared GT',
        'gt_empty': 'skeleton_recall and BDR undefined; preserve old clDice both-empty NaN / one-empty zero',
        'prediction_empty': 'recall=0 if GT skeleton exists, precision=NaN, BDR=0 only if effective GT branches exist',
        'no_valid_gt_branches': 'BDR and coverage mean NaN, no zero/one imputation',
        'cldice_legacy_policy': 'both empty NaN; either empty zero; exact perfect nonempty mask one; otherwise max(1,skeleton count) harmonic',
        'cldice_absolute_difference_tolerance': 1e-12,
        'aggregation': 'per-patient mean/sample_std ddof=1/median/n_valid, separately by split and class; no branch pooling',
        'definition_limit': 'unrooted topological GT branches, not anatomical names or branch levels; coverage BDR does not ensure continuity',
        'fixed_cases': [{key: case[key] for key in ('case_id', 'split', 'gt_path')} for case in cohort['cases']],
    }


def load_or_build_graph(mask, gt_geometry, case, name, cache_root, core):
    header = {'algorithm': ALGORITHM_VERSION, 'branch_source': core['source_code_sha256']['branch_metrics.py'],
              'skeleton_source': core['source_code_sha256']['skeleton_metrics.py'], 'skimage': skimage.__version__,
              'case_id': case['case_id'], 'split': case['split'], 'class': name,
              'geometry': gt_geometry, 'gt_foreground_voxels': int(mask.sum()), 'minimum_length_mm': 3.0}
    cache_key = hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()[:16]
    path = cache_root / f"{case['split']}_{case['case_id']}_{name}_{cache_key}.pkl.gz"
    if path.exists():
        with gzip.open(path, 'rb') as stream:
            cached = pickle.load(stream)
        if cached['header'] != header:
            raise ValueError('GT cache protocol or reference geometry differs.')
        graph = cached['graph']
        coords = graph['coords_zyx']
        if len(coords) and not mask[tuple(coords.T)].all():
            raise ValueError('Cached GT skeleton does not lie inside the fixed GT.')
        return graph, str(path), True
    coords = skeleton_coordinates(mask)
    graph = extract_reference_branches(coords, gt_geometry['spacing_xyz_mm'], gt_geometry['direction'])
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.gz.part')
    with gzip.open(temporary, 'wb', compresslevel=3) as stream:
        pickle.dump({'header': header, 'graph': graph}, stream, protocol=4)
    temporary.replace(path)
    return graph, str(path), False


def plot_graph(graph, details, reference_geometry, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    coords = graph['coords_zyx'][:, ::-1].astype(float)
    matrix = np.asarray(reference_geometry['direction']).reshape(3, 3) @ np.diag(reference_geometry['spacing_xyz_mm'])
    points = coords @ matrix.T + np.asarray(reference_geometry['origin_lps_mm'])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    projections = [(0, 1), (0, 2), (1, 2)]
    for column, (xaxis, yaxis) in enumerate(projections):
        for index, (branch, detail) in enumerate(zip(graph['branches'], details)):
            edges = branch['edges']
            if not len(edges):
                continue
            segments = points[edges][:, :, [xaxis, yaxis]]
            excluded = bool(detail['exclusion_reason'])
            color = '0.75' if excluded else plt.cm.turbo((index * 0.61803398875) % 1)
            axes[0, column].add_collection(LineCollection(segments, colors=[color], linewidths=0.7))
            color = '0.75' if excluded else plt.cm.RdYlGn(detail['coverage'])
            axes[1, column].add_collection(LineCollection(segments, colors=[color], linewidths=0.8))
        for row in range(2):
            axis = axes[row, column]
            axis.autoscale()
            axis.set_aspect('equal')
            axis.set_xlabel('XYZ'[xaxis] + ' (mm)')
            axis.set_ylabel('XYZ'[yaxis] + ' (mm)')
        for node in graph['nodes']:
            members = points[node['members']]
            center = members.mean(axis=0)
            axes[0, column].plot(center[xaxis], center[yaxis], '.', color='black', markersize=2)
        axes[0, column].set_title('GT branches / junctions; excluded=gray')
        axes[1, column].set_title('Same-class prediction coverage: red=0, green=1')
    fig.suptitle(path.stem)
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.06, top=0.91, hspace=0.28, wspace=0.22)
    color_axis = fig.add_axes([0.95, 0.13, 0.012, 0.28])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap='RdYlGn'), cax=color_axis)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_one(task):
    case, gt_root, pred_root, output, cache_root, core, old, plots = task
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    started = time.monotonic()
    case_id, split = case['case_id'], case['split']
    gt_path = gt_root / case['gt_path']
    prediction_path = pred_root / split / f'{case_id}.nii.gz'
    payload = {'case_id': case_id, 'split': split, 'gt_path': str(gt_path),
               'prediction_path': str(prediction_path), 'rows': [], 'details': [], 'exceptions': []}
    try:
        if not prediction_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(f'gt_exists={gt_path.is_file()}, prediction_exists={prediction_path.is_file()}')
        gt_image, pred_image = read_label(gt_path, 'unified'), read_label(prediction_path, 'unified')
        pred_image, alignment = align_prediction(pred_image, gt_image)
        payload['geometry'] = alignment
        gt, pred = sitk.GetArrayViewFromImage(gt_image), sitk.GetArrayViewFromImage(pred_image)
        ref_geometry = geometry(gt_image)
        for value, name in CLASSES.items():
            base = {'case_id': case_id, 'split': split, 'class': name}
            try:
                g, p = gt == value, pred == value
                graph, cache_path, cache_hit = load_or_build_graph(g, ref_geometry, case, name, cache_root, core)
                skeletal, issues = score_skeletons(g, p, graph['coords_zyx'])
                branches, details = branch_coverages(graph, p)
                expected = old.get((split, case_id, name))
                result = skeletal['cldice_recomputed']
                finite_result = math.isfinite(result)
                delta = abs(result - expected) if finite_result and expected is not None else None
                matched = (delta <= 1e-12) if delta is not None else (not finite_result and expected is None)
                if not matched:
                    # Diagnose using the unchanged legacy implementation, without
                    # replacing any stored old values or metric definitions.
                    from evaluate_airrc import score_class
                    legacy_now = score_class(g, p, np.asarray(ref_geometry['spacing_xyz_mm'][::-1]))['cldice']
                    issues.append(('cldice_recomputed', f'legacy_mismatch old={expected} recomputed={result} current_score_class={legacy_now}; geometry={alignment["action"]}; skimage={skimage.__version__}'))
                if not branches['gt_branch_count']:
                    issues.append(('branch_detection_rate_80', 'no_effective_gt_branches_after_3mm_and_cycle_exclusions'))
                row = {**base, **skeletal, **branches, 'old_cldice': expected,
                       'cldice_absolute_difference': delta, 'cldice_matches_legacy': matched,
                       'status': 'computed', 'gt_cache_path': cache_path, 'gt_cache_hit': cache_hit}
                payload['rows'].append(row)
                payload['details'].extend({**base, **item} for item in details)
                payload['exceptions'].extend({**base, 'metric': metric, 'reason': reason} for metric, reason in issues)
                if plots:
                    plot_graph(graph, details, ref_geometry, output / 'previews' / f'{split}_{case_id}_{name}.png')
            except Exception as error:
                payload['rows'].append({**base, **dict.fromkeys(METRICS + EXTRA), 'status': 'failed'})
                payload['exceptions'].append({**base, 'metric': 'all', 'reason': f'{type(error).__name__}: {error}'})
    except Exception as error:
        for name in CLASSES.values():
            base = {'case_id': case_id, 'split': split, 'class': name}
            payload['rows'].append({**base, **dict.fromkeys(METRICS + EXTRA), 'status': 'missing' if isinstance(error, FileNotFoundError) else 'failed'})
            payload['exceptions'].append({**base, 'metric': 'all', 'reason': f'{type(error).__name__}: {error}'})
    payload['seconds'] = time.monotonic() - started
    write_json(output / 'cases' / f'{split}_{case_id}.json', clean(payload))
    return clean(payload)


def stats(values):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return {'mean': float(np.mean(finite)) if finite else None,
            'sample_std': float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            'median': float(np.median(finite)) if finite else None, 'n_valid': len(finite)}


def output_csv(path, fields, rows, nan_fields=()):
    temporary = path.with_suffix('.csv.part')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ('nan' if key in nan_fields and row.get(key) is None else row.get(key)) for key in fields})
    temporary.replace(path)


def aggregate(output, cases, core):
    paths = [output / 'cases' / f"{case['split']}_{case['case_id']}.json" for case in cases]
    records = [read_json(path) for path in paths if path.exists()]
    rows = [row for record in records for row in record['rows']]
    details = [row for record in records for row in record['details']]
    exceptions = [row for record in records for row in record['exceptions']]
    summary = {'state': 'done' if len(records) == len(cases) else 'running', 'patients_expected': len(cases),
               'patients_completed': len(records), 'protocol_id': core['protocol_id'], 'scores': {},
               'failed_structure_rows': sum(row['status'] != 'computed' for row in rows),
               'legacy_cldice_mismatches': sum(row.get('cldice_matches_legacy') is False for row in rows),
               'exception_count': len(exceptions),
               'maximum_cldice_absolute_difference': max((row['cldice_absolute_difference'] for row in rows
                                                         if row.get('cldice_absolute_difference') is not None), default=None)}
    for split in ('val', 'test'):
        summary['scores'][split] = {}
        for name in CLASSES.values():
            selected = [row for row in rows if row['split'] == split and row['class'] == name]
            summary['scores'][split][name] = {metric: stats([row.get(metric) for row in selected]) for metric in METRICS + EXTRA}
    write_json(output / 'skeleton_branch_summary.json', clean(summary))
    row_fields = ['case_id', 'split', 'class', *METRICS, *EXTRA, 'old_cldice',
                  'cldice_absolute_difference', 'cldice_matches_legacy', 'legacy_cldice_policy', 'status']
    output_csv(output / 'skeleton_branch_per_case.csv', row_fields, rows, METRICS + EXTRA)
    output_csv(output / 'branch_details.csv', ['case_id','split','class','branch_id','length_mm','coverage',
                                             'detected80','detected90','exclusion_reason'], details, ('coverage',))
    output_csv(output / 'skeleton_branch_exceptions.csv', ['case_id','split','class','metric','reason'], exceptions)
    missing = [{'case_id': record['case_id'], 'split': record['split'],
                'prediction_path': record['prediction_path'], 'gt_path': record['gt_path']}
               for record in records if any(row['status'] == 'missing' for row in record['rows'])]
    output_csv(output / 'missing_predictions.csv', ['case_id','split','prediction_path','gt_path'], missing)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'gt-root', 'prediction-root', 'previous-per-case', 'output', 'cache-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pilot-count', type=int, default=0)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    cohort = read_json(args.cohort)
    cases = validate_cohort(cohort)
    if CLASSES != {1: 'pulmonary_artery', 2: 'pulmonary_vein', 3: 'airway_lumen'} or cohort.get('gt_label_encoding') != 'unified':
        raise ValueError('Actual project label mapping differs from the requested unified encoding.')
    core = protocol(cohort)
    if skimage.__version__ != '0.21.0':
        raise RuntimeError('Use the original evaluation runtime with scikit-image 0.21.0; do not silently change skeletonization version.')
    chosen = [case for case in cases if case['split'] == 'val'][:args.pilot_count] if args.pilot_count else cases
    output = args.output.resolve()
    for name in ('cases', 'previews'):
        (output / name).mkdir(parents=True, exist_ok=True)
    core.update(gt_root=str(args.gt_root), prediction_root=str(args.prediction_root),
                previous_per_case=str(args.previous_per_case), selected_cases=[{'case_id': case['case_id'], 'split': case['split']} for case in chosen],
                cache_root=str(args.cache_root))
    protocol_path = output / 'evaluation_protocol.json'
    if protocol_path.exists() and read_json(protocol_path) != core:
        raise ValueError('Existing branch evaluation used a different frozen protocol or input path.')
    write_json(protocol_path, core)
    old = {}
    with args.previous_per_case.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            old[(row['split'], row.get('case_id') or row['patient'], row['class'])] = float(row['cldice']) if row['cldice'] else None
    expected_keys = {(case['split'], case['case_id'], name) for case in cases for name in CLASSES.values()}
    if set(old) != expected_keys:
        raise ValueError('Legacy per-case scores and fixed cohort do not cover the same patients/classes.')
    tasks = [(case, args.gt_root, args.prediction_root, output, args.cache_root, core, old, args.plots)
             for case in chosen if not (output / 'cases' / f"{case['split']}_{case['case_id']}.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(evaluate_one, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            current = aggregate(output, chosen, core)
            print(f"BRANCH {current['patients_completed']}/{len(chosen)} {result['split']} {result['case_id']} "
                  f"{result['seconds']:.1f}s failed_rows={current['failed_structure_rows']} "
                  f"cldice_mismatches={current['legacy_cldice_mismatches']}", flush=True)
    summary = aggregate(output, chosen, core)
    if summary['failed_structure_rows'] or summary['legacy_cldice_mismatches']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
