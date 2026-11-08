"""Evaluate one completed supervised group on its frozen native-grid labels."""
import argparse
import csv
import math
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import skimage
import torch

from .preprocess import DATASET, CONFIGURATION, configure_paths, read_json, write_json


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def stats(values):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return {'mean': float(np.mean(finite)) if finite else None,
            'sample_std': float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            'median': float(np.median(finite)) if finite else None, 'n_valid': len(finite)}


def write_csv(path, rows, fields):
    temporary = path.with_suffix('.csv.part')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ('nan' if row.get(key) is None and key not in ['case_id', 'split', 'empty'] else row.get(key)) for key in fields})
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--group', choices=['P0', 'P1', 'P2'], required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    args = parser.parse_args()
    if skimage.__version__ != '0.21.0':
        raise RuntimeError('Use the frozen scikit-image0.21.0 metric runtime.')
    root, group = args.root.resolve(), args.group
    shared, work = root / 'shared', root / group
    configure_paths(shared, work / 'results')
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from evaluate_airrc import predict_case, score_class, METRICS as BASIC
    from evaluation.label_geometry import read_label, geometry, same_grid
    from evaluation.skeleton_metrics import score_skeletons
    from evaluation.branch_metrics import branch_coverages
    from evaluation.evaluate_skeleton_branches import protocol, load_or_build_graph
    from .trainer import build_network, seed_everything
    complete = read_json(work / 'training_complete.json')
    assert complete['optimizer_steps'] == 100000
    config, split = read_json(work / 'experiment_config.json'), read_json(root / 'parse22_seed2026.json')
    checkpoint = Path(complete['checkpoint_best'])
    state = torch.load(checkpoint, map_location='cpu')
    if state['experiment_family'] != 'parse22_supervised' or state['group'] != group or state['run_kind'] != 'full' or state['split_sha256'] != config['split_sha256']:
        raise ValueError('Checkpoint does not belong to this independent formal experiment.')
    selected = {'group': group, 'seed': 2026, 'filename': 'checkpoint_best.pth',
                'selected_optimizer_steps': state['optimizer_steps'], 'selected_epoch': state['current_epoch'],
                'validation_dice': state['best_volume_dice'], 'training_budget_optimizer_steps': 100000,
                'split_sha256': config['split_sha256']}
    prior = read_json(work / 'evaluation_config.json')
    if prior.get('checkpoint') is not None and prior['checkpoint'] != selected:
        raise ValueError('Saved predictions belong to another selected checkpoint.')
    write_json(work / 'evaluation_config.json', {'state': 'running', 'checkpoint': selected,
               'inference': config['inference'], 'metrics': list(BASIC) + ['skeleton_recall', 'skeleton_precision', 'branch_detection_rate_80', 'branch_detection_rate_90'],
               'class': 'pulmonary_artery', 'av_metrics_computed': False})
    seed_everything(2026)
    plans = read_json(shared / 'preprocessed' / DATASET / 'nnUNetPlans.json')
    dataset = read_json(shared / 'raw' / DATASET / 'dataset.json')
    manager = PlansManager(plans)
    configuration = manager.get_configuration(CONFIGURATION)
    network, _ = build_network(manager, configuration, dataset, group, torch.device('cuda', 0), deep_supervision=False)
    network.load_state_dict(state['network_weights'], strict=True)
    network.eval()
    del state
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
                 perform_everything_on_device=True, device=torch.device('cuda', 0),
                 verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(network, manager, configuration, None, dataset, 'Parse22Trainer', None)
    lookup = {row['case_id']: row for row in read_json(args.source_manifest)['cases']}
    cases = [{'case_id': case_id, 'split': partition, 'gt_path': case_id + '.nii.gz'}
             for partition in ['val', 'test'] for case_id in split[partition]]
    core = protocol({'cases': cases})
    core.update(dataset=DATASET, evaluated_classes=['pulmonary_artery'], gt_allowed_labels=[0, 1],
                labels={'0': 'background', '1': 'pulmonary_artery'})
    core.pop('raw_airrc_mapping')
    core['gt_encoding'] = 'native PARSE22 binary0 background/1 pulmonary_artery; not AirRC mapping'
    inherited = read_json(root / 'reference_branch_protocol.json')
    for name in ['branch_metrics.py', 'skeleton_metrics.py', 'evaluate_airrc.py']:
        if core['source_code_sha256'][name] != inherited['source_code_sha256'][name]:
            raise ValueError('Frozen skeleton/branch metric implementation changed.')
    write_json(work / 'branch_evaluation_protocol.json', core)
    (work / 'case_results').mkdir(exist_ok=True)
    rows, details, exceptions, predictions = [], [], [], []
    for case in cases:
        case_id, partition = case['case_id'], case['split']
        result_path = work / 'case_results' / (partition + '_' + case_id + '.json')
        prediction_path = work / 'predictions' / partition / (case_id + '.nii.gz')
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path.exists():
            result = read_json(result_path)
            if result['checkpoint'] != selected or not prediction_path.exists():
                raise ValueError('Existing per-case result or prediction has inconsistent provenance.')
        else:
            target_image = read_label(Path(lookup[case_id]['label_path']), 'unified')
            gt = sitk.GetArrayViewFromImage(target_image)
            if not set(np.unique(gt)).issubset({0, 1}):
                raise ValueError('PARSE22 GT must be binary.')
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            if prediction_path.exists():
                image = read_label(prediction_path, 'unified')
                timing = prediction_path.with_suffix('.timing.json')
                inference_seconds = read_json(timing)['inference_seconds'] if timing.exists() else None
            else:
                cache = shared / 'preprocessed' / DATASET / configuration.data_identifier / case_id if partition == 'val' else None
                predicted, properties = predict_case(predictor, Path(lookup[case_id]['image_path']), cache)
                partial = prediction_path.with_name(case_id + '.part.nii.gz')
                manager.image_reader_writer_class().write_seg(predicted.astype(np.uint8), str(partial), properties)
                inference_seconds = time.monotonic() - started
                write_json(prediction_path.with_suffix('.timing.json'), {'checkpoint': selected, 'inference_seconds': inference_seconds})
                partial.replace(prediction_path)
                image = read_label(prediction_path, 'unified')
            if not same_grid(geometry(image), geometry(target_image)):
                raise ValueError('Exported prediction and native GT physical grids differ.')
            pred = sitk.GetArrayViewFromImage(image)
            if not set(np.unique(pred)).issubset({0, 1}):
                raise ValueError('Binary model prediction contains nonbinary labels.')
            metrics_started = time.monotonic()
            reference_mask, prediction_mask = gt == 1, pred == 1
            scores = score_class(reference_mask, prediction_mask, np.asarray(target_image.GetSpacing()[::-1]))
            graph, _, _ = load_or_build_graph(reference_mask, geometry(target_image), case, 'pulmonary_artery', shared / 'native_gt_branch_cache', core)
            skeletal, issues = score_skeletons(reference_mask, prediction_mask, graph['coords_zyx'])
            branches, case_details = branch_coverages(graph, prediction_mask)
            if not branches['gt_branch_count']:
                issues.extend((name, 'no_effective_gt_branches_after_fixed_short_branch_and_cycle_exclusions')
                              for name in ['branch_detection_rate_80', 'branch_detection_rate_90', 'branch_coverage_mean'])
            difference = abs(scores['cldice'] - skeletal['cldice_recomputed']) if scores['cldice'] is not None else None
            if difference is not None and difference > 1e-12:
                raise ValueError('Recomputed clDice differs from the unchanged basic metric.')
            result = clean({'case_id': case_id, 'split': partition, **scores, **skeletal, **branches,
                       'cldice_absolute_difference': difference, 'inference_seconds': inference_seconds,
                       'evaluation_seconds': time.monotonic() - metrics_started,
                       'peak_inference_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                       'checkpoint': selected, 'inference_config': config['inference'], 'reference_geometry': geometry(target_image),
                       'prediction_relative_path': str(prediction_path.relative_to(work)),
                       'branch_details': case_details,
                       'exceptions': [{'metric': metric, 'reason': reason} for metric, reason in issues]})
            write_json(result_path, result)
        rows.append(result)
        details.extend({'case_id': case_id, 'split': partition, 'class': 'pulmonary_artery', **item} for item in result['branch_details'])
        exceptions.extend({'case_id': case_id, 'split': partition, **item} for item in result['exceptions'])
        predictions.append({'case_id': case_id, 'split': partition, 'path': result['prediction_relative_path'],
                            'checkpoint': selected, 'reference_geometry': result['reference_geometry']})
        print('PARSE_EVAL', group, len(rows), '/36', case_id, partition, result['dice'], flush=True)
    extra = ['skeleton_recall', 'skeleton_precision', 'cldice_recomputed', 'cldice_absolute_difference',
             'branch_detection_rate_80', 'branch_detection_rate_90',
             'branch_coverage_mean', 'gt_branch_count', 'detected_branch_count_80', 'detected_branch_count_90',
             'excluded_short_branch_count', 'excluded_cycle_component_count', 'excluded_short_branch_total_length_mm',
             'degenerate_structure_count', 'inference_seconds', 'peak_inference_allocated_gib']
    fields = ['case_id', 'split', *BASIC, *extra, 'empty', 'legacy_cldice_policy']
    for partition, filename in [('test', 'per_case.csv'), ('val', 'validation_per_case.csv')]:
        write_csv(work / filename, [row for row in rows if row['split'] == partition], fields)
    summary = {'state': 'done', 'experiment_family': 'parse22_supervised', 'group': group,
               'patient_counts': {'val': 16, 'test': 20}, 'primary_partition': 'test', 'checkpoint': selected,
               'aggregation': 'patient macro mean/sample_std(ddof=1)/median/n_valid; no voxel or branch pooling',
               'scores': {partition: {metric: stats([row.get(metric) for row in rows if row['split'] == partition])
                                      for metric in list(BASIC) + extra} for partition in ['val', 'test']},
               'exception_count': len(exceptions), 'av_metrics_computed': False, 'airway_or_vein_evaluated': False}
    summary['training'] = complete
    summary['model'] = read_json(work / 'initialization.json')
    if abs(summary['scores']['val']['dice']['mean'] - selected['validation_dice']) > 1e-6:
        raise ValueError('Final validation Dice does not agree with checkpoint selection.')
    write_json(work / 'summary.json', clean(summary))
    write_json(work / 'prediction_manifest.json', {'state': 'done', 'checkpoint': selected, 'predictions': predictions})
    write_csv(work / 'branch_details.csv', details, ['case_id', 'split', 'class', 'branch_id', 'length_mm', 'coverage', 'detected80', 'detected90', 'exclusion_reason'])
    write_csv(work / 'evaluation_exceptions.csv', exceptions, ['case_id', 'split', 'metric', 'reason'])
    write_json(work / 'evaluation_config.json', {'state': 'done', 'checkpoint': selected,
               'inference': config['inference'], 'class': 'pulmonary_artery', 'av_metrics_computed': False})
    print('PARSE_SUPERVISED_EVALUATION_DONE', group, flush=True)


if __name__ == '__main__':
    main()
