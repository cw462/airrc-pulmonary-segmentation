"""Evaluate the frozen AirRC checkpoint on the public HiPaS NPZ cohort.

The published NPZ data have no patient origin/direction. The shared benchmark
grid uses the published per-case XYZ spacing, identity LPS direction and zero
origin for both CT and GT; this is not a recovered patient-world affine.
The source layout is Y,X,Z, matching the author's NPZ-to-image axis swap.
"""

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk

from .add_av_metrics import METRICS as AV_METRICS, FORMULAS, read_json, write_json
from .av_metrics import compute_av_metrics
from .label_geometry import geometry, same_grid


def prepare_case(case, inputs, temporary, label_path):
    def load(name):
        with np.load(inputs / name, allow_pickle=False) as archive:
            return archive['data']

    ct, artery, vein = load('ct.npz'), load('artery.npz'), load('vein.npz')
    if ct.ndim != 3 or ct.shape != artery.shape or ct.shape != vein.shape:
        raise ValueError('HiPaS CT/artery/vein array shapes differ.')
    if not np.issubdtype(ct.dtype, np.number) or not np.isfinite(ct).all():
        raise ValueError('Nonfinite or nonnumeric HiPaS CT.')
    low, high = float(ct.min()), float(ct.max())
    # The actual pinned Zenodo release is raw int16 HU, unlike the normalized
    # examples described in parts of the upstream README. Do not invert twice.
    if low >= -200 or high <= 2:
        raise ValueError(f'Unexpected CT intensity for the pinned raw-HU release: {low}, {high}')
    if not np.isin(artery, (0, 1)).all() or not np.isin(vein, (0, 1)).all():
        raise ValueError('Expected separate binary HiPaS artery/vein references.')
    artery, vein = artery.astype(bool), vein.astype(bool)
    overlap = int(np.count_nonzero(artery & vein))
    labels = np.zeros(ct.shape, dtype=np.uint8)
    labels[vein] = 2
    labels[artery] = 1  # Explicit author recommendation, GitHub issue #10.
    spacing = tuple(float(value) for value in case['spacing_xyz_mm'])
    if len(spacing) != 3 or any(not math.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError('Invalid published per-case XYZ spacing.')
    ct_image = sitk.GetImageFromArray(np.ascontiguousarray(ct.transpose(2, 0, 1)))
    ct_image.SetSpacing(spacing)
    gt_image = sitk.GetImageFromArray(np.ascontiguousarray(labels.transpose(2, 0, 1)))
    gt_image.CopyInformation(ct_image)
    ct_path = temporary / 'ct.nii'
    sitk.WriteImage(ct_image, str(ct_path))
    sitk.WriteImage(gt_image, str(label_path), True)
    return ct_path, gt_image, {
        'source_shape_yxz': list(ct.shape), 'source_dtype': str(ct.dtype),
        'source_hu_min': low, 'source_hu_max': high, 'ct_intensity_conversion': 'none; actual source is raw HU',
        'source_artery_voxels': int(artery.sum()), 'source_vein_voxels': int(vein.sum()),
        'overlap_voxels_assigned_artery': overlap,
        'overlap_policy': 'artery priority per author issue #10; original NPZ references remain unchanged',
        'geometry_basis': 'published XYZ spacing and YXZ layout; common canonical LPS benchmark frame, not original patient origin/direction',
    }


def evaluate_case(predictor, case, work):
    from evaluate_airrc import predict_case, score_class

    case_id = case['case_id']
    temporary = work / 'tmp' / case_id
    temporary.mkdir(parents=True, exist_ok=True)
    label_path = work / 'labels' / f'{case_id}.nii.gz'
    started = time.monotonic()
    ct_path, gt_image, provenance = prepare_case(case, work / 'inputs' / case_id, temporary, label_path)
    prediction, properties = predict_case(predictor, ct_path)
    reader = predictor.plans_manager.image_reader_writer_class()
    reference, ref_props = reader.read_seg(str(label_path))
    if reference[0].shape != prediction.shape or ref_props['sitk_stuff'] != properties['sitk_stuff']:
        raise ValueError(f'Prediction did not return to the shared reference grid for {case_id}.')
    prediction_path = work / 'predictions' / f'{case_id}.nii.gz'
    partial = prediction_path.with_name(case_id + '.part.nii.gz')
    reader.write_seg(prediction.astype(np.uint8), str(partial), properties)
    exported = sitk.ReadImage(str(partial))
    if not same_grid(geometry(exported), geometry(gt_image)):
        raise ValueError(f'Exported prediction has a different physical grid for {case_id}.')
    partial.replace(prediction_path)
    spacing_zyx = np.asarray(ref_props['spacing'], dtype=float)
    scores = {name: score_class(reference[0] == label, prediction == label, spacing_zyx)
              for label, name in ((1, 'pulmonary_artery'), (2, 'pulmonary_vein'))}
    av = compute_av_metrics(reference[0], prediction)
    av = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
          for key, value in av.items()}
    result = {
        'case_id': case_id, 'scores': scores, 'av': av, 'seconds': time.monotonic() - started,
        'prediction_path': str(prediction_path), 'gt_path': str(label_path),
        'same_reference_grid': True, 'reference_geometry': geometry(gt_image), **provenance,
    }
    # Only a generated transient image is removed; original source archives,
    # reference labels, predictions and all case results are retained.
    ct_path.unlink()
    return result


def statistics(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return {'mean': float(np.mean(values)) if values else None,
            'sample_std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'median': float(np.median(values)) if values else None, 'n_valid': len(values)}


def summarize(work, config, cases):
    from evaluate_airrc import METRICS as STRUCTURE_METRICS

    rows = [read_json(work / 'results/cases' / f"{case['case_id']}.json") for case in cases
            if (work / 'results/cases' / f"{case['case_id']}.json").exists()]
    summary = {
        **config, 'state': 'done' if len(rows) == len(cases) else 'partial',
        'patients_expected': len(cases), 'patients_completed': len(rows),
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'aggregation': 'one value per patient; arithmetic mean, sample_std ddof=1, median; no voxel pooling',
        'scores': {structure: {metric: statistics([row['scores'][structure][metric] for row in rows])
                               for metric in STRUCTURE_METRICS}
                   for structure in ('pulmonary_artery', 'pulmonary_vein')},
        'av_metrics': {metric: statistics([row['av'][metric] for row in rows]) for metric in AV_METRICS},
        'av_formulas': FORMULAS,
        'overlap_cases': [{'case_id': row['case_id'], 'voxels': row['overlap_voxels_assigned_artery']}
                          for row in rows if row['overlap_voxels_assigned_artery']],
        'airway_gt_available': False,
    }
    write_json(work / 'results/summary.json', summary)
    if rows:
        csv_path = work / 'results/per_case.csv'
        with csv_path.with_suffix('.csv.part').open('w', newline='', encoding='utf-8') as stream:
            fields = ['case_id', 'structure', *STRUCTURE_METRICS, 'empty', *AV_METRICS,
                      'overlap_voxels_assigned_artery']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                for structure, scores in row['scores'].items():
                    writer.writerow({'case_id': row['case_id'], 'structure': structure, **scores,
                                     **{key: row['av'][key] for key in AV_METRICS},
                                     'overlap_voxels_assigned_artery': row['overlap_voxels_assigned_artery']})
        csv_path.with_suffix('.csv.part').replace(csv_path)
        counts = {key for key in rows[0]['av'] if key not in AV_METRICS}
        with (work / 'results/av_confusion_counts.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['case_id', *sorted(counts), 'overlap_voxels_assigned_artery'])
            writer.writeheader()
            for row in rows:
                writer.writerow({'case_id': row['case_id'], **{key: row['av'][key] for key in sorted(counts)},
                                 'overlap_voxels_assigned_artery': row['overlap_voxels_assigned_artery']})
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint-epoch', type=int, default=400)
    args = parser.parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    work = args.work.resolve()
    cohort = read_json(args.manifest)
    cases = cohort['cases']
    if len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate HiPaS case identifiers.')
    for name in ('results/cases', 'predictions', 'labels', 'tmp', 'inputs'):
        (work / name).mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in cohort.items() if key != 'cases'}
    config.update(checkpoint=str(args.model / 'fold_0/checkpoint_best.pth'), checkpoint_epoch=args.checkpoint_epoch,
                  checkpoint_step=100000, inference='unchanged AirRC M0; 1 mm preprocessing; native reference-grid export; no TTA')
    from evaluate_airrc import load_model
    predictor = None
    completed = summarize(work, config, cases)
    while completed < len(cases):
        ready = [case for case in cases
                 if not (work / 'results/cases' / f"{case['case_id']}.json").exists()
                 and (work / 'inputs' / case['case_id'] / 'ready.json').is_file()]
        write_json(work / 'heartbeat.json', {'state': 'evaluating' if ready else 'waiting_for_data',
                                           'patients_completed': completed, 'updated_at': datetime.now(timezone.utc).isoformat()})
        if not ready:
            time.sleep(10)
            continue
        if predictor is None:
            predictor = load_model(args.model, args.checkpoint_epoch)
        for case in ready:
            result = evaluate_case(predictor, case, work)
            write_json(work / 'results/cases' / f"{case['case_id']}.json", result)
            completed = summarize(work, config, cases)
            print(f"HIPAS {completed}/{len(cases)} {case['case_id']} {result['seconds']:.1f}s "
                  f"A_Dice={result['scores']['pulmonary_artery']['dice']} "
                  f"V_Dice={result['scores']['pulmonary_vein']['dice']}", flush=True)
    write_json(work / 'heartbeat.json', {'state': 'done', 'patients_completed': completed})


if __name__ == '__main__':
    main()
