"""Training-only planning for the independent binary PARSE22 experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil

import numpy as np
import SimpleITK as sitk


DATASET_ID = 930
DATASET = 'Dataset930_PARSE22Supervised'
CONFIGURATION = '3d_fullres'
PATCH = [128, 128, 128]
BATCH = 2


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def configure_paths(shared, results=None):
    from nnunetv2 import paths
    values = {'nnUNet_raw': shared / 'raw', 'nnUNet_preprocessed': shared / 'preprocessed',
              'nnUNet_results': results or shared / 'planning_results'}
    for name, path in values.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
        setattr(paths, name, str(path))


def stage_read_only_pairs(shared, split_path, source_manifest):
    split = read_json(split_path)
    sets = [set(split[key]) for key in ('train', 'val', 'test')]
    assert [len(s) for s in sets] == [64, 16, 20]
    assert len(set.union(*sets)) == 100 and not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    lookup = {row['case_id']: row for row in read_json(source_manifest)['cases']}
    if set(lookup) != set.union(*sets):
        raise ValueError('Source manifest differs from the frozen100 cases.')
    root = shared / 'raw' / DATASET
    for partition, suffix in [('train', 'Tr'), ('val', 'Va')]:
        for case_id in split[partition]:
            for kind, folder, name in [('image', 'images' + suffix, case_id + '_0000.nii.gz'),
                                       ('label', 'labels' + suffix, case_id + '.nii.gz')]:
                source = Path(lookup[case_id][kind + '_path']).resolve()
                if not source.is_file():
                    raise FileNotFoundError(source)
                target = root / folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    if target.resolve() != source:
                        raise ValueError('Existing source symlink differs from the frozen manifest.')
                elif target.exists():
                    raise ValueError('Raw supervised inputs must remain read-only aliases, not replacements.')
                else:
                    target.symlink_to(source)
    dataset = {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'pulmonary_artery': 1},
               'numTraining': 64, 'file_ending': '.nii.gz', 'name': DATASET,
               'description': 'Independent PARSE22 supervised train64/val16/test20; only imagesTr fit statistics'}
    for name, value in [('dataset.json', dataset), ('splits_final.json', [{'train': split['train'], 'val': split['val']}])]:
        path = root / name
        if path.exists() and read_json(path) != value:
            raise ValueError('Frozen raw dataset definition changed.')
        write_json(path, value)
    return split, lookup, dataset


def inspect_training_resolution(shared, split, lookup):
    from evaluation.skeleton_metrics import skeleton_coordinates
    report_path = shared / 'training_resolution_study.json'
    if report_path.exists():
        report = read_json(report_path)
        if report['training_cases'] != split['train']:
            raise ValueError('Resolution study uses a different training cohort.')
        if report['state'] != 'frozen':
            raise ValueError('Training resolution requires review before planning.')
        return report
    spacings = {}
    for case_id in split['train']:
        reader = sitk.ImageFileReader()
        reader.SetFileName(lookup[case_id]['image_path'])
        reader.ReadImageInformation()
        spacings[case_id] = list(reader.GetSpacing())
    ordered = sorted(split['train'], key=lambda key: (np.prod(spacings[key]), key))
    selected = [ordered[index] for index in np.linspace(0, 63, 8).astype(int)]
    results = []
    chosen = None
    # Fixed engineering rule before seeing these training-only measurements.
    for target_spacing in [1.0, 0.8, 0.75]:
        rows = []
        for case_id in selected:
            image = sitk.ReadImage(lookup[case_id]['label_path'])
            mask = sitk.GetArrayFromImage(image) == 1
            size = np.maximum(1, np.round(np.asarray(image.GetSize()) * np.asarray(image.GetSpacing()) / target_spacing).astype(int))
            low = sitk.Resample(image, size.tolist(), sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor,
                                image.GetOrigin(), [target_spacing] * 3, image.GetDirection(), 0, sitk.sitkUInt8)
            restored = sitk.Resample(low, image, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            restored_mask = sitk.GetArrayViewFromImage(restored) == 1
            coords = skeleton_coordinates(mask)
            recall = float(restored_mask[tuple(coords.T)].mean())
            original_volume = float(mask.sum() * np.prod(image.GetSpacing()))
            volume_ratio = float(np.count_nonzero(sitk.GetArrayViewFromImage(low)) * target_spacing ** 3 / original_volume)
            rows.append({'case_id': case_id, 'native_spacing_xyz_mm': spacings[case_id],
                         'target_spacing_mm': target_spacing, 'roundtrip_gt_skeleton_recall': recall,
                         'foreground_physical_volume_ratio': volume_ratio})
        passed = float(np.mean([r['roundtrip_gt_skeleton_recall'] for r in rows])) >= .95 and min(r['roundtrip_gt_skeleton_recall'] for r in rows) >= .90
        results.append({'target_spacing_mm': target_spacing, 'cases': rows, 'retention_rule_passed': passed})
        print('TRAIN_ONLY_RESOLUTION', target_spacing, 'mean_skeleton_retention', np.mean([r['roundtrip_gt_skeleton_recall'] for r in rows]), 'passed', passed, flush=True)
        if passed:
            chosen = target_spacing
            break
    values = np.asarray(list(spacings.values()))
    report = {'state': 'frozen' if chosen else 'needs_review', 'training_cases': split['train'],
              'spacing_xyz_mm': {'min': values.min(0).tolist(), 'median': np.median(values, axis=0).tolist(), 'max': values.max(0).tolist()},
              'representative_training_cases': selected, 'selection_rule': 'Coarsest candidate among1.0,0.8,0.75mm with8 training-case mean GT centerline NN-roundtrip retention>=0.95 and minimum>=0.90; no validation/test images or scores used',
              'label_interpolation': 'nearest_neighbor', 'target_spacing_xyz_mm': [chosen] * 3 if chosen else None,
              'measurements': results}
    write_json(report_path, report)
    if chosen is None:
        raise RuntimeError('Training-only fine-vessel resampling retention needs review; no formal training started.')
    return report


def cache_case(task):
    out, image, label, plans, dataset, provenance = task
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    manager = PlansManager(plans)
    config = manager.get_configuration(CONFIGURATION)
    data, seg, props = DefaultPreprocessor().run_case([str(image)], str(label), manager, config, dataset)
    for extension, array in [('.npy', data.astype(np.float16)), ('_seg.npy', seg.astype(np.int8))]:
        path = out.parent / (out.name + extension)
        temporary = path.with_suffix('.npy.part')
        with temporary.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path)
    out.with_suffix('.pkl.part').write_bytes(pickle.dumps(props))
    out.with_suffix('.pkl.part').replace(out.with_suffix('.pkl'))
    write_json(out.with_suffix('.cache_provenance.json'), provenance)


def prepare(shared, split_path, source_manifest):
    configure_paths(shared)
    split, lookup, dataset = stage_read_only_pairs(shared, split_path, source_manifest)
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    resolution = inspect_training_resolution(shared, split, lookup)
    completed_path = shared / 'preprocessing_config.json'
    base = shared / 'preprocessed' / DATASET
    plan_path = base / 'nnUNetPlans.json'
    if completed_path.exists():
        config = read_json(completed_path)
        if config['split_sha256'] != split_hash:
            raise ValueError('Existing preprocessing belongs to a different split.')
        return read_json(plan_path), dataset
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments
    extract_fingerprints([DATASET_ID], num_processes=4, clean=False, verbose=False)
    plan_experiments([DATASET_ID], overwrite_target_spacing=resolution['target_spacing_xyz_mm'])
    plans = read_json(plan_path)
    intensity = plans['foreground_intensity_properties_per_channel']['0']
    if intensity['min'] < -2048 or intensity['max'] > 5000:
        raise ValueError('Training CT statistics conflict with the checked raw HU ranges; inspect source copies before training.')
    config = plans['configurations'][CONFIGURATION]
    if np.any(np.asarray(PATCH) % np.prod(config['pool_op_kernel_sizes'], axis=0)):
        raise ValueError('Common128-cube patch is incompatible with planned strides.')
    config['patch_size'], config['batch_size'] = PATCH, BATCH
    config['resampling_fn_seg_kwargs'].update(order=0, order_z=0)
    # Continuous CT and probability interpolation remain the existing defaults.
    write_json(plan_path, plans)
    write_json(base / 'splits_final.json', [{'train': split['train'], 'val': split['val']}])
    write_json(base / 'dataset.json', dataset)
    storage = base / config['data_identifier']
    storage.mkdir(parents=True, exist_ok=True)
    pending, required = [], 0
    plan_sha = hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()
    for partition, suffix in [('train', 'Tr'), ('val', 'Va')]:
        for case_id in split[partition]:
            out = storage / case_id
            provenance = {'plan_sha256': plan_sha, 'image_path': lookup[case_id]['image_path'],
                          'label_path': lookup[case_id]['label_path']}
            provenance_path = out.with_suffix('.cache_provenance.json')
            if (all((storage / (case_id + extension)).is_file() for extension in ['.npy', '_seg.npy', '.pkl'])
                    and provenance_path.exists() and read_json(provenance_path) == provenance):
                continue
            reader = sitk.ImageFileReader()
            reader.SetFileName(lookup[case_id]['image_path'])
            reader.ReadImageInformation()
            voxels = np.prod(np.asarray(reader.GetSize()) * np.asarray(reader.GetSpacing()) / resolution['target_spacing_xyz_mm'])
            # Existing derived arrays are replaced atomically one case at a
            # time. Reserve the final growth plus one temporary case per worker.
            expected = int(voxels * 3.3) + 512 * 1024
            existing = sum((storage / (case_id + extension)).stat().st_size for extension in ['.npy', '_seg.npy', '.pkl']
                           if (storage / (case_id + extension)).exists())
            required += max(0, expected - existing)
            pending.append((out, shared / 'raw' / DATASET / ('images' + suffix) / (case_id + '_0000.nii.gz'),
                            shared / 'raw' / DATASET / ('labels' + suffix) / (case_id + '.nii.gz'), plans, dataset, provenance))
    if shutil.disk_usage(storage).free < required + 8 * 1024 ** 3:
        raise RuntimeError('Insufficient shared-preprocessing space with6GiB reserve; no old results deleted.')
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        for index, _ in enumerate(pool.map(cache_case, pending), 1):
            print('PREPROCESSED', index, '/', len(pending), flush=True)
    write_json(completed_path, {'state': 'done', 'dataset': DATASET, 'split_sha256': split_hash,
               'normalization_fit_cases': split['train'], 'planning_fit_cases': split['train'],
               'source_manifest_sha256': hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest(),
               'normalization': plans['foreground_intensity_properties_per_channel'],
               'target_spacing_xyz_mm': resolution['target_spacing_xyz_mm'], 'patch': PATCH, 'batch': BATCH,
               'cached_cases': split['train'] + split['val'], 'cached_test_cases': [],
               'ct_interpolation': config['resampling_fn_data_kwargs'], 'label_interpolation': config['resampling_fn_seg_kwargs'],
               'stored_image_dtype': 'float16', 'stored_label_dtype': 'int8', 'all_groups_share_plans': True})
    return plans, dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.shared, args.split, args.source_manifest)


if __name__ == '__main__':
    main()
