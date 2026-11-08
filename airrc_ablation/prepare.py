"""Regenerate disposable M0-format caches from its frozen plans; never fit new statistics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
import SimpleITK as sitk

from .common import DATASET, CONFIGURATION, configure_paths, digest, read_json, write_json

GIB = 1024 ** 3


def prepare(root, source, persistent):
    root.mkdir(parents=True, exist_ok=True)
    shared = root / 'shared'
    shared.mkdir(exist_ok=True)
    for name in ('raw', 'preprocessed'):
        target = shared / name
        expected = source / 'full' / name
        if target.is_symlink():
            assert target.resolve() == expected.resolve()
        else:
            assert not target.exists()
            target.symlink_to(expected, target_is_directory=True)
    configure_paths(shared, shared / 'planning_results_unused')
    plans_path = shared / 'preprocessed' / DATASET / 'nnUNetPlans.json'
    plans = read_json(plans_path)
    trained = read_json(source / 'full/results' / DATASET / 'AirRCTrainer__nnUNetPlans__3d_fullres/plans.json')
    if plans != trained:
        raise ValueError('Preprocessing plans differ from the completed M0 model; do not refit or silently change them.')
    dataset = read_json(shared / 'raw' / DATASET / 'dataset.json')
    assert dataset['labels'] == {'background': 0, 'pulmonary_artery': 1, 'pulmonary_vein': 2, 'airway_lumen': 3}
    split = read_json(source / 'evaluation/holdout/patient_splits.json')
    train_val = read_json(shared / 'preprocessed' / DATASET / 'splits_final.json')[0]
    assert train_val == {key: split[key] for key in ('train', 'val')}
    assert [len(split[key]) for key in ('train', 'val', 'test')] == [178, 38, 38]
    write_json(root / 'patient_splits.json', split)
    write_json(root / 'source_reference.json', {'M0_root': str(source / 'full'),
               'test_root': str(source / 'evaluation/holdout'), 'branch_cache': str(source / 'skeleton_branch/gt_cache'),
               'original_train_val_split_sha256': digest(source / 'full/raw' / DATASET / 'splits_final.json'),
               'patient_split_sha256': digest(root / 'patient_splits.json'), 'frozen_plan_sha256': digest(plans_path)})
    config = plans['configurations'][CONFIGURATION]
    assert config['patch_size'] == [128]*3 and config['batch_size'] == 2 and config['spacing'] == [1.0]*3
    storage = shared / 'preprocessed' / DATASET / config['data_identifier']
    storage.mkdir(exist_ok=True)
    disk_cache = persistent / 'cache'
    disk_cache.mkdir(parents=True, exist_ok=True)
    tasks = []
    for partition, suffix in [('train', 'Tr'), ('val', 'Va')]:
        for case_id in split[partition]:
            image = shared / 'raw' / DATASET / ('images'+suffix) / (case_id+'_0000.nii.gz')
            label = shared / 'raw' / DATASET / ('labels'+suffix) / (case_id+'.nii.gz')
            assert image.is_file() and label.is_file()
            reader = sitk.ImageFileReader(); reader.SetFileName(str(image)); reader.ReadImageInformation()
            estimate = int(np.prod(reader.GetSize())*3.2)+512*1024
            complete = all((storage/(case_id+ext)).is_file() for ext in ['.npy', '_seg.npy', '.pkl'])
            tasks.append({'case_id': case_id, 'image': image, 'label': label, 'estimate': estimate, 'complete': complete})
    pending = [task for task in tasks if not task['complete']]
    required = sum(task['estimate'] for task in pending)
    ram_budget = max(0, shutil.disk_usage(storage).free-6*GIB)
    # Keep8GiB SSD free plus3GiB for both groups' model/optimizer checkpoints.
    ssd_budget = max(0, shutil.disk_usage(persistent).free-11*GIB)
    on_ssd = set()
    for task in sorted(pending, key=lambda row: row['estimate'], reverse=True):
        if required <= ram_budget:
            break
        if task['estimate'] > ssd_budget:
            continue
        on_ssd.add(task['case_id']); required -= task['estimate']; ssd_budget -= task['estimate']
    if required > ram_budget:
        raise RuntimeError('Cache capacity insufficient with preserved original data and checkpoint reserves.')
    from train_airrc import cache_case
    work = []
    for task in pending:
        prefix = (disk_cache if task['case_id'] in on_ssd else storage) / task['case_id']
        work.append((prefix, task['image'], task['label'], plans, dataset))
    print('AIRRC_FROZEN_CACHE_PLAN', len(work), 'cases', len(on_ssd), 'onSSD; original plans/statistics unchanged', flush=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        for index, _ in enumerate(pool.map(cache_case, work), 1):
            print('AIRRC_CACHE', index, '/', len(work), flush=True)
    for case_id in on_ssd:
        for extension in ['.npy', '_seg.npy', '.pkl']:
            path = storage / (case_id+extension)
            assert not path.exists()
            path.symlink_to(disk_cache/(case_id+extension))
    assert all(all((storage/(task['case_id']+ext)).is_file() for ext in ['.npy', '_seg.npy', '.pkl']) for task in tasks)
    write_json(shared / 'preprocessing_config.json', {'state': 'done', 'reused_M0_plans_and_training_statistics': True,
               'no_fingerprint_refit': True, 'cached_train_val_counts': {'train':178,'val':38}, 'test_cached': False,
               'plans_sha256': digest(plans_path), 'patient_split_sha256': digest(root/'patient_splits.json'),
               'spacing_mm':[1,1,1], 'patch':[128]*3, 'batch_size':2, 'data_dtype':'float16',
               'SSD_cache_cases':sorted(on_ssd), 'source_arrays_identical_format_to_M0':True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--persistent', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.root, args.source, args.persistent)


if __name__ == '__main__':
    main()
