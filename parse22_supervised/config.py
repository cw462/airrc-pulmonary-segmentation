"""Frozen experiment definitions, separate from all cross-dataset reports."""
import hashlib
from pathlib import Path

from .preprocess import DATASET, read_json, write_json


def make_config(root, group, kind):
    shared = root / 'shared'
    split_path = root / 'parse22_seed2026.json'
    preprocessing = read_json(shared / 'preprocessing_config.json')
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if preprocessing['split_sha256'] != split_sha or preprocessing['state'] != 'done':
        raise ValueError('A completed, training-only preprocessing configuration is required.')
    base = Path(__file__).resolve().parent.parent
    files = ['train_airrc.py', 'models/ct_local_refiner.py', 'parse22_supervised/preprocess.py',
             'parse22_supervised/trainer.py', 'parse22_supervised/skeleton_recall.py',
             'parse22_supervised/config.py', 'parse22_supervised/train.py',
             'parse22_supervised/evaluate.py', 'parse22_supervised/compare.py',
             'parse22_supervised/scheduling.py', 'parse22_supervised/pipeline.py']
    sources = {name: hashlib.sha256((base / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for name in files}
    common = {
        'experiment_family': 'parse22_supervised', 'dataset': DATASET, 'role': 'within_dataset_supervised',
        'split_file': 'data/splits/parse22_seed2026.json', 'split_sha256': split_sha,
        'patient_counts': {'train': 64, 'val': 16, 'test': 20}, 'backbone': 'U-Mamba bottleneck3D',
        'labels': {'background': 0, 'pulmonary_artery': 1}, 'output_channels': 2,
        'seed': 2026, 'data_sampling_seed': 2026, 'initialization': 'from_scratch_no_AirRC_or_other_trained_checkpoint',
        'shared_backbone_initialization_verified_by_tensor_hash': True,
        'target_spacing_xyz_mm': preprocessing['target_spacing_xyz_mm'], 'patch': [128, 128, 128], 'batch_size': 2,
        'normalization': preprocessing['normalization'], 'normalization_fitted_on': 'train64_only',
        'automatic_planning_fitted_on': 'train64_only', 'data_cache': {'CT': 'float16', 'segmentation': 'int8'},
        'ct_interpolation': preprocessing['ct_interpolation'], 'label_interpolation': 'nearest_neighbor_order0',
        'training_label_augmentation_interpolation': 'nearest_neighbor_order0',
        'augmentation': 'Existing nnU-Net/U-Mamba pipeline; resolved parameter snapshot saved separately',
        'augmentation_processes': 0, 'augmentation_order': 'deterministic SingleThreadedAugmenter; same seed for all groups',
        'foreground_oversampling_fraction': .33,
        'optimizer': {'name': 'SGD', 'initial_lr': .01, 'momentum': .99, 'nesterov': True, 'weight_decay': 3e-5,
                      'gradient_clip_norm': 12, 'schedule': 'existing polynomial learning-rate scheduler over400epochs'},
        'optimizer_steps': 100000, 'epochs': 400, 'steps_per_epoch': 250,
        'amp': True, 'amp_overflow_policy': 'Retry same batch after normal GradScaler backoff; count only successful optimizer updates',
        'base_loss': 'Existing foreground Dice + cross entropy, original deep-supervision weights',
        'checkpoint_selection': {'metric': 'mean_per_patient_pulmonary_artery_Dice', 'partition': 'val16',
                                 'full_validation_every_steps': 2000, 'tie_rule': 'retain earlier checkpoint',
                                 'test_used': False, 'latest_checkpoint_every_steps': 250},
        'inference': {'tile_step_size': .5, 'overlap_fraction': .5, 'gaussian_weighting': True, 'TTA': False,
                      'postprocessing': 'none', 'grid': 'original GT physical grid after existing inverse preprocessing'},
        'prior_exposure': 'All100 cases previously evaluated using frozen AirRC M0. Test20 is a new supervised holdout, not study-wide never-viewed blind data.',
        'not_a_precise_vesmamba_training_reproduction': True,
        'source_code_sha256': sources,
    }
    matrix = {**common, 'groups': {
        'P0': {'skeleton_recall_weight': 0, 'ct_local_refiner': False},
        'P1': {'skeleton_recall_weight': 1, 'ct_local_refiner': False},
        'P2': {'skeleton_recall_weight': 1, 'ct_local_refiner': {'hidden_channels': 16, 'max_delta': 1.0,
                'gate_floor': .25, 'refiner_dtype': 'FP32 under AMP', 'refiner_seed': 2026,
                'conditions': 'normalizedCT + detached probabilities + detached normalized entropy',
                'output': 'highest-resolution logits only; before Gaussian patch fusion',
                'initial_residual': 'exact_zero', 'extra_residual_regularization': False,
                'locality_limit': '7x7x7 convolution path; GroupNorm spatial statistics mean full module is not strictly finite-neighbourhood'}}},
        'skeleton_recall': {'reference': 'MIC-DKFZ/Skeleton-Recall@c7bac3c72ca7bb4f3d891235f051f0c4d07d7229',
                            'training_tube': 'Lee GT skeleton,2-voxel Manhattan dilation intersectGT, after augmentation',
                            'smooth': 1e-5, 'deep_supervision': 'highest-resolution target only; original DS weights retained'},
        'comparison_boundary': 'P1-P0 SRL increment; P2-P1 refiner increment conditional onSRL; P2-P0 full method; no standalone-refiner or complementarity claim'}
    frozen_path = root / 'experiment_matrix.json'
    path = root / 'experiment_matrix_smoke_candidate.json' if kind == 'smoke' and not frozen_path.exists() else frozen_path
    if path.exists() and path == frozen_path:
        if read_json(path) != matrix:
            raise ValueError('Frozen experimental conditions or source changed; do not silently mix runs.')
        # Matched frozen metadata is read-only during concurrent P1/P2 startup.
    else:
        write_json(path, matrix)
    return {**common, 'group': group, 'run_kind': kind, 'optimizer_steps': 50 if kind == 'smoke' else 100000,
            'formal_budget_optimizer_steps': 100000, **matrix['groups'][group],
            'inference': {**common['inference'], 'ct_local_refiner_enabled': group == 'P2',
                          'refiner_stage': 'per_normalized_patch_before_Gaussian_fusion' if group == 'P2' else None},
            'skeleton_recall_protocol': matrix['skeleton_recall']}


def initialize_reports(work, config, split):
    work.mkdir(parents=True, exist_ok=True)
    path = work / 'experiment_config.json'
    if path.exists() and read_json(path) != config:
        raise ValueError('Existing experiment configuration differs.')
    write_json(path, config)
    write_json(work / 'split_manifest.json', {'split_sha256': config['split_sha256'], **split})
    defaults = {
        'evaluation_config.json': {'state': 'pending_training', 'group': config['group'], 'inference': config['inference'],
                                  'checkpoint_selection': config['checkpoint_selection'], 'labels': config['labels']},
        'summary.json': {'state': 'not_evaluated', 'group': config['group'], 'role': 'within_dataset_supervised',
                         'patient_counts': config['patient_counts'], 'scores': None},
        'prediction_manifest.json': {'state': 'not_evaluated', 'group': config['group'], 'predictions': []},
    }
    for name, value in defaults.items():
        if not (work / name).exists():
            write_json(work / name, value)
    headers = {'training_history.csv': 'epoch,optimizer_steps,train_loss,patch_validation_loss\n',
               'validation_history.csv': 'optimizer_steps,mean_patient_dice,n_patients,full_validation_seconds\n',
               'per_case.csv': 'case_id,split,dice,iou,cldice,hd95_mm,precision,recall,skeleton_recall,skeleton_precision,branch_detection_rate_80,branch_detection_rate_90\n'}
    for name, header in headers.items():
        if not (work / name).exists():
            (work / name).write_text(header, encoding='utf-8')
