"""Run one from-scratch PARSE22 group or its isolated plumbing smoke test."""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch

from .preprocess import CONFIGURATION, DATASET, configure_paths, read_json, write_json


def run_smoke(trainer):
    from .trainer import build_network, seed_everything
    trainer.on_train_start()
    trainer.on_train_epoch_start()
    batch = next(trainer.dataloader_train)
    target = batch['target'][0]
    if not set(torch.unique(target[:, :1]).tolist()).issubset({0, 1}):
        raise ValueError('Smoke labels are not binary.')
    start = time.monotonic()
    fit = [float(trainer.train_step(batch)['loss']) for _ in range(30)]
    for _ in range(20):
        trainer.train_step(next(trainer.dataloader_train))
    elapsed = time.monotonic() - start
    assert trainer.optimizer_steps == 50 and all(math.isfinite(v) for v in fit)
    assert all(math.isfinite(v) for v in trainer.gradient_norms) and max(trainer.gradient_norms) > 0
    if fit[-1] >= fit[0]:
        raise RuntimeError('Fixed-patch loss did not decrease during smoke; investigate before formal training.')
    peak = {'allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'reserved_gib': torch.cuda.max_memory_reserved() / 1024**3}
    trainer.network.eval()
    with torch.no_grad():
        patch_validation = trainer.validation_step(next(trainer.dataloader_val))
    patch_loss = float(patch_validation['loss'])
    if not math.isfinite(patch_loss):
        raise FloatingPointError('Smoke patch validation loss is nonfinite.')
    data = batch['data'][:1].to(trainer.device)
    trainer.network.eval()
    trainer.set_deep_supervision_enabled(False)
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True):
        before = trainer.network(data).float().cpu()
    checkpoint = trainer.work / 'checkpoint_smoke.pth'
    trainer.save_checkpoint(checkpoint)
    restored, _ = build_network(trainer.plans_manager, trainer.configuration_manager, trainer.dataset_json,
                                 trainer.group, trainer.device, deep_supervision=False)
    state = torch.load(checkpoint, map_location='cpu')
    restored.load_state_dict(state['network_weights'], strict=True)
    restored.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True):
        after = restored(data).float().cpu()
    difference = float((before - after).abs().max())
    if difference > 1e-5:
        raise RuntimeError('Real-model checkpoint reload output differs.')
    del restored, before, after, data
    torch.cuda.empty_cache()
    val = trainer.validate_volumes(export_first=True, case_ids=trainer.do_split()[1][:1])
    refiner_gradients = None
    if trainer.group == 'P2':
        head = trainer.network.refiner.residual_head.weight.grad
        features = trainer.network.refiner.features[0].block[0].weight.grad
        assert head is not None and features is not None and head.abs().sum() > 0 and features.abs().sum() > 0
        refiner_gradients = {'head_abs_sum': float(head.abs().sum()), 'features_abs_sum_after_head_update': float(features.abs().sum())}
    result = {'state': 'passed', 'group': trainer.group, 'optimizer_steps': 50, 'initial_fit_loss': fit[0],
              'final_fit_loss': fit[-1], 'seconds_per_update_including_sampling': elapsed / 50,
              'gradient_norm_min': min(trainer.gradient_norms), 'gradient_norm_max': max(trainer.gradient_norms),
              'amp_retries': trainer.amp_retries, 'peak_training_memory': peak,
              'patch_validation_loss': patch_loss,
              'checkpoint_reload_max_absolute_difference': difference, 'native_grid_validation': val,
              'classes': 2, 'refiner_gradients': refiner_gradients, 'used_for_formal_initialization': False}
    write_json(trainer.work / 'smoke_report.json', result)
    print('SMOKE_PASSED', result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--group', choices=['P0', 'P1', 'P2'], required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    shared = root / 'shared'
    work = root / args.group / 'smoke' if args.mode == 'smoke' else root / args.group
    configure_paths(shared, work / 'results')
    from .config import make_config, initialize_reports
    from .trainer import Parse22Trainer, seed_everything
    config = make_config(root, args.group, args.mode)
    split = read_json(root / 'parse22_seed2026.json')
    if args.group != 'P0' and args.mode == 'full':
        previous = 'P0' if args.group == 'P1' else 'P1'
        completed = root / previous / 'training_complete.json'
        if not completed.exists() or read_json(completed)['optimizer_steps'] != 100000:
            raise ValueError('Formal experiment order is P0 thenP1 thenP2; previous group not complete.')
    if args.mode == 'full':
        quality = read_json(root / 'server_source_pair_verification.json')
        if quality.get('state') != 'done' or quality.get('patients') != 100:
            raise ValueError('Server CT/GT pairs must match the checked original cohort before formal training.')
        smoke = root / args.group / 'smoke/smoke_report.json'
        if not smoke.exists() or read_json(smoke)['state'] != 'passed':
            raise ValueError('Run this group isolated smoke test before formal training.')
        if args.group == 'P0':
            for group in ['P1', 'P2']:
                check = root / group / 'smoke/smoke_report.json'
                if not check.exists() or read_json(check)['state'] != 'passed':
                    raise ValueError('Verify all three groups engineering smoke before freezing the formal matrix.')
        if args.resume is None and (work / 'training_history.csv').exists() and len((work / 'training_history.csv').read_text().splitlines()) > 1:
            raise ValueError('Existing formal training must resume its own valid checkpoint, not restart over results.')
    initialize_reports(work, config, split)
    seed_everything(2026)
    torch.set_num_threads(4)
    plans = read_json(shared / 'preprocessed' / DATASET / 'nnUNetPlans.json')
    dataset = read_json(shared / 'raw' / DATASET / 'dataset.json')
    trainer = Parse22Trainer(plans, CONFIGURATION, 0, dataset, unpack_dataset=False, device=torch.device('cuda', 0))
    trainer.group, trainer.shared, trainer.work = args.group, shared, work
    trainer.run_kind, trainer.experiment_config = args.mode, config
    trainer.num_epochs, trainer.num_iterations_per_epoch, trainer.validate_every = 400, 250, 8
    trainer.initial_lr, trainer.weight_decay = .01, 3e-5
    trainer.initialize()
    if args.resume:
        trainer.load_checkpoint(str(args.resume))
    if args.mode == 'smoke':
        if args.resume:
            raise ValueError('Smoke is an isolated from-scratch plumbing test, not a continuation experiment.')
        run_smoke(trainer)
    else:
        started = time.monotonic()
        trainer.run_training()
        if trainer.optimizer_steps != 100000:
            raise RuntimeError('Formal experiment did not finish100000 successful optimizer updates.')
        write_json(work / 'training_complete.json', {'state': 'trained', 'group': args.group,
                   'optimizer_steps': trainer.optimizer_steps, 'epochs': trainer.current_epoch,
                   'this_process_wall_seconds': time.monotonic() - started,
                   'training_wall_seconds_on_saved_trajectory': trainer.elapsed_before_resume + time.monotonic() - trainer.training_started_monotonic,
                   'best_validation_patient_dice': trainer.best_volume_dice,
                   'peak_training_allocated_gib': trainer.training_peak_allocated_gib,
                   'peak_training_reserved_gib': trainer.training_peak_reserved_gib,
                   'checkpoint_best': str(Path(trainer.output_folder) / 'checkpoint_best.pth'),
                   'split_sha256': config['split_sha256'], 'airrc_checkpoint_loaded': False})


if __name__ == '__main__':
    main()
