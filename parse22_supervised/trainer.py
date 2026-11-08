"""Binary PARSE22 training with the existing U-Mamba training machinery."""
import csv
import hashlib
import math
from pathlib import Path
import random
import time

import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from train_airrc import AirRCTrainer

from models.ct_local_refiner import RefinedSegmenter
from .preprocess import DATASET, write_json
from .skeleton_recall import SegmentationWithSkeletonRecall, SkeletonTargetTransform


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def transform_snapshot(value, depth=0):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): transform_snapshot(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [transform_snapshot(v, depth + 1) for v in value]
    if depth < 8 and hasattr(value, '__dict__'):
        return {'class': value.__class__.__module__ + '.' + value.__class__.__name__,
                'parameters': transform_snapshot(vars(value), depth + 1)}
    return getattr(value, '__module__', '') + '.' + getattr(value, '__qualname__', value.__class__.__name__)


def build_network(plans_manager, configuration_manager, dataset, group, device, deep_supervision=True):
    if dataset['labels'] != {'background': 0, 'pulmonary_artery': 1}:
        raise ValueError('PARSE22 requires a newly initialized binary output head.')
    backbone = nnUNetTrainerUMambaBot.build_network_architecture(plans_manager, dataset, configuration_manager, 1, deep_supervision)
    initial_hash = tensor_state_hash(backbone)
    network = backbone
    if group == 'P2':
        # Preserve the shared-backbone/data RNG stream while initializing the
        # supplied extra module from a fixed, independent RNG scope.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            network = RefinedSegmenter(backbone, num_classes=2, hidden_channels=16, max_delta=1.0, gate_floor=.25)
        assert torch.count_nonzero(network.refiner.residual_head.weight) == 0
        assert torch.count_nonzero(network.refiner.residual_head.bias) == 0
    return network.to(device), initial_hash


class Parse22Trainer(AirRCTrainer):
    """Inherit scheduling, optimizer, augmentations and full-volume selection.

    Set group/shared/work/experiment_config on the instance before initialize.
    No custom constructor: the installed framework introspects its signature.
    """
    group = 'P0'
    optimizer_steps = 0
    amp_retries = 0
    resuming = False

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError('Trainer already initialized.')
        if self.is_ddp:
            raise RuntimeError('This frozen experiment uses one GPU, not DDP.')
        self.num_input_channels = 1
        self.network, initial = build_network(self.plans_manager, self.configuration_manager, self.dataset_json,
                                               self.group, self.device, self.enable_deep_supervision)
        shared_init = self.shared / 'shared_backbone_initialization.json'
        if shared_init.exists():
            from .preprocess import read_json
            assert read_json(shared_init)['backbone_initial_sha256'] == initial, 'Shared initialization differs.'
        else:
            if self.group != 'P0':
                raise ValueError('P0 must establish the common from-scratch initialization first.')
            write_json(shared_init, {'backbone_initial_sha256': initial, 'seed': 2026, 'classes': 2,
                                     'loaded_pretrained_checkpoint': False})
        self.initial_backbone_hash = initial
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.loss = self._build_loss()
        self.was_initialized = True
        self.training_records = []
        self.gradient_norms = []
        self.training_peak_allocated_gib = 0.0
        self.training_peak_reserved_gib = 0.0
        self.sample_order_prefix = []
        self.elapsed_before_resume = 0.0
        optimizer_ids = {id(p) for g in self.optimizer.param_groups for p in g['params']}
        if self.group == 'P2':
            assert {id(p) for p in self.network.refiner.parameters()}.issubset(optimizer_ids)
        write_json(self.work / 'initialization.json', {'group': self.group, 'classes': 2,
                   'backbone_initial_sha256': initial, 'seed': 2026, 'refiner_seed': 2026 if self.group == 'P2' else None,
                   'loaded_pretrained_checkpoint': False, 'parameters': sum(p.numel() for p in self.network.parameters()),
                   'trainable_parameters': sum(p.numel() for p in self.network.parameters() if p.requires_grad),
                   'refiner_parameters': sum(p.numel() for p in self.network.refiner.parameters()) if self.group == 'P2' else 0,
                   'optimizer_includes_all_refiner_parameters': True if self.group == 'P2' else None})
        write_json(self.work / 'resolved_training_config.json', {
            'group': self.group, 'optimizer': self.optimizer.__class__.__name__,
            'optimizer_hyperparameters': {key: value for key, value in self.optimizer.param_groups[0].items() if key != 'params'},
            'scheduler': self.lr_scheduler.__class__.__name__,
            'deep_supervision_weights': list(self.loss.weight_factors),
            'foreground_oversampling_fraction': self.oversample_foreground_percent,
            'augmentation_processes': 0, 'network_compile': False, 'DDP': False})

    def _build_loss(self):
        loss = super()._build_loss()
        if self.group in ('P1', 'P2'):
            from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
            if not isinstance(loss, DeepSupervisionWrapper):
                raise ValueError('Expected the shared deep-supervision loss wrapper.')
            loss.loss = SegmentationWithSkeletonRecall(loss.loss, batch_dice=self.configuration_manager.batch_dice, weight=1.0)
        return loss

    def get_training_transforms(self, *args, **kwargs):
        kwargs['order_resampling_seg'] = 0
        transforms = super().get_training_transforms(*args, **kwargs)
        if self.group in ('P1', 'P2'):
            if transforms.transforms[-1].__class__.__name__ != 'NumpyToTensor':
                raise ValueError('Unexpected augmentation order; do not insert skeletonization blindly.')
            transforms.transforms.insert(-1, SkeletonTargetTransform())
        write_json(self.work / 'training_augmentation_snapshot.json', transform_snapshot(transforms))
        return transforms

    def get_validation_transforms(self, *args, **kwargs):
        transforms = super().get_validation_transforms(*args, **kwargs)
        if self.group in ('P1', 'P2'):
            if transforms.transforms[-1].__class__.__name__ != 'NumpyToTensor':
                raise ValueError('Unexpected validation transform order.')
            transforms.transforms.insert(-1, SkeletonTargetTransform())
        write_json(self.work / 'validation_augmentation_snapshot.json', transform_snapshot(transforms))
        return transforms

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()
        # Extra-module initialization must never alter the data sampling seed.
        if not self.resuming:
            seed_everything(2026)
        super().on_train_start()
        if self.resuming:
            py, np_state, cpu, cuda = self.resume_rng
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda)
        self.training_started_monotonic = time.monotonic()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.gradient_norms = []
        self.epoch_started_monotonic = time.monotonic()

    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        targets = batch['target']
        targets = [target.to(self.device, non_blocking=True) for target in targets] if isinstance(targets, list) else targets.to(self.device, non_blocking=True)
        if len(self.sample_order_prefix) < 16:
            self.sample_order_prefix.append(list(batch.get('keys', [])))
        for retry in range(8):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=True):
                outputs = self.network(data)
                if self.optimizer_steps == 0:
                    shapes = [list(value.shape) for value in outputs] if isinstance(outputs, (list, tuple)) else [list(outputs.shape)]
                    if shapes[0][1] != 2 or shapes[0][2:] != list(data.shape[2:]):
                        raise ValueError('Deep supervision output0 must be the binary highest-resolution logits.')
                    write_json(self.work / 'output_shape_probe.json', {'input': list(data.shape), 'outputs': shapes,
                                'highest_resolution_index': 0, 'classes': 2})
                loss = self.loss(outputs, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; no optimizer step counted.')
            scale = self.grad_scaler.get_scale()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            if self.grad_scaler.get_scale() >= scale:
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient accepted by optimizer.')
                self.optimizer_steps += 1
                self.gradient_norms.append(float(norm))
                if self.optimizer_steps == 16:
                    write_json(self.work / 'sample_order_prefix.json', {'seed': 2026, 'case_ids': self.sample_order_prefix})
                    common_order = self.shared / ('sample_order_' + self.run_kind + '.json')
                    if common_order.exists():
                        from .preprocess import read_json
                        if read_json(common_order)['case_ids'] != self.sample_order_prefix:
                            raise ValueError('Initial shared-seed sample ordering differs across groups.')
                    else:
                        write_json(common_order, {'seed': 2026, 'case_ids': self.sample_order_prefix})
                if self.group == 'P2' and self.optimizer_steps % 250 == 0:
                    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True):
                        # Same training batch, no model update or RNG sampling.
                        _, aux = self.network(data[:1], return_aux=True)
                        delta = aux['delta'].detach().abs().flatten()
                        self.last_delta_statistics = {'delta_abs_mean': float(delta.mean()),
                            **{'delta_abs_p' + str(int(q * 100)): float(torch.quantile(delta, q)) for q in [.5, .9, .99]}}
                return {'loss': loss.detach().cpu().numpy()}
            self.amp_retries += 1
            self.print_to_log_file('AMP overflow: retrying same batch, optimizer step not counted', retry + 1)
        raise FloatingPointError('Eight AMP retries failed; stop without changing the experiment definition.')

    def validation_step(self, batch):
        if self.group == 'P0':
            return super().validation_step(batch)
        target = [item.to(self.device, non_blocking=True) for item in batch['target']]
        original_loss = self.loss
        self.loss = lambda output, _: original_loss(output, target)
        try:
            return super().validation_step({**batch, 'target': [item[:, :1] for item in target]})
        finally:
            self.loss = original_loss

    @torch.no_grad()
    def validate_volumes(self, export_first=False, case_ids=None):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
                                    perform_everything_on_device=True, device=self.device,
                                    verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                         self.dataset_json, self.__class__.__name__, None)
        cases = self.do_split()[1] if case_ids is None else case_ids
        allowed = set(self.do_split()[1])
        if not set(cases).issubset(allowed):
            raise ValueError('Full validation can only access the frozen16 validation cases.')
        dataset = nnUNetDataset(self.preprocessed_dataset_folder, cases)
        reader = self.plans_manager.image_reader_writer_class()
        rows = []
        try:
            for case_id in cases:
                started = time.monotonic()
                data, _, props = dataset.load_case(case_id)
                logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(np.asarray(data, dtype=np.float32))).cpu()
                prediction = convert_predicted_logits_to_segmentation_with_correct_shape(logits, self.plans_manager,
                                self.configuration_manager, self.label_manager, props)
                del logits
                path = self.shared / 'raw' / DATASET / 'labelsVa' / (case_id + '.nii.gz')
                target, properties = reader.read_seg(str(path))
                if prediction.shape != target[0].shape or properties['sitk_stuff'] != props['sitk_stuff']:
                    raise ValueError('Validation prediction did not return to native GT physical grid.')
                if not np.isin(np.unique(prediction), [0, 1]).all():
                    raise ValueError('Nonbinary validation prediction.')
                gt, pred = target[0] == 1, prediction == 1
                dice = 2 * int(np.count_nonzero(gt & pred)) / (int(gt.sum()) + int(pred.sum()))
                if export_first and not rows:
                    reader.write_seg(prediction.astype(np.uint8), str(self.work / 'smoke_native_prediction.nii.gz'), props)
                rows.append({'case': case_id, 'seconds': time.monotonic() - started, 'dice': {'1': dice}})
                self.print_to_log_file('PARSE full validation', case_id, rows[-1]['seconds'], dice)
        finally:
            self.set_deep_supervision_enabled(True)
        mean = float(np.mean([row['dice']['1'] for row in rows]))
        result = {'macro_dice': mean, 'mean_dice': {'1': mean}, 'cases': rows,
                  'selection_metric': 'mean patient pulmonary-artery Dice on frozen16 cases', 'optimizer_steps': self.optimizer_steps}
        if case_ids is None:
            if len(rows) != 16:
                raise ValueError('Checkpoint selection requires all16 cases.')
            history = self.work / 'validation_history.csv'
            new_file = not history.exists()
            with history.open('a', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['optimizer_steps', 'mean_patient_dice', 'n_patients', 'full_validation_seconds'])
                if new_file:
                    writer.writeheader()
                writer.writerow({'optimizer_steps': self.optimizer_steps, 'mean_patient_dice': mean,
                                 'n_patients': len(rows), 'full_validation_seconds': sum(row['seconds'] for row in rows)})
        return result

    def save_checkpoint(self, filename):
        if self.disable_checkpointing:
            return
        path = Path(filename)
        temporary = path.with_suffix('.pth.part')
        state = {'network_weights': self.network.state_dict(), 'optimizer_state': self.optimizer.state_dict(),
                 'grad_scaler_state': self.grad_scaler.state_dict(), 'scheduler_state': self.lr_scheduler.state_dict(),
                 'logging': self.logger.get_checkpoint(), '_best_ema': self._best_ema,
                 'best_volume_dice': self.best_volume_dice, 'current_epoch': self.current_epoch + 1,
                 'init_args': self.my_init_kwargs, 'trainer_name': self.__class__.__name__,
                 'inference_allowed_mirroring_axes': None, 'experiment_family': 'parse22_supervised',
                 'group': self.group, 'run_kind': self.run_kind, 'optimizer_steps': self.optimizer_steps,
                 'split_sha256': self.experiment_config['split_sha256'], 'initial_backbone_sha256': self.initial_backbone_hash,
                 'training_records': self.training_records, 'amp_retries': self.amp_retries,
                 'peak_training_allocated_gib': self.training_peak_allocated_gib,
                 'peak_training_reserved_gib': self.training_peak_reserved_gib,
                 'accumulated_training_wall_seconds': self.elapsed_before_resume + (time.monotonic() - self.training_started_monotonic if hasattr(self, 'training_started_monotonic') else 0),
                 'sample_order_prefix': self.sample_order_prefix,
                 'rng_state': (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())}
        torch.save(state, temporary)
        temporary.replace(path)

    def load_checkpoint(self, filename):
        state = torch.load(filename, map_location='cpu')
        if state.get('experiment_family') != 'parse22_supervised' or state.get('group') != self.group or state.get('run_kind') != self.run_kind:
            raise ValueError('Cannot initialize a formal experiment from another group, AirRC or a smoke checkpoint.')
        if state['split_sha256'] != self.experiment_config['split_sha256']:
            raise ValueError('Resume split differs.')
        super().load_checkpoint(str(filename))
        self.optimizer_steps = state['optimizer_steps']
        self.training_records = state['training_records']
        self.amp_retries = state['amp_retries']
        self.sample_order_prefix = state['sample_order_prefix']
        self.resume_rng = state['rng_state']
        self.resuming = True
        self.elapsed_before_resume = state.get('accumulated_training_wall_seconds', 0)
        self.training_peak_allocated_gib = state.get('peak_training_allocated_gib', 0)
        self.training_peak_reserved_gib = state.get('peak_training_reserved_gib', 0)
        # A crash between writing validation rows and saving a new latest
        # checkpoint must not leave duplicated or future observations on resume.
        history = self.work / 'validation_history.csv'
        if history.exists():
            with history.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = [row for row in reader if int(row['optimizer_steps']) <= self.optimizer_steps]
            with history.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def on_epoch_end(self):
        self.training_peak_allocated_gib = max(self.training_peak_allocated_gib, torch.cuda.max_memory_allocated(self.device) / 1024 ** 3)
        self.training_peak_reserved_gib = max(self.training_peak_reserved_gib, torch.cuda.max_memory_reserved(self.device) / 1024 ** 3)
        # Save the epoch checkpoint only after any required full validation
        # finishes, so resuming can never silently skip a2000-update selection.
        self.disable_checkpointing = True
        try:
            nnUNetTrainer.on_epoch_end(self)
        finally:
            self.disable_checkpointing = False
        epoch = self.current_epoch
        if self.optimizer_steps != epoch * 250:
            raise ValueError('Epoch count and actual optimizer update budget diverged.')
        if epoch % self.validate_every == 0 or epoch == self.num_epochs:
            result = self.validate_volumes()
            write_json(Path(self.output_folder) / ('validation_epoch_%04d.json' % epoch), result)
            if result['macro_dice'] > self.best_volume_dice:
                self.best_volume_dice = result['macro_dice']
                self.current_epoch -= 1
                try:
                    self.save_checkpoint(Path(self.output_folder) / 'checkpoint_best.pth')
                finally:
                    self.current_epoch = epoch
        logs = self.logger.get_checkpoint()
        record = {'epoch': epoch, 'optimizer_steps': self.optimizer_steps, 'train_loss': float(logs['train_losses'][-1]),
                  'patch_validation_loss': float(logs['val_losses'][-1]), 'learning_rate': self.optimizer.param_groups[0]['lr'],
                  'epoch_seconds': time.monotonic() - self.epoch_started_monotonic,
                  'peak_training_allocated_gib': self.training_peak_allocated_gib,
                  'peak_training_reserved_gib': self.training_peak_reserved_gib,
                  'amp_retries_total': self.amp_retries, 'mean_gradient_norm': float(np.mean(self.gradient_norms)),
                  **{key: getattr(self, 'last_delta_statistics', {}).get(key) for key in ['delta_abs_mean', 'delta_abs_p50', 'delta_abs_p90', 'delta_abs_p99']}}
        self.training_records.append(record)
        with (self.work / 'training_history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(self.training_records)
        # A common250-update latest-checkpoint cadence limits recoverable loss;
        # best-checkpoint selection remains the shared2000-update full validation.
        self.current_epoch -= 1
        try:
            self.save_checkpoint(Path(self.output_folder) / 'checkpoint_latest.pth')
        finally:
            self.current_epoch = epoch

    def on_train_end(self):
        # The installed base assumes multiprocess augmenters with _finish().
        # Our common deterministic SingleThreadedAugmenter has no such method.
        self.current_epoch -= 1
        try:
            self.save_checkpoint(Path(self.output_folder) / 'checkpoint_final.pth')
        finally:
            self.current_epoch += 1
        for loader in (self.dataloader_train, self.dataloader_val):
            finish = getattr(loader, '_finish', None)
            if finish is not None:
                finish()
        torch.cuda.empty_cache()
        self.print_to_log_file('Training done:100000 successful optimizer updates.')
