"""Original AirRC M0 training machinery with declared loss/refiner additions only."""
import csv
import fcntl
import hashlib
from pathlib import Path
import random
import time

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from train_airrc import AirRCTrainer
from models.ct_local_refiner import RefinedSegmenter

from .common import write_json, read_json
from .skeleton_recall import SegmentationWithSkeletonRecall, SkeletonTargetTransform


def seed_everything():
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)


def tensor_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class AirRCVariantTrainer(AirRCTrainer):
    group = 'P1'
    iteration_count = 0
    successful_updates = 0
    amp_skips = 0
    resuming = False

    def initialize(self):
        super().initialize()
        if self.is_ddp or self.label_manager.num_segmentation_heads != 4:
            raise ValueError('AirRC variants require one GPU and four output labels.')
        initial = tensor_hash(self.network)
        reference = self.shared / 'variant_initialization.json'
        with (self.shared/'variant_initialization.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if reference.exists():
                if read_json(reference)['backbone_initial_sha256'] != initial:
                    raise ValueError('P1/P2 backbone initializations differ.')
            else:
                write_json(reference, {'backbone_initial_sha256': initial, 'seed': 2026,
                                      'historical_M0_initial_tensors_saved': False,
                                      'M0_comparison': 'same documented seed/architecture; no retrospective claim of verified initial tensor identity'})
        self.initial_hash = initial
        if self.group == 'P2':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2026)
                self.network = RefinedSegmenter(self.network, 4, hidden_channels=16, max_delta=1., gate_floor=.25).to(self.device)
            # The original optimizer has no state before the first step. Rebuild
            # after wrapping so it includes every new parameter, with same settings.
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
            assert all(id(p) in ids for p in self.network.refiner.parameters())
            assert not torch.count_nonzero(self.network.refiner.residual_head.weight)
        self.records = []
        self.peak_allocated = self.peak_reserved = 0.
        self.elapsed_before_resume = 0.
        write_json(self.work / 'initialization.json', {'group': self.group, 'seed': 2026, 'labels': 4,
                   'backbone_initial_sha256': initial, 'loaded_trained_M0_weights': False,
                   'parameters': sum(p.numel() for p in self.network.parameters()),
                   'refiner_parameters': sum(p.numel() for p in self.network.refiner.parameters()) if self.group=='P2' else 0})
        write_json(self.work / 'resolved_training_config.json', {'optimizer': self.optimizer.__class__.__name__,
                   'optimizer_hyperparameters': {k:v for k,v in self.optimizer.param_groups[0].items() if k!='params'},
                   'deep_supervision_weights': list(self.loss.weight_factors), 'train_augmentation_workers':4,
                   'validation_augmentation_workers':2, 'foreground_oversampling':self.oversample_foreground_percent,
                   'worker_seeds':None, 'batch_arrival':'existing M0 NonDetMultiThreaded pipeline; not bitwise-replayable',
                   'iteration_budget':100000,'AMP_skip_policy':'same stock M0 behavior; record successful updates separately'})

    def _build_loss(self):
        loss = super()._build_loss()
        loss.loss = SegmentationWithSkeletonRecall(loss.loss, self.configuration_manager.batch_dice)
        return loss

    def get_training_transforms(self, *args, **kwargs):
        transform = super().get_training_transforms(*args, **kwargs)
        assert transform.transforms[-1].__class__.__name__=='NumpyToTensor'
        transform.transforms.insert(-1, SkeletonTargetTransform())
        return transform

    def get_validation_transforms(self, *args, **kwargs):
        transform = super().get_validation_transforms(*args, **kwargs)
        assert transform.transforms[-1].__class__.__name__=='NumpyToTensor'
        transform.transforms.insert(-1, SkeletonTargetTransform())
        return transform

    def on_train_start(self):
        if not self.resuming:
            seed_everything()
        super().on_train_start()
        self.started = time.monotonic()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.epoch_started = time.monotonic()

    def train_step(self, batch):
        scale = self.grad_scaler.get_scale()
        result = super().train_step(batch)
        self.iteration_count += 1
        if self.grad_scaler.get_scale() >= scale:
            self.successful_updates += 1
        else:
            self.amp_skips += 1
        if self.group=='P2' and self.iteration_count % 250==0:
            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True):
                _, aux = self.network(batch['data'][:1].to(self.device), return_aux=True)
                delta = aux['delta'].abs().flatten()
                self.delta_statistics = {'delta_abs_mean':float(delta.mean()),
                    **{'delta_abs_p'+str(int(q*100)):float(torch.quantile(delta,q)) for q in [.5,.9,.99]}}
        return result

    def validation_step(self, batch):
        target = [value.to(self.device) for value in batch['target']]
        loss = self.loss
        self.loss = lambda outputs, _: loss(outputs, target)
        try:
            return super().validation_step({**batch,'target':[value[:,:1] for value in target]})
        finally:
            self.loss = loss

    def validate_volumes(self, export_first=False):
        started = time.monotonic()
        result = super().validate_volumes(export_first)
        if len(result['cases']) != 38:
            raise ValueError('Complete selection validation requires the fixed38 patients.')
        history = self.work / 'validation_history.csv'
        with history.open('a',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['iterations','selection_macro_dice','n_patients','seconds'])
            if stream.tell()==0:writer.writeheader()
            writer.writerow({'iterations':self.iteration_count,'selection_macro_dice':result['macro_dice'],
                             'n_patients':38,'seconds':time.monotonic()-started})
        return result

    def save_checkpoint(self, filename):
        if self.disable_checkpointing:return
        path = Path(filename)
        state = {'network_weights':self.network.state_dict(),'optimizer_state':self.optimizer.state_dict(),
                 'scheduler_state':self.lr_scheduler.state_dict(),'grad_scaler_state':self.grad_scaler.state_dict(),
                 'logging':self.logger.get_checkpoint(),'_best_ema':self._best_ema,'best_volume_dice':self.best_volume_dice,
                 'current_epoch':self.current_epoch+1,'init_args':self.my_init_kwargs,'trainer_name':self.__class__.__name__,
                 'inference_allowed_mirroring_axes':None,'experiment_family':'airrc_ablation','group':self.group,
                 'run_kind':self.run_kind,'iteration_count':self.iteration_count,'successful_optimizer_updates':self.successful_updates,
                 'amp_skips':self.amp_skips,'patient_split_sha256':self.experiment_config['patient_split_sha256'],
                 'initial_backbone_sha256':self.initial_hash,'training_records':self.records,
                 'peak_allocated_gib':self.peak_allocated,'peak_reserved_gib':self.peak_reserved,
                 'elapsed_wall_seconds':self.elapsed_before_resume+(time.monotonic()-self.started if hasattr(self,'started') else 0),
                 'rng_state':(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())}
        temporary=path.with_suffix('.pth.part');torch.save(state,temporary);temporary.replace(path)

    def load_checkpoint(self, filename):
        state=torch.load(filename,map_location='cpu')
        if (state.get('experiment_family')!='airrc_ablation' or state.get('group')!=self.group
                or state.get('run_kind')!=self.run_kind or state['patient_split_sha256']!=self.experiment_config['patient_split_sha256']):
            raise ValueError('Resume only the same AirRC group/split/formal experiment, never M0 or another variant.')
        super().load_checkpoint(str(filename))
        self.iteration_count=state['iteration_count'];self.successful_updates=state['successful_optimizer_updates']
        self.amp_skips=state['amp_skips'];self.records=state['training_records'];self.resuming=True
        self.peak_allocated=state['peak_allocated_gib'];self.peak_reserved=state['peak_reserved_gib']
        self.elapsed_before_resume=state['elapsed_wall_seconds']
        self.lr_scheduler.load_state_dict(state['scheduler_state'])
        random.setstate(state['rng_state'][0]);np.random.set_state(state['rng_state'][1])
        torch.set_rng_state(state['rng_state'][2]);torch.cuda.set_rng_state(state['rng_state'][3],self.device)
        # A crash during validation may leave rows beyond the last committed
        # checkpoint. Keep only the validation history represented by it.
        history=self.work/'validation_history.csv'
        if history.exists():
            with history.open(newline='',encoding='utf-8') as stream:
                reader=csv.DictReader(stream);fields=reader.fieldnames
                rows=[r for r in reader if int(r['iterations'])<=self.iteration_count]
            with history.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

    def on_epoch_end(self):
        self.peak_allocated=max(self.peak_allocated,torch.cuda.max_memory_allocated()/1024**3)
        self.peak_reserved=max(self.peak_reserved,torch.cuda.max_memory_reserved()/1024**3)
        self.disable_checkpointing=True
        try:nnUNetTrainer.on_epoch_end(self)
        finally:self.disable_checkpointing=False
        epoch=self.current_epoch
        if self.iteration_count!=epoch*250:raise ValueError('Original M0 iteration budget diverged.')
        epoch_seconds=time.monotonic()-self.epoch_started
        validation_seconds=0.
        if epoch%8==0 or epoch==400:
            started=time.monotonic();result=self.validate_volumes();validation_seconds=time.monotonic()-started
            write_json(Path(self.output_folder)/('validation_epoch_%04d.json'%epoch),result)
            if result['macro_dice']>self.best_volume_dice:
                self.best_volume_dice=result['macro_dice'];self.current_epoch-=1
                try:self.save_checkpoint(Path(self.output_folder)/'checkpoint_best.pth')
                finally:self.current_epoch=epoch
        logs=self.logger.get_checkpoint()
        row={'epoch':epoch,'iterations':self.iteration_count,'successful_optimizer_updates':self.successful_updates,
             'amp_skips':self.amp_skips,'train_loss':float(logs['train_losses'][-1]),
             'patch_validation_loss':float(logs['val_losses'][-1]),'epoch_seconds_without_full_validation':epoch_seconds,
             'full_validation_seconds':validation_seconds,'learning_rate':self.optimizer.param_groups[0]['lr'],
             'peak_allocated_gib':self.peak_allocated,'peak_reserved_gib':self.peak_reserved,
             **{name:getattr(self,'delta_statistics',{}).get(name) for name in ['delta_abs_mean','delta_abs_p50','delta_abs_p90','delta_abs_p99']}}
        self.records.append(row)
        with (self.work/'training_history.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(self.records)
        self.current_epoch-=1
        try:self.save_checkpoint(Path(self.output_folder)/'checkpoint_latest.pth')
        finally:self.current_epoch=epoch
