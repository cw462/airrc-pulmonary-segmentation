"""Independent from-scratch AirRC P1/P2, preserving the completed M0 training recipe."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .common import DATASET,CONFIGURATION,configure_paths,read_json,write_json


def smoke(trainer):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
    from evaluate_airrc import predict_case
    from models.ct_local_refiner import RefinedSegmenter
    trainer.on_train_start();trainer.on_train_epoch_start()
    batch=next(trainer.dataloader_train)
    assert set(torch.unique(batch['target'][0][:,:1]).tolist()).issubset({0,1,2,3})
    fit=[float(trainer.train_step(batch)['loss']) for _ in range(30)]
    for _ in range(20):trainer.train_step(next(trainer.dataloader_train))
    started=time.monotonic()
    for _ in range(200):trainer.train_step(next(trainer.dataloader_train))
    seconds=time.monotonic()-started
    assert trainer.iteration_count==250 and np.isfinite(fit).all() and fit[-1]<fit[0]
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.network.parameters())
    gradients=None
    if trainer.group=='P2':
        gradients={name:float(param.grad.abs().sum()) for name,param in
          [('head',trainer.network.refiner.residual_head.weight),('features',trainer.network.refiner.features[0].block[0].weight)]}
        assert min(gradients.values())>0
    peak={'allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
    trainer.network.eval()
    with torch.no_grad():patch=trainer.validation_step(next(trainer.dataloader_val))
    assert np.isfinite(float(patch['loss']))
    trainer.set_deep_supervision_enabled(False)
    x=batch['data'][:1].to(trainer.device)
    with torch.no_grad(),torch.autocast('cuda',enabled=True):before=trainer.network(x).float().cpu()
    checkpoint=trainer.work/'checkpoint_smoke.pth';trainer.save_checkpoint(checkpoint)
    restored=nnUNetTrainerUMambaBot.build_network_architecture(trainer.plans_manager,trainer.dataset_json,trainer.configuration_manager,1,False)
    if trainer.group=='P2':restored=RefinedSegmenter(restored,4,hidden_channels=16,max_delta=1.,gate_floor=.25)
    state=torch.load(checkpoint,map_location='cpu');restored.load_state_dict(state['network_weights'],strict=True)
    del state
    restored=restored.to(trainer.device).eval()
    with torch.no_grad(),torch.autocast('cuda',enabled=True):after=restored(x).float().cpu()
    delta=float((before-after).abs().max());assert delta<=1e-5
    del restored,x,before,after;torch.cuda.empty_cache()
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,perform_everything_on_device=True,
          device=trainer.device,verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    predictor.manual_initialization(trainer.network,trainer.plans_manager,trainer.configuration_manager,None,
                                   trainer.dataset_json,'AirRCVariantTrainer',None)
    case=trainer.do_split()[1][0];raw=trainer.work/'raw'/DATASET
    cache=Path(trainer.preprocessed_dataset_folder)/case
    started=time.monotonic();pred,props=predict_case(predictor,raw/'imagesVa'/(case+'_0000.nii.gz'),cache)
    reader=trainer.plans_manager.image_reader_writer_class()
    gt,ref_props=reader.read_seg(str(raw/'labelsVa'/(case+'.nii.gz')))
    assert pred.shape==gt[0].shape and props['sitk_stuff']==ref_props['sitk_stuff'] and np.isin(np.unique(pred),[0,1,2,3]).all()
    reader.write_seg(pred.astype(np.uint8),str(trainer.work/'smoke_native_prediction.nii.gz'),props)
    report={'state':'passed','group':trainer.group,'iterations':250,'timed_fresh_batch_iterations':200,
       'seconds_per_training_iteration':seconds/200,'initial_fixed_patch_loss':fit[0],'final_fixed_patch_loss':fit[-1],
       'successful_optimizer_updates':trainer.successful_updates,'AMP_skips':trainer.amp_skips,
       'peak_training_memory':peak,'checkpoint_reload_max_difference':delta,'refiner_gradients':gradients,
       'native_grid_case':case,'native_grid_verified':True,'native_grid_inference_seconds':time.monotonic()-started,
       'formal_initialization_uses_smoke_weights':False}
    write_json(trainer.work/'smoke_report.json',report);print('AIRRC_VARIANT_SMOKE_PASSED',report,flush=True)
    for loader in [trainer.dataloader_train,trainer.dataloader_val]:
        finish=getattr(loader,'_finish',None)
        if finish is not None:finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--group',required=True,choices=['P1','P2'])
    parser.add_argument('--mode',required=True,choices=['smoke','full'])
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args();root=args.root.resolve();work=root/args.group
    if args.mode=='smoke':work=work/'smoke'
    configure_paths(root/'shared',work/'results')
    from .config import make_config,initialize_reports
    from .trainer import AirRCVariantTrainer,seed_everything
    config=make_config(root,args.group,args.mode)
    if args.mode=='full':
        assert read_json(root/'engineering_review.json')['state']=='passed'
        assert read_json(root/args.group/'smoke/smoke_report.json')['state']=='passed'
        if args.resume is None and (work/'training_history.csv').exists() and len((work/'training_history.csv').read_text().splitlines())>1:
            raise ValueError('Existing training must resume its own checkpoint, not overwrite progress.')
    initialize_reports(work,config,read_json(root/'patient_splits.json'))
    alias=work/'raw'
    if not alias.exists():alias.symlink_to(root/'shared/raw',target_is_directory=True)
    seed_everything();torch.set_num_threads(4)
    plans=read_json(root/'shared/preprocessed'/DATASET/'nnUNetPlans.json')
    dataset=read_json(root/'shared/raw'/DATASET/'dataset.json')
    trainer=AirRCVariantTrainer(plans,CONFIGURATION,0,dataset,unpack_dataset=False,device=torch.device('cuda',0))
    trainer.group=args.group;trainer.shared=root/'shared';trainer.work=work;trainer.run_kind=args.mode;trainer.experiment_config=config
    trainer.num_epochs=400;trainer.num_iterations_per_epoch=250;trainer.validate_every=8
    trainer.initial_lr=.01;trainer.weight_decay=3e-5
    trainer.initialize()
    if args.resume:trainer.load_checkpoint(str(args.resume))
    if args.mode=='smoke':smoke(trainer)
    else:
        trainer.run_training()
        assert trainer.iteration_count==100000
        write_json(work/'training_complete.json',{'state':'trained','group':args.group,'training_iterations':trainer.iteration_count,
            'successful_optimizer_updates':trainer.successful_updates,'AMP_skips':trainer.amp_skips,
            'best_validation_macro_dice':trainer.best_volume_dice,'best_checkpoint':str(Path(trainer.output_folder)/'checkpoint_best.pth'),
            'wall_seconds':trainer.elapsed_before_resume+time.monotonic()-trainer.started,
            'peak_allocated_gib':trainer.peak_allocated,'peak_reserved_gib':trainer.peak_reserved,
            'patient_split_sha256':config['patient_split_sha256'],'initialized_from_trained_M0':False})


if __name__=='__main__':main()
