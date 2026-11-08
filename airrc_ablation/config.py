"""AirRC M0-compatible scientific settings; variants add only the declared method."""
from pathlib import Path
from .common import DATASET, CLASSES, read_json, write_json, digest


def make_config(root, group, mode):
    if group not in ('P1','P2'):raise ValueError('The existing M0 is the P0 baseline; do not retrain it here.')
    preprocessing=read_json(root/'shared/preprocessing_config.json')
    if preprocessing['state']!='done':raise ValueError('Frozen M0 cache is not ready.')
    source=read_json(root/'source_reference.json')
    base=Path(__file__).resolve().parent.parent
    files=['train_airrc.py','models/ct_local_refiner.py','airrc_ablation/common.py','airrc_ablation/skeleton_recall.py',
           'airrc_ablation/trainer.py','airrc_ablation/config.py','airrc_ablation/train.py']
    common={'experiment_family':'airrc_ablation','dataset':DATASET,'baseline':'completed AirRC M0, no new baseline training',
        'labels':{'background':0,**{name:label for label,name in CLASSES.items()}},'output_channels':4,
        'patient_counts':{'train':178,'val':38,'test':38},'patient_split_sha256':source['patient_split_sha256'],
        'frozen_M0_plan_sha256':preprocessing['plans_sha256'],'normalization':'reuse frozen M0 training178 statistics; no refit',
        'spacing_mm':[1,1,1],'patch':[128,128,128],'batch_size':2,'seed':2026,'initialization':'from_scratch',
        'epochs':400,'iterations_per_epoch':250,'training_iterations':100000,
        'iteration_semantics':'exact stock M0 AMP iteration budget; successful optimizer updates/skips recorded separately; historical M0 skip count unavailable',
        'optimizer':{'name':'SGD','lr':.01,'momentum':.99,'nesterov':True,'weight_decay':3e-5,'gradient_clip':12,'scheduler':'original polynomial400 epochs'},
        'base_loss':'original foreground Dice + cross entropy and unchanged deep-supervision weights',
        'augmentation':'original AirRC M0 pipeline,4 training/2 validation workers, stock nondeterministic batch arrival',
        'worker_seeds':None,'foreground_oversampling':.33,'AMP':True,'DDP':False,'network_compile':False,
        'validation':{'frequency_iterations':2000,'patients':38,'selection':'mean of three foreground-class patient-mean Dice','tie':'keep earlier'},
        'inference':{'tile_step_size':.5,'Gaussian':True,'TTA':False,'postprocessing':'none','grid':'same prepared1mm AirRC GT evaluation grid'},
        'skeleton_recall':{'foreground_classes':[1,2,3],'weight':1.,'class_weighting':'equal foreground macro',
            'tube':'Lee skeleton,2-voxel Manhattan dilation intersect own GT, after spatial augmentation',
            'smoothing':1e-5,'deep_supervision':'highest resolution only, retain original outer DS weights',
            'reference':'MIC-DKFZ/Skeleton-Recall@c7bac3c72ca7bb4f3d891235f051f0c4d07d7229'},
        'training_source_sha256':{name:digest(base/name) for name in files}}
    matrix={**common,'groups':{'P1':{'refiner':False},'P2':{'refiner':{'source':'user supplied CT_Local_Refiner_Codex.md',
           'num_classes':4,'hidden_channels':16,'max_delta':1.,'gate_floor':.25,'dtype':'FP32 under AMP',
           'initial_residual':'zero','conditions_detached':True,'stage':'highest-resolution patch logits before Gaussian fusion',
           'extra_residual_regularizer':False}}}}
    frozen=root/'experiment_matrix.json'
    if mode=='full':
        if frozen.exists() and read_json(frozen)!=matrix:raise ValueError('Frozen AirRC conditions/source changed.')
        if not frozen.exists():write_json(frozen,matrix)
    return {**common,**matrix['groups'][group],'group':group,'run_kind':mode}


def initialize_reports(work,config,split):
    work.mkdir(parents=True,exist_ok=True)
    path=work/'experiment_config.json'
    if path.exists() and read_json(path)!=config:raise ValueError('Existing experiment configuration differs.')
    write_json(path,config);write_json(work/'split_manifest.json',split)
    for name,value in {'summary.json':{'state':'not_evaluated','group':config['group'],'scores':None},
        'evaluation_config.json':{'state':'pending_training','inference':config['inference']},
        'prediction_manifest.json':{'state':'not_evaluated','predictions':[]}}.items():
        if not (work/name).exists():write_json(work/name,value)
    for name,header in {'training_history.csv':'epoch,iterations,train_loss,patch_validation_loss\n',
          'validation_history.csv':'iterations,selection_macro_dice,n_patients,seconds\n',
          'per_case.csv':'split,case_id,class,dice,iou,cldice,hd95_mm,precision,recall\n'}.items():
        if not (work/name).exists():(work/name).write_text(header,encoding='utf-8')
