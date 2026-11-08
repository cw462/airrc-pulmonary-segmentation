"""Native fixed-GT AirRC variant evaluation using unchanged M0 metrics and branch protocol."""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import skimage
import torch

from .common import DATASET,CONFIGURATION,CLASSES,read_json,write_json,configure_paths
from .evaluation_utils import clean,stats,csv_output

SKELETON=['skeleton_recall','skeleton_precision','cldice_recomputed','cldice_absolute_difference',
          'branch_detection_rate_80','branch_detection_rate_90','branch_coverage_mean','gt_branch_count',
          'detected_branch_count_80','detected_branch_count_90','excluded_short_branch_count',
          'excluded_short_branch_total_length_mm','excluded_cycle_component_count','degenerate_structure_count']
AV=['av_mcs','artery_to_vein_rate','vein_to_artery_rate','merged_vessel_dice','merged_vessel_precision','merged_vessel_recall']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--group',required=True,choices=['P1','P2'])
    args=parser.parse_args();root=args.root.resolve();work=root/args.group
    if skimage.__version__!='0.21.0':raise ValueError('Use the frozen M0 skeleton runtime0.21.0.')
    configure_paths(root/'shared',work/'results')
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from models.ct_local_refiner import RefinedSegmenter
    from evaluate_airrc import predict_case,score_class,METRICS
    from evaluation.label_geometry import read_label,geometry,same_grid
    from evaluation.evaluate_skeleton_branches import protocol,load_or_build_graph
    from evaluation.skeleton_metrics import score_skeletons
    from evaluation.branch_metrics import branch_coverages
    from evaluation.av_metrics import compute_av_metrics
    split=read_json(root/'patient_splits.json');source=read_json(root/'source_reference.json');config=read_json(work/'experiment_config.json')
    completed=read_json(work/'training_complete.json');assert completed['training_iterations']==100000
    checkpoint=Path(completed['best_checkpoint']);state=torch.load(checkpoint,map_location='cpu')
    assert state['experiment_family']=='airrc_ablation' and state['group']==args.group and state['run_kind']=='full'
    assert state['patient_split_sha256']==config['patient_split_sha256']
    selected={'group':args.group,'checkpoint_filename':checkpoint.name,'selected_iteration':state['iteration_count'],
              'selected_epoch':state['current_epoch'],'validation_macro_dice':state['best_volume_dice'],
              'patient_split_sha256':config['patient_split_sha256']}
    prior=read_json(work/'evaluation_config.json')
    if 'checkpoint' in prior and prior['checkpoint']!=selected:raise ValueError('Existing predictions refer to another checkpoint.')
    write_json(work/'evaluation_config.json',{'state':'running','checkpoint':selected,'inference':config['inference']})
    plans=read_json(root/'shared/preprocessed'/DATASET/'nnUNetPlans.json')
    dataset=read_json(root/'shared/raw'/DATASET/'dataset.json');manager=PlansManager(plans);configuration=manager.get_configuration(CONFIGURATION)
    network=nnUNetTrainerUMambaBot.build_network_architecture(manager,dataset,configuration,1,False)
    if args.group=='P2':network=RefinedSegmenter(network,4,hidden_channels=16,max_delta=1.,gate_floor=.25)
    network.load_state_dict(state['network_weights'],strict=True);network=network.to('cuda:0').eval();del state
    torch.set_num_threads(4)
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,perform_everything_on_device=True,
        device=torch.device('cuda',0),verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    predictor.manual_initialization(network,manager,configuration,None,dataset,'AirRCVariantTrainer',None)
    cases=[{'case_id':case,'split':partition,'gt_path':case+'.nii.gz'} for partition in ['val','test'] for case in split[partition]]
    core=protocol({'cases':cases});reference=read_json(root/'reference_branch_protocol.json')
    for name in ['branch_metrics.py','skeleton_metrics.py','evaluate_airrc.py']:
        assert core['source_code_sha256'][name]==reference['source_code_sha256'][name]
    write_json(work/'evaluation_protocol.json',{'branch_protocol':core,'labels':dataset['labels'],
        'av_formula':'(GT_artery_pred_vein + GT_vein_pred_artery)/(GT_A_or_V_count + Pred_A_or_V_count); no factor2',
        'HiPaS_implementation_identity_claimed':False,'original_M0_metric_implementations_retained':True})
    (work/'case_results').mkdir(exist_ok=True)
    rows=[];av_rows=[];counts=[];details=[];exceptions=[];manifest=[]
    for index,case in enumerate(cases,1):
        case_id=case['case_id'];partition=case['split']
        pred_path=work/'predictions'/partition/(case_id+'.nii.gz');pred_path.parent.mkdir(parents=True,exist_ok=True)
        case_path=work/'case_results'/(partition+'_'+case_id+'.json')
        if case_path.exists():
            result=read_json(case_path);assert result['checkpoint']==selected and pred_path.exists()
        else:
            raw=root/'shared/raw'/DATASET if partition=='val' else Path(source['test_root'])
            suffix='Va' if partition=='val' else 'Ts'
            target=read_label(raw/('labels'+suffix)/(case_id+'.nii.gz'),'unified')
            started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            timing=pred_path.with_suffix('.timing.json')
            if not pred_path.exists():
                cache=root/'shared/preprocessed'/DATASET/configuration.data_identifier/case_id if partition=='val' else None
                predicted,properties=predict_case(predictor,raw/('images'+suffix)/(case_id+'_0000.nii.gz'),cache)
                temporary=pred_path.with_name(case_id+'.part.nii.gz')
                manager.image_reader_writer_class().write_seg(predicted.astype(np.uint8),str(temporary),properties)
                write_json(timing,{'checkpoint':selected,'seconds':time.monotonic()-started})
                temporary.replace(pred_path)
            image=read_label(pred_path,'unified');assert same_grid(geometry(image),geometry(target))
            gt=sitk.GetArrayViewFromImage(target);pred=sitk.GetArrayViewFromImage(image)
            structures=[];case_details=[];issues=[]
            for label,name in CLASSES.items():
                g=gt==label;p=pred==label
                basic=score_class(g,p,np.array(target.GetSpacing()[::-1]))
                graph,_,_=load_or_build_graph(g,geometry(target),case,name,Path(source['branch_cache']),core)
                skeletal,problems=score_skeletons(g,p,graph['coords_zyx']);branches,branch_rows=branch_coverages(graph,p)
                difference=abs(basic['cldice']-skeletal['cldice_recomputed']) if basic['cldice'] is not None else None
                if difference is not None and difference>1e-12:raise ValueError('Recomputed clDice differs from original metric.')
                if not branches['gt_branch_count']:
                    problems.extend((metric,'no_effective_GT_branches') for metric in ['branch_detection_rate_80','branch_detection_rate_90','branch_coverage_mean'])
                structures.append({'split':partition,'case_id':case_id,'class':name,**basic,**skeletal,**branches,'cldice_absolute_difference':difference})
                case_details.extend({'split':partition,'case_id':case_id,'class':name,**row} for row in branch_rows)
                issues.extend({'split':partition,'case_id':case_id,'class':name,'metric':metric,'reason':reason} for metric,reason in problems)
            av=compute_av_metrics(gt,pred)
            for metric in AV:
                if not math.isfinite(av[metric]):issues.append({'split':partition,'case_id':case_id,'class':'A_V','metric':metric,'reason':'zero_denominator'})
            result=clean({'checkpoint':selected,'case_id':case_id,'split':partition,'structures':structures,'av':av,
                         'branch_details':case_details,'exceptions':issues,'geometry':geometry(target),
                         'inference_seconds':read_json(timing)['seconds'] if timing.exists() else None,
                         'peak_inference_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
            write_json(case_path,result)
        rows.extend(result['structures']);details.extend(result['branch_details']);exceptions.extend(result['exceptions'])
        av_rows.append({'case_id':case_id,'split':partition,**{key:result['av'][key] for key in AV}})
        counts.append({'case_id':case_id,'split':partition,**{key:value for key,value in result['av'].items() if key not in AV}})
        manifest.append({'case_id':case_id,'split':partition,'prediction':str(pred_path.relative_to(work)),'checkpoint':selected,
                         'geometry':result['geometry'],'inference_seconds':result['inference_seconds'],
                         'peak_inference_allocated_gib':result['peak_inference_allocated_gib']})
        print('AIRRC_VARIANT_EVAL',args.group,index,'/76',case_id,partition,flush=True)
    csv_output(work/'per_case.csv',rows,['split','case_id','class',*METRICS,*SKELETON,'empty','legacy_cldice_policy'])
    csv_output(work/'av_metrics_per_case.csv',av_rows,['case_id','split',*AV])
    csv_output(work/'av_confusion_counts.csv',counts,list(counts[0]))
    csv_output(work/'branch_details.csv',details,['split','case_id','class','branch_id','length_mm','coverage','detected80','detected90','exclusion_reason'])
    csv_output(work/'evaluation_exceptions.csv',exceptions,['split','case_id','class','metric','reason'])
    scores={partition:{name:{metric:stats([row.get(metric) for row in rows if row['split']==partition and row['class']==name])
                     for metric in list(METRICS)+SKELETON} for name in CLASSES.values()} for partition in ['val','test']}
    macro=np.mean([scores['val'][name]['dice']['mean'] for name in CLASSES.values()])
    if abs(macro-selected['validation_macro_dice'])>1e-3:raise ValueError('Final validation differs from checkpoint selection.')
    summary={'state':'done','experiment_family':'airrc_ablation','group':args.group,'patient_counts':{'val':38,'test':38},
        'scores':scores,'av_scores':{partition:{metric:stats([row[metric] for row in av_rows if row['split']==partition]) for metric in AV} for partition in ['val','test']},
        'checkpoint':selected,'training':completed,'model':read_json(work/'initialization.json'),'exceptions':len(exceptions),
        'inference_seconds':stats([row['inference_seconds'] for row in manifest]),
        'aggregation':'patient macro mean/sample_std(ddof=1)/median/n_valid; validation and test separated'}
    write_json(work/'summary.json',clean(summary));write_json(work/'prediction_manifest.json',{'state':'done','predictions':manifest,'checkpoint':selected})
    write_json(work/'evaluation_config.json',{'state':'done','checkpoint':selected,'inference':config['inference']})
    print('AIRRC_VARIANT_EVALUATION_DONE',args.group,flush=True)


if __name__=='__main__':main()
