"""Publish a report allowlist; never copy volumes, model weights, or connections."""
import csv
from pathlib import Path
import shutil

from .common import read_json, write_json

JSON_FILES = ['experiment_config.json', 'split_manifest.json', 'summary.json', 'evaluation_config.json',
              'prediction_manifest.json', 'initialization.json', 'resolved_training_config.json',
              'training_complete.json', 'evaluation_protocol.json', 'pause_state.json', 'resume_state.json']
CSV_FILES = ['training_history.csv', 'validation_history.csv', 'per_case.csv', 'av_metrics_per_case.csv',
             'av_confusion_counts.csv', 'evaluation_exceptions.csv', 'branch_details.csv']


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items() if key not in (
            'pid', 'gpu', 'cuda_visible_devices', 'allowed_cuda_visible_devices', 'queue_pid', 'pid_before_pause', 'adopted_training_pid')}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str) and (value.startswith('/') or (len(value)>2 and value[1:3] in (':/', ':\\'))):
        return value.replace('\\', '/').rsplit('/', 1)[-1]
    return value


def export(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for group in ('P0_M0', 'P1', 'P2'):
        work = source/group; target = destination/group
        if not work.exists():
            continue
        target.mkdir(exist_ok=True)
        for name in JSON_FILES:
            if (work/name).exists():
                write_json(target/name, sanitize(read_json(work/name)))
        for name in CSV_FILES:
            if (work/name).exists():
                shutil.copyfile(work/name, target/name)
        if (work/'smoke/smoke_report.json').exists():
            write_json(target/'engineering_smoke_report.json', sanitize(read_json(work/'smoke/smoke_report.json')))
    for name in ('experiment_matrix.json', 'engineering_review.json', 'execution_schedule_public.json',
                 'runtime_repair_public.json', 'source_version.json', 'status.json', 'comparison_protocol.json',
                 'user_control.json', 'execution_revision_pause_public.json', 'execution_revision_gpu0_public.json'):
        if (source/name).exists():
            write_json(destination/name, sanitize(read_json(source/name)))
    for name in ('comparison.csv', 'comparison.md'):
        if (source/name).exists():
            shutil.copyfile(source/name, destination/name)
    text = ['# AirRC M0 / P1 / P2', '',
            'M0 (P0_M0) is the completed four-label U-Mamba baseline. P1 adds Skeleton Recall Loss for all three foreground classes. P2 adds the user supplied CT guided residual refiner to P1.',
            'P1 and P2 start independently from scratch, each using one GPU. Device scheduling is recorded in execution_schedule_public.json. PARSE22 supervised training is paused by user instruction; its saved checkpoint is retained.', '',
            'The original fixed178/38/38 split,1mm preprocessing,128³ patch,batch2,4 training/2 validation augmentation workers and100000 stock AMP training iterations are reused. Successful optimizer updates and AMP skips are recorded separately; the historical M0 skip count is unavailable.',
            'Full validation38 every2000 iterations selects the mean patient Dice across PA/PV/airway lumen. Test38 is excluded from selection. M0 scores are imported unchanged from reports/airrc_m0.',
            'Basic, skeleton/branch and AV metrics retain the frozen project protocols. clDSC(%) equals clDice×100. BDR80 measures GT branch length coverage, not continuous connection.', '',
            'Each summary uses patient mean, sample_std(ddof=1), median and n_valid, separately for validation and test. Ratios are stored0–1; HD95 is mm. Paired bootstrap uses seed2026 and10000 resamples.',
            'One training seed does not establish robustness to training randomness. These three groups do not establish module complementarity or refinement alone.', '']
    status = source/'status.json'
    if status.exists():
        text.append('Current pipeline state: **{}**.'.format(read_json(status)['pipeline']['state']))
    policy = source/'user_control.json'
    if policy.exists() and read_json(policy).get('paused_groups'):
        text.append('User requested pause: **{}**. These groups await explicit user instruction to resume.'.format(
            ', '.join(sorted(read_json(policy)['paused_groups']))))
    if policy.exists() and read_json(policy).get('allowed_gpu_indices') == [0]:
        text.append('Current user scheduling: all remaining training and evaluation use GPU0 serially. GPU1 is reserved for the user’s other experiments.')
    (destination/'README.md').write_text('\n\n'.join(text)+'\n', encoding='utf-8')
