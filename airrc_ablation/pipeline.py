"""Two independent single-GPU AirRC variants; completed M0 is never retrained."""
import argparse
import fcntl
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from .common import read_json, write_json

LOCK_FD = None


def process_matches(pid, module, root, group=None):
    try:
        args = (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
        return module in args and str(root) in args and (group is None or
            ('--group' in args and args[args.index('--group')+1] == group))
    except (FileNotFoundError, ProcessLookupError):
        return False


def trained(root, group):
    path = root/group/'training_complete.json'
    return path.exists() and read_json(path).get('training_iterations') == 100000 and read_json(path).get('group') == group


def training_args(root, group, mode):
    arguments = ['--root', str(root), '--group', group, '--mode', mode]
    if mode == 'full' and not trained(root, group):
        checkpoints = list((root/group/'results').glob('*/AirRCVariantTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_latest.pth'))
        if len(checkpoints) > 1:
            raise ValueError('Ambiguous same-group recovery checkpoint.')
        if checkpoints:
            arguments += ['--resume', str(checkpoints[0])]
        elif (root/group/'training_history.csv').exists() and len((root/group/'training_history.csv').read_text().splitlines()) > 1:
            raise ValueError('Training progress exists but its latest checkpoint is missing.')
    return arguments


def gpu_available(uuid):
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], universal_newlines=True)
    return not any(line.split(',')[0].strip() == uuid for line in output.splitlines() if line.strip())


def launch(root, module, arguments, log_name, label, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu
    with (root/log_name).open('ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-u', '-m', module, *arguments],
            env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, pass_fds=(LOCK_FD,))
    path = root/'active_children.json'; children = read_json(path) if path.exists() else {}
    children[label] = {'pid': child.pid, 'module': module, 'arguments': arguments,
                       'log': log_name, 'started_unix': time.time(), 'cuda_visible_devices': gpu}
    write_json(path, children)
    return child


def run_parallel(root, assignments, mode, poll_seconds=5):
    pending = [g for g in ('P1', 'P2') if not (trained(root, g) if mode == 'full' else
               (root/g/'smoke/smoke_report.json').exists() and read_json(root/g/'smoke/smoke_report.json')['state'] == 'passed')]
    processes, failures = {}, {}
    while pending or processes:
        for group in list(pending):
            if gpu_available(assignments[group]):
                processes[group] = launch(root, 'airrc_ablation.train', training_args(root, group, mode),
                    group+'_'+mode+'.log', group, assignments[group])
                pending.remove(group)
        for group, child in list(processes.items()):
            if child.poll() is not None:
                complete = trained(root, group) if mode == 'full' else (root/group/'smoke/smoke_report.json').exists()
                if child.returncode or not complete:
                    failures[group] = {'exit_code': child.returncode, 'log': group+'_'+mode+'.log'}
                del processes[group]
        write_json(root/'pipeline_state.json', {'state': 'training_partial_failure' if failures else
            ('training' if mode == 'full' else 'engineering_smokes'), 'groups': list(processes),
            'waiting_for_gpu': pending, 'failures': failures, 'pid': os.getpid(),
            'completed_groups': [g for g in ('P1', 'P2') if trained(root, g)], 'PARSE22': 'paused_by_user'})
        if pending or processes:
            time.sleep(poll_seconds)
    if failures:
        raise RuntimeError('Failed groups {}; healthy peers were allowed to continue.'.format(failures))


def main():
    global LOCK_FD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=['smoke', 'formal'])
    args = parser.parse_args(); root = args.root.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); LOCK_FD = lock.fileno()
        try:
            assignments = read_json(root/'execution_schedule.json')['cuda_visible_devices']
            assert set(assignments) == {'P1', 'P2'} and len(set(assignments.values())) == 2
            assert read_json(root/'shared/preprocessing_config.json')['state'] == 'done'
            if args.mode == 'formal':
                assert read_json(root/'engineering_review.json')['state'] == 'passed'
                assert (root/'experiment_matrix.json').exists()
                for group in ('P1', 'P2'):
                    assert read_json(root/group/'smoke/smoke_report.json')['state'] == 'passed'
            run_parallel(root, assignments, 'smoke' if args.mode == 'smoke' else 'full')
            if args.mode == 'smoke':
                write_json(root/'pipeline_state.json', {'state': 'smoke_suite_done', 'pid': os.getpid()})
                return
            for group in ('P1', 'P2'):
                if read_json(root/group/'summary.json').get('state') == 'done':
                    continue
                while not gpu_available(assignments[group]):
                    write_json(root/'pipeline_state.json', {'state': 'waiting_evaluation_gpu', 'group': group, 'pid': os.getpid()})
                    time.sleep(20)
                write_json(root/'pipeline_state.json', {'state': 'evaluating', 'group': group, 'pid': os.getpid()})
                child = launch(root, 'airrc_ablation.evaluate', ['--root', str(root), '--group', group],
                    group+'_evaluation.log', group+'_evaluation', assignments[group])
                if child.wait():
                    raise RuntimeError(group+' evaluation failed; inspect saved predictions and evaluation log.')
            child = launch(root, 'airrc_ablation.compare', ['--root', str(root)], 'comparison.log', 'comparison')
            if child.wait():
                raise RuntimeError('Patient-paired comparison failed.')
            write_json(root/'pipeline_state.json', {'state': 'done', 'finished_unix': time.time(), 'pid': os.getpid()})
        except Exception as error:
            path = root/'pipeline_state.json'; previous = read_json(path) if path.exists() else {}
            write_json(path, {**previous, 'state': 'failed', 'error': str(error), 'failed_unix': time.time()})
            raise


if __name__ == '__main__':
    main()
