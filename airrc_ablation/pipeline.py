"""Independent single-GPU AirRC variants; shared devices run serially."""
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


def process_matches(pid, module, root, group=None, start_ticks=None):
    try:
        args = (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
        matching = module in args and str(root) in args and (group is None or
            ('--group' in args and args[args.index('--group')+1] == group))
        if matching and start_ticks is not None:
            actual = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]
            matching = str(start_ticks) == actual
        return matching
    except (FileNotFoundError, ProcessLookupError):
        return False


def trained(root, group):
    path = root/group/'training_complete.json'
    return path.exists() and read_json(path).get('training_iterations') == 100000 and read_json(path).get('group') == group


def paused_groups(root):
    path = root/'user_control.json'
    return sorted(read_json(path).get('paused_groups', {})) if path.exists() else []


def validate_assignments(root, assignments):
    if set(assignments) != {'P1', 'P2'} or not all(isinstance(gpu, str) and gpu for gpu in assignments.values()):
        raise ValueError('Both AirRC groups require an explicit single GPU assignment.')
    path = root/'user_control.json'
    allowed = read_json(path).get('allowed_cuda_visible_devices') if path.exists() else None
    if allowed is not None and not set(assignments.values()).issubset(set(allowed)):
        raise ValueError('GPU assignments violate the user device reservation policy.')


def training_args(root, group, mode):
    if group in paused_groups(root):
        raise RuntimeError(group+' is paused_by_user; explicit user instruction is required to resume.')
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
        paused = paused_groups(root)
        busy_devices = {assignments[g] for g in processes}
        for group in list(pending):
            if group not in paused and assignments[group] not in busy_devices and gpu_available(assignments[group]):
                processes[group] = launch(root, 'airrc_ablation.train', training_args(root, group, mode),
                    group+'_'+mode+'.log', group, assignments[group])
                busy_devices.add(assignments[group])
                pending.remove(group)
        for group, child in list(processes.items()):
            if child.poll() is not None:
                complete = trained(root, group) if mode == 'full' else (root/group/'smoke/smoke_report.json').exists()
                if group in paused and not complete:
                    pending.append(group)
                elif child.returncode or not complete:
                    failures[group] = {'exit_code': child.returncode, 'log': group+'_'+mode+'.log'}
                del processes[group]
        write_json(root/'pipeline_state.json', {'state': 'training_partial_failure' if failures else
            ('waiting_user_resume' if not processes and pending and all(g in paused for g in pending) else
             'training' if mode == 'full' else 'engineering_smokes'), 'groups': list(processes),
            'paused_groups': paused, 'waiting_for_gpu': [g for g in pending if g not in paused], 'failures': failures, 'pid': os.getpid(),
            'completed_groups': [g for g in ('P1', 'P2') if trained(root, g)], 'PARSE22': 'paused_by_user'})
        if pending or processes:
            time.sleep(poll_seconds)
    if failures:
        raise RuntimeError('Failed groups {}; healthy peers were allowed to continue.'.format(failures))


def wait_for_existing_training(root, group, pid, start_ticks, poll_seconds=10):
    # The adopted trainer holds the previous controller's inherited file lock.
    # Wait by identity, without signaling it or opening a second training run.
    while process_matches(pid, 'airrc_ablation.train', root, group, start_ticks):
        write_json(root/'pipeline_state.json', {'state': 'training', 'groups': [group],
            'paused_groups': paused_groups(root), 'adopted_training_pid': pid,
            'failures': {}, 'pid': os.getpid(), 'PARSE22': 'paused_by_user'})
        time.sleep(poll_seconds)
    if not trained(root, group):
        raise RuntimeError('Adopted '+group+' exited before completing its fixed training budget.')


def main():
    global LOCK_FD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=['smoke', 'formal'])
    parser.add_argument('--adopt-group', choices=['P1', 'P2'])
    parser.add_argument('--adopt-pid', type=int)
    parser.add_argument('--adopt-start-ticks')
    args = parser.parse_args(); root = args.root.resolve()
    if any([args.adopt_group, args.adopt_pid, args.adopt_start_ticks]) and not all([args.adopt_group, args.adopt_pid, args.adopt_start_ticks]):
        parser.error('Adoption requires group, PID and process start ticks together.')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with (root/'pipeline.lock').open('a') as lock:
        try:
            if args.adopt_pid:
                wait_for_existing_training(root, args.adopt_group, args.adopt_pid, args.adopt_start_ticks)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); LOCK_FD = lock.fileno()
            assignments = read_json(root/'execution_schedule.json')['cuda_visible_devices']
            validate_assignments(root, assignments)
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
