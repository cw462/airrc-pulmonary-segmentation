"""Locked queue: complete P0, then run independent P1/P2 on separate GPUs."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .preprocess import read_json, write_json
from .scheduling import training_arguments, training_finished

QUEUE_LOCK_FD = None


def process_matches(pid, module, root, group=None, start_ticks=None):
    proc = Path('/proc') / str(pid)
    try:
        arguments = (proc / 'cmdline').read_bytes().decode().split('\0')
        if module not in arguments or str(root) not in arguments:
            return False
        if group is not None and ('--group' not in arguments or arguments[arguments.index('--group') + 1] != group):
            return False
        if start_ticks is not None:
            actual = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
            if str(start_ticks) != actual:
                return False
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False


def launch(root, module, arguments, log_name, label, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu
    with (root / log_name).open('ab', buffering=0) as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', module, *arguments], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   pass_fds=(QUEUE_LOCK_FD,))
    path = root / 'active_children.json'
    children = read_json(path) if path.exists() else {}
    info = {'pid': process.pid, 'module': module, 'arguments': arguments,
            'log': log_name, 'started_unix': time.time(), 'cuda_visible_devices': gpu}
    children[label] = info
    write_json(path, children)
    write_json(root / 'active_child.json', info)
    return process


def child(root, module, arguments, log_name, label, gpu=None):
    process = launch(root, module, arguments, log_name, label, gpu)
    code = process.wait()
    if code:
        raise RuntimeError('{} exited{}; inspect{}'.format(module, code, log_name))


def gpu_available(uuid):
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], universal_newlines=True)
    return not any(row.split(',')[0].strip() == uuid for row in text.splitlines() if row.strip())


def wait_for_adopted_p0(root, pid, start_ticks, poll_seconds=10):
    # Process identity is checked independently of inherited lock descriptors.
    while process_matches(pid, 'parse22_supervised.train', root, 'P0', start_ticks):
        write_json(root / 'pipeline_state.json', {'state': 'training', 'group': 'P0', 'pid': os.getpid(),
                   'adopted_p0_pid': pid, 'schedule': 'P0_then_P1_P2_parallel'})
        time.sleep(poll_seconds)


def parallel_training(root, assignments, poll_seconds=2):
    processes, failures = {}, {}
    pending = [group for group in ('P1', 'P2') if not training_finished(root, group)]
    # Spawn both before waiting. One visible GPU per process keeps logical
    # cuda:0 and the original batch2 for each independent model, without DDP.
    while processes or pending:
        for group in list(pending):
            if gpu_available(assignments[group]):
                processes[group] = launch(root, 'parse22_supervised.train', training_arguments(root, group),
                                          group + '_full.log', group, assignments[group])
                pending.remove(group)
        running = []
        for group, process in list(processes.items()):
            code = process.poll()
            if code is None:
                running.append(group)
            else:
                if code or not training_finished(root, group):
                    failures[group] = {'exit_code': code, 'log': group + '_full.log'}
                del processes[group]
        write_json(root / 'pipeline_state.json', {
            'state': 'training_partial_failure' if failures else 'training',
            'groups': running, 'waiting_for_gpu': list(pending),
            'completed_groups': [g for g in ('P0', 'P1', 'P2') if training_finished(root, g)],
            'failures': failures, 'pid': os.getpid(), 'schedule': 'P0_then_P1_P2_parallel'})
        if processes or pending:
            time.sleep(poll_seconds)
    if failures:
        # Do not cancel a healthy peer when another independent group fails.
        raise RuntimeError('Training failed for{}; other started groups were allowed to finish.'.format(','.join(failures)))


def main():
    global QUEUE_LOCK_FD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--mode', choices=['smoke', 'formal'], required=True)
    parser.add_argument('--adopt-p0', type=int)
    parser.add_argument('--adopt-start-ticks')
    args = parser.parse_args()
    root = args.root.resolve()
    state_path = root / 'pipeline_state.json'
    with (root / 'pipeline.lock').open('a') as lock:
        if args.adopt_p0:
            # The running P0 inherited this lock from the previous dispatcher.
            # Wait without signaling or restarting that training process.
            wait_for_adopted_p0(root, args.adopt_p0, args.adopt_start_ticks)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        QUEUE_LOCK_FD = lock.fileno()
        try:
            while not (root / 'shared/preprocessing_config.json').exists():
                pid = int((root / 'preprocess.pid').read_text())
                if not Path('/proc/{}/cmdline'.format(pid)).exists():
                    raise RuntimeError('Preprocessing exited before completed configuration was written.')
                write_json(state_path, {'state': 'waiting_preprocessing', 'mode': args.mode, 'pid': os.getpid()})
                time.sleep(20)
            if read_json(root / 'shared/preprocessing_config.json')['state'] != 'done':
                raise RuntimeError('Preprocessing incomplete.')
            if args.mode == 'smoke':
                for group in ('P0', 'P1', 'P2'):
                    report = root / group / 'smoke/smoke_report.json'
                    if report.exists() and read_json(report)['state'] == 'passed':
                        continue
                    write_json(state_path, {'state': 'running_smoke', 'group': group, 'pid': os.getpid()})
                    child(root, 'parse22_supervised.train', ['--root', str(root), '--group', group, '--mode', 'smoke'], group + '_smoke.log', group)
                write_json(state_path, {'state': 'smoke_suite_done', 'pid': os.getpid(), 'finished_unix': time.time()})
                return
            if read_json(root / 'engineering_smoke_review.json')['state'] != 'passed':
                raise RuntimeError('Review engineering smoke outputs before formal training.')
            schedule = read_json(root / 'execution_schedule.json')
            assignments = schedule['cuda_visible_devices']
            if len(set(assignments.values())) != 2 or assignments['P0'] != assignments['P1'] or assignments['P1'] == assignments['P2']:
                raise ValueError('Expected P0/P1 on one GPU and independent P2 on the other.')
            if not training_finished(root, 'P0'):
                write_json(state_path, {'state': 'training', 'group': 'P0', 'pid': os.getpid()})
                child(root, 'parse22_supervised.train', training_arguments(root, 'P0'), 'P0_full.log', 'P0', assignments['P0'])
            parallel_training(root, assignments)
            # Test evaluation remains deferred until all training is complete.
            for group in ('P0', 'P1', 'P2'):
                if read_json(root / group / 'summary.json').get('state') == 'done':
                    continue
                write_json(state_path, {'state': 'evaluating', 'group': group, 'pid': os.getpid()})
                child(root, 'parse22_supervised.evaluate', ['--root', str(root), '--group', group,
                      '--source-manifest', str(root / 'source_manifest.json')], group + '_evaluation.log', group + '_evaluation', assignments[group])
            child(root, 'parse22_supervised.compare', ['--root', str(root)], 'comparison.log', 'comparison')
            write_json(state_path, {'state': 'done', 'pid': os.getpid(), 'finished_unix': time.time()})
        except Exception as error:
            old = read_json(state_path) if state_path.exists() else {}
            write_json(state_path, {**old, 'state': 'failed', 'mode': args.mode, 'error': str(error), 'failed_unix': time.time()})
            raise


if __name__ == '__main__':
    main()
