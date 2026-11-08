"""Single-GPU, locked queue for the independent PARSE22 experiment family."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .preprocess import read_json, write_json

QUEUE_LOCK_FD = None


def child(root, module, arguments, log_name):
    with (root / log_name).open('ab', buffering=0) as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', module, *arguments],
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   pass_fds=(QUEUE_LOCK_FD,))
        write_json(root / 'active_child.json', {'pid': process.pid, 'module': module, 'arguments': arguments,
                                               'log': log_name, 'started_unix': time.time()})
        code = process.wait()
    if code:
        raise RuntimeError('{} exited{}; inspect{}'.format(module, code, log_name))


def main():
    global QUEUE_LOCK_FD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--mode', choices=['smoke', 'formal'], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    state_path = root / 'pipeline_state.json'
    with (root / 'pipeline.lock').open('a') as lock:
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
                for group in ['P0', 'P1', 'P2']:
                    report = root / group / 'smoke/smoke_report.json'
                    if report.exists() and read_json(report)['state'] == 'passed':
                        continue
                    write_json(state_path, {'state': 'running_smoke', 'group': group, 'pid': os.getpid()})
                    child(root, 'parse22_supervised.train', ['--root', str(root), '--group', group, '--mode', 'smoke'], group + '_smoke.log')
                write_json(state_path, {'state': 'smoke_suite_done', 'pid': os.getpid(), 'finished_unix': time.time()})
                return
            if not (root / 'engineering_smoke_review.json').exists() or read_json(root / 'engineering_smoke_review.json')['state'] != 'passed':
                raise RuntimeError('Review all engineering smoke outputs before starting formal training.')
            for group in ['P0', 'P1', 'P2']:
                work = root / group
                if (work / 'training_complete.json').exists() and read_json(work / 'training_complete.json')['optimizer_steps'] == 100000:
                    continue
                write_json(state_path, {'state': 'training', 'group': group, 'pid': os.getpid(), 'started_unix': time.time()})
                arguments = ['--root', str(root), '--group', group, '--mode', 'full']
                latest = list((work / 'results').glob('**/checkpoint_latest.pth')) if (work / 'results').exists() else []
                if len(latest) > 1:
                    raise RuntimeError('Ambiguous latest checkpoint paths; do not guess.')
                if latest:
                    arguments += ['--resume', str(latest[0])]
                child(root, 'parse22_supervised.train', arguments, group + '_full.log')
            # Test scores become visible only after all three formal training
            # trajectories have finished with the already frozen settings.
            for group in ['P0', 'P1', 'P2']:
                if read_json(root / group / 'summary.json').get('state') == 'done':
                    continue
                write_json(state_path, {'state': 'evaluating', 'group': group, 'pid': os.getpid()})
                child(root, 'parse22_supervised.evaluate', ['--root', str(root), '--group', group,
                      '--source-manifest', str(root / 'source_manifest.json')], group + '_evaluation.log')
            child(root, 'parse22_supervised.compare', ['--root', str(root)], 'comparison.log')
            write_json(state_path, {'state': 'done', 'pid': os.getpid(), 'finished_unix': time.time()})
        except Exception as error:
            old = read_json(state_path) if state_path.exists() else {}
            write_json(state_path, {**old, 'state': 'failed', 'mode': args.mode, 'error': str(error), 'failed_unix': time.time()})
            raise


if __name__ == '__main__':
    main()
