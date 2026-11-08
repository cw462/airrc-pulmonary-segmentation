"""Execution order only; independent one-GPU models keep the frozen budget."""
from pathlib import Path

from .preprocess import read_json


def require_completed_baseline(root, group):
    if group not in ('P0', 'P1', 'P2'):
        raise ValueError('Unknown supervised group.')
    if group == 'P0':
        return
    path = Path(root) / 'P0/training_complete.json'
    if not path.exists() or read_json(path).get('optimizer_steps') != 100000:
        raise ValueError('P1/P2 may start only after P0 completes100000 optimizer updates.')


def training_arguments(root, group):
    root = Path(root)
    require_completed_baseline(root, group)
    arguments = ['--root', str(root), '--group', group, '--mode', 'full']
    folder = root / group / 'results'
    latest = list(folder.glob('**/checkpoint_latest.pth')) if folder.exists() else []
    if len(latest) > 1:
        raise RuntimeError('Ambiguous latest checkpoint paths; do not guess.')
    if latest:
        arguments += ['--resume', str(latest[0])]
    return arguments


def training_finished(root, group):
    path = Path(root) / group / 'training_complete.json'
    return path.exists() and read_json(path).get('optimizer_steps') == 100000
