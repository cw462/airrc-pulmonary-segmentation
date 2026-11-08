"""Synthetic scheduling tests; no model training or clinical scores are produced."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from parse22_supervised.scheduling import require_completed_baseline, training_arguments


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for group in ('P0', 'P1', 'P2'):
            (self.root / group).mkdir()
        self.complete('P0')

    def tearDown(self):
        self.temp.cleanup()

    def complete(self, group):
        (self.root / group / 'training_complete.json').write_text(json.dumps({'optimizer_steps': 100000}))

    def test_p1_and_p2_both_depend_only_on_completed_p0(self):
        require_completed_baseline(self.root, 'P1')
        require_completed_baseline(self.root, 'P2')
        (self.root / 'P0/training_complete.json').unlink()
        for group in ('P1', 'P2'):
            with self.assertRaises(ValueError):
                require_completed_baseline(self.root, group)

    def test_resume_uses_only_own_checkpoint(self):
        for group in ('P0', 'P1', 'P2'):
            path = self.root / group / 'results/fold_0/checkpoint_latest.pth'
            path.parent.mkdir(parents=True)
            path.write_text('synthetic fixture; not a model')
        for group in ('P1', 'P2'):
            arguments = training_arguments(self.root, group)
            self.assertEqual(arguments[-1], str(self.root / group / 'results/fold_0/checkpoint_latest.pth'))

    def test_frozen_matrix_is_not_rewritten_by_concurrent_initialization(self):
        from concurrent.futures import ThreadPoolExecutor
        from parse22_supervised import config
        repo = Path(__file__).resolve().parent.parent
        (self.root / 'shared').mkdir()
        shutil.copyfile(repo / 'reports/parse22_supervised/preprocessing_config.json', self.root / 'shared/preprocessing_config.json')
        shutil.copyfile(repo / 'data/splits/parse22_seed2026.json', self.root / 'parse22_seed2026.json')
        config.make_config(self.root, 'P0', 'full')
        matrix = self.root / 'experiment_matrix.json'
        original = matrix.read_bytes()
        with mock.patch.object(config, 'write_json', side_effect=AssertionError('Frozen metadata must be read-only')):
            with ThreadPoolExecutor(2) as pool:
                results = list(pool.map(lambda group: config.make_config(self.root, group, 'full'), ['P1', 'P2']))
        self.assertEqual([row['group'] for row in results], ['P1', 'P2'])
        self.assertEqual(matrix.read_bytes(), original)

    @unittest.skipUnless(os.name == 'posix', 'Server dispatcher uses POSIX process locks')
    def test_workers_overlap_with_distinct_gpu_environments(self):
        from parse22_supervised import pipeline
        children = []
        def launch(root, module, arguments, log, label, gpu):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            # Both synthetic workers must enter before either can finish.
            script = """
import json,os,sys,time
from pathlib import Path
r=Path(sys.argv[1]); g=sys.argv[2]; other='P2' if g=='P1' else 'P1'
(r/g/'started').write_text(os.environ['CUDA_VISIBLE_DEVICES'])
end=time.monotonic()+5
while not (r/other/'started').exists():
 if time.monotonic()>end:sys.exit(8)
 time.sleep(.02)
(r/g/'training_complete.json').write_text(json.dumps({'optimizer_steps':100000}))
"""
            p = subprocess.Popen([sys.executable, '-c', script, str(root), label], env=env)
            children.append(p)
            return p
        try:
            with mock.patch.object(pipeline, 'launch', side_effect=launch), mock.patch.object(pipeline, 'gpu_available', return_value=True):
                pipeline.parallel_training(self.root, {'P1': 'SYNTHETIC_GPU_0', 'P2': 'SYNTHETIC_GPU_1'}, poll_seconds=.02)
            self.assertEqual((self.root / 'P1/started').read_text(), 'SYNTHETIC_GPU_0')
            self.assertEqual((self.root / 'P2/started').read_text(), 'SYNTHETIC_GPU_1')
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
                p.wait()

    @unittest.skipUnless(os.name == 'posix', 'Server dispatcher uses POSIX process locks')
    def test_adoption_waits_without_terminating_existing_p0(self):
        import time
        from parse22_supervised import pipeline
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(.3)',
                                    'parse22_supervised.train', str(self.root), '--group', 'P0'])
        try:
            started = time.monotonic()
            ticks = (Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
            pipeline.wait_for_adopted_p0(self.root, process.pid, ticks, poll_seconds=.02)
            self.assertGreater(time.monotonic() - started, .2)
            self.assertEqual(process.wait(), 0)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()

    @unittest.skipUnless(os.name == 'posix', 'Server dispatcher uses POSIX process locks')
    def test_failed_peer_does_not_cancel_healthy_worker(self):
        from parse22_supervised import pipeline
        children = []
        def launch(root, module, arguments, log, label, gpu):
            script = """
import json,sys,time
from pathlib import Path
r=Path(sys.argv[1]);g=sys.argv[2]
if g=='P1':sys.exit(7)
time.sleep(.15)
(r/g/'training_complete.json').write_text(json.dumps({'optimizer_steps':100000}))
"""
            p = subprocess.Popen([sys.executable, '-c', script, str(root), label])
            children.append(p)
            return p
        try:
            with mock.patch.object(pipeline, 'launch', side_effect=launch), mock.patch.object(pipeline, 'gpu_available', return_value=True):
                with self.assertRaises(RuntimeError):
                    pipeline.parallel_training(self.root, {'P1': 'SYNTHETIC_GPU_0', 'P2': 'SYNTHETIC_GPU_1'}, poll_seconds=.02)
            self.assertTrue((self.root / 'P2/training_complete.json').exists())
            self.assertEqual(children[1].returncode, 0)
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
                p.wait()


if __name__ == '__main__':
    unittest.main()
