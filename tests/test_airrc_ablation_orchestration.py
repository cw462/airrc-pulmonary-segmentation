import importlib.util
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys
from unittest.mock import patch

from airrc_ablation.baseline import import_baseline
from airrc_ablation.common import read_json, write_json
from airrc_ablation.compare import paired_difference


class AirRCOrchestrationTests(unittest.TestCase):
    def test_existing_m0_numbers_and_all_76_patient_ids_are_retained(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/'patient_splits.json', read_json(repo/'data/splits/airrc_seed2026.json'))
            summary = import_baseline(root, repo/'reports/airrc_m0')
            self.assertEqual(summary['patient_counts'], {'val': 38, 'test': 38})
            for part in ('val', 'test'):
                self.assertEqual(summary['av_scores'][part]['av_mcs']['n_valid'], 38)
                for name in ('pulmonary_artery', 'pulmonary_vein', 'airway_lumen'):
                    self.assertEqual(summary['scores'][part][name]['dice']['n_valid'], 38)

    def test_paired_bootstrap_does_not_zero_fill_nan_or_reorder_pairing(self):
        result = paired_difference({'a': 1., 'b': 2., 'c': None}, {'c': 5., 'b': 2.1, 'a': 1.1})
        self.assertEqual(result['n_pairs'], 2)
        self.assertEqual(result['excluded_cases'], ['c'])
        self.assertAlmostEqual(result['mean_difference'], .1)
        self.assertAlmostEqual(result['CI95_low'], .1)
        self.assertAlmostEqual(result['CI95_high'], .1)
        with self.assertRaises(ValueError):
            paired_difference({'a': 1}, {'b': 1})

    @unittest.skipUnless(importlib.util.find_spec('fcntl'), 'Linux server process queue')
    def test_resume_uses_only_own_group_and_requires_checkpoint_for_existing_progress(self):
        from airrc_ablation.pipeline import training_args
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = 'results/Dataset926_AirRC/AirRCVariantTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_latest.pth'
            p1 = root/'P1'/relative; p1.parent.mkdir(parents=True); p1.touch()
            p2 = root/'P2'; p2.mkdir()
            self.assertEqual(training_args(root, 'P2', 'full'), ['--root', str(root), '--group', 'P2', '--mode', 'full'])
            self.assertEqual(training_args(root, 'P1', 'full')[-2:], ['--resume', str(p1)])
            (p2/'training_history.csv').write_text('epoch,iterations\n1,250\n')
            with self.assertRaises(ValueError):
                training_args(root, 'P2', 'full')

    @unittest.skipUnless(importlib.util.find_spec('fcntl'), 'Linux server process queue')
    def test_failed_peer_does_not_cancel_healthy_independent_training(self):
        from airrc_ablation.pipeline import run_parallel
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processes = []

            def launch(folder, module, arguments, log_name, label, gpu):
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpu
                if label == 'P1':
                    code = 'import time;time.sleep(.1);raise SystemExit(1)'
                else:
                    code = ("import time,os,json;from pathlib import Path;time.sleep(.4);"
                            "p=Path({!r})/'P2';p.mkdir();"
                            "(p/'training_complete.json').write_text(json.dumps({{'training_iterations':100000,'group':'P2'}}));"
                            "(p/'gpu.txt').write_text(os.environ['CUDA_VISIBLE_DEVICES'])").format(str(root))
                child = subprocess.Popen([sys.executable, '-c', code], env=env)
                processes.append(child)
                return child

            with patch('airrc_ablation.pipeline.launch', launch), patch('airrc_ablation.pipeline.gpu_available', return_value=True):
                with self.assertRaises(RuntimeError):
                    run_parallel(root, {'P1': 'gpu-A', 'P2': 'gpu-B'}, 'full', poll_seconds=.01)
            self.assertEqual((root/'P2/gpu.txt').read_text(), 'gpu-B')
            self.assertEqual([child.returncode for child in processes], [1, 0])
            self.assertIn('P1', read_json(root/'pipeline_state.json')['failures'])

    @unittest.skipUnless(importlib.util.find_spec('fcntl'), 'Linux server process queue')
    def test_user_paused_group_never_queries_or_uses_reserved_gpu(self):
        from airrc_ablation.pipeline import run_parallel, training_args
        class EndObservation(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/'P1/training_complete.json', {'training_iterations': 100000, 'group': 'P1'})
            write_json(root/'user_control.json', {'paused_groups': {'P2': {'automatic_resume': False}}})
            with self.assertRaisesRegex(RuntimeError, 'paused_by_user'):
                training_args(root, 'P2', 'full')
            with patch('airrc_ablation.pipeline.gpu_available', side_effect=AssertionError('Reserved GPU must not be queried for dispatch')), \
                 patch('airrc_ablation.pipeline.launch', side_effect=AssertionError('Paused group must not launch')), \
                 patch('airrc_ablation.pipeline.time.sleep', side_effect=EndObservation):
                with self.assertRaises(EndObservation):
                    run_parallel(root, {'P1': 'gpu-A', 'P2': 'gpu-B'}, 'full')
            state = read_json(root/'pipeline_state.json')
            self.assertEqual(state['state'], 'waiting_user_resume')
            self.assertEqual(state['paused_groups'], ['P2'])
            self.assertEqual(state['failures'], {})

    @unittest.skipUnless(importlib.util.find_spec('fcntl'), 'Linux server process queue')
    def test_adoption_observes_existing_pid_without_restarting_or_signaling_it(self):
        from airrc_ablation.pipeline import wait_for_existing_training
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = ("import time,json;from pathlib import Path;time.sleep(.2);"
                    "p=Path({!r})/'P1';p.mkdir();"
                    "(p/'training_complete.json').write_text(json.dumps({{'training_iterations':100000,'group':'P1'}}))").format(str(root))
            child = subprocess.Popen([sys.executable, '-c', code, 'airrc_ablation.train', str(root), '--group', 'P1'])
            ticks = (Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]
            with patch('airrc_ablation.pipeline.launch', side_effect=AssertionError('Adoption must not launch')), \
                 patch('airrc_ablation.pipeline.os.kill', side_effect=AssertionError('Adoption must not signal')):
                wait_for_existing_training(root, 'P1', child.pid, ticks, poll_seconds=.01)
            self.assertEqual(child.wait(), 0)


if __name__ == '__main__':
    unittest.main()
