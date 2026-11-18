"""Local regression tests; no GPU, credentials, training data, or W&B connection."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import train_flip_pillow_lora as launcher
from import_loss_to_wandb import read_history, write_history


class ReportingTest(unittest.TestCase):
    def test_launch_command(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 6), __version__='mock')
        env = {'CUDA_VISIBLE_DEVICES': '1,2,3,4,5,6', 'NUM_GPUS': '6', 'WANDB_PROJECT': 'pillow-test'}
        for flags, backend in (([], 'none'), (['--report-to', 'wandb'], 'wandb')):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                argv = ['launcher', 'train', '--data', tmp, '--output', f'{tmp}/new-run', *flags]
                sdk = Mock() if flags else None
                with (patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', argv),
                      patch.dict(sys.modules, {'torch': torch, 'wandb': sdk}),
                      patch.object(launcher, 'check_data'), patch.object(launcher, 'check_weights'),
                      patch.object(os, 'chdir'), patch.object(os, 'execv') as launch):
                    launcher.main()
                    args = launch.call_args.args[1]
                    self.assertEqual([s for s in args if s.startswith('report_to=')], [f'report_to={backend}'])
                    for s in ('wandb_project="pillow-test"', '--nproc-per-node=6', 'logging_steps=10',
                              'max_steps=1000', 'save_steps=200', 'fps.yam=50', 'upload_checkpoints=false'):
                        self.assertIn(s, args)
                    if sdk is not None:
                        sdk.init.assert_not_called()  # only the HF callback should create the run
                        self.assertEqual(os.environ['WANDB_LOG_MODEL'], 'false')

    def test_disabled_env_rejected_for_explicit_reporting(self):
        for env in ({'WANDB_DISABLED': 'true'}, {'WANDB_MODE': 'disabled'}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, '环境禁用了上报'):
                    launcher.reporting_options('wandb')

    def test_history_matches_live_metric_names_and_merges_sparse_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'loss_log.jsonl'
            path.write_text('{"step":0,"action_loss_avg":0.12}\n{"step":1,"loss":0.17,"unrelated":"secret"}\n'
                            '{"step":0,"dynamics_loss_avg":0.056}\n{"step":1,"learning_rate":0}\n'
                            '{"step":10,"loss":0.1328}\n', encoding='utf-8')
            run = Mock()
            write_history(run, read_history(path))
            self.assertEqual(run.log.call_args_list, [
                call({'train/global_step': 0, 'train/action_loss_avg': .12, 'train/dynamics_loss_avg': .056}),
                call({'train/global_step': 1, 'train/loss': .17, 'train/learning_rate': 0}),
                call({'train/global_step': 10, 'train/loss': .1328}),
            ])
            run.define_metric.assert_any_call('train/*', step_metric='train/global_step')

    def test_reject_invalid_completed_log_before_upload(self):
        for text in ('', '{"step":0,"learning_rate":0}', '{"step":1,"loss":NaN}', '{"step":true,"loss":0}', '{'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'loss_log.jsonl'
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_history(path)


if __name__ == '__main__':
    unittest.main()
