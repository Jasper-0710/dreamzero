"""CPU-only SwanLab tests: fake SDK, no credentials, network, model or GPU."""

import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ws.src.monitoring import check_swanlab
from ws.src.training import train_flip_pillow_lora as train
from ws.src.training import launch_pillow_training as launch


class SwanLabReportingTest(unittest.TestCase):
    def args(self, *flags):
        return launch.parser().parse_args(['train', '--train_dir', 'data/flip_150', '--gpus', '1,2,3,4,5,6', *flags])

    def test_backend_flags_aliases_and_environment(self):
        args = self.args('--report_to', 'swanlab', '--swanlab_project', 'pillow-test',
                         '--swanlab_mode', 'online', '--swanlab_workspace', 'team')
        launch.validate_args(args)
        original = {'WANDB_MODE': 'offline', 'WANDB_DISABLED': 'true', 'SWANLAB_MODE': 'disabled',
                    'SWANLAB_LOG_MODEL': 'true', 'LOAD_TRT_ENGINE': 'stale.engine'}
        with patch.dict(os.environ, original, clear=True):
            env = launch.gpu_environment(args)
            self.assertEqual(dict(os.environ), original)  # Parent terminal is not modified.
        self.assertEqual(env['SWANLAB_MODE'], 'online')
        self.assertEqual(env['SWANLAB_PROJECT'], 'pillow-test')
        self.assertEqual(env['SWANLAB_WORKSPACE'], 'team')
        self.assertEqual(env['NUM_GPUS'], '6')
        self.assertNotIn('SWANLAB_LOG_MODEL', env)
        self.assertNotIn('LOAD_TRT_ENGINE', env)

    def test_swanlab_default_and_offline_modes(self):
        with patch.dict(os.environ, {'SWANLAB_MODE': 'disabled'}, clear=True):
            env = launch.gpu_environment(self.args('--report-to', 'swanlab'))
            self.assertEqual(env['SWANLAB_MODE'], 'online')
            self.assertEqual(env['SWANLAB_PROJECT'], 'dreamzero-flip-pillow')
            env = launch.gpu_environment(self.args('--report-to', 'swanlab', '--swanlab-mode', 'offline'))
            self.assertEqual(env['SWANLAB_MODE'], 'offline')

    def test_reject_wrong_backend_or_empty_names(self):
        for flags in (['--swanlab-mode', 'online'], ['--swanlab-project', 'p'],
                      ['--swanlab-workspace', 'team'], ['--report-to', 'wandb', '--swanlab-mode', 'offline'],
                      ['--report-to', 'swanlab', '--wandb-mode', 'online'],
                      ['--report-to', 'swanlab', '--swanlab-project', ' '],
                      ['--report-to', 'swanlab', '--swanlab-workspace', '']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                launch.validate_args(self.args(*flags))

    def test_explicit_reporting_does_not_init_sdk_or_upload_model(self):
        for mode in ('online', 'offline'):
            sdk = Mock()
            env = {'SWANLAB_MODE': mode, 'SWANLAB_PROJECT': 'pillow-test', 'SWANLAB_LOG_MODEL': 'false',
                   'WANDB_MODE': 'online'}
            with (self.subTest(mode=mode), patch.dict(os.environ, env, clear=True),
                  patch.dict(sys.modules, {'swanlab': sdk})):
                options = train.reporting_options('swanlab', output=Path('/tmp/pillow-test'))
                self.assertEqual(options, ['report_to=none', 'wandb_project="dreamzero"',
                    'trainer._target_=ws.src.training.swanlab_reporting.create_swanlab_trainer'])
                self.assertEqual(os.environ['SWANLAB_PROJ_NAME'], 'pillow-test')
                self.assertEqual(os.environ['SWANLAB_LOGDIR'], '/tmp/pillow-test/swanlog')
                self.assertEqual(os.environ['SWANLAB_LOG_DIR'], '/tmp/pillow-test/swanlog')
                self.assertNotIn('SWANLAB_LOG_MODEL', os.environ)
                sdk.init.assert_not_called()

    def test_missing_sdk_fails_before_gpu(self):
        with patch.dict(sys.modules, {'swanlab': None}), self.assertRaisesRegex(ValueError, 'pip install swanlab'):
            train.reporting_options('swanlab')

    def test_low_level_launch_selects_only_swanlab(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 6), __version__='mock')
        sdk = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            env = {'CUDA_VISIBLE_DEVICES': '1,2,3,4,5,6', 'NUM_GPUS': '6', 'SWANLAB_PROJECT': 'pillow-test'}
            argv = ['train', 'train', '--data', tmp, '--output', f'{tmp}/new-run', '--report-to', 'swanlab']
            with (patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', argv),
                  patch.dict(sys.modules, {'torch': torch, 'swanlab': sdk}),
                  patch.object(train, 'check_data'),
                  patch.object(train, 'check_weights'), patch.object(os, 'chdir'), patch.object(os, 'execv') as run):
                train.main()
                options = run.call_args.args[1]
                self.assertEqual([s for s in options if s.startswith('report_to=')], ['report_to=none'])
                self.assertIn('trainer._target_=ws.src.training.swanlab_reporting.create_swanlab_trainer', options)
                self.assertIn('--nproc-per-node=6', options)
                self.assertIn('upload_checkpoints=false', options)
                self.assertEqual(os.environ['SWANLAB_LOGDIR'], f'{tmp}/new-run/swanlog')
                sdk.init.assert_not_called()

    def test_connection_check_is_explicit_and_finishes(self):
        for mode in ('online', 'offline'):
            sdk = Mock()
            with (self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp,
                  patch.dict(os.environ, {'SWANLAB_LOG_MODEL': 'true'}, clear=True),
                  patch.dict(sys.modules, {'swanlab': sdk}),
                  redirect_stdout(StringIO())):
                check_swanlab.main(['--project', 'pillow-test', '--workspace', 'team', '--mode', mode, '--output', tmp])
                self.assertEqual(sdk.init.call_args.kwargs['project'], 'pillow-test')
                self.assertEqual(sdk.init.call_args.kwargs['workspace'], 'team')
                self.assertEqual(sdk.init.call_args.kwargs['mode'], mode)
                self.assertEqual(sdk.init.call_args.kwargs['logdir'], f'{tmp}/swanlog')
                sdk.log.assert_called_once_with({'train/connection_test': 1., 'train/global_step': 0}, step=0)
                sdk.get_run.assert_not_called()
                sdk.finish.assert_called_once()
                self.assertNotIn('torch', sdk.init.call_args.kwargs)
                self.assertNotIn('SWANLAB_LOG_MODEL', os.environ)
                self.assertEqual(os.environ['SWANLAB_ON_ONLINE_ERROR'], 'raise')

    def test_connection_check_cleans_up_after_log_error(self):
        sdk = Mock()
        sdk.log.side_effect = RuntimeError('mock network error')
        with (patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {'swanlab': sdk}),
              redirect_stdout(StringIO()),
              self.assertRaisesRegex(RuntimeError, 'mock network error')):
            check_swanlab.main([])
        sdk.finish.assert_called_once()

    def test_swanlab_dry_run_does_not_connect_or_start_workers(self):
        with (patch.dict(os.environ, {}, clear=True), patch.object(launch, 'validate_training_view') as view,
              patch.object(launch.subprocess, 'run') as run, redirect_stdout(StringIO()) as output):
            launch.main(['--mode', 'train', '--train_dir', 'data/flip_150', '--gpus', '1,2,3,4,5,6',
                         '--report_to', 'swanlab', '--swanlab_project', 'pillow-test', '--dry_run'])
            view.assert_not_called()
            run.assert_not_called()
            self.assertIn('--report-to swanlab', output.getvalue())


if __name__ == '__main__':
    unittest.main()
