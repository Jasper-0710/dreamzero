"""Exercise the real callback lifecycle with both old and new fake SDK behavior."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ws.src.training.swanlab_reporting import DreamZeroSwanLabCallback, create_swanlab_trainer


class SwanLabCallbackTest(unittest.TestCase):
    def args(self):
        return SimpleNamespace(output_dir='/tmp/dz-swanlab-test', run_name='pillow-test',
                               max_steps=4000, learning_rate=1e-5, per_device_train_batch_size=1,
                               gradient_accumulation_steps=1, world_size=6, report_to=[],
                               hub_token='must-not-log', unused_secret='must-not-log')

    def state(self, rank=0, step=10):
        return SimpleNamespace(is_world_process_zero=rank == 0, global_step=step)

    def test_no_active_run_regression_both_sdk_semantics(self):
        for new_sdk in (False, True):
            with self.subTest(new_sdk=new_sdk):
                sdk = Mock()
                sdk.get_run.return_value = None
                if new_sdk:
                    sdk.get_run.side_effect = RuntimeError('No active Run. Call swanlab.init() first.')
                callback = DreamZeroSwanLabCallback()
                args, state = self.args(), self.state()
                env = {'SWANLAB_PROJECT': 'pillow-project', 'SWANLAB_MODE': 'online',
                       'SWANLAB_WORKSPACE': 'pillow-team', 'SWANLAB_LOG_MODEL': 'true',
                       'WANDB_MODE': 'online', 'WANDB_API_KEY': 'must-not-log'}
                with patch.dict(sys.modules, {'swanlab': sdk}), patch.dict(os.environ, env, clear=True):
                    callback.on_train_begin(args, state, None)
                    callback.on_train_begin(args, state, None)  # Idempotent initialization.
                    logs = {'loss': .1, 'grad_norm': .5, 'learning_rate': 1e-5,
                            'action_loss_avg': .07, 'dynamics_loss_avg': .03, 'epoch': 0.,
                            'eval_loss': .2, 'test_loss': .3, 'not_scalar': 'ignore', 'nan': float('nan')}
                    callback.on_log(args, state, None, logs=logs)
                    # The original per-step JSONL callback still needs the unchanged logs.
                    self.assertEqual(logs['not_scalar'], 'ignore')
                    callback.on_save(args, state, None)  # Inherited no-op; no checkpoint upload.
                    callback.on_train_end(args, state, None)
                    callback.on_train_end(args, state, None)  # One finish per run.
                sdk.get_run.assert_not_called()
                sdk.init.assert_called_once()
                options = sdk.init.call_args.kwargs
                self.assertEqual(options['project'], 'pillow-project')
                self.assertEqual(options['workspace'], 'pillow-team')
                self.assertEqual(options['experiment_name'], 'pillow-test')
                self.assertEqual(options['logdir'], '/tmp/dz-swanlab-test/swanlog')
                self.assertEqual(options['config']['effective_batch_size'], 6)
                self.assertNotIn('must-not-log', repr(options))
                sdk.log.assert_called_once_with({
                    'train/loss': .1, 'train/grad_norm': .5, 'train/learning_rate': 1e-5,
                    'train/action_loss_avg': .07, 'train/dynamics_loss_avg': .03,
                    'train/epoch': 0., 'eval/loss': .2, 'test/loss': .3, 'train/global_step': 10,
                }, step=10)
                sdk.finish.assert_called_once()
                self.assertEqual([c[0] for c in sdk.method_calls], ['init', 'log', 'finish'])

    def test_six_ranks_create_one_run_total(self):
        sdk = Mock()
        sdk.get_run.side_effect = RuntimeError('No active Run. Call swanlab.init() first.')
        with patch.dict(sys.modules, {'swanlab': sdk}), patch.dict(os.environ, {}, clear=True):
            for rank in range(6):
                callback, args, state = DreamZeroSwanLabCallback(), self.args(), self.state(rank)
                callback.on_train_begin(args, state, None)
                callback.on_log(args, state, None, logs={'loss': .1})
                callback.on_train_end(args, state, None)
        sdk.init.assert_called_once()
        sdk.log.assert_called_once()
        sdk.finish.assert_called_once()
        sdk.get_run.assert_not_called()

    def test_offline_defaults_and_log_without_begin(self):
        sdk = Mock()
        args, state = self.args(), self.state(step=0)
        args.run_name = None
        callback = DreamZeroSwanLabCallback()
        with patch.dict(sys.modules, {'swanlab': sdk}), patch.dict(os.environ, {'SWANLAB_MODE': 'offline'}, clear=True):
            callback.on_log(args, state, None, logs={})
            sdk.init.assert_not_called()
            callback.on_log(args, state, None, logs={'loss': .1})
            callback.on_log(args, state, None, logs={'action_loss_avg': .05})  # Same step, sparse logs.
            callback.on_train_end(args, state, None)
        options = sdk.init.call_args.kwargs
        self.assertEqual(options['mode'], 'offline')
        self.assertEqual(options['project'], 'dreamzero-flip-pillow')
        self.assertEqual(options['experiment_name'], Path(args.output_dir).name)
        self.assertNotIn('workspace', options)
        sdk.init.assert_called_once()
        self.assertEqual(sdk.log.call_count, 2)

    def test_failed_initialization_does_not_log_or_finish(self):
        sdk = Mock()
        sdk.init.side_effect = RuntimeError('mock authentication failure')
        callback, args, state = DreamZeroSwanLabCallback(), self.args(), self.state()
        with patch.dict(sys.modules, {'swanlab': sdk}), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'authentication failure'):
                callback.on_train_begin(args, state, None)
            callback.on_train_end(args, state, None)
        sdk.log.assert_not_called()
        sdk.finish.assert_not_called()

    def test_factory_keeps_official_trainer_kwargs_and_callbacks(self):
        trainer = Mock()
        cls = Mock(return_value=trainer)
        kwargs = dict(args=self.args(), model=object(), data_collator=object(),
                      callbacks=[object()], output_dir='/tmp/test', compute_dtype='bfloat16')
        with patch.dict(sys.modules, {'groot.vla.experiment': SimpleNamespace(VLATrainer=cls)}):
            result = create_swanlab_trainer(**kwargs)
        self.assertIs(result, trainer)
        cls.assert_called_once_with(**kwargs)
        trainer.add_callback.assert_called_once()
        self.assertIsInstance(trainer.add_callback.call_args.args[0], DreamZeroSwanLabCallback)

    def test_factory_rejects_native_integrations_to_prevent_duplicate_runs(self):
        args = self.args()
        for backend in (['swanlab'], ['wandb'], ['all']):
            args.report_to = backend
            with self.assertRaisesRegex(ValueError, 'report_to=none'):
                create_swanlab_trainer(args=args)


if __name__ == '__main__':
    unittest.main()
