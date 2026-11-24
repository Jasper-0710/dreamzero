"""CPU-only launcher tests: fake metadata/files, no model, CUDA, conversion or training."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.common.pillow import TASK, VIEWS
from ws.src.training import launch_pillow_training as launch


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def write_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in data), encoding='utf-8')


def raw_fixture(root, count=4, task=TASK):
    write_json(root / 'meta/info.json', {'fps': 50, 'total_episodes': count})
    write_jsonl(root / 'meta/episodes.jsonl', [{'episode_index': i} for i in range(count)])
    write_jsonl(root / 'meta/tasks.jsonl', [{'task_index': 0, 'task': task}])


def converted_fixture(root, source_name, origins):
    # The launcher only checks structure/nonempty files. These are NOT decodable datasets.
    features = {k: {'shape': [14], 'dtype': 'float32'} for k in ('observation.state', 'action')}
    features.update({k: {'dtype': 'video'} for k in VIEWS})
    info = dict(fps=50, features=features, total_episodes=len(origins), total_frames=30 * len(origins),
                total_videos=3 * len(origins), chunks_size=1000,
                data_path='data/episode_{episode_index:06d}.parquet',
                video_path='videos/{video_key}/episode_{episode_index:06d}.mp4')
    write_json(root / 'meta/info.json', info)
    write_json(root / 'meta/conversion.json', {'source_dataset': source_name, 'source_fps': 50})
    write_json(root / 'meta/embodiment.json', {'embodiment_tag': 'yam'})
    for name in ('modality', 'stats', 'relative_stats_dreamzero'):
        write_json(root / f'meta/{name}.json', {})
    write_jsonl(root / 'meta/tasks.jsonl', [{'task_index': 0, 'task': TASK}])
    write_jsonl(root / 'meta/episodes.jsonl', [
        {'episode_index': row['episode_index'], 'length': 30, 'tasks': [TASK]} for row in origins])
    write_jsonl(root / 'meta/source_episodes.jsonl', origins)
    for row in origins:
        fields = dict(episode_index=row['episode_index'])
        for pattern in [info['data_path'], *[info['video_path'].format(video_key=k, **fields) for k in VIEWS]]:
            path = root / pattern.format(**fields)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'test placeholder, not real parquet/video')


def split_fixture(paths, origins, count=4, holdout=1):
    held = list(range(count - holdout, count))
    converted_fixture(paths.train, paths.raw.name, [r for r in origins if r['source_episode_index'] not in held])
    write_json(paths.train / 'meta/holdout.json', dict(full_dataset=str(paths.converted),
               train_source_ids=list(range(count - holdout)), held_out_source_ids=held,
               held_out_episode_ids_in_full_dataset=[r['episode_index'] for r in origins if r['source_episode_index'] in held]))


class LauncherTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.patch_root = patch.object(launch, 'ROOT', self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.origins = [{'episode_index': i, 'source_episode_index': i // 2,
                         'source_segment_index': i % 2} for i in range(8)]

    def args(self, mode='train', *flags):
        return launch.parser().parse_args([mode, '--dataset', 'data/flip_150', '--holdout', '1', *flags])

    def dataset(self, *, split=True, task=TASK):
        paths = launch.paths_for(self.args())
        raw_fixture(paths.raw, task=task)
        converted_fixture(paths.converted, paths.raw.name, self.origins)
        if split:
            split_fixture(paths, self.origins)
        return paths

    def run_main(self, mode='train', *flags):
        with redirect_stdout(StringIO()):
            launch.main([mode, '--dataset', 'data/flip_150', '--holdout', '1', *flags])

    def test_dataset_specific_paths_and_smoke_isolation(self):
        train = launch.paths_for(self.args())
        smoke = launch.paths_for(self.args('smoke'))
        self.assertEqual(train.raw, self.root / 'data/flip_150')
        self.assertEqual(train.converted, self.root / 'data/flip_150_dz')
        self.assertEqual(train.train, self.root / 'data/flip_150_dz_train')
        self.assertEqual(train.output, self.root / 'checkpoints/flip_150_lora_01')
        self.assertEqual(smoke.output.name, 'flip_150_lora_01_smoke')
        other = launch.paths_for(self.args('train', '--dataset', 'data/新数据 集'))
        self.assertEqual(other.train, self.root / 'data/新数据 集_dz_train')
        self.assertNotEqual(other.output, train.output)

    def test_paths_are_repo_relative_and_customizable(self):
        with patch('os.getcwd', return_value='/tmp'):
            paths = launch.paths_for(self.args('train', '--converted', 'custom/full', '--train-data', 'custom/train',
                                                '--output', str(self.root / 'runs/one')))
        self.assertEqual(paths.converted, self.root / 'custom/full')
        self.assertEqual(paths.train, self.root / 'custom/train')
        self.assertEqual(paths.output, self.root / 'runs/one')

    def test_reject_overlapping_paths_and_path_run_names(self):
        for flags in (['--converted', 'data/flip_150'], ['--train-data', 'data/flip_150/train'],
                      ['--output', 'data'], ['--run-name', '../old'], ['--run-name', '..']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                launch.paths_for(self.args('train', *flags))
        # The default output must resolve symlinks too, not just explicit --output.
        (self.root / 'data').mkdir()
        (self.root / 'checkpoints').symlink_to(self.root / 'data', target_is_directory=True)
        with self.assertRaises(ValueError):
            launch.paths_for(self.args('train', '--run-name', 'flip_150'))

    def test_step_defaults(self):
        for mode, flags, expected in [('train', [], (2000, 200)), ('smoke', [], (10, 10)),
                                      ('train', ['--steps', '50'], (50, 50)),
                                      ('smoke', ['--steps', '3'], (3, 3))]:
            with self.subTest(mode=mode, flags=flags):
                self.assertEqual(launch.validate_args(self.args(mode, *flags)), expected)

    def test_invalid_flags(self):
        for flags in (['--steps', '0'], ['--save-steps', '2001'], ['--holdout', '0'], ['--batch-size', '0'],
                      ['--task-text', 'another task'], ['--reuse-converted'], ['--wandb-mode', 'offline']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                launch.validate_args(self.args('train', *flags))

    def test_gpu_environment_explicit_and_stale_count(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'NUM_GPUS': '8'}, clear=True):
            env = launch.gpu_environment(self.args('train', '--gpus', '1, 2,3,4,5,6'))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1,2,3,4,5,6')
            self.assertEqual(env['NUM_GPUS'], '6')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')
            env = launch.gpu_environment(self.args('train', '--gpus', '1,2', '--num-gpus', '1'))
            self.assertEqual(env['NUM_GPUS'], '1')

    def test_gpu_inheritance_and_wandb_settings(self):
        original = {'CUDA_VISIBLE_DEVICES': 'GPU-abc,MIG-GPU-def/1/0', 'WANDB_MODE': 'offline', 'WANDB_PROJECT': 'inherited'}
        with patch.dict(os.environ, original, clear=True):
            env = launch.gpu_environment(self.args())
            self.assertEqual(env['NUM_GPUS'], '2')
            self.assertEqual(env['WANDB_PROJECT'], 'inherited')
            env = launch.gpu_environment(self.args('train', '--report-to', 'wandb', '--wandb-mode', 'online',
                                                   '--wandb-project', 'new-project'))
            self.assertEqual(env['WANDB_MODE'], 'online')
            self.assertEqual(env['WANDB_PROJECT'], 'new-project')

    def test_reject_invalid_gpu_masks_counts_and_ignore_old_trt(self):
        with patch.dict(os.environ, {}, clear=True):
            for mask in ('', '1-6', '1,,2', '1,1', '-1', 'all'):
                with self.subTest(mask=mask), self.assertRaises(ValueError):
                    launch.gpu_environment(self.args('train', '--gpus', mask))
            for count in ('0', '3'):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    launch.gpu_environment(self.args('train', '--gpus', '1,2', '--num-gpus', count))
            with self.assertRaises(ValueError):
                launch.gpu_environment(self.args())
        with patch.dict(os.environ, {'LOAD_TRT_ENGINE': 'old.engine', 'WANDB_DISABLED': 'true'}, clear=True):
            env = launch.gpu_environment(self.args('train', '--gpus', '1', '--report-to', 'wandb'))
            self.assertNotIn('LOAD_TRT_ENGINE', env)
            self.assertNotIn('WANDB_DISABLED', env)
            self.assertIn('LOAD_TRT_ENGINE', os.environ)  # Parent shell is not changed.

    def test_dry_run_does_not_read_data_launch_process_or_write(self):
        with (patch.dict(os.environ, {}, clear=True), patch.object(launch, 'source_ids') as raw,
              patch.object(launch, 'check_data') as check, patch.object(launch, 'validate_training_view') as view,
              patch.object(launch.subprocess, 'run') as run):
            self.run_main('prepare', '--task-text', TASK, '--dry-run')
            self.run_main('train', '--gpus', '1,2,3,4,5,6', '--dry-run')
        raw.assert_not_called()
        check.assert_not_called()
        view.assert_not_called()
        run.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_split_uses_whole_source_recordings_including_segments(self):
        paths = self.dataset()
        with redirect_stdout(StringIO()):
            launch.validate_training_view(paths, 1)
        # Holdout=1 here means source 3, which owns TWO converted episodes (6, 7).
        split = launch.read_json(paths.train / 'meta/holdout.json')
        self.assertEqual(split['held_out_episode_ids_in_full_dataset'], [6, 7])
        with self.assertRaises(ValueError):
            launch.validate_training_view(paths, 2)

    def test_reject_stale_conversion_source_or_incomplete_mapping(self):
        paths = self.dataset()
        write_json(paths.converted / 'meta/conversion.json', {'source_dataset': 'old_dataset', 'source_fps': 50})
        with self.assertRaises(ValueError):
            launch.validate_conversion(paths)
        write_json(paths.converted / 'meta/conversion.json', {'source_dataset': paths.raw.name, 'source_fps': 50})
        write_jsonl(paths.converted / 'meta/source_episodes.jsonl', self.origins[:-1])
        with self.assertRaises(ValueError):
            launch.validate_conversion(paths)

    def test_reject_wrong_split_directory_or_leaked_source_mapping(self):
        paths = self.dataset()
        split_path = paths.train / 'meta/holdout.json'
        split = launch.read_json(split_path)
        write_json(split_path, dict(split, full_dataset=str(self.root / 'another_dataset')))
        with self.assertRaises(ValueError):
            launch.validate_training_view(paths, 1)
        write_json(split_path, split)
        write_jsonl(paths.train / 'meta/source_episodes.jsonl', self.origins)
        with self.assertRaises(ValueError):
            launch.validate_training_view(paths, 1)

    def test_training_dispatch_uses_selected_data_output_and_environment(self):
        paths = self.dataset()
        with patch.dict(os.environ, {}, clear=True), patch.object(launch, 'run_command') as run:
            self.run_main('train', '--gpus', '1,2,3,4,5,6', '--steps', '1600', '--save-steps', '200',
                          '--batch-size', '2', '--report-to', 'wandb', '--wandb-mode', 'offline')
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:3], [sys.executable, str(self.root / 'ws/src/training/train_flip_pillow_lora.py'), 'train'])
        for flag, expected in [('--data', paths.train), ('--output', paths.output), ('--steps', 1600),
                               ('--save-steps', 200), ('--batch-size', 2), ('--report-to', 'wandb')]:
            self.assertEqual(cmd[cmd.index(flag) + 1], str(expected))
        self.assertEqual(run.call_args.kwargs['env']['NUM_GPUS'], '6')
        self.assertEqual(run.call_args.kwargs['env']['WANDB_MODE'], 'offline')

    def test_nonempty_checkpoint_is_not_overwritten(self):
        paths = self.dataset()
        paths.output.mkdir(parents=True)
        (paths.output / 'old-weights').write_text('do not touch')
        with (patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}, clear=True),
              patch.object(launch, 'run_command') as run):
            with self.assertRaisesRegex(ValueError, '不覆盖'):
                self.run_main('train')
            self.run_main('check')  # Read-only checks may inspect an existing run path.
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][2], 'check')
        self.assertEqual((paths.output / 'old-weights').read_text(), 'do not touch')

    def test_prepare_order_labels_and_no_gpu_requirement(self):
        paths = launch.paths_for(self.args('prepare'))
        raw_fixture(paths.raw, task='pick_cup')

        def fake_run(cmd, **kwargs):
            if cmd[1].endswith('convert_aloha_to_dreamzero.py'):
                converted_fixture(paths.converted, paths.raw.name, self.origins)
            elif cmd[1].endswith('prepare_pillow_holdout.py'):
                split_fixture(paths, self.origins)

        with patch.dict(os.environ, {}, clear=True), patch.object(launch, 'run_command', side_effect=fake_run) as run:
            self.run_main('prepare', '--task-text', TASK)
        calls = [call.args[0] for call in run.call_args_list]
        self.assertEqual([Path(c[1]).name for c in calls], ['convert_aloha_to_dreamzero.py',
                         *['check_aloha_alignment.py'] * 3, 'prepare_pillow_holdout.py'])
        self.assertEqual([int(c[c.index('--episode') + 1]) for c in calls[1:-1]], [0, 4, 7])
        self.assertEqual(calls[0][calls[0].index('--task-text') + 1], TASK)

    def test_prepare_correct_original_label_is_preserved(self):
        paths = launch.paths_for(self.args('prepare'))
        raw_fixture(paths.raw)
        with patch.object(launch, 'run_command', side_effect=RuntimeError('stop at converter')) as run:
            with self.assertRaisesRegex(RuntimeError, 'stop at converter'):
                self.run_main('prepare')
        self.assertEqual(run.call_args.args[0][-1], '--keep-task-text')

    def test_multiple_old_labels_are_inspected_by_converter_and_confirmation_is_forwarded(self):
        paths = launch.paths_for(self.args('prepare'))
        raw_fixture(paths.raw, task='pick_cup')
        write_jsonl(paths.raw / 'meta/tasks.jsonl', [
            {'task_index': 0, 'task': 'pick_cup'}, {'task_index': 1, 'task': 'flip_the_pillow'}])
        for confirm in ([], ['--confirm_same_task'], ['--confirm-same-task']):
            with (self.subTest(confirm=confirm),
                  patch.object(launch, 'run_command', side_effect=RuntimeError('stop at converter')) as run):
                with self.assertRaisesRegex(RuntimeError, 'stop at converter'):
                    self.run_main('prepare', '--task-text', TASK, *confirm)
                cmd = run.call_args.args[0]
                self.assertEqual(cmd[cmd.index('--task-text') + 1], TASK)
                self.assertEqual('--confirm-same-task' in cmd, bool(confirm))

    def test_confirm_same_task_requires_explicit_instruction_and_preparation_mode(self):
        for mode, flags in [('all', []), ('prepare', []), ('train', ['--task_text', TASK])]:
            with (self.subTest(mode=mode), patch.object(launch, 'run_command') as run,
                  self.assertRaisesRegex(ValueError, '--confirm_same_task')):
                self.run_main(mode, '--confirm_same_task', *flags)
            run.assert_not_called()

    def test_prepare_requires_explicit_relabel_and_valid_holdout(self):
        paths = launch.paths_for(self.args('prepare'))
        raw_fixture(paths.raw, task='pick_cup')
        with patch.object(launch, 'run_command') as run:
            with self.assertRaisesRegex(ValueError, '显式传'):
                self.run_main('prepare')
            with self.assertRaises(ValueError):
                self.run_main('prepare', '--holdout', '4', '--task-text', TASK)
        run.assert_not_called()

    def test_prepare_existing_outputs_require_explicit_reuse(self):
        paths = self.dataset(split=False)
        with patch.object(launch, 'run_command') as run:
            with self.assertRaisesRegex(ValueError, '转换目录已存在'):
                self.run_main('prepare')
        run.assert_not_called()

        def fake_run(cmd, **kwargs):
            if cmd[1].endswith('prepare_pillow_holdout.py'):
                split_fixture(paths, self.origins)

        with patch.object(launch, 'run_command', side_effect=fake_run) as run:
            self.run_main('prepare', '--reuse-converted')
        self.assertFalse(any('convert_aloha_to_dreamzero.py' == Path(c.args[0][1]).name for c in run.call_args_list))
        with patch.object(launch, 'run_command') as run, self.assertRaisesRegex(ValueError, '训练集已存在'):
            self.run_main('prepare', '--reuse-converted')
        run.assert_not_called()

    def test_prepare_subprocess_failure_stops_pipeline(self):
        paths = launch.paths_for(self.args('prepare'))
        raw_fixture(paths.raw)
        failure = subprocess.CalledProcessError(7, 'converter')
        with patch.object(launch, 'run_command', side_effect=failure) as run, self.assertRaises(subprocess.CalledProcessError):
            self.run_main('prepare')
        self.assertEqual(run.call_count, 1)
        self.assertFalse(paths.train.exists())

    def test_simple_cli_prepares_checks_and_trains_without_user_environment(self):
        paths = launch.paths_for(self.args())
        raw_fixture(paths.raw, task='pick_cup')

        def fake_run(cmd, **kwargs):
            if cmd[1].endswith('convert_aloha_to_dreamzero.py'):
                converted_fixture(paths.converted, paths.raw.name, self.origins)
            elif cmd[1].endswith('prepare_pillow_holdout.py'):
                split_fixture(paths, self.origins)

        with (patch.dict(os.environ, {}, clear=True), redirect_stdout(StringIO()),
              patch.object(launch, 'run_command', side_effect=fake_run) as run):
            launch.main(['--train_dir', 'data/flip_150', '--gpus', '1,2,3,4,5,6', '--holdout', '1',
                         '--save_steps', '200', '--batch_size', '1', '--report_to', 'none'])
        calls = [c.args[0] for c in run.call_args_list]
        self.assertEqual([Path(c[1]).name for c in calls], ['convert_aloha_to_dreamzero.py',
                         *['check_aloha_alignment.py'] * 3, 'prepare_pillow_holdout.py',
                         'train_flip_pillow_lora.py', 'train_flip_pillow_lora.py'])
        self.assertEqual([c[2] for c in calls[-2:]], ['check', 'train'])
        self.assertEqual(calls[0][-2:], ['--task-text', TASK])
        self.assertEqual(run.call_args.kwargs['env']['NUM_GPUS'], '6')

    def test_simple_cli_reuses_valid_prepared_data(self):
        self.dataset()
        with (patch.dict(os.environ, {}, clear=True), redirect_stdout(StringIO()),
              patch.object(launch, 'prepare') as prep, patch.object(launch, 'run_command') as run):
            launch.main(['--train_dir', 'data/flip_150', '--gpus', '1', '--holdout', '1'])
        prep.assert_not_called()
        self.assertEqual([c.args[0][2] for c in run.call_args_list], ['check', 'train'])

    def test_simple_cli_reuses_conversion_and_creates_split(self):
        paths = self.dataset(split=False)

        def fake_run(cmd, **kwargs):
            if cmd[1].endswith('prepare_pillow_holdout.py'):
                split_fixture(paths, self.origins)

        with (patch.dict(os.environ, {}, clear=True), redirect_stdout(StringIO()),
              patch.object(launch, 'run_command', side_effect=fake_run) as run):
            launch.main(['--train_dir', 'data/flip_150', '--gpus', '1', '--holdout', '1'])
        self.assertEqual([Path(c.args[0][1]).name for c in run.call_args_list],
                         [*['check_aloha_alignment.py'] * 3, 'prepare_pillow_holdout.py',
                          'train_flip_pillow_lora.py', 'train_flip_pillow_lora.py'])

    def test_simple_cli_failed_check_never_starts_training(self):
        self.dataset()
        with (patch.dict(os.environ, {}, clear=True), redirect_stdout(StringIO()),
              patch.object(launch, 'run_command', side_effect=subprocess.CalledProcessError(3, 'check')) as run,
              self.assertRaises(subprocess.CalledProcessError)):
            launch.main(['--train_dir', 'data/flip_150', '--gpus', '1', '--holdout', '1'])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][2], 'check')

    def test_simple_cli_nonempty_checkpoint_stops_before_preparation(self):
        paths = launch.paths_for(self.args())
        paths.output.mkdir(parents=True)
        (paths.output / 'existing').write_text('keep me')
        with (patch.dict(os.environ, {}, clear=True), redirect_stdout(StringIO()),
              patch.object(launch, 'prepare') as prep, patch.object(launch, 'run_command') as run,
              self.assertRaisesRegex(ValueError, '不覆盖')):
            launch.main(['--train_dir', 'data/flip_150', '--gpus', '1'])
        prep.assert_not_called()
        run.assert_not_called()

    def test_named_mode_and_weight_directory_flags(self):
        paths = self.dataset()
        with (patch.dict(os.environ, {'DZ_CKPT_DIR': '/old/base'}, clear=True), redirect_stdout(StringIO()),
              patch.object(launch, 'run_command') as run):
            launch.main(['--mode', 'smoke', '--train_dir', 'data/flip_150', '--gpus', '1', '--holdout', '1',
                         '--base_dir', 'weights/agibot', '--wan_dir', 'weights/wan',
                         '--tokenizer_dir', 'weights/tokenizer', '--run_name', 'test_run'])
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[2], 'smoke')
        self.assertEqual(cmd[cmd.index('--output') + 1], str(self.root / 'checkpoints/test_run_smoke'))
        self.assertEqual(cmd[cmd.index('--data') + 1], str(paths.train))
        env = run.call_args.kwargs['env']
        self.assertEqual(env['DZ_CKPT_DIR'], str(self.root / 'weights/agibot'))
        self.assertEqual(env['WAN_CKPT_DIR'], str(self.root / 'weights/wan'))
        self.assertEqual(env['TOKENIZER_DIR'], str(self.root / 'weights/tokenizer'))

    def test_simple_cli_dry_run_is_read_only(self):
        with (patch.dict(os.environ, {}, clear=True), redirect_stdout(StringIO()),
              patch.object(launch, 'source_ids') as raw, patch.object(launch, 'validate_training_view') as view,
              patch.object(launch.subprocess, 'run') as run):
            launch.main(['--train_dir', 'data/flip_150', '--gpus', '1,2,3,4,5,6', '--dry_run'])
            launch.main(['--mode', 'prepare', '--train_dir', 'data/flip_150', '--dry_run'])
        raw.assert_not_called()
        view.assert_not_called()
        run.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_subprocess_uses_same_python_repo_cwd_and_no_shell(self):
        cmd = launch.command('training/train_flip_pillow_lora.py', 'check', '--data', 'data/带 空格')
        env = {'CUDA_VISIBLE_DEVICES': '1'}
        with patch.object(launch.subprocess, 'run') as run, redirect_stdout(StringIO()):
            launch.run_command(cmd, env=env)
        run.assert_called_once_with(cmd, cwd=self.root, env=env, check=True)
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[-1], 'data/带 空格')


if __name__ == '__main__':
    unittest.main()
