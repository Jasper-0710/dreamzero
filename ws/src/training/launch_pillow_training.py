#!/usr/bin/env python3
"""Dataset-selectable launcher for the existing Mobile ALOHA pillow LoRA recipe.

This is ordinary task adaptation, NOT DreamZero-Flash and NOT checkpoint resume.
The default workflow prepares data if needed, checks it, then starts training.
No model is loaded in this orchestration process.
"""

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.common.pillow import ROOT, TASK, check_data, jsonl, read_json


@dataclass(frozen=True)
class Paths:
    raw: Path
    converted: Path
    train: Path
    output: Path


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def paths_for(args):
    raw = repo_path(args.dataset)
    converted = repo_path(args.converted) if args.converted else raw.with_name(raw.name + '_dz')
    train = repo_path(args.train_data) if args.train_data else converted.with_name(converted.name + '_train')
    name = args.run_name or raw.name + '_lora_01'
    if name in ('.', '..') or not name.strip() or '/' in name or '\\' in name:
        raise ValueError('--run-name 必须是单个目录名，不得包含路径分隔符；自定义路径请用 --output')
    if args.mode == 'smoke':
        name += '_smoke'
    output = repo_path(args.output) if args.output else (ROOT / 'checkpoints' / name).resolve()
    paths = Paths(raw, converted, train, output)
    values = list(vars(paths).values())
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError(f'原始、转换、训练集和 checkpoint 目录必须互不重叠：{a} / {b}')
    return paths


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = ['all', 'prepare', 'check', 'smoke', 'train']
    p.add_argument('mode', nargs='?', default='all', choices=modes,
                   help='默认 all：按需准备 → 预检 → 训练；也兼容原位置参数')
    p.add_argument('--mode', dest='selected_mode', choices=modes,
                   help='可选阶段：prepare / check / smoke / train；默认 all')
    p.add_argument('--train_dir', '--train-dir', '--dataset', dest='dataset', required=True, type=Path,
                   help='原始录制目录，例如 data/flip_150；不是转换后的 _dz_train；相对路径按仓库根目录解析')
    p.add_argument('--converted', type=Path, help='转换输出目录，默认 <dataset>_dz')
    p.add_argument('--train-data', '--train_data', type=Path, help='划分后的训练集目录，默认 <converted>_train')
    p.add_argument('--holdout', type=int, default=10, help='留出最后 N 条完整原始录制，默认 10；各阶段保持一致')
    p.add_argument('--task-text', '--task_text', help='prepare 时显式修正原录制标签；当前仅支持红胶带面朝上的翻枕头任务')
    p.add_argument('--confirm-same-task', '--confirm_same_task', action='store_true',
                   help='与 --task_text 配合：明确确认不同旧标签对应的录制实际上全部是同一翻枕头任务')
    p.add_argument('--reuse-converted', '--reuse_converted', action='store_true', help='仅 prepare：显式复用已有转换结果，仍检查来源并重新核对对齐')
    p.add_argument('--gpus', help='已分配的 GPU，逗号分隔，例如 1,2,3,4,5,6；否则使用 CUDA_VISIBLE_DEVICES')
    p.add_argument('--num-gpus', '--num_gpus', type=int, help='训练进程数，默认使用所选 GPU 数；不沿用可能过期的 NUM_GPUS')
    p.add_argument('--run-name', '--run_name', help='checkpoint 目录名，默认 <dataset目录名>_lora_01；smoke 自动加 _smoke')
    p.add_argument('--output', type=Path, help='显式指定 checkpoint 输出路径；已有非空目录不覆盖、不自动续训')
    p.add_argument('--steps', type=int, help='默认 train=2000，smoke=10')
    p.add_argument('--save-steps', '--save_steps', type=int, help='默认 train=每 200 步，smoke=最后一步')
    p.add_argument('--batch-size', '--batch_size', type=int, default=1, help='每张 GPU 的 batch size，默认 1')
    p.add_argument('--report-to', '--report_to', choices=['none', 'wandb', 'swanlab'], default='none')
    p.add_argument('--wandb-mode', '--wandb_mode', choices=['online', 'offline'], help='仅 --report-to wandb 时有效')
    p.add_argument('--wandb-project', '--wandb_project', help='仅 --report-to wandb 时有效；不传则沿用 WANDB_PROJECT')
    p.add_argument('--swanlab-mode', '--swanlab_mode', choices=['online', 'offline'],
                   help='仅 SwanLab：默认 online，offline 仅本地记录，不实时显示云端网页')
    p.add_argument('--swanlab-project', '--swanlab_project', help='仅 SwanLab：默认 dreamzero-flip-pillow')
    p.add_argument('--swanlab-workspace', '--swanlab_workspace', help='仅 SwanLab：工作空间名，不传则使用个人空间或 SDK 配置')
    p.add_argument('--base_dir', type=Path, help='AgiBot 基座目录，默认 checkpoints/DreamZero-AgiBot')
    p.add_argument('--wan_dir', type=Path, help='Wan 组件目录，默认 checkpoints/Wan2.1-I2V-14B-480P')
    p.add_argument('--tokenizer_dir', type=Path, help='Tokenizer 目录，默认 checkpoints/umt5-xxl')
    p.add_argument('--dry-run', '--dry_run', action='store_true', help='只打印计划；不访问数据/权重、不建目录、不启动子进程或 GPU')
    return p


def validate_args(args):
    if args.holdout < 1 or args.batch_size < 1:
        raise ValueError('--holdout 和 --batch-size 必须为正整数')
    steps = args.steps if args.steps is not None else (10 if args.mode == 'smoke' else 2000)
    save = args.save_steps if args.save_steps is not None else (steps if args.mode == 'smoke' else min(200, steps))
    if steps < 1 or not 1 <= save <= steps:
        raise ValueError('需要 1 <= save-steps <= steps')
    if args.task_text is not None and args.task_text.strip() != TASK:
        raise ValueError(f'当前训练/验证接口仅支持任务 {TASK!r}；其他任务需要先适配任务检查，不能静默改标签')
    if args.confirm_same_task and (args.task_text is None or args.mode not in ('all', 'prepare')):
        raise ValueError('--confirm_same_task 仅用于 all / prepare，且需要显式传 --task_text；不要用于合并真实不同的任务')
    if args.reuse_converted and args.mode not in ('prepare', 'all'):
        raise ValueError('--reuse-converted 只用于 prepare / all')
    if (args.wandb_mode or args.wandb_project) and args.report_to != 'wandb':
        raise ValueError('--wandb-mode / --wandb-project 需要同时设置 --report-to wandb')
    swanlab_flags = (args.swanlab_mode, args.swanlab_project, args.swanlab_workspace)
    if any(value is not None for value in swanlab_flags) and args.report_to != 'swanlab':
        raise ValueError('--swanlab-mode / --swanlab-project / --swanlab-workspace 需要同时设置 --report-to swanlab')
    for value in (args.swanlab_project, args.swanlab_workspace):
        if value is not None and not value.strip():
            raise ValueError('SwanLab 项目名和工作空间名不能是空字符串')
    return steps, save


def gpu_environment(args):
    env = os.environ.copy()
    mask = args.gpus if args.gpus is not None else env.get('CUDA_VISIBLE_DEVICES', '')
    parts = [part.strip() for part in mask.split(',')]
    if not mask.strip() or any(not part for part in parts) or len(parts) != len(set(parts)):
        raise ValueError('请用 --gpus 1,2,3,4,5,6 或 CUDA_VISIBLE_DEVICES 显式选择已分配的 GPU（不要重复）')
    if any(not (part.isdecimal() or part.startswith(('GPU-', 'MIG-'))) for part in parts):
        raise ValueError('GPU 编号必须用逗号分隔，例如 --gpus 1,2,3,4,5,6，不是 1-6')
    count = args.num_gpus if args.num_gpus is not None else len(parts)
    if not 1 <= count <= len(parts):
        raise ValueError('--num-gpus 必须在 1 到所选 GPU 数之间')
    env.update(CUDA_VISIBLE_DEVICES=','.join(parts), NUM_GPUS=str(count), PYTHONUNBUFFERED='1')
    if args.wandb_mode:
        env['WANDB_MODE'] = args.wandb_mode
    if args.wandb_project:
        env['WANDB_PROJECT'] = args.wandb_project
    # Runtime settings are managed here, not by user-written export/unset commands.
    env.pop('LOAD_TRT_ENGINE', None)
    if args.report_to == 'wandb':
        env.pop('WANDB_DISABLED', None)
        if env.get('WANDB_MODE', '').lower() == 'disabled':
            env['WANDB_MODE'] = 'offline'
    if args.report_to == 'swanlab':
        # Explicit backend selection must not inherit an old disabled/local mode.
        env['SWANLAB_MODE'] = args.swanlab_mode or 'online'
        env['SWANLAB_PROJECT'] = (args.swanlab_project or env.get('SWANLAB_PROJECT', '').strip()
                                  or 'dreamzero-flip-pillow')
        if args.swanlab_workspace is not None:
            env['SWANLAB_WORKSPACE'] = args.swanlab_workspace.strip()
        # Older HF callbacks treat ANY nonempty SWANLAB_LOG_MODEL (even 'false') as enabled.
        env.pop('SWANLAB_LOG_MODEL', None)
    for flag, key, default in (('base_dir', 'DZ_CKPT_DIR', 'DreamZero-AgiBot'),
                               ('wan_dir', 'WAN_CKPT_DIR', 'Wan2.1-I2V-14B-480P'),
                               ('tokenizer_dir', 'TOKENIZER_DIR', 'umt5-xxl')):
        env[key] = str(repo_path(getattr(args, flag) or ROOT / 'checkpoints' / default))
    return env


def command(script, *flags):
    return [sys.executable, str(ROOT / 'ws/src' / script), *map(str, flags)]


def run_command(cmd, *, env=None, dry_run=False):
    print('\n$ ' + shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def source_ids(raw):
    info = read_json(raw / 'meta/info.json')
    episodes = jsonl(raw / 'meta/episodes.jsonl')
    ids = [ep['episode_index'] for ep in episodes]
    if info['fps'] != 50 or not ids or ids != list(range(len(ids))) or len(ids) != info['total_episodes']:
        raise ValueError('原始数据必须为 50 Hz，且 episodes.jsonl 与 info.json 中的录制 ID/数量一致')
    return ids


def validate_conversion(paths):
    ids = source_ids(paths.raw)
    meta = paths.converted / 'meta'
    provenance = read_json(meta / 'conversion.json')
    origins = jsonl(meta / 'source_episodes.jsonl')
    total = read_json(meta / 'info.json')['total_episodes']
    if ([row['episode_index'] for row in origins] != list(range(total))
            or provenance['source_dataset'] != paths.raw.name or provenance['source_fps'] != 50
            or sorted(set(row['source_episode_index'] for row in origins)) != ids):
        raise ValueError('转换目录的来源名称、帧率或原始录制 ID 不匹配；拒绝混用数据集或不完整的转换结果')
    check_data(paths.converted)
    return origins


def validate_training_view(paths, holdout):
    origins = validate_conversion(paths)
    ids = source_ids(paths.raw)
    if not 1 <= holdout < len(ids):
        raise ValueError(f'--holdout 必须在 1 到 {len(ids) - 1} 之间；不会根据数据集名称猜测数量')
    meta = paths.train / 'meta'
    split = read_json(meta / 'holdout.json')
    held = ids[-holdout:]
    expected = [row['episode_index'] for row in origins if row['source_episode_index'] in held]
    if (Path(split['full_dataset']).resolve() != paths.converted
            or split['train_source_ids'] != ids[:-holdout]
            or split['held_out_source_ids'] != held
            or split['held_out_episode_ids_in_full_dataset'] != expected):
        raise ValueError('训练集 holdout.json 与所选数据集/--holdout 不一致；拒绝误用旧划分或全量数据')
    check_data(paths.train)
    actual = jsonl(meta / 'source_episodes.jsonl')
    expected_train = [row for row in origins if row['source_episode_index'] not in held]
    if actual != expected_train:
        raise ValueError('训练集 source_episodes.jsonl 与预期留出划分不一致')
    print(f'训练原始录制 {len(ids) - holdout} 条；留出 {len(held)} 条，ID={held}', flush=True)


def prepare(args, paths):
    convert = command('data_prep/convert_aloha_to_dreamzero.py', '--input', paths.raw, '--output', paths.converted)
    # Label replacement must be explicitly requested, even if the old recorder used "pick_cup".
    convert += ['--task-text', TASK] if args.task_text else ['--keep-task-text']
    if args.confirm_same_task:
        convert += ['--confirm-same-task']
    split = command('data_prep/prepare_pillow_holdout.py', '--input', paths.converted,
                    '--output', paths.train, '--holdout', args.holdout)
    if args.dry_run:
        if args.reuse_converted:
            print('计划：显式复用转换结果；实际执行时仍验证来源和数据结构。')
        else:
            run_command(convert, dry_run=True)
        run_command(command('data_prep/check_aloha_alignment.py', '--input', paths.raw,
                            '--output', paths.converted, '--episode', 0, '--samples', 5), dry_run=True)
        print('实际执行会根据转换后的 episode 数，核对首条、中间条和末条（去重）。')
        run_command(split, dry_run=True)
        return
    ids = source_ids(paths.raw)
    if not 1 <= args.holdout < len(ids):
        raise ValueError(f'--holdout 必须在 1 到 {len(ids) - 1} 之间')
    if paths.train.exists() or paths.train.is_symlink():
        raise ValueError(f'训练集已存在，不覆盖：{paths.train}；已准备好请改用 check / smoke / train')
    if paths.converted.exists() or paths.converted.is_symlink():
        if not args.reuse_converted:
            raise ValueError(f'转换目录已存在，不覆盖：{paths.converted}；确认来源后用 --reuse-converted 继续划分')
    else:
        if args.reuse_converted:
            raise ValueError('--reuse-converted 指定的转换目录不存在')
        tasks = jsonl(paths.raw / 'meta/tasks.jsonl')
        if not args.task_text and any(t['task'] != TASK for t in tasks):
            raise ValueError(f'原始任务标签为 {tasks!r}；确认全部录制均为红面朝上后显式传 --task-text {TASK!r}')
        print(f'原始任务表：{tasks!r}；转换器将先读取每条录制的 task_index，统计实际使用的标签。', flush=True)
        run_command(convert)
    validate_conversion(paths)
    n = read_json(paths.converted / 'meta/info.json')['total_episodes']
    for ep in sorted({0, n // 2, n - 1}):
        run_command(command('data_prep/check_aloha_alignment.py', '--input', paths.raw,
                            '--output', paths.converted, '--episode', ep, '--samples', 5))
    run_command(split)
    validate_training_view(paths, args.holdout)
    print('\n数据准备完成；尚未加载模型或启动训练。下一步使用 check，再 smoke，最后 train。', flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    args.mode = args.selected_mode or args.mode
    if args.mode == 'all' and args.task_text is None and not args.confirm_same_task:
        args.task_text = TASK
    steps, save = validate_args(args)
    paths = paths_for(args)
    print('=== Mobile ALOHA 翻枕头：普通 LoRA（非 Flash，不自动续训旧 checkpoint） ===', flush=True)
    for label, path in zip(('原始录制', '转换全量', '仅训练集', 'checkpoint'), vars(paths).values()):
        print(f'{label}: {path}', flush=True)
    print(f'留出原始录制数: {args.holdout}', flush=True)
    if args.mode == 'all':
        print(f'本任务指令: {args.task_text}（请确认本批全部是红胶带面朝上）', flush=True)
    if args.dry_run:
        print('DRY RUN：只展示计划，不验证数据/权重/显存，不写入文件或执行命令。', flush=True)
    if args.mode == 'prepare':
        prepare(args, paths)
        return
    env = gpu_environment(args)
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}；训练进程数={env['NUM_GPUS']}", flush=True)
    if not args.dry_run:
        if args.mode != 'check' and paths.output.exists() and (not paths.output.is_dir() or any(paths.output.iterdir())):
            raise ValueError(f'checkpoint 输出非空，不覆盖、不自动续训：{paths.output}；请选择新 --run-name 或 --output')
    if args.mode == 'all':
        if args.dry_run:
            print('新数据计划如下；实际执行时，已有训练集先验证后复用，已有全量转换则核对后划分。', flush=True)
        if args.dry_run or not (paths.train.exists() or paths.train.is_symlink()):
            prep_args = argparse.Namespace(**vars(args))
            prep_args.mode = 'prepare'
            prep_args.reuse_converted = args.reuse_converted or (not args.dry_run and paths.converted.exists())
            prepare(prep_args, paths)
        else:
            print('训练集已存在：校验后复用，不重新转换或划分。', flush=True)
    if not args.dry_run:
        validate_training_view(paths, args.holdout)
    for mode in (('check', 'train') if args.mode == 'all' else (args.mode,)):
        cmd = command('training/train_flip_pillow_lora.py', mode, '--data', paths.train,
                      '--output', paths.output, '--steps', steps, '--save-steps', save,
                      '--batch-size', args.batch_size, '--report-to', args.report_to)
        run_command(cmd, env=env, dry_run=args.dry_run)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode if exc.returncode > 0 else 1)
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f'启动失败：{exc}')
    except KeyboardInterrupt:
        sys.exit(130)
