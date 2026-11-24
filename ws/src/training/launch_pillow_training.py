#!/usr/bin/env python3
"""Dataset-selectable launcher for the existing Mobile ALOHA pillow LoRA recipe.

This is ordinary task adaptation, NOT DreamZero-Flash and NOT checkpoint resume.
Preparation is deliberately separate from GPU training. No model is loaded here.
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
    p.add_argument('mode', choices=['prepare', 'check', 'smoke', 'train'],
                   help='prepare 转换/对齐/留出；check 预检；smoke 10 步；train 正式训练')
    p.add_argument('--dataset', required=True, type=Path, help='原始录制目录，例如 data/flip_150；相对路径均以仓库根目录为基准')
    p.add_argument('--converted', type=Path, help='转换输出目录，默认 <dataset>_dz')
    p.add_argument('--train-data', type=Path, help='划分后的训练集目录，默认 <converted>_train')
    p.add_argument('--holdout', type=int, default=10, help='留出最后 N 条完整原始录制，默认 10；各阶段保持一致')
    p.add_argument('--task-text', help='prepare 时显式修正原录制标签；当前仅支持红胶带面朝上的翻枕头任务')
    p.add_argument('--reuse-converted', action='store_true', help='仅 prepare：显式复用已有转换结果，仍检查来源并重新核对对齐')
    p.add_argument('--gpus', help='已分配的 GPU，逗号分隔，例如 1,2,3,4,5,6；否则使用 CUDA_VISIBLE_DEVICES')
    p.add_argument('--num-gpus', type=int, help='训练进程数，默认使用所选 GPU 数；不沿用可能过期的 NUM_GPUS')
    p.add_argument('--run-name', help='checkpoint 目录名，默认 <dataset目录名>_lora_01；smoke 自动加 _smoke')
    p.add_argument('--output', type=Path, help='显式指定 checkpoint 输出路径；已有非空目录不覆盖、不自动续训')
    p.add_argument('--steps', type=int, help='默认 train=2000，smoke=10')
    p.add_argument('--save-steps', type=int, help='默认 train=每 200 步，smoke=最后一步')
    p.add_argument('--batch-size', type=int, default=1, help='每张 GPU 的 batch size，默认 1')
    p.add_argument('--report-to', choices=['none', 'wandb'], default='none')
    p.add_argument('--wandb-mode', choices=['online', 'offline'], help='仅 --report-to wandb 时有效')
    p.add_argument('--wandb-project', help='仅 --report-to wandb 时有效；不传则沿用 WANDB_PROJECT')
    p.add_argument('--dry-run', action='store_true', help='只打印计划；不访问数据/权重、不建目录、不启动子进程或 GPU')
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
    if args.reuse_converted and args.mode != 'prepare':
        raise ValueError('--reuse-converted 只用于 prepare')
    if (args.wandb_mode or args.wandb_project) and args.report_to != 'wandb':
        raise ValueError('--wandb-mode / --wandb-project 需要同时设置 --report-to wandb')
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
    # Never pass an engine for a different model to this unquantized LoRA recipe.
    if env.get('LOAD_TRT_ENGINE'):
        raise ValueError('普通 LoRA 训练请先 unset LOAD_TRT_ENGINE；本入口不加载 TensorRT 引擎')
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
        if len(tasks) != 1 or (not args.task_text and tasks[0]['task'] != TASK):
            raise ValueError(f'原始任务标签为 {tasks!r}；确认全部录制均为红面朝上后显式传 --task-text {TASK!r}')
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
    steps, save = validate_args(args)
    paths = paths_for(args)
    print('=== Mobile ALOHA 翻枕头：普通 LoRA（非 Flash，不自动续训旧 checkpoint） ===', flush=True)
    for label, path in zip(('原始录制', '转换全量', '仅训练集', 'checkpoint'), vars(paths).values()):
        print(f'{label}: {path}', flush=True)
    print(f'留出原始录制数: {args.holdout}', flush=True)
    if args.dry_run:
        print('DRY RUN：只展示计划，不验证数据/权重/显存，不写入文件或执行命令。', flush=True)
    if args.mode == 'prepare':
        prepare(args, paths)
        return
    env = gpu_environment(args)
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}；训练进程数={env['NUM_GPUS']}", flush=True)
    if not args.dry_run:
        validate_training_view(paths, args.holdout)
        if args.mode != 'check' and paths.output.exists() and (not paths.output.is_dir() or any(paths.output.iterdir())):
            raise ValueError(f'checkpoint 输出非空，不覆盖、不自动续训：{paths.output}；请选择新 --run-name 或 --output')
    cmd = command('training/train_flip_pillow_lora.py', args.mode, '--data', paths.train,
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
