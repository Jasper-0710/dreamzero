#!/usr/bin/env python3
"""Import a finished DreamZero loss_log.jsonl into a separate W&B run; no training or GPU."""

import argparse
import json
import math
import os
from pathlib import Path
from uuid import uuid4

METRICS = ('loss', 'action_loss_avg', 'dynamics_loss_avg', 'learning_rate')


def read_history(path):
    # The Trainer may write several records for one optimizer step, including step 0.
    by_step = {}
    with path.open(encoding='utf-8') as src:
        for n, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'{path}:{n}: 无效 JSON，请确认运行已结束：{exc}') from exc
            if not isinstance(row, dict) or type(row.get('step')) is not int or row['step'] < 0:
                raise ValueError(f'{path}:{n}: 缺少有效训练步 step')
            for key in METRICS:
                if key not in row:
                    continue
                val = row[key]
                if type(val) not in (int, float) or not math.isfinite(val):
                    raise ValueError(f'{path}:{n}: {key} @ step {row["step"]} 不是有限数值')
                by_step.setdefault(row['step'], {})[key] = val
    if not any('loss' in metrics for metrics in by_step.values()):
        raise ValueError(f'{path}: 没有总 loss 记录，请选择已完成训练的 loss_log.jsonl')
    return sorted(by_step.items())


def write_history(run, history):
    # Match Transformers' axis and metric names so imported and live runs overlay.
    run.define_metric('train/global_step')
    run.define_metric('train/*', step_metric='train/global_step')
    for step, metrics in history:
        run.log({'train/global_step': step, **{f'train/{k}': v for k, v in metrics.items()}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path, help='已结束的训练输出目录里的 loss_log.jsonl')
    p.add_argument('--name', help='默认 <输出目录名>-imported；每次导入均为独立 run')
    p.add_argument('--dry-run', action='store_true', help='只校验并统计数据，不需要 wandb，不访问网络')
    args = p.parse_args()
    path = args.log.expanduser().resolve()
    try:
        history = read_history(path)
    except (OSError, ValueError) as exc:
        p.exit(1, f'未导入：{exc}\n')
    counts = ', '.join(f'{k}: {sum(k in row for _, row in history)}' for k in METRICS)
    print(f'step {history[0][0]}–{history[-1][0]}；各指标实际记录点数：{counts}', flush=True)
    if args.dry_run:
        return
    mode = os.environ.get('WANDB_MODE', 'online').lower()
    if mode == 'disabled':
        p.error('WANDB_MODE=disabled；请选 online（网页）或 offline（稍后同步）')
    try:
        import wandb
    except ImportError as exc:
        p.exit(1, f'无法导入 W&B SDK：{exc}。若未安装，请在当前环境运行 python -m pip install wandb\n')
    # A backfill must never attach to an active training run, even with WANDB_RUN_ID set.
    # Host/GPU samples taken during import would not be historical training metrics.
    settings = wandb.Settings(disable_git=True, console='off', _disable_stats=True)
    with wandb.init(project=os.environ.get('WANDB_PROJECT', '').strip() or 'dreamzero',
                    name=args.name or f'{path.parent.name}-imported', id=uuid4().hex,
                    resume='never', job_type='metrics-import', config={'source': 'loss_log.jsonl'},
                    save_code=False, settings=settings) as run:
        write_history(run, history)
        if not run.offline and run.url:
            print(f'W&B 曲线：{run.url}', flush=True)


if __name__ == '__main__':
    main()
