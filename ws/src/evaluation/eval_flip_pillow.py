#!/usr/bin/env python3
"""Compare DreamZero checkpoints offline on the Mobile ALOHA pillow holdout (no robot/W&B)."""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

if not __package__:  # Make both the workspace packages and the official groot package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.common.pillow import TASK
from ws.src.evaluation.pillow_eval_policy import independent_policy, recipe, single_gpu_mask
from ws.src.evaluation.pillow_open_loop import (
    FPS, H, Holdout, action_chunk, anchor_images, crossings, match_events, metrics, observation, read_rows,
)

SIDES = ('left', 'right')
DIRECTIONS = ('open', 'close')
EVENT_FIELDS = ('checkpoint', 'source_id', 'episode_index', 'side', 'direction', 'status',
                'gt_source_frame', 'pred_source_frame', 'lag_frames', 'lag_ms')


def plan(episodes, max_blocks):
    """Prefix cap is global per checkpoint; with no cap every real row is scored once."""
    jobs = []
    for ep in episodes:
        n = ep.length if max_blocks is None else min(ep.length, max_blocks * H)
        if n == 0:
            break
        jobs.append((ep, n))
        if max_blocks is not None:
            max_blocks -= (n + H - 1) // H
    return jobs


def make_predictions(data, ep, n, state, predict):
    pred = np.empty((n, 14), dtype=np.float32)
    end = 0
    for t, images in anchor_images(ep, n):
        if t != end:
            raise ValueError(f'Non-contiguous anchors for episode {ep.id}: {t} not {end}')
        obs = observation(images, state[t], data.state)
        chunk = action_chunk(predict(obs), data.action)  # official unapply already returned absolute commands
        end = min(t + H, n)
        pred[t:end] = chunk[:end-t]
    if end != n:
        raise ValueError(f'Missing predictions for episode {ep.id}: scored {end}/{n}')
    return pred


def events_for(label, data, ep, gt, pred, tolerance):
    rows = []
    for side in SIDES:
        col = data.action[side + '_gripper_pos'].start
        ref, est = (crossings(x[:, col], data.thresholds[side]) for x in (gt, pred))
        for direction in DIRECTIONS:
            for g, p in match_events(ref[direction], est[direction], tolerance):
                lag = p-g if p is not None and g is not None else None
                rows.append(dict(checkpoint=label, source_id=ep.source, episode_index=ep.id, side=side,
                                 direction=direction, status='matched' if lag is not None else 'missed' if p is None else 'extra',
                                 gt_source_frame=None if g is None else ep.source_start+g,
                                 pred_source_frame=None if p is None else ep.source_start+p,
                                 lag_frames=lag, lag_ms=None if lag is None else lag * 1000/FPS))
    return rows


def event_counts(events):
    lags = [e['lag_ms'] for e in events if e['status'] == 'matched']
    missed = sum(e['status'] == 'missed' for e in events)
    extra = sum(e['status'] == 'extra' for e in events)
    m = len(lags)
    return dict(grip_gt=m+missed, grip_pred=m+extra, grip_matched=m, grip_missed=missed, grip_extra=extra,
                grip_f1=2*m/(2*m+missed+extra) if 2*m+missed+extra else None,
                grip_mean_lag_ms=float(np.mean(lags)) if m else None,
                grip_mean_abs_lag_ms=float(np.abs(lags).mean()) if m else None)


def report(label, segments, events, layout):
    """Within a source use all its frames; across sources report the arithmetic mean of source scores."""
    by_source = defaultdict(list)
    for ep, gt, pred in segments:
        by_source[ep.source].append((gt, pred))
    per_source = []
    for source, parts in sorted(by_source.items()):
        gt, pred = (np.concatenate(x) for x in zip(*parts))
        scores = metrics(pred, gt, layout)
        per_source.append(dict(checkpoint=label, source_id=source, n_segments=len(parts), n_frames=len(gt), **scores,
                               **event_counts([e for e in events if e['source_id'] == source])))
    summary = dict(checkpoint=label, n_sources=len(per_source), n_segments=len(segments),
                   n_frames=sum(r['n_frames'] for r in per_source),
                   **{k: float(np.mean([r[k] for r in per_source])) for k in scores}, **event_counts(events))
    timing = [dict(checkpoint=label, side=s, direction=d,
                   **event_counts([e for e in events if e['side'] == s and e['direction'] == d]))
              for s in SIDES for d in DIRECTIONS]
    return {'comparison': [summary], 'per_source': per_source, 'gripper_summary': timing, 'events': events}


def save_trace(folder, data, ep, n, predict):
    state, gt = read_rows(ep)
    state, gt = state[:n], gt[:n]
    pred = make_predictions(data, ep, n, state, predict)
    np.savez_compressed(folder / f'episode_{ep.id:06}.npz', gt=gt, pred=pred, state=state,
                        source_id=ep.source, source_frames=ep.source_start+np.arange(n),
                        action_names=data.names, anchors=np.arange(0, n, H))
    return ep, state, gt, pred


def score_traces(step, data, traces, tolerance, baseline):
    segments, events, holds, hold_events = [], [], [], []
    for ep, state, gt, pred in traces:
        n = len(gt)
        segments.append((ep, gt, pred))
        events.extend(events_for(step, data, ep, gt, pred, tolerance))  # never match across segments
        if baseline:
            hold = np.empty_like(pred)
            for k, s in data.action.items():
                hold[:, s] = state[(np.arange(n)//H)*H, data.state[k]]
            holds.append((ep, gt, hold))
            hold_events.extend(events_for('hold_state', data, ep, gt, hold, tolerance))
    reports = [report('hold_state', holds, hold_events, data.action)] if baseline else []
    reports.append(report(step, segments, events, data.action))
    return reports


def evaluate_step(step, data, jobs, predict, output, tolerance, baseline=False):
    """predict receives current observations only. Truth is used exclusively for scoring."""
    folder = output / f'checkpoint-{step}'
    folder.mkdir()
    def traces():
        for j, (ep, n) in enumerate(jobs, 1):
            print(f'[{step}] {j}/{len(jobs)} source={ep.source}, episode={ep.id}: {n} frames, {(n+H-1)//H} blocks', flush=True)
            yield save_trace(folder, data, ep, n, predict)
    return score_traces(step, data, traces(), tolerance, baseline)


def gpu_step(step, ckpt, verified, data, jobs, opts, baseline):
    with independent_policy(ckpt, verified, opts.seed) as (_, joint):
        return evaluate_step(step, data, jobs, lambda obs: joint(obs)[0].act, opts.output, opts.event_tolerance, baseline)


def save_tables(output, reports):
    for name in ('comparison', 'per_source', 'gripper_summary', 'events'):
        rows = [row for r in reports for row in r[name]]
        tmp = output / f'.{name}.csv.tmp'
        with tmp.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=EVENT_FIELDS if name == 'events' else list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        tmp.replace(output / f'{name}.csv')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--run', type=Path, default=Path('checkpoints/flip_pillow_lora_01'), help='包含 checkpoint-N 的训练目录')
    p.add_argument('--steps', type=int, nargs='+', default=[1200, 1600, 2000])
    p.add_argument('--train-data', type=Path, default=Path('data/flip_the_pillow_dz_train'), help='含 meta/holdout.json 的训练视图')
    p.add_argument('--full-data', type=Path, help='全量转换集，默认读 holdout.json 中的 full_dataset')
    p.add_argument('--base', type=Path, help='原始 DreamZero-AgiBot 权重目录，默认读取训练配置；不是 LoRA 节点')
    p.add_argument('--output', type=Path, default=Path('results/pillow_open_loop'), help='必须是新目录，不覆盖')
    p.add_argument('--seed', type=int, default=1140, help='所有窗口及节点使用相同推理噪声种子')
    p.add_argument('--event-tolerance', type=int, default=25, metavar='FRAMES', help='同侧同向开/关事件匹配的最大帧差（含边界）')
    p.add_argument('--limit-recordings', type=int, help='只取前 N 条留出录制（冒烟）')
    p.add_argument('--max-blocks', type=int, help='每个节点最多评估前 N 个窗口，跨录制合计（冒烟）')
    p.add_argument('--check-only', action='store_true', help='只检留出映射、行/视频帧、配置和权重文件头；不加载模型，不创建输出')
    o = p.parse_args(argv)
    if (min(o.steps) < 1 or len(o.steps) != len(set(o.steps)) or not 0 <= o.seed < 2**32 or o.event_tolerance < 0
            or any(n is not None and n < 1 for n in (o.limit_recordings, o.max_blocks))):
        p.error('steps 必须为不同正整数、limit/max-blocks 为正整数、seed 为 uint32、event-tolerance 不可为负')
    return o


def main(opts):
    train = opts.train_data.expanduser().resolve(strict=True)
    run = opts.run.expanduser().resolve(strict=True)
    data = Holdout(train, opts.full_data and opts.full_data.expanduser(), opts.limit_recordings)
    jobs = plan(data.episodes, opts.max_blocks)
    planned = {step: run / f'checkpoint-{step}' for step in opts.steps}
    absent = [str(p) for p in planned.values() if not p.is_dir()]
    if absent:
        raise ValueError(f"缺少节点：{absent}。{run} 现有：{sorted(p.name for p in run.glob('checkpoint-*') if p.is_dir())}")
    verified = {step: recipe(p, train, data, opts.base) for step, p in planned.items()}
    frames = sum(n for _, n in jobs)
    blocks = sum((n+H-1)//H for _, n in jobs)
    print(f'留出原始 ID={sorted({ep.source for ep, _ in jobs})}，{len(jobs)} 段；每个节点 {frames} 帧 / {blocks} 窗口。'
          f'夹爪开关阈值 (low, high)={data.thresholds}，匹配容差 ±{opts.event_tolerance} 帧。', flush=True)
    if opts.check_only:
        for ep, n in jobs:
            read_rows(ep)
            for _ in anchor_images(ep, n):
                pass
        print('预检通过（未加载模型，未运行推理，未写入文件）。')
        return
    mask = single_gpu_mask()
    if opts.output.is_symlink():
        raise ValueError(f'输出不能是软链接：{opts.output}')
    opts.output = opts.output.expanduser().resolve()
    for path in [train, data.full, run, *[p for v in verified.values() for p in v.protected_dirs]]:
        if opts.output == path or path in opts.output.parents or opts.output in path.parents:
            raise ValueError(f'输出目录与输入重叠：{opts.output} / {path}')
    opts.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(steps=opts.steps, seed=opts.seed, train=str(train), full=str(data.full), run=str(run),
                    task=TASK, horizon=H, fps=FPS, tolerance_frames=opts.event_tolerance, thresholds=data.thresholds,
                    recordings_limit=opts.limit_recordings, blocks_limit=opts.max_blocks,
                    frames_per_checkpoint=frames, blocks_per_checkpoint=blocks,
                    segments=[dict(source_id=ep.source, episode_index=ep.id, source_start=ep.source_start, frames=n) for ep, n in jobs],
                    aggregation='MAE/RMSE macro by source; events pooled within side/direction after per-segment matching',
                    base_by_step={s: str(v.base_files[0].parent) for s, v in verified.items()},
                    cuda_visible_devices=mask, num_dit_steps=os.environ.get('NUM_DIT_STEPS', '8'),
                    dynamic_cache_schedule=os.environ.get('DYNAMIC_CACHE_SCHEDULE', 'False'))
    (opts.output/'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False)+'\n')
    reports = []
    for i, (step, ckpt) in enumerate(planned.items()):
        print(f'开始 checkpoint-{step}（单 GPU 串行）：{ckpt}', flush=True)
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as pool:
            reports.extend(pool.submit(gpu_step, step, ckpt, verified[step], data, jobs, opts, i == 0).result())
        save_tables(opts.output, reports)  # preserve completed checkpoints even if a subsequent one fails
        s = reports[-1]['comparison'][0]
        print(f"[{step}] 双臂 MAE (rad) 左={s['left_arm_mae_rad']:.5f}, 右={s['right_arm_mae_rad']:.5f}; "
              f"夹爪匹配={s['grip_matched']}, 漏报={s['grip_missed']}, 误报={s['grip_extra']}", flush=True)
    print(f'完成：{opts.output / "comparison.csv"}；夹爪分侧/方向详见 gripper_summary.csv。')


if __name__ == '__main__':
    try:
        main(parse_args())
    except (OSError, ValueError, KeyError, ImportError) as exc:
        sys.exit(f'评估失败：{exc}')
