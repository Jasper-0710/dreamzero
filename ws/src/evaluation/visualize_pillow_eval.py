#!/usr/bin/env python3
"""把 eval_flip_pillow.py 已保存的结果制作成一个离线 HTML；不加载模型。"""
import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

TEMPLATE = Path(__file__).with_name('pillow_eval_report.html')
EVENT_COUNTS = ('grip_matched', 'grip_missed', 'grip_extra', 'grip_f1', 'grip_mean_lag_ms', 'grip_mean_abs_lag_ms')
ERRORS = tuple(f'{side}_{m}' for m in ('arm_mae_rad', 'arm_rmse_rad', 'gripper_mae') for side in ('left', 'right'))
TEXT = {'checkpoint', 'side', 'direction', 'status'}


def read_table(root, name, columns):
    """Keep checkpoint IDs categorical, numerical zeros as zeros, and empty delays/F1 as null."""
    path = root / (name + '.csv')
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path.name} 缺列：{sorted(missing)}')
        rows = []
        for row in reader:
            item = {}
            for k in columns:
                v = row[k]
                if k in TEXT:
                    item[k] = v
                elif v == '':
                    item[k] = None
                else:
                    number = float(v)
                    if not math.isfinite(number):
                        raise ValueError(f'{path.name} 第 {reader.line_num} 行，{k} 不是有限值')
                    item[k] = int(v) if v.lstrip('-').isdigit() else number
            rows.append(item)
        return rows


def series(a):
    # Channel-major for the browser; only the drawn lines are rounded. CSV metrics are used as written.
    return np.round(a.T.astype(np.float64), 6).tolist()


def build_data(root):
    with (root / 'protocol.json').open(encoding='utf-8') as f:
        p = json.load(f)
    summary = read_table(root, 'comparison', ('checkpoint', 'n_sources', 'n_segments', 'n_frames', *ERRORS))
    planned = [str(int(s)) for s in p['steps']]
    labels = [r['checkpoint'] for r in summary]
    if len(set(labels)) != len(labels) or 'hold_state' not in labels or set(labels) - {'hold_state', *planned}:
        raise ValueError('comparison.csv 的节点与 protocol.json 不一致')
    steps = [s for s in planned if s in labels]  # Ignore NPZ files left by an unfinished checkpoint.
    if not steps:
        raise ValueError('尚无已完成的 checkpoint；请等评估脚本写完汇总表再复制')
    groups = read_table(root, 'gripper_summary', ('checkpoint', 'side', 'direction', *EVENT_COUNTS))
    expected = {(k, s, d) for k in labels for s in ('left', 'right') for d in ('open', 'close')}
    if {(r['checkpoint'], r['side'], r['direction']) for r in groups} != expected or len(groups) != len(expected):
        raise ValueError('夹爪汇总表与节点对照表不同步；请重新复制已写完的结果')
    events = read_table(root, 'events', ('checkpoint', 'episode_index', 'side', 'direction', 'status',
                                        'gt_source_frame', 'pred_source_frame', 'lag_ms'))
    fps, horizon = p['fps'], p['horizon']
    if fps <= 0 or horizon <= 0 or not p['segments']:
        raise ValueError('protocol.json 的采样率、窗口或段信息无效')
    episodes, names = [], None
    for seg in p['segments']:
        eid, start, n = int(seg['episode_index']), int(seg['source_start']), int(seg['frames'])
        if n < 1:
            raise ValueError(f'episode {eid} 为空')
        anchors = np.arange(0, n, horizon)
        ep = dict(source=seg['source_id'], id=eid, start=start, n=n, anchors=anchors.tolist(), preds={})
        reference = None
        for step in steps:
            path = root / f'checkpoint-{step}' / f'episode_{eid:06d}.npz'
            with np.load(path, allow_pickle=False) as z:
                columns, gt, pred = z['action_names'].tolist(), z['gt'], z['pred']
                if (gt.shape != (n, 14) or pred.shape != gt.shape or not np.isfinite(gt).all()
                        or not np.isfinite(pred).all() or len(columns) != 14 or len(set(columns)) != 14
                        or z['source_id'].item() != ep['source'] or not np.array_equal(z['anchors'], anchors)
                        or not np.array_equal(z['source_frames'], np.arange(start, start+n))):
                    raise ValueError(f'{path} 的帧、窗口或动作维度与评估协议不符')
                if names is None:
                    names = columns
                if columns != names or (reference is not None and not np.array_equal(gt, reference)):
                    raise ValueError(f'{path} 动作列或录制真值与其他节点不一致，不能叠图')
                if reference is None:
                    reference, ep['gt'] = gt, series(gt)
                ep['preds'][step] = series(pred)
        episodes.append(ep)
    frames = sum(e['n'] for e in episodes)
    blocks = sum(len(e['anchors']) for e in episodes)
    sources = len({e['source'] for e in episodes})
    if (frames != p['frames_per_checkpoint'] or blocks != p['blocks_per_checkpoint']
            or any((r['n_frames'], r['n_segments'], r['n_sources']) != (frames, len(episodes), sources) for r in summary)):
        raise ValueError('汇总表的覆盖范围与逐帧轨迹不一致')
    return dict(steps=steps, planned=planned, fps=fps, horizon=horizon, tolerance=p['tolerance_frames'],
                thresholds=p['thresholds'], limited=any(p.get(k) is not None for k in ('blocks_limit', 'recordings_limit')),
                frames=frames, blocks=blocks, sources=sources, names=names, summary=summary, groups=groups,
                events=[r for r in events if r['checkpoint'] in steps], episodes=episodes)


def write_report(root, output=None):
    data = build_data(root)
    dest = output if output is not None else root / 'report.html'
    if dest.suffix.lower() != '.html' or dest.resolve() == TEMPLATE.resolve() or dest.is_symlink():
        raise ValueError('请使用独立的 .html 输出路径，不要覆盖模板或软链接')
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', r'\u003c')
    html = TEMPLATE.read_text(encoding='utf-8').replace('__REPORT_DATA__', payload)
    with tempfile.NamedTemporaryFile(dir=dest.parent, suffix='.html.tmp', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        tmp_path.write_text(html, encoding='utf-8')
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='含 comparison.csv、protocol.json 和 checkpoint-N/*.npz 的评估输出目录')
    parser.add_argument('--output', type=Path, help='默认 <结果目录>/report.html，可重复生成')
    args = parser.parse_args()
    try:
        path = write_report(args.results.expanduser(), args.output.expanduser() if args.output else None)
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        parser.exit(1, f'未生成报告：{exc}\n请确认复制了同一次评估的完整结果目录（不是训练权重目录）。\n')
    print(f'离线报告：{path.resolve()}\n只需用浏览器打开这个 HTML 文件。')


if __name__ == '__main__':
    main()
