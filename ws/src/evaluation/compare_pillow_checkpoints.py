#!/usr/bin/env python3
"""一次比较多个 DreamZero 节点：全留出集动作误差 + 同帧预测视频，无真机或 W&B。"""
import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.evaluation import eval_flip_pillow as ev, export_pillow_video as video
from ws.src.evaluation.pillow_compare_report import mosaic_frames, video_section
from ws.src.evaluation.pillow_eval_policy import clear_window_cache, independent_policy, recipe
from ws.src.evaluation.pillow_video_workers import run_workers, video_devices
from ws.src.evaluation.visualize_pillow_eval import write_report


@dataclass
class Lane:
    gpu: str
    step: int
    actions: list = field(default_factory=list)  # never split an episode for gripper event scoring
    videos: list = field(default_factory=list)
    cost: int = 0


def waves(steps, devices, actions, windows):
    """Spread each wave across checkpoints; balance by inference windows, not episode count."""
    for start in range(0, len(steps), len(devices)):
        batch = steps[start:start+len(devices)]
        q, r = divmod(len(devices), len(batch))
        idx, wave = 0, []
        for i, step in enumerate(batch):
            count = min(q + (i < r), len(actions) + len(windows))
            lanes = [Lane(gpu, step) for gpu in devices[idx:idx+count]]
            idx += count
            for ep, n in sorted(actions, key=lambda x: -x[1]):
                lane = min(lanes, key=lambda p: p.cost)
                lane.actions.append((ep, n)); lane.cost += (n+ev.H-1)//ev.H
            for ep, a in windows:  # a = segment-local anchor
                lane = min(lanes, key=lambda p: p.cost)
                lane.videos.append((ep, a)); lane.cost += 1
            wave.extend(lanes)
        yield wave


def worker(lane, ckpt, verified, data, out, o, *, load_slots, ready):
    folder = out / ckpt.name
    with ExitStack() as stack:
        with load_slots:
            print(f'[GPU {lane.gpu}] 加载 {ckpt.name}', flush=True)
            policy, joint = stack.enter_context(independent_policy(ckpt, verified, o.seed))
        ready.wait()
        head = policy.trained_model.action_head
        def forecast(obs):
            _, latent = joint(obs)
            clear_window_cache(head)
            return video.decode_prediction(head, latent, o.tiled_decode)
        for i, (ep, n) in enumerate(lane.actions, 1):
            print(f'[GPU {lane.gpu} / {ckpt.name}] 动作：{i}/{len(lane.actions)} source={ep.source} episode={ep.id}', flush=True)
            ev.save_trace(folder, data, ep, n, lambda obs: joint(obs)[0].act)
        for ep, a in lane.videos:
            frames = video.paired_frames([(ep, [a])], data.state, forecast,
                                         lambda obs: video.recorded_grid(policy, obs), lane.step)
            with (folder/f'window-{ep.source_start+a}.npyseq').open('xb') as f:
                for gt, pred, _, _, _ in frames:
                    np.save(f, np.stack((gt, pred)), allow_pickle=False)
        return bool(head.tiled or o.tiled_decode)


def checkpoints(run, steps):
    paths = {s: run/f'checkpoint-{s}' for s in steps}
    missing = [s for s,p in paths.items() if not p.is_dir()]
    if missing:
        available = sorted(int(p.name.split('-')[-1]) for p in run.glob('checkpoint-*')
                           if p.is_dir() and p.name.removeprefix('checkpoint-').isdecimal())
        raise ValueError(f'{run} 缺少 step {missing}；实际可用 steps：{available}。请显式改选，不会自动替换。')
    return paths


def saved_traces(folder, jobs):
    for ep, n in jobs:
        with np.load(folder/f'episode_{ep.id:06}.npz', allow_pickle=False) as z:
            if z['pred'].shape != (n, 14):
                raise ValueError(f'{folder}/episode {ep.id}: 轨迹长度或维度不符')
            yield ep, z['state'], z['gt'], z['pred']


def main(o):
    run = o.run.expanduser().resolve(strict=True)
    ckpts = checkpoints(run, o.steps)  # before importing torch or reading the dataset
    train = o.train_data.expanduser().resolve(strict=True)
    data = ev.Holdout(train, o.full_data and o.full_data.expanduser(), o.limit_recordings)
    jobs = ev.plan(data.episodes, o.max_blocks)
    grouped = video.select_windows(data.episodes, o.source_id, o.anchors)
    windows = [(ep,a) for ep, local in grouped for a in local]
    devices = video_devices(o.check_only)
    recipes = {s: recipe(p, train, data, o.base) for s,p in ckpts.items()}
    for step, ckpt in ckpts.items():
        video.check_video_recipe(ckpt, recipes[step])
    video.check_encoder()
    frames = sum(n for _,n in jobs)
    blocks = sum((n+ev.H-1)//ev.H for _,n in jobs)
    print(f'每个节点：动作 {len({e.source for e,_ in jobs})} 条录制 / {frames} 帧 / {blocks} 窗口；'
          f'视频 source={o.source_id} anchors={list(o.anchors)}；相同 seed={o.seed}。', flush=True)
    schedule = list(waves(o.steps, devices, jobs, windows)) if devices else []
    for i, wave in enumerate(schedule, 1):
        for lane in wave:
            print(f'批次 {i}：GPU {lane.gpu} → checkpoint-{lane.step}，动作段 {len(lane.actions)}，视频窗 {len(lane.videos)}', flush=True)
    if o.check_only:
        for ep,n in jobs:
            ev.read_rows(ep)
            for _ in ev.anchor_images(ep,n):
                pass
        for ep, anchors in grouped:
            for _ in video.windows_rgb(ep,anchors):
                pass
        print('预检通过，没有加载模型或写文件。')
        return
    out = o.output.expanduser()
    if out.is_symlink():
        raise ValueError(f'拒绝软链接输出：{out}')
    out = out.resolve()
    for p in (train, data.full, run, *(x for v in recipes.values() for x in v.protected_dirs)):
        if out == p or out in p.parents or p in out.parents:
            raise ValueError(f'输出不能与数据或模型目录重叠：{out} / {p}')
    out.mkdir(parents=True, exist_ok=False)
    for s in o.steps:
        (out/ckpts[s].name).mkdir()
    maps = [dict(wave=i, device=p.gpu, step=p.step, episodes=[e.id for e,_ in p.actions],
                 video_anchors=[e.source_start+a for e,a in p.videos]) for i,w in enumerate(schedule,1) for p in w]
    protocol = dict(steps=o.steps, seed=o.seed, run=str(run), train=str(train), full=str(data.full), task=ev.TASK,
                    horizon=ev.H, fps=ev.FPS, thresholds=data.thresholds, tolerance_frames=o.event_tolerance,
                    segments=[dict(source_id=e.source, episode_index=e.id, source_start=e.source_start, frames=n) for e,n in jobs],
                    frames_per_checkpoint=frames, blocks_per_checkpoint=blocks, blocks_limit=o.max_blocks,
                    recordings_limit=o.limit_recordings, gpu_assignments=maps, num_dit_steps=os.getenv('NUM_DIT_STEPS','8'),
                    dynamic_cache_schedule=os.getenv('DYNAMIC_CACHE_SCHEDULE','False'))
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    modes, reports = set(), []
    try:
        for wave in schedule:
            tasks = [(p.gpu, (p, ckpts[p.step], recipes[p.step], data, out, o)) for p in wave]
            modes.update(run_workers(worker, tasks, o.parallel_loads))
            if len(modes) != 1:
                raise ValueError('不同节点/分片的 VAE 解码模式不一致')
            for step in dict.fromkeys(p.step for p in wave):
                reports.extend(ev.score_traces(step, data, saved_traces(out/ckpts[step].name, jobs),
                                               o.event_tolerance, baseline=step == o.steps[0]))
                m = reports[-1]['comparison'][0]
                print(f"[step {step}] 左/右臂 MAE (rad) {m['left_arm_mae_rad']:.5f} / {m['right_arm_mae_rad']:.5f}，"
                      f"夹爪匹配/漏报/误报 {m['grip_matched']}/{m['grip_missed']}/{m['grip_extra']}", flush=True)
            ev.save_tables(out, reports)
        tmp = out/'comparison.partial.mp4'
        n = video.encode_mp4(tmp, mosaic_frames(out, o.steps, o.source_id, o.anchors), video.RATE)
        if n != len(o.anchors)*len(video.OFFSETS):
            raise ValueError(f'对照视频帧数错误：{n}')
        tmp.rename(out/'comparison.mp4')
        meta = video.manifest(grouped, ckpts[o.steps[0]], o.seed, next(iter(modes)))
        del meta['checkpoint']
        meta['checkpoints'] = [str(p) for p in ckpts.values()]
        (out/'windows.json').write_text(json.dumps(meta, indent=2)+'\n')
        write_report(out, out/'index.html', video_section=video_section(o.source_id, o.anchors))
    finally:  # retain completed metrics/NPZ on failure, not bulk intermediate RGB data
        for p in out.glob('checkpoint-*/window-*.npyseq'):
            p.unlink()
        (out/'comparison.partial.mp4').unlink(missing_ok=True)
    print(f'完成：{out}/index.html（动作误差 + 同步视频），指标：{out}/comparison.csv')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('checkpoints/flip_pillow_lora_01'))
    p.add_argument('--steps', type=int, nargs='+', default=[1200,1600,2000], help='同一 run 中实际保存的节点')
    p.add_argument('--train-data', type=Path, default=Path('data/flip_the_pillow_dz_train'))
    p.add_argument('--full-data', type=Path)
    p.add_argument('--base', type=Path)
    p.add_argument('--source-id', type=int, default=91, help='视频用的原始录制号，默认 91；不缩小动作评估范围')
    p.add_argument('--anchors', type=int, nargs='+', help='视频使用的原始起始帧，录制 91 默认 168 192 216 240 264 288')
    p.add_argument('--seed', type=int, default=1140)
    p.add_argument('--parallel-loads', type=int, default=1, help='允许几张卡同时读权重；不限制并行推理')
    p.add_argument('--event-tolerance', type=int, default=25)
    p.add_argument('--tiled-decode', action='store_true')
    p.add_argument('--limit-recordings', type=int, help='冒烟：只读取前 N 条留出录制，必须包含视频 source')
    p.add_argument('--max-blocks', type=int, help='冒烟：每节点最多评估前 N 个动作窗口，不改变视频窗口')
    p.add_argument('--check-only', action='store_true', help='仅预检，不加载模型，不写输出')
    p.add_argument('--output', type=Path, default=Path('results/pillow_compare'), help='新目录，不覆盖旧结果')
    o = p.parse_args(argv)
    if len(set(o.steps)) != len(o.steps) or min(o.steps) < 1 or not 0 <= o.seed < 2**32 or o.event_tolerance < 0:
        p.error('steps 必须为不重复正整数，seed 为 uint32，event-tolerance 不可为负')
    if any(v is not None and v < 1 for v in (o.parallel_loads, o.limit_recordings, o.max_blocks)):
        p.error('parallel-loads/limit-recordings/max-blocks 必须为正')
    if o.anchors is None:
        if o.source_id != 91:
            p.error('非录制 91 请显式提供 --anchors')
        o.anchors = video.SOURCE91
    return o


if __name__ == '__main__':
    try:
        main(parse_args())
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        sys.exit(f'多节点评估失败：{exc}')
