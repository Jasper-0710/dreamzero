#!/usr/bin/env python3
"""Export held-out DreamZero RGB forecasts beside time-aligned recorded RGB. No robot, W&B or retraining."""
import argparse
import json
import os
import sys
import tempfile
from contextlib import ExitStack, nullcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.common.pillow import TASK, VIEWS
from ws.src.evaluation.pillow_eval_policy import clear_window_cache, independent_policy, recipe
from ws.src.evaluation.pillow_open_loop import FPS, H, Holdout, observation, read_rows, selected_images
from ws.src.evaluation.pillow_video_report import check_encoder, encode_mp4, page, side_by_side
from ws.src.evaluation.pillow_video_workers import assignments, run_workers, video_devices

# ShardedLeRobotSubLangSingleActionChunkDatasetDROID._uniform_sample_from_language_ranges:
# the eight offsets 0,3,...,21, then last_idx+3 at the end of the sequence.
# First causal call: 1 observed latent + 2 denoised latents; Wan2.1 VAE decodes 1+4*2 RGBs.
STRIDE = 3
OFFSETS = tuple(range(0, H + 1, STRIDE))
RATE = Fraction(FPS, STRIDE)
SOURCE91 = (168, 192, 216, 240, 264, 288)  # original 50 Hz frames; includes the post-flip window
CAMERAS = ['video.' + v.removeprefix('observation.images.') for v in VIEWS]


def check_video_recipe(ckpt, verified):
    import yaml
    cfg = yaml.safe_load((ckpt/'experiment_cfg/conf.yaml').read_text())  # resolved by Trainer
    inner = verified.config['action_head_cfg']['config']
    concat_orders = [t['video_concat_order'] for t in cfg['transforms']['yam']['transforms']
                     if t['_target_'] == 'groot.vla.data.transform.ConcatTransform']
    if (cfg['train_dataset']['dataset_class'] != 'groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetDROID'
            or cfg['modality_configs']['yam']['video']['modality_keys'] != CAMERAS
            or cfg['modality_configs']['yam']['video']['eval_delta_indices'] != [0]
            or concat_orders != [CAMERAS]
            or inner['num_frame_per_block'] != 2 or inner['diffusion_model_cfg']['num_action_per_block'] != H
            or inner['vae_cfg']['_target_'] != 'groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE'):
        raise ValueError('节点不是预期的单帧条件 YAM / Wan2.1、2 个视频 latent / 24 动作配置；拒绝猜测对齐。')


def select_windows(episodes, source, anchors):
    """Map original (not converted-local) frame IDs to complete windows within ONE segment each."""
    segments = [ep for ep in episodes if ep.source == source]
    if not segments:
        raise ValueError(f'原始录制 {source} 不在留出集中；可选 {sorted({e.source for e in episodes})}')
    if not anchors or list(anchors) != sorted(set(anchors)) or anchors[0] < 0:
        raise ValueError('请按递增顺序提供不重复的原始录制起始帧（从 0 计）')
    jobs = []
    for a in anchors:
        match = [ep for ep in segments if ep.source_start <= a and a + H < ep.source_start + ep.length]
        if len(match) != 1:
            spans = [(e.source_start, e.source_start + e.length - 1) for e in segments]
            raise ValueError(f'source {source}: [{a}, {a + H}] 不是段内完整窗口（含终点）；可用段 {spans}。不补帧、不跨暂停。')
        ep, = match
        if not jobs or jobs[-1][0].id != ep.id:
            jobs.append((ep, []))
        jobs[-1][1].append(a - ep.source_start)
    return jobs


def windows_rgb(ep, anchors):
    """Decode each selected row once, with PTS checks. Retain at most one rolling window of raw RGBs."""
    wanted = {a + d for a in anchors for d in OFFSETS}
    buf, j = {}, 0
    for row, views in selected_images(ep, max(wanted) + 1, wanted):
        buf[row] = views
        a = anchors[j]
        if row == a + H:
            yield a, [buf[a + d] for d in OFFSETS]
            j += 1
            buf = {i: v for i, v in buf.items() if j < len(anchors) and i >= anchors[j]}
    if j != len(anchors):
        raise ValueError(f'episode {ep.id}: missing video windows {j}/{len(anchors)}')


def decode_prediction(head, latent, force_tiled=False):
    """Official VAE decode and [-1,1] -> RGB conversion; never substitute/copy GT for the predicted half."""
    import torch
    if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or tuple(latent.shape[:3]) != (1, 16, 3):
        raise ValueError(f'应返回 Wan2.1 首窗口 latent [1,16,3,h,w]，实际 {getattr(latent, "shape", None)}')
    with torch.inference_mode():
        x = head.vae.decode(latent, tiled=force_tiled or head.tiled,
                            tile_size=(head.tile_size_height, head.tile_size_width),
                            tile_stride=(head.tile_stride_height, head.tile_stride_width))
        if x.ndim != 5 or tuple(x.shape[:3]) != (1, 3, len(OFFSETS)) or not torch.isfinite(x).all().item():
            raise ValueError(f'VAE 应解码 1 张条件重建 + 8 张未来 RGB（有限值）；实际 {tuple(x.shape)}。不截断/猜帧。')
        return ((x[0].permute(1, 2, 3, 0).float() + 1) * 127.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()


def recorded_grid(policy, obs):
    """Use the checkpoint's actual eval crop/resize/concat transform; NEVER invoke the model for GT."""
    from groot.vla.model.n1_5.sim_policy import unsqueeze_dict_values
    from tianshou.data import Batch
    # Mirror lazy_joint_forward_causal's batch-of-one path, stopping before the forward pass.
    x = policy.apply(Batch(obs=unsqueeze_dict_values(obs))).normalized_obs['images']
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 5 or x.shape[:2] != (1, 1) or x.shape[-1] != 3 or x.dtype != np.uint8:
        raise ValueError(f'Unexpected transformed RGB [1,1,h,w,3]: {x.shape}, {x.dtype}')
    return x[0, 0]


def paired_frames(jobs, layout, forecast, reference, step):
    for ep, anchors in jobs:
        state, _ = read_rows(ep)
        for a, views in windows_rgb(ep, anchors):
            src = ep.source_start + a
            print(f'[checkpoint-{step}] source={ep.source}, episode={ep.id}, 原始帧 {src}–{src+H}', flush=True)
            # This is the ONLY forecast call for the window. It takes only the 3 single-frame
            # images at a, state[a], and the task; no GT actions, future states or future images.
            pred = forecast(observation(views[0], state[a], layout))
            if pred.ndim != 4 or pred.shape[0] != len(OFFSETS) or pred.shape[-1] != 3 or pred.dtype != np.uint8:
                raise ValueError(f'Unexpected decoded RGB sequence: {pred.shape}, {pred.dtype}')
            for i, offset in enumerate(OFFSETS):
                # Only NOW prepare the reference using the saved eval transform. State stays fixed;
                # it is included only for the transform interface, not passed to a forecasting model.
                gt = reference(observation(views[i], state[a], layout))
                yield gt, pred[i], ep.source, src, offset


def comparison_frames(jobs, layout, forecast, reference, step):
    for gt, pred, source, anchor, offset in paired_frames(jobs, layout, forecast, reference, step):
        yield side_by_side(gt, pred, step, source, anchor, offset)


def manifest(jobs, ckpt, seed, tiled):
    windows = []
    for ep, anchors in jobs:
        for a in anchors:
            src = ep.source_start + a
            windows.append(dict(episode_index=ep.id, local_anchor=a, source_anchor=src,
                                source_frames=[src+d for d in OFFSETS], first_output_frame=len(windows)*len(OFFSETS)))
    return dict(checkpoint=str(ckpt), source_id=jobs[0][0].source, task=TASK, seed=seed, source_fps=FPS,
                output_fps=str(RATE), offsets=list(OFFSETS), frames_per_window=len(OFFSETS),
                frame_0='conditioning image (VAE reconstruction), not a forecast',
                protocol='independent open-loop windows, cache reset at every anchor, no robot execution',
                layout=[['top', 'right_wrist'], ['left_wrist', 'unused']], vae_tiled=bool(tiled),
                num_dit_steps=os.environ.get('NUM_DIT_STEPS', '8'), windows=windows)


def render_part(path, jobs, layout, ckpt, verified, o, *, load_slots=None, ready=None):
    """One full replica. Single-card exports stream to MP4; parallel workers spool lossless frames."""
    expected = sum(len(a) for _, a in jobs) * len(OFFSETS)
    with ExitStack() as stack:
        with (load_slots if load_slots is not None else nullcontext()):
            print(f'[GPU {os.environ["CUDA_VISIBLE_DEVICES"]}] 正在加载 {ckpt.name}', flush=True)
            policy, joint = stack.enter_context(independent_policy(ckpt, verified, o.seed))
        if ready is not None:
            print(f'[GPU {os.environ["CUDA_VISIBLE_DEVICES"]}] 权重就绪，等待其余 GPU。', flush=True)
            ready.wait()  # outside the load semaphore, so every worker can reach the barrier
        head = policy.trained_model.action_head

        def forecast(obs):
            _, latent = joint(obs)
            clear_window_cache(head)
            return decode_prediction(head, latent, o.tiled_decode)

        frames = comparison_frames(jobs, layout, forecast, lambda obs: recorded_grid(policy, obs), o.step)
        if ready is None:
            n = encode_mp4(path, frames, RATE)
        else:
            n = 0
            with path.open('xb') as f:
                for n, rgb in enumerate(frames, 1):
                    np.save(f, rgb, allow_pickle=False)
        if n != expected:
            raise ValueError(f'{path}: 应有 {expected} 帧，实际 {n}')
        return bool(o.tiled_decode or head.tiled)


def ordered_frames(parts):
    for path, expected in parts:  # assignment order, not worker completion order
        with path.open('rb') as f:
            for _ in range(expected):
                yield np.load(f, allow_pickle=False)
            if f.read(1):
                raise ValueError(f'发现多余的分片帧：{path}')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--run', type=Path, default=Path('checkpoints/flip_pillow_lora_01'))
    p.add_argument('--step', type=int, default=2000, help='只加载这一个 checkpoint')
    p.add_argument('--source-id', type=int, default=91, help='留出的原始录制 ID，不是转换后的 episode_index')
    p.add_argument('--anchors', type=int, nargs='+', help='原始录制帧号（0 起）。录制 91 默认 168 192 216 240 264 288')
    p.add_argument('--parallel-loads', type=int, default=1, help='最多同时加载权重的 GPU 数（仅影响加载，不限制推理并行度）')
    p.add_argument('--train-data', type=Path, default=Path('data/flip_the_pillow_dz_train'))
    p.add_argument('--full-data', type=Path, help='若移动过全量转换集，请指定新路径')
    p.add_argument('--base', type=Path, help='可选：原始 DreamZero-AgiBot 权重目录')
    p.add_argument('--seed', type=int, default=1140, help='与动作评估使用相同的固定种子')
    p.add_argument('--output', type=Path, help='新输出目录；默认 results/pillow_video_s<source>_ckpt<step>')
    p.add_argument('--tiled-decode', action='store_true', help='仅 VAE 解码使用官方空间分块；解码显存紧张时用')
    p.add_argument('--check-only', action='store_true', help='读权重头、Parquet、视频及 PTS，检查编码器；不加载模型/写文件')
    o = p.parse_args(argv)
    if o.parallel_loads < 1:
        p.error('--parallel-loads 必须大于 0')
    if o.step <= 0 or o.source_id < 0 or not 0 <= o.seed < 2**32:
        p.error('step 必须为正数、source-id 非负、seed 为 uint32')
    if o.anchors is None:
        if o.source_id != 91:
            p.error('非录制 91 请显式提供 --anchors（原始录制帧号）')
        o.anchors = SOURCE91
    if o.output is None:
        o.output = Path(f'results/pillow_video_s{o.source_id}_ckpt{o.step}')
    return o


def main(o):
    groups = assignments(o.anchors, video_devices(o.check_only))
    train, run = (p.expanduser().resolve(strict=True) for p in (o.train_data, o.run))
    data = Holdout(train, o.full_data and o.full_data.expanduser())
    jobs = select_windows(data.episodes, o.source_id, o.anchors)
    ckpt = run / f'checkpoint-{o.step}'
    verified = recipe(ckpt, train, data, o.base)
    check_video_recipe(ckpt, verified)
    check_encoder()
    print(f'source {o.source_id}，起始帧 {list(o.anchors)}；偏移 {list(OFFSETS)} / {FPS} Hz。', flush=True)
    for gpu, starts in groups:
        print(f'GPU {gpu} ← 原始起始帧 {starts}', flush=True)
    if o.check_only:
        for ep, anchors in jobs:
            read_rows(ep)
            for _ in windows_rgb(ep, anchors):
                pass
        print('视频预检通过：没有加载模型、运行推理或写入文件。')
        return
    out = o.output.expanduser()
    if out.is_symlink():
        raise ValueError(f'拒绝软链接输出：{out}')
    out = out.resolve()
    for inp in (train, data.full, run, ckpt.resolve(), *verified.protected_dirs):
        if inp == out or out in inp.parents or inp in out.parents:
            raise ValueError(f'输出不能与数据或模型目录重叠：{out} / {inp}')
    if out.exists():
        raise FileExistsError(f'输出已存在，请换新目录：{out}')
    # Stage in the same filesystem: a failed forecast never publishes a partial or stale video.
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix='.pillow-video-') as d:
        stage = Path(d)
        n = len(o.anchors) * len(OFFSETS)
        if len(groups) == 1:
            tiled = render_part(stage/'comparison.mp4', jobs, data.state, ckpt, verified, o)
        else:
            tasks, parts = [], []
            for i, (gpu, starts) in enumerate(groups):
                shard = select_windows(data.episodes, o.source_id, starts)
                part = stage/f'part-{i}.npyseq'
                tasks.append((gpu, (part, shard, data.state, ckpt, verified, o)))
                parts.append((part, len(starts) * len(OFFSETS)))
            modes = run_workers(render_part, tasks, o.parallel_loads)
            if len(set(modes)) != 1:
                raise ValueError('各 GPU 的 VAE 解码模式不一致，拒绝合并')
            tiled = modes[0]
            if encode_mp4(stage/'comparison.mp4', ordered_frames(parts), RATE) != n:
                raise ValueError(f'输出帧数应为 {n}')
        meta = manifest(jobs, ckpt, o.seed, tiled)
        meta['gpu_assignments'] = [{'device': d, 'anchors': a} for d, a in groups]
        (stage/'windows.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (stage/'index.html').write_text(page(meta), encoding='utf-8')
        out.mkdir(exist_ok=False)  # exclusive, even if another job just finished at this path
        for name in ('comparison.mp4', 'windows.json', 'index.html'):
            (stage/name).replace(out/name)
    print(f'完成：{out}/comparison.mp4（{n} 帧）；可在服务器此目录启动网页服务查看 index.html。')


if __name__ == '__main__':
    try:
        main(parse_args())
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        sys.exit(f'视频导出失败：{exc}')
