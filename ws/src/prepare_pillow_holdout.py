#!/usr/bin/env python3
"""Make a training-only view of converted episodes, reserving the last N whole source recordings."""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from convert_aloha_to_dreamzero import ACTION, STATE, ep_path, gear, relative_statistics, write_json, write_jsonl
from train_flip_pillow_lora import VIEWS, check_data, jsonl, read_json


def prepare(full, stage, count):
    # The converter emits episodes in source order. Holding out a suffix means the
    # remaining Parquet files already have the correct global and per-episode indices.
    meta = full / 'meta'
    info = read_json(meta / 'info.json')
    episodes = jsonl(meta / 'episodes.jsonl')
    origins = jsonl(meta / 'source_episodes.jsonl')
    ids = [row['source_episode_index'] for row in origins]
    sequential = list(range(len(episodes)))
    if (ids != sorted(ids) or [e['episode_index'] for e in episodes] != sequential
            or [row['episode_index'] for row in origins] != sequential):
        raise ValueError('需要转换器按录制顺序生成的连续 episode 和完整的 source_episodes.jsonl')
    recordings = sorted(set(ids))
    if not 1 <= count < len(recordings):
        raise ValueError(f'--holdout 必须在 1 到 {len(recordings) - 1} 之间')
    held = recordings[-count:]
    ntrain = ids.index(held[0])
    outmeta = stage / 'meta'
    outmeta.mkdir(parents=True)
    paths, pairs = [], []
    for ep in range(ntrain):
        parquet = ep_path(full, info['data_path'], ep, info['chunks_size'])
        files = [parquet] + [ep_path(full, info['video_path'], ep, info['chunks_size'], key) for key in VIEWS]
        for src in files:
            dest = stage / src.relative_to(full)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Use absolute links: the staged directory is renamed one level up at publish time.
            dest.symlink_to(src)
        paths.append(parquet)
        with pq.ParquetFile(parquet) as pf:
            table = pf.read(columns=[STATE, ACTION])
        pairs.append(tuple(np.stack(table[k].to_numpy()) for k in (STATE, ACTION)))

    for name in ('embodiment.json', 'modality.json', 'tasks.jsonl', 'conversion.json'):
        shutil.copyfile(meta / name, outmeta / name)
    write_jsonl(outmeta / 'episodes.jsonl', episodes[:ntrain])
    write_jsonl(outmeta / 'source_episodes.jsonl', origins[:ntrain])
    chunk = info['chunks_size']
    write_json(outmeta / 'info.json', dict(info, total_episodes=ntrain, total_frames=sum(e['length'] for e in episodes[:ntrain]),
               total_chunks=(ntrain + chunk - 1) // chunk, total_videos=ntrain * len(VIEWS), splits={'train': f'0:{ntrain}'}))
    columns = [k for k, f in info['features'].items() if f['dtype'].startswith('float')]
    write_json(outmeta / 'stats.json', gear.compute_stats(paths, columns))
    write_json(outmeta / 'relative_stats_dreamzero.json', relative_statistics(pairs))
    write_json(outmeta / 'holdout.json', {'full_dataset': str(full), 'train_source_ids': recordings[:-count],
               'held_out_source_ids': held, 'held_out_episode_ids_in_full_dataset': sequential[ntrain:]})
    check_data(stage)
    return ntrain, held


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('data/flip_the_pillow_dz'))
    p.add_argument('--output', type=Path, default=Path('data/flip_the_pillow_dz_train'))
    p.add_argument('--holdout', type=int, default=10, help='预留最后多少条原始录制（包括它们的所有切段），默认 10')
    opt = p.parse_args()
    full, out = opt.input.expanduser().resolve(strict=True), opt.output.expanduser().absolute()
    if out.exists() or out.is_symlink():
        p.error(f'输出已存在，不覆盖：{out}')
    out = out.resolve()
    if full == out or full in out.parents or out in full.parents:
        p.error('输入、输出目录不能重叠')
    check_data(full)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=f'.{out.name}-') as tmp:
        stage = Path(tmp) / 'train'
        ntrain, held = prepare(full, stage, opt.holdout)
        if out.exists() or out.is_symlink():
            p.error(f'输出已存在，不覆盖：{out}')
        stage.rename(out)
    print(f'训练集：{out}（{ntrain} 个输出 episode）；独立留出的原始录制 ID：{held}。请保留全量目录不动。')


if __name__ == '__main__':
    main()
