#!/usr/bin/env python3
"""Independently compare a converted episode with its recorded rows and JPEGs (read only)."""

import argparse
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.data_prep.aloha_video import check_video
from ws.src.data_prep.convert_aloha_to_dreamzero import ACTION, CAMERAS, SLICES, STATE, ep_path, read_jsonl, timestamps


def decoded_samples(path, indices, width, height):
    # 'n' is the presentation-order frame number, not an imprecise seek in seconds.
    select = 'select=' + '+'.join(f'eq(n\\,{i})' for i in indices)
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-filter_threads', '2', '-threads', '2',
           '-i', str(path), '-map', '0:v:0', '-vf', select, '-vsync', '0', '-pix_fmt', 'rgb24',
           '-threads', '1', '-f', 'rawvideo', 'pipe:1']
    pixels = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    if len(pixels) != len(indices) * width * height * 3:
        raise ValueError(f'{path}: decoded byte count differs from requested sample count')
    return np.frombuffer(pixels, np.uint8).reshape(len(indices), height, width, 3)


def compare(args):
    src, dst, ep = args.input, args.output, args.episode
    src_info = json.loads((src / 'meta/info.json').read_text())
    info = json.loads((dst / 'meta/info.json').read_text())
    mod = json.loads((dst / 'meta/modality.json').read_text())
    origin = {r['episode_index']: r for r in read_jsonl(dst / 'meta/source_episodes.jsonl')}[ep]
    src_ep, lo, hi = (origin[k] for k in ('source_episode_index', 'source_frame_start', 'source_frame_stop'))
    length, fps, chunk = hi - lo, info['fps'], info['chunks_size']
    episodes = read_jsonl(dst / 'meta/episodes.jsonl')
    episode = episodes[ep]
    offset = sum(e['length'] for e in episodes[:ep])
    if fps != src_info['fps'] or length <= 0:
        raise ValueError('Invalid provenance or different source/target control frequencies')
    src_path = ep_path(src, src_info['data_path'], src_ep, src_info['chunks_size'])
    dst_path = ep_path(dst, info['data_path'], ep, chunk)
    with pq.ParquetFile(src_path) as raw, pq.ParquetFile(dst_path) as converted:
        numeric = converted.read()
        frames = np.arange(length)
        if (episode['episode_index'] != ep or len(numeric) != episode['length'] or len(numeric) != length or
                not np.all(numeric['episode_index'].to_numpy() == ep) or
                not np.array_equal(numeric['frame_index'].to_numpy(), frames) or
                not np.array_equal(numeric['index'].to_numpy(), offset + frames)):
            raise ValueError('Output episode/frame/global indices or length disagree with metadata')
        if not np.array_equal(numeric['timestamp'].to_numpy(), timestamps(length, fps)):
            raise ValueError('Parquet timestamps differ from the MP4 frame grid')
        for family, column in [('state', STATE), ('action', ACTION)]:
            for key, (a, b) in SLICES.items():
                m = mod[family][key]
                if (m['original_key'], m['start'], m['end'], m['absolute']) != (column, a, b, True):
                    raise ValueError(f'Unexpected {family}.{key} mapping')
        fixed = [k for k in src_info['features'] if k not in CAMERAS and
                 k not in ('frame_index', 'index', 'episode_index', 'timestamp', 'task_index')]
        original = raw.read(columns=fixed).slice(lo, length)
        if not original.equals(numeric.select(fixed)):
            raise ValueError('At least one source state/action/diagnostic row is not copied unchanged')
        langcol = mod['annotation']['task']['original_key']
        ids = np.unique(numeric[langcol].to_numpy())
        if ids.tolist() != [origin['task_index']]:
            raise ValueError('Language does not resolve to the declared task_index')
        tasks = {t['task_index']: t['task'] for t in read_jsonl(dst / 'meta/tasks.jsonl')}
        if episode['tasks'] != [tasks[int(ids[0])]]:
            raise ValueError('Episode instruction and tasks.jsonl disagree')
        print(f'Output ep {ep} = source ep {src_ep}, rows [{lo}:{hi}), {length} rows @ {fps} Hz')
        print(f'Task: {origin["source_task"]!r} -> {tasks[int(ids[0])]!r}; all state/action/diagnostic rows match.')
        sample = np.unique(np.linspace(0, length - 1, min(args.samples, length), dtype=int)).tolist()
        for source_key, expected_video_key in CAMERAS.items():
            logical_key = expected_video_key.removeprefix('observation.images.')
            video_key = mod['video'][logical_key]['original_key']
            if video_key != expected_video_key:
                raise ValueError(f'Unexpected camera mapping for {logical_key}: {video_key}')
            path = ep_path(dst, info['video_path'], ep, chunk, video_key)
            h, w, _ = src_info['features'][source_key]['shape']
            check_video(path, fps, (w, h), length)
            rgb = decoded_samples(path, sample, w, h)
            jpegs = raw.read(columns=[source_key])[source_key]
            errors = []
            for frame_no, pixels in zip(sample, rgb):
                with Image.open(io.BytesIO(jpegs[lo + frame_no].as_py()['bytes'])) as image:
                    ref = np.asarray(image.convert('RGB'), dtype=np.int16)
                errors.append(float(np.abs(ref - pixels.astype(np.int16)).mean()))
            worst = int(np.argmax(errors))
            print(f'{logical_key}: {length} PTS checked; {len(sample)} JPEG/MP4 pairs; '
                  f'worst mean RGB error {errors[worst]:.2f}/255 at frame {sample[worst]}')
            if errors[worst] > args.max_mae:
                raise ValueError('Picture mismatch: check camera identity, color and frame order')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='original dataset')
    p.add_argument('--output', type=Path, required=True, help='already converted dataset (read only)')
    p.add_argument('--episode', type=int, default=0, help='OUTPUT episode ID')
    p.add_argument('--samples', type=int, default=5, help='evenly spaced RGB comparisons per camera')
    p.add_argument('--max-mae', type=float, default=12.0, help='per sampled image, in 8-bit RGB levels')
    opts = p.parse_args()
    if opts.samples < 1 or not math.isfinite(opts.max_mae) or opts.max_mae < 0:
        p.error('Need positive --samples and finite nonnegative --max-mae')
    compare(opts)
