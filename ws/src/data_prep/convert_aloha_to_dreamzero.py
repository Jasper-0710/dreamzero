#!/usr/bin/env python3
"""Convert aloha_record's embedded-JPEG LeRobot v2.1 data for this repo's yam_relative loader."""

import argparse
import fcntl
import json
import math
import shutil
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if not __package__:  # Also support launching this file by path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.common.pillow import ROOT
from ws.src.data_prep.aloha_video import jpeg_rows, write_video

# Reuse the converter in THIS checkout; its CLI alone cannot extract embedded images.
sys.path.insert(0, str(ROOT / 'scripts/data'))
import convert_lerobot_to_gear as gear


DATA = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
VIDEO = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
STATE, ACTION = 'observation.state', 'action'
HORIZON = 24  # yam_relative: state [0], actions [0..23], next video boundary [24].
JOINTS = [f'{side}_{joint}' for side in ('left', 'right') for joint in
          ('waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper')]
SLICES = {'left_joint_pos': [0, 6], 'left_gripper_pos': [6, 7],
          'right_joint_pos': [7, 13], 'right_gripper_pos': [13, 14]}
# Actual input column -> external video feature. gear.detect_features uses the short output names.
CAMERAS = {f'observation.images.{src}': f'observation.images.{dst}_camera-images-rgb'
           for src, dst in [('cam_high', 'top'), ('cam_left_wrist', 'left'), ('cam_right_wrist', 'right')]}


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def ep_path(root, pattern, idx, chunk_size, video_key=''):
    return root / pattern.format(episode_index=idx, episode_chunk=idx // chunk_size, video_key=video_key)


def timestamps(n, fps):
    # Match the recorder's Python i / fps -> Arrow float32 (not accumulated timestamps).
    return (np.arange(n, dtype=np.float64) / fps).astype(np.float32)


def source_metadata(root):
    meta = root / 'meta'
    info = json.loads((meta / 'info.json').read_text(encoding='utf-8'))
    eps, tasks = read_jsonl(meta / 'episodes.jsonl'), read_jsonl(meta / 'tasks.jsonl')
    if (info['codebase_version'], info['robot_type'], info['data_path']) != ('v2.1', 'mobile_aloha', DATA):
        raise ValueError('Expected the mobile_aloha v2.1 layout from aloha_record/hf_dataset_writer.py')
    if any(type(info[k]) is not int or info[k] <= 0 for k in ('fps', 'chunks_size')):
        raise ValueError('Source fps and chunks_size must be positive integers')
    if (not eps or len(eps) != info['total_episodes'] or
            [e['episode_index'] for e in eps] != list(range(len(eps))) or
            any(type(e['length']) is not int or e['length'] < 1 for e in eps) or
            sum(e['length'] for e in eps) != info['total_frames']):
        raise ValueError('episodes.jsonl and info.json disagree on episode indices/lengths')
    if (not tasks or len(tasks) != info['total_tasks'] or
            [t['task_index'] for t in tasks] != list(range(len(tasks))) or
            any(not isinstance(t['task'], str) or not t['task'].strip() for t in tasks)):
        raise ValueError('Invalid task table in tasks.jsonl')
    features = info['features']
    for col in (STATE, ACTION):
        f = features[col]
        if (f['dtype'], f['shape'], f['names']) != ('float32', [14], JOINTS):
            raise ValueError(f'{col}: expected [left 6 joints, left gripper, right 6 joints, right gripper]')
    for col, dtype in [('timestamp', 'float32'), ('recording.elapsed_time', 'float64'),
                       *((k, 'int64') for k in ('frame_index', 'episode_index', 'index', 'task_index',
                                                 'recording.segment_index'))]:
        if (features[col]['dtype'], features[col]['shape']) != (dtype, [1]):
            raise ValueError(f'{col}: expected a scalar {dtype}')
    if {k for k, f in features.items() if f['dtype'] in ('image', 'video')} != CAMERAS.keys():
        raise ValueError(f'Expected exactly the three recorded cameras: {list(CAMERAS)}')
    for key in CAMERAS:
        f = features[key]
        h, w, c = f['shape']
        if (f['dtype'] != 'image' or f['names'] != ['height', 'width', 'channels'] or c != 3 or
                min(w, h) <= 0 or w % 2 or h % 2):
            raise ValueError(f'{key}: expected embedded RGB images with even width and height for yuv420p')
    return info, eps, tasks


def lowdim(parquet, features):
    """Read numeric columns only. Return the unmodified Arrow table plus checked numpy views."""
    if set(parquet.schema_arrow.names) != set(features):
        raise ValueError('Parquet columns differ from meta/info.json features')
    table = parquet.read(columns=[k for k in features if k not in CAMERAS]).replace_schema_metadata(None)
    arrays = {}
    for k in table.column_names:
        col = table[k].combine_chunks()
        dim, = features[k]['shape']
        dtype = pa.from_numpy_dtype(features[k]['dtype'])
        expected = dtype if dim == 1 else pa.list_(dtype, dim)
        if col.type != expected or col.null_count:
            raise ValueError(f'{k}: expected non-null {expected}, got {col.type}')
        flat = col if dim == 1 else col.flatten()
        if flat.null_count:
            raise ValueError(f'{k}: null component')
        a = flat.to_numpy(zero_copy_only=False)
        if not np.isfinite(a).all():
            raise ValueError(f'{k}: NaN/Inf')
        arrays[k] = a if dim == 1 else a.reshape(len(table), dim)
    return table, arrays


def source_spans(a, episode, offset, fps, tasks, max_gap):
    """Validate the recorded timeline and split at the same discontinuities as replay."""
    n, idx = episode['length'], episode['episode_index']
    frames = np.arange(n, dtype=np.int64)
    if (not np.array_equal(a['frame_index'], frames) or not np.array_equal(a['index'], offset + frames) or
            not np.all(a['episode_index'] == idx) or not np.array_equal(a['timestamp'], timestamps(n, fps))):
        raise ValueError('Invalid source row count, frame/global/episode indices, or CFR timestamps')
    task = int(a['task_index'][0])
    if (not 0 <= task < len(tasks) or not np.all(a['task_index'] == task) or
            episode['tasks'] != [tasks[task]['task']]):
        raise ValueError('Expected one consistent source task per recorded episode')
    seg, elapsed = a['recording.segment_index'], a['recording.elapsed_time']
    ds, dt = np.diff(seg), np.diff(elapsed)
    if seg[0] < 0 or elapsed[0] < 0 or np.any(ds < 0) or np.any(dt <= 0):
        raise ValueError('Invalid recording.segment_index or non-increasing recording.elapsed_time')
    seg_starts = [0, *(np.flatnonzero(ds != 0) + 1).tolist()]
    if episode.get('segments') != [{'segment_index': int(seg[i]), 'start_frame': i} for i in seg_starts]:
        raise ValueError('Recorded segment boundaries disagree with episodes.jsonl')
    cuts = [0, *(np.flatnonzero((ds != 0) | (dt > max_gap)) + 1).tolist(), n]
    spans = []
    for lo, hi in zip(cuts, cuts[1:]):
        if hi - lo <= HORIZON:
            raise ValueError(f'frames [{lo}:{hi}) too short: need at least {HORIZON + 1}; '
                             'inspect the pause/gap rather than padding or joining unrelated actions')
        reason = 'episode_start' if lo == 0 else ('segment_change' if ds[lo - 1] else 'elapsed_gap')
        spans.append((lo, hi, int(seg[lo]), reason))
    return task, spans


def write_numeric(path, source, lo, hi, idx, offset, task, fps):
    """Do not reconstruct or cast any state/action. Only reindex segment-local coordinates."""
    n = hi - lo
    table = source.slice(lo, n)
    ids = np.arange(n, dtype=np.int64)
    for key, values in {'frame_index': ids, 'index': offset + ids, 'timestamp': timestamps(n, fps),
                        'episode_index': np.full(n, idx, np.int64), 'task_index': np.full(n, task, np.int64)}.items():
        table = table.set_column(table.schema.get_field_index(key), key, pa.array(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression='snappy')
    if not pq.ParquetFile(path).read().equals(table):
        raise ValueError(f'{path}: numeric read-back differs from the source slice plus declared index changes')


def relative_statistics(pairs):
    """Match LeRobotSingleDataset._calculate_relative_stats_for_key, including the last full window."""
    total = sum((len(s) - HORIZON + 1) * HORIZON for s, _ in pairs)
    out = {}
    for key, (first, end) in SLICES.items():
        stats = {k: [] for k in ('min', 'max', 'mean', 'std', 'q01', 'q99')}
        for j in range(first, end):
            # Exact quantiles, one coordinate at a time; no image or dataset-wide (N,H,14) tensor.
            pooled = np.empty(total, dtype=np.float32)
            pos = 0
            for state, action in pairs:
                windows = np.lib.stride_tricks.sliding_window_view(action[:, j], HORIZON)
                m = len(windows)
                target = pooled[pos:pos + windows.size].reshape(m, HORIZON)
                np.subtract(windows, state[:m, j, None], out=target)
                pos += windows.size
            q01, q99 = np.quantile(pooled, [0.01, 0.99])
            for name, val in [('min', pooled.min()), ('max', pooled.max()), ('mean', pooled.mean(dtype=np.float64)),
                              ('std', pooled.std(dtype=np.float64)), ('q01', q01), ('q99', q99)]:
                stats[name].append(float(val))
        out[key] = stats
    return out


def plan_tasks(root, info, episodes, tasks, task_text, confirm_same_task=False):
    """Inspect actual task references before encoding; unused historical labels are harmless."""
    if confirm_same_task and not task_text:
        raise ValueError('--confirm-same-task requires an explicit --task-text')
    usage = {}
    for ep in episodes:
        path = ep_path(root, DATA, ep['episode_index'], info['chunks_size'])
        with pq.ParquetFile(path) as pf:
            column = pf.read(columns=['task_index'])['task_index']
        if column.type != pa.int64() or column.null_count or len(column) != ep['length']:
            raise ValueError(f'{path}: invalid task_index type, nulls or row count')
        ids = np.unique(column.to_numpy()).tolist()
        if (len(ids) != 1 or not 0 <= ids[0] < len(tasks) or
                ep['tasks'] != [tasks[ids[0]]['task']]):
            raise ValueError(f'{path}: Expected one consistent source task per recorded episode')
        task = ids[0]
        usage[task] = usage.get(task, 0) + 1
    summary = [{'source_task_index': idx, 'source_task': tasks[idx]['task'], 'recordings': count}
               for idx, count in usage.items()]
    print(f'Actual source task usage ({len(episodes)} recordings): {summary!r}', flush=True)
    if task_text is not None and len(usage) > 1 and not confirm_same_task:
        raise ValueError(
            f'实际录制使用多个任务标签：{summary!r}。--task-text 不会自动合并多个任务；'
            '仅当确认这些录制实际上全部是同一任务时，同时传 --confirm-same-task '
            '（启动脚本也支持 --confirm_same_task）。若确有 pick_cup 等其他任务，请先分开数据。')
    mapping = {src: (0 if task_text is not None else dst) for dst, src in enumerate(usage)}
    output_tasks = ([{'task_index': 0, 'task': task_text}] if task_text is not None else
                    [{'task_index': dst, 'task': tasks[src]['task']} for src, dst in mapping.items()])
    if task_text is not None:
        print(f'Task in output only: {[tasks[i]["task"] for i in usage]!r} -> {task_text!r}; '
              f'confirmed_same_task={confirm_same_task}', flush=True)
    return mapping, output_tasks, summary


def output_info(source, episodes, output_tasks):
    fps, chunk = source['fps'], source['chunks_size']
    features = {k: v for k, v in source['features'].items() if k not in CAMERAS}
    for src, dst in CAMERAS.items():
        h, w, c = source['features'][src]['shape']
        features[dst] = {'dtype': 'video', 'shape': [h, w, c], 'names': ['height', 'width', 'channels'],
                         'info': {'video.height': h, 'video.width': w, 'video.channels': c, 'video.fps': fps,
                                  'video.codec': 'h264', 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False,
                                  'has_audio': False}}
    count = len(episodes)
    return {'codebase_version': 'v2.1', 'robot_type': source['robot_type'], 'fps': fps, 'chunks_size': chunk,
            'data_path': DATA, 'video_path': VIDEO, 'features': features, 'total_episodes': count,
            'total_frames': sum(e['length'] for e in episodes), 'total_tasks': len(output_tasks),
            'total_videos': count * len(CAMERAS), 'total_chunks': math.ceil(count / chunk), 'splits': {'train': f'0:{count}'}}


def convert(root, stage, args):
    info, all_episodes, tasks = source_metadata(root)
    fps, chunk = info['fps'], info['chunks_size']
    start = args.start_episode
    selected = all_episodes[start:None if args.limit is None else start + args.limit]
    if not selected:
        raise ValueError('No source episodes selected')
    task_map, output_tasks, task_usage = plan_tasks(
        root, info, selected, tasks, args.task_text, args.confirm_same_task)
    meta = stage / 'meta'
    meta.mkdir(parents=True)
    episodes, origins, paths, pairs = [], [], [], []
    offset, src_offset = 0, sum(e['length'] for e in all_episodes[:start])
    for ep in selected:
        src_id = ep['episode_index']
        src_path = ep_path(root, DATA, src_id, chunk)
        print(f'Source episode {src_id}: {ep["length"]} rows', flush=True)
        try:
            with pq.ParquetFile(src_path) as pf:
                table, arrays = lowdim(pf, info['features'])
                src_task, spans = source_spans(arrays, ep, src_offset, fps, tasks, args.max_gap)
                task = task_map[src_task]
                streams = {k: jpeg_rows(pf, k) for k in CAMERAS}
                for lo, hi, segment, reason in spans:
                    idx, n = len(episodes), hi - lo
                    path = ep_path(stage, DATA, idx, chunk)
                    write_numeric(path, table, lo, hi, idx, offset, task, fps)
                    for src, dst in CAMERAS.items():
                        h, w, _ = info['features'][src]['shape']
                        write_video(ep_path(stage, VIDEO, idx, chunk, dst), streams[src], fps, (w, h), n)
                    paths.append(path)
                    pairs.append((arrays[STATE][lo:hi], arrays[ACTION][lo:hi]))
                    episodes.append({'episode_index': idx, 'length': n, 'tasks': [args.task_text or tasks[src_task]['task']]})
                    origins.append({'episode_index': idx, 'source_episode_index': src_id, 'source_segment_index': segment,
                                    'source_frame_start': lo, 'source_frame_stop': hi, 'boundary': reason,
                                    'source_task_index': src_task, 'source_task': tasks[src_task]['task'], 'task_index': task})
                    offset += n
                if any(next(s, None) is not None for s in streams.values()):
                    raise ValueError('More image rows than numeric rows')
        except Exception as exc:
            raise ValueError(f'{src_path}: {exc}') from exc
        src_offset += ep['length']
    target = output_info(info, episodes, output_tasks)
    if target['total_frames'] != sum(e['length'] for e in selected):
        raise ValueError('Internal frame accounting mismatch')
    modality = gear.build_modality_json(target, gear.detect_features(target), SLICES, SLICES, task_key=None)
    # The loader resolves this numeric column via tasks.jsonl; no duplicated or stale string columns.
    modality['annotation'] = {'task': {'original_key': 'task_index'}}
    warnings = gear.validate_dataset(stage, target, modality)
    if warnings:
        raise ValueError('; '.join(warnings))
    write_json(meta / 'info.json', target)
    write_json(meta / 'modality.json', modality)
    write_json(meta / 'embodiment.json', {'embodiment_tag': 'yam'})
    write_jsonl(meta / 'tasks.jsonl', output_tasks)
    write_jsonl(meta / 'episodes.jsonl', episodes)
    write_jsonl(meta / 'source_episodes.jsonl', origins)
    write_json(meta / 'conversion.json', {'source_dataset': root.name, 'source_camera_to_output': CAMERAS,
                                         'source_fps': fps, 'max_gap_seconds': args.max_gap, 'action_horizon': HORIZON,
                                         'task_text': args.task_text, 'confirmed_same_task': args.confirm_same_task,
                                         'source_task_usage': task_usage,
                                         'task_mapping': [{'source_task_index': src, 'source_task': tasks[src]['task'],
                                                           'task_index': dst, 'task': output_tasks[dst]['task']}
                                                          for src, dst in task_map.items()]})
    # Official stats helper reads the NEW image-free parquets. Never run it on the huge embedded JPEG tables.
    floats = [k for k, f in target['features'].items() if f['dtype'].startswith('float')]
    write_json(meta / 'stats.json', gear.compute_stats(paths, floats))
    print(f'Computing {HORIZON}-step anchor-relative statistics (absolute actions stay on disk)', flush=True)
    write_json(meta / 'relative_stats_dreamzero.json', relative_statistics(pairs))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='committed aloha_record dataset root')
    parser.add_argument('--output', required=True, type=Path, help='NEW sibling/separate output directory')
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument('--task-text', help='output instruction; multiple used source labels also require --confirm-same-task')
    task.add_argument('--keep-task-text', action='store_true', help='explicitly keep recorded task text')
    parser.add_argument('--confirm-same-task', '--confirm_same_task', action='store_true',
                        help='confirm that ALL selected recordings are the same real task despite different source labels')
    parser.add_argument('--start-episode', type=int, default=0, help='first source episode ID (inclusive)')
    parser.add_argument('--limit', type=int, help='at most this many SOURCE episodes (default: all remaining)')
    parser.add_argument('--max-gap', type=float, default=0.25, help='split at elapsed gaps above this many seconds (replay default: 0.25)')
    args = parser.parse_args()
    if args.confirm_same_task and args.task_text is None:
        parser.error('--confirm-same-task requires --task-text; do not use it for genuinely different tasks')
    if args.start_episode < 0 or (args.limit is not None and args.limit < 1):
        parser.error('--start-episode must be nonnegative and --limit positive')
    if args.task_text is not None:
        args.task_text = args.task_text.strip()
        if not args.task_text:
            parser.error('--task-text cannot be empty')
    if not math.isfinite(args.max_gap) or args.max_gap <= 0:
        parser.error('--max-gap must be finite and positive')
    root = args.input.expanduser().resolve(strict=True)
    out = args.output.expanduser()
    if out.is_symlink() or out.exists():
        parser.error(f'Output already exists; no overwriting: {out}')
    dest = out.resolve()
    if dest == root or root in dest.parents or dest in root.parents:
        parser.error('Input and output must not overlap')
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            parser.error(f'{tool} not found in PATH (FFmpeg with libx264 required)')
    with ExitStack() as stack:
        lockpath = root / '.record.lock'
        if lockpath.exists():
            lock = stack.enter_context(lockpath.open('rb'))
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error('The recorder still owns this dataset; stop recording first')
        if (root / '.hf-pending.json').exists():
            parser.error('Uncommitted recorder journal: finish/recover the recording before converting')
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix=f'.{dest.name}.partial-', dir=dest.parent))
        stage = Path(tmp) / 'dataset'
        result = convert(root, stage, args)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f'Output appeared during conversion; not publishing: {dest}')
        stage.rename(dest)
    print(f'OK: {dest} | {result["total_episodes"]} episodes, {result["total_frames"]} rows, '
          f'{result["total_videos"]} checked MP4s @ {result["fps"]} Hz')


if __name__ == '__main__':
    main()
