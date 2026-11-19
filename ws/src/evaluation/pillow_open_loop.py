"""CPU-side protocol for Mobile ALOHA: disjoint recordings, absolute actions, no ROS."""
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ws.src.common.pillow import TASK, VIEWS, jsonl, read_json

H = 24
FPS = 50
WIDTHS = {'left_joint_pos': 6, 'left_gripper_pos': 1, 'right_joint_pos': 6, 'right_gripper_pos': 1}


def slices(modality, kind):
    fields = modality[kind]
    result = {k: slice(fields[k]['start'], fields[k]['end']) for k in WIDTHS}
    if (set(fields) != set(WIDTHS) or sorted(i for s in result.values() for i in range(s.start, s.stop)) != list(range(14))
            or any(s.stop - s.start != WIDTHS[k] or not fields[k]['absolute'] or
                   fields[k]['original_key'] != ('observation.state' if kind == 'state' else 'action')
                   for k, s in result.items())):
        raise ValueError(f'Invalid compact absolute 14-D {kind} layout: {fields}')
    return result


@dataclass(frozen=True)
class Episode:
    id: int
    source: int
    source_start: int
    length: int
    task_id: int
    parquet: Path
    cameras: dict


class Holdout:
    def __init__(self, train, full=None, limit=None):
        meta = train / 'meta'
        split = read_json(meta / 'holdout.json')
        self.full = (full or Path(split['full_dataset'])).resolve(strict=True)
        fm = self.full / 'meta'
        train_ids, held = set(split['train_source_ids']), sorted(split['held_out_source_ids'])
        if not held or len(set(held)) != len(held) or train_ids.intersection(held):
            raise ValueError('Invalid train/holdout split')
        train_origins = jsonl(meta / 'source_episodes.jsonl')
        full_origins = jsonl(fm / 'source_episodes.jsonl')
        ids = [o['episode_index'] for o in full_origins if o['source_episode_index'] in held]
        if ({o['source_episode_index'] for o in train_origins} != train_ids
                or {o['source_episode_index'] for o in full_origins} != train_ids.union(held)
                or [o for o in full_origins if o['source_episode_index'] in train_ids] != train_origins
                or ids != split['held_out_episode_ids_in_full_dataset'] or len(set(ids)) != len(ids)):
            raise ValueError('holdout.json does not match training/full source_episodes.jsonl')
        self.modality = read_json(meta / 'modality.json')
        if self.modality != read_json(fm / 'modality.json'):
            raise ValueError('Different full and training modalities')
        self.state, self.action = (slices(self.modality, k) for k in ('state', 'action'))
        self.names = [f'{name}_{j}' for name, span in sorted(self.action.items(), key=lambda p: p[1].start)
                      for j in range(span.stop-span.start)]
        info = read_json(fm / 'info.json')
        video = {f'video.{k}': v['original_key'] for k, v in self.modality['video'].items()}
        if (info['fps'] != FPS or read_json(meta/'info.json')['fps'] != FPS or set(video.values()) != set(VIEWS)
                or any(k != 'video.' + v.removeprefix('observation.images.') for k, v in video.items())
                or read_json(meta/'embodiment.json')['embodiment_tag'] != 'yam'):
            raise ValueError('Expected the 50 Hz 3-camera YAM conversion')
        tasks = jsonl(fm / 'tasks.jsonl')
        if len(tasks) != 1 or tasks[0]['task'] != TASK or tasks != jsonl(meta / 'tasks.jsonl'):
            raise ValueError(f'Expected converted training task, not original pick_cup label: {tasks}')
        records = {e['episode_index']: e for e in jsonl(fm / 'episodes.jsonl')}
        self.episodes = []
        for o in full_origins:
            i, source = o['episode_index'], o['source_episode_index']
            if source not in held[:limit]:
                continue
            length = records[i]['length']
            if (length < 1 or length != o['source_frame_stop'] - o['source_frame_start']
                    or records[i]['tasks'] != [TASK] or o['task_index'] != tasks[0]['task_index']):
                raise ValueError(f'Invalid held-out segment or prompt: episode {i}')
            kw = dict(episode_index=i, episode_chunk=i // info['chunks_size'])
            pq = self.full / info['data_path'].format(**kw)
            cameras = {k: self.full / info['video_path'].format(video_key=v, **kw) for k, v in video.items()}
            for path in [pq, *cameras.values()]:
                if not path.is_file():
                    raise FileNotFoundError(path)
            self.episodes.append(Episode(i, source, o['source_frame_start'], length, o['task_index'], pq, cameras))
        if {e.source for e in self.episodes} != set(held[:limit]):
            raise ValueError('Not all selected held-out recordings have segments')
        stats = read_json(meta/'stats.json')['action']  # ABSOLUTE commands, training data only
        self.thresholds = {}
        for side in ('left', 'right'):
            i = self.action[side+'_gripper_pos'].start
            lo, hi = stats['q01'][i], stats['q99'][i]
            if not np.isfinite([lo, hi]).all() or hi <= lo:
                raise ValueError(f'No usable training gripper range for {side}: {lo}, {hi}')
            self.thresholds[side] = (lo + .35*(hi-lo), lo + .65*(hi-lo))


def read_rows(ep):
    import pyarrow.parquet as pq
    cols = ['observation.state', 'action', 'frame_index', 'episode_index', 'task_index', 'timestamp']
    with pq.ParquetFile(ep.parquet) as pf:
        tab = pf.read(columns=cols)
    state, action = (np.stack(tab[k].to_numpy()).astype(np.float32) for k in cols[:2])
    n = ep.length
    if (state.shape != (n, 14) or action.shape != state.shape or not np.isfinite(state).all()
            or not np.isfinite(action).all() or not np.array_equal(tab['frame_index'].to_numpy(), np.arange(n))
            or not np.all(tab['episode_index'].to_numpy() == ep.id) or not np.all(tab['task_index'].to_numpy() == ep.task_id)
            or not np.allclose(tab['timestamp'].to_numpy(), np.arange(n)/FPS, rtol=0, atol=2e-5)):
        raise ValueError(f'Invalid row/frame alignment: {ep.parquet}')
    return state, action


def anchor_images(ep, n):
    """Decode three RGB streams in presentation order. Only yield frame t, never t+1."""
    import av
    with ExitStack() as stack:
        readers = {}
        for k, p in ep.cameras.items():
            container = stack.enter_context(av.open(str(p)))
            stream, = container.streams.video
            stream.codec_context.thread_count = 1
            readers[k] = iter(container.decode(stream))
        for t in range(n):
            obs = {}
            for k, stream in readers.items():
                frame = next(stream, None)
                if frame is None or frame.pts is None or frame.pts * frame.time_base * FPS != t:
                    raise ValueError(f'Video/row mismatch: episode {ep.id}, {k}, frame {t}')
                if t % H == 0:
                    obs[k] = frame.to_ndarray(format='rgb24')[None, ...]
            if obs:
                yield t, obs
        if n == ep.length and any(next(r, None) is not None for r in readers.values()):
            raise ValueError(f'Extra video frames after row {n-1}: episode {ep.id}')


def action_chunk(act, layout):
    out = np.empty((H, 14), np.float64)
    for k, s in layout.items():
        x = act['action.' + k]
        if hasattr(x, 'detach'):
            x = x.detach().float().cpu().numpy()
        x = np.asarray(x)
        d = s.stop-s.start
        if x.shape == (1, H, d):
            x = x[0]
        if d == 1 and x.shape == (H,):  # official unbatched policy squeezes grippers
            x = x[:, None]
        if x.shape != (H, d) or not np.isfinite(x).all():
            raise ValueError(f'Invalid prediction for {k}: {x.shape} (expected {(H,d)} and finite)')
        out[:, s] = x
    return out


def metrics(pred, truth, layout):
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != 14 or len(pred) == 0:
        raise ValueError('Expected matching nonempty (frames, 14) arrays')
    err = np.asarray(pred, dtype=np.float64) - truth
    ret = {}
    for k, s in layout.items():
        e = err[:, s]
        joint = k.endswith('_joint_pos')
        name = k.replace('_joint_pos', '_arm').replace('_gripper_pos', '_gripper')
        unit = '_rad' if joint else ''
        ret[name+'_mae'+unit] = float(np.abs(e).mean())
        ret[name+'_rmse'+unit] = float(np.sqrt(np.mean(e**2)))
    return ret


def crossings(values, thresholds):
    """Hysteresis: initial observed state is not itself an event."""
    low, high = thresholds
    state = None
    events = {'open': [], 'close': []}
    for i, v in enumerate(values):
        new = 'open' if v >= high else 'close' if v <= low else state
        if state is not None and new != state:
            events[new].append(i)
        state = new
    return events


def match_events(gt, pred, tol):
    """Monotone, same-direction one-to-one matching: max matches, then min total |lag|."""
    # Track (match count, negative cost) lexicographically; keep only two score rows.
    m, n = len(gt), len(pred)
    choice = np.zeros((m, n), dtype=np.int8)
    below = [(0, 0)] * (n+1)
    for i in range(m-1, -1, -1):
        row = [(0, 0)] * (n+1)
        for j in range(n-1, -1, -1):
            score, move = (below[j], 1) if below[j] >= row[j+1] else (row[j+1], 2)
            delta = abs(pred[j]-gt[i])
            if delta <= tol:
                count, cost = below[j+1]
                pair = (count+1, cost-delta)
                if pair >= score:
                    score, move = pair, 3
            row[j], choice[i, j] = score, move
        below = row
    out = []
    i = j = 0
    while i < m or j < n:
        move = 2 if i == m else 1 if j == n else choice[i, j]
        out.append((gt[i] if move != 2 else None, pred[j] if move != 1 else None))
        i += int(move != 2)
        j += int(move != 1)
    return out
