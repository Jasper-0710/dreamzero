"""Pillow task metadata and read-only dataset / model file checks (stdlib only)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = 'Flip the pillow over so the side with the red tape faces up.'
VIEWS = [f'observation.images.{v}_camera-images-rgb' for v in ('top', 'left', 'right')]
WAN_FILES = ('models_t5_umt5-xxl-enc-bf16.pth', 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth', 'Wan2.1_VAE.pth')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def require_file(path):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f'文件缺失或为空：{path}')


def check_data(root):
    meta = root / 'meta'
    info = read_json(meta / 'info.json')
    features = info['features']
    if (info['fps'] != 50 or any(features[k]['shape'] != [14] for k in ('observation.state', 'action'))
            or set(k for k, v in features.items() if v['dtype'] == 'video') != set(VIEWS)
            or read_json(meta / 'embodiment.json')['embodiment_tag'] != 'yam'):
        raise ValueError(f'{root} 不是本任务的 50 Hz、双臂 14 维、三视角 YAM 接口数据')
    tasks = jsonl(meta / 'tasks.jsonl')
    if len(tasks) != 1 or tasks[0]['task'] != TASK:
        raise ValueError(f'期望任务 {TASK!r}，实际 {tasks!r}；请检查所选数据目录')
    for name in ('modality.json', 'stats.json', 'relative_stats_dreamzero.json'):
        require_file(meta / name)
    episodes = jsonl(meta / 'episodes.jsonl')
    if (not episodes or len(episodes) != info['total_episodes']
            or sum(e['length'] for e in episodes) != info['total_frames']
            or len(VIEWS) * len(episodes) != info['total_videos']):
        raise ValueError('info.json 的轨迹数、总帧数或视频数与 episodes.jsonl 不符')
    for ordinal, ep in enumerate(episodes):
        idx = ep['episode_index']
        if idx != ordinal or ep['tasks'] != [TASK]:
            raise ValueError(f'episodes.jsonl 第 {ordinal + 1} 行的 ID 或任务不符')
        fields = dict(episode_index=idx, episode_chunk=idx // info['chunks_size'])
        require_file(root / info['data_path'].format(**fields))
        for view in VIEWS:
            require_file(root / info['video_path'].format(video_key=view, **fields))
    print(f"数据：{root}\n{len(episodes)} episodes，{info['total_frames']:,} 帧，{info['total_videos']} 个 MP4，50 Hz\n任务：{TASK}", flush=True)


def check_weights(base, wan, tokenizer):
    index = base / 'model.safetensors.index.json'
    files = ([base / name for name in sorted(set(read_json(index)['weight_map'].values()))]
             if index.is_file() else [base / 'model.safetensors'])
    if not files:
        raise ValueError(f'空的权重分片索引：{index}')
    files += [wan / name for name in WAN_FILES] + [tokenizer / 'tokenizer_config.json']
    files.append(tokenizer / ('tokenizer.json' if (tokenizer / 'tokenizer.json').is_file() else 'spiece.model'))
    missing = []
    for path in files:
        if not path.is_file() or not path.stat().st_size:
            missing.append(str(path))
        else:
            with path.open('rb') as src:
                if src.read(43).startswith(b'version https://git-lfs.github.com/spec/'):
                    missing.append(f'{path}（仅 Git LFS 指针）')
    if missing:
        raise ValueError('未找到完整本地权重/tokenizer（AgiBot 支持 model.safetensors 或索引及全部分片）；'
                         '按 ws/src/README.md 准备或指定路径：\n  ' + '\n  '.join(missing))
    print(f'AgiBot: {base}\nWan: {wan}\nTokenizer: {tokenizer}', flush=True)
