#!/usr/bin/env python3
"""Localhost-only, single-GPU, independent-window Mobile ALOHA shadow inference (no actuation)."""
import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import time
import traceback

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ws.src.common.pillow import TASK, VIEWS, jsonl, read_json
from ws.src.evaluation.pillow_eval_policy import independent_policy, recipe, single_gpu_mask
from ws.src.evaluation.pillow_open_loop import FPS, H, WIDTHS, action_chunk, observation, slices
from ws.src.inference.pillow_wire import MIME, NAMES, SHAPE, SIZE, unpack


@dataclass
class Layout:
    state: dict
    action: dict


def train_layout(train):
    # No episodes or holdout video needed; retain the offline loader's cross-check of the normalizers.
    m = train / 'meta'
    mod, info = read_json(m/'modality.json'), read_json(m/'info.json')
    features = info['features']
    if (info['fps'] != FPS or read_json(m/'embodiment.json')['embodiment_tag'] != 'yam'
            or [r['task'] for r in jsonl(m/'tasks.jsonl')] != [TASK]
            or mod['video'] != {v.split('.')[-1]: {'original_key': v} for v in VIEWS}
            or any(features[v]['dtype'] != 'video' or tuple(features[v]['shape']) != SHAPE for v in VIEWS)
            or any(features[k]['names'] != NAMES or features[k]['shape'] != [14] for k in ('observation.state', 'action'))):
        raise ValueError('Expected the training metadata for the 50-Hz, 3-camera red-tape pillow task')
    layout = Layout(slices(mod, 'state'), slices(mod, 'action'))
    # slices() permits re-ordered layouts in generic eval. The client wire format is fixed.
    ofs = 0
    for key, width in WIDTHS.items():
        s = slice(ofs, ofs + width)
        if layout.state[key] != s or layout.action[key] != s:
            raise ValueError('Training layout is not left arm, left gripper, right arm, right gripper')
        ofs += width
    return layout


def infer(q, frames, layout, predict):
    images = {'video.' + v.split('.')[-1]: frame[None] for v, frame in zip(VIEWS, frames)}
    batch, _ = predict(observation(images, q, layout.state))
    return action_chunk(batch.act, layout.action)  # official unapply() already restored absolute actions


def handler(run, checkpoint):
    class Handler(BaseHTTPRequestHandler):
        timeout = 20  # bound partial network reads, not inference time

        def reply(self, code, result):
            body = json.dumps(result, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                return self.send_error(404)
            self.reply(200, dict(mode='shadow', checkpoint=checkpoint, task=TASK, fps=FPS, action_shape=[H, 14]))

        def do_POST(self):
            if self.path != '/predict':
                return self.send_error(404)
            if (self.headers.get('Content-Type') != MIME or self.headers.get('Transfer-Encoding') is not None
                    or self.headers.get_all('Content-Length') != [str(SIZE)]):
                return self.send_error(400, 'Expected one fixed-size pillow observation')
            try:
                q, frames = unpack(self.rfile.read(SIZE))
            except (ValueError, OSError) as exc:
                return self.send_error(400, str(exc))
            try:
                action, ms, peak = run(q, frames)
            except Exception:
                traceback.print_exc()
                return self.send_error(500, 'Inference failed; inspect server log')
            self.reply(200, dict(mode='shadow', checkpoint=checkpoint, fps=FPS, action_names=NAMES,
                                 action=action.tolist(), model_ms=ms, cuda_peak_reserved_gib=peak))
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/flip_pillow_lora_01/checkpoint-1600'))
    p.add_argument('--train-data', type=Path, default=Path('data/flip_the_pillow_dz_train'), help='只读 meta/，不需要视频')
    p.add_argument('--base', type=Path, help='DreamZero-AgiBot 基座，不是 DROID 或 LoRA')
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if not 1 <= a.port <= 65535:
        p.error('port must be 1..65535')
    ckpt, train = a.checkpoint.expanduser().resolve(strict=True), a.train_data.expanduser().resolve(strict=True)
    layout = train_layout(train)
    r = recipe(ckpt, train, layout, a.base)
    if a.check_only:
        print('部署预检通过：训练统计、布局和权重文件头已核对；未加载 GPU、未启动服务。')
        return
    single_gpu_mask()
    import torch
    with independent_policy(ckpt, r, 1140) as (_, predict):
        def run(q, frames):
            torch.cuda.synchronize()
            start = time.perf_counter()
            act = infer(q, frames, layout, predict)
            torch.cuda.synchronize()
            return act, (time.perf_counter() - start) * 1000, torch.cuda.max_memory_reserved() / 2**30
        # Single-threaded: the model's cache/seed are reset per prediction; no shared concurrent calls.
        with HTTPServer(('127.0.0.1', a.port), handler(run, ckpt.name)) as server:
            print(f'READY: {ckpt.name} / {torch.cuda.get_device_name(0)} / 127.0.0.1:{a.port} / shadow only', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
