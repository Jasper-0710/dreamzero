#!/usr/bin/env python3
"""Compare BF16 and INT8 weight storage for the same pillow checkpoint in shadow mode."""

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if not __package__:
    sys.path.insert(0, str(ROOT))
from ws.src.inference.pillow_client import post, recorded
from ws.src.inference.pillow_wire import pack

MODES = ('bf16', 'int8-weight-only')
DIT_STEPS = 8


def wait_ready(process, log, timeout):
    deadline = time.monotonic() + timeout
    with log.open(encoding='utf-8', errors='replace') as stream:
        while time.monotonic() < deadline:
            for line in stream:
                if line.startswith('READY: '):
                    _, _, gpu_name, address, _ = line.strip().split(' / ')
                    host, port = address.rsplit(':', 1)
                    if host != '127.0.0.1':
                        raise RuntimeError(f'Expected a loopback service, got {address}')
                    return gpu_name, int(port)
            if process.poll() is not None:
                raise RuntimeError(f'Service exited with code {process.returncode}. See {log}')
            time.sleep(0.5)
    raise TimeoutError(f'Service did not become ready within {timeout:g} s. See {log}')


def check_identity(reply, expected):
    actual = {key: reply.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f'Expected {expected}, got {actual}')


def run_mode(command, env, output, mode, identity, body, observation, state, count, timeout):
    log = output / f'{mode}.log'
    expected = dict(identity, weight_mode=mode)
    args = command + ['--port', '0', '--weight-mode', mode]
    print(f'{mode}: loading model, log {log}', flush=True)
    with log.open('xb') as stream, subprocess.Popen(args, env=env, stdout=stream, stderr=subprocess.STDOUT) as process:
        try:
            gpu_name, port = wait_ready(process, log, timeout)
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            try:
                conn.request('GET', '/health')
                response = conn.getresponse()
                if response.status != 200:
                    raise RuntimeError(f'Unexpected /health status {response.status}. See {log}')
                check_identity(json.load(response), dict(expected, action_shape=[24, 14]))
            finally:
                conn.close()

            rows = []
            with (output / f'{mode}.jsonl').open('x', encoding='utf-8') as data:
                for i in range(count):
                    start = time.perf_counter()
                    pred = post(port, body, expected['checkpoint'])
                    rtt_ms = 1000 * (time.perf_counter() - start)
                    check_identity(pred, expected)
                    row = dict(pred, source='recording', observation=observation, state=state, rtt_ms=rtt_ms)
                    data.write(json.dumps(row, allow_nan=False) + '\n')
                    data.flush()
                    rows.append(row)
                    print(f"{mode} [{i+1}/{count}] model={row['model_ms']:.0f} ms RTT={rtt_ms:.0f} ms "
                          f"peak allocated={row['cuda_peak_allocated_gib']:.2f} GiB "
                          f"reserved={row['cuda_peak_reserved_gib']:.2f} GiB", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    steps = [int(n) for n in re.findall(r'DIT Compute Steps (\d+) steps', log.read_text(errors='replace'))]
    if steps != [DIT_STEPS] * count:
        raise ValueError(f'Expected {count} predictions at {DIT_STEPS} DiT steps, got {steps}. See {log}')
    return gpu_name, rows, steps


def summarize(rows, steps):
    stats = dict(warm_count=len(rows)-1, dit_compute_steps=steps)
    for key in ('model_ms', 'rtt_ms'):
        p50, p95 = np.percentile([r[key] for r in rows[1:]], [50, 95])
        stats[key] = dict(p50=float(p50), p95=float(p95))
    for key in ('cuda_peak_allocated_gib', 'cuda_peak_reserved_gib'):
        stats[key] = max(r[key] for r in rows)
    return stats


def action_delta(bf16, int8):
    reference = np.asarray([r['action'] for r in bf16[1:]], dtype=np.float64)
    candidate = np.asarray([r['action'] for r in int8[1:]], dtype=np.float64)
    error = np.abs(candidate - reference)
    summary = {}
    for name, columns, unit in (
        ('left_arm', slice(0, 6), 'rad'), ('left_gripper', slice(6, 7), 'normalized'),
        ('right_arm', slice(7, 13), 'rad'), ('right_gripper', slice(13, 14), 'normalized'),
    ):
        values = error[..., columns]
        summary[name] = dict(mae=float(values.mean()), max_abs=float(values.max()), unit=unit)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True, help='一张已分配的空闲 GPU 编号')
    p.add_argument('--checkpoint', type=Path,
                   default=ROOT/'checkpoints/flip_150_lora_4000_swanlab_01/checkpoint-4000')
    p.add_argument('--train-data', type=Path, default=ROOT/'data/flip_150_dz_train')
    p.add_argument('--data', type=Path, default=ROOT/'data/flip_150')
    p.add_argument('--episode', type=int, help='默认取 holdout.json 中第一个 held_out_source_id')
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--count', type=int, default=6, help='每种模式的总请求数，含首个预热请求')
    p.add_argument('--startup-timeout', type=int, default=1200, help='每轮模型加载超时，单位秒')
    a = p.parse_args()
    if a.count < 2 or min(a.gpu, a.frame) < 0 or a.startup_timeout < 1:
        p.error('Require count >= 2, gpu/frame >= 0 and a positive startup timeout')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    if mask is not None and str(a.gpu) not in mask.replace(' ', '').split(','):
        p.error(f'GPU {a.gpu} is outside the current CUDA_VISIBLE_DEVICES={mask}')

    ckpt, train = a.checkpoint.resolve(strict=True), a.train_data.resolve(strict=True)
    held_out = json.loads((train/'meta/holdout.json').read_text())['held_out_source_ids']
    if not held_out:
        p.error('No held-out source IDs found')
    episode = held_out[0] if a.episode is None else a.episode
    if episode not in held_out:
        p.error(f'Episode {episode} is outside the held-out split')
    q, frames, observation = recorded(a.data, episode, a.frame)
    body = pack(q, frames)
    state = q.tolist()

    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(a.gpu),
               NUM_DIT_STEPS=str(DIT_STEPS), DYNAMIC_CACHE_SCHEDULE='False',
               ENABLE_TENSORRT='False', PYTHONUNBUFFERED='1')
    env.pop('LOAD_TRT_ENGINE', None)
    command = [sys.executable, str(ROOT/'ws/src/inference/serve_pillow.py'),
               '--checkpoint', str(ckpt), '--train-data', str(train)]
    subprocess.run(command + ['--check-only'], env=env, check=True)
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f'{ckpt.parent.name}_{ckpt.name}_precision_', dir=results))
    identity = dict(mode='shadow', checkpoint_run=ckpt.parent.name, checkpoint=ckpt.name, fps=50)
    print(f"{ckpt} on GPU {a.gpu}, episode {episode}, frame {a.frame}. Output {output}", flush=True)
    runs = {mode: run_mode(command, env, output, mode, identity, body, observation, state, a.count, a.startup_timeout)
            for mode in MODES}
    bf16_gpu, bf16, bf16_steps = runs['bf16']
    int8_gpu, int8, int8_steps = runs['int8-weight-only']
    if bf16_gpu != int8_gpu:
        raise ValueError(f'GPU mismatch: {bf16_gpu} versus {int8_gpu}')
    summary = dict(identity, checkpoint_path=str(ckpt), train_data=str(train), data=str(a.data.resolve()),
                   observation=observation, observation_sha256=hashlib.sha256(body).hexdigest(),
                   gpu_index=a.gpu, gpu_name=bf16_gpu, seed=1140, action_horizon_ms=480,
                   bf16=summarize(bf16, bf16_steps), int8_weight_only=summarize(int8, int8_steps),
                   action_delta_int8_vs_bf16=action_delta(bf16, int8))
    text = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    (output/'summary.json').write_text(text, encoding='utf-8')
    print(text, end='')
    print(f'Summary {output}/summary.json')


if __name__ == '__main__':
    main()
