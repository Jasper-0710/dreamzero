"""Window-parallel video export. The coordinator never loads a CUDA model."""
import multiprocessing as mp
import os
import re
from multiprocessing.connection import wait
from unittest.mock import patch


def video_devices(check_only):
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if os.environ.get('WORLD_SIZE', '1') != '1' or 'LOCAL_RANK' in os.environ:
        raise ValueError('请直接用 python 启动视频导出（不要用 torchrun）。')
    if check_only and mask in ('', '-1'):
        return ()  # CPU-only data preflight
    devices = tuple(d.strip() for d in mask.split(','))
    if (not all(re.fullmatch(r'0|[1-9][0-9]*|GPU-[0-9a-fA-F-]+|MIG-(?:GPU-)?[0-9a-fA-F/-]+', d)
                for d in devices) or len(set(devices)) != len(devices)):
        raise ValueError('请设置 CUDA_VISIBLE_DEVICES 为不重复的已分配 GPU，例如 1,2,3,4,5,6。')
    if not check_only and 'LOAD_TRT_ENGINE' in os.environ:
        raise ValueError('请先 unset LOAD_TRT_ENGINE，以免复用其他模型的 TensorRT 权重。')
    return devices


def assignments(anchors, devices):
    if not devices:
        return []
    if len(anchors) < len(devices):
        raise ValueError(f'{len(anchors)} 个窗口不足以分给 {len(devices)} 张卡；请增加有效起始帧或减少卡数。')
    q, r = divmod(len(anchors), len(devices))
    start, result = 0, []
    for i, gpu in enumerate(devices):
        stop = start + q + (i < r)
        result.append((gpu, list(anchors[start:stop])))
        start = stop
    return result


def _run_one(render, args, idx, tiled, load_slots, ready):
    tiled[idx] = int(render(*args, load_slots=load_slots, ready=ready))


def run_workers(render, tasks, parallel_loads):
    """Pin each fresh interpreter before CUDA import; kill waiting siblings if any worker fails."""
    ctx = mp.get_context('spawn')
    n = len(tasks)
    slots, ready = ctx.BoundedSemaphore(min(n, parallel_loads)), ctx.Barrier(n)
    tiled = ctx.Array('b', [-1] * n, lock=False)
    procs = []
    try:
        for i, (gpu, args) in enumerate(tasks):
            p = ctx.Process(name=f'pillow-gpu-{gpu}', target=_run_one, args=(render, args, i, tiled, slots, ready))
            # Only the coordinator starts children; restrict visibility *before* importing their main module.
            env = {'CUDA_VISIBLE_DEVICES': gpu}
            for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                env[k] = os.environ.get(k) or '1'
            with patch.dict(os.environ, env):
                p.start()
            procs.append((gpu, p))
        pending = {p.sentinel: (gpu, p) for gpu, p in procs}
        while pending:
            for handle in wait(pending):
                gpu, p = pending.pop(handle)
                p.join()
                if p.exitcode:
                    raise RuntimeError(f'GPU {gpu} 进程退出码 {p.exitcode}，未发布视频；请查看该子进程日志。')
        if any(x not in (0, 1) for x in tiled):
            raise RuntimeError('子进程没有返回有效的解码信息')
        return list(tiled)
    finally:
        for _, p in procs:
            if p.is_alive():
                p.terminate()
        for _, p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
            p.close()
