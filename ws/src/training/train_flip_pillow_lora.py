#!/usr/bin/env python3
"""Check or launch DreamZero's YAM LoRA recipe for the converted 50 Hz Mobile ALOHA dataset."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
import sys
from pathlib import Path

if not __package__:  # Support python ws/src/training/train_flip_pillow_lora.py as well as python -m.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.common.pillow import ROOT, WAN_FILES, check_data, check_weights


def configure_swanlab(output=None):
    """Configure HF's native callback without authenticating or creating a run."""
    try:
        import swanlab  # noqa: F401 -- no run or GPU created in this process.
    except ImportError as exc:
        raise ValueError('当前 Python 无法导入 swanlab；请在服务器 dreamzero 环境运行 '
                         'python -m pip install swanlab') from exc
    # Native SwanLab reporting was added in transformers 4.50.0. Do not import
    # the full Transformers/Torch stack merely to check its installed version.
    from packaging.version import Version
    try:
        installed = version('transformers')
    except PackageNotFoundError as exc:
        raise ValueError('未安装 transformers；请按项目依赖安装 transformers==4.51.3') from exc
    if Version(installed) < Version('4.50.0'):
        raise ValueError(f'当前 transformers={installed} 不支持原生 SwanLab 上报；'
                         '本项目要求 transformers==4.51.3，请不要直接升级到最新版')
    mode = os.environ.get('SWANLAB_MODE', 'online').strip().lower()
    if mode not in ('online', 'offline'):
        raise ValueError('SwanLab 训练上报需要 online 或 offline；请通过启动器 --swanlab_mode 选择')
    project = os.environ.get('SWANLAB_PROJECT', '').strip() or 'dreamzero-flip-pillow'
    os.environ.update(SWANLAB_PROJECT=project, SWANLAB_PROJ_NAME=project, SWANLAB_MODE=mode,
                      SWANLAB_DISABLE_GIT='True', SWANLAB_PROBE_GIT='false')
    # HF 4.51.3 checks `is not None`, so setting this to 'false' is NOT sufficient.
    os.environ.pop('SWANLAB_LOG_MODEL', None)
    if output is not None:
        logdir = str(Path(output) / 'swanlog')
        # Legacy and current SDK spellings; keep logs next to local checkpoints.
        os.environ.update(SWANLAB_LOGDIR=logdir, SWANLAB_LOG_DIR=logdir)
    print(f'SwanLab 项目：{project}；模式={mode}；不上传 checkpoint。', flush=True)
    if mode == 'offline':
        print('当前为离线记录，不会生成实时云端网页曲线。', flush=True)
    return project


def reporting_options(backend, *, output=None):
    if backend == 'swanlab':
        configure_swanlab(output)
        # Only one native callback: do NOT fall back to W&B or report_to=all.
        # wandb_project is still a required field in DreamZero's base Hydra config.
        return ['report_to=swanlab', 'wandb_project="dreamzero"']
    project = os.environ.get('WANDB_PROJECT', '').strip() or 'dreamzero'
    if backend == 'wandb':
        mode = os.environ.get('WANDB_MODE', 'online').lower()
        if mode == 'disabled' or os.environ.get('WANDB_DISABLED', '').upper() in {'1', 'TRUE', 'YES', 'ON'}:
            raise ValueError('已选 W&B，但环境禁用了上报。请 unset WANDB_DISABLED，并设置 WANDB_MODE=online 或 offline')
        try:
            import wandb  # noqa: F401 -- verify the SDK before starting workers; do not create a run here.
        except ImportError as exc:
            raise ValueError(f'当前 Python 无法导入 wandb：{exc}。若未安装，请在服务器的 dreamzero 环境运行 python -m pip install wandb') from exc
        os.environ['WANDB_PROJECT'] = project  # Keep the SDK and the saved Hydra config in agreement.
        os.environ['WANDB_LOG_MODEL'] = 'false'  # Keep model artifacts/checkpoints on the server.
        os.environ['WANDB_WATCH'] = 'false'      # Scalar grad_norm is still logged by Trainer.
        os.environ['WANDB_DISABLE_GIT'] = 'true'  # No repository patch or automatic source snapshot.
        os.environ['WANDB_DISABLE_CODE'] = 'true'
        print(f'W&B 项目：{project}；WANDB_MODE={mode}；不上传模型。', flush=True)
        if mode in ('offline', 'dryrun'):
            print('当前为离线记录，不会生成实时网页曲线。', flush=True)
    return [f'report_to={backend}', f'wandb_project={json.dumps(project, ensure_ascii=False)}']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'smoke', 'train'], help='check 只做预检，不训练或下载')
    p.add_argument('--data', type=Path, default=os.environ.get('YAM_DATA_ROOT', ROOT / 'data/flip_the_pillow_dz_train'),
                   help='默认使用 YAM_DATA_ROOT 或留出之后的训练目录，而不是全量数据')
    p.add_argument('--output', type=Path, help='新运行的输出目录；默认 checkpoints/flip_pillow_lora_<模式>')
    p.add_argument('--steps', type=int, help='默认 smoke=10，train=1000')
    p.add_argument('--save-steps', type=int, help='默认 smoke=最后一步，train=每 200 步（不超过总步数）')
    p.add_argument('--batch-size', type=int, default=1, help='每张卡的 batch，默认 1')
    p.add_argument('--report-to', choices=['none', 'wandb', 'swanlab'], default='none',
                   help='默认 none 只写本地日志；wandb / swanlab 使用 Trainer 原生上报；推荐通过 launch_pillow_training.sh 设置项目参数')
    args = p.parse_args()
    steps = args.steps if args.steps is not None else (1000 if args.mode == 'train' else 10)
    save = args.save_steps if args.save_steps is not None else (min(200, steps) if args.mode == 'train' else steps)
    if not (1 <= save <= steps and args.batch_size >= 1):
        p.error('需要 1 <= save-steps <= steps，且 batch-size >= 1')

    data = args.data.expanduser().resolve()
    base, wan, tokenizer = [Path(os.environ.get(var, ROOT / 'checkpoints' / name)).expanduser().resolve() for var, name in
                            [('DZ_CKPT_DIR', 'DreamZero-AgiBot'), ('WAN_CKPT_DIR', 'Wan2.1-I2V-14B-480P'), ('TOKENIZER_DIR', 'umt5-xxl')]]
    output = (args.output or ROOT / 'checkpoints' / f'flip_pillow_lora_{args.mode}').expanduser().resolve()
    reporting = reporting_options(args.report_to, output=output)
    if not data.is_dir():
        p.error(f'找不到训练集：{data}。先运行 ws/src/data_prep/prepare_pillow_holdout.py；若另有独立测试数据，可显式用 --data 指向全量目录')
    check_data(data)
    check_weights(base, wan, tokenizer)
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not mask:
        p.error('请先激活服务器环境并显式设置 CUDA_VISIBLE_DEVICES 为分配给你的卡；不会自动占用所有 GPU')
    import torch
    visible = torch.cuda.device_count()
    procs = int(os.environ.get('NUM_GPUS', visible))
    if not 1 <= procs <= visible:
        p.error(f'NUM_GPUS={procs}，但 PyTorch 在 CUDA_VISIBLE_DEVICES={mask} 中只看到 {visible} 张卡')
    print(f'Python: {sys.executable}\nPyTorch: {torch.__version__}，CUDA_VISIBLE_DEVICES={mask}，训练进程数={procs}', flush=True)
    if args.mode == 'check':
        print('预检通过。尚未加载模型或启动训练。')
        return
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        p.error(f'输出目录非空：{output}；请用 --output 选一个新目录，避免误续训或跳过训练')

    # Mirror the architecture/LoRA flags from scripts/train/yam_training.sh. The stock
    # shell launcher does not forward CLI overrides. Do NOT change the 24-step action block.
    options = '''data=dreamzero/yam_relative fps.yam=50 train_architecture=lora
        model=dreamzero/vla model/dreamzero/action_head=wan_flow_matching_action_tf
        model/dreamzero/transform=dreamzero_cotrain num_frames=33 action_horizon=24 num_views=3
        num_frame_per_block=2 num_action_per_block=24 num_state_per_block=1 max_chunk_size=4 frame_seqlen=880
        image_resolution_width=320 image_resolution_height=176 seed=42 training_args.learning_rate=1e-5
        training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json training_args.warmup_ratio=0.05
        weight_decay=1e-5 save_total_limit=5 upload_checkpoints=false bf16=true tf32=true eval_bf16=true
        dataloader_pin_memory=false dataloader_num_workers=1 save_lora_only=true save_strategy=steps
        +train_dataset.dataset_kwargs.num_steps_per_shard=2000
        ++action_head_cfg.config.skip_component_loading=true ++action_head_cfg.config.defer_lora_injection=true'''.split()
    options += reporting + [f'max_steps={steps}', f'save_steps={save}', f'per_device_train_batch_size={args.batch_size}',
                f'logging_steps={1 if args.mode == "smoke" else min(10, steps)}']
    paths = dict(yam_data_root=data, output_dir=output, pretrained_model_path=base, dit_version=wan,
                 tokenizer_path=tokenizer, text_encoder_pretrained_path=wan / WAN_FILES[0],
                 image_encoder_pretrained_path=wan / WAN_FILES[1], vae_pretrained_path=wan / WAN_FILES[2])
    options += [f'{k}={json.dumps(str(v), ensure_ascii=False)}' for k, v in paths.items()]
    os.chdir(ROOT)
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH')]))
    os.environ['HYDRA_FULL_ERROR'] = '1'
    print(f'启动 {args.mode}：{steps} 步，每卡 batch={args.batch_size}，每 {save} 步保存 → {output}', flush=True)
    os.execv(sys.executable, [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc-per-node={procs}',
                              'groot/vla/experiment/experiment.py', *options])


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, ImportError) as exc:
        sys.exit(f'预检失败：{exc}')
