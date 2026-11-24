#!/usr/bin/env python3
"""Test SwanLab logging before loading DreamZero; creates one tiny test experiment."""

import argparse
import os
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.common.pillow import ROOT
from ws.src.training.train_flip_pillow_lora import configure_swanlab


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', default='dreamzero-flip-pillow')
    p.add_argument('--workspace', help='工作空间名，不传则使用个人空间或 SDK 配置')
    p.add_argument('--mode', choices=['online', 'offline'], default='online')
    p.add_argument('--output', type=Path, default=ROOT / 'results/swanlab_check',
                   help='仅保存本次连接测试日志，不是训练 checkpoint 目录')
    args = p.parse_args(argv)
    if not args.project.strip() or (args.workspace is not None and not args.workspace.strip()):
        p.error('项目名和工作空间名不能为空')
    os.environ.update(SWANLAB_PROJECT=args.project.strip(), SWANLAB_MODE=args.mode,
                      SWANLAB_ON_ONLINE_ERROR='raise')  # A connection test must not silently go offline.
    if args.workspace is not None:
        os.environ['SWANLAB_WORKSPACE'] = args.workspace.strip()
    output = args.output.expanduser().resolve()
    project = configure_swanlab(output)
    import swanlab
    kwargs = dict(project=project, experiment_name='connection-check', mode=args.mode,
                  logdir=str(output / 'swanlog'), config={'purpose': 'connection-test-not-training'})
    workspace = os.environ.get('SWANLAB_WORKSPACE', '').strip()
    if workspace:
        kwargs['workspace'] = workspace
    print('只创建一个连接测试实验并写入一条测试指标；不加载模型、不读取数据、不占用 GPU。', flush=True)
    swanlab.init(**kwargs)
    try:
        swanlab.log({'connection_test': 1}, step=0)
    finally:
        swanlab.finish()
    print('SwanLab 初始化和日志写入流程已完成。', flush=True)
    if args.mode == 'online':
        print('请确认上方打印了在线实验地址，并能在网页看到 connection_test；确认后再启动训练。')
    else:
        print('这是离线测试，不验证云端访问，不生成实时云端网页。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        sys.exit(f'SwanLab 测试失败（未启动训练）：{exc}')
