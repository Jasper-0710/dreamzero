# DreamZero 项目交接文档

> 编写日期：2026-06-29

---

## 1. 项目概述

**DreamZero** 是 NVIDIA GEAR Lab 开发的世界动作模型（World Action Model），能够联合预测动作和视频，在未见任务上展现强大的零样本性能。

192.168.2.52:/mnt/data0/xhx/dreamzero

192.168.2.42:/mnt/data1/xhx/dreamzero

- **论文**: [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
- **许可证**: Apache 2.0
- **主干模型**: Wan2.1-I2V-14B-480P（视频生成扩散模型）
- **语言编码器**: umt5-xxl
- **视觉编码器**: CLIP OpenCLIP XLM-RoBERTa-Large ViT-Huge-14

---

## 2. 目录结构

```
dreamzero/
├── groot/vla/                         # 核心代码
│   ├── model/dreamzero/               # DreamZero 模型定义
│   │   ├── base_vla.py                # VLA 基类
│   │   ├── action_head/               # 动作头（Wan Flow Matching）
│   │   ├── transform/                 # 数据变换（collate 等）
│   │   └── modules/                   # VAE、视频编解码等底层模块
│   ├── data/                          # 数据处理
│   │   ├── dataset/                   # 数据集加载（LeRobot Sharded）
│   │   ├── schema/                    # 数据模式（EmbodimentTag 等）
│   │   └── transform/                 # 数据增强/归一化
│   ├── configs/                       # Hydra 配置
│   │   ├── model/dreamzero/           # 模型配置
│   │   ├── data/dreamzero/            # 数据配置（各 embodiment）
│   │   └── deepspeed/                 # DeepSpeed 配置
│   └── experiment/experiment.py       # 训练入口
├── scripts/
│   ├── inference/                     # 推理服务脚本
│   │   ├── serve_dreamzero_g1.py      # ★ G1 推理服务（单 GPU）
│   │   ├── build_trt_engine.py/.sh    # TensorRT 引擎构建
│   │   └── validate_g1_checkpoint.py  # checkpoint 验证
│   ├── train/                         # 训练脚本
│   │   ├── g1_training_full_finetune.sh  # ★ G1 全量微调
│   │   ├── g1_training_lora.sh        # ★ G1 LoRA 微调
│   │   ├── g1_training_resume.sh      # ★ G1 断点续训
│   │   ├── droid_training_*.sh        # DROID 训练
│   │   ├── agibot_training.sh         # AgiBot 训练
│   │   └── yam_training.sh            # YAM 训练
│   └── data/                          # 数据转换脚本
│       ├── convert_lerobot_to_gear.py # LeRobot → GEAR 格式转换
│       └── convert_g1_to_dreamzero.py # G1 数据转换
├── eval_utils/                        # 评估工具
│   ├── policy_server.py               # WebSocket 策略服务
│   ├── policy_client.py               # WebSocket 客户端
│   └── serve_dreamzero_wan22.py       # Wan2.2 推理服务
├── eval_dreamzero_g1.py               # ★ G1 仿真评估（OmniGibson）
├── docs/                              # 文档
│   ├── DATASET_TO_GEAR_AND_TRAIN.md   # 新 embodiment 训练指南
│   ├── DROID_CONVERSION.md            # DROID 数据转换说明
│   └── WAN22_BACKBONE.md              # Wan2.2 主干说明
├── model/                             # 模型权重
│   ├── Wan2.1-I2V-14B-480P/           # Wan2.1 基础模型 (~28GB)
│   └── DreamZero-DROID/               # DROID 预训练 checkpoint
├── checkpoints/                       # 训练 checkpoint
│   ├── Wan2.1-I2V-14B-480P/           # Wan 基础权重
│   ├── umt5-xxl/                      # 分词器
│   ├── DreamZero-AgiBot/              # AgiBot 预训练 checkpoint
│   └── dreamzero_g1_finetuned_*/      # ★ G1 微调 checkpoint
├── lerobot/                           # LeRobot 格式数据集
│   └── omnigibson_dex_g1_56_converted/ # ★ G1 数据集（GEAR 格式）
└── OOM_FIX_NOTES.md                   # ★ OOM 问题修复说明
```

---

## 3. 本项目核心工作：G1 机器人适配

本仓库在原始 DreamZero 基础上，做了以下**面向 G1 机器人（OmniGibson 仿真）** 的适配和优化：

### 3.1 数据集

- **数据来源**: OmniGibson 仿真环境采集的 G1 遥操作数据
- **数据路径**: `lerobot/omnigibson_dex_g1_56_converted/`
- **数据规模**: 56 episodes, ~13,193 frames
- **格式**: LeRobot v2 → 经 `convert_lerobot_to_gear.py` 转为 GEAR 格式
- **关键特征**:
  - 单视角（`exterior_image_1_left`，640×480）
  - **36 维完整关节位置**（不做 state/action 拆分，不用 gripper 单独通道）
  - 图像自动 resize 到 320×176

### 3.2 训练

#### 3.2.1 全量微调（Full Fine-Tuning）

- **训练脚本**: `scripts/train/g1_training_full_finetune.sh`
- **硬件**: 8× RTX 6000D (86GB VRAM each)，PCIe 拓扑，无 NVLink
- **策略**: 全量微调（`train_architecture=full`），基于 DreamZero-AgiBot checkpoint
- **关键参数**:
  - `num_views=1`（单摄像头）
  - `max_state_dim=36, max_action_dim=36`
  - `frame_seqlen=220`（G1 专用：单摄像头 video latent × channels）
  - `image_resolution_width=320, image_resolution_height=176`
  - `per_device_train_batch_size=1`（14B 模型显存限制）
  - DeepSpeed ZeRO-2 + CPU offload（`zero2_offload.json`）
  - NCCL 超时 1800s（大模型 allreduce 较慢）
  - `train_architecture=full`，`save_lora_only=false`
- **已训得 checkpoint**:
  - `checkpoints/dreamzero_g1_finetuned_20260615_1805/checkpoint-1000/`, `checkpoint-2000/`
  - `checkpoints/dreamzero_g1_finetuned_20260616_1536/checkpoint-5000/`, `checkpoint-10000/`
- **启动命令**:
  ```bash
  export G1_DATA_ROOT=/mnt/data1/xhx/dreamzero/lerobot/omnigibson_dex_g1_56_converted
  export WAN_CKPT_DIR=./model/Wan2.1-I2V-14B-480P
  bash scripts/train/g1_training_full_finetune.sh
  ```

#### 3.2.2 LoRA 微调（LoRA Fine-Tuning）

LoRA 微调是显存更友好的替代方案，仅训练低秩 adapter 而不更新全部参数，适合快速实验和数据量较少的场景。

- **训练脚本**: `scripts/train/g1_training_lora.sh`
- **硬件**: 8× RTX 6000D (86GB VRAM each)
- **策略**: LoRA 微调（`train_architecture=lora`），基于 DreamZero-AgiBot checkpoint
- **与全量微调的关键差异**:

  | 参数 | 全量微调 | LoRA 微调 |
  |------|----------|-----------|
  | `train_architecture` | `full` | `lora` |
  | `learning_rate` | `1e-5` | `1e-4`（更高） |
  | `save_lora_only` | `false` | `true`（只保存 adapter 权重，文件更小） |
  | `deepspeed` | `zero2_offload.json`（CPU offload） | `zero2.json`（无 offload，LoRA 显存小） |
  | `dataloader_num_workers` | `8` | `1` |
  | `dataloader_pin_memory` | `true` | `false` |

- **LoRA 特有逻辑**:
  - 训练脚本通过 `++action_head_cfg.config.defer_lora_injection=true` 延迟 LoRA 注入，避免加载 pretrained checkpoint 时出错
  - 输出 checkpoint 仅包含 LoRA adapter 权重（`adapter_model.safetensors`），文件远小于全量模型
- **启动命令**:
  ```bash
  export G1_DATA_ROOT=/mnt/data1/xhx/dreamzero/lerobot/omnigibson_dex_g1_56_converted
  export WAN_CKPT_DIR=./model/Wan2.1-I2V-14B-480P
  bash scripts/train/g1_training_lora.sh

  # 自定义参数
  MAX_STEPS=5000 LEARNING_RATE=2e-4 bash scripts/train/g1_training_lora.sh
  ```

### 3.3 推理

- **推理服务**: `scripts/inference/serve_dreamzero_g1.py`
- **通信协议**: WebSocket（基于 `eval_utils/policy_server.py`）
- **硬件要求**: 单 GPU（RTX 6000D 86GB 可运行，推荐 ≥48GB）
- **默认设备**: `cuda:6`（可修改）
- **关键优化**:
  - 分布式初始化 → 模型整体显式迁移到目标 GPU
  - `local_attn_size` 从 21 扩展到 41（KV cache 生命周期从 ~6 步延长到 ~11 步）
  - 帧缓冲策略：首次推理 1 帧（预热 KV cache），后续每次 4 帧
  - 支持 `DEBUG_SAVE_VIDEO_PRED` 环境变量保存模型预测视频
- **启动命令示例**:
  ```bash
  python scripts/inference/serve_dreamzero_g1.py \
      --model_path ./checkpoints/dreamzero_g1_finetuned_20260616_1536/checkpoint-7000 \
      --port 8001
  ```

### 3.4 推理验证（validate_g1_checkpoint.py）

训练完成后，需要验证 checkpoint 的基本可用性和推理质量。`scripts/inference/validate_g1_checkpoint.py` 提供三种验证模式：

#### 3.4.1 基础验证模式（默认）

对数据集上的多个样本逐一推理，检查：
- 模型加载是否成功
- 前向推理是否正常（不 crash）
- 输出 action 是否包含 NaN / Inf
- 输出 shape 是否正确
- 推理耗时基线
- 预测 action 与 GT action 的 MSE 对比

```bash
python scripts/inference/validate_g1_checkpoint.py \
    --checkpoint ./checkpoints/dreamzero_g1_finetuned_20260616_1536/checkpoint-5000 \
    --device cuda:0 \
    --num_samples 30
```

#### 3.4.2 自回归 Rollout 模式（`--rollout`）

闭环模式：用模型自己的预测帧和预测状态反复输入，生成完整任务视频。
- 首次推理用数据集中的 4 帧真实帧
- 后续每步：取上一轮 VAE 解码的像素帧（需 resize 回 640×480）喂回模型
- 用模型预测的 action（绝对关节位置）更新 state，保持视频帧和关节状态一致
- 每次生成 ~12 帧视频（模型输出 33 帧，去除冗余）

```bash
python scripts/inference/validate_g1_checkpoint.py \
    --checkpoint ./checkpoints/dreamzero_g1_finetuned_20260616_1536/checkpoint-5000 \
    --rollout \
    --rollout_steps 20
# 视频保存到 eval_videos/rollout_20steps.mp4
```

#### 3.4.3 开环评估模式（`--openloop`）

开环模式：始终用数据集的真实连续帧和真实状态喂给模型，模拟"如果模型看到这个观测会怎么预测"。
- 每步取 4 帧真实连续帧 [t-3, t-2, t-1, t] + 真实 state[t]
- 模型预测 24 步 action，与 GT action[t:t+24] 对比
- 每步前重置 KV cache（`current_start_frame=0`），确保各步独立
- 同时解码模型生成的视频帧，输出拼接视频
- 按关键 DOF（肩俯仰、肘弯、前臂、腕转、右爪）打印逐维度 MSE/MAE
- 按预测时间步（第 0/5/11/17/23 步）打印误差衰减趋势

```bash
python scripts/inference/validate_g1_checkpoint.py \
    --checkpoint ./checkpoints/dreamzero_g1_finetuned_20260616_1536/checkpoint-5000 \
    --openloop \
    --stride 30 \
    --openloop_steps 50
# 输出视频:
#   eval_videos/openloop_50steps_gen.mp4  (模型生成的视频)
#   eval_videos/openloop_50steps_real.mp4 (数据集真实帧)
```

#### 关键参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | *(必填)* | checkpoint 路径 |
| `--dataset` | G1 数据集默认路径 | LeRobot v2 格式数据集 |
| `--device` | `cuda:0` | 推理设备 |
| `--num_samples` | `30` | 基础验证的采样数 |
| `--start_idx` | `0` | 起始行号 |
| `--rollout` | false | 启用闭环 rollout 模式 |
| `--rollout_steps` | `20` | rollout 步数（每步 ~12 帧） |
| `--openloop` | false | 启用开环评估模式 |
| `--stride` | `30` | 开环评估步幅（帧/步） |
| `--openloop_steps` | `50` | 开环评估最大步数 |

#### 理解三种模式

```
数据集真实数据 ──→ 模型 ──→ action + 生成视频
     ↑                          │
     │                          │  (rollout: 闭环反馈)
     │                          ↓
     └──── 预测的 action/帧 ────┘

基础验证:   输入=数据集真实, 只检查输出不反馈
开环(openloop): 输入=数据集真实, 每步独立推理, 对比 GT
闭环(rollout):  输入=模型自己预测, 反复自回归, 看累积效果
```

### 3.5 仿真评估

- **评估脚本**: `eval_dreamzero_g1.py`
- **仿真环境**: OmniGibson + G1WithDex3Hand 机器人
- **动作映射**: 36 维模型输出 → controller_action_idx 顺序的 env_action
  - 使用 `zip(controller_action_idx[comp], controller.dof_idx)` 精确映射
  - 跳过 trunk 关节（0-2）
  - 夹爪支持增益放大（`--gripper-gain`/`--gripper-scale`）和延迟闭合（`--gripper-delay-steps`）
- **重规划**: 每 8 步重新查询模型（`--replan-steps`）
- **推理服务器与仿真机分离部署**：仿真在 OmniGibson 机器运行，推理在 GPU 服务器运行

---

## 4. 已解决的关键问题

### 4.1 OOM 问题（详见 `OOM_FIX_NOTES.md`）

| 问题 | 解决方案 |
|------|----------|
| Shard 过大导致 CPU 内存溢出 | `num_steps_per_shard: 500`, `dataset_shard_sampling_rate: 0.5` |
| `num_shards_to_sample` 默认值过大 | 传入 `mixture_kwargs.num_shards_to_sample: 100` |
| Parquet 重复读取 Bug | `get_shard()` 中按 `episode_index` 过滤 |
| Batch size 过大 | 降至 `per_device_train_batch_size=1` |

### 4.2 状态/动作维度问题

- 原始代码将 36 维拆分为 7+1，导致模型预期与实际不匹配
- 修复：`modality.json` 中 state/action 定义为完整 36 维，YAML 配置中只用 `joint_position` 单键
- `max_state_dim=36, max_action_dim=36` 与模型一致

### 4.3 推理设备一致性问题

- 分布式初始化后 `torch.cuda.set_device(0)` 会覆盖目标设备
- 修复：初始化后显式 `torch.cuda.set_device(target_device)` + 整体 `model.to(device)`
- `action_head._device` 同步更新

### 4.4 动作反归一化

- 模型输出已经是反归一化后的绝对关节位置（服务器端 `sim_policy.unapply()` 处理）
- 客户端无需手动反归一化

### 4.5 Wan2.1 KV Cache 问题

- `local_attn_size` 默认 21，每 ~6 步重置一次 → 上下文丢失
- 扩展到 41，每 ~11 步重置，改善长时序一致性

---

## 5. 环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.11 |
| CUDA | 12.9+ |
| PyTorch | 2.8.0+ |
| GPU（训练） | 8× RTX 6000D 或同等 48GB+ VRAM |
| GPU（推理） | 1× 48GB+ VRAM（当前用 cuda:6） |
| DeepSpeed | ZeRO Stage 2 + CPU offload |
| 关键依赖 | flash-attn, transformers 4.51.3, diffusers 0.30.2 |

### 安装命令

```bash
conda create -n dreamzero python=3.11
conda activate dreamzero
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
MAX_JOBS=8 pip install --no-build-isolation flash-attn
```

---

## 6. 常用操作速查

### 训练 G1

```bash
# 全量微调
export G1_DATA_ROOT=/mnt/data1/xhx/dreamzero/lerobot/omnigibson_dex_g1_56_converted
export WAN_CKPT_DIR=./model/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=./checkpoints/umt5-xxl
bash scripts/train/g1_training_full_finetune.sh

# LoRA 微调（显存更友好，适合快速实验）
MAX_STEPS=5000 LEARNING_RATE=2e-4 bash scripts/train/g1_training_lora.sh
```

### 验证 Checkpoint

```bash
# 基础验证（30 个样本，检查 NaN/Inf/MSE）
python scripts/inference/validate_g1_checkpoint.py \
    --checkpoint ./checkpoints/dreamzero_g1_finetuned_20260616_1536/checkpoint-5000 \
    --device cuda:0 \
    --num_samples 30

# 开环评估（对比 GT action，输出 gen/real 对比视频）
python scripts/inference/validate_g1_checkpoint.py \
    --checkpoint ./checkpoints/dreamzero_g1_finetuned_20260616_1536/checkpoint-5000 \
    --openloop --stride 30 --openloop_steps 50

# 闭环 Rollout（模型自回归生成完整任务视频）
python scripts/inference/validate_g1_checkpoint.py \
    --checkpoint ./checkpoints/dreamzero_g1_finetuned_20260616_1536/checkpoint-5000 \
    --rollout --rollout_steps 20
```

### 启动推理服务

```bash
# 服务端（GPU 机器）
python scripts/inference/serve_dreamzero_g1.py \
    --model_path ./checkpoints/dreamzero_g1_finetuned_20260616_1536/checkpoint-5000 \
    --port 8001
```

### 仿真评估

```bash
# 客户端（OmniGibson 机器）
python eval_dreamzero_g1.py \
    --server-host <GPU_SERVER_IP> \
    --server-port 8001 \
    --instruction "Pick up the bottle" \
    --episodes 5 \
    --replan-steps 8
```

### 调试视频预测

```bash
DEBUG_SAVE_VIDEO_PRED=1 python scripts/inference/serve_dreamzero_g1.py --port 8001
# 视频保存在 debug_video_pred/ 目录
```

### 验证 Checkpoint

```bash
python scripts/inference/validate_g1_checkpoint.py \
    --model_path ./checkpoints/dreamzero_g1_finetuned_20260616_1536/checkpoint-5000
```
