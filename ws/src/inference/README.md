# Mobile ALOHA：用翻枕头 checkpoint 做只读推理

本服务回答“现在这组观测下，模型会发出什么动作”。沿用留出评估的 `recipe`、LoRA 恢复、相机变换和 `unapply()`；一次请求只用当前时刻，每次清除历史缓存、使用相同种子，**不是跨请求自回归控制器**。可以遥操作采样对照，但本客户端不创建电机发布者、不调用驱动服务、不改变扭矩、不控制底盘。无需 W&B、重新训练或把权重搬上机器人。

## 1. GPU 服务器

在**本机 DreamZero 仓库根目录**同步代码。评估程序已同步过时只需发新增目录：

```bash
rsync -av --exclude='__pycache__/' ws/src/inference/ \
  omnisky@192.168.2.40:/data1/wp_ws/dreamzero-git/ws/src/inference/
```

在**服务器**已激活的 `(dreamzero)` 环境：

```bash
cd /data1/wp_ws/dreamzero-git
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1  # 在已分配的 1–6 中选一张空闲卡
unset LOAD_TRT_ENGINE
CKPT=checkpoints/flip_pillow_lora_01/checkpoint-1600
python ws/src/inference/serve_pillow.py --checkpoint "$CKPT" --check-only
python ws/src/inference/serve_pillow.py --checkpoint "$CKPT"
```

等出现 `READY: checkpoint-1600 ... / 127.0.0.1:8766 / shadow only`。预检查元数据/权重文件头，不分配 GPU，不检验显存够不够。服务只绑定服务器回环，预期单客户端；不是对外的生产 API。无需 `torchrun`。端口冲突可在服务端、隧道、客户端一起换端口。

## 2. 本机先发录制帧，再试实时观测

机器人/本机**终端 A**建立隧道并保持连接（若客户端就在模型服务器上则跳过）：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8766:127.0.0.1:8766 omnisky@192.168.2.40
```

**终端 B**，从 DreamZero 根目录，验证连接、用你原始录制的第 91 条第 216 帧测试；客户端录制模式只需 Python、NumPy、PyArrow、Pillow，不需要 CUDA/ROS。原始文件使用内嵌 JPEG，不是 `_dz_train`。读同一行三图与 state，不读未来、GT action，不以相同帧的多次推理来评价准确率。

```bash
curl -fsS http://127.0.0.1:8766/health
python ws/src/inference/pillow_client.py recording --episode 91 --frame 216 --count 6 \
  --output results/pillow_shadow_1600_recording.jsonl
```

再在**连接 ALOHA 的机器、平时能读取 ROS1 的 Python 环境**，保持相机/从臂状态驱动已启动。不用运行录制、回放或启动新的电机控制程序：

```bash
python3 ws/src/inference/pillow_client.py ros --count 6 \
  --constants ../aloha_record/aloha_scripts/constants.py \
  --output results/pillow_shadow_1600_ros.jsonl
```

- ROS 模式只需 NumPy、ROS1 `rospy/message_filters/sensor_msgs/cv_bridge` 和录制时使用的 `constants.py`，不需要 PyArrow、Pillow 或 PyTorch。`/usb_cam_high/image_raw`、`/usb_cam_left_wrist/image_raw`、`/usb_cam_right_wrist/image_raw` 转为 RGB 640×480；`/puppet_left/joint_states`、`/puppet_right/joint_states` 分别取六关节和 `PUPPET_GRIPPER_POSITION_NORMALIZE_FN(position[7])`（指爪直线反馈，不是 position[6] 电机角）。驱动名称/标定不同先核对而非跳过校验。建议相机安装、视角、左右、光照与录制一致，尤其不要让红/黑胶带因颜色通道弄反而失真。
- 客户端按五路 ROS 消息 **header 时间**配对（默认时间跨度 ≤40 ms、最旧输入年龄 ≤150 ms；不等同硬件触发）。要求传感器使用共同的 ROS 系统时钟；无新鲜同步组则报错，不填零或沿用旧状态。请求串行，一次只取最新同步组。每个原始 RGB 请求约 2.64 MiB；建议走有线局域网而非把接口暴露到公网。
- 返回 `action` 是 **24×14**：左 6 关节目标(rad) + 左归一化夹爪指令 + 右 6 关节目标(rad) + 右归一化夹爪指令。是录制动作空间中的**绝对目标**，不是增量/速度，勿再加 state，也不是 14 个电机弧度。无底盘指令。JSONL 另有 `action_names`、输入关节、观测标识/时间和计时。`--output` 不覆盖，重跑请换名；不指定则只看终端摘要。换节点需同时改服务端 `--checkpoint` 和客户端 `--expect-checkpoint checkpoint-N`。

## 3. 迁移和验收限制

- **GPU 端**要有同一版仓库（含 `ws/src/common/`、`evaluation/`、`inference/` 和 `groot/`）、完整 `checkpoint-1600/`（含 `config.json`、`experiment_cfg/`、权重和相应索引分片）、**DreamZero-AgiBot** 原始基座、Wan2.1 的 T5/CLIP/VAE 三组件、`umt5-xxl` tokenizer。不要只拷 LoRA 文件，也不要换成 DROID 或 Wan2.2 基座。检查所需 Wan 文件名见 `ws/src/common/pillow.py::WAN_FILES`。
- 服务端只用 `data/flip_the_pillow_dz_train/meta/`（`info.json`、`modality.json`、`tasks.jsonl`、`embodiment.json`、`stats.json`、`relative_stats_dreamzero.json`）。保留原训练统计与节点内 `metadata.json` 互校；**不需要整份视频/Parquet 或 holdout**。搬家用 `--train-data /新目录/flip_the_pillow_dz_train`、`--base /新目录/DreamZero-AgiBot`，并 `export WAN_CKPT_DIR=/新目录/Wan2.1-I2V-14B-480P TOKENIZER_DIR=/新目录/umt5-xxl`；`--train-data` 指 `meta/` 的上一级。
- 复用服务器已经跑通的 Linux x86-64、Python 3.11、torch 2.8 + cu129、兼容 NVIDIA 驱动和 flash-attn。这里沿用单 GPU 的评估路径，不是仓库根目录按 DROID 7+1 维动作服务的 `socket_test_optimized_AR.py`，也不是同事的 G1 控制器。磁盘可在权重机 `du -sh checkpoints/{DreamZero-AgiBot,Wan2.1-I2V-14B-480P,umt5-xxl,flip_pillow_lora_01/checkpoint-1600}` 测实际占用；注意 CPU 加载时内存峰值。要记录峰值可用 `/usr/bin/time -v python ws/src/inference/serve_pillow.py`，退出后查最大 RSS。
- **不能把 LoRA 文件大小当作推理显存**：本次运行日志里 DiT 约 165 亿参数，按 BF16 粗算，仅这一部分就约 30 GiB。当前完整单卡路径不适用于标准 RTX 4090 24 GB；同事的 `ws/src/doc/DreamZero实验.md` 是另一套 G1，记录了 86 GB 卡单卡可运行，不等于本节点 48 GB 一定够。先在已能评估该节点的服务器测显存/延迟，再决定是否迁移。
- 终端显示同步计时的 `model`、客户端 `RTT`（组包到取回并校验结果）、进程累计 PyTorch GPU `peak reserved`（含加载，不含驱动及其他进程；另看 `nvidia-smi`），ROS 另显示 `oldest_obs_at_reply`（最老传感器时间到收到动作）。先测通，再 `--count 30` 在目标网络上看抛去首请求后的 p50/p95；固定录制帧的统计不能代替实时传感器测试。
- **24 / 50 = 0.48 秒**是一个动作块的控制周期总长，并不是一次推理可以用 0.48 秒之外的旧观测随意补发命令。若未来每 6 步刷新一次，预算只有 120 ms，还需给采集、传输、调度、刹停留余量。延迟、夹爪抓放时刻、跟踪误差、超时/限位保护及急停没验收前，保持只读；不要把 JSONL 喂给回放或驱动。

本地 CPU 协议测试：`python -m unittest ws.src.tests.test_pillow_shadow -v`。该测试不加载真权重、不证明显存、真实 ROS 联通或任务成功率。
