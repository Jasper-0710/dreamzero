# 将 Mobile ALOHA 的翻枕头录制数据转成 DreamZero 训练集

输入是 `../aloha_record/aloha_scripts/hf_dataset_writer.py` 写的 LeRobot v2.1：图像是 **Parquet 内嵌 JPEG**，不是 MP4。目标是本仓库 `data=dreamzero/yam_relative` 的三视角、双臂 14 维训练接口。**原始数据只读，输出必须是新目录。** 不连接机器人，不需要 ROS、CUDA 或模型权重。

## 先转换两条、核对，再转全量

以下命令均从 **DreamZero 仓库根目录**执行。用 Python 3.11；推荐独立环境，避免改变训练环境里的 NumPy/Pandas。系统还需 `ffmpeg`（带 `libx264` 编码器）和 `ffprobe` 在 `PATH` 中。

```bash
# 如果已有兼容的环境，可跳过创建环境这两行
python3.11 -m venv .venv-aloha-convert
source .venv-aloha-convert/bin/activate
python -m pip install -r ws/src/requirements-convert.txt
ffmpeg -hide_banner -encoders | grep libx264
ffprobe -version | head -1
```

当前 `data/flip_the_pillow/meta/info.json` 记载 **101 条 episode、51,934 个控制步、50 Hz**，三个相机均为 **640×480**。原始 `tasks.jsonl` 写的是 **`pick_cup`**。这批示范已确认都是翻到红胶带面朝上，请显式传 `--task-text`；该操作只改**输出**任务表，不会更改原始录制。如果原始标签正确，可改用 `--keep-task-text`。多任务数据不允许用一个 `--task-text` 覆盖。

```bash
# 1. 小样本：源 episode 0、1；不会启动训练
python ws/src/convert_aloha_to_dreamzero.py \
  --input data/flip_the_pillow --output data/flip_the_pillow_dz_smoke \
  --limit 2 --task-text 'Flip the pillow over so the side with the red tape faces up.'

# 2. 从原始数据独立核对一个输出 episode；也可改 --episode 1
python ws/src/check_aloha_alignment.py \
  --input data/flip_the_pillow --output data/flip_the_pillow_dz_smoke \
  --episode 0 --samples 5

# 3. 确认任务和视角无误后，转全量到另一个新目录
python ws/src/convert_aloha_to_dreamzero.py \
  --input data/flip_the_pillow --output data/flip_the_pillow_dz \
  --task-text 'Flip the pillow over so the side with the red tape faces up.'
```

按当前源元数据，前两条合计 **1,477 行、6 个 MP4**。核对脚本对选定 episode 的**全部** state、action 及未重排的诊断列做等值比较，对三路 MP4 检查全部解码帧的时间戳；再抽取首、中、尾等位置，把解码 RGB 和**同相机、同源行**的 JPEG 比较，打印每路最差一张的平均绝对像素误差。H.264 是有损的，不要求像素相等；该抽查不等于验证硬件曝光时刻。还可以单独执行 `python ws/src/test_aloha_temporal.py`，测试固定 anchor、最后一个完整窗口，以及暂停/长间隔边界。

转换不覆盖已有目录。在目标目录的同一文件系统内临时写入，校验成功后才发布；正常报错/中断不会留下看似完整的训练集。按原录制 episode 选取子集可以用 `--start-episode N --limit M`（前者含 N）。训练和评估必须按**整条源录制**隔离；已完成全量转换时可用下文的 `prepare_pillow_holdout.py`，不要将同一次示范的相邻帧随机拆开。

## 对齐规则（按录制和回放代码，而不是猜偏移）

| 训练含义 | 录制来源 | 写入方式 |
|---|---|---|
| 当前观测 `state[t]` | `record_episodes_hf.py::tick` 先取观测；`real_env.py::get_qpos` 读取从臂实测关节和夹爪 | 同行原样保存到 `observation.state` |
| 当前指令 `action[t]` | `real_env.py::get_action` 取得主臂绝对关节目标、归一化夹爪指令，随后 `env.step(action)` 下发 | 同行原样保存到 `action`；**不取 `state[t+1]`，不平移，不裁剪** |
| 当前三路图像 | 该 tick 取到的各相机最新画面；writer 保存为 JPEG | 第 `t` 行 JPEG → 对应视频第 `t` 帧；不去重、不插帧、不单独降帧 |
| `base_action` | 录制时的底盘 CAN 速度反馈，不是遥操作指令 | 保留诊断列；不拼进 14 维 action，不参与训练模态 |

`hf_dataset_writer.py::append` 将上述内容写到同一行。两条臂各 6 个关节（弧度）+ 1 个夹爪，**state 和 action 使用相同切片**：

| `modality.json` 的子键 | 14 维数组上的 Python 切片 |
|---|---|
| `left_joint_pos` | `[0:6]` |
| `left_gripper_pos` | `[6:7]` |
| `right_joint_pos` | `[7:13]` |
| `right_gripper_pos` | `[13:14]` |

夹爪 state 是从臂开度归一化值，夹爪 action 是主臂指令归一化值，二者不必相等，指令也不一定严格落在 `[0,1]`。`replay_episodes_hf.py` 为实机回放所做的夹爪缩放、限幅和速度处理，**都不用于生成训练标签**。

- **时间轴**：`timestamp=i/fps` 是控制步时间；`recording.elapsed_time` 是 tick 的实际经过时间，**不是**相机硬件时间戳。没有独立曝光时间可重建，摄像头未更新时的重复帧也属于对应的控制行。输出原频率 CFR 视频，保留 `recording.elapsed_time`、`recording.segment_index`、速度、力矩和底盘反馈以便追溯。
- **暂停/掉采样**：遇到 `recording.segment_index` 变化或实际间隔 `> --max-gap`（默认 `0.25` 秒，取自回放脚本），拆成新 episode，帧号/视频时间从零开始。任一段不足 25 行即报错，**不跨暂停拼窗口、不补帧、不悄悄丢数据**。本批录制元数据均为单段；只读检查的最大实际相邻间隔约为 0.0284 秒。
- **相对动作**：Parquet 保持绝对值。YAM 读取器按每个 24 步块用同一个 `state[t]` 计算 `action[t+k] - state[t]`（`k=0..23`，含夹爪）。相对统计在每段内按读取器口径，汇集**每个完整**的 24 步窗口，包括最后一个；不会跨 episode 或把已经相对化的数据再减一次。

## 输出和训练接入

```text
data/flip_the_pillow_dz/
  data/chunk-000/episode_000000.parquet        # 低维列；无内嵌 JPEG
  videos/chunk-000/observation.images.top_camera-images-rgb/episode_000000.mp4
  videos/chunk-000/observation.images.left_camera-images-rgb/episode_000000.mp4
  videos/chunk-000/observation.images.right_camera-images-rgb/episode_000000.mp4
  meta/info.json, modality.json, embodiment.json
  meta/episodes.jsonl, tasks.jsonl, stats.json, relative_stats_dreamzero.json
  meta/source_episodes.jsonl, conversion.json   # 输出 ID → 源 episode、段、[起,止) 行、原任务和相机映射
```

`cam_high`、`cam_left_wrist`、`cam_right_wrist` 分别映射到 **top / left / right**；`annotation.task` 通过整数 `task_index` 查 `tasks.jsonl`。如全量转换成功，按这批数据应为 **101 条输出轨迹、51,934 个控制步、303 个 MP4**。

- 本工具直接复用本仓库 `scripts/data/convert_lerobot_to_gear.py` 的模态构建、绝对统计和基础校验函数。其原 CLI **不会**把内嵌 JPEG 变成视频，所以不要直接对原目录运行它，也不必转换后再运行一次。相对统计以实际读取器 `groot/vla/data/dataset/lerobot.py::_calculate_relative_stats_for_key` 为准。
- 输出针对 DreamZero 的 GEAR 读取器，不直接复制原 `episodes_stats.jsonl` 里的 JPEG 像素统计。训练读的是重新计算的 `stats.json` 和 `relative_stats_dreamzero.json`。
- `info.json` 的真实机器人仍是 `mobile_aloha`；`embodiment.json` 的 `yam` 仅用于复用此仓库已有的双臂三视角接口，**不是说 ALOHA 和 YAM 硬件相同**。
- 后续微调时把 `YAM_DATA_ROOT` 指向**用于训练的转换后目录**（如果划分留出集，就指向训练子集）。`scripts/train/yam_training.sh` 使用 `data=dreamzero/yam_relative`，但本仓库的 `fps.yam` 默认是 **30**。保留本批 50 Hz 数据时，在你自己的启动脚本的 `torchrun ...` 参数中加入 **`fps.yam=50`**（24 个控制周期为 0.48 秒）。这个 shell 脚本**不会**转发额外命令行参数；在 `bash scripts/train/yam_training.sh` 后面追加该参数并不生效。
- 不要直接用 `scripts/open_loop_yam.py` 检查动作：它硬编码了另一套关节偏移（32–46）且直接从 `annotation.task` 列读文本。这里是紧凑的 14 维切片和 `task_index`，请用上面的核对脚本验证转换。评估模型时，另用按本数据集 `meta/modality.json` 取值的代码。

## 转换完成后：在服务器微调翻枕头策略

下面只在服务器 `/data1/wp_ws/dreamzero-git` 执行。先将 `ws/src` 的更新同步到服务器。已经得到 `data/flip_the_pillow_dz` 就直接使用，不需要再编码视频，也不需要修改绝对动作标签。

### 1. 核对全量输出，再留出未见过的示范

```bash
cd /data1/wp_ws/dreamzero-git
# 新终端若已部署服务器激活脚本，可先 source /data1/wp_ws/activate_dreamzero.sh
cat data/flip_the_pillow_dz/meta/tasks.jsonl
for ep in 0 50 100; do
  python ws/src/check_aloha_alignment.py --input data/flip_the_pillow \
    --output data/flip_the_pillow_dz --episode "$ep" --samples 5 || break
done

# 已有独立测试录制时，这一步可选。否则保留最后 10 条原始录制用于评估。
python ws/src/prepare_pillow_holdout.py --holdout 10
export YAM_DATA_ROOT="$PWD/data/flip_the_pillow_dz_train"
```

核对成功应显示指令是红胶带朝上、所查轨迹的 state/action 全行一致、三路视频的 PTS 和抽样像素比较通过。留出脚本只读全量数据：新目录对已有 Parquet 和 MP4 建符号链接，不复制或转码视频；在新 `meta` 里只用训练轨迹重算绝对及相对统计。**不要移动或删除全量目录**。此次 101 条原始录制对应训练 0–90、留出 91–100；预计 46,921 个训练帧、273 个训练 MP4；实际 ID 见训练目录的 `meta/holdout.json`。留出的是整条录制（包括暂停切段），不是随机视频帧。若另有独立评估数据、不留出这 10 条，则把 `YAM_DATA_ROOT` 指向全量转换后的目录。

### 2. 检查分配到的 GPU 和已有权重

```bash
nvidia-smi
free -h
df -h .
# 新终端请使用服务器激活脚本的设置；仅在确实获分配物理 1–6 时使用：
# export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 NUM_GPUS=6
# 默认根为 ./checkpoints。如果服务器已有权重，直接导出已有目录：
# export DZ_CKPT_DIR=/实际路径/DreamZero-AgiBot
# export WAN_CKPT_DIR=/实际路径/Wan2.1-I2V-14B-480P
# export TOKENIZER_DIR=/实际路径/umt5-xxl
python ws/src/train_flip_pillow_lora.py check
```

预检只查表、文件（含 AgiBot 索引的分片和 Git LFS 指针）、解释器及 PyTorch 可见卡数，不启动训练，也不证明显存一定足够。缺权重时才执行对应的下载命令。下面是这份**训练入口**所需的下载项；其 DiT 使用 AgiBot 检查点初始化，再注入 LoRA，因此不需要另下一套 Wan DiT 分片。若日后使用其他推理入口，按其加载方式准备相应基座组件。

```bash
# 若你的环境仅有 huggingface-cli，将下面的 hf download 替换为 huggingface-cli download。
hf download GEAR-Dreams/DreamZero-AgiBot --local-dir "${DZ_CKPT_DIR:-./checkpoints/DreamZero-AgiBot}"
hf download Wan-AI/Wan2.1-I2V-14B-480P \
  models_t5_umt5-xxl-enc-bf16.pth models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth Wan2.1_VAE.pth \
  --local-dir "${WAN_CKPT_DIR:-./checkpoints/Wan2.1-I2V-14B-480P}"
hf download google/umt5-xxl --include '*.json' 'spiece.model' --local-dir "${TOKENIZER_DIR:-./checkpoints/umt5-xxl}"
```

### 3. 先验证 10 步，再做第一轮训练

```bash
# 在上面同一终端运行，沿用 YAM_DATA_ROOT 和明确的 CUDA_VISIBLE_DEVICES
python ws/src/train_flip_pillow_lora.py smoke
tail -n 12 checkpoints/flip_pillow_lora_smoke/loss_log.jsonl
ls -lh checkpoints/flip_pillow_lora_smoke/checkpoint-10/

# smoke 正常后：从 AgiBot 重新开始训练，不续用 smoke 的 checkpoint
python ws/src/train_flip_pillow_lora.py train --steps 1000 --save-steps 200
```

两种模式分别写 `checkpoints/flip_pillow_lora_smoke` 和 `checkpoints/flip_pillow_lora_train`，遇非空输出目录会停止；新实验可加 `--output checkpoints/flip_pillow_lora_run02`。默认每卡 batch 1、学习率 `1e-5`、仅写本地日志；希望用 W&B 的话，**不执行上面最后那条 `train`**，改用第 4 节命令，只训练一次。显存和吞吐确认后才考虑 `--batch-size`。启动器保留 YAM 的 `3 views / 24 actions per block / num_frames=33`、LoRA 和 ZeRO-2，并显式 `fps.yam=50`；将每个数据分片的目标大小设为 2,000 步，以降低多卡三路原始 RGB 解码及预取的**主机内存**占用。这与 GPU 显存是不同的限制。

**验收分两层：** 10 步结束、loss 和动作损失有限、写出 `checkpoint-10`，只说明训练管线通了。1,000 步也只是方便比较的第一轮预算，不是最优值。训练入口不会自动构造验证集。后续需在 `holdout.json` 指定的**未参与训练的录制**上，用训练集的统计和 `modality.json` 解码动作，在同一动作空间比较左右臂关节误差和夹爪开合时机；不要直接运行旧 `open_loop_yam.py`。最终另做多种未见枕头起始位置的受控真机测试（限速、有人看护、可急停），记录“翻后红胶带面朝上、松手后枕头在桌面稳定”的成功次数 / 总次数。

### 4. 用 W&B 看已完成的 smoke / 实时看下一轮训练

把本机 `ws/src/train_flip_pillow_lora.py`、`ws/src/import_loss_to_wandb.py` 同步到服务器仓库的 `ws/src/`。以下命令在**之前通过 smoke 的服务器终端**执行。新终端请先恢复同一环境、分配的 GPU、权重和训练集路径。

```bash
cd /data1/wp_ws/dreamzero-git
python -c 'import wandb; print(wandb.__version__)'  # 仅在没安装时：python -m pip install wandb
unset WANDB_DISABLED
export WANDB_MODE=online WANDB_PROJECT=dreamzero-flip-pillow
# 若要放入团队空间，可再 export WANDB_ENTITY=你的团队标识
wandb login  # 在服务器终端交互登录，不要把 API key 写到命令、代码或聊天里
```

**已完成的 smoke**：原 `report_to=none` 没有创建 W&B run；日志中的 `Copying wandb_config.json` 不是同步成功的提示。无需重跑，把当时的 JSONL 指标导成单独的 run：

```bash
python ws/src/import_loss_to_wandb.py checkpoints/flip_pillow_lora_smoke/loss_log.jsonl
```

打开输出的 run 链接，应能看到 `flip_pillow_lora_smoke-imported`。加 `--dry-run` 可只检查点数，不导入；真正导入运行一次即可，重复运行会新建 run。它按真实训练步合并各指标，不补零、不伪造时间、梯度、显存或其他缺失的历史数据。横轴选 `train/global_step`，不要用导入时的 wall time。

**正式训练**：同一终端执行下列命令，保留 smoke 通过的训练设置，从 AgiBot 基座开始，不续训/覆盖 `checkpoint-10`。`--output` 指定一个新的空目录或尚不存在的路径；目录名也是网页上的 run 名。

```bash
python ws/src/train_flip_pillow_lora.py train --report-to wandb \
  --steps 1000 --save-steps 200 --output checkpoints/flip_pillow_lora_wandb_01
```

打开终端中 W&B 输出的 **run** 链接。横轴用 `train/global_step`，看 `train/loss`（总损失，正式训练每 10 步一条）、`train/action_loss_avg`（动作）、`train/dynamics_loss_avg`（视频预测），并可同时看 `train/learning_rate`、`train/grad_norm`。6 个训练进程共用一个 Trainer 跟踪的 run；不会上传模型权重或示范视频。线上会记录指标、训练配置和运行环境等信息，请使用获准的团队/项目。本地 `loss_log.jsonl` 和 checkpoint 照常保留。

**服务器不能联网**：训练前 `export WANDB_MODE=offline`，训练命令仍保留 `--report-to wandb`。这时不提供实时云面板。待网络恢复，登录并对 W&B 输出的**具体离线 run 目录**执行 `wandb sync <离线run目录>`（本例在 `checkpoints/flip_pillow_lora_wandb_01/wandb/offline-run-*` 下）。旧 smoke 的普通 JSONL 不能直接 `wandb sync`，用上面的历史导入命令。

### 5. 不用 W&B 时：在本机浏览器看本地日志

网页只读训练器原有的 `loss_log.jsonl`，不写入 checkpoint 目录、不占用 GPU，不需要安装 TensorBoard 或登录 W&B。可以先开网页再训练，也可以在训练中途或结束后打开。网页服务仅监听服务器的 `127.0.0.1`，不需要开放防火墙端口。

**本机终端**（在本机 DreamZero 仓库根目录，不是在服务器上）：将下面的 `DZ_SSH` 换成你平时 `ssh` 使用的地址或别名。若脚本已同步，可跳过 `scp`。`ssh` 这条命令同时启动网页和本地端口转发，请保持该终端打开。

```bash
DZ_SSH='omnisky@你的服务器IP'
scp ws/src/serve_loss.py "$DZ_SSH":/data1/wp_ws/dreamzero-git/ws/src/
ssh -t -o ExitOnForwardFailure=yes -L 127.0.0.1:8765:127.0.0.1:8765 "$DZ_SSH" \
  'cd /data1/wp_ws/dreamzero-git && /data1/wp_ws/envs/dreamzero/bin/python ws/src/serve_loss.py'
```

在**本机浏览器**打开 `http://127.0.0.1:8765/`。默认显示 **Smoke 测试**，点击 **正式训练** 切换到正式训练目录。若使用 VS Code Remote-SSH，也可在服务器新终端运行 `python3 ws/src/serve_loss.py`，再在 VS Code 的“端口”面板转发 `8765`（与上面的 SSH 命令二选一）。网页每 3 秒刷新；初次加载模型期间显示“等待首条日志”正常。

**原来通过预检的服务器训练终端**中运行 `python ws/src/train_flip_pillow_lora.py smoke`。查看 `loss`（总损失）、`action_loss_avg`（动作损失）、`dynamics_loss_avg`（视频预测损失）和 `learning_rate`。总损失在 smoke 中每步记录；分项均值分开写，默认约每 10 步记录，10 步时只有一个点也正常。网页不会把缺失指标补成 0；遇到 NaN/Inf 会提示。结束时检查 `checkpoints/flip_pillow_lora_smoke/checkpoint-10/` 是否写出。退出网页终端不影响独立启动的训练。

若训练命令使用了 `--output`，启动网页时用 `--smoke-dir 你的输出目录` 或 `--train-dir 你的输出目录` 与之对应。若仅**本机**端口 8765 已占用，可将 SSH 的 `-L` 改为 `127.0.0.1:8766:127.0.0.1:8765` 并在浏览器访问本机 8766；若**服务器**端口已占用，则用 `--port` 和 `-L` 相应地一起调整。
