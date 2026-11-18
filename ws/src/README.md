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

当前 `data/flip_the_pillow/meta/info.json` 记载 **101 条 episode、51,934 个控制步、50 Hz**，三个相机均为 **640×480**。原始 `tasks.jsonl` 写的是 **`pick_cup`**。如果这批示范都是翻枕头，请显式传 `--task-text`；该操作只改**输出**任务表，不会更改原始录制。如果原始标签正确，可改用 `--keep-task-text`。多任务数据不允许用一个 `--task-text` 覆盖。

```bash
# 1. 小样本：源 episode 0、1；不会启动训练
python ws/src/convert_aloha_to_dreamzero.py \
  --input data/flip_the_pillow --output data/flip_the_pillow_dz_smoke \
  --limit 2 --task-text 'Flip the pillow over.'

# 2. 从原始数据独立核对一个输出 episode；也可改 --episode 1
python ws/src/check_aloha_alignment.py \
  --input data/flip_the_pillow --output data/flip_the_pillow_dz_smoke \
  --episode 0 --samples 5

# 3. 确认任务和视角无误后，转全量到另一个新目录
python ws/src/convert_aloha_to_dreamzero.py \
  --input data/flip_the_pillow --output data/flip_the_pillow_dz \
  --task-text 'Flip the pillow over.'
```

按当前源元数据，前两条合计 **1,477 行、6 个 MP4**。核对脚本对选定 episode 的**全部** state、action 及未重排的诊断列做等值比较，对三路 MP4 检查全部解码帧的时间戳；再抽取首、中、尾等位置，把解码 RGB 和**同相机、同源行**的 JPEG 比较，打印每路最差一张的平均绝对像素误差。H.264 是有损的，不要求像素相等；该抽查不等于验证硬件曝光时刻。还可以单独执行 `python ws/src/test_aloha_temporal.py`，测试固定 anchor、最后一个完整窗口，以及暂停/长间隔边界。

转换不覆盖已有目录。在目标目录的同一文件系统内临时写入，校验成功后才发布；正常报错/中断不会留下看似完整的训练集。按原录制 episode 选取子集可以用 `--start-episode N --limit M`（前者含 N）。如果要隔离训练/验证，请在**源 episode 层面**先分开转换，不要将同一次示范的相邻帧随机拆开。

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
- 后续微调时把 `YAM_DATA_ROOT` 指向**全量输出目录**。`scripts/train/yam_training.sh` 使用 `data=dreamzero/yam_relative`，但本仓库的 `fps.yam` 默认是 **30**。保留本批 50 Hz 数据时，在你自己的启动脚本的 `torchrun ...` 参数中加入 **`fps.yam=50`**（24 个控制周期为 0.48 秒）。这个 shell 脚本**不会**转发额外命令行参数；在 `bash scripts/train/yam_training.sh` 后面追加该参数并不生效。
- 不要直接用 `scripts/open_loop_yam.py` 检查动作：它硬编码了另一套关节偏移（32–46）且直接从 `annotation.task` 列读文本。这里是紧凑的 14 维切片和 `task_index`，请用上面的核对脚本，或让评估代码按本数据集的 `meta/modality.json` 取值。
