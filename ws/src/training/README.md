# 按数据集启动翻枕头 LoRA 训练

入口：`bash ws/src/training/launch_pillow_training.sh <阶段> --dataset <原始录制目录> ...`

此入口复用已有的转换、对齐检查、整条录制留出和 `train_flip_pillow_lora.py`，不重写训练算法。**这是普通任务 LoRA 微调，不是 DreamZero-Flash 蒸馏，也不会自动续训之前的 checkpoint。** 当前接口仅适用于同样的 Mobile ALOHA 三相机、双臂 14 维、50 Hz 数据，以及“翻到红胶带面朝上”的任务；换数据集不等于已支持任意机器人或任意任务。

## 1. 同步与环境

在本机仓库根目录同步整个 `ws/src`，避免漏掉依赖：

```bash
rsync -av --exclude='__pycache__/' --exclude='*.py[co]' \
  ws/src/ omnisky@192.168.2.40:/data1/wp_ws/dreamzero-git/ws/src/
```

之后全部在**服务器**执行：

```bash
cd /data1/wp_ws/dreamzero-git
conda activate dreamzero

LAUNCH=ws/src/training/launch_pillow_training.sh
DATA=data/flip_150
RUN=flip_150_lora_01
GPUS=1,2,3,4,5,6  # 仅在这些卡已分配给你时使用
unset LOAD_TRT_ENGINE
```

脚本使用当前环境的 `python`，不自动安装依赖或下载权重；可用 `PYTHON_BIN=/实际/python` 切换解释器。相对路径统一按**仓库根目录**解析。原始数据及现有转换结果不会被修改。

默认数据流：

```text
data/flip_150                 原始录制，只读
    ↓ prepare：转换 + 首/中/末 episode 对齐核对
data/flip_150_dz              全量转换结果，供留出评估使用
    ↓ 按完整原始录制留出最后 10 条，并仅用训练集重算统计
data/flip_150_dz_train        训练集（Parquet / MP4 是指向全量目录的符号链接）
    ↓ smoke / train
checkpoints/flip_150_lora_01_smoke   冒烟测试输出
checkpoints/flip_150_lora_01         正式训练输出
```

请保留原始、全量转换和训练目录；不能移动/删除全量目录，否则训练集的符号链接会失效。真实录制数量从元数据读取，**不会根据 `flip_150` 这个名字假定有 150 条**。

## 2. 数据准备：只运行一次

如果这批数据依然全部是“红胶带面朝上”，执行：

```bash
bash "$LAUNCH" prepare --dataset "$DATA" --holdout 10 \
  --task-text 'Flip the pillow over so the side with the red tape faces up.'
```

请先停止录制，后续也不要继续向本次数据目录追加录制；新增数据另用目录。

这一步不使用 GPU、不加载模型：

1. 转换原始 Parquet 内嵌 JPEG 为 DreamZero 所需的 Parquet + 三路 MP4。
2. 核对转换后的首条、中间条、末条 episode：全部行的 state/action、视频时间戳以及三视角抽样图像。
3. 留出最后 10 条**完整原始录制**（包括其所有暂停切段），训练集重算绝对和相对统计。
4. 校验转换来源、训练/留出 ID 和训练文件结构。

`--task-text` 只更改输出标签，不改原始录制。如果原始标签已经正确，可以省略；旧的 `pick_cup` 等标签不能静默沿用。多任务数据不能用一条指令覆盖。

**不要反复运行 `prepare`**。两个输出目录必须是新目录；成功后直接进入 `check`。如果只有全量转换已经完成、尚未划分，可明确复用：

```bash
bash "$LAUNCH" prepare --dataset "$DATA" --holdout 10 --reuse-converted
```

复用时仍检查来源，并重新运行首/中/末对齐核对；不会覆盖已有的训练目录。若要更改留出数量，请另选 `--train-data` 新目录，并在之后所有阶段保持该参数和 `--holdout` 一致。

## 3. 预检

```bash
bash "$LAUNCH" check --dataset "$DATA" --holdout 10 \
  --gpus "$GPUS" --run-name "$RUN"
```

检查数据集、留出划分、本地权重、Python / PyTorch 和所选 GPU 数量；**不加载模型、不保证显存必然足够**。权重路径默认沿用仓库中的：

- `checkpoints/DreamZero-AgiBot`
- `checkpoints/Wan2.1-I2V-14B-480P`
- `checkpoints/umt5-xxl`

权重已在其他目录时，用 `DZ_CKPT_DIR`、`WAN_CKPT_DIR`、`TOKENIZER_DIR` 指向已有位置，见[完整训练说明](../README.md)。

## 4. 冒烟测试：先验证 10 步

```bash
bash "$LAUNCH" smoke --dataset "$DATA" --holdout 10 \
  --gpus "$GPUS" --run-name "$RUN"

tail -n 12 "checkpoints/${RUN}_smoke/loss_log.jsonl"
ls -lh "checkpoints/${RUN}_smoke/checkpoint-10/"
```

默认 10 步、每卡 batch=1，最后一步保存。**会真正训练**，但输出自动与正式运行分开；正式训练从官方 AgiBot 初始化，不接着 smoke 的结果训练。重跑 smoke 需要换新 `--run-name` 或 `--output`。

## 5. 正式训练

先按当前配方训练 2000 个优化步、每 200 步保存，之后比较留出集效果；**不是 2000 个 epoch**。这只是第一轮基线，不表示步数最优。

```bash
unset WANDB_DISABLED
bash "$LAUNCH" train --dataset "$DATA" --holdout 10 \
  --gpus "$GPUS" --run-name "$RUN" \
  --steps 2000 --save-steps 200 --batch-size 1 \
  --report-to wandb --wandb-mode offline \
  --wandb-project dreamzero-flip-pillow
```

- 之前服务器 W&B 出现过 403，所以上面选 `offline`：记录到服务器本地，**不会实时出现在 W&B 网页**。网络和登录恢复后可改 `--wandb-mode online`；同步离线记录用 `wandb sync <本次运行的offline-run目录>`。
- 完全不用 W&B：去掉三个 W&B 参数，改为 `--report-to none`；本地 `loss_log.jsonl` 仍保存。已有环境中若设置了 `WANDB_DISABLED`，使用 W&B 前需 `unset WANDB_DISABLED`。
- 延续现有配方：LoRA、学习率 `1e-5`、bf16、DeepSpeed ZeRO-2、50 Hz、每卡 batch=1。6 卡时每个优化步使用 6 个样本（当前梯度累积为 1）。
- checkpoint 在 `checkpoints/$RUN/checkpoint-<step>/`。原训练入口设置 **`save_total_limit=5`，只保留最近 5 个周期性 checkpoint**；2000 / 200 的设置最终通常保留 1200、1400、1600、1800、2000。不要假定更早的节点仍存在。
- 非空输出目录会拒绝覆盖，**不自动恢复训练**；重做实验请用 `--run-name flip_150_lora_02`。已有旧实验和旧数据不受影响。
- CPU 预检不能测出加载/训练峰值显存；若冒烟 OOM，先保留日志再调整配置，不要直接开长训练。

## 6. 换数据集与常用参数

下一个同格式、同任务的数据集准备好后，只需更换 `DATA`，并给新实验选择 `RUN`：

```bash
DATA=data/flip_200
RUN=flip_200_lora_01
# 对新 DATA 重新执行 prepare → check → smoke → train。
```

| 参数 | 用途 / 默认值 |
|---|---|
| `--dataset` | 必填，原始录制目录；不是转换后的目录 |
| `--converted` | 全量转换目录，默认 `<dataset>_dz` |
| `--train-data` | 留出后的训练目录，默认 `<converted>_train` |
| `--holdout` | 最后 N 条完整录制留出，默认 10；各阶段保持一致 |
| `--gpus` | 分配给你的 GPU，逗号分隔；省略则必须已有 `CUDA_VISIBLE_DEVICES` |
| `--num-gpus` | 进程数，默认等于所选 GPU 数；不会误用旧 `NUM_GPUS` |
| `--run-name` | 默认 `<dataset目录名>_lora_01`；smoke 自动加 `_smoke` |
| `--output` | 直接指定输出路径；传入时以此为准，不再自动加 smoke 后缀，注意选择独立目录 |
| `--steps` | train 默认 2000、smoke 默认 10 |
| `--save-steps` | train 默认每 200 步（不超过总步数），smoke 默认最后一步 |
| `--batch-size` | 每 GPU，默认 1 |
| `--report-to` | `none`（默认）或 `wandb` |
| `--wandb-mode` | `online` 或 `offline`；不传沿用已有 `WANDB_MODE`，SDK 通常默认 online |
| `--wandb-project` | 不传沿用已有 `WANDB_PROJECT`，底层入口兜底 `dreamzero` |
| `--dry-run` | 只打印执行计划，不校验实际数据/权重、不写文件、不启动子进程 / GPU |

可以先在没有新数据的本机查看正式启动计划：

```bash
bash ws/src/training/launch_pillow_training.sh train \
  --dataset data/flip_150 --gpus 1,2,3,4,5,6 \
  --steps 2000 --save-steps 200 --dry-run
```

训练结束后，用这次的 `data/flip_150_dz_train/meta/holdout.json`、对应的全量数据目录和新 checkpoint 做离线评估，不能继续把旧数据集的 91–100 当成本次留出集。现有评估入口与参数见[评估说明](../README.md#checkpoint-eval)。

例如，训练结束后先预检这次的动作评估配置：

```bash
python ws/src/evaluation/eval_flip_pillow.py \
  --run "checkpoints/$RUN" --steps 1200 1600 2000 \
  --train-data "${DATA}_dz_train" --check-only
```

若设置了自定义 `--train-data`，这里也使用相同目录。六卡动作 + 视频比较入口还必须显式选择**本次留出集**里的 `--source-id` 和合适的 `--anchors`，不要沿用旧录制 91 的默认窗口。
