# 翻枕头训练：参数直接传，不用设置环境变量

在服务器仓库根目录、已激活的 `dreamzero` Python 环境中直接执行：

```bash
./ws/src/training/launch_pillow_training.sh \
  --train_dir data/flip_150 \
  --gpus 1,2,3,4,5,6
```

`--train_dir` 指向**原始录制目录**，不是转换后的 `_dz_train` 目录。GPU 参数填写已经分配给你的卡。不需要先写 `DATA=...`、`RUN=...`、`export` 或 `unset`。

默认自动执行：

```text
按需准备数据 → 检查数据、权重、GPU → 正式 LoRA 训练
```

- 新数据：转换 → 首/中/末 episode 的 state/action、时间戳和三相机抽样图像核对 → 留出最后 10 条完整原始录制，重算仅训练数据的统计量。
- 已有全量转换但未划分：核对转换来源和对齐后划分，不重新编码。
- 已有训练集：校验来源和留出划分后复用，不重新转换或划分。
- 预检或任一步失败，立即停止，不继续训练。
- **这是普通 LoRA，不是 Flash，不自动接着旧 checkpoint 续训。**

这是专门的翻枕头入口，默认任务指令为：

> Flip the pillow over so the side with the red tape faces up.

请确认本批仍全部是“红胶带面朝上”的任务；默认完整流程会用该指令替换输出标签，即使原录制误标为 `pick_cup`。原始数据不修改。不要拿此入口给其他任务统一改标签。仅支持目前相同的 Mobile ALOHA 三相机、14 维、50 Hz 数据；先停止录制，不要边训练边向本次原始目录追加数据。

## 完整参数示例

```bash
./ws/src/training/launch_pillow_training.sh \
  --train_dir data/flip_150 \
  --gpus 1,2,3,4,5,6 \
  --steps 2000 \
  --save_steps 200 \
  --batch_size 1 \
  --holdout 10 \
  --output checkpoints/flip_150_lora_01 \
  --report_to wandb \
  --wandb_mode offline \
  --wandb_project dreamzero-flip-pillow
```

W&B 在线网页记录改为 `--wandb_mode online`（需要网络和登录正常）；离线模式不实时显示网页曲线。完全不用 W&B 就用默认的 `--report_to none`，本地 loss 日志仍保存。GPU、进程数、权重路径、W&B 配置和旧 TRT 设置由启动器在内部处理，不需要你手动设置或清理环境变量。API key 不要写进脚本，在线使用前按正常登录流程认证。

默认 2000 **优化步**，不是 2000 个 epoch；每 200 步保存、每卡 batch=1。沿用已有 LoRA 配方：学习率 `1e-5`、bf16、DeepSpeed ZeRO-2。先作为新数据的基线，再用留出集选择节点。

## 只执行某个阶段

```bash
# 只准备数据，不使用 GPU；原始任务标签不正确时显式指定输出任务
./ws/src/training/launch_pillow_training.sh \
  --mode prepare --train_dir data/flip_150 \
  --task_text 'Flip the pillow over so the side with the red tape faces up.'

# 数据准备好后只做预检，不加载模型
./ws/src/training/launch_pillow_training.sh \
  --mode check --train_dir data/flip_150 --gpus 1,2,3,4,5,6

# 数据准备好后只跑 10 步冒烟训练，默认输出目录自动加 _smoke
./ws/src/training/launch_pillow_training.sh \
  --mode smoke --train_dir data/flip_150 --gpus 1,2,3,4,5,6

# 数据准备好后只启动正式训练
./ws/src/training/launch_pillow_training.sh \
  --mode train --train_dir data/flip_150 --gpus 1,2,3,4,5,6

# 只打印完整计划，不访问实际数据或权重、不写文件、不启动 GPU
./ws/src/training/launch_pillow_training.sh \
  --train_dir data/flip_150 --gpus 1,2,3,4,5,6 --dry_run
```

`prepare` 单阶段不会覆盖已有转换/训练目录；如果只有全量转换完成，传 `--reuse_converted` 继续核对和划分。默认完整流程会自动识别这两种情况。冒烟会真正训练，正式训练不继承 smoke 的权重。

## 输出及常用参数

以 `data/flip_150` 为例，默认生成：

```text
data/flip_150_dz/                      全量转换数据（包含留出集）
data/flip_150_dz_train/                仅训练数据 + holdout.json
checkpoints/flip_150_lora_01/          正式 checkpoint / loss 日志
checkpoints/flip_150_lora_01_smoke/    单独冒烟时的输出
```

训练集的 Parquet / MP4 链接到全量目录，不能移动/删除全量目录。真实录制数量来自元数据，不根据目录名猜测数量。各阶段的 `--holdout`、`--converted`、`--train_data` 必须保持一致。

| 参数 | 默认值 / 含义 |
|---|---|
| `--train_dir` | 必填，原始录制目录；兼容 `--dataset` |
| `--mode` | `all`，即按需准备 → 预检 → 训练 |
| `--gpus` | 例如 `1,2,3,4,5,6`，显式选择已分配 GPU |
| `--num_gpus` | 默认等于所选 GPU 数 |
| `--converted` | `<train_dir>_dz` |
| `--train_data` | `<converted>_train` |
| `--holdout` | 10 条完整原始录制 |
| `--steps` | train=2000，smoke=10 |
| `--save_steps` | train=200（不超过总步数），smoke=最后一步 |
| `--batch_size` | 每卡 1 |
| `--run_name` | `<原始目录名>_lora_01`；用于生成默认输出路径 |
| `--output` | 自定义 checkpoint 目录；指定后不再自动加 smoke 后缀 |
| `--base_dir` | `checkpoints/DreamZero-AgiBot` |
| `--wan_dir` | `checkpoints/Wan2.1-I2V-14B-480P` |
| `--tokenizer_dir` | `checkpoints/umt5-xxl` |
| `--report_to` | `none` 或 `wandb` |
| `--wandb_mode` | W&B 选 `online` / `offline` |
| `--wandb_project` | W&B 项目名 |
| `--dry_run` | 只打印计划，不实际执行 |

非空 checkpoint 输出目录会拒绝覆盖，重做实验请改 `--output` 或 `--run_name`。底层训练只保留最近 5 个周期性 checkpoint；2000 / 200 的配置最终通常保留 1200、1400、1600、1800、2000。预检不能保证峰值显存足够，可先单独做冒烟测试。

训练完评估时指定新训练集，不要沿用旧录制 91–100：

```bash
python ws/src/evaluation/eval_flip_pillow.py \
  --run checkpoints/flip_150_lora_01 --steps 1200 1600 2000 \
  --train-data data/flip_150_dz_train --check-only
```

六卡动作 + 视频比较时，还要选择本次留出集里的 `--source-id` 和合适的 `--anchors`，不能照搬旧录制 91 的默认窗口。其他入口见[总说明](../README.md)。

## 同步脚本到服务器

在本机仓库根目录执行，无需复制数据或模型：

```bash
rsync -av --exclude='__pycache__/' --exclude='*.py[co]' \
  ws/src/ omnisky@192.168.2.40:/data1/wp_ws/dreamzero-git/ws/src/
```

旧的 `prepare / check / smoke / train --dataset ...` 位置参数形式仍兼容；新用法推荐上面的 `--train_dir` 和 `--mode`。
