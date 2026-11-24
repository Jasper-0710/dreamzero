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

### 同一任务混用了多个旧标签（例如 `pick_cup` / `flip_the_pillow`）

转换器会先只读取每条录制的 `task_index` 列，打印实际使用的标签和对应录制数量，不把任务表里未使用的历史标签当作多任务。若实际录制用了多个标签，单独传 `--task_text` 不会自动合并，以免把真实抓杯子数据误当作翻枕头。

**仅当你确认整批实际上全部都是“翻到红胶带面朝上”**时，使用：

```bash
./ws/src/training/launch_pillow_training.sh \
  --train_dir data/flip_150 \
  --gpus 1,2,3,4,5,6 \
  --task_text 'Flip the pillow over so the side with the red tape faces up.' \
  --confirm_same_task
```

`--confirm_same_task` 必须与显式的 `--task_text` 配合。仅输出数据统一为一条任务、所有输出 Parquet 的 `task_index=0`；原标签和 ID 仍保存于 `meta/source_episodes.jsonl`，使用情况与映射记录在 `meta/conversion.json`。不会修改原始数据，也不会改变 state/action 数值或时间对齐。若确有其他任务，请先按真实任务分开数据，不能靠确认参数修复。

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

## 使用 SwanLab 网页看 loss（替代 W&B）

已通过 `swanlab_reporting.py` 接入 SwanLab 官方 SDK 的 `init / log / finish`，
保留官方 `VLATrainer`，只增加标量日志 callback，不改模型、数据或 checkpoint 格式。
不再使用 Transformers 自带的 SwanLab callback：部分旧 Transformers 期待 `get_run()` 在未初始化时返回
`None`，而新版 SwanLab 会抛出 `No active Run. Call swanlab.init() first.`，导致训练在第 0 步退出。
修复版不调用 `get_run()`，只在 torchrun worker 的 **world rank 0** 初始化一个实验，其他 rank 不写 SwanLab。
无需因此升级或降级环境；项目的 `transformers==4.51.3` 依赖保持不变。

### 1. 在服务器的 dreamzero 环境安装、登录

```bash
python -m pip install swanlab
swanlab login
```

按终端提示在 SwanLab 网页获取 **SwanLab 自己的 API key** 后粘贴；不能使用 W&B key，不能把 key 写进启动脚本或发到聊天。

### 2. 先检查云端记录，不加载模型、不占 GPU

```bash
python ws/src/monitoring/check_swanlab.py --project dreamzero-flip-pillow
```

这会使用**训练同款 callback** 完整走一遍 `on_train_begin → on_log → on_train_end`，
创建一个名为 `connection-check` 的小实验并写入一条 `train/connection_test` 指标，
不读取录制数据、不加载模型、不启动训练。测试会实际导入 Transformers 的基础 callback，但不占 GPU。
确认终端打印了在线实验地址，且网页能看到该指标，再启动大模型。账号权限、服务器出口仍需此步实际确认。
团队项目给测试脚本加 `--workspace 团队空间名`，训练启动器使用对应的 `--swanlab_workspace 团队空间名`。

### 3. 已完成数据准备：直接训练并在线上报

```bash
./ws/src/training/launch_pillow_training.sh \
  --mode train \
  --train_dir data/flip_150 \
  --gpus 1,2,3,4,5,6 \
  --steps 4000 \
  --save_steps 200 \
  --batch_size 1 \
  --run_name flip_150_lora_4000_swanlab_02 \
  --report_to swanlab \
  --swanlab_mode online \
  --swanlab_project dreamzero-flip-pillow
```

`--mode train` 要求数据已经准备好；首次转换的新数据使用默认 `all`，任务确认/留出规则与上文保持一致。
已有失败运行也可能占用了旧输出目录，请使用新 `--run_name`，不删除旧数据、不自动续训。
`--report_to swanlab` **只启用 SwanLab，不同时启用 W&B**；残留的 W&B key 或离线设置不参与这个 callback。
内部打印的 Hydra 参数是 `report_to=none` 和
`trainer._target_=ws.src.training.swanlab_reporting.create_swanlab_trainer`：
这是有意关闭有兼容问题的原生集成，再由工厂为官方训练器添加修复版 callback，不是没有开启上报。

训练启动时 SwanLab 会打印项目/实验网页地址。网页看 `train/loss`、`train/grad_norm`、`train/learning_rate`；
训练器上报时还会显示 `train/action_loss_avg`、`train/dynamics_loss_avg` 等细分指标。
正式训练每 10 步记录总 loss；首次加载大模型时没有曲线属于正常等待。
checkpoint 仍按原训练配方保存到本地 `checkpoints/.../checkpoint-*`，不会通过该 callback 上传模型；
SwanLab 本地日志放在本次输出目录的 `swanlog/`，原有 `loss_log.jsonl` 仍然保留。
代码仍填写内部必须的 `wandb_project` Hydra 字段，但这不表示创建了 W&B run。

离线使用 `--swanlab_mode offline`，只在本地记录，**不能实时从云端网页看曲线**。
若只想检查命令、不安装 SDK、不访问网络或 GPU，可在训练命令末尾加 `--dry_run`。

接口依据：[SwanLab Transformers 集成](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-transformers.html)、
[官方环境变量说明](https://docs.swanlab.cn/api/environment-variable.html)。

### `No active Run` 修复后的同步和重试

本机在项目根目录同步两个代码目录（不删除服务器上的其他文件）：

```bash
rsync -av --exclude='__pycache__/' --exclude='*.pyc' \
  ws/src/training ws/src/monitoring \
  omnisky@192.168.2.40:/data1/wp_ws/dreamzero-git/ws/src/
```

服务器先运行上述 `check_swanlab.py`，确认测试指标显示后再执行六卡训练命令。
原失败日志停在 step 0；本次不必重新转换数据，用 `--mode train` 复用已准备的训练集。
`_01` 目录可能已写入配置，即使没有训练 checkpoint 也不覆盖，例子改用 `_02`；若 `_02` 也被占用再换新名字。

CPU 回归（SDK 使用测试替身，无网络/凭据/GPU）覆盖旧、新 `get_run()` 行为、六 rank 只建立一个实验、
重复初始化/结束、动作/视频 loss 指标转发、保留官方训练器参数、不上传 checkpoint：

```bash
python -m unittest ws.src.tests.test_swanlab_callback ws.src.tests.test_swanlab_reporting -v
```

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
| `--task_text` | 显式输出任务指令；兼容 `--task-text` |
| `--confirm_same_task` | 仅 all / prepare，与显式 `--task_text` 配合，确认多个旧标签实际是同一任务；兼容 `--confirm-same-task` |
| `--steps` | train=2000，smoke=10 |
| `--save_steps` | train=200（不超过总步数），smoke=最后一步 |
| `--batch_size` | 每卡 1 |
| `--run_name` | `<原始目录名>_lora_01`；用于生成默认输出路径 |
| `--output` | 自定义 checkpoint 目录；指定后不再自动加 smoke 后缀 |
| `--base_dir` | `checkpoints/DreamZero-AgiBot` |
| `--wan_dir` | `checkpoints/Wan2.1-I2V-14B-480P` |
| `--tokenizer_dir` | `checkpoints/umt5-xxl` |
| `--report_to` | `none`、`wandb` 或 `swanlab` |
| `--wandb_mode` | W&B 选 `online` / `offline` |
| `--wandb_project` | W&B 项目名 |
| `--swanlab_mode` | SwanLab 选 `online`（默认）/ `offline` |
| `--swanlab_project` | SwanLab 项目名，默认 `dreamzero-flip-pillow` |
| `--swanlab_workspace` | SwanLab 团队空间名，可选 |
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
