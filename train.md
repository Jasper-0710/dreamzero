可以，**推荐用 `tmux`，断开 SSH、关闭终端后训练仍会继续，还能重新进入查看进度。** 这里是让训练在后台继续，不是暂停训练。

## 1. 在服务器上创建训练会话

```bash
tmux new -s dreamzero_flip150
```

进入新终端后，执行下面完整命令：

```bash
cd /data1/wp_ws/dreamzero-git
source ./activate_dreamzero_server.sh
mkdir -p logs

./ws/src/training/launch_pillow_training.sh \
  --mode train \
  --train_dir data/flip_150 \
  --gpus 1,2,3,4,5,6 \
  --steps 4000 \
  --save_steps 200 \
  --batch_size 1 \
  --run_name flip_150_lora_4000_swanlab_03 \
  --report_to swanlab \
  --swanlab_mode online \
  --swanlab_project dreamzero-flip-pillow \
  2>&1 | tee -a logs/flip_150_lora_4000_swanlab_03.log
```

**注意：如果 `_03` 已经启动过、checkpoint 目录非空，脚本会拒绝重新启动；新实验请把 `run_name` 和日志名一起改成 `_04`。这不是断点续训。**

## 2. 退出会话，让训练继续运行

依次按：

**先按 `Ctrl + B`，松开，再按 `D`。**

看到类似下面提示，就说明已退出会话，训练仍在继续：

```text
[detached (from session dreamzero_flip150)]
```

此时可以关闭 SSH。**不要用 `Ctrl + C` 退出训练画面，那会中断训练。**

## 3. 之后查看训练

### 重新进入训练画面

```bash
tmux attach -t dreamzero_flip150
```

### 不进入会话，只查看日志

在服务器另一个终端执行：

```bash
cd /data1/wp_ws/dreamzero-git
tail -n 100 -f logs/flip_150_lora_4000_swanlab_03.log
```

这里按 `Ctrl + C` **只退出日志查看，不影响训练**。

SwanLab 在线上报也会继续，网页可以继续看 loss；后台运行不会改变上报方式。

## 4. 如果要停止训练

```bash
tmux attach -t dreamzero_flip150
```

进入训练画面后按 **`Ctrl + C`**。

你设置了每 **200 步**保存 checkpoint；中断后应以**最后一个完整保存成功的 checkpoint** 为准，中断时的当前步骤不保证保存。

> 如果你的训练现在已经在普通终端前台运行，**先不要重复执行上述启动命令，也不要为了切后台而中断它**；这种情况需要用另一种方式处理。