#!/usr/bin/env bash
# 适用于 omnisky-Rack-Server；必须 source，才能激活当前终端。
# 服务器用法：source /data1/wp_ws/activate_dreamzero.sh
# 本脚本复用已安装环境，不安装软件、不切换 editable 安装、不启动训练。

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    printf '请用 source 激活当前终端：\n  source %q\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi

_dz_server_root=/data1/wp_ws
_dz_server_env="$_dz_server_root/envs/dreamzero"
_dz_server_code="$_dz_server_root/dreamzero-git"
_dz_server_cuda="$_dz_server_root/toolchains/cuda-12.9-home"

for _dz_server_required in \
    "$_dz_server_root/miniforge3/etc/profile.d/conda.sh" \
    "$_dz_server_env/bin/python" \
    "$_dz_server_code/pyproject.toml" \
    "$_dz_server_cuda/bin/nvcc"; do
    if [[ ! -f $_dz_server_required ]]; then
        printf '[失败] 缺少服务器文件：%s\n' "$_dz_server_required" >&2
        return 1
    fi
done

# 使用独立 Conda 环境，保留简短的 (dreamzero) 提示符。
unset PYTHONHOME PYTHONPATH
export CONDA_ENV_PROMPT='({name}) '
source "$_dz_server_root/miniforge3/etc/profile.d/conda.sh" || return 1
conda activate "$_dz_server_env" || return 1

# 只允许使用物理 GPU 1–6；逻辑 cuda:0 对应物理 GPU 1。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
export NUM_GPUS=6

# 与 PyTorch 2.8.0+cu129 配套的独立 CUDA 12.9 编译工具链。
export CUDA_HOME="$_dz_server_cuda"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
case "$PATH" in
    "$CUDA_HOME/bin":*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
esac
case "${LIBRARY_PATH:-}" in
    "$CUDA_HOME/lib64"|"$CUDA_HOME/lib64":*) ;;
    *) export LIBRARY_PATH="$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}" ;;
esac

export CONDA_PKGS_DIRS="$_dz_server_root/cache/conda"
export PIP_CACHE_DIR="$_dz_server_root/cache/pip"
export HF_HOME="$_dz_server_root/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$_dz_server_root/cache/torch_extensions/dreamzero_torch28_cu129"
export TMPDIR="$_dz_server_root/tmp"
export MAX_JOBS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset DS_SKIP_CUDA_CHECK

mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" \
    "$TORCH_EXTENSIONS_DIR" "$TMPDIR" || return 1
cd "$_dz_server_code" || return 1

printf '\n[已激活] DreamZero\n'
printf 'Python：%s\n' "$(command -v python)"
printf '代码目录：%s\n' "$PWD"
printf '物理 GPU：%s（共 %s 张）\n' "$CUDA_VISIBLE_DEVICES" "$NUM_GPUS"
printf 'CUDA_HOME：%s\n\n' "$CUDA_HOME"

unset _dz_server_root _dz_server_env _dz_server_code _dz_server_cuda _dz_server_required
