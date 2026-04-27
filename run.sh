#!/bin/bash
# A1 Robot 训练启动脚本 (AutoDL)
# 用法: bash run.sh [选项]

set -e

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate a1_robot

# 设置环境变量
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/a1_robot/lib:$LD_LIBRARY_PATH

# 进入项目目录
cd ~/bionic_robot_project/scripts

echo "=========================================="
echo "A1 Robot RL 训练启动"
echo "=========================================="

# 默认参数
SIM_DEVICE="cuda:0"
HEADLESS="--headless"
MAX_ITERS=""
RESUME=""
NUM_ENVS=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            SIM_DEVICE="cpu"
            echo "模式: CPU"
            shift
            ;;
        --gui)
            HEADLESS=""
            echo "模式: 图形界面 (仅本地可用)"
            shift
            ;;
        --resume)
            RESUME="--resume"
            echo "恢复上次训练"
            shift
            ;;
        --max_iterations=*)
            MAX_ITERS="$1"
            echo "最大迭代: ${1#*=}"
            shift
            ;;
        --num_envs=*)
            NUM_ENVS="$1"
            echo "环境数: ${1#*=}"
            shift
            ;;
        --help)
            echo "用法: bash run.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --cpu                  使用 CPU 模式"
            echo "  --gui                  启用图形界面 (仅本地)"
            echo "  --resume               恢复训练"
            echo "  --max_iterations=N     设置最大迭代数"
            echo "  --num_envs=N           设置并行环境数"
            echo "  --help                 显示此帮助"
            echo ""
            echo "示例:"
            echo "  bash run.sh                              # GPU 无头模式"
            echo "  bash run.sh --max_iterations=1000        # 训练 1000 轮"
            echo "  bash run.sh --resume                     # 恢复训练"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python3 train_a1.py --sim_device=$SIM_DEVICE $HEADLESS"

if [ -n "$MAX_ITERS" ]; then
    CMD="$CMD $MAX_ITERS"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD $RESUME"
fi

if [ -n "$NUM_ENVS" ]; then
    CMD="$CMD $NUM_ENVS"
fi

echo "命令: $CMD"
echo "=========================================="
echo ""

# 启动训练
$CMD

echo ""
echo "=========================================="
echo "训练结束"
echo "=========================================="
