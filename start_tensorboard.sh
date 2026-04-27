#!/bin/bash
# TensorBoard 启动脚本 (AutoDL)
# 用法: bash start_tensorboard.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate a1_robot

echo "启动 TensorBoard..."
echo "访问地址: http://$(hostname -I | awk '{print $1}'):6006"
echo "AutoDL 需开放 6006 端口"
echo ""

# 创建日志目录（如果不存在）
mkdir -p ~/legged_gym/logs/a1_custom

tensorboard --logdir=~/legged_gym/logs/a1_custom \
    --host=0.0.0.0 \
    --port=6006
