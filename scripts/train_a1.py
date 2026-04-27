"""
A1机器人训练脚本
用法:
    python train_a1.py --headless              # 无渲染训练
    python train_a1.py --num_envs=2048         # 指定环境数
    python train_a1.py --resume                # 恢复训练
    python train_a1.py --max_iterations=3000   # 指定最大迭代数
"""

import numpy as np
import os
import sys
from datetime import datetime

# 添加legged_gym到路径（支持两种安装位置）
# 1. 项目目录下的legged_gym（当前位置）
# 2. home目录下的legged_gym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legged_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'legged_gym'))
sys.path.insert(0, os.path.expanduser('~/legged_gym'))

import isaacgym
from isaacgym import gymapi

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, Logger
from legged_gym.utils.helpers import class_to_dict
import torch
import torch.nn as nn

# 导入自定义配置
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
from a1_custom_config import A1CustomCfg, A1CustomCfgPPO


def setup_custom_task():
    """注册自定义A1任务"""
    from legged_gym.utils.task_registry import task_registry
    from legged_gym.envs.base.legged_robot import LeggedRobot as A1
    
    # 注册自定义配置
    task_registry.register(
        'a1_custom', 
        A1, 
        A1CustomCfg(), 
        A1CustomCfgPPO()
    )


def train(args):
    """主训练函数"""
    
    # 注册自定义任务
    setup_custom_task()
    
    # 自动检测 CUDA 可用性，选择合适的设备
    if not torch.cuda.is_available():
        print("=" * 60)
        print("自动检测: 无 GPU 可用，切换到 CPU 模式")
        print("=" * 60)
        args.rl_device = 'cpu'
        args.sim_device = 'cpu'
        args.use_gpu = False
        args.use_gpu_pipeline = False
    else:
        print("=" * 60)
        print(f"自动检测: GPU 可用，使用 {torch.cuda.get_device_name(0)}")
        print("=" * 60)
        args.rl_device = 'cuda:0'
        args.sim_device = 'cuda:0'
        args.use_gpu = True
        args.use_gpu_pipeline = True
    
    # 创建环境和配置
    print("=" * 60)
    print("创建环境...")
    print("=" * 60)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 创建PPO训练器
    print("\n创建PPO训练器...")
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, 
        name=args.task, 
        args=args
    )
    
    # 打印训练信息
    print("\n" + "=" * 60)
    print("训练配置信息")
    print("=" * 60)
    print(f"任务名称: {args.task}")
    print(f"并行环境数: {env_cfg.env.num_envs}")
    print(f"观测维度: {env_cfg.env.num_observations}")
    print(f"动作维度: {env_cfg.env.num_actions}")
    print(f"回合长度: {env_cfg.env.episode_length_s}秒")
    print(f"控制频率: {1 / (env_cfg.sim.dt * env_cfg.control.decimation):.1f}Hz")
    print(f"最大迭代数: {train_cfg.runner.max_iterations}")
    print(f"学习率: {train_cfg.algorithm.learning_rate}")
    print(f"实验名称: {train_cfg.runner.experiment_name}")
    print("=" * 60)
    
    # 打印奖励函数配置
    print("\n奖励函数配置:")
    reward_scales = class_to_dict(env_cfg.rewards.scales)
    for key, value in reward_scales.items():
        if value != 0:
            print(f"  {key}: {value}")
    
    # 开始训练
    print("\n开始训练...")
    print("提示: 按V键切换渲染/停止渲染（如有图形界面）")
    print("      按ESC键退出训练\n")
    
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )
    
    print("\n训练完成!")
    print(f"模型保存在: logs/{train_cfg.runner.experiment_name}/")


def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("系统信息")
    print("=" * 60)
    
    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # CUDA信息
    if torch.cuda.is_available():
        print(f"CUDA可用: 是")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA可用: 否")
        print("警告: 没有可用的GPU，训练将非常缓慢!")
        print("提示: 如需GPU支持，请检查NVIDIA驱动和虚拟机GPU直通配置")
    
    # Isaac Gym信息
    try:
        gym = gymapi.acquire_gym()
        print(f"Isaac Gym: 已加载")
    except Exception as e:
        print(f"Isaac Gym: 加载失败 - {e}")
        print("提示: 尝试设置 export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")
    
    print("=" * 60)


def main():
    """主函数"""
    # 解析命令行参数
    args = get_args()
    
    # 设置默认任务（覆盖legged_gym的默认anymal_c_flat）
    if args.task == 'anymal_c_flat':
        args.task = 'a1_custom'
    
    # 如果命令行明确指定了 sim_device，则使用指定的
    # 否则让 train() 函数自动检测
    if args.sim_device == 'cuda:0' and not torch.cuda.is_available():
        print("警告: 命令行指定了 CUDA 但 GPU 不可用，将自动切换到 CPU")
    
    # 打印系统信息
    print_system_info()
    
    # 开始训练
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
