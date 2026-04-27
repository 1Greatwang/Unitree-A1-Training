"""
A1机器人训练脚本 V2 - 优化版
基于改进的配置进行训练
"""

import os
import sys

# Isaac Gym 必须在 torch 之前导入
from isaacgym import gymapi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legged_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

import torch
from datetime import datetime
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot import LeggedRobot as A1

# 导入优化版配置
from a1_custom_config_v2 import A1CustomCfgV2, A1CustomCfgPPOV2


def main():
    """主函数"""
    print("="*70)
    print("🚀 A1 机器人强化学习训练 V2 - 优化版")
    print("="*70)
    
    # 注册任务
    task_registry.register('a1_custom_v2', A1, A1CustomCfgV2(), A1CustomCfgPPOV2())
    
    # 获取参数
    args = get_args()
    args.task = 'a1_custom_v2'
    
    # 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs('a1_custom_v2')
    
    # 打印改进信息
    print("\n📊 改进版本对比：")
    print("-"*50)
    print(f"并行环境数: 1024 -> {env_cfg.env.num_envs}")
    print(f"回合长度: 20s -> {env_cfg.env.episode_length_s}s")
    print(f"地形等级: 3 -> {env_cfg.terrain.max_init_terrain_level}")
    print(f"速度范围: [-1, 1] -> {env_cfg.commands.ranges.lin_vel_x}")
    print(f"训练轮数: 1500 -> {train_cfg.runner.max_iterations}")
    print(f"学习率: 1e-3 -> {train_cfg.algorithm.learning_rate}")
    print(f"步态奖励: 1.0 -> {env_cfg.rewards.scales.feet_air_time}")
    print("-"*50)
    
    # 检查是否从V1继续训练
    print("\n📝 训练选项：")
    print("1. 从头开始训练 (V2全新训练)")
    print("2. 从V1继续训练 (加载之前的模型)")
    print("")
    
    # 可以选择从V1加载
    v1_model_path = None
    v1_log_dir = "/root/a1_robot_project/legged_gym/logs/a1_custom"
    if os.path.exists(v1_log_dir):
        import glob
        model_files = glob.glob(os.path.join(v1_log_dir, "*/model_*.pt"))
        if model_files:
            latest = max(model_files, key=os.path.getmtime)
            v1_model_path = latest
            print(f"✓ 发现V1模型: {os.path.basename(v1_model_path)}")
    
    # 创建环境
    print("\n🎯 创建并行环境...")
    print(f"   环境数量: {env_cfg.env.num_envs}")
    print(f"   观测维度: {env_cfg.env.num_observations}")
    print(f"   动作维度: {env_cfg.env.num_actions}")
    
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    env = task_registry.get_task_class('a1_custom_v2')(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device, headless=args.headless
    )
    
    # 创建训练器
    print("\n🎓 创建PPO训练器...")
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name='a1_custom_v2', args=args)
    
    # 如果从V1加载
    if v1_model_path and input("\n是否从V1模型继续训练? (y/n): ").lower() == 'y':
        try:
            ppo_runner.load(v1_model_path)
            print(f"✅ 已加载V1模型，继续训练...")
            print(f"   将从第1500轮继续到第5000轮")
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            print("   将从头开始训练")
    
    # 打印网络结构
    print("\n🧠 Actor-Critic 网络结构：")
    print(ppo_runner.alg.actor_critic)
    
    # 打印配置信息
    print("\n" + "="*70)
    print("📋 训练配置信息")
    print("="*70)
    print(f"任务名称: {args.task}")
    print(f"并行环境数: {env_cfg.env.num_envs}")
    print(f"观测维度: {env_cfg.env.num_observations}")
    print(f"动作维度: {env_cfg.env.num_actions}")
    print(f"回合长度: {env_cfg.env.episode_length_s}秒")
    print(f"控制频率: {200/env_cfg.control.decimation:.1f}Hz")
    print(f"最大迭代数: {train_cfg.runner.max_iterations}")
    print(f"学习率: {train_cfg.algorithm.learning_rate}")
    print(f"实验名称: {train_cfg.runner.experiment_name}")
    print("="*70)
    
    print("\n🎮 奖励函数配置 (V2改进):")
    reward_items = vars(env_cfg.rewards.scales)
    for key, value in reward_items.items():
        if not key.startswith('_') and value != 0:
            print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("🚀 开始训练...")
    print("="*70)
    print("提示: 按V键切换渲染/停止渲染（如有图形界面）")
    print("      按ESC键退出训练")
    
    # 开始训练
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )
    
    print("\n✅ 训练完成！")
    print(f"模型保存在: ~/legged_gym/logs/{train_cfg.runner.experiment_name}/")


if __name__ == '__main__':
    main()
