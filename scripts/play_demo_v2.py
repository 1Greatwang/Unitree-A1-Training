"""
A1机器人演示脚本 V2 - 加载训练好的模型并可视化
使用 a1_custom_v2 配置和模型

用法:
    python play_demo_v2.py              # 加载最新模型演示
    python play_demo_v2 --checkpoint=5000   # 加载指定检查点
    
控制:
    ESC - 退出
    V   - 暂停/继续渲染
"""

import os
import sys

# 设置路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legged_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

# Isaac Gym 必须在 torch 之前导入
from isaacgym import gymapi, gymtorch

import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path, class_to_dict
from legged_gym.envs.base.legged_robot import LeggedRobot as A1
from a1_custom_config_v2 import A1CustomCfgV2, A1CustomCfgPPOV2


def main():
    # 注册任务
    task_registry.register('a1_custom_v2', A1, A1CustomCfgV2(), A1CustomCfgPPOV2())
    
    # 获取参数
    args = get_args()
    args.task = 'a1_custom_v2'
    args.headless = False  # 显示图形
    
    # 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs('a1_custom_v2')
    env_cfg.env.num_envs = 4  # 多环境演示：3x3 网格
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 9  # 最高难度
    env_cfg.env.env_spacing = 4.0  # 环境间距（默认）
    
    # 使用类似单环境的相机设置，但更高更远以看到所有机器人
    env_cfg.viewer.pos = [12, 0, 8]      # 相机位置
    env_cfg.viewer.lookat = [13, 5, 2]   # 看向中心区域
    
    # 创建环境
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("="*60)
    print("A1 四足机器人 - 模型推理演示 (多环境)")
    print("="*60)
    print(f"任务: {args.task}")
    print(f"环境数: {env_cfg.env.num_envs} (同时展示 {env_cfg.env.num_envs} 条狗)")
    print(f"观察维度: {env_cfg.env.num_observations}")
    print(f"动作维度: {env_cfg.env.num_actions}")
    print("="*60)
    
    print("\n创建环境...")
    env = task_registry.get_task_class('a1_custom_v2')(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device, headless=False
    )
    
    # 加载模型
    print("加载模型...")
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name='a1_custom_v2', args=args)
    
    # 直接指定模型路径
    checkpoint = args.checkpoint if hasattr(args, 'checkpoint') and args.checkpoint else -1
    if checkpoint > 0:
        model_path = f'/home/ubuntu/data/最终版/legged_gym/logs/a1_custom_v2/Mar31_23-07-49_/model_{checkpoint}.pt'
    else:
        model_path = '/home/ubuntu/data/最终版/legged_gym/logs/a1_custom_v2/Mar31_23-07-49_/model_5000.pt'
    
    if os.path.exists(model_path):
        ppo_runner.load(model_path)
        print(f"✓ 已加载模型: {os.path.basename(model_path)}")
    else:
        print(f"✗ 未找到模型: {model_path}")
        return
    
    print("\n" + "="*60)
    print("演示开始 - 按 ESC 退出")
    print("="*60 + "\n")
    
    # 运行演示
    obs = env.get_observations()
    steps = 0
    episode_rewards = torch.zeros(env_cfg.env.num_envs, device=env.device)
    episode_count = 0
    
    with torch.no_grad():
        while True:
            if env.gym.query_viewer_has_closed(env.viewer):
                break
            
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "QUIT":
                    print("\n退出")
                    return
            
            # 使用策略网络进行推理
            actions = ppo_runner.alg.actor_critic.act_inference(obs)
            obs, _, rewards, dones, _ = env.step(actions)
            
            episode_rewards += rewards
            steps += 1
            
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            
            # 回合结束（任意一个环境结束就打印）
            done_envs = dones.nonzero(as_tuple=True)[0]
            if len(done_envs) > 0:
                for env_id in done_envs:
                    episode_count += 1
                    print(f"回合 {episode_count}: 环境{env_id.item()} {steps}步, 奖励: {episode_rewards[env_id].item():.2f}")
                    episode_rewards[env_id] = 0


if __name__ == '__main__':
    main()
