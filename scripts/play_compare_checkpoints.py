"""
训练进度对比展示 - 展示不同训练阶段的模型
同时显示早期模型和最终模型，直观对比进步

用法:
    python play_compare_checkpoints.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legged_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from isaacgym import gymapi, gymtorch
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot import LeggedRobot as A1
from a1_custom_config_v2 import A1CustomCfgV2, A1CustomCfgPPOV2


def main():
    task_registry.register('a1_custom_v2', A1, A1CustomCfgV2(), A1CustomCfgPPOV2())
    
    args = get_args()
    args.task = 'a1_custom_v2'
    args.headless = False
    
    env_cfg, train_cfg = task_registry.get_cfgs('a1_custom_v2')
    
    # 两个环境并排：左边早期模型，右边最终模型
    env_cfg.env.num_envs = 2
    env_cfg.env.env_spacing = 5.0  # 分开展示
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    
    # 相机位置能看到两个机器人
    env_cfg.viewer.pos = [5, 0, 4]
    env_cfg.viewer.lookat = [5, 2.5, 0]
    
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("="*60)
    print("🎬 训练进度对比展示")
    print("="*60)
    print("左侧 (红): 早期模型 (model_100.pt) - 刚学会站立")
    print("右侧 (绿): 最终模型 (model_5000.pt) - 熟练奔跑")
    print("="*60)
    
    env = task_registry.get_task_class('a1_custom_v2')(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device, headless=False
    )
    
    # 创建两个 runner，加载不同检查点
    base_dir = '/home/ubuntu/data/最终版/legged_gym/logs/a1_custom_v2/Mar31_23-07-49_'
    
    print("\n加载早期模型...")
    ppo_runner_early, _ = task_registry.make_alg_runner(env=env, name='a1_custom_v2', args=args)
    ppo_runner_early.load(f'{base_dir}/model_100.pt')
    
    print("加载最终模型...")
    # 重新创建 runner 加载最终模型
    ppo_runner_final, _ = task_registry.make_alg_runner(env=env, name='a1_custom_v2', args=args)
    ppo_runner_final.load(f'{base_dir}/model_5000.pt')
    
    print("\n演示开始 - 按 ESC 退出")
    print("观察：左侧机器人笨拙vs右侧机器人流畅的对比\n")
    
    obs = env.get_observations()
    steps = 0
    
    # 给两个机器人不同颜色的指令（可选）
    env.commands[0, 0] = 1.0  # 左侧：向前 1m/s
    env.commands[1, 0] = 1.0  # 右侧：同样指令
    
    with torch.no_grad():
        while True:
            if env.gym.query_viewer_has_closed(env.viewer):
                break
            
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "QUIT":
                    print("\n退出")
                    return
            
            # 分别推理：环境0用早期模型，环境1用最终模型
            obs_batch = env.get_observations()
            
            # 早期模型推理环境0
            actions_early = ppo_runner_early.alg.actor_critic.act_inference(obs_batch[0:1])
            
            # 最终模型推理环境1  
            actions_final = ppo_runner_final.alg.actor_critic.act_inference(obs_batch[1:2])
            
            # 合并动作
            actions = torch.cat([actions_early, actions_final], dim=0)
            
            obs, _, rewards, dones, _ = env.step(actions)
            
            steps += 1
            
            # 打印对比信息
            if steps % 200 == 0:
                reward_early = rewards[0].item()
                reward_final = rewards[1].item()
                print(f"步数 {steps:4d} | 早期奖励: {reward_early:6.2f} | "
                      f"最终奖励: {reward_final:6.2f} | 提升: {(reward_final-reward_early):6.2f}")
            
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)


if __name__ == '__main__':
    main()
