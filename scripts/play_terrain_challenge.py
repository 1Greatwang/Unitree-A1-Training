"""
地形挑战测试 - 展示机器人在复杂地形上的表现
包含：斜坡、楼梯、崎岖地面等地形类型
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legged_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from isaacgym import gymapi, gymtorch
import torch
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
    
    # 16 个环境，4 种地形，每种 4 个
    env_cfg.env.num_envs = 16
    env_cfg.env.env_spacing = 4.0
    
    # 启用复杂地形
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.curriculum = True  # 启用课程学习，地形有难度
    env_cfg.terrain.max_init_terrain_level = 9
    
    # 地形比例：只保留有挑战性的地形
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.4, 0.4, 0.2]  # [平滑, 粗糙, 上楼梯, 下楼梯, 离散]
    
    # 相机俯瞰
    env_cfg.viewer.pos = [12, 12, 10]
    env_cfg.viewer.lookat = [6, 6, 0]
    
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("="*60)
    print("🏔️ 地形挑战测试")
    print("="*60)
    print("地形类型分布:")
    print("  - 上楼梯 (40%)")
    print("  - 下楼梯 (40%)")
    print("  - 离散地形 (20%)")
    print("="*60)
    print("观察：机器人在不同地形的适应能力")
    print("="*60)
    
    env = task_registry.get_task_class('a1_custom_v2')(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device, headless=False
    )
    
    print("\n加载模型...")
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name='a1_custom_v2', args=args)
    model_path = '/home/ubuntu/data/最终版/legged_gym/logs/a1_custom_v2/Mar31_23-07-49_/model_5000.pt'
    ppo_runner.load(model_path)
    
    print("\n按数字键 1-4 切换视角关注不同区域")
    print("ESC 退出\n")
    
    obs = env.get_observations()
    steps = 0
    
    # 注册视角切换按键
    for i in range(1, 5):
        env.gym.subscribe_viewer_keyboard_event(env.viewer, getattr(gymapi, f'KEY_{i}'), f"VIEW_{i}")
    
    with torch.no_grad():
        while True:
            if env.gym.query_viewer_has_closed(env.viewer):
                break
            
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "QUIT":
                    print("\n退出")
                    return
                elif evt.action.startswith("VIEW_") and evt.value > 0:
                    # 切换视角到不同区域
                    region = int(evt.action[-1])
                    x, y = (region - 1) % 2 * 8, (region - 1) // 2 * 8
                    env_cfg.viewer.pos = [x + 4, y + 4, 6]
                    env_cfg.viewer.lookat = [x + 2, y + 2, 0]
                    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
                    print(f"切换到区域 {region}")
            
            actions = ppo_runner.alg.actor_critic.act_inference(obs)
            obs, _, rewards, dones, _ = env.step(actions)
            
            steps += 1
            
            # 统计存活率和平均奖励
            if steps % 100 == 0:
                alive = (dones == 0).sum().item()
                avg_reward = rewards.mean().item()
                print(f"步数: {steps:4d} | 存活: {alive:2d}/{env_cfg.env.num_envs} | "
                      f"平均奖励: {avg_reward:6.2f}")
            
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)


if __name__ == '__main__':
    main()
