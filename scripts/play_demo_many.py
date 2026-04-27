"""
A1机器人 - 多狗演示（几十上百条狗）
使用简化地形，专注展示机器人动作
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
    
    # 关键：大量环境 + 简化地形
    env_cfg.env.num_envs = 64  # 64 条狗！8x8 网格
    env_cfg.env.env_spacing = 2.0  # 紧凑排列
    
    # 简化地形 - 使用平面，不生成复杂地形
    env_cfg.terrain.mesh_type = 'plane'  # 平面地形，不吃性能
    env_cfg.terrain.curriculum = False
    
    # 或者用地形但大幅减少尺寸
    # env_cfg.terrain.mesh_type = 'trimesh'
    # env_cfg.terrain.num_rows = 2  # 最少行数
    # env_cfg.terrain.num_cols = 2  # 最少列数
    
    # 相机设置 - 俯瞰视角
    env_cfg.viewer.pos = [10, 10, 15]
    env_cfg.viewer.lookat = [8, 8, 0]
    
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("="*60)
    print(f"🐕 A1 四足机器人 - {env_cfg.env.num_envs} 条狗同时演示！")
    print("="*60)
    print("地形: 平面 (简化渲染)")
    print(f"布局: {int(env_cfg.env.num_envs**0.5)}x{int(env_cfg.env.num_envs**0.5)} 网格")
    print("="*60)
    
    print("\n创建环境...")
    env = task_registry.get_task_class('a1_custom_v2')(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device, headless=False
    )
    
    print("加载模型...")
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name='a1_custom_v2', args=args)
    
    checkpoint = args.checkpoint if hasattr(args, 'checkpoint') and args.checkpoint else -1
    if checkpoint > 0:
        model_path = f'/home/ubuntu/data/最终版/legged_gym/logs/a1_custom_v2/Mar31_23-07-49_/model_{checkpoint}.pt'
    else:
        model_path = '/home/ubuntu/data/最终版/legged_gym/logs/a1_custom_v2/Mar31_23-07-49_/model_5000.pt'
    
    if os.path.exists(model_path):
        ppo_runner.load(model_path)
        print(f"✓ 已加载: {os.path.basename(model_path)}")
    else:
        print(f"✗ 未找到模型")
        return
    
    print("\n" + "="*60)
    print("演示开始 - 按 ESC 退出，V 切换渲染同步")
    print("="*60 + "\n")
    
    obs = env.get_observations()
    steps = 0
    
    with torch.no_grad():
        while True:
            if env.gym.query_viewer_has_closed(env.viewer):
                break
            
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "QUIT":
                    print("\n退出")
                    return
                elif evt.action == "toggle_viewer_sync":
                    env.enable_viewer_sync = not env.enable_viewer_sync
                    print(f"渲染同步: {'开启' if env.enable_viewer_sync else '关闭(加速)'}")
            
            actions = ppo_runner.alg.actor_critic.act_inference(obs)
            obs, _, rewards, dones, _ = env.step(actions)
            
            steps += 1
            
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            
            # 每100步打印一次
            if steps % 100 == 0:
                alive = env.reset_buf.logical_not().sum().item()
                print(f"步数: {steps}, 存活: {alive}/{env_cfg.env.num_envs}")


if __name__ == '__main__':
    main()
