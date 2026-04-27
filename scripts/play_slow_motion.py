"""
慢动作演示 - 展示机器人步态细节
使用正常的物理仿真，只降低渲染速度

控制:
    +/- 调整渲染速度 (0.2x - 2.0x)
    P   暂停/继续
    ESC 退出
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legged_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from isaacgym import gymapi, gymtorch
import torch
import time
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
    env_cfg.env.num_envs = 1  # 单环境，专注观察
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    # 保持默认 decimation=4，不改控制频率
    
    # 相机近距离观察
    env_cfg.viewer.pos = [3, 3, 2]
    env_cfg.viewer.lookat = [0, 0, 0.5]
    
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("="*60)
    print("🎥 慢动作演示 - 观察步态细节")
    print("="*60)
    print("控制:")
    print("  +/- : 调整渲染速度 (不改变物理仿真)")
    print("  P   : 暂停")
    print("  ESC : 退出")
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
    
    # 注册按键
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_EQUAL, "SPEED_UP")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_MINUS, "SPEED_DOWN")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_P, "PAUSE")
    
    obs = env.get_observations()
    render_speed = 0.5  # 默认 0.5x 慢速渲染
    paused = False
    
    print(f"\n当前渲染速度: {render_speed:.1f}x")
    print("注意：物理仿真保持正常，只降低渲染速度\n")
    
    with torch.no_grad():
        while True:
            if env.gym.query_viewer_has_closed(env.viewer):
                break
            
            # 处理事件
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "QUIT":
                    return
                elif evt.action == "SPEED_UP" and evt.value > 0:
                    render_speed = min(render_speed + 0.1, 2.0)
                    print(f"渲染速度: {render_speed:.1f}x")
                elif evt.action == "SPEED_DOWN" and evt.value > 0:
                    render_speed = max(render_speed - 0.1, 0.2)
                    print(f"渲染速度: {render_speed:.1f}x")
                elif evt.action == "PAUSE" and evt.value > 0:
                    paused = not paused
                    print("⏸️ 暂停" if paused else "▶️ 继续")
            
            if not paused:
                # 正常物理仿真步
                actions = ppo_runner.alg.actor_critic.act_inference(obs)
                obs, _, _, _, _ = env.step(actions)
            
            # 渲染（带速度控制）
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            
            # 通过控制渲染间隔实现慢动作（不影响物理）
            if render_speed < 1.0:
                time.sleep((1.0 / render_speed - 1.0) * 0.016)  # 假设 60fps


if __name__ == '__main__':
    main()
