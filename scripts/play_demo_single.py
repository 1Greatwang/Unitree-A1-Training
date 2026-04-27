"""单环境演示测试"""
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
    env_cfg.env.num_envs = 1  # 单环境
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 9
    
    # 使用默认相机位置
    env_cfg.viewer.pos = [10, 0, 6]
    env_cfg.viewer.lookat = [11, 5, 3]
    
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("创建单环境...")
    env = task_registry.get_task_class('a1_custom_v2')(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device, headless=False
    )
    
    print("加载模型...")
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name='a1_custom_v2', args=args)
    model_path = '/home/ubuntu/data/最终版/legged_gym/logs/a1_custom_v2/Mar31_23-07-49_/model_5000.pt'
    
    if os.path.exists(model_path):
        ppo_runner.load(model_path)
        print(f"已加载: {os.path.basename(model_path)}")
    
    print("演示开始 - 按 ESC 退出")
    obs = env.get_observations()
    
    with torch.no_grad():
        while True:
            if env.gym.query_viewer_has_closed(env.viewer):
                break
            
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "QUIT":
                    print("\n退出")
                    return
            
            actions = ppo_runner.alg.actor_critic.act_inference(obs)
            obs, _, rewards, dones, _ = env.step(actions)
            
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)

if __name__ == '__main__':
    main()
