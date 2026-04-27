"""
A1机器人演示录制脚本 V2
录制训练好的模型为 GIF/视频

用法:
    python record_demo_v2.py --checkpoint=5000 --steps=500
"""

import os
import sys

# 设置路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legged_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from isaacgym import gymapi, gymtorch

import torch
import numpy as np
from PIL import Image
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
    args.headless = True  # 无头模式
    
    # 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs('a1_custom_v2')
    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 9
    
    # 创建环境
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("="*60)
    print("A1 四足机器人 - 录制演示")
    print("="*60)
    
    print("\n创建环境...")
    env = task_registry.get_task_class('a1_custom_v2')(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device, headless=True
    )
    
    # 加载模型
    print("加载模型...")
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name='a1_custom_v2', args=args)
    
    checkpoint = args.checkpoint if hasattr(args, 'checkpoint') and args.checkpoint else -1
    load_path = get_load_path(ppo_runner.log_dir, load_run=-1, checkpoint=checkpoint)
    if load_path:
        ppo_runner.load(load_path)
        print(f"✓ 已加载: {os.path.basename(load_path)}")
    else:
        print("✗ 未找到模型")
        return
    
    # 录制参数
    steps = args.record_steps if hasattr(args, 'record_steps') and args.record_steps else 300
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'videos')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'a1_demo_v2_{checkpoint if checkpoint > 0 else "latest"}.gif')
    
    print(f"\n开始录制 {steps} 帧...")
    
    obs = env.get_observations()
    frames = []
    
    with torch.no_grad():
        for i in range(steps):
            actions = ppo_runner.alg.actor_critic.act_inference(obs)
            obs, _, rewards, dones, _ = env.step(actions)
            
            # 渲染帧
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            
            # 获取图像
            if i % 3 == 0:  # 每3帧取一帧，减小GIF大小
                # 使用 viewer 相机获取图像
                img = env.gym.get_camera_image(env.sim, env.envs[0], 0, gymapi.IMAGE_COLOR)
                if img is not None and img.shape[0] > 0:
                    frames.append(Image.fromarray(img[:, :, :3]))
            
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{steps}")
    
    # 保存 GIF
    if frames:
        print(f"\n保存 GIF: {output_path}")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # 50ms = 20fps
            loop=0
        )
        print(f"✓ 录制完成: {len(frames)} 帧")
    else:
        print("✗ 没有捕获到帧")


if __name__ == '__main__':
    main()
