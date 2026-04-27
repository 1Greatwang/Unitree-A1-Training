"""
高清录制演示 - 生成高质量视频用于展示
录制特定视角，自动旋转相机，生成专业展示效果
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legged_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

from isaacgym import gymapi, gymtorch
import torch
import numpy as np
from PIL import Image
from datetime import datetime

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot import LeggedRobot as A1
from a1_custom_config_v2 import A1CustomCfgV2, A1CustomCfgPPOV2


def main():
    task_registry.register('a1_custom_v2', A1, A1CustomCfgV2(), A1CustomCfgPPOV2())
    
    args = get_args()
    args.task = 'a1_custom_v2'
    args.headless = False  # 需要图形界面来捕获画面
    
    env_cfg, train_cfg = task_registry.get_cfgs('a1_custom_v2')
    
    # 设置环境数
    env_cfg.env.num_envs = 16
    env_cfg.env.env_spacing = 2.5
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    
    # 初始相机位置
    env_cfg.viewer.pos = [10, 0, 6]
    env_cfg.viewer.lookat = [8, 0, 0]
    
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("="*60)
    print("🎬 高清录制模式")
    print("="*60)
    
    env = task_registry.get_task_class('a1_custom_v2')(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device, headless=False
    )
    
    print("加载模型...")
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name='a1_custom_v2', args=args)
    model_path = '/home/ubuntu/data/最终版/legged_gym/logs/a1_custom_v2/Mar31_23-07-49_/model_5000.pt'
    ppo_runner.load(model_path)
    
    # 录制参数
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'videos', 'recordings')
    os.makedirs(output_dir, exist_ok=True)
    
    total_frames = 300  # 录制 300 帧 (约 10 秒 @ 30fps)
    rotate_camera = True  # 相机旋转
    
    print(f"\n开始录制 {total_frames} 帧...")
    print("相机将自动旋转展示全景")
    print("按 Ctrl+C 可提前结束\n")
    
    obs = env.get_observations()
    frames = []
    camera_angle = 0
    
    # 固定指令让机器人向前跑
    env.commands[:, 0] = 1.5  # x方向速度
    env.commands[:, 1] = 0.0  # y方向速度
    env.commands[:, 2] = 0.0  # 旋转速度
    
    try:
        with torch.no_grad():
            for i in range(total_frames):
                actions = ppo_runner.alg.actor_critic.act_inference(obs)
                obs, _, _, _, _ = env.step(actions)
                
                # 相机旋转
                if rotate_camera:
                    camera_angle += 2 * np.pi / total_frames  # 完整转一圈
                    radius = 10
                    cam_x = 8 + radius * np.cos(camera_angle)
                    cam_y = radius * np.sin(camera_angle)
                    env.set_camera([cam_x, cam_y, 6], [8, 0, 0])
                
                # 渲染
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.draw_viewer(env.viewer, env.sim, True)
                
                # 截图 (每隔几帧)
                if i % 2 == 0:  # 15 fps
                    # 使用 viewer 截图
                    filename = f"/tmp/frame_{i:04d}.png"
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    frames.append(filename)
                
                if (i + 1) % 50 == 0:
                    print(f"进度: {i+1}/{total_frames} ({(i+1)/total_frames*100:.0f}%)")
                    
    except KeyboardInterrupt:
        print("\n录制中断")
    
    # 保存为 GIF
    if frames:
        print(f"\n生成 GIF ({len(frames)} 帧)...")
        output_file = os.path.join(output_dir, f"demo_{datetime.now().strftime('%m%d_%H%M')}.gif")
        
        images = [Image.open(f) for f in frames]
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            duration=67,  # 15 fps
            loop=0
        )
        print(f"✓ 已保存: {output_file}")
        
        # 清理临时文件
        for f in frames:
            os.remove(f)
    
    print("\n录制完成！")


if __name__ == '__main__':
    main()
