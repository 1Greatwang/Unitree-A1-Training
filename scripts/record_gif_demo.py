"""
A1机器人 GIF 录制脚本 - 无需显示界面
使用软件渲染录制视频

输出: videos/a1_walking_demo.gif
"""

import os
import sys

# Isaac Gym 必须在 torch 之前导入
from isaacgym import gymapi, gymtorch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legged_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))

import torch
import numpy as np
from PIL import Image
import glob

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot import LeggedRobot as A1
from a1_custom_config import A1CustomCfg, A1CustomCfgPPO


def render_to_file(env, filename):
    """将当前画面保存为图片"""
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    
    # 使用 Isaac Gym 的查看器截图功能
    # 由于没有相机，我们通过获取 Viewer 的图像
    if hasattr(env, 'viewer') and env.viewer is not None:
        # 这里需要使用其他方法
        pass
    return False


def main():
    print("="*60)
    print("🎬 A1机器人 GIF 录制")
    print("="*60)
    
    # 注册任务
    task_registry.register('a1_custom', A1, A1CustomCfg(), A1CustomCfgPPO())
    
    # 获取参数 - 强制使用 CPU 渲染避免 GLX 问题
    args = get_args()
    args.task = 'a1_custom'
    args.headless = True
    args.graphics_device_id = -1  # CPU 渲染
    
    # 配置
    env_cfg, train_cfg = task_registry.get_cfgs('a1_custom')
    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    
    # 视频目录
    video_dir = os.path.join(os.path.dirname(__file__), '..', 'videos')
    os.makedirs(video_dir, exist_ok=True)
    
    # 创建环境
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("创建环境...")
    env = task_registry.get_task_class('a1_custom')(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device, headless=True
    )
    
    # 加载模型
    print("加载模型...")
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name='a1_custom', args=args)
    
    # 找最新模型 (V2)
    model_files = glob.glob('/root/a1_robot_project/legged_gym/logs/a1_custom_v2/*/model_*.pt')
    model_files = [f for f in model_files if 'model_0.pt' not in f]
    if model_files:
        load_path = max(model_files, key=os.path.getmtime)
        ppo_runner.load(load_path)
        checkpoint = int(os.path.basename(load_path).split('_')[1].split('.')[0])
        print(f"✅ 已加载: model_{checkpoint}.pt")
    else:
        print("❌ 未找到模型")
        return
    
    # 录制参数
    max_steps = 1500
    fps = 20
    
    print(f"\n录制 {max_steps} 帧...")
    
    # 存储机器人位置用于绘制动画
    positions = []
    velocities = []
    
    obs = env.get_observations()
    total_reward = 0
    
    with torch.no_grad():
        for step in range(max_steps):
            actions = ppo_runner.alg.actor_critic.act_inference(obs)
            obs, _, rewards, dones, _ = env.step(actions)
            
            total_reward += rewards.item()
            
            # 记录机器人位置
            pos = env.root_states[0, :3].cpu().numpy()
            vel = env.root_states[0, 7:10].cpu().numpy()  # 线速度
            positions.append(pos)
            velocities.append(vel)
            
            if dones.item():
                break
            
            if step % 50 == 0:
                print(f"  录制: {step}/{max_steps} 步...", end='\r')
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    print(f"\n✅ 录制完成: {len(positions)} 帧, 奖励: {total_reward:.2f}")
    
    # 创建动画 GIF
    print("生成 GIF...")
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    
    frames = []
    
    for i in range(0, len(positions), 3):  # 每隔3帧取一帧
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：轨迹
        ax1.plot(positions[:i+1, 0], positions[:i+1, 1], 'b-', linewidth=2, alpha=0.7)
        ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(positions[i, 0], positions[i, 1], c='red', s=100, marker='x', label='Current', zorder=5)
        
        # 绘制速度向量
        if i > 0:
            scale = 0.5
            ax1.arrow(positions[i, 0], positions[i, 1], 
                     velocities[i, 0]*scale, velocities[i, 1]*scale,
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'A1 Robot Walking Trajectory\nStep: {i}, Reward: {total_reward:.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 右图：速度曲线
        time_steps = np.arange(len(velocities[:i+1])) * 0.02  # 50Hz
        ax2.plot(time_steps, velocities[:i+1, 0], label='Vx', alpha=0.7)
        ax2.plot(time_steps, velocities[:i+1, 1], label='Vy', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Robot Velocity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存为图片
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close()
    
    # 保存 GIF
    if frames:
        gif_path = os.path.join(video_dir, f'a1_walking_demo_{checkpoint}iters.gif')
        images = [Image.fromarray(f) for f in frames]
        images[0].save(
            gif_path, save_all=True, append_images=images[1:],
            duration=1000//fps, loop=0, optimize=True
        )
        print(f"✅ GIF 已保存: {gif_path}")
        print(f"   文件大小: {os.path.getsize(gif_path)/1024:.1f} KB")
        
        # 同时保存最后一帧为静态图
        png_path = os.path.join(video_dir, f'a1_walking_final_{checkpoint}iters.png')
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax.scatter(positions[0, 0], positions[0, 1], c='green', s=150, marker='o', label='Start', zorder=5)
        ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=150, marker='x', label='End', zorder=5)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(f'A1 Robot Walking Trajectory (Checkpoint {checkpoint})\nTotal Distance: {np.linalg.norm(positions[-1] - positions[0]):.2f}m', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 静态图已保存: {png_path}")
    
    print("\n" + "="*60)
    print("完成！下载文件查看:")
    print(f"  scp root@58.144.141.111:{video_dir}/a1_walking_* .")
    print("="*60)


if __name__ == '__main__':
    main()
