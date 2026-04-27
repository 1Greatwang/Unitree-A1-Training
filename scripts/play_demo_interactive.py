"""
A1机器人交互式演示脚本 - 完整相机控制版

用法:
    python play_demo_interactive.py --checkpoint=5000
    
相机控制 (键盘):
    WASD          - 前后左右移动相机
    Q/E           - 上升/下降
    方向键↑↓      - 俯仰角 (Pitch)
    方向键←→      - 偏航角 (Yaw)
    +/-           - 缩放 (改变视野距离)
    
视角模式:
    1             - 自由视角 (默认)
    2             - 跟踪模式 (跟随第一个机器人)
    3             - 俯瞰模式 (鸟瞰)
    4             - 侧面跟随模式
    
其他功能:
    TAB           - 切换跟踪的机器人
    R             - 重置相机到默认位置
    C             - 截图保存到 videos/screenshots/
    SPACE         - 暂停/继续
    V             - 切换渲染同步 (加速/减速)
    ESC           - 退出
    
鼠标控制 (Isaac Gym 原生):
    左键拖拽      - 旋转视角
    右键拖拽      - 平移视角
    滚轮          - 缩放
"""

import os
import sys
import math

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


class CameraController:
    """相机控制器 - 支持多种视角模式和键盘控制"""
    
    # 视角模式
    MODE_FREE = 0       # 自由视角
    MODE_FOLLOW = 1     # 跟踪模式
    MODE_TOPDOWN = 2    # 俯瞰模式
    MODE_SIDE = 3       # 侧面跟随
    
    def __init__(self, env, env_cfg):
        self.env = env
        self.cfg = env_cfg
        self.gym = env.gym
        self.viewer = env.viewer
        
        # 当前模式
        self.mode = self.MODE_FREE
        self.target_env = 0  # 跟踪的机器人编号
        
        # 相机参数
        self.pos = np.array(env_cfg.viewer.pos, dtype=np.float64)
        self.lookat = np.array(env_cfg.viewer.lookat, dtype=np.float64)
        
        # 距离和角度 (用于轨道相机)
        self.distance = np.linalg.norm(self.pos - self.lookat)
        self.yaw = math.atan2(self.lookat[1] - self.pos[1], self.lookat[0] - self.pos[0])
        self.pitch = math.asin((self.pos[2] - self.lookat[2]) / self.distance) if self.distance > 0 else 0
        
        # 移动速度
        self.move_speed = 2.0
        self.rotate_speed = 1.0
        self.zoom_speed = 0.5
        
        # 暂停状态
        self.paused = False
        
        # 截图目录
        self.screenshot_dir = os.path.join(os.path.dirname(__file__), '..', 'videos', 'screenshots')
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # 初始化键盘事件
        self._setup_keyboard_events()
        
    def _setup_keyboard_events(self):
        """注册键盘事件"""
        # 基础控制
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "MOVE_FORWARD")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "MOVE_BACKWARD")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "MOVE_LEFT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "MOVE_RIGHT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "MOVE_UP")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "MOVE_DOWN")
        
        # 旋转
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "YAW_LEFT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "YAW_RIGHT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "PITCH_UP")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "PITCH_DOWN")
        
        # 缩放
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_EQUAL, "ZOOM_IN")  # +
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_MINUS, "ZOOM_OUT") # -
        
        # 模式切换
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_1, "MODE_FREE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_2, "MODE_FOLLOW")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_3, "MODE_TOPDOWN")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_4, "MODE_SIDE")
        
        # 其他功能
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_TAB, "NEXT_ROBOT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "RESET_CAMERA")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "SCREENSHOT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "PAUSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "TOGGLE_SYNC")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        
    def update(self, dt):
        """每帧更新相机"""
        # 处理键盘事件
        events = self.gym.query_viewer_action_events(self.viewer)
        for evt in events:
            if evt.value > 0:  # 按键按下
                self._handle_key(evt.action)
        
        # 根据模式更新相机
        if self.mode == self.MODE_FOLLOW:
            self._update_follow_mode()
        elif self.MODE_SIDE:
            self._update_side_mode()
        
        # 应用相机位置
        self._apply_camera()
        
    def _handle_key(self, action):
        """处理键盘输入"""
        # 移动控制 (只在自由模式下有效)
        if self.mode == self.MODE_FREE:
            if action == "MOVE_FORWARD":
                self._move_camera(1, 0, 0)
            elif action == "MOVE_BACKWARD":
                self._move_camera(-1, 0, 0)
            elif action == "MOVE_LEFT":
                self._move_camera(0, -1, 0)
            elif action == "MOVE_RIGHT":
                self._move_camera(0, 1, 0)
            elif action == "MOVE_UP":
                self._move_camera(0, 0, 1)
            elif action == "MOVE_DOWN":
                self._move_camera(0, 0, -1)
            elif action == "YAW_LEFT":
                self.yaw += self.rotate_speed * 0.1
            elif action == "YAW_RIGHT":
                self.yaw -= self.rotate_speed * 0.1
            elif action == "PITCH_UP":
                self.pitch = min(self.pitch + self.rotate_speed * 0.1, math.pi/2 - 0.1)
            elif action == "PITCH_DOWN":
                self.pitch = max(self.pitch - self.rotate_speed * 0.1, -math.pi/2 + 0.1)
        
        # 缩放 (所有模式)
        if action == "ZOOM_IN":
            self.distance = max(self.distance - self.zoom_speed, 1.0)
        elif action == "ZOOM_OUT":
            self.distance = min(self.distance + self.zoom_speed, 50.0)
            
        # 模式切换
        elif action == "MODE_FREE":
            self.mode = self.MODE_FREE
            print("📷 模式: 自由视角")
        elif action == "MODE_FOLLOW":
            self.mode = self.MODE_FOLLOW
            print(f"📷 模式: 跟踪机器人 #{self.target_env}")
        elif action == "MODE_TOPDOWN":
            self.mode = self.MODE_TOPDOWN
            print("📷 模式: 俯瞰视角")
            self._set_topdown_view()
        elif action == "MODE_SIDE":
            self.mode = self.MODE_SIDE
            print(f"📷 模式: 侧面跟随机器人 #{self.target_env}")
            
        # 其他功能
        elif action == "NEXT_ROBOT":
            self.target_env = (self.target_env + 1) % self.env.cfg.env.num_envs
            print(f"🎯 切换到机器人 #{self.target_env}")
        elif action == "RESET_CAMERA":
            self._reset_camera()
            print("📷 相机已重置")
        elif action == "SCREENSHOT":
            self._take_screenshot()
        elif action == "PAUSE":
            self.paused = not self.paused
            print("⏸️ 暂停" if self.paused else "▶️ 继续")
        elif action == "QUIT":
            print("\n退出演示")
            sys.exit(0)
            
    def _move_camera(self, forward, right, up):
        """移动相机 (本地坐标系)"""
        forward_vec = np.array([math.cos(self.yaw), math.sin(self.yaw), 0])
        right_vec = np.array([-math.sin(self.yaw), math.cos(self.yaw), 0])
        up_vec = np.array([0, 0, 1])
        
        delta = (forward_vec * forward + right_vec * right + up_vec * up) * self.move_speed * 0.1
        self.pos += delta
        self.lookat += delta
        
    def _update_follow_mode(self):
        """更新跟踪模式相机"""
        try:
            # 获取目标机器人的位置
            root_states = self.env.root_states
            if root_states.shape[0] > self.target_env:
                target_pos = root_states[self.target_env, :3].cpu().numpy()
                
                # 计算相机位置 (从后方跟随)
                offset = np.array([
                    -math.cos(self.yaw) * self.distance,
                    -math.sin(self.yaw) * self.distance,
                    self.distance * 0.5  # 稍微高一点
                ])
                
                self.pos = target_pos + offset
                self.lookat = target_pos + np.array([0, 0, 0.5])  # 看机器人的中心
        except:
            pass
            
    def _update_side_mode(self):
        """更新侧面跟随模式"""
        try:
            root_states = self.env.root_states
            if root_states.shape[0] > self.target_env:
                target_pos = root_states[self.target_env, :3].cpu().numpy()
                
                # 从侧面跟随
                offset = np.array([0, -self.distance, self.distance * 0.3])
                self.pos = target_pos + offset
                self.lookat = target_pos + np.array([0, 0, 0.5])
        except:
            pass
            
    def _set_topdown_view(self):
        """设置俯瞰视角"""
        # 计算所有机器人的中心
        try:
            root_states = self.env.root_states[:, :3].cpu().numpy()
            center = np.mean(root_states, axis=0)
        except:
            center = np.array([0, 0, 0])
            
        self.pos = center + np.array([0, 0, self.distance])
        self.lookat = center
        
    def _reset_camera(self):
        """重置相机到默认位置"""
        self.pos = np.array(self.cfg.viewer.pos, dtype=np.float64)
        self.lookat = np.array(self.cfg.viewer.lookat, dtype=np.float64)
        self.distance = np.linalg.norm(self.pos - self.lookat)
        self.yaw = math.atan2(self.lookat[1] - self.pos[1], self.lookat[0] - self.pos[0])
        self.pitch = math.asin((self.pos[2] - self.lookat[2]) / self.distance) if self.distance > 0 else 0
        self.mode = self.MODE_FREE
        
    def _apply_camera(self):
        """应用相机位置到 viewer"""
        self.env.set_camera(self.pos, self.lookat)
        
    def _take_screenshot(self):
        """截图保存"""
        try:
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            self.gym.write_viewer_image_to_file(self.viewer, filepath)
            print(f"📸 截图已保存: {filepath}")
        except Exception as e:
            print(f"截图失败: {e}")
            
    def print_help(self):
        """打印控制帮助"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    🎮 相机控制指南                            ║
╠══════════════════════════════════════════════════════════════╣
║  移动: WASD (前后左右)  Q/E (上升/下降)                       ║
║  旋转: 方向键 (↑↓俯仰, ←→偏航)                              ║
║  缩放: +/-                                                   ║
╠══════════════════════════════════════════════════════════════╣
║  视角模式: 1=自由  2=跟踪  3=俯瞰  4=侧面                     ║
║  功能: TAB=切换机器人  R=重置  C=截图  SPACE=暂停  ESC=退出   ║
╠══════════════════════════════════════════════════════════════╣
║  鼠标: 左键拖拽旋转  右键拖拽平移  滚轮缩放                  ║
╚══════════════════════════════════════════════════════════════╝
        """)


def main():
    # 注册任务
    task_registry.register('a1_custom_v2', A1, A1CustomCfgV2(), A1CustomCfgPPOV2())
    
    # 获取参数
    args = get_args()
    args.task = 'a1_custom_v2'
    args.headless = False
    
    # 配置
    env_cfg, train_cfg = task_registry.get_cfgs('a1_custom_v2')
    env_cfg.env.num_envs = 9  # 多环境演示
    env_cfg.env.env_spacing = 4.0
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 9
    
    # 初始相机位置
    env_cfg.viewer.pos = [8, 8, 10]
    env_cfg.viewer.lookat = [4, 4, 0]
    
    # 创建环境
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("="*60)
    print("A1 四足机器人 - 交互式演示 (完整相机控制)")
    print("="*60)
    
    env = task_registry.get_task_class('a1_custom_v2')(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device, headless=False
    )
    
    # 加载模型
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
    
    # 创建相机控制器
    camera = CameraController(env, env_cfg)
    camera.print_help()
    
    print("\n" + "="*60)
    print("演示开始 - 按 ESC 退出, H 查看帮助")
    print("="*60 + "\n")
    
    # 运行演示
    obs = env.get_observations()
    steps = 0
    episode_rewards = torch.zeros(env_cfg.env.num_envs, device=env.device)
    
    with torch.no_grad():
        while True:
            if env.gym.query_viewer_has_closed(env.viewer):
                break
            
            # 更新相机
            camera.update(env.dt)
            
            if camera.paused:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.draw_viewer(env.viewer, env.sim, True)
                continue
            
            # 执行动作
            actions = ppo_runner.alg.actor_critic.act_inference(obs)
            obs, _, rewards, dones, _ = env.step(actions)
            
            episode_rewards += rewards
            steps += 1
            
            # 渲染
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            
            # 回合结束提示
            done_envs = dones.nonzero(as_tuple=True)[0]
            if len(done_envs) > 0:
                for env_id in done_envs:
                    print(f"🐕 机器人{env_id.item()}: {steps}步, 奖励:{episode_rewards[env_id].item():.2f}")
                    episode_rewards[env_id] = 0


if __name__ == '__main__':
    main()
