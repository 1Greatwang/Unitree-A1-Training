"""
生成训练曲线图用于展示
将TensorBoard数据导出为高清图片

用法:
    python plot_training.py
    # 生成图片到 videos/figures/ 目录
"""

import os
import sys
import glob
from datetime import datetime

# 尝试导入tensorboard数据读取器
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("[错误] 需要安装 tensorboard")
    print("  pip install tensorboard")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("[错误] 需要安装 matplotlib")
    print("  pip install matplotlib")
    sys.exit(1)


def load_tensorboard_data(log_dir):
    """加载TensorBoard数据"""
    # 找到最新的日志文件
    event_files = glob.glob(os.path.join(log_dir, "*/events.out.tfevents.*"))
    if not event_files:
        print(f"[错误] 在 {log_dir} 中找不到日志文件")
        return None
    
    latest_file = max(event_files, key=os.path.getctime)
    print(f"[信息] 加载日志: {latest_file}")
    
    # 读取数据
    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()
    
    return ea


def plot_reward_curve(ea, save_path):
    """绘制奖励曲线"""
    try:
        data = ea.Scalars('Train/mean_reward')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, linewidth=2, color='#2E86AB')
        plt.fill_between(steps, values, alpha=0.3, color='#2E86AB')
        plt.xlabel('Training Iteration', fontsize=12)
        plt.ylabel('Mean Reward', fontsize=12)
        plt.title('A1 Robot Training Progress', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加关键数值标注
        if len(values) > 0:
            max_reward = max(values)
            final_reward = values[-1]
            plt.axhline(y=final_reward, color='r', linestyle='--', alpha=0.5, 
                       label=f'Final: {final_reward:.2f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[保存] 奖励曲线: {save_path}")
        return True
    except Exception as e:
        print(f"[警告] 无法绘制奖励曲线: {e}")
        return False


def plot_episode_length(ea, save_path):
    """绘制回合长度曲线"""
    try:
        data = ea.Scalars('Train/mean_episode_length')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, linewidth=2, color='#A23B72')
        plt.fill_between(steps, values, alpha=0.3, color='#A23B72')
        plt.xlabel('Training Iteration', fontsize=12)
        plt.ylabel('Mean Episode Length (steps)', fontsize=12)
        plt.title('Episode Length Over Training', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加目标线
        plt.axhline(y=400, color='g', linestyle='--', alpha=0.5, label='Target: 400 steps')
        if len(values) > 0:
            plt.axhline(y=values[-1], color='r', linestyle='--', alpha=0.5,
                       label=f'Final: {values[-1]:.0f} steps')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[保存] 回合长度曲线: {save_path}")
        return True
    except Exception as e:
        print(f"[警告] 无法绘制回合长度: {e}")
        return False


def plot_learning_curves(ea, save_path):
    """绘制学习曲线（损失函数）"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Value loss
        try:
            data = ea.Scalars('Loss/value_function')
            steps = [x.step for x in data]
            values = [x.value for x in data]
            axes[0, 0].plot(steps, values, color='#F18F01')
            axes[0, 0].set_title('Value Function Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        except:
            axes[0, 0].text(0.5, 0.5, 'No Data', ha='center', va='center')
        
        # Policy loss
        try:
            data = ea.Scalars('Loss/surrogate')
            steps = [x.step for x in data]
            values = [x.value for x in data]
            axes[0, 1].plot(steps, values, color='#C73E1D')
            axes[0, 1].set_title('Policy Loss (Surrogate)', fontweight='bold')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        except:
            axes[0, 1].text(0.5, 0.5, 'No Data', ha='center', va='center')
        
        # Reward components
        try:
            reward_keys = [
                'Episode/rew_tracking_lin_vel',
                'Episode/rew_feet_air_time',
                'Episode/rew_torques'
            ]
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            
            for key, color in zip(reward_keys, colors):
                try:
                    data = ea.Scalars(key)
                    steps = [x.step for x in data]
                    values = [x.value for x in data]
                    label = key.split('_')[-3:]  # 简化标签
                    label = '_'.join(label)
                    axes[1, 0].plot(steps, values, label=label, color=color)
                except:
                    pass
            axes[1, 0].set_title('Reward Components', fontweight='bold')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        except:
            axes[1, 0].text(0.5, 0.5, 'No Data', ha='center', va='center')
        
        # Learning rate
        try:
            data = ea.Scalars('Loss/learning_rate')
            steps = [x.step for x in data]
            values = [x.value for x in data]
            axes[1, 1].plot(steps, values, color='#3B1F2B')
            axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        except:
            axes[1, 1].text(0.5, 0.5, 'No Data', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[保存] 学习曲线: {save_path}")
        return True
    except Exception as e:
        print(f"[警告] 无法绘制学习曲线: {e}")
        return False


def create_summary(ea, save_path):
    """创建训练总结图"""
    try:
        # 获取关键数据
        reward_data = ea.Scalars('train/mean_reward')
        length_data = ea.Scalars('train/mean_episode_length')
        
        final_reward = reward_data[-1].value if reward_data else 0
        final_length = length_data[-1].value if length_data else 0
        max_reward = max([x.value for x in reward_data]) if reward_data else 0
        total_iterations = len(reward_data)
        
        # 创建总结图
        fig = plt.figure(figsize=(12, 8))
        
        # 标题
        fig.suptitle('A1 Robot Training Summary', fontsize=20, fontweight='bold', y=0.98)
        
        # 关键指标
        ax_text = plt.subplot(2, 2, 1)
        ax_text.axis('off')
        
        summary_text = f"""
Training Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Iterations: {total_iterations}
Final Reward: {final_reward:.2f}
Max Reward: {max_reward:.2f}
Final Episode Length: {final_length:.0f} steps

Status: {'✓ Training Successful' if final_reward > 20 else '⚠ Needs More Training'}

Key Metrics:
• Mean Reward: {final_reward:.2f}
• Episode Length: {final_length:.0f} steps
• Performance: {'Good' if final_reward > 30 else 'Moderate' if final_reward > 15 else 'Poor'}
        """
        
        ax_text.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
        
        # 奖励曲线（小图）
        ax_reward = plt.subplot(2, 2, 2)
        if reward_data:
            steps = [x.step for x in reward_data]
            values = [x.value for x in reward_data]
            ax_reward.plot(steps, values, linewidth=2, color='#2E86AB')
            ax_reward.set_title('Reward Curve', fontweight='bold')
            ax_reward.set_xlabel('Iteration')
            ax_reward.set_ylabel('Reward')
            ax_reward.grid(True, alpha=0.3)
        
        # 回合长度（小图）
        ax_length = plt.subplot(2, 2, 3)
        if length_data:
            steps = [x.step for x in length_data]
            values = [x.value for x in length_data]
            ax_length.plot(steps, values, linewidth=2, color='#A23B72')
            ax_length.set_title('Episode Length', fontweight='bold')
            ax_length.set_xlabel('Iteration')
            ax_length.set_ylabel('Steps')
            ax_length.grid(True, alpha=0.3)
        
        # 训练进度条
        ax_progress = plt.subplot(2, 2, 4)
        ax_progress.axis('off')
        
        progress = min(100, (total_iterations / 1500) * 100)
        progress_bar = '█' * int(progress / 5) + '░' * (20 - int(progress / 5))
        
        progress_text = f"""
Training Progress:
[{progress_bar}] {progress:.1f}%

Iterations: {total_iterations}/1500

Estimated Time Remaining:
{'N/A' if progress >= 100 else f'{((1500-total_iterations)*2.5/60):.1f} minutes'}
        """
        
        ax_progress.text(0.1, 0.5, progress_text, fontsize=11, family='monospace',
                        verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[保存] 训练总结: {save_path}")
        return True
    except Exception as e:
        print(f"[警告] 无法创建总结图: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("生成训练可视化图表")
    print("=" * 60)
    print()
    
    # 查找日志目录
    log_dir = "/root/a1_robot_project/legged_gym/logs/a1_custom_v2"
    
    if not os.path.exists(log_dir):
        print(f"[错误] 找不到日志目录: {log_dir}")
        print("请先完成训练")
        return
    
    print(f"[信息] 日志目录: {log_dir}")
    print()
    
    # 加载数据
    ea = load_tensorboard_data(log_dir)
    if ea is None:
        return
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'videos', 'figures')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[信息] 输出目录: {output_dir}")
    print()
    
    # 生成图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("正在生成图表...")
    print()
    
    files_generated = []
    
    if plot_reward_curve(ea, os.path.join(output_dir, f'reward_curve_{timestamp}.png')):
        files_generated.append('reward_curve')
    
    if plot_episode_length(ea, os.path.join(output_dir, f'episode_length_{timestamp}.png')):
        files_generated.append('episode_length')
    
    if plot_learning_curves(ea, os.path.join(output_dir, f'learning_curves_{timestamp}.png')):
        files_generated.append('learning_curves')
    
    if create_summary(ea, os.path.join(output_dir, f'training_summary_{timestamp}.png')):
        files_generated.append('training_summary')
    
    print()
    print("=" * 60)
    print("生成完成!")
    print("=" * 60)
    print()
    print(f"生成的图表保存在:")
    print(f"  {output_dir}/")
    print()
    print("文件列表:")
    for fname in os.listdir(output_dir):
        if timestamp in fname:
            print(f"  ✓ {fname}")
    print()
    print("这些图片可以直接插入到PPT中展示!")
    print()
    
    # 尝试打开文件夹（WSL2可能不支持）
    try:
        import subprocess
        subprocess.run(['explorer.exe', output_dir], check=False)
    except:
        pass


if __name__ == '__main__':
    main()
