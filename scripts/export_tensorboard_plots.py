"""
TensorBoard 图表批量导出脚本
将所有训练曲线导出为高清 PNG 图片

用法:
    python export_tensorboard_plots.py
    
输出:
    videos/tensorboard_export/ 目录下的所有图表
"""

import os
import sys
import glob
from datetime import datetime

# 设置日志路径
LOG_DIR = "/root/a1_robot_project/legged_gym/logs/a1_custom_v2"
OUTPUT_DIR = "/root/a1_robot_project/videos/tensorboard_export"

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
    """加载 TensorBoard 数据"""
    event_files = glob.glob(os.path.join(log_dir, "*/events.out.tfevents.*"))
    if not event_files:
        print(f"[错误] 在 {log_dir} 中找不到日志文件")
        return None
    
    # 使用最新的日志文件
    latest_file = max(event_files, key=os.path.getctime)
    print(f"[信息] 加载日志: {latest_file}")
    
    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()
    
    return ea


def get_all_scalars(ea):
    """获取所有标量数据"""
    scalar_tags = ea.Tags()['scalars']
    print(f"[信息] 找到 {len(scalar_tags)} 个标量指标")
    return scalar_tags


def plot_scalar(ea, tag, save_path, title=None):
    """绘制单个标量图表"""
    try:
        data = ea.Scalars(tag)
        steps = [x.step for x in data]
        values = [x.value for x in data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, linewidth=1.5)
        plt.xlabel('Training Iteration', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        
        # 设置标题
        if title:
            plt.title(title, fontsize=14, fontweight='bold')
        else:
            plt.title(tag, fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"  [警告] 无法绘制 {tag}: {e}")
        return False


def categorize_tags(tags):
    """对标签进行分类"""
    categories = {
        'reward': [],
        'loss': [],
        'episode': [],
        'train': [],
        'other': []
    }
    
    for tag in tags:
        tag_lower = tag.lower()
        if 'reward' in tag_lower:
            categories['reward'].append(tag)
        elif 'loss' in tag_lower:
            categories['loss'].append(tag)
        elif 'episode' in tag_lower:
            categories['episode'].append(tag)
        elif 'train' in tag_lower:
            categories['train'].append(tag)
        else:
            categories['other'].append(tag)
    
    return categories


def create_summary_grid(ea, tags, save_path, max_plots=9):
    """创建汇总网格图"""
    try:
        # 选择前 max_plots 个标签
        selected_tags = tags[:max_plots]
        n_plots = len(selected_tags)
        
        # 计算网格大小
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        for idx, tag in enumerate(selected_tags):
            try:
                data = ea.Scalars(tag)
                steps = [x.step for x in data]
                values = [x.value for x in data]
                
                ax = axes[idx] if n_plots > 1 else axes[0]
                ax.plot(steps, values, linewidth=1.5)
                ax.set_title(tag, fontsize=10, fontweight='bold')
                ax.set_xlabel('Iteration', fontsize=8)
                ax.set_ylabel('Value', fontsize=8)
                ax.grid(True, alpha=0.3)
            except Exception as e:
                ax = axes[idx] if n_plots > 1 else axes[0]
                ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
        
        # 隐藏多余的子图
        for idx in range(n_plots, len(axes) if isinstance(axes, np.ndarray) else 1):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"  [警告] 无法创建汇总图: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("TensorBoard 图表批量导出工具")
    print("=" * 70)
    print()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载数据
    print(f"[信息] 日志目录: {LOG_DIR}")
    print(f"[信息] 输出目录: {OUTPUT_DIR}")
    print()
    
    ea = load_tensorboard_data(LOG_DIR)
    if ea is None:
        return
    
    # 获取所有标量
    tags = get_all_scalars(ea)
    if not tags:
        print("[错误] 没有找到任何标量数据")
        return
    
    print()
    
    # 分类标签
    categories = categorize_tags(tags)
    
    # 导出各类图表
    total_exported = 0
    
    for category, cat_tags in categories.items():
        if not cat_tags:
            continue
        
        print(f"\n导出 {category.upper()} 类别 ({len(cat_tags)} 个图表)...")
        
        # 创建类别子目录
        cat_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        for tag in cat_tags:
            # 清理文件名
            safe_name = tag.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(cat_dir, f"{safe_name}_{timestamp}.png")
            
            if plot_scalar(ea, tag, save_path):
                total_exported += 1
                print(f"  ✓ {tag}")
    
    # 创建汇总网格图
    print("\n创建汇总网格图...")
    important_tags = [t for t in tags if any(x in t.lower() for x in ['reward', 'length', 'loss'])]
    if important_tags:
        summary_path = os.path.join(OUTPUT_DIR, f"summary_grid_{timestamp}.png")
        if create_summary_grid(ea, important_tags, summary_path):
            print(f"  ✓ 汇总图已保存")
    
    # 生成 HTML 索引文件
    print("\n生成 HTML 索引文件...")
    html_path = os.path.join(OUTPUT_DIR, f"index_{timestamp}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write(f"<title>TensorBoard Export - {timestamp}</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }\n")
        f.write("h1 { color: #333; }\n")
        f.write("h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }\n")
        f.write(".category { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n")
        f.write(".plot-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }\n")
        f.write(".plot-item { text-align: center; }\n")
        f.write(".plot-item img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }\n")
        f.write(".plot-item p { margin-top: 10px; color: #666; font-size: 14px; }\n")
        f.write("</style>\n</head>\n<body>\n")
        f.write(f"<h1>TensorBoard 训练图表导出</h1>\n")
        f.write(f"<p>导出时间: {timestamp}</p>\n")
        f.write(f"<p>总共导出: {total_exported} 个图表</p>\n")
        
        for category in ['reward', 'loss', 'episode', 'train', 'other']:
            cat_dir = os.path.join(OUTPUT_DIR, category)
            if os.path.exists(cat_dir):
                images = glob.glob(os.path.join(cat_dir, f"*_{timestamp}.png"))
                if images:
                    f.write(f"<div class='category'>\n")
                    f.write(f"<h2>{category.upper()} ({len(images)} 个)</h2>\n")
                    f.write(f"<div class='plot-grid'>\n")
                    for img_path in sorted(images):
                        img_name = os.path.basename(img_path)
                        tag_name = img_name.replace(f'_{timestamp}.png', '').replace('_', '/')
                        rel_path = os.path.relpath(img_path, OUTPUT_DIR)
                        f.write(f"<div class='plot-item'>\n")
                        f.write(f"<img src='{rel_path}' alt='{tag_name}' loading='lazy'>\n")
                        f.write(f"<p>{tag_name}</p>\n")
                        f.write(f"</div>\n")
                    f.write(f"</div>\n")
                    f.write(f"</div>\n")
        
        f.write("</body>\n</html>")
    
    print(f"  ✓ HTML 索引: {html_path}")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("导出完成!")
    print("=" * 70)
    print(f"\n总计导出: {total_exported} 个图表")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n查看方式:")
    print(f"  1. 直接查看图片: {OUTPUT_DIR}/")
    print(f"  2. 浏览器查看:   打开 {html_path}")
    print()
    print("文件列表:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        level = root.replace(OUTPUT_DIR, '').count(os.sep)
        indent = '  ' * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = '  ' * (level + 1)
        for file in sorted(files)[:10]:  # 只显示前10个
            print(f'{subindent}{file}')
        if len(files) > 10:
            print(f'{subindent}... and {len(files)-10} more')


if __name__ == '__main__':
    main()
