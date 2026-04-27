#!/usr/bin/env python3
"""
环境检查脚本 - 验证所有依赖是否正确安装
用法:
    python3 check_env.py
"""

import sys
import os

def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_check(name, success, message=""):
    status = "✅" if success else "❌"
    print(f"{status} {name}", end="")
    if message:
        print(f" - {message}")
    else:
        print()

def check_python():
    """检查Python版本"""
    version = sys.version_info
    required = (3, 8)
    success = version >= required
    print_check(
        f"Python {version.major}.{version.minor}.{version.micro}",
        success,
        "需要 >= 3.8" if not success else "版本正常"
    )
    return success

def check_pytorch():
    """检查PyTorch安装"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        print_check(f"PyTorch {version}", True)
        
        if cuda_available:
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            print_check(f"CUDA {cuda_version}", True, device_name)
        else:
            print_check("CUDA", False, "GPU不可用，训练将非常缓慢")
        
        return True
    except ImportError:
        print_check("PyTorch", False, "未安装")
        return False

def check_isaacgym():
    """检查Isaac Gym安装"""
    try:
        import isaacgym
        print_check("Isaac Gym", True)
        return True
    except ImportError:
        print_check("Isaac Gym", False, "请安装: cd ~/isaacgym/python && pip install -e .")
        return False

def check_legged_gym():
    """检查Legged Gym安装"""
    try:
        import legged_gym
        print_check("Legged Gym", True)
        return True
    except ImportError:
        print_check("Legged Gym", False, "请安装: cd ~/legged_gym && pip install -e .")
        return False

def check_rsl_rl():
    """检查RSL-RL安装"""
    try:
        import rsl_rl
        print_check("RSL-RL", True)
        return True
    except ImportError:
        print_check("RSL-RL", False, "请安装: cd ~/rsl_rl && pip install -e .")
        return False

def check_tensorboard():
    """检查TensorBoard安装"""
    try:
        import tensorboard
        print_check("TensorBoard", True)
        return True
    except ImportError:
        print_check("TensorBoard", False, "请安装: pip install tensorboard")
        return False

def check_matplotlib():
    """检查Matplotlib安装"""
    try:
        import matplotlib
        print_check("Matplotlib", True)
        return True
    except ImportError:
        print_check("Matplotlib", False, "请安装: pip install matplotlib")
        return False

def check_project_structure():
    """检查项目结构"""
    print_header("项目结构检查")
    
    required_files = [
        "../configs/a1_custom_config.py",
        "train_a1.py",
        "play_demo.py",
        "plot_training.py"
    ]
    
    all_ok = True
    for file in required_files:
        exists = os.path.exists(file)
        print_check(file, exists)
        if not exists:
            all_ok = False
    
    return all_ok

def check_gpu_memory():
    """检查GPU显存"""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / 1e9
            print_check(f"GPU显存: {total_gb:.2f} GB", total_gb >= 4, 
                       "建议4GB+" if total_gb < 4 else "足够")
            return total_gb >= 4
    except:
        pass
    return False

def main():
    print_header("A1机器人训练项目 - 环境检查")
    print("项目目录:", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 检查依赖
    print_header("Python依赖检查")
    results = {
        "python": check_python(),
        "pytorch": check_pytorch(),
        "isaacgym": check_isaacgym(),
        "legged_gym": check_legged_gym(),
        "rsl_rl": check_rsl_rl(),
        "tensorboard": check_tensorboard(),
        "matplotlib": check_matplotlib(),
    }
    
    # 检查GPU
    print_header("GPU检查")
    check_gpu_memory()
    
    # 检查项目结构
    check_project_structure()
    
    # 总结
    print_header("检查结果总结")
    
    core_deps = ["python", "pytorch", "isaacgym", "legged_gym", "rsl_rl"]
    core_ok = all(results[d] for d in core_deps)
    
    if core_ok:
        print("✅ 核心依赖已就绪，可以开始训练!")
        print("\n下一步:")
        print("  1. 测试训练: python3 train_a1.py --headless --max_iterations=5")
        print("  2. 完整训练: python3 train_a1.py --headless")
        print("  3. 运行演示: python3 play_demo.py")
    else:
        print("❌ 缺少必要的依赖，请按照AGENTS.md安装")
        print("\n快速安装命令:")
        print("  conda activate a1_robot")
        print("  pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117")
        print("  cd ~/isaacgym/python && pip install -e .")
        print("  cd ~/legged_gym && pip install -e .")
        print("  cd ~/rsl_rl && pip install -e .")
        print("  pip install tensorboard matplotlib")
    
    return 0 if core_ok else 1

if __name__ == "__main__":
    sys.exit(main())
