"""
Unitree A1 四足机器人自定义配置
基于 legged_gym 的配置文件，针对 RTX 3050 Ti 4GB 显存优化

配置说明：
- num_envs: 1024 (适合4GB显存的并行环境数)
- 奖励权重: 优化后的平衡配置
- 地形: 启用课程学习，从平地到复杂地形
- 观测: 48维基础观测 + 可选地形高度
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class A1CustomCfg(LeggedRobotCfg):
    """A1机器人环境配置类"""
    
    # ==================== 环境参数 ====================
    class env(LeggedRobotCfg.env):
        num_envs = 1024              # 并行环境数（3050Ti推荐1024，可调2048）
        num_observations = 48        # 基础观测维度（不含地形高度）
        num_privileged_obs = None    # 特权观测（非对称训练时给critic，None表示不使用）
        num_actions = 12             # 12个关节
        env_spacing = 3.             # 环境间距（平面地形用）
        send_timeouts = True         # 发送超时信息给算法
        episode_length_s = 20        # 回合长度20秒
        # RMA配置

    
    # ==================== 深度相机参数 ====================
    
    # ==================== 地形参数 ====================
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'        # 地形类型: 'plane', 'heightfield', 'trimesh'
        horizontal_scale = 0.1       # 水平分辨率 [m]
        vertical_scale = 0.005       # 垂直分辨率 [m]
        border_size = 25             # 边界大小 [m]
        curriculum = True            # 启用地形课程学习
        static_friction = 1.0        # 静摩擦系数
        dynamic_friction = 1.0       # 动摩擦系数
        restitution = 0.             # 恢复系数
        
        # 地形高度测量（感知地形用）
        measure_heights = False      # 关闭以节省显存，如需地形感知设为True
        # 测量点网格（以机器人为中心）
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                             0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 
                             0.1, 0.2, 0.3, 0.4, 0.5]
        
        # 课程学习参数
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 3   # 初始最大地形等级（0-9）
        terrain_length = 8.          # 单个地形块长度 [m]
        terrain_width = 8.           # 单个地形块宽度 [m]
        num_rows = 10                # 地形行数（等级数）
        num_cols = 20                # 地形列数（类型数）
        
        # 地形类型比例 [平滑坡, 粗糙坡, 上楼梯, 下楼梯, 离散地形]
        terrain_proportions = [0.15, 0.15, 0.30, 0.25, 0.15]
        
        slope_treshold = 0.75        # 坡度阈值，超过则修正为垂直面
    
    # ==================== 指令参数 ====================
    class commands(LeggedRobotCfg.commands):
        curriculum = False           # 是否启用指令课程
        max_curriculum = 1.
        num_commands = 4             # [线速度x, 线速度y, 角速度yaw, 航向角]
        resampling_time = 10.        # 指令重新采样间隔 [s]
        heading_command = True       # 使用航向角指令模式
        
        class ranges:
            lin_vel_x = [-1.0, 1.0]      # 前后速度范围 [m/s]
            lin_vel_y = [-0.6, 0.6]      # 左右速度范围 [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # 偏航角速度 [rad/s]
            heading = [-3.14, 3.14]      # 航向角范围 [rad]
    
    # ==================== 初始状态 ====================
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]       # 初始位置 [x, y, z] [m]
        rot = [0.0, 0.0, 0.0, 1.0]   # 初始旋转 [x, y, z, w] (四元数)
        lin_vel = [0.0, 0.0, 0.0]    # 初始线速度 [m/s]
        ang_vel = [0.0, 0.0, 0.0]    # 初始角速度 [rad/s]
        
        # 默认关节角度（动作=0时的目标角度）[rad]
        default_joint_angles = {
            # 髋部关节
            'FL_hip_joint': 0.1,      # 前左
            'RL_hip_joint': 0.1,      # 后左
            'FR_hip_joint': -0.1,     # 前右
            'RR_hip_joint': -0.1,     # 后右
            # 大腿关节
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            # 小腿关节
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5
        }
    
    # ==================== 控制参数 ====================
    class control(LeggedRobotCfg.control):
        control_type = 'P'           # 控制类型: 'P'(位置), 'V'(速度), 'T'(力矩)
        stiffness = {'joint': 20.}   # PD控制P增益 [N*m/rad]
        damping = {'joint': 0.5}     # PD控制D增益 [N*m*s/rad]
        action_scale = 0.25          # 动作缩放系数
        decimation = 4               # 控制频率 = 仿真频率 / decimation
                                     # 200Hz / 4 = 50Hz控制频率
    
    # ==================== 资产参数 ====================
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"           # 足部链接名称
        penalize_contacts_on = ["thigh", "calf"]  # 惩罚这些部位的接触
        terminate_after_contacts_on = ["base"]    # 这些部位接触则终止
        self_collisions = 1          # 1禁用自碰撞, 0启用
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
        fix_base_link = False
        
        # 物理参数
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
    
    # ==================== 领域随机化 ====================
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
    
    # ==================== 奖励函数 ====================
    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True     # 只保留正奖励（避免早终止问题）
        tracking_sigma = 0.25            # 速度跟踪奖励的带宽参数
        soft_dof_pos_limit = 0.9         # 关节位置软限制（相对于URDF限制）
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25        # 目标机身高度 [m]
        max_contact_force = 100.         # 最大接触力惩罚阈值
        
        class scales:
            # ========== 任务奖励（主要目标）==========
            tracking_lin_vel = 1.0       # 跟踪线速度指令（核心）
            tracking_ang_vel = 0.5       # 跟踪角速度指令
            feet_air_time = 1.0          # 鼓励周期性步态
            
            # ========== 正则化奖励（约束行为）==========
            lin_vel_z = -2.0             # 惩罚垂直跳动
            ang_vel_xy = -0.05           # 惩罚横滚俯仰角速度
            orientation = -0.0           # 惩罚非水平姿态（设为0关闭）
            torques = -0.0002            # 惩罚力矩（节能）
            dof_vel = -0.0               # 惩罚关节速度
            dof_acc = -2.5e-7            # 惩罚关节加速度（平滑）
            base_height = -0.0           # 惩罚偏离目标高度
            collision = -1.0             # 惩罚非足部接触
            feet_stumble = -0.0          # 惩罚足部绊倒
            action_rate = -0.01          # 惩罚动作变化率（平滑）
            stand_still = -0.0           # 零指令时惩罚运动
            dof_pos_limits = -10.0       # 惩罚关节超限
            dof_vel_limits = -0.0
            torque_limits = -0.0
            termination = -0.0           # 终止惩罚
            feet_contact_forces = -0.0   # 惩罚接触力
    
    # ==================== 归一化 ====================
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0                # 线速度缩放
            ang_vel = 0.25               # 角速度缩放
            dof_pos = 1.0                # 关节位置缩放
            dof_vel = 0.05               # 关节速度缩放
            height_measurements = 5.0    # 高度测量缩放
        clip_observations = 100.         # 观测裁剪值
        clip_actions = 100.              # 动作裁剪值
    
    # ==================== 噪声 ====================
    class noise(LeggedRobotCfg.noise):
        add_noise = True                 # 添加观测噪声
        noise_level = 1.0                # 噪声缩放系数
        class noise_scales:
            dof_pos = 0.01               # 关节位置噪声 [rad]
            dof_vel = 1.5                # 关节速度噪声 [rad/s]
            lin_vel = 0.1                # 线速度噪声 [m/s]
            ang_vel = 0.2                # 角速度噪声 [rad/s]
            gravity = 0.05               # 重力方向噪声
            height_measurements = 0.1    # 高度测量噪声 [m]
    
    # ==================== 视角相机 ====================
    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0                      # 参考环境索引
        pos = [10, 0, 6]                 # 相机位置 [m]
        lookat = [11., 5, 3.]            # 相机朝向点 [m]
    
    # ==================== 仿真参数 ====================
    class sim(LeggedRobotCfg.sim):
        dt = 0.005                       # 仿真步长 [s] (200Hz)
        substeps = 1
        gravity = [0., 0., -9.81]        # 重力加速度 [m/s^2]
        up_axis = 1                      # 上轴方向: 0=y, 1=z
        
        class physx:
            num_threads = 10             # 物理线程数
            solver_type = 1              # 0=PGS, 1=TGS
            num_position_iterations = 4  # 位置迭代次数
            num_velocity_iterations = 0  # 速度迭代次数
            contact_offset = 0.01        # 接触偏移 [m]
            rest_offset = 0.0            # 静止偏移 [m]
            bounce_threshold_velocity = 0.5  # 反弹阈值 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2       # 0=从不, 1=最后子步, 2=所有子步


class A1CustomCfgPPO(LeggedRobotCfgPPO):
    """PPO训练配置类"""
    
    seed = 1                         # 随机种子
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        init_noise_std = 1.0         # 初始动作噪声标准差
        actor_hidden_dims = [512, 256, 128]   # Actor网络隐藏层
        critic_hidden_dims = [512, 256, 128]  # Critic网络隐藏层
        activation = 'elu'           # 激活函数: 'elu', 'relu', 'selu', etc.
    
    class algorithm:
        # 训练参数
        value_loss_coef = 1.0        # 价值损失系数
        use_clipped_value_loss = True
        clip_param = 0.2             # PPO裁剪参数
        entropy_coef = 0.01          # 熵系数（鼓励探索）
        num_learning_epochs = 5      # 每次数据收集后的学习轮数
        num_mini_batches = 4         # mini-batch数量
        learning_rate = 1.e-3        # 学习率
        schedule = 'adaptive'        # 学习率调度: 'adaptive', 'fixed'
        gamma = 0.99                 # 折扣因子
        lam = 0.95                   # GAE参数
        desired_kl = 0.01            # 目标KL散度（自适应学习率用）
        max_grad_norm = 1.           # 梯度裁剪
    
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24       # 每次迭代每个环境的步数
        max_iterations = 1500        # 最大训练迭代数
        
        # 日志记录
        save_interval = 50           # 保存检查点间隔
        experiment_name = 'a1_custom' # 实验名称
        run_name = ''                # 运行名称（自动生成时间戳）
        
        # 恢复训练
        resume = False
        load_run = -1                # -1表示最新运行
        checkpoint = -1              # -1表示最新检查点
        resume_path = None
