"""
Unitree A1 四足机器人配置 V2 - 优化版
基于第一次训练的经验改进

主要改进：
1. 增加训练轮数 3000 -> 5000
2. 优化学习率调度
3. 增强奖励函数（步态奖励）
4. 增加地形难度
5. 增加并行环境数（显存允许）
6. 优化课程学习
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class A1CustomCfgV2(LeggedRobotCfg):
    """A1机器人环境配置类 V2 - 优化版"""
    
    # ==================== 环境参数 ====================
    class env(LeggedRobotCfg.env):
        num_envs = 2048              # 增加并行环境数（RTX 2080 Ti 有22GB显存，可以支持）
        num_observations = 48
        num_privileged_obs = None
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 24        # 增加回合长度到24秒（原来20秒）
    
    # ==================== 地形参数 ====================
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                             0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 
                             0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5   # 增加初始地形等级（原来3）
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        
        # 调整地形比例：增加复杂地形
        terrain_proportions = [0.10, 0.15, 0.30, 0.25, 0.20]  # 增加离散地形比例
        
        slope_treshold = 0.75
    
    # ==================== 指令参数 ====================
    class commands(LeggedRobotCfg.commands):
        curriculum = True            # 启用指令课程学习
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = True
        
        class ranges:
            lin_vel_x = [-1.5, 1.5]      # 增加速度范围（原来[-1, 1]）
            lin_vel_y = [-0.8, 0.8]      # 增加侧移速度
            ang_vel_yaw = [-1.5, 1.5]    # 增加旋转速度
            heading = [-3.14, 3.14]
    
    # ==================== 初始状态 ====================
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5
        }
    
    # ==================== 控制参数 ====================
    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 25.}   # 增加刚度（原来20，更精准控制）
        damping = {'joint': 0.6}     # 增加阻尼
        action_scale = 0.25
        decimation = 4
    
    # ==================== 资产参数 ====================
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
        fix_base_link = False
        
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
        friction_range = [0.4, 1.5]  # 扩大摩擦范围
        randomize_base_mass = True   # 启用质量随机化
        added_mass_range = [-1.5, 3.]  # 更大的质量变化
        push_robots = True
        push_interval_s = 10         # 更频繁的推力干扰
        max_push_vel_xy = 1.5        # 更大的推力
    
    # ==================== 奖励函数（关键优化）====================
    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.30    # 提高目标高度（原来0.25）
        max_contact_force = 100.
        
        class scales:
            # ========== 任务奖励（增加权重）==========
            tracking_lin_vel = 1.5       # 增加：1.0 -> 1.5
            tracking_ang_vel = 0.8       # 增加：0.5 -> 0.8
            feet_air_time = 2.0          # 大幅增加步态奖励：1.0 -> 2.0
            
            # ========== 正则化奖励（微调）==========
            lin_vel_z = -2.0             # 保持：惩罚垂直跳动
            ang_vel_xy = -0.08           # 略微增加惩罚
            orientation = -0.5           # 开启姿态奖励（原来0）
            torques = -0.0001            # 略微降低惩罚，鼓励动态
            dof_vel = -0.0
            dof_acc = -3.0e-7            # 略微增加平滑度要求
            base_height = -1.0           # 开启高度保持
            collision = -1.5             # 增加碰撞惩罚
            stumble = -0.5             # 开启绊倒惩罚
            action_rate = -0.005         # 略微降低平滑度要求
            stand_still = -0.5           # 开启静止惩罚
            dof_pos_limits = -10.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            termination = -0.0
            # feet_contact_forces = -0.0
    
    # ==================== 归一化 ====================
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
    
    # ==================== 噪声（降低噪声）====================
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.8                # 降低噪声水平（原来1.0）
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
    
    # ==================== 视角相机 ====================
    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]
    
    # ==================== 仿真参数 ====================
    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1
        
        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 1    # 增加速度迭代（原来0）
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class A1CustomCfgPPOV2(LeggedRobotCfgPPO):
    """PPO训练配置类 V2 - 优化版"""
    
    seed = 42                        # 更换随机种子
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
    
    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005             # 降低熵系数（后期训练减少探索）
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 5.e-4            # 降低学习率（更稳定）
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.008               # 降低目标KL（更保守的更新）
        max_grad_norm = 1.
    
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 5000            # 大幅增加：1500 -> 5000
        
        save_interval = 100              # 增加保存间隔：50 -> 100
        experiment_name = 'a1_custom_v2'
        run_name = ''
        
        # 从之前的模型继续训练
        resume = False                   # 可以设为 True 继续训练
        load_run = -1
        checkpoint = -1
        resume_path = None
