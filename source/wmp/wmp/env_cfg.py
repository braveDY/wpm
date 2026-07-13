from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .a1_cfg import WMP_A1_CFG
from .events import randomize_base_com, randomize_material_per_env
from .terrain import WMP_ROUGH_TERRAINS_CFG


def forward_grid_pattern(cfg: ForwardGridPatternCfg, device: str):
    x = torch.arange(cfg.x_range[0], cfg.x_range[1] + 1.0e-9, cfg.resolution, device=device)
    y = torch.arange(cfg.y_range[0], cfg.y_range[1] + 1.0e-9, cfg.resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    starts = torch.zeros((grid_x.numel(), 3), device=device)
    starts[:, 0] = grid_x.flatten()
    starts[:, 1] = grid_y.flatten()
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.0
    return starts, directions


@configclass
class ForwardGridPatternCfg(patterns.PatternBaseCfg):
    func = forward_grid_pattern
    resolution: float = 0.1
    x_range: tuple[float, float] = (0.0, 2.0)
    y_range: tuple[float, float] = (-1.2, 1.2)


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=randomize_material_per_env,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": (0.5, 2.0),
            "restitution_range": (0.0, 0.0),
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (0.0, 3.0),
            "operation": "add",
        },
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_(hip|thigh|calf|foot).*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=randomize_base_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )


@configclass
class WmpEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 12
    observation_space = 45
    state_space = 285

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.5,
            gpu_max_rigid_contact_count=2**23,
        ),
    )
    viewer = ViewerCfg(eye=(10.0, 10.0, 5.0), lookat=(0.0, 0.0, 0.5))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    events: EventCfg | None = EventCfg()

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=WMP_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = WMP_A1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        update_period=0.02,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0), ordering="yx"),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    forward_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        update_period=0.02,
        ray_alignment="yaw",
        pattern_cfg=ForwardGridPatternCfg(),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    depth_camera: RayCasterCameraCfg | None = None

    action_scale = 0.25
    action_clip = 6.0
    nominal_stiffness = 40.0
    nominal_damping = 1.0
    nominal_torque_limit = 33.5
    randomize_gains = True
    gain_multiplier_range = (0.8, 1.2)
    randomize_motor_strength = True
    motor_strength_range = (0.8, 1.2)
    randomize_action_latency = True
    max_action_latency_steps = 1

    command_resampling_time_s = 10.0
    command_x_range = (0.0, 0.8)
    command_y_range = (0.0, 0.0)
    command_yaw_range = (-1.0, 1.0)
    heading_range = (0.0, 0.0)
    push_robots = True
    push_interval_s = 15.0
    max_push_velocity_xy = 1.0

    clip_observations = 100.0
    base_height_reference = 0.3
    lin_vel_obs_scale = 1.0
    ang_vel_obs_scale = 0.25
    joint_pos_obs_scale = 1.0
    joint_vel_obs_scale = 0.05
    height_obs_scale = 5.0
    contact_force_obs_scale = 0.005
    com_obs_scale = 20.0
    gain_obs_scale = 5.0
    history_length = 5

    tracking_sigma = 0.15
    lin_vel_clip = 0.1
    reward_scales = {
        "tracking_lin_vel": 1.5,
        "tracking_ang_vel": 0.5,
        "torques": -0.0001,
        "dof_acc": -2.5e-7,
        "feet_air_time": 0.5,
        "collision": -1.0,
        "feet_stumble": -0.1,
        "action_rate": -0.03,
        "feet_edge": -1.0,
        "dof_error": -0.04,
        "lin_vel_z": -1.0,
        "cheat": -1.0,
        "stuck": -1.0,
    }
    reward_curriculum_start = 4000
    reward_curriculum_end = 10000
    reward_curriculum_initial = 0.1
    reward_curriculum_final = 1.0
    rollout_steps_per_iteration = 24
    terrain_curriculum = True

    terrain_gap_range = (0.35, 0.60)
    terrain_climb_range = (0.60, 0.85)
    terrain_tilt_range = (0.85, 0.90)
    terrain_crawl_range = (0.90, 0.95)
    terrain_rough_flat_range = (0.95, 1.00)


@configclass
class WmpEnvCfg_PLAY(WmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.events = None
        self.push_robots = False
        self.randomize_gains = False
        self.randomize_motor_strength = False
        self.randomize_action_latency = False
        self.terrain_curriculum = False
        self.terrain.max_init_terrain_level = None
        self.terrain.terrain_generator.num_rows = 5
        self.terrain.terrain_generator.num_cols = 20
        self.terrain.terrain_generator.curriculum = True


@configclass
class WmpVisualEnvCfg(WmpEnvCfg):
    depth_camera = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/trunk",
        mesh_prim_paths=["/World/ground"],
        update_period=0.1,
        max_distance=2.0,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.27, 0.0, 0.03), convention="world"),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=26.6068,
            width=64,
            height=64,
        ),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        debug_vis=False,
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024


__all__ = ["WmpEnvCfg", "WmpEnvCfg_PLAY", "WmpVisualEnvCfg"]
