"""共享的速度任务 MDP 扩展。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """按行走表现调整地形难度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """机器人接近地形边界时终止。"""
    if env.scene.cfg.terrain.terrain_type == "plane":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if env.scene.cfg.terrain.terrain_type != "generator":
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")

    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    grid_width, grid_length = terrain_gen_cfg.size
    map_width = terrain_gen_cfg.num_rows * grid_width + 2 * terrain_gen_cfg.border_width
    map_height = terrain_gen_cfg.num_cols * grid_length + 2 * terrain_gen_cfg.border_width

    asset: RigidObject = env.scene[asset_cfg.name]
    x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
    y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
    return torch.logical_or(x_out_of_bounds, y_out_of_bounds)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """奖励足端离地时间。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """双足任务的正向离地奖励。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚足端打滑。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def height_scan_image(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5
) -> torch.Tensor:
    """将高度扫描重排为单通道二维高度图。"""
    height_values = base_mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    pattern_cfg = sensor.cfg.pattern_cfg

    if not hasattr(pattern_cfg, "resolution") or not hasattr(pattern_cfg, "size"):
        raise ValueError("高度图观测要求 `height_scanner` 使用 GridPatternCfg。")

    resolution = pattern_cfg.resolution
    x_count = int(round(pattern_cfg.size[0] / resolution)) + 1
    y_count = int(round(pattern_cfg.size[1] / resolution)) + 1

    if height_values.shape[1] != x_count * y_count:
        raise ValueError(
            f"高度扫描数量与网格尺寸不一致: got {height_values.shape[1]}, expected {x_count * y_count}."
        )

    ordering = getattr(pattern_cfg, "ordering", "xy")
    if ordering == "xy":
        height_map = height_values.reshape(env.num_envs, y_count, x_count)
    elif ordering == "yx":
        height_map = height_values.reshape(env.num_envs, x_count, y_count).transpose(1, 2)
    else:
        raise ValueError(f"不支持的 GridPattern 排列方式: {ordering}")

    return height_map.unsqueeze(1)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励偏航对齐坐标系下的线速度跟踪。"""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励世界系下的偏航角速度跟踪。"""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """小速度指令下惩罚关节偏离默认位姿。"""
    command = env.command_manager.get_command(command_name)
    return base_mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)
