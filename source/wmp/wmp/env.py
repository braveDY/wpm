from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
from isaaclab.utils.math import quat_apply, quat_inv, wrap_to_pi, yaw_quat

from .a1_cfg import WMP_JOINT_NAMES
from .env_cfg import WmpEnvCfg, WmpEnvCfg_PLAY, WmpVisualEnvCfg


class WmpEnv(DirectRLEnv):
    cfg: WmpEnvCfg | WmpEnvCfg_PLAY | WmpVisualEnvCfg

    def __init__(
        self,
        cfg: WmpEnvCfg | WmpEnvCfg_PLAY | WmpVisualEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._joint_ids, joint_names = self._robot.find_joints(WMP_JOINT_NAMES, preserve_order=True)
        if tuple(joint_names) != WMP_JOINT_NAMES:
            raise RuntimeError(f"WMP 关节顺序不匹配: {joint_names}")
        self._feet_ids, foot_names = self._contact_sensor.find_bodies(
            ["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
        )
        if len(self._feet_ids) != 4:
            raise RuntimeError(f"WMP 足端数量不匹配: {foot_names}")
        self._penalized_body_ids, penalized_names = self._contact_sensor.find_bodies(
            [
                "FL_thigh",
                "FL_calf",
                "FR_thigh",
                "FR_calf",
                "RL_thigh",
                "RL_calf",
                "RR_thigh",
                "RR_calf",
            ],
            preserve_order=True,
        )
        if len(self._penalized_body_ids) != 8:
            raise RuntimeError(f"WMP 碰撞身体数量不匹配: {penalized_names}")
        self._base_id, _ = self._contact_sensor.find_bodies("trunk")
        self._robot_base_id, _ = self._robot.find_bodies("trunk")

        action_shape = (self.num_envs, len(self._joint_ids))
        self._actions = torch.zeros(action_shape, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._torques = torch.zeros_like(self._actions)
        self._commands = torch.zeros((self.num_envs, 4), device=self.device)
        self._history = torch.zeros(
            (self.num_envs, self.cfg.history_length, 42), dtype=torch.float32, device=self.device
        )
        self._previous_joint_vel = self._robot.data.joint_vel[:, self._joint_ids].clone()
        self._forward_vector = torch.zeros((self.num_envs, 3), device=self.device)
        self._forward_vector[:, 0] = 1.0
        self._physics_substep = 0
        self._latency_steps = 0
        self._last_history_counter = -1
        self._training_iteration_offset = 0.0
        self._feet_air_time = torch.zeros((self.num_envs, len(self._feet_ids)), device=self.device)
        self._last_foot_contacts = torch.zeros(
            (self.num_envs, len(self._feet_ids)), dtype=torch.bool, device=self.device
        )
        self._contact_filter = torch.zeros_like(self._last_foot_contacts)
        self._wmp_observations: dict[str, torch.Tensor] = {}

        self._p_gains = torch.full(action_shape, self.cfg.nominal_stiffness, device=self.device)
        self._d_gains = torch.full(action_shape, self.cfg.nominal_damping, device=self.device)
        self._resample_gains(self._all_env_ids)

        if not hasattr(self, "_wmp_friction"):
            self._wmp_friction = torch.ones((self.num_envs, 1), device=self.device)
        if not hasattr(self, "_wmp_restitution"):
            self._wmp_restitution = torch.zeros((self.num_envs, 1), device=self.device)
        if not hasattr(self, "_wmp_com_offsets"):
            self._wmp_com_offsets = torch.zeros((self.num_envs, 3), device=self.device)
        masses = self._robot.root_physx_view.get_masses().to(self.device)
        default_base_mass = self._robot.data.default_mass[:, self._robot_base_id[0]].to(self.device)
        self._added_base_mass = (
            masses[:, self._robot_base_id[0]] - default_base_mass
        ).unsqueeze(-1)

        self._command_interval = max(1, round(self.cfg.command_resampling_time_s / self.step_dt))
        self._push_interval = max(1, round(self.cfg.push_interval_s / self.step_dt))
        self._episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for name in self.cfg.reward_scales
        }
        self._termination_reasons = {
            "base_contact": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "velocity": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "fall": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self._forward_height_scanner = RayCaster(self.cfg.forward_height_scanner)
        self.scene.sensors["forward_height_scanner"] = self._forward_height_scanner
        self._depth_camera = None
        if self.cfg.depth_camera is not None:
            self._depth_camera = RayCasterCamera(self.cfg.depth_camera)
            self.scene.sensors["depth_camera"] = self._depth_camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._ensure_finite("actions", actions)
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip))
        self._physics_substep = 0
        if self.cfg.randomize_action_latency:
            self._latency_steps = int(
                torch.randint(0, self.cfg.max_action_latency_steps + 1, (1,), device=self.device).item()
            )
        else:
            self._latency_steps = 0

    def _apply_action(self) -> None:
        control_actions = self._previous_actions if self._physics_substep < self._latency_steps else self._actions
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids]
        joint_target = default_joint_pos + self.cfg.action_scale * control_actions
        torques = self._p_gains * (joint_target - joint_pos) - self._d_gains * joint_vel
        torques = torch.clamp(torques, -self.cfg.nominal_torque_limit, self.cfg.nominal_torque_limit)
        if self.cfg.randomize_motor_strength:
            motor_strength = torch.empty_like(torques).uniform_(*self.cfg.motor_strength_range)
            torques = torques * motor_strength
        self._ensure_finite("torques", torques)
        self._torques.copy_(torques)
        self._robot.set_joint_effort_target(self._torques, joint_ids=self._joint_ids)
        self._physics_substep += 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_commands_and_pushes()
        contact_forces = self._contact_sensor.data.net_forces_w
        base_contact = torch.any(torch.norm(contact_forces[:, self._base_id], dim=-1) > 1.0, dim=1)
        velocity_error = self._robot.data.root_lin_vel_b[:, 0] - self._commands[:, 0]
        velocity_violation = (
            ((velocity_error > 1.5) & (self._commands[:, 0] < 0.0))
            | ((velocity_error < -1.5) & (self._commands[:, 0] > 0.0))
        ) & (self._terrain.terrain_levels > 3)
        fall = (self._robot.data.root_lin_vel_w[:, 2] < -3.0) | (
            self._robot.data.projected_gravity_b[:, 2] > 0.0
        )
        self._termination_reasons["base_contact"].copy_(base_contact)
        self._termination_reasons["velocity"].copy_(velocity_violation)
        self._termination_reasons["fall"].copy_(fall)
        terminated = base_contact | velocity_violation | fall
        time_out = self.episode_length_buf > self.max_episode_length
        self.extras["terminal_amp"] = self._compute_amp_observation().clone()
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids]

        linear_velocity = root_lin_vel_b[:, :2]
        upper_bound = torch.where(
            self._commands[:, :2] < 0.0,
            torch.full_like(self._commands[:, :2], 1.0e5),
            self._commands[:, :2] + self.cfg.lin_vel_clip,
        )
        lower_bound = torch.where(
            self._commands[:, :2] > 0.0,
            torch.full_like(self._commands[:, :2], -1.0e5),
            self._commands[:, :2] - self.cfg.lin_vel_clip,
        )
        clipped_velocity = torch.clamp(linear_velocity, min=lower_bound, max=upper_bound)
        linear_error = torch.sum(torch.square(self._commands[:, :2] - clipped_velocity), dim=1)
        tracking_lin_vel = torch.exp(-linear_error / self.cfg.tracking_sigma)
        angular_error = torch.square(self._commands[:, 2] - root_ang_vel_b[:, 2])
        tracking_ang_vel = torch.exp(-angular_error / self.cfg.tracking_sigma)

        joint_acceleration = (joint_vel - self._previous_joint_vel) / self.step_dt
        contact_forces = self._contact_sensor.data.net_forces_w
        foot_forces = contact_forces[:, self._feet_ids]
        contact = foot_forces[..., 2] > 1.0
        self._contact_filter = torch.logical_or(contact, self._last_foot_contacts)
        self._last_foot_contacts.copy_(contact)
        first_contact = (self._feet_air_time > 0.0) & self._contact_filter
        self._feet_air_time += self.step_dt
        feet_air_time = torch.sum((self._feet_air_time - 0.5) * first_contact.float(), dim=1)
        feet_air_time *= (torch.norm(self._commands[:, :2], dim=1) > 0.1).float()
        self._feet_air_time *= (~self._contact_filter).float()

        penalized_contact = torch.norm(contact_forces[:, self._penalized_body_ids], dim=-1) > 0.1
        collision = torch.sum(penalized_contact.float(), dim=1)
        feet_stumble = torch.any(
            torch.norm(foot_forces[:, :, :2], dim=-1) > 4.0 * torch.abs(foot_forces[:, :, 2]), dim=1
        ).float()
        obstacle_mask = self._terrain_mask(self.cfg.terrain_gap_range) | self._terrain_mask(
            self.cfg.terrain_climb_range
        )
        feet_stumble *= obstacle_mask & (self._terrain.terrain_levels > 3)

        heading = self._current_heading()
        rough_flat = self._terrain_mask(self.cfg.terrain_rough_flat_range)
        cheat = ((heading > 1.0) | (heading < -1.0)) & ~rough_flat
        stuck = (torch.abs(root_lin_vel_b[:, 0]) < 0.1) & (torch.abs(self._commands[:, 0]) > 0.1)
        feet_edge = self._compute_feet_edge_reward()
        curriculum = self._reward_curriculum_coefficient()

        raw_terms = {
            "tracking_lin_vel": tracking_lin_vel,
            "tracking_ang_vel": tracking_ang_vel,
            "torques": torch.sum(torch.square(self._torques), dim=1),
            "dof_acc": torch.sum(torch.square(joint_acceleration), dim=1),
            "feet_air_time": feet_air_time,
            "collision": collision,
            "feet_stumble": feet_stumble,
            "action_rate": torch.sum(torch.square(self._actions - self._previous_actions), dim=1),
            "feet_edge": feet_edge * curriculum,
            "dof_error": torch.sum(torch.square(joint_pos - default_joint_pos), dim=1),
            "lin_vel_z": torch.square(root_lin_vel_b[:, 2]),
            "cheat": cheat.float(),
            "stuck": stuck.float(),
        }
        rewards = {
            name: value * self.cfg.reward_scales[name] * self.step_dt for name, value in raw_terms.items()
        }
        for name, value in rewards.items():
            self._episode_sums[name] += value
        self._previous_joint_vel.copy_(joint_vel)
        reward = compute_rewards(rewards)
        self._ensure_finite("rewards", reward)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        contact_forces = self._contact_sensor.data.net_forces_w
        contact_flags = (torch.norm(contact_forces[:, self._penalized_body_ids], dim=-1) > 0.1).float()
        foot_forces = contact_forces[:, self._feet_ids].flatten(1) * self.cfg.contact_force_obs_scale
        damping_randomization = (self._d_gains / self.cfg.nominal_damping - 1.0) * self.cfg.gain_obs_scale
        stiffness_randomization = (self._p_gains / self.cfg.nominal_stiffness - 1.0) * self.cfg.gain_obs_scale
        root_lin_vel = self._robot.data.root_lin_vel_b * self.cfg.lin_vel_obs_scale
        root_ang_vel = self._robot.data.root_ang_vel_b * self.cfg.ang_vel_obs_scale
        projected_gravity = self._robot.data.projected_gravity_b
        command = self._commands[:, :3] * torch.tensor(
            [self.cfg.lin_vel_obs_scale, self.cfg.lin_vel_obs_scale, self.cfg.ang_vel_obs_scale],
            device=self.device,
        )
        joint_pos = (
            self._robot.data.joint_pos[:, self._joint_ids]
            - self._robot.data.default_joint_pos[:, self._joint_ids]
        ) * self.cfg.joint_pos_obs_scale
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids] * self.cfg.joint_vel_obs_scale
        height = self._compute_height_observation()

        full_observation = torch.cat(
            (
                contact_flags,
                foot_forces,
                damping_randomization,
                stiffness_randomization,
                self._wmp_com_offsets * self.cfg.com_obs_scale,
                self._added_base_mass,
                self._wmp_restitution,
                self._wmp_friction,
                root_lin_vel,
                root_ang_vel,
                projected_gravity,
                command,
                joint_pos,
                joint_vel,
                self._actions,
                height,
            ),
            dim=-1,
        )
        if full_observation.shape[-1] != 285:
            raise RuntimeError(f"WMP critic 观测维度应为 285，实际为 {full_observation.shape[-1]}")

        clipped_full_observation = torch.clamp(full_observation, -self.cfg.clip_observations, self.cfg.clip_observations)
        history_frame = torch.cat((clipped_full_observation[:, 53:59], clipped_full_observation[:, 62:98]), dim=-1)
        if self._last_history_counter != self.common_step_counter:
            self._history = torch.cat((self._history[:, 1:], history_frame.unsqueeze(1)), dim=1)
            self._last_history_counter = self.common_step_counter

        policy_observation = clipped_full_observation[:, 53:98]
        critic_observation = clipped_full_observation
        self._wmp_observations = {
            "history": self._history.flatten(1),
            "command": clipped_full_observation[:, 59:62],
            "world_model": clipped_full_observation[:, 53:86],
            "forward_height": self._compute_forward_height_observation(),
            "amp": self._compute_amp_observation(),
        }
        if self._depth_camera is not None:
            depth = self._depth_camera.data.output["distance_to_image_plane"]
            self._wmp_observations["depth"] = torch.clamp(depth, 0.0, 2.0) / 2.0 - 0.5

        self._ensure_finite("policy observations", policy_observation)
        self._ensure_finite("critic observations", critic_observation)
        for name, value in self._wmp_observations.items():
            self._ensure_finite(f"WMP observation[{name}]", value)
        return {"policy": policy_observation, "critic": critic_observation}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self._all_env_ids
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        self._update_terrain_curriculum(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._log_episode(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._torques[env_ids] = 0.0
        self._history[env_ids] = 0.0
        self._feet_air_time[env_ids] = 0.0
        self._last_foot_contacts[env_ids] = False
        self._contact_filter[env_ids] = False
        self._resample_gains(env_ids)
        self._resample_commands(env_ids)

        default_joint_pos = self._robot.data.default_joint_pos[env_ids][:, self._joint_ids]
        joint_pos = default_joint_pos * torch.empty_like(default_joint_pos).uniform_(0.5, 1.5)
        joint_vel = torch.zeros_like(joint_pos)
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        root_state[:, :2] += torch.empty((len(env_ids), 2), device=self.device).uniform_(-1.0, 1.0)
        constrained = self._terrain_mask(self.cfg.terrain_gap_range)[env_ids] | self._terrain_mask(
            self.cfg.terrain_tilt_range
        )[env_ids]
        root_state[constrained, 1] = self._terrain.env_origins[env_ids[constrained], 1]
        root_state[:, 7:13] = torch.empty((len(env_ids), 6), device=self.device).uniform_(-0.5, 0.5)

        self._previous_joint_vel[env_ids] = joint_vel
        self._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            joint_pos, joint_vel, joint_ids=self._joint_ids, env_ids=env_ids
        )

    def _resample_gains(self, env_ids: torch.Tensor):
        if self.cfg.randomize_gains:
            self._p_gains[env_ids] = self.cfg.nominal_stiffness * torch.empty(
                (len(env_ids), len(self._joint_ids)), device=self.device
            ).uniform_(*self.cfg.gain_multiplier_range)
            self._d_gains[env_ids] = self.cfg.nominal_damping * torch.empty(
                (len(env_ids), len(self._joint_ids)), device=self.device
            ).uniform_(*self.cfg.gain_multiplier_range)
        else:
            self._p_gains[env_ids] = self.cfg.nominal_stiffness
            self._d_gains[env_ids] = self.cfg.nominal_damping

    def _resample_commands(self, env_ids: torch.Tensor):
        self._commands[env_ids, 0].uniform_(*self.cfg.command_x_range)
        self._commands[env_ids, 1].uniform_(*self.cfg.command_y_range)
        self._commands[env_ids, 2] = 0.0
        self._commands[env_ids, 3].uniform_(*self.cfg.heading_range)
        rough_flat_ids = env_ids[self._terrain_mask(self.cfg.terrain_rough_flat_range)[env_ids]]
        if len(rough_flat_ids) > 0:
            self._commands[rough_flat_ids, 2].uniform_(*self.cfg.command_yaw_range)
        moving = torch.norm(self._commands[env_ids, :2], dim=1) > 0.2
        self._commands[env_ids, :2] *= moving.unsqueeze(1)

    def _update_commands_and_pushes(self):
        resample_ids = (self.episode_length_buf % self._command_interval == 0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)
        rough_flat = self._terrain_mask(self.cfg.terrain_rough_flat_range)
        heading_error = wrap_to_pi(self._commands[:, 3] - self._current_heading())
        self._commands[~rough_flat, 2] = torch.clamp(0.5 * heading_error[~rough_flat], -1.0, 1.0)
        if self.cfg.push_robots and self.common_step_counter % self._push_interval == 0:
            root_velocity = self._robot.data.root_com_vel_w.clone()
            root_velocity[:, :2] = torch.empty((self.num_envs, 2), device=self.device).uniform_(
                -self.cfg.max_push_velocity_xy, self.cfg.max_push_velocity_xy
            )
            self._robot.write_root_com_velocity_to_sim(root_velocity)

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        generator = self.cfg.terrain.terrain_generator
        if not self.cfg.terrain_curriculum or generator is None or not generator.curriculum or self.common_step_counter == 0:
            return
        distance = torch.norm(
            self._robot.data.root_link_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1
        )
        move_up = distance > 0.5 * generator.size[0]
        expected_distance = torch.norm(self._commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        move_down = (distance < expected_distance) & ~move_up
        self._terrain.update_env_origins(env_ids, move_up, move_down)

    def _terrain_mask(self, terrain_range: tuple[float, float]) -> torch.Tensor:
        num_cols = self.cfg.terrain.terrain_generator.num_cols
        fraction = (self._terrain.terrain_types.float() + 0.5) / num_cols
        return (fraction >= terrain_range[0]) & (fraction < terrain_range[1])

    def _current_heading(self) -> torch.Tensor:
        forward = quat_apply(self._robot.data.root_link_quat_w, self._forward_vector)
        return torch.atan2(forward[:, 1], forward[:, 0])

    def _compute_height_observation(self) -> torch.Tensor:
        ray_height = self._height_scanner.data.ray_hits_w[..., 2]
        height = self._robot.data.root_link_pos_w[:, 2].unsqueeze(1) - self.cfg.base_height_reference - ray_height
        return torch.clamp(height, -1.0, 1.0) * self.cfg.height_obs_scale

    def _compute_forward_height_observation(self) -> torch.Tensor:
        ray_height = self._forward_height_scanner.data.ray_hits_w[..., 2]
        height = self._robot.data.root_link_pos_w[:, 2].unsqueeze(1) - self.cfg.base_height_reference - ray_height
        return torch.clamp(height, -1.0, 1.0) * self.cfg.height_obs_scale

    def _compute_amp_observation(self) -> torch.Tensor:
        return torch.cat(
            (
                self._robot.data.joint_pos[:, self._joint_ids],
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.joint_vel[:, self._joint_ids],
            ),
            dim=-1,
        )

    def _compute_feet_edge_reward(self) -> torch.Tensor:
        ray_height = self._height_scanner.data.ray_hits_w[..., 2].reshape(self.num_envs, 17, 11)
        edge = torch.zeros_like(ray_height, dtype=torch.bool)
        x_difference = torch.abs(ray_height[:, 1:] - ray_height[:, :-1]) > 0.075
        edge[:, 1:] |= x_difference
        edge[:, :-1] |= x_difference

        foot_position = self._robot.data.body_pos_w[:, self._feet_ids] - self._robot.data.root_link_pos_w.unsqueeze(1)
        yaw_inverse = quat_inv(yaw_quat(self._robot.data.root_link_quat_w)).unsqueeze(1).expand(-1, 4, -1)
        foot_position = quat_apply(yaw_inverse, foot_position)
        x_index = torch.clamp(torch.round((foot_position[:, :, 0] + 0.8) / 0.1).long(), 0, 16)
        y_index = torch.clamp(torch.round((foot_position[:, :, 1] + 0.5) / 0.1).long(), 0, 10)
        env_index = self._all_env_ids.unsqueeze(1).expand(-1, 4)
        foot_at_edge = edge[env_index, x_index, y_index]
        obstacle_mask = self._terrain_mask(self.cfg.terrain_gap_range) | self._terrain_mask(
            self.cfg.terrain_climb_range
        )
        terrain_gate = self._terrain.terrain_levels > 3
        return torch.sum((foot_at_edge & self._contact_filter).float(), dim=1) * obstacle_mask * terrain_gate

    def _reward_curriculum_coefficient(self) -> float:
        iteration = self._training_iteration_offset + self.common_step_counter / self.cfg.rollout_steps_per_iteration
        progress = (iteration - self.cfg.reward_curriculum_start) / (
            self.cfg.reward_curriculum_end - self.cfg.reward_curriculum_start
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.cfg.reward_curriculum_initial + progress * (
            self.cfg.reward_curriculum_final - self.cfg.reward_curriculum_initial
        )

    def set_training_iteration(self, iteration: int | float) -> None:
        self._training_iteration_offset = float(iteration)

    def get_wmp_observations(self) -> dict[str, torch.Tensor]:
        return self._wmp_observations

    def _ensure_finite(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0 or torch.isfinite(tensor).all():
            return
        if tensor.ndim > 0 and tensor.shape[0] == self.num_envs:
            flat = tensor.reshape(self.num_envs, -1)
            bad_env_ids = torch.nonzero(~torch.isfinite(flat).all(dim=1), as_tuple=False).flatten()
            env_ids = bad_env_ids[:8].detach().cpu().tolist()
            raise RuntimeError(f"{name} 含非有限值: env_ids={env_ids}")
        raise RuntimeError(f"{name} 含非有限值")

    def _log_episode(self, env_ids: torch.Tensor):
        log = {}
        for name, episode_sum in self._episode_sums.items():
            log[f"/Episode/Reward/{name}"] = torch.mean(episode_sum[env_ids]) / self.max_episode_length_s
            episode_sum[env_ids] = 0.0
        log["/Episode/TerrainLevel"] = torch.mean(self._terrain.terrain_levels.float())
        log["/Episode/Reward/feet_edge_curriculum"] = self._reward_curriculum_coefficient()
        for name, reason in self._termination_reasons.items():
            log[f"/Episode/Termination/{name}"] = torch.count_nonzero(reason[env_ids]).item()
        log["/Episode/Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = log


def compute_rewards(reward_terms: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sum(torch.stack(tuple(reward_terms.values())), dim=0).clamp_min(0.0)


__all__ = ["WmpEnv", "compute_rewards"]
