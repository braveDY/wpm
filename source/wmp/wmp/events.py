from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def randomize_material_per_env(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
):
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    materials = asset.root_physx_view.get_material_properties()
    friction = torch.empty((len(env_ids), 1), device="cpu").uniform_(*friction_range)
    restitution = torch.empty((len(env_ids), 1), device="cpu").uniform_(*restitution_range)
    materials[env_ids, :, 0] = friction
    materials[env_ids, :, 1] = friction
    materials[env_ids, :, 2] = restitution
    asset.root_physx_view.set_material_properties(materials, env_ids)
    env._wmp_friction = torch.zeros((env.scene.num_envs, 1), device=asset.device)
    env._wmp_restitution = torch.zeros((env.scene.num_envs, 1), device=asset.device)
    env._wmp_friction[env_ids.to(asset.device)] = friction.to(asset.device)
    env._wmp_restitution[env_ids.to(asset.device)] = restitution.to(asset.device)


def randomize_base_com(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
):
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    body_ids = torch.tensor(asset_cfg.body_ids, device="cpu", dtype=torch.long)
    ranges = torch.tensor([com_range[axis] for axis in ("x", "y", "z")], device="cpu")
    offsets = torch.empty((len(env_ids), 3), device="cpu")
    offsets.uniform_(0.0, 1.0)
    offsets = ranges[:, 0] + offsets * (ranges[:, 1] - ranges[:, 0])
    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids[:, None], body_ids, :3] += offsets[:, None, :]
    asset.root_physx_view.set_coms(coms, env_ids)
    env._wmp_com_offsets = torch.zeros((env.scene.num_envs, 3), device=asset.device)
    env._wmp_com_offsets[env_ids.to(asset.device)] = offsets.to(asset.device)


__all__ = ["randomize_base_com", "randomize_material_per_env"]
