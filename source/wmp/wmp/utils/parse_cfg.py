"""任务配置解析工具。"""

from __future__ import annotations

import collections
import importlib
import os
import re

import gymnasium as gym
import yaml

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict | object:
    """从 gym 注册表加载任务配置。"""
    cfg_entry_point = gym.spec(task_name.split(":")[-1]).kwargs.get(entry_point_key)
    if cfg_entry_point is None:
        agents = collections.defaultdict(list)
        for key in gym.spec(task_name.split(":")[-1]).kwargs:
            if key.endswith("_cfg_entry_point") and key != "env_cfg_entry_point":
                spec = key.replace("_cfg_entry_point", "").replace("rl_games", "rl-games").replace("rsl_rl", "rsl-rl")
                spec_items = spec.split("_")
                agent = spec_items[0].replace("-", "_")
                algorithms = [item.upper() for item in (spec_items[1:] if len(spec_items) > 1 else ["PPO"])]
                agents[agent].extend(algorithms)

        message = "\nExisting RL library (and algorithms) config entry points: "
        for agent, algorithms in agents.items():
            message += f"\n  |-- {agent}: {', '.join(algorithms)}"
        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'."
            f"\nPlease check that the gym registry has the entry point: '{entry_point_key}'."
            f"{message if agents else ''}"
        )

    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            config_file = cfg_entry_point
        else:
            module_name, file_name = cfg_entry_point.split(":")
            module_path = os.path.dirname(importlib.import_module(module_name).__file__)
            config_file = os.path.join(module_path, file_name)
        print(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as file:
            cfg = yaml.full_load(file)
    else:
        if callable(cfg_entry_point):
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            module_name, attr_name = cfg_entry_point.split(":")
            module = importlib.import_module(module_name)
            cfg_cls = getattr(module, attr_name)
        else:
            cfg_cls = cfg_entry_point

        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
        cfg = cfg_cls() if callable(cfg_cls) else cfg_cls

    return cfg


def parse_env_cfg(
    task_name: str, device: str = "cuda:0", num_envs: int | None = None, use_fabric: bool | None = None
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    """解析环境配置并应用命令行覆盖。"""
    cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for the task: '{task_name}' is not a class. Please provide a class.")

    cfg.sim.device = device
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    if num_envs is not None:
        cfg.scene.num_envs = num_envs

    return cfg


def get_checkpoint_path(
    log_path: str,
    run_dir: str = ".*",
    checkpoint: str = ".*",
    other_dirs: list[str] | None = None,
    sort_alpha: bool = True,
) -> str:
    """从日志目录解析 checkpoint 路径。"""
    try:
        runs = [os.path.join(log_path, run.name) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)]
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)
        run_path = os.path.join(runs[-1], *other_dirs) if other_dirs is not None else runs[-1]
    except IndexError as exc:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.") from exc

    model_checkpoints = [file_name for file_name in os.listdir(run_path) if re.match(checkpoint, file_name)]
    if not model_checkpoints:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")

    model_checkpoints.sort(key=lambda name: f"{name:0>15}")
    return os.path.join(run_path, model_checkpoints[-1])
