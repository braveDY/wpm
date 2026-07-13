"""A1 任务注册。"""

import gymnasium as gym


gym.register(
    id="Wmp-Velocity-Rough-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:UnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Wmp-Velocity-Rough-A1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UnitreeA1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:UnitreeA1RoughPPORunnerCfg",
    },
)
