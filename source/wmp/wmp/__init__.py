import gymnasium as gym

_COMMON_KWARGS = {"rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"}

gym.register(
    id="Isaac-Wmp-Direct-v0",
    entry_point=f"{__name__}.env:WmpEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:WmpEnvCfg", **_COMMON_KWARGS},
)
gym.register(
    id="Isaac-Wmp-Direct-Play-v0",
    entry_point=f"{__name__}.env:WmpEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:WmpEnvCfg_PLAY", **_COMMON_KWARGS},
)
gym.register(
    id="Isaac-Wmp-Visual-Direct-v0",
    entry_point=f"{__name__}.env:WmpEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:WmpVisualEnvCfg", **_COMMON_KWARGS},
)
gym.register(
    id="Template-Wmp-Direct-v0",
    entry_point=f"{__name__}.env:WmpEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:WmpEnvCfg", **_COMMON_KWARGS},
)
