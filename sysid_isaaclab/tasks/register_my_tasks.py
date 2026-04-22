import gymnasium as gym

gym.register(
    id="IsaacLab-Pace-Go2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.go2_pace_env_cfg:Go2PaceEnvCfg",
    },
)

gym.register(
    id="IsaacLab-Pace-Z1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.z1_pace_env_cfg:Z1PaceEnvCfg",
    },
)

gym.register(
    id="IsaacLab-Pace-Aliengo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.aliengo_pace_env_cfg:AliengoPaceEnvCfg",
    },
)