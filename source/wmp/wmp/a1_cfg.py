from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

WMP_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

WMP_A1_CFG = UNITREE_A1_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_(hip|thigh|calf)_joint"],
            effort_limit=40.2,
            velocity_limit=21.0,
            effort_limit_sim=40.2,
            velocity_limit_sim=21.0,
            stiffness=0.0,
            damping=0.0,
        )
    },
)

__all__ = ["WMP_A1_CFG", "WMP_JOINT_NAMES"]
