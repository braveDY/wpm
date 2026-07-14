"""A1 粗糙地形速度任务配置。"""

from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from wmp.common.env_cfg import LocomotionVelocityRoughEnvCfg
from wmp.terrains.finetune_terrain_cfg import FINETUNE_ROUGH_TERRAINS_CFG
from wmp.terrains.terrain_cfg import ROUGH_TERRAINS_CFG
FINETUNE = False

@configclass
class UnitreeA1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """A1 粗糙地形训练配置。"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.height_scanner.pattern_cfg.resolution = 0.05

        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"

        if FINETUNE:
            self.scene.terrain.terrain_generator = FINETUNE_ROUGH_TERRAINS_CFG
            self.events.reset_base.params = {
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            }
        else:
            self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
            self.events.reset_base.params = {
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            }
        self.scene.terrain.terrain_generator.curriculum = True

        
@configclass
class UnitreeA1RoughEnvCfg_PLAY(UnitreeA1RoughEnvCfg):
    """A1 粗糙地形播放配置。"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.observations.height_map.enable_corruption = False
        self.events.base_external_force_torque = None
