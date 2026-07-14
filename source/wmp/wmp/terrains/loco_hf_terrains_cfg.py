"""自定义高度场地形配置类。"""

from dataclasses import MISSING

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import loco_hf_terrains


@configclass
class HfStonesBridgeTerrainCfg(HfTerrainBaseCfg):
    """石桥类高度场地形配置。"""

    function = loco_hf_terrains.stones_bridge_terrain

    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_length_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    stone_lateral_distance_range: tuple[float, float] = MISSING
    holes_depth: float = -10.0
    platform_width: float = 1.0


@configclass
class HfConcentricGapTerrainCfg(HfTerrainBaseCfg):
    """同心环形缺口地形配置。"""

    function = loco_hf_terrains.concentric_gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    ground_width_range: tuple[float, float] = MISSING
    ground_height_max: float = MISSING
    gap_depth: float = -2.0
    platform_width: float = 1.0


@configclass
class HfDoubleColumnStakesTerrainCfg(HfTerrainBaseCfg):
    """双列梅花桩地形配置。"""

    function = loco_hf_terrains.double_column_stakes_terrain

    stake_height_max: float = MISSING
    stake_side_range: tuple[float, float] = MISSING
    stake_gap_range: tuple[float, float] = MISSING
    column_gap_range: tuple[float, float] = MISSING
    column_jitter: float = 0.0
    holes_depth: float = -2.0
    platform_width: float = 1.0
    seed: int | None = None


@configclass
class HfAlternateColumnStakesTerrainCfg(HfTerrainBaseCfg):
    """交错双列梅花桩地形配置。"""

    function = loco_hf_terrains.alternate_column_stakes_terrain

    stake_height_max: float = MISSING
    stake_side_range: tuple[float, float] = MISSING
    stake_gap_range: tuple[float, float] = MISSING
    column_gap_range: tuple[float, float] = MISSING
    column_jitter: float = 0.0
    holes_depth: float = -2.0
    platform_width: float = 1.0
    seed: int | None = None

