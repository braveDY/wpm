from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


def _box(size: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    transform = trimesh.transformations.translation_matrix(center)
    return trimesh.creation.box(size, transform)


def _ground(size: tuple[float, float], top: float = 0.0) -> trimesh.Trimesh:
    return _box((size[0], size[1], 1.0), (0.5 * size[0], 0.5 * size[1], top - 0.5))


def wmp_gap_terrain(difficulty: float, cfg: WmpGapTerrainCfg):
    gap = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    corridor = np.random.uniform(*cfg.corridor_width_range)
    length, width = cfg.size
    pit_depth = cfg.pit_depth
    center_y = 0.5 * width
    segments = ((gap, 3.0 - gap), (3.0, 6.0), (6.0 + gap, length))
    meshes = [_ground(cfg.size, top=-pit_depth)]
    for start, end in segments:
        if end <= start:
            continue
        meshes.append(_box((end - start, corridor, pit_depth), (0.5 * (start + end), center_y, -0.5 * pit_depth)))
    return meshes, np.array([0.5 * length, center_y, 0.0])


def wmp_climb_terrain(difficulty: float, cfg: WmpClimbTerrainCfg):
    height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )
    length, width = cfg.size
    first_length = np.random.uniform(*cfg.obstacle_length_range)
    second_length = np.random.uniform(*cfg.obstacle_length_range)
    meshes = [
        _ground(cfg.size),
        _box((first_length, width, height), (1.0 + 0.5 * first_length, 0.5 * width, 0.5 * height)),
        _box((second_length, width, height), (6.0 + 0.5 * second_length, 0.5 * width, 0.5 * height)),
    ]
    return meshes, np.array([0.5 * length, 0.5 * width, 0.0])


def wmp_tilt_terrain(difficulty: float, cfg: WmpTiltTerrainCfg):
    corridor = cfg.corridor_width_range[1] - difficulty * (
        cfg.corridor_width_range[1] - cfg.corridor_width_range[0]
    )
    block_length = np.random.uniform(*cfg.block_length_range)
    length, width = cfg.size
    side_width = 0.5 * (width - corridor)
    left_y = 0.5 * side_width
    right_y = width - left_y
    front_x = 6.0 + 0.5 * block_length
    back_x = 2.0 - 0.5 * block_length
    meshes = [_ground(cfg.size)]
    for x in (back_x, front_x):
        meshes.append(_box((block_length, side_width, cfg.wall_height), (x, left_y, 0.5 * cfg.wall_height)))
        meshes.append(_box((block_length, side_width, cfg.wall_height), (x, right_y, 0.5 * cfg.wall_height)))
    return meshes, np.array([0.5 * length, 0.5 * width, 0.0])


def wmp_crawl_terrain(difficulty: float, cfg: WmpCrawlTerrainCfg):
    clearance = cfg.clearance_range[1] - difficulty * (cfg.clearance_range[1] - cfg.clearance_range[0])
    bar_length = np.random.uniform(*cfg.bar_length_range)
    length, width = cfg.size
    meshes = [
        _ground(cfg.size),
        _box(
            (bar_length, width, cfg.bar_height),
            (6.0 + 0.5 * bar_length, 0.5 * width, clearance + 0.5 * cfg.bar_height),
        ),
    ]
    return meshes, np.array([0.5 * length, 0.5 * width, 0.0])


@configclass
class WmpGapTerrainCfg(SubTerrainBaseCfg):
    function = wmp_gap_terrain
    gap_width_range: tuple[float, float] = (0.0, 1.0)
    corridor_width_range: tuple[float, float] = (1.0, 2.0)
    pit_depth: float = 5.0


@configclass
class WmpClimbTerrainCfg(SubTerrainBaseCfg):
    function = wmp_climb_terrain
    obstacle_height_range: tuple[float, float] = (0.0, 0.6)
    obstacle_length_range: tuple[float, float] = (1.0, 1.2)


@configclass
class WmpTiltTerrainCfg(SubTerrainBaseCfg):
    function = wmp_tilt_terrain
    corridor_width_range: tuple[float, float] = (0.28, 0.32)
    block_length_range: tuple[float, float] = (0.4, 0.8)
    wall_height: float = 1.0


@configclass
class WmpCrawlTerrainCfg(SubTerrainBaseCfg):
    function = wmp_crawl_terrain
    clearance_range: tuple[float, float] = (0.20, 0.35)
    bar_length_range: tuple[float, float] = (0.2, 0.4)
    bar_height: float = 1.0


WMP_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "rough_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=3.0, border_width=0.25
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
        ),
        "gap": WmpGapTerrainCfg(proportion=0.25),
        "climb": WmpClimbTerrainCfg(proportion=0.25),
        "tilt": WmpTiltTerrainCfg(proportion=0.05),
        "crawl": WmpCrawlTerrainCfg(proportion=0.05),
        "rough_flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.05,
            noise_range=(-0.05, 0.05),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
    },
)

__all__ = ["WMP_ROUGH_TERRAINS_CFG"]
