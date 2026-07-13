# WMP A1 Isaac Lab PPO Baseline

本仓库提供 WMP A1 粗糙地形任务在 Isaac Lab 上的首轮迁移结果，当前目标是“真实 WMP/A1 环境语义 + Isaac Lab 标准 `rsl-rl-lib==5.0.1` PPO 基线”。

当前实现已经覆盖：

- canonical Python 入口：`wmp.env`、`wmp.env_cfg`、`wmp.rsl_rl_ppo_cfg`
- Gym 任务：`Isaac-Wmp-Direct-v0`、`Isaac-Wmp-Direct-Play-v0`、`Isaac-Wmp-Visual-Direct-v0`
- 兼容别名：`Template-Wmp-Direct-v0`
- 训练、恢复训练、播放、JIT 导出、ONNX 导出

当前 `source` 目录已经收口为单一 canonical Python 包结构，仓库不再提供 `wmp.tasks.*` 旧导入路径。

当前不包含：

- AMP 训练链路
- Dreamer / World Model 训练链路
- 深度策略训练
- 自定义 WMP runner

这些能力需要在标准 PPO 基线之外单独设计和验证。

## 版本锁定

- 原始 WMP 参考仓库：`/home/brave/robot_rl/WMP`
- 原始 WMP 固定 commit：`c232c115ada4517453ebded5019078ba055456de`
- Isaac Sim：`5.1.0.0`
- Isaac Lab：`2.3.2`
- Isaac Lab Python 包元数据：`0.54.4`
- `isaaclab_rl`：`0.5.2`
- `rsl-rl-lib`：`5.0.1`
- PyTorch：`2.7.0+cu128`

`scripts/rsl_rl/train.py` 和 `scripts/rsl_rl/play.py` 会在启动 Isaac Sim 之前检查 `rsl-rl-lib==5.0.1`，版本不一致会直接报错退出。

## 安装

以下命令默认在已经安装 Isaac Lab 的 Python 环境中执行：

```bash
export PYTHONPATH="$PWD/source/wmp${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -e source/wmp
python -m pip install "rsl-rl-lib==5.0.1"
```

如果你使用的是 `isaaclab.sh -p` 或 Conda 环境，只需要把上面的 `python` 替换成对应解释器即可。

## 任务与接口

| 类型 | 标识 |
| --- | --- |
| 标准训练任务 | `Isaac-Wmp-Direct-v0` |
| 播放任务 | `Isaac-Wmp-Direct-Play-v0` |
| Visual 采集任务 | `Isaac-Wmp-Visual-Direct-v0` |
| 兼容别名 | `Template-Wmp-Direct-v0` |

Python 导入入口只保留：

- `wmp.env`
- `wmp.env_cfg`
- `wmp.rsl_rl_ppo_cfg`

标准 PPO 基线的环境契约如下：

- 动作维度：12
- actor 观测维度：45
- critic 观测维度：285
- 控制方式：显式 PD，`action_scale=0.25`
- 动作裁剪：`[-6.0, 6.0]`
- 标称刚度 / 阻尼：`40.0 / 1.0`
- 标称力矩限幅：`33.5`

训练脚本只消费 `policy` 和 `critic` 两组观测。WMP 辅助观测通过环境公共接口 `get_wmp_observations()` 暴露，当前可包含：

- `history`
- `command`
- `world_model`
- `forward_height`
- `amp`
- `depth`（仅 Visual 任务）

标准 PPO actor 不消费 depth，`Isaac-Wmp-Visual-Direct-v0` 当前主要用于数据采集和后续自定义链路接入。

## 常用命令

列出任务：

```bash
python scripts/list_envs.py --keyword Wmp
```

零动作 smoke：

```bash
python scripts/zero_agent.py --task Isaac-Wmp-Direct-v0 --num_envs 1 --num_steps 32 --headless
```

随机动作 smoke：

```bash
python scripts/random_agent.py --task Isaac-Wmp-Direct-v0 --num_envs 8 --num_steps 32 --headless
```

Visual smoke：

```bash
python scripts/zero_agent.py --task Isaac-Wmp-Visual-Direct-v0 --num_envs 1 --num_steps 4 --headless --enable_cameras
```

短训练：

```bash
python scripts/rsl_rl/train.py --task Isaac-Wmp-Direct-v0 --headless --num_envs 8 --max_iterations 1
```

恢复训练：

```bash
python scripts/rsl_rl/train.py --task Isaac-Wmp-Direct-v0 --headless --num_envs 8 --resume --load_run <run_name> --checkpoint <checkpoint> --max_iterations 1
```

说明：

- 传入 `--load_run` 或 `--checkpoint` 时必须同时传 `--resume`
- 恢复训练时，`--max_iterations` 表示“额外继续训练多少轮”，不是总训练预算

播放并导出 JIT / ONNX：

```bash
python scripts/rsl_rl/play.py --task Isaac-Wmp-Direct-Play-v0 --headless --num_envs 1 --num_steps 200 --checkpoint <checkpoint_path>
```

导出产物会写入 `<run_dir>/exported/policy.pt` 和 `<run_dir>/exported/policy.onnx`。

## checkpoint 边界

当前 checkpoint 语义是标准 RSL-RL 参数级恢复：

- 会恢复 actor、critic、optimizer 和迭代计数
- 不会恢复 Python / NumPy / PyTorch RNG 状态
- 不会恢复环境内部状态
- 不会恢复 rollout storage

`play.py` 只加载 actor 权重，不依赖 critic 或 optimizer，因此更适合作为推理与导出入口。

旧 checkpoint 仅在以下条件同时满足时才有机会兼容：

- 动作维度一致
- 观测维度与顺序一致
- 归一化契约一致
- 网络结构一致

如果这些条件被破坏，默认需要重新训练。

## 资产、缓存与已知差异

当前实现使用 Isaac Lab / Isaac Sim 提供的 Unitree A1 USD 资产，而不是原始 WMP 仓库中的 URDF 资产。这意味着：

- 需要能够访问 Isaac Sim / Nucleus 资产缓存
- 首次运行可能会触发资产下载或缓存构建
- 质量、惯量、碰撞和接触的数值细节不保证与原始 WMP 逐项一致

已知尚未严格完成的部分：

- 原始 WMP 资产与当前 USD 资产的严格动力学等价
- 原始固定轨迹 / 奖励分项 / 参考 checkpoint 的逐项数值对齐
- 目标规模并行环境下的最终性能验证
- 原 terrain 中部分 rough noise、负坡和随机台阶宽度细节

按严格口径，这些缺口意味着“首轮 PPO 基线已实现并验证”，但还不能宣称全部迁移工作完全结束。

## 许可说明

当前仓库代码基于 Isaac Lab 外部项目模板扩展而来，机器人资产使用 Isaac Lab / Isaac Sim 提供的 Unitree A1 资源。相关资产许可请以 Isaac Lab / Isaac Sim 分发内容中的 BSD-3-Clause 与原始资源声明为准。


