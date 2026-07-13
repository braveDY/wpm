# WMP A1 Rough Task

基于 Isaac Lab 的 Unitree A1 粗糙地形速度跟踪任务，采用 manager-based 环境配置，并使用 RSL-RL PPO 训练。

当前项目已经将官方 `isaaclab_tasks` 中的 A1 rough velocity 任务迁移到本地 `wmp` 包中，运行时不再依赖 `isaaclab_tasks`。

## 当前任务

- 训练任务：`Wmp-Velocity-Rough-A1-v0`
- 播放任务：`Wmp-Velocity-Rough-A1-Play-v0`
- 机器人资产：`isaaclab_assets.robots.unitree.UNITREE_A1_CFG`
- 训练算法：`RSL-RL PPO`

## 目录结构

```text
wmp/
├── scripts/
│   ├── list_envs.py
│   ├── random_agent.py
│   ├── zero_agent.py
│   └── rsl_rl/
│       ├── cli_args.py
│       ├── play.py
│       └── train.py
└── source/
    └── wmp/
        ├── config/
        ├── setup.py
        └── wmp/
            ├── __init__.py
            ├── common/
            │   ├── env_cfg.py
            │   └── mdp.py
            ├── robots/
            │   └── a1/
            │       ├── __init__.py
            │       ├── agent_cfg.py
            │       └── env_cfg.py
            └── utils/
                ├── hydra.py
                └── parse_cfg.py
```

## 代码组织

- `source/wmp/wmp/common/env_cfg.py`
  - 共享的 velocity rough 任务基类。
  - 定义场景、观测、动作、奖励、事件、终止条件和 curriculum。
- `source/wmp/wmp/common/mdp.py`
  - velocity 任务专用的 `mdp` 扩展。
  - 在 `isaaclab.envs.mdp` 基础上补充 reward、termination 和 curriculum 逻辑。
- `source/wmp/wmp/robots/a1/env_cfg.py`
  - A1 的特化环境配置。
  - 负责替换机器人资产、调整地形参数、设置 A1 专用 reward 和 event。
- `source/wmp/wmp/robots/a1/agent_cfg.py`
  - A1 的 PPO 训练超参数。
- `source/wmp/wmp/robots/a1/__init__.py`
  - 注册 `Wmp-Velocity-Rough-A1-v0` 和 `Wmp-Velocity-Rough-A1-Play-v0`。
- `source/wmp/wmp/utils/parse_cfg.py`
  - 本地化的任务配置解析工具。
- `source/wmp/wmp/utils/hydra.py`
  - 本地化的 Hydra 任务配置适配层。

## 环境要求

- Python `>= 3.10`
- Isaac Lab
- `isaaclab`
- `isaaclab_assets`
- `isaaclab_rl`
- `rsl-rl-lib==5.0.1`
- `trimesh`

项目当前通过 `source/wmp/setup.py` 安装 Python 包，子包发现已改为 `find_packages()`。

## 安装

如果你已经处于 Isaac Lab 的 Python 环境中，通常可直接在项目根目录执行：

```bash
pip install -e source/wmp
```

如果你的 Isaac Lab 工作流要求通过扩展方式加载，也可以继续沿用现有扩展加载方式，只要确保 `wmp` 包已在 Python 路径中可见。

## 查看已注册任务

```bash
python scripts/list_envs.py --keyword Wmp
```

如果注册正常，应至少能看到：

```text
Wmp-Velocity-Rough-A1-v0
Wmp-Velocity-Rough-A1-Play-v0
```

## 训练

```bash
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0
```

常见附加参数：

```bash
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0 --num_envs 1024
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0 --max_iterations 1000
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0 --seed 42
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0 --video
```

训练日志默认写入：

```text
logs/rsl_rl/unitree_a1_rough/
```

## 播放训练结果

```bash
python scripts/rsl_rl/play.py --task Wmp-Velocity-Rough-A1-Play-v0
```

如果要指定某个 checkpoint：

```bash
python scripts/rsl_rl/play.py --task Wmp-Velocity-Rough-A1-Play-v0 --checkpoint /abs/path/to/model.pt
```

## 随机动作和零动作调试

随机动作：

```bash
python scripts/random_agent.py --task Wmp-Velocity-Rough-A1-v0
```

零动作：

```bash
python scripts/zero_agent.py --task Wmp-Velocity-Rough-A1-v0
```

这两个脚本主要用于检查：

- 任务是否成功注册
- 环境是否能正常创建
- 观测和动作空间是否符合预期

## 扩展到其他机器人

当前目录按机器人组织，后续扩展建议保持同样约定：

```text
source/wmp/wmp/robots/
├── a1/
├── go1/
├── go2/
└── ...
```

新增一个机器人时，建议最少补齐三部分：

- `robots/<robot>/env_cfg.py`
- `robots/<robot>/agent_cfg.py`
- `robots/<robot>/__init__.py`

其中：

- `env_cfg.py` 继承 `wmp.common.env_cfg.LocomotionVelocityRoughEnvCfg`
- `agent_cfg.py` 放该机器人的 PPO 配置
- `__init__.py` 负责 `gym.register(...)`

## 与官方任务的关系

当前实现来源于 Isaac Lab 官方 A1 rough velocity manager-based 任务，但已经做了本地化拆分：

- 去掉了对 `isaaclab_tasks` 的运行时依赖
- 将公共 velocity 逻辑下沉到 `wmp.common`
- 将任务解析和 Hydra 适配下沉到 `wmp.utils`
- 将机器人特化配置集中到 `wmp.robots.<robot>`

这种拆分便于后续继续添加其他机器人的 locomotion velocity 任务，而不需要重新依赖官方任务包。

## 当前限制

- 当前只迁移了 `A1 rough velocity` 任务
- 当前默认训练算法只接了 `RSL-RL`
- 当前没有加入 `flat` 版本任务
- 当前没有主动补充测试或静态检查脚本

## 下一步建议

- 增加 `go1` 或 `go2` 的同构任务
- 统一 `scripts/list_envs.py` 的说明文字和过滤逻辑
- 补一套最小自检流程，验证任务注册、环境创建和配置加载
