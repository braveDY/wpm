## 下载模型参数
```bash
rsync -avzP --ignore-existing autodl:/root/wpm/logs/ /home/brave/isaaclab_pj/wmp/logs/
```

说明：当前项目本地化的是任务相关代码；训练、播放和配置解析仍复用 `isaaclab_tasks.utils`。


## 训练

```bash
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0
```

常见附加参数：

```bash
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0 --num_envs 1024
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0 --max_iterations 1000
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0 --headless
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0 --video
```

训练日志默认写入：

```text
logs/rsl_rl/unitree_a1_rough/
```

复杂地形精调时，先把 [env_cfg.py](/home/brave/isaaclab_pj/wmp/source/wmp/wmp/robots/a1/env_cfg.py:9) 里的 `FINETUNE` 改成 `True`，然后继续加载已有 checkpoint：

```bash
python scripts/rsl_rl/train.py --task Wmp-Velocity-Rough-A1-v0 --resume --load_run <原训练目录名> --checkpoint <checkpoint文件名>
```

精调日志默认写入：

```text
logs/rsl_rl/unitree_a1_rough_finetune/
```

## 播放训练结果

```bash
python scripts/rsl_rl/play.py --task Wmp-Velocity-Rough-A1-Play-v0 --headless --video --video_length 1000
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

- 将公共 velocity 逻辑下沉到 `wmp.common`
- 将机器人特化配置集中到 `wmp.robots.<robot>`
- 继续复用 `isaaclab_tasks.utils` 中现成的配置工具和 Hydra 适配

这种拆分便于后续继续添加其他机器人的 locomotion velocity 任务，同时避免重复复制官方通用工具代码。

## 当前限制

- 当前只迁移了 `A1 rough velocity` 任务
- 当前默认训练算法只接了 `RSL-RL`
- 当前没有加入 `flat` 版本任务
- 当前没有主动补充测试或静态检查脚本

## 下一步建议

- 增加 `go1` 或 `go2` 的同构任务
- 统一 `scripts/list_envs.py` 的说明文字和过滤逻辑
- 补一套最小自检流程，验证任务注册、环境创建和配置加载
