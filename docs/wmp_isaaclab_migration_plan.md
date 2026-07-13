# WMP → Isaac Lab 迁移修订方案

> 状态：首轮 PPO 基线已实施并完成定向验证  
> 基线：2026-07-12 当前工作区 + 原始 WMP `c232c115ada4517453ebded5019078ba055456de`  
> 目标：在保持可回滚和公共接口可控的前提下，将 WMP 的环境语义、训练链路与交付方式迁移到 Isaac Lab。
> 当前结论：标准 PPO 基线链路已闭环，但严格资产/数值对齐与目标规模性能验证仍未完成，不能宣称全部迁移结束。

## 0. 当前实施状态

### 0.1 当前范围

当前工作区已经实现并验证的范围是：

- 真实 WMP/A1 语义的 `DirectRLEnv` 环境
- `rsl-rl-lib==5.0.1` 标准 PPO 基线
- 训练、恢复训练、播放、JIT 导出和 ONNX 导出
- canonical 入口 `wmp.env`、`wmp.env_cfg`、`wmp.rsl_rl_ppo_cfg`
- canonical 任务 `Isaac-Wmp-Direct-v0`、`Isaac-Wmp-Direct-Play-v0`、`Isaac-Wmp-Visual-Direct-v0`
- 兼容别名 `Template-Wmp-Direct-v0`
- `source` 目录已进一步收口，不再保留 `wmp.tasks.*` 旧 Python 兼容子树

当前明确不在本轮范围内的内容：

- AMP
- Dreamer / world model 训练
- 深度策略训练
- 自定义 WMP runner

这些能力需要在标准 PPO 基线之外单独立项。

### 0.2 Phase 实施结果

| Phase | 当前状态 | 说明 |
| --- | --- | --- |
| Phase 0 | 部分完成 | 已固定源码与版本组合，但缺少原始固定轨迹、奖励分项样本和参考 checkpoint |
| Phase 1 | 完成 | canonical 路径、任务注册和打包入口已收口，旧 `wmp.tasks.*` 兼容子树已移除 |
| Phase 2 | 基础实现完成 | A1 资产、地形、传感器、随机化、控制频率和 reset 链路已接入 |
| Phase 3 | 完成 | 12 维动作、显式 PD、限幅和关节顺序校验已落地 |
| Phase 4 | 基础实现完成 | 45 维 actor、285 维 critic、done/time_out、WMP 辅助观测接口已落地 |
| Phase 5 | 基础实现完成 | 奖励、命令、push、terrain curriculum、reward curriculum 已接入 |
| Phase 6 | 完成 | 短训练、恢复训练、播放、JIT 导出、ONNX 导出已通过 |
| Phase 7 | 部分完成 | 仅完成 1/8 环境 smoke 与短训练，目标规模性能仍未验证 |

### 0.3 当前版本矩阵

| 组件 | 版本 |
| --- | --- |
| 原始 WMP | `c232c115ada4517453ebded5019078ba055456de` |
| Isaac Sim | `5.1.0.0` |
| Isaac Lab | `2.3.2` |
| Isaac Lab Python 包元数据 | `0.54.4` |
| `isaaclab_rl` | `0.5.2` |
| `rsl-rl-lib` | `5.0.1` |
| PyTorch | `2.7.0+cu128` |

### 0.4 已执行验证

当前验证使用的命令前缀为：

```bash
PYTHONPATH="$PWD/source/wmp" /home/brave/miniconda3/envs/env_isaaclab/bin/python
```

已通过的定向验证如下：

1. 8 环境随机动作 32 步

   ```bash
   python scripts/random_agent.py --task Isaac-Wmp-Direct-v0 --num_envs 8 --num_steps 32 --headless
   ```

2. Visual 单环境 4 步

   ```bash
   python scripts/zero_agent.py --task Isaac-Wmp-Visual-Direct-v0 --num_envs 1 --num_steps 4 --headless --enable_cameras
   ```

3. 8 环境、1 轮 PPO 短训练

   ```bash
   python scripts/rsl_rl/train.py --task Isaac-Wmp-Direct-v0 --headless --num_envs 8 --max_iterations 1
   ```

   产物：

   ```text
   logs/rsl_rl/wmp_a1/<timestamp>_baseline/
   ├── model_0.pt
   ├── params/env.yaml
   ├── params/agent.yaml
   └── git/wmp.diff
   ```

4. checkpoint 恢复训练

   ```bash
   python scripts/rsl_rl/train.py --task Isaac-Wmp-Direct-v0 --headless --num_envs 8 --max_iterations 1 --resume --load_run <run_name> --checkpoint model_0.pt
   ```

5. 播放 + JIT / ONNX 导出

   ```bash
   python scripts/rsl_rl/play.py --task Isaac-Wmp-Direct-Play-v0 --headless --num_envs 1 --num_steps 4 --checkpoint <checkpoint_path>
   ```

   产物：

   ```text
   exported/policy.pt
   exported/policy.onnx
   ```

### 0.5 当前仍受限的事项

- 当前 A1 USD 资产不等同于原始 WMP URDF，严格动力学数值等价尚未完成
- 缺少原始固定轨迹、奖励分项样本和参考 checkpoint，无法完成严格离线对齐
- 目标规模并行环境下的吞吐、显存和稳定性尚未完成最终验证
- AMP / Dreamer / 深度策略训练不在首轮范围内

## 1. 总体判断

当前仓库已经具备 Isaac Lab 外部项目模板、`DirectRLEnv` 环境骨架、Gym 注册入口和 RSL-RL 训练/播放脚本，但还不能视为完成了 WMP 迁移：

1. 当前 `WmpEnv`、`WmpEnvCfg` 和 `PPORunnerCfg` 仍是 Cartpole Direct 示例的改名版本，资产、动作、观测、奖励、终止条件和 PPO 实验名都不是 WMP 的真实定义。
2. 当前工作区主要完成了从 `wmp.tasks.direct.wmp.*` 到 `wmp.*` 的目录扁平化，环境行为本身没有变化。
3. 扁平化尚未闭环：仓库自带的任务列表、零动作、随机动作、训练和播放脚本仍导入已经删除的 `wmp.tasks`。
4. 根目录的 `rsl_rl/` 是独立的上游源码仓库，不是 WMP 实现；运行脚本仍依赖 Python 环境中安装的 `rsl-rl-lib` 和 `isaaclab_rl` 适配层。
5. `DirectRLEnv` 适合首轮“行为等价迁移”。首轮不应同时改写为 Manager-Based 环境、重写 PPO、改变 checkpoint 格式或做大范围目录重构。

因此，迁移应拆成两条连续轨道：

- **轨道 A：集成结构收口。** 先修复注册、导入、打包和兼容路径，使现有行为在新路径下完整可用。
- **轨道 B：WMP 行为迁移。** 在原 WMP 契约冻结后，逐项替换 Cartpole 占位资产和 MDP 逻辑，并用数值对齐验证语义。

两条轨道不能混在一个不可分割的改动中。目录迁移、环境语义迁移、算法升级和无关文件清理都应具有独立验收点和回滚边界。

## 2. 实施前基线与主要问题（追溯）

| 项目 | 当前状态 | 迁移影响 |
| --- | --- | --- |
| Gym 任务 ID | `Template-Wmp-Direct-v0` | 结构迁移阶段应保持稳定；正式改名需要兼容别名或单独的破坏性变更 |
| 注册位置 | `wmp.__init__` 顶层注册 | 可作为唯一注册所有者，但所有调用方必须统一导入 `wmp` |
| 环境入口 | `wmp.env:WmpEnv` | 新的扁平 canonical 路径 |
| 环境配置入口 | `wmp.env_cfg:WmpEnvCfg` | 新的扁平 canonical 路径 |
| RSL-RL 配置入口 | `wmp.rsl_rl_ppo_cfg:PPORunnerCfg` | 新的扁平 canonical 路径 |
| 旧路径 | `wmp.tasks.direct.wmp.*` | 历史基线中曾保留；当前工作区已移除该兼容子树 |
| 环境实现 | Cartpole，动作 1 维、策略观测 4 维 | 只能验证 Isaac Lab 接线，不能代表 WMP 语义 |
| 仿真频率 | `dt=1/120`、`decimation=2`，策略频率 60 Hz | 必须与原 WMP 控制频率核对，不能默认沿用 |
| Episode | 5 秒，即当前配置下 300 个策略步 | 必须与原 WMP 超时定义核对 |
| 训练实验名 | `cartpole_direct` | 会污染日志与 checkpoint 目录，应在行为迁移阶段改为稳定的 WMP 名称 |
| 训练脚本 | Isaac Lab 标准 RSL-RL 入口 | 应复用，不需要另写训练循环 |
| RSL-RL 版本 | 当前运行环境与仓库内源码可能不一致 | 必须锁定实际安装版本，禁止依据未安装的本地源码推断运行行为 |
| 打包 | `setup.py` 仅列出 `packages=["wmp"]` | 若保留旧路径兼容子包，需要改为包发现，否则 wheel 可能漏包 |
| UI 与变更日志 | 工作区删除，但 README 仍引用 UI 示例 | 与 WMP 环境迁移无直接关系，应单独决策 |

当前环境还存在以下契约风险：

- 动作没有在环境侧明确约定裁剪范围，而随机动作脚本默认生成 `[-1, 1]`。
- `initial_pole_angle_range` 的注释写弧度，使用时又乘以 `pi`，说明单位约定尚未冻结。
- `compute_rewards()` 接收但未使用 `cart_pos`，且没有把奖励分项写入 `extras["log"]`。
- 当前只有 `"policy"` 观测组，`state_space=0`，没有显式 actor/critic 或 privileged observation 映射。
- 当前 PPO 配置依赖 Isaac Lab 对旧式 `policy` 字段和默认 `"policy"` 观测组的兼容处理；升级 RSL-RL 时存在移除风险。
- 新顶层 Python 文件是未跟踪文件时，单看普通 `git diff` 可能漏掉实际迁移内容，实施时必须同时检查 `git status`。

## 3. 迁移目标与非目标

### 3.1 目标

1. WMP 任务能够通过稳定 Gym ID 注册，并由 Isaac Lab 标准脚本创建。
2. Isaac Lab 环境与原 WMP 在动作、观测、奖励、重置、终止、超时和随机化语义上可解释地对齐。
3. 环境输出满足 `RslRlVecEnvWrapper` 和 RSL-RL runner 的张量契约。
4. 训练、恢复训练、播放、JIT 导出和 ONNX 导出使用同一套注册与配置入口。
5. 每个迁移阶段都有明确的输入、产物、验收条件和回滚方式。
6. 外部调用方未知时，旧 Python 导入路径至少保留一个明确的弃用周期。

### 3.2 首轮非目标

- 不在首轮把 `DirectRLEnv` 改写为 Manager-Based 环境。
- 不重写 Isaac Lab 的训练、播放或 Hydra 配置加载流程。
- 不同时修改 PPO 算法、网络结构和环境语义来追求训练曲线。
- 不默认引入或维护 `rsl_rl/` 源码 fork。
- 不把 UI 示例、CHANGELOG、项目元数据清理与核心环境迁移绑定。
- 不在行为等价之前做大规模性能优化或领域随机化扩展。
- 不承诺自动兼容观察维度、动作维度或网络结构已经改变的旧 checkpoint。

## 4. 默认假设

在没有新增需求说明时，按以下假设实施：

1. 原 WMP 的可运行代码、配置、资产和至少一个参考 checkpoint 是行为真值来源。
2. 首轮迁移采用单智能体、连续动作、向量化的 `DirectRLEnv`。
3. GPU 物理仿真是主要运行路径；CPU 只承担最小规模调试，不作为目标吞吐基准。
4. 结构迁移期间保持 `Template-Wmp-Direct-v0`、`WmpEnv`、`WmpEnvCfg` 和 `PPORunnerCfg` 不变。
5. 正式任务名默认改为 `Isaac-Wmp-Direct-v0`，但只能在兼容别名和调用方迁移方案确定后单独实施。
6. 新 canonical Python 路径为 `wmp.env`、`wmp.env_cfg` 和 `wmp.rsl_rl_ppo_cfg`。
7. 外部是否存在旧深层导入调用方目前未知，因此默认保留一个弃用周期的只读重导出兼容层。
8. RSL-RL 通过 Isaac Lab 的 `isaaclab_rl.rsl_rl` 配置与 wrapper 接入；本地 `rsl_rl/` 默认不安装、不修改、不纳入主迁移依赖。
9. 原 WMP 的观测顺序、动作含义、坐标系、单位、奖励公式和终止逻辑优先于当前 Cartpole 占位实现。
10. 领域随机化、课程学习和传感器噪声在基础确定性语义对齐后再恢复。
11. 旧 checkpoint 只有在动作/观测维度、顺序、归一化和网络结构都不变时才默认可直接加载。
12. 如果原 WMP 资料缺失，最多只能完成轨道 A，不能宣称完成行为迁移。

实施 Phase 0 前必须确认的外部输入包括：

- 原 WMP 仓库地址、固定 commit 或发布版本。
- Isaac Sim、Isaac Lab、`isaaclab_rl`、`rsl-rl-lib` 和 PyTorch 的目标版本组合。
- 机器人 USD/URDF/MJCF、地形、材质、传感器和其他资源的来源及许可证。
- 原 WMP 的训练和评估启动命令、默认 seed、目标硬件与目标并行环境数。
- 可用于数值对齐的状态样本、轨迹、奖励分项、episode 统计和 checkpoint。

## 5. 目标架构与依赖边界

### 5.1 推荐目录

首轮保持单任务扁平 canonical 接口：

```text
source/wmp/wmp/
├── __init__.py            # 唯一 Gym 注册所有者
├── env.py                 # WmpEnv 与纯张量 MDP 辅助函数
├── env_cfg.py             # WmpEnvCfg 与资产/仿真/任务配置
└── rsl_rl_ppo_cfg.py      # PPORunnerCfg
```

当前不再保留旧 `wmp.tasks.*` Python 兼容层。如果后续确定会维护多个独立任务，应单独决定是否恢复 Isaac Lab 常见的 `tasks/direct/<task>` 组织方式，不能在同一版本里长期混用两套 canonical 路径。

### 5.2 依赖边界

- `WmpEnvCfg` 负责声明仿真、场景、资产、空间维度、控制参数、重置范围和奖励权重。
- `WmpEnv` 只负责 Isaac Lab 生命周期、张量缓存、动作应用、观测、奖励、终止和 reset。
- 机器人资源配置可以拆为独立资产配置文件，但环境不得依赖训练算法内部对象。
- `PPORunnerCfg` 只描述 runner、actor、critic、PPO 和观测组映射，不承载环境业务逻辑。
- 训练与播放继续使用 `scripts/rsl_rl/train.py` 和 `scripts/rsl_rl/play.py`。
- `RslRlVecEnvWrapper` 是 Isaac Lab 环境与 RSL-RL 之间的唯一适配边界。
- 原 WMP 特有算法若确实需要修改 RSL-RL，应在基础 PPO 迁移通过后建立独立 RFC 和独立依赖策略。

## 6. 公共接口与数据契约

### 6.1 稳定 Python/Gym 接口

| 接口 | 结构迁移期约定 |
| --- | --- |
| Gym ID | 保持 `Template-Wmp-Direct-v0` |
| 环境入口 | `wmp.env:WmpEnv` |
| 环境配置入口 | `wmp.env_cfg:WmpEnvCfg` |
| RSL-RL 配置入口 | `wmp.rsl_rl_ppo_cfg:PPORunnerCfg` |
| 注册键 | `env_cfg_entry_point`、`rsl_rl_cfg_entry_point` |
| 注册所有者 | 仅 `wmp.__init__` |
| 旧环境路径 | 当前不再提供 |
| 旧配置路径 | 当前不再提供 |
| 旧 PPO 路径 | 当前不再提供 |

任务正式改名时采用以下策略：

1. 新增 `Isaac-Wmp-Direct-v0` 作为 canonical ID。
2. 旧 ID 暂时注册到同一环境和同一配置，并明确弃用窗口。
3. 训练日志、文档、脚本示例和新 checkpoint 只使用新 ID。
4. 确认没有外部依赖后，再单独移除旧 ID。

### 6.2 `DirectRLEnv` 生命周期契约

`WmpEnv` 必须实现并保持以下职责：

| 方法 | 输入/输出 | 约定 |
| --- | --- | --- |
| `_setup_scene()` | 无返回值 | 创建资产、地面/地形、灯光、传感器，克隆环境并注册到 `self.scene` |
| `_pre_physics_step(actions)` | `[N, A]` | 校验、裁剪、缩放或解码策略动作，只更新动作目标缓存 |
| `_apply_action()` | 无返回值 | 每个物理子步向 actuator 写入目标，不重复做策略级随机处理 |
| `_get_dones()` | `terminated, time_out` | 两个布尔张量均为 `[N]`，严格区分任务失败与时间截断 |
| `_get_rewards()` | `[N]` | 返回有限的浮点奖励，并记录可诊断的奖励分项 |
| `_get_observations()` | 观测字典 | 至少返回 `{"policy": tensor}`；如有特权信息，显式增加 `"critic"` |
| `_reset_idx(env_ids)` | 环境索引 | 只重置指定环境，调用基类清理 episode 状态，并按环境原点写回 root/joint 状态 |

应以 Isaac Lab 实际版本的调用顺序为准，并在迁移说明中固定以下语义：

- 结束步的 reward 和 done 对应重置前状态。
- 自动重置后返回的 observation 对应重置后状态。
- `time_out` 不能与任务失败混为一类，否则会改变价值 bootstrap。
- 不依赖 `self.robot._ALL_INDICES` 等私有字段作为长期公共接口；优先使用稳定公开接口或自行维护全环境索引。

### 6.3 张量契约

| 数据 | 形状 | dtype/device | 语义要求 |
| --- | --- | --- | --- |
| 策略动作 | `[num_envs, action_dim]` | 通常 `float32`，环境设备 | 明确范围、单位、scale、offset、关节顺序和 actuator 类型 |
| `observations["policy"]` | `[num_envs, actor_obs_dim]` | `float32`，环境设备 | 顺序、坐标系、归一化、裁剪和历史堆叠必须固定 |
| `observations["critic"]` | `[num_envs, critic_obs_dim]`，可选 | `float32`，环境设备 | 只包含训练允许使用的特权信息 |
| reward | `[num_envs]` | `float32`，环境设备 | 每个策略步一个标量，禁止 NaN/Inf |
| terminated | `[num_envs]` | `bool`，环境设备 | 任务失败或成功终止 |
| time_out | `[num_envs]` | `bool`，环境设备 | 仅时间限制截断 |
| episode 日志 | 标量或 `[num_envs]` | 可归约 | 通过 `extras["log"]` 使用稳定、带命名空间的键 |

冻结观测契约时必须逐项记录：

- 字段名称、拼接顺序和维度。
- 世界系、机器人基座系、局部航向系等坐标系。
- 米、弧度、牛顿、牛米、秒等单位。
- 是否做重力投影、角度 wrap、速度缩放、归一化、裁剪或噪声。
- 当前值、历史值、命令、上一动作和相位等时间信息。
- actor 与 critic 各自使用哪些观测组。

### 6.4 RSL-RL 接口

训练链路保持：

```text
Gym 注册
  → hydra_task_config 加载 env_cfg/agent_cfg
  → gym.make()
  → RslRlVecEnvWrapper
  → OnPolicyRunner
  → learn()/load()/export
```

公共 CLI 覆盖项至少包括：

- `--task`、`--num_envs`、`--device`、`--seed`、`--max_iterations`。
- `--resume`、`--load_run`、`--checkpoint`。
- `--logger`、`--experiment_name`、`--run_name`。
- 需要录制时的 `--video`、`--video_length` 和相机开关。

版本锁定后，`PPORunnerCfg` 应显式声明：

- `obs_groups` 中 actor、critic 及其他模型输入的映射。
- actor 和 critic 网络结构。
- 动作分布初始标准差、观测归一化与动作裁剪。
- PPO 超参数、rollout 长度、mini-batch 数量和保存周期。
- `experiment_name`、seed、device、resume 和 logger 默认值。

播放与导出必须保证：

- `runner.load()` 加载的配置与 checkpoint 网络结构匹配。
- 在线策略按 runner 约定接收 TensorDict 观测。
- JIT/ONNX 导出输入按 `obs_groups["actor"]` 的稳定顺序构造。
- 有循环状态的策略在 episode done 后重置内部状态。
- 恢复训练和只做推理分别验证；不能以播放成功代替恢复训练兼容性。

## 7. 分阶段实施

### Phase 0：冻结基线与原 WMP 契约

**工作项**

1. 固定原 WMP、Isaac Lab、Isaac Sim、RSL-RL 和 PyTorch 版本。
2. 形成资产、DOF/body 名称与顺序、控制频率和 actuator 配置清单。
3. 形成动作、观测、奖励、终止、超时、reset 和随机化契约表。
4. 从原 WMP 保存一组固定 seed 的 reset 样本、状态轨迹、动作序列、奖励分项和 episode 汇总。
5. 保存至少一个可复现的参考训练或评估结果。

**验收门槛**

- 所有维度、顺序、坐标系、单位和公式都有明确来源。
- 行为迁移范围内不存在影响实现的 `TBD`。
- 缺少原实现或参考数据时，停止在轨道 A，不进入 Phase 2。

**回滚**

- 本阶段只产出规格和参考数据，不改变运行代码。

### Phase 1：完成集成结构收口

**工作项**

1. 以 `wmp.env`、`wmp.env_cfg`、`wmp.rsl_rl_ppo_cfg` 为唯一权威实现。
2. 保证只有 `wmp.__init__` 调用 `gym.register()`。
3. 已完成：仓库内部调用方已统一迁移为 `import wmp`。
4. 当前不再保留兼容子包，`setup.py` 仅声明 canonical `wmp` 包。
5. 保持 Gym ID、空间维度、环境行为和 PPO 参数不变。
6. 将 UI、CHANGELOG、README 清理和 `rsl_rl/` 引入决策拆出本阶段。

**验收门槛**

- 新旧约定路径均能按方案导入。
- 同一 Gym ID 只有一个注册点，不发生重复注册。
- 仓库任务列表、dummy agent、训练和播放入口使用一致的注册导入方式。
- wheel/editable install 的包内容一致，不遗漏兼容模块。

**回滚**

- 未发布新路径时可整体恢复旧目录。
- 新路径已被使用后，回滚只能切换实现或注册入口，不能直接删除新路径。

### Phase 2：迁移资产、场景与仿真配置

**工作项**

1. 用真实 WMP 机器人配置替换 `CARTPOLE_CFG`。
2. 配置正确的 prim path、关节、刚体、执行器、碰撞、质量、阻尼、摩擦和初始状态。
3. 按原 WMP 恢复地形、环境间距、传感器、灯光及全局碰撞对象。
4. 对齐 `dt`、`decimation`、render interval、episode 时长和 physics material。
5. 在初始化时解析并校验关键 joint/body 索引，名称不匹配时尽早失败。

**验收门槛**

- 单环境能够创建、reset 和无控制步进。
- 多环境 root pose 正确叠加 `scene.env_origins`，环境之间无非预期耦合。
- 静止状态、重力方向、接触状态和关节默认状态与原 WMP 一致或差异有解释。

**回滚**

- 保留 Cartpole 仅作为独立的基础接线样例，不与 WMP 正式任务共用同一 ID。

### Phase 3：迁移动作与控制链路

**工作项**

1. 固定策略动作范围和 `action_dim`。
2. 明确动作到执行器目标的公式，包括 scale、offset、默认姿态、关节顺序和单位。
3. 区分 position、velocity、effort、PD target 或混合控制。
4. 明确动作裁剪由环境、wrapper 或策略哪一层负责，避免重复裁剪。
5. 明确 decimation 内动作保持、延迟、滤波、上一动作和控制噪声语义。

**验收门槛**

- 给定固定状态和固定动作，Isaac Lab 生成的 actuator 目标与原 WMP 数值一致。
- 零动作、边界动作和超范围动作行为明确且无 NaN/Inf。
- 关节顺序错误能够通过断言或诊断信息立即发现。

**回滚**

- 动作适配层保持独立，必要时可切回旧映射而不改观测和奖励。

### Phase 4：迁移 reset、观测、终止与超时

**工作项**

1. 先迁移确定性 reset，再恢复随机 reset。
2. 对齐 root pose、root velocity、joint state、命令、历史缓冲和传感器状态。
3. 按冻结顺序实现 `policy` 和可选 `critic` 观测。
4. 对齐角度表示、坐标变换、归一化、裁剪、噪声和历史堆叠。
5. 独立实现 `terminated` 与 `time_out`，并核对结束步/重置后观测语义。
6. 确保 `_reset_idx(env_ids)` 不修改未选中的环境。

**验收门槛**

- 固定 seed 下，reset 分布统计与原 WMP 对齐。
- 对同一批离线状态，全部观测字段逐项通过误差阈值。
- 单环境 reset 不影响其他环境。
- terminated 与 time_out 的触发条件、优先级和计数一致。

**回滚**

- 确定性 reset、随机 reset、观测噪声和历史堆叠分别通过配置开关隔离。

### Phase 5：迁移奖励、命令与随机化

**工作项**

1. 将奖励拆成可单独计算和记录的纯张量分项。
2. 固定每个分项的公式、权重、单位、dt 缩放、裁剪和终止奖励时机。
3. 恢复命令采样、课程学习、push、质量/摩擦随机化和传感器噪声。
4. 在 `extras["log"]` 输出稳定的奖励、成功率、失败原因和 episode 长度指标。
5. 对奖励函数增加有限值保护和关键状态诊断。

**验收门槛**

- 对相同离线状态和 done 标记，奖励分项与总奖励在阈值内对齐。
- 关闭随机化时结果可复现；逐项开启时能定位统计变化。
- 训练日志能够区分超时、失败和成功终止。

**回滚**

- 每个奖励分项和随机化项均可单独禁用，不需要回退整个环境。

### Phase 6：接入 RSL-RL 训练、恢复、播放与导出

**工作项**

1. 锁定实际运行的 `rsl-rl-lib` 版本，解决环境安装版与本地源码版差异。
2. 将配置迁移到目标版本的显式 actor、critic、`obs_groups` 和 algorithm 字段。
3. 先保持原 WMP PPO 超参数；环境语义对齐后再调参。
4. 统一 `experiment_name`、run 命名、checkpoint 路径和配置落盘。
5. 分别验证新训练、恢复训练、播放、JIT 导出和 ONNX 导出。
6. 明确旧 checkpoint 的可加载范围，必要时提供单独转换脚本或宣布不兼容。

**验收门槛**

- 短训练能够产生 `params/env.yaml`、`params/agent.yaml` 和 checkpoint。
- 恢复训练的迭代、optimizer、normalizer 和随机状态行为符合预期。
- 播放时动作与观测均为有限值，episode 能正常 reset。
- JIT 和 ONNX 产物能够在固定输入上与在线策略输出对齐。

**回滚**

- 保留行为迁移前最后一个可用配置；算法配置升级不与环境张量变化混在同一步。

### Phase 7：规模、性能、复现与切换

**工作项**

1. 按 1、8/32、目标规模三个层级扩大 `num_envs`。
2. 记录每秒环境步数、GPU/CPU 利用率、显存、仿真耗时和学习耗时。
3. 在固定版本、seed 和硬件下重复短训练，比较 episode 指标和初期学习曲线。
4. 更新任务名称、安装说明、训练/播放命令、资产来源和兼容声明。
5. 确认外部调用方迁移后，独立移除旧导入路径和旧 Gym ID。
6. 决定 `rsl_rl/`、UI 示例和 CHANGELOG 的最终归属，不把这些决策混入核心迁移提交。

**验收门槛**

- 目标并行规模下无持续 NaN、OOM、异常接触或环境间串扰。
- 性能达到预先约定阈值，或与基线差异有明确解释。
- 固定 seed 的统计波动在约定范围内。
- 文档示例、任务注册、日志目录和导出产物命名一致。

**回滚**

- 正式切换前保留旧任务 ID 和上一套可运行配置。
- 清理兼容层作为最后的独立改动，可单独恢复。

## 8. 验证方案

以下验证项中，L1-L6 已在当前工作区完成定向执行，L7 仍是后续扩大规模时的计划项。

### 8.1 L0：静态契约核对

- 核对任务 ID、三个 entry point 和唯一注册位置。
- 搜索所有 `wmp.tasks`、旧模块字符串和 `Template-` 调用方。
- 核对打包结果会包含 canonical 模块和兼容模块。
- 核对动作、观测、reward、terminated、time_out 的 shape/dtype/device 文档。

### 8.2 L1：注册与单环境 smoke

建议命令：

```bash
python scripts/list_envs.py --keyword Wmp
python scripts/zero_agent.py --task Isaac-Wmp-Direct-v0 --num_envs 1 --headless
python scripts/random_agent.py --task Isaac-Wmp-Direct-v0 --num_envs 1 --headless
```

验收：

- 任务能被发现并创建。
- reset、零动作和随机动作可连续执行。
- observation、reward、done 和关键状态均为有限值。

### 8.3 L2：张量与 reset 隔离

- 分别以 1、8/32 个环境检查全部 shape、dtype 和 device。
- 只 reset 指定 env IDs，确认其他环境状态和 episode buffer 不变。
- 检查动作边界、关节索引、环境原点和 timeout 的边界条件。
- 检查结束步 reward/done 与重置后 observation 的时序。

### 8.4 L3：原 WMP 数值对齐

- 使用固定 seed、固定初始状态和固定动作序列。
- 对齐 actuator 目标、关键物理状态、全部观测字段和奖励分项。
- 对 reset 分布、终止原因、episode 长度和成功率做统计比较。
- 连续动力学无法逐帧完全一致时，提前约定绝对/相对误差和统计阈值。

### 8.5 L4：RSL-RL 短训练

建议命令：

```bash
python scripts/rsl_rl/train.py --task Isaac-Wmp-Direct-v0 --headless --num_envs 8 --max_iterations 1
```

验收：

- 生成环境配置、agent 配置和 checkpoint。
- rollout、loss、动作标准差、奖励和 episode 指标均为有限值。
- 日志目录使用正确的 WMP experiment name。

### 8.6 L5：恢复训练

建议命令：

```bash
python scripts/rsl_rl/train.py --task Isaac-Wmp-Direct-v0 --headless --num_envs 8 --resume --load_run <run> --checkpoint <checkpoint> --max_iterations 1
```

验收：

- checkpoint 被正确解析并加载。
- 迭代编号、optimizer、normalizer 和模型参数连续。
- 明确 `max_iterations` 在目标 runner 中表示总轮数还是额外轮数。

### 8.7 L6：播放与导出

建议命令：

```bash
python scripts/rsl_rl/play.py --task Isaac-Wmp-Direct-Play-v0 --headless --num_envs 1 --checkpoint <checkpoint>
```

验收：

- 策略动作有限，episode 能正常终止和 reset。
- 生成 `exported/policy.pt` 和 `exported/policy.onnx`。
- 导出模型在固定输入上的输出与在线策略在误差阈值内一致。

### 8.8 L7：规模与性能

- 从 32/64 个环境逐步扩大到目标规模，禁止直接从单环境跳到最大规模。
- 分别记录有无相机、不同 terrain/传感器配置下的吞吐与显存。
- 检查长时间运行中的 NaN、显存增长、异常 reset 和环境串扰。
- 与原 WMP 或既定 Isaac Lab 基线比较吞吐、样本效率和任务指标。

## 9. 风险与缓解

| 风险 | 表现 | 缓解措施 |
| --- | --- | --- |
| 把目录搬移误判为功能迁移 | 环境能启动但实际仍是 Cartpole | Phase 0 冻结原 WMP 契约，Phase 2 后才开始替换业务语义 |
| 注册半迁移 | `wmp.__init__` 已注册，但脚本仍导入 `wmp.tasks` | 结构收口阶段原子迁移全部调用方或保留完整兼容包 |
| 重复 Gym 注册 | 新旧模块都调用 `gym.register()` | 只允许 `wmp.__init__` 注册，兼容层只重导出 |
| wheel 漏包 | editable install 可用，普通安装缺兼容子包 | 使用包发现并检查构建产物 |
| 动作重复缩放/裁剪 | 训练不稳定、执行器饱和 | 冻结动作公式，明确环境与 wrapper 的责任边界 |
| 观测顺序或坐标系漂移 | checkpoint 可加载但策略失效 | 为每个字段建立索引、单位和坐标系表，并做离线数值对齐 |
| timeout 语义错误 | value bootstrap 和回报计算偏差 | 分离 terminated/time_out，并验证 wrapper 的 `time_outs` 处理 |
| reset 串扰 | 一个环境 reset 改变其他环境 | 针对 env IDs 做隔离检查，避免共享缓冲误写 |
| RSL-RL 版本错配 | 配置字段弃用、checkpoint 无法恢复 | 锁定实际安装版本，显式配置 `obs_groups`，分别验证恢复与播放 |
| 本地源码遮蔽安装包 | 调试行为与生产环境不同 | 默认不把根目录 `rsl_rl/` 加入 Python 路径，明确唯一依赖来源 |
| 过早随机化 | 数值差异无法定位 | 先确定性对齐，再逐项开启随机化 |
| 性能优化改变语义 | 曲线变化无法判断来源 | 行为验收后再优化，并保留基线配置 |

## 10. 兼容、checkpoint 与回滚边界

### 10.1 可安全兼容的变更

- 仅移动 Python 模块，同时保留重导出兼容层。
- 更换 canonical 注册字符串但保持 Gym ID、空间和环境语义不变。
- 更新文档、日志命名和非行为配置。

### 10.2 需要显式迁移的破坏性变更

- 修改 Gym ID 且不保留别名。
- 修改动作维度、动作顺序、范围、scale 或 actuator 语义。
- 修改观测维度、字段顺序、归一化或 actor/critic 观测映射。
- 修改网络结构、normalizer、runner 或 checkpoint 字典布局。
- 修改 reward、termination、timeout 或 episode 时序。

发生破坏性变更时必须：

1. 提升任务或配置版本。
2. 记录旧新契约差异。
3. 判断 checkpoint 是可转换、仅可推理还是完全不兼容。
4. 给出兼容窗口、转换工具或明确的重新训练要求。

### 10.3 回滚原则

- 每个 Phase 独立提交和验收，禁止把结构、行为、算法和清理混为一步。
- 新路径发布后，回滚时保留新旧导入别名，只切换实现或注册目标。
- 正式任务 ID 切换前保留旧 ID。
- 环境语义变更前保留最后一套可运行配置和参考 checkpoint。
- 兼容层、UI、CHANGELOG、README 清理和本地 `rsl_rl/` 决策都作为独立改动。

## 11. 完成定义

按本节的严格定义，当前仓库还不能宣称“WMP → Isaac Lab 迁移完全结束”。目前仍缺第 5 条中的原始 WMP 离线数值对齐，以及第 7 条中的目标规模性能验证。

只有同时满足以下条件，才能宣布 WMP → Isaac Lab 迁移完成：

1. 任务注册、Python 模块、训练、播放和文档使用一致的 canonical 接口。
2. 仓库内不存在失效的 `wmp.tasks` 导入或重复 Gym 注册。
3. 真实 WMP 资产、控制和 MDP 逻辑已经替换 Cartpole 占位实现。
4. 动作、观测、奖励、reset、terminated 和 time_out 契约有完整文档。
5. 原 WMP 的离线数值和 episode 统计在约定阈值内对齐。
6. 单环境、多环境、短训练、恢复、播放、JIT 和 ONNX 验证全部通过。
7. 目标规模下性能、显存和稳定性达到预先约定标准。
8. RSL-RL 的实际版本、依赖来源和 checkpoint 兼容范围明确。
9. 旧路径和旧任务 ID 的弃用计划已执行或有明确截止版本。
10. 所有与核心迁移无关的删除、依赖引入和文档清理均已独立审查。

## 12. 推荐实施顺序

最终推荐顺序为：

```text
冻结原 WMP 契约
  → 收口当前目录/注册迁移
  → 迁移真实资产与仿真
  → 迁移动作与控制
  → 迁移 reset/观测/done
  → 迁移奖励/命令/随机化
  → 接入并锁定 RSL-RL
  → 数值对齐
  → 短训练/恢复/播放/导出
  → 规模与性能验证
  → 切换正式任务 ID
  → 清理兼容层和无关模板内容
```

这一路径优先保证行为可解释、接口可控和问题可定位；任何阶段失败时，都能回到上一条已验证链路，而不需要同时回退环境、算法和包结构。
