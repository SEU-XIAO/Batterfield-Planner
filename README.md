# Battlefield-Planner 项目说明（DQN 风险规避路径规划）

本文档面向“第一次接触本项目”的读者，目标是让你快速理解：
- 这个项目在解决什么问题；
- 强化学习的完整数据流和决策流；
- 每个模块分别负责什么；
- 奖励函数是怎么设计的、为什么这么设计；
- 怎么训练、验证和可视化。

---

## 1. 项目要解决的问题

这是一个二维网格路径规划问题：
- 智能体从 `start_pos` 出发，到达 `goal_pos`；
- 地图上有若干敌人，每个敌人有一个扇形探测区域（方向、角度、分段半径与概率）；
- 智能体目标是：**尽量低风险地到达终点**。

本项目不是“最短路径优先”，而是“**风险优先，路径次优**”。

---

## 2. 一句话理解整体方法

项目采用的是 **Dueling Double DQN + 风险感知动作后处理（hybrid）**：
1. 神经网络学习每个动作的 Q 值；
2. 动作执行前做一层风险优先的策略筛选（`risk_aware_action`）；
3. 如果检测到“卡住/打转”，触发全局最小风险突围动作；
4. 训练时使用经验回放 + 目标网络稳定学习。

> 换句话说：不是纯 DQN，而是“DQN 学习 + 规则兜底”的工程化方案。

---

## 3. 目录与模块作用

### 3.1 训练与运行入口

- `entry_dqn.py`
  - 主训练脚本。
  - 负责：场景采样、epsilon 探索、回放更新、周期验证、保存最佳模型。
  - 训练使用的实际奖励来自 `shaped_reward(...)`（不是直接用环境原始 reward）。

- `evaluate_dqn_batch.py`
  - 批量评估脚本。
  - 对一个目录内的多个场景做多次 rollout，输出：成功率、平均步数、路径风险指标。

- `visualize_dqn_path.py`
  - 可视化单条轨迹。
  - 使用训练好的模型逐步执行动作，显示敌人扇区和智能体轨迹表现。

### 3.2 环境与物理层

- `core/physics.py` (`DetectionPhysics`)
  - 输入某格点与敌人参数，计算“被该敌人发现的概率”。
  - 关键点：
    - 兼容敌人坐标格式：`pos` 或 `x/y`；
    - 若踩到敌人所在格子，概率直接返回 `1.0`；
    - 角度采用统一约定：`0°` 向右，`90°` 向上；
    - 概率由 `detection_zones` 分段给出。

- `logic/cost_evaluator.py` (`CostEvaluator`)
  - 将多个敌人的发现概率合成为生存概率，转成风险代价：
    - `p_survival = Π(1 - p_i)`
    - `risk_cost = risk_weight * (-log(p_survival)) + base_cost`
  - 这个代价也被环境 step 内部用于基础风险惩罚。

- `core/environment.py` (`GridEnvironment`)
  - Gym 风格环境，负责状态转移与基础 reward。
  - 动作空间：8 方向（上下左右+四对角）。
  - 提供风险查询接口：
    - `estimate_max_discovery_probability(...)`
    - `estimate_combined_discovery_probability(...)`
  - `step(...)` 内含较强风险惩罚（联合风险、最大风险、风险增量、分段重罚）。

### 3.3 模型与决策层

- `logic/dqn_agent.py`
  - `DuelingQNetwork`：Dueling 架构（Value + Advantage）。
  - `DQNAgent`：
    - `select_action`（epsilon-greedy），
    - `update`（Double DQN 目标计算 + Huber Loss）。
  - `ReplayBuffer`：经验回放。
  - `build_state_features`：构造 14 维状态特征。
  - `risk_aware_action`：核心决策后处理（风险优先 + 防打转 + 全局突围）。

---

## 4. 状态、动作、转移是怎么定义的

## 4.1 状态表示

环境内部观测是离散索引：
- `state = row * grid_width + col`

训练输入网络时，会转换成 14 维特征（`build_state_features`）：
1. 当前位置归一化 `(row, col)` → 2 维；
2. 目标相对位移归一化 `(goal - pos)` → 2 维；
3. 当前点联合风险、最大单敌风险 → 2 维；
4. 8 个邻居点的联合风险 → 8 维。

合计：`2 + 2 + 2 + 8 = 14`。

## 4.2 动作空间

8 个动作：
- 0: 上，1: 下，2: 左，3: 右
- 4: 左上，5: 右上，6: 左下，7: 右下

## 4.3 转移逻辑

`env.step(action)`：
- 越界：位置不变并给负奖励；
- 合法移动：计算风险相关代价，更新位置；
- 到达终点：`terminated=True`。

---

## 5. 奖励函数设计（重点）

本项目存在两层奖励：

## 5.1 环境基础奖励（`core/environment.py`）

环境内部 reward 强化“高风险强惩罚”：
- 基础步代价来自风险代价 `step_cost`；
- 叠加：最大风险、联合风险、风险增量、进入风险区门槛惩罚、分段重罚；
- 到达终点加正奖励。

这层 reward 主要保证环境反馈方向正确。

## 5.2 训练塑形奖励（`entry_dqn.py -> shaped_reward`）

训练时真正写入回放的是 `shaped_reward(...)`，它更稳定、可控：

1) **风险优先项**
- 若下一步联合风险几乎 0：`+0.8`
- 否则惩罚：`-(8.0 + 22.0*next_combined + 10.0*next_max)`

2) **风险上升惩罚**
- `risk_increase = max(0, next_combined - curr_combined)`
- 额外惩罚：`-18.0 * risk_increase`

3) **进度奖励（次优目标）**
- `progress = curr_dist - next_dist`
- 奖励：`+0.6 * progress`

4) **防停滞/防打转项**
- 原地不动惩罚：`-2.5`
- 重复访问惩罚：`-min(2.0, 0.25*(visit_count-1))`

5) **终点奖励**
- 到达终点：`+35.0`

6) **奖励裁剪**
- 最终 `clip` 到 `[-25, 25]`，提升训练稳定性。

### 5.3 为什么要两层奖励？

- 环境层 reward 更接近“物理风险代价”；
- 训练层 reward 更偏“学习友好”，控制梯度尺度与探索方向；
- 这样能兼顾语义正确性与优化稳定性。

---

## 6. 如何解决“局部打转”

核心在 `risk_aware_action`：

1. **风险优先词典序筛选**
   - 先选低风险动作，再看离终点距离，再看访问次数，最后才看 Q 值。

2. **反复横跳抑制**
   - 用 `last_state` 避免立即回退到上一步状态。

3. **卡住检测**
   - 使用 `visit_counts` + `no_progress_steps` 判断是否进入循环。

4. **全局突围（兜底）**
   - 触发 `_global_risk_first_action`：
   - 以“累计风险最小、步数最短”为目标做全局搜索，给出下一步突围动作。

这套机制让策略在复杂包围/窄口场景不容易无限循环。

---

## 7. 训练流程（`entry_dqn.py`）

1. 读取 `config/train_scenarios/*.json` 与 `config/val_scenarios/*.json`；
2. 每个 episode 按权重采样训练场景（难例权重更高）；
3. 每步执行：
   - 组装状态特征；
   - epsilon 探索或 `risk_aware_action` 选动作；
   - 环境转移；
   - 计算 `shaped_reward`；
   - 写入回放池；
   - 按步频更新网络；
4. 每 20 个 episode 做验证集评估，打印 `EvalDetail`；
5. 若指标提升则保存 `dqn_model.pth`；
6. 若长期无提升，可回滚最佳模型继续训练。

---

## 8. 评估指标如何看

`evaluate_dqn_batch.py` 输出：
- `success_rate`：达到终点的比例；
- `avg_steps`：平均步数（越低通常越好）；
- `avg_max_risk`：路径上遇到的最大联合风险均值；
- `avg_path_risk`：整条路径联合风险平均值。

常见判读：
- 高成功率 + 低 `avg_path_risk`：策略稳健；
- 高成功率但 `avg_max_risk` 高：能到终点但会冒险；
- 低成功率且 `avg_path_risk` 低：可能过于保守、陷入打转。

---

## 9. 场景配置格式（JSON）

示例结构：

```json
{
  "map": {
    "grid_size": 30,
    "start_pos": [0, 0],
    "goal_pos": [29, 29]
  },
  "enemies": [
    {
      "id": 1,
      "pos": [14, 10],
      "theta": 70.0,
      "alpha": 120.0,
      "detection_zones": [
        {"r": 4, "p": 0.9},
        {"r": 6, "p": 0.6},
        {"r": 8, "p": 0.2}
      ]
    }
  ]
}
```

字段说明：
- `theta`：敌人朝向角（0°右、90°上）；
- `alpha`：扇形张角；
- `detection_zones`：分段半径与概率，按距离匹配。

---

## 10. 如何运行

## 10.1 环境准备

建议 Python 3.10+，安装依赖：

```bash
pip install numpy torch gymnasium
```

`tkinter` 通常随 Python 自带（用于可视化）。

## 10.2 训练

```bash
python entry_dqn.py
```

## 10.3 批量验证

```bash
python evaluate_dqn_batch.py --scenario-dir config/val_scenarios --runs 5 --max-steps 1200
```

## 10.4 可视化

```bash
python visualize_dqn_path.py
```

---

## 11. 这个项目当前范式总结

- 算法内核：Dueling Double DQN；
- 决策范式：风险优先的动作后处理（hybrid，而非纯端到端）；
- 工程策略：多场景训练 + 验证驱动 + 难例加权 + 最佳模型回滚；
- 目标取向：优先规避风险，其次提高到达效率。

如果你后续想改成“纯 DQN（不带规则后处理）”，可以把 `risk_aware_action` 替换为 `agent.select_action(..., epsilon=0)` 的纯贪心推理，并把“防打转能力”完全交给奖励函数与训练数据来学习。
