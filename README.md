# Battlefield-Planner（简版说明）

## 项目在做什么

这是一个网格地图上的风险规避路径规划项目：
- 从起点走到终点；
- 敌人有扇形视野，格点会有被发现概率；
- 目标是优先降低风险，再追求较短路径。

当前方案不是纯 DQN，而是：
- Dueling Double DQN 负责学习动作价值；
- 风险感知动作层负责把“风险优先”落地；
- 卡住时用全局最小风险突围做兜底。

---

## 核心思路（训练与决策）

1. 状态特征（14维）
- 位置归一化 2 维；
- 目标相对位移 2 维；
- 当前联合风险与最大单敌风险 2 维；
- 8 个邻居的联合风险 8 维。

2. 模型
- 网络：Dueling Q Network（Value + Advantage）；
- 学习：Double DQN 目标计算 + 经验回放 + 目标网络。

3. 动作选择
- 训练时：epsilon 探索，否则走 risk_aware_action；
- 推理时：risk_aware_action。

4. 防打转策略
- 词典序风险优先筛选（风险→距离→访问次数→Q值）；
- 立即回退抑制（last_state）；
- 无进展检测（no_progress_steps）；
- 必要时调用全局最小风险动作（_global_risk_first_action）。

---

## 模块职责

- entry_dqn.py
  - 训练入口：多场景采样、epsilon 调度、回放更新、验证、保存最佳模型。
  - 定义训练塑形奖励 shaped_reward。

- logic/dqn_agent.py
  - DQNAgent、DuelingQNetwork、ReplayBuffer。
  - build_state_features 负责特征构造。
  - risk_aware_action 负责风险优先决策与防打转兜底。

- core/environment.py
  - Gym 环境：状态转移、动作空间、环境基础奖励。
  - 提供联合风险与最大风险查询接口。

- core/physics.py
  - 单敌发现概率计算（距离、角度、分段探测区）。
  - 踩敌人坐标概率直接视为 1.0。
  - 支持 `danger_shape`：`sector`（默认扇形）或 `circle`（圆形）。

- logic/cost_evaluator.py
  - 多敌风险融合：p_survival = Π(1-p_i)，并映射为风险代价。

- evaluate_dqn_batch.py
  - 批量评估：成功率、平均步数、路径最大风险、路径平均风险。

- visualize_dqn_path.py
  - 可视化推理轨迹，观察绕行、突围与风险水平。

---

## 奖励函数设计（详细）

本项目是“双层奖励”设计：
- 环境层奖励（core/environment.py）：强调物理风险真实性，约束行为边界；
- 训练层奖励（entry_dqn.py::shaped_reward）：强调学习稳定性，引导策略收敛。

### A. 环境层奖励（core/environment.py）

环境 step 中对每一步给出基础回报：

1) 非法动作惩罚
- 越界：`reward = -50`
- 撞障碍：`reward = -50`

2) 合法动作基础代价
- `step_cost = risk_cost * move_dist`
- `risk_cost = risk_weight * (-log(p_survival)) + base_cost`
- 其中：`p_survival = Π(1 - p_i)`，`p_i` 是每个敌人的发现概率。

3) 额外风险惩罚（核心）
- 最大单敌风险非线性惩罚：
  - `move_dist * (600*max_p + 3000*max_p^2)`
- 联合风险非线性惩罚：
  - `move_dist * (2200*p_combined_next + 7000*p_combined_next^2)`
- 进入风险区门槛惩罚：
  - 若 `p_combined_next > 0`，额外 `220*move_dist`
- 风险上升惩罚：
  - `1000 * max(0, p_next - p_curr) * move_dist`
- 分段重罚：
  - `max_p >= 0.3` 加 `500*move_dist`
  - `max_p >= 0.6` 加 `1200*move_dist`
  - `p_combined_next >= 0.25` 加 `600*move_dist`
  - `p_combined_next >= 0.5` 加 `1500*move_dist`

4) 终点奖励
- 到达目标额外 `+60`

5) 汇总
- 合法移动时：`reward = -(step_cost + extra_risk_penalty)`，到终点再加奖励。

这层奖励的作用是“风险硬约束”：让高风险动作在环境层面天然不划算。

### B. 训练层塑形奖励（entry_dqn.py::shaped_reward）

训练时写入 replay 的不是环境原始回报，而是塑形回报：

1) 风险优先主项
- 若下一步几乎零风险：`+0.8`
- 否则：`-(8 + 22*next_combined + 10*next_max)`

2) 风险上升惩罚
- `risk_increase = max(0, next_combined - curr_combined)`
- 惩罚：`-18 * risk_increase`

3) 距离进度项（次优目标）
- `progress = curr_dist - next_dist`
- 奖励：`+0.6 * progress`

4) 防停滞项
- 原地不动：`-2.5`
- 重复访问：`-min(2.0, 0.25*(visit_count-1))`

5) 终点奖励
- 到达目标：`+35`

6) 奖励裁剪
- 最终 `clip` 到 `[-25, 25]`

这层奖励的作用是“学习友好”：
- 让风险、进度、反打转信号更平衡；
- 控制极端值，降低训练震荡。

### C. 两层奖励为什么要并存

- 只用环境奖励：语义真实，但可能梯度大、学习慢；
- 只用塑形奖励：学习快，但可能偏离真实风险语义；
- 两层结合：既保留风险物理意义，又保证训练稳定。

### D. 参数调节建议（实战）

- 现象：能到终点但经常擦边冒险
  - 增大环境层联合风险惩罚系数（2200/7000）或门槛重罚。

- 现象：太保守，宁可打转也不突围
  - 适当减小环境层分段重罚，或增大训练层进度系数（0.6）。

- 现象：局部来回抖动
  - 增大训练层重复访问惩罚上限；
  - 配合启用全局兜底器进行对照评估。

- 现象：训练后期不稳定
  - 收紧奖励裁剪区间，或放缓 epsilon 衰减。

一句话总结：
奖励设计的优先级是“先不被发现，再尽快到达”，并通过反打转项保证策略可执行。

---

## 快速使用

- 训练：python entry_dqn.py
- 批量验证：python evaluate_dqn_batch.py --scenario-dir config/val_scenarios --runs 5 --max-steps 1200
- 可视化：python visualize_dqn_path.py

---

## 圆形危险区配置（新）

每个敌人可选字段：
- `danger_shape`: `sector` 或 `circle`
  - 不写时默认 `sector`
  - 设为 `circle` 时，会忽略 `theta/alpha`，按同心圆风险区计算

示例（单圆）：

```json
{
  "id": 1,
  "pos": [14, 14],
  "danger_shape": "circle",
  "detection_zones": [
    {"r": 9, "p": 0.45}
  ]
}
```

示例（多圆）：

```json
{
  "id": 1,
  "pos": [10, 10],
  "danger_shape": "circle",
  "detection_zones": [
    {"r": 3, "p": 0.90},
    {"r": 6, "p": 0.60},
    {"r": 10, "p": 0.30}
  ]
}
```

已提供完整示例文件：
- `config/circle_scenarios/circle_one_zone.json`
- `config/circle_scenarios/circle_multi_zone.json`

如果要改成纯 DQN（不带规则后处理），可把 risk_aware_action 替换为 select_action(epsilon=0) 的纯贪心推理。
