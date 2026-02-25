import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json

from logic.cost_evaluator import CostEvaluator


class GridEnvironment(gym.Env):
    """基于 scenario.json 的网格环境，用于 Q-learning 训练。

    - 使用 map.grid_size 作为宽高（正方形网格）
    - start_pos / goal_pos 作为起点与终点
    - 暂时不显式建墙和障碍，后续可以根据敌人威胁区构造代价或虚拟障碍
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, scenario_path: str, render_mode: str = "ansi", renderer=None):
        super().__init__()

        with open(scenario_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        m = cfg["map"]
        self.grid_width = int(m["grid_size"])
        self.grid_height = int(m["grid_size"])
        self.start_pos = np.array(m["start_pos"], dtype=int)
        self.goal_pos = np.array(m["goal_pos"], dtype=int)

        # 目前不从场景里读取硬障碍，后续可扩展
        self.obstacles = []
        self.enemies = cfg.get("enemies", [])

        # 风险优先：提高风险项权重（比路径长度更重要）
        self.evaluator = CostEvaluator(risk_weight=320.0)

        self.agent_pos: np.ndarray | None = None
        self.render_mode = render_mode
        self.renderer = renderer

        # 8 个动作: 与 A* 一致，支持对角线
        # 0~3: 上下左右，4~7: 四个对角
        self.action_space = spaces.Discrete(8)

        # 状态用一个离散索引表示: row * width + col
        self.observation_space = spaces.Discrete(self.grid_width * self.grid_height)

        self.actions = {
            0: np.array([-1, 0], dtype=int),   # Up
            1: np.array([1, 0], dtype=int),    # Down
            2: np.array([0, -1], dtype=int),   # Left
            3: np.array([0, 1], dtype=int),    # Right
            4: np.array([-1, -1], dtype=int),  # Up-Left
            5: np.array([-1, 1], dtype=int),   # Up-Right
            6: np.array([1, -1], dtype=int),   # Down-Left
            7: np.array([1, 1], dtype=int),    # Down-Right
        }

    def _pos_to_state(self, pos: np.ndarray) -> int:
        return int(pos[0] * self.grid_width + pos[1])

    def _in_bounds(self, pos: np.ndarray) -> bool:
        return 0 <= pos[0] < self.grid_height and 0 <= pos[1] < self.grid_width

    def _max_discovery_probability(self, point: list[int]) -> float:
        max_p = 0.0
        for enemy in self.enemies:
            p = self.evaluator.physics.compute_discovery_probability(np.array(point), enemy)
            if p > max_p:
                max_p = p
        return float(max_p)

    def _combined_discovery_probability(self, point: list[int]) -> float:
        """联合被发现概率：1 - Π(1-p_i)"""
        p_not_discovered = 1.0
        point_arr = np.array(point)
        for enemy in self.enemies:
            p_i = self.evaluator.physics.compute_discovery_probability(point_arr, enemy)
            p_not_discovered *= (1.0 - p_i)
        return float(1.0 - p_not_discovered)

    def estimate_max_discovery_probability(self, pos: np.ndarray) -> float:
        point = [int(pos[0]), int(pos[1])]
        return self._max_discovery_probability(point)

    def estimate_combined_discovery_probability(self, pos: np.ndarray) -> float:
        point = [int(pos[0]), int(pos[1])]
        return self._combined_discovery_probability(point)

    def estimate_cell_cost(self, pos: np.ndarray) -> float:
        point = [int(pos[0]), int(pos[1])]
        return float(self.evaluator.evaluate_grid_cost(point, self.enemies, base_cost=1.0))

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()

        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.render(self.agent_pos)

        observation = self._pos_to_state(self.agent_pos)
        info: dict = {}
        return observation, info

    def step(self, action: int):
        move = self.actions[int(action)]
        next_pos = self.agent_pos + move

        # 默认 reward 与 A* 的代价对齐：reward = - step_cost

        # 边界检查：撞墙严惩，位置不变
        if not self._in_bounds(next_pos):
            reward = -50.0
            next_pos = self.agent_pos
        # 障碍检查（暂时为空列表，预留）
        elif any(np.array_equal(next_pos, obs) for obs in self.obstacles):
            reward = -50.0
            next_pos = self.agent_pos
        else:
            # 与 A* 相同的风险代价：(-log P_survival * risk_weight + base_cost) * 距离
            grid_point = [int(next_pos[0]), int(next_pos[1])]
            risk_cost = self.evaluator.evaluate_grid_cost(grid_point, self.enemies, base_cost=1.0)
            move_dist = float(np.linalg.norm(move))
            step_cost = risk_cost * move_dist

            # 额外风险惩罚：优先避免进入有探测概率的区域
            max_p = self._max_discovery_probability(grid_point)
            p_combined_next = self._combined_discovery_probability(grid_point)
            p_combined_curr = self._combined_discovery_probability([int(self.agent_pos[0]), int(self.agent_pos[1])])

            # 非线性惩罚：随着探测概率提升，惩罚快速增大
            extra_risk_penalty = move_dist * (600.0 * max_p + 3000.0 * (max_p ** 2))

            # 对联合概率做更强约束：即使是扇形边缘的低风险也不鼓励穿越
            extra_risk_penalty += move_dist * (2200.0 * p_combined_next + 7000.0 * (p_combined_next ** 2))

            # 进入任何风险区的“门槛惩罚”（软约束）
            if p_combined_next > 0.0:
                extra_risk_penalty += 220.0 * move_dist

            # 若下一步风险高于当前点，增加梯度惩罚，抑制朝风险区边缘靠近
            risk_increase = max(0.0, p_combined_next - p_combined_curr)
            extra_risk_penalty += 1000.0 * risk_increase * move_dist

            # 分段重罚：高风险区域显著不划算
            if max_p >= 0.3:
                extra_risk_penalty += 500.0 * move_dist
            if max_p >= 0.6:
                extra_risk_penalty += 1200.0 * move_dist
            if p_combined_next >= 0.25:
                extra_risk_penalty += 600.0 * move_dist
            if p_combined_next >= 0.5:
                extra_risk_penalty += 1500.0 * move_dist

            reward = -(step_cost + extra_risk_penalty)

            # 到达目标时额外给予正奖励，鼓励成功到达（但不压过避险优先级）
            if np.array_equal(next_pos, self.goal_pos):
                reward += 60.0

        self.agent_pos = next_pos

        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = False  # 如需步数上限，可以在外部控制或在此处加入逻辑

        observation = self._pos_to_state(self.agent_pos)

        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.render(self.agent_pos)

        info: dict = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            grid = np.full((self.grid_height, self.grid_width), "_", dtype=str)
            grid[self.start_pos[0], self.start_pos[1]] = "S"
            grid[self.goal_pos[0], self.goal_pos[1]] = "G"
            for obs in self.obstacles:
                grid[obs[0], obs[1]] = "X"
            if self.agent_pos is not None:
                grid[self.agent_pos[0], self.agent_pos[1]] = "A"
            return "\n".join(" ".join(row) for row in grid)

        elif self.render_mode == "human" and self.renderer is not None:
            self.renderer.render(self.agent_pos)

    def close(self):
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
