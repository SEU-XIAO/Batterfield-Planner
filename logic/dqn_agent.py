from collections import deque
import heapq
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_head(feat)
        adv = self.adv_head(feat)
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        target_update_freq: int = 2000,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.online_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer()

    def select_action(self, state_vec: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states_t).gather(1, next_actions)
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q_target

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


def state_to_pos(state: int, grid_width: int) -> np.ndarray:
    return np.array([state // grid_width, state % grid_width], dtype=np.int32)


def build_state_features(env, state: int) -> np.ndarray:
    pos = state_to_pos(state, env.grid_width).astype(np.float32)
    goal = env.goal_pos.astype(np.float32)

    grid_scale = float(max(1, env.grid_width - 1))
    pos_norm = pos / grid_scale
    goal_delta = (goal - pos) / grid_scale

    curr_combined = env.estimate_combined_discovery_probability(pos)
    curr_max = env.estimate_max_discovery_probability(pos)

    neighbor_risks = []
    for action in range(env.action_space.n):
        move = env.actions[action]
        next_pos = pos.astype(np.int32) + move
        if not env._in_bounds(next_pos):
            next_pos = pos.astype(np.int32)
        neighbor_risks.append(env.estimate_combined_discovery_probability(next_pos))

    feat = np.concatenate(
        [
            pos_norm,
            goal_delta,
            np.array([curr_combined, curr_max], dtype=np.float32),
            np.array(neighbor_risks, dtype=np.float32),
        ]
    ).astype(np.float32)

    return feat


def _global_risk_first_action(
    env,
    state: int,
    visit_counts: dict[int, int] | None = None,
) -> int | None:
    start_pos_arr = state_to_pos(state, env.grid_width)
    start = (int(start_pos_arr[0]), int(start_pos_arr[1]))
    goal = (int(env.goal_pos[0]), int(env.goal_pos[1]))

    if start == goal:
        return 0

    best_cost: dict[tuple[int, int], tuple[float, int]] = {start: (0.0, 0)}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    pq: list[tuple[float, int, int, int]] = [(0.0, 0, start[0], start[1])]

    while pq:
        risk_sum, steps, r, c = heapq.heappop(pq)
        node = (r, c)
        curr_best = best_cost.get(node)
        if curr_best is None:
            continue
        if (risk_sum, steps) != curr_best:
            continue

        if node == goal:
            break

        for _, move in env.actions.items():
            nr = r + int(move[0])
            nc = c + int(move[1])
            next_pos = np.array([nr, nc], dtype=np.int32)
            if not env._in_bounds(next_pos):
                continue

            base_risk = float(env.estimate_combined_discovery_probability(next_pos))
            if base_risk < 1e-6:
                base_risk = 0.0

            revisit_penalty = 0.0
            if visit_counts is not None:
                next_state = int(nr * env.grid_width + nc)
                v = int(visit_counts.get(next_state, 0))
                revisit_penalty = min(0.03, 0.003 * max(0, v - 1))

            new_cost = (risk_sum + base_risk + revisit_penalty, steps + 1)
            next_node = (nr, nc)
            old = best_cost.get(next_node)
            if old is None or new_cost < old:
                best_cost[next_node] = new_cost
                parent[next_node] = node
                heapq.heappush(pq, (new_cost[0], new_cost[1], nr, nc))

    if goal not in best_cost:
        return None

    step = goal
    while step in parent and parent[step] != start:
        step = parent[step]

    if step not in parent:
        return None

    dr = step[0] - start[0]
    dc = step[1] - start[1]
    for action, move in env.actions.items():
        if int(move[0]) == dr and int(move[1]) == dc:
            return int(action)

    return None


def risk_aware_action(
    agent: DQNAgent,
    env,
    state: int,
    visit_counts: dict[int, int] | None = None,
    last_state: int | None = None,
    no_progress_steps: int = 0,
    enable_global_fallback: bool = True,
) -> int:
    state_vec = build_state_features(env, state)
    state_t = torch.tensor(state_vec, dtype=torch.float32, device=agent.device).unsqueeze(0)

    with torch.no_grad():
        q_values = agent.online_net(state_t).squeeze(0).cpu().numpy()

    # 防止出现 NaN/Inf 导致动作选择失效
    q_values = np.nan_to_num(q_values, nan=-1e6, posinf=1e6, neginf=-1e6)

    curr_pos = state_to_pos(state, env.grid_width)
    goal = env.goal_pos

    candidates = []
    zero_risk_eps = 1e-6
    for action in range(env.action_space.n):
        move = env.actions[action]
        next_pos = curr_pos + move
        if not env._in_bounds(next_pos):
            continue

        risk_combined_raw = env.estimate_combined_discovery_probability(next_pos)
        risk_combined = 0.0 if risk_combined_raw < zero_risk_eps else float(risk_combined_raw)
        dist = float(np.linalg.norm(next_pos - goal))
        q_val = float(q_values[action])

        if not np.isfinite(q_val):
            q_val = -1e6

        next_state = int(next_pos[0] * env.grid_width + next_pos[1])
        visit_penalty = float(visit_counts.get(next_state, 0)) if visit_counts is not None else 0.0

        # 词典序目标：
        # 1) 最小 risk_combined
        # 2) 在同风险内最小 dist（更短路径）
        # 3) 在同风险同距离下优先更少重复访问（防打转）
        # 4) 最后用 q_val 打破平局
        candidates.append((int(action), risk_combined, dist, visit_penalty, q_val, next_state))

    if not candidates:
        return 0

    curr_visits = int(visit_counts.get(int(state), 0)) if visit_counts is not None else 0
    has_zero_risk = any(item[1] <= 1e-9 for item in candidates)
    hard_stuck = (curr_visits >= 3) or (no_progress_steps >= 8)
    if enable_global_fallback and (hard_stuck or (not has_zero_risk)):
        global_action = _global_risk_first_action(env, state, visit_counts=visit_counts)
        if global_action is not None:
            return int(global_action)

    min_risk = min(item[1] for item in candidates)

    # 默认严格最小风险；若长时间无进展，适度放宽风险带帮助脱离局部最优
    risk_eps = 1e-9
    relax = 0.0
    if no_progress_steps >= 25:
        relax = min(0.20, 0.01 * ((no_progress_steps - 25) // 5 + 1))
    risk_band = min_risk + risk_eps + relax

    low_risk = [item for item in candidates if item[1] <= risk_band]

    # 避免立即来回折返（若存在其它同级候选）
    if last_state is not None:
        non_backtrack = [item for item in low_risk if item[5] != last_state]
        if non_backtrack:
            low_risk = non_backtrack
        elif enable_global_fallback and hard_stuck:
            global_action = _global_risk_first_action(env, state, visit_counts=visit_counts)
            if global_action is not None:
                return int(global_action)

    min_dist = min(item[2] for item in low_risk)
    dist_eps = 1e-6
    shortest = [item for item in low_risk if item[2] <= min_dist + dist_eps]

    min_visit = min(item[3] for item in shortest)
    least_visited = [item for item in shortest if item[3] <= min_visit + 1e-9]

    best = max(least_visited, key=lambda x: x[4])
    return int(best[0])
