import numpy as np
import os
import glob

from core.environment import GridEnvironment
from logic.dqn_agent import DQNAgent, build_state_features, risk_aware_action, state_to_pos


SCENARIO_PATH = "config/scenario.json"
DQN_MODEL_PATH = "dqn_model.pth"
TRAIN_SCENARIOS_DIR = "config/train_scenarios"
VAL_SCENARIOS_DIR = "config/val_scenarios"


def shaped_reward(
    env,
    curr_state: int,
    next_state: int,
    env_reward: float,
    terminated: bool,
    next_visit_count: int = 1,
) -> float:
    curr_pos = state_to_pos(curr_state, env.grid_width)
    next_pos = state_to_pos(next_state, env.grid_width)

    curr_combined = env.estimate_combined_discovery_probability(curr_pos)
    next_combined = env.estimate_combined_discovery_probability(next_pos)
    next_max = env.estimate_max_discovery_probability(next_pos)

    curr_dist = float(np.linalg.norm(curr_pos - env.goal_pos))
    next_dist = float(np.linalg.norm(next_pos - env.goal_pos))

    reward = 0.0

    # 1) 风险优先：先最小化被发现概率
    # 无风险动作给轻微激励；有风险动作给显著惩罚（软约束）
    if next_combined <= 1e-8:
        reward += 0.8
    else:
        reward -= (8.0 + 22.0 * next_combined + 10.0 * next_max)

    # 风险上升额外惩罚：防止向更危险区域深入
    risk_increase = max(0.0, next_combined - curr_combined)
    reward -= 18.0 * risk_increase

    # 2) 距离次之：只在风险同级别下鼓励更短路径
    progress = curr_dist - next_dist
    reward += 0.6 * progress

    # 原地不动惩罚（包含撞边界后停留）
    if np.array_equal(curr_pos, next_pos):
        reward -= 2.5

    # 重复访问惩罚：防止在边界和低风险边缘来回打转
    if next_visit_count > 1:
        reward -= min(2.0, 0.25 * (next_visit_count - 1))

    # 成功到达目标的主奖励（不能压过长期风险累计）
    if terminated:
        reward += 35.0

    # 裁剪奖励，提升训练稳定性
    reward = float(np.clip(reward, -25.0, 25.0))
    return reward


def evaluate_policy_on_files(
    agent: DQNAgent,
    scenario_files: list[str],
    eval_runs_per_scenario: int = 2,
    max_steps: int = 1200,
) -> tuple[float, float]:
    success_count = 0
    total_steps = 0
    total_runs = 0

    for scenario_path in scenario_files:
        env = GridEnvironment(scenario_path=scenario_path, render_mode="ansi")
        for _ in range(eval_runs_per_scenario):
            state, _ = env.reset()
            done = False
            steps = 0
            visit_counts = {int(state): 1}
            last_state = None
            no_progress_steps = 0

            while not done and steps < max_steps:
                curr_pos = state_to_pos(state, env.grid_width)
                curr_dist = float(np.linalg.norm(curr_pos - env.goal_pos))

                action = risk_aware_action(
                    agent,
                    env,
                    state,
                    visit_counts=visit_counts,
                    last_state=last_state,
                    no_progress_steps=no_progress_steps,
                )
                next_state, _, terminated, truncated, _ = env.step(action)

                next_pos = state_to_pos(next_state, env.grid_width)
                next_dist = float(np.linalg.norm(next_pos - env.goal_pos))
                if next_dist < curr_dist - 1e-6:
                    no_progress_steps = 0
                else:
                    no_progress_steps += 1

                last_state = state
                state = next_state
                visit_counts[int(state)] = visit_counts.get(int(state), 0) + 1
                done = terminated or truncated
                steps += 1

            if done:
                success_count += 1
            total_steps += steps
            total_runs += 1

        env.close()

    if total_runs == 0:
        return 0.0, float(max_steps)

    success_rate = success_count / total_runs
    avg_steps = total_steps / total_runs
    return success_rate, avg_steps


def evaluate_policy_by_scenario(
    agent: DQNAgent,
    scenario_files: list[str],
    eval_runs_per_scenario: int = 3,
    max_steps: int = 1200,
) -> dict[str, tuple[float, float]]:
    result = {}
    for scenario_path in scenario_files:
        success_rate, avg_steps = evaluate_policy_on_files(
            agent,
            [scenario_path],
            eval_runs_per_scenario=eval_runs_per_scenario,
            max_steps=max_steps,
        )
        result[os.path.basename(scenario_path)] = (success_rate, avg_steps)
    return result


def pick_training_scenario(
    train_scenario_files: list[str],
    train_stats: dict[str, dict[str, int]],
    hard_scenarios: dict[str, float] | None = None,
) -> str:
    weights = []
    for path in train_scenario_files:
        s = train_stats[path]
        attempts = s["attempts"]
        successes = s["successes"]
        if attempts < 5:
            weight = 2.0
        else:
            success_rate = successes / max(1, attempts)
            weight = 1.0 + 5.0 * (1.0 - success_rate)

        if hard_scenarios is not None and path in hard_scenarios:
            weight *= hard_scenarios[path]
        weights.append(weight)

    weights = np.array(weights, dtype=np.float64)
    probs = weights / np.sum(weights)
    idx = int(np.random.choice(len(train_scenario_files), p=probs))
    return train_scenario_files[idx]


def main():
    np.random.seed(42)

    train_scenario_files = sorted(glob.glob(os.path.join(TRAIN_SCENARIOS_DIR, "*.json")))
    if not train_scenario_files:
        train_scenario_files = [SCENARIO_PATH]

    val_scenario_files = sorted(glob.glob(os.path.join(VAL_SCENARIOS_DIR, "*.json")))
    if not val_scenario_files:
        val_scenario_files = [SCENARIO_PATH]

    # 用首个训练场景创建环境以读取动作/状态维度
    env = GridEnvironment(scenario_path=train_scenario_files[0], render_mode="ansi")

    state_dim = 14
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        target_update_freq=2000,
    )

    # 若已有模型，则在其基础上继续微调，避免丢掉已学能力
    if os.path.exists(DQN_MODEL_PATH):
        try:
            agent.load(DQN_MODEL_PATH)
            print(f"Loaded existing model from {DQN_MODEL_PATH}, continue fine-tuning.")
        except Exception as e:
            print(f"Load existing model failed, start from scratch. reason: {e}")

    # 默认先跑一个可接受时长的训练配置
    max_episodes = 800
    max_steps_per_episode = 1000

    epsilon_start = 1.0
    epsilon_final = 0.02
    epsilon_decay_steps = 25_000

    global_step = 0
    best_success_rate = 0.0
    best_avg_steps = float("inf")
    no_improve_evals = 0
    update_every = 4
    warmup_steps = 2000
    recent_successes = 0
    train_stats = {path: {"attempts": 0, "successes": 0} for path in train_scenario_files}
    hard_scenario_weights = {path: 1.0 for path in train_scenario_files}

    for episode in range(1, max_episodes + 1):
        # 每个回合随机采样一个训练场景
        chosen_scenario = pick_training_scenario(train_scenario_files, train_stats, hard_scenario_weights)
        if env is not None:
            env.close()
        env = GridEnvironment(scenario_path=str(chosen_scenario), render_mode="ansi")

        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        updates_in_episode = 0
        visit_counts = {int(state): 1}
        last_state = None
        no_progress_steps = 0

        for step_in_episode in range(1, max_steps_per_episode + 1):
            epsilon = max(
                epsilon_final,
                epsilon_start - (epsilon_start - epsilon_final) * (global_step / epsilon_decay_steps),
            )

            state_vec = build_state_features(env, state)
            if np.random.rand() < epsilon:
                action = np.random.randint(action_dim)
            else:
                action = risk_aware_action(
                    agent,
                    env,
                    state,
                    visit_counts=visit_counts,
                    last_state=last_state,
                    no_progress_steps=no_progress_steps,
                )

            prev_state = int(state)
            curr_pos = state_to_pos(prev_state, env.grid_width)
            curr_dist = float(np.linalg.norm(curr_pos - env.goal_pos))
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_pos = state_to_pos(next_state, env.grid_width)
            next_dist = float(np.linalg.norm(next_pos - env.goal_pos))
            if next_dist < curr_dist - 1e-6:
                no_progress_steps = 0
            else:
                no_progress_steps += 1

            next_visit_count = visit_counts.get(int(next_state), 0) + 1

            reward = shaped_reward(
                env,
                state,
                next_state,
                env_reward,
                terminated,
                next_visit_count=next_visit_count,
            )
            next_state_vec = build_state_features(env, next_state)

            agent.replay.push(state_vec, action, reward, next_state_vec, done)
            if global_step >= warmup_steps and global_step % update_every == 0:
                loss = agent.update()
                if loss is not None:
                    updates_in_episode += 1

            state = next_state
            last_state = prev_state
            visit_counts[int(state)] = next_visit_count
            episode_reward += reward
            global_step += 1

            if done:
                break

        if done:
            recent_successes += 1
            train_stats[str(chosen_scenario)]["successes"] += 1
        train_stats[str(chosen_scenario)]["attempts"] += 1

        if episode % 10 == 0:
            print(
                f"Episode {episode}/{max_episodes} | "
                f"TrainScenario {os.path.basename(str(chosen_scenario))} | "
                f"Steps {step_in_episode} | "
                f"Reward {episode_reward:.2f} | "
                f"Epsilon {epsilon:.3f} | "
                f"Updates {updates_in_episode} | "
                f"RecentSuccess(10ep) {recent_successes}/10"
            )
            recent_successes = 0

        if episode % 20 == 0:
            success_rate, avg_steps = evaluate_policy_on_files(
                agent,
                val_scenario_files,
                eval_runs_per_scenario=2,
                max_steps=max_steps_per_episode,
            )
            detail = evaluate_policy_by_scenario(
                agent,
                val_scenario_files,
                eval_runs_per_scenario=2,
                max_steps=max_steps_per_episode,
            )
            print(
                f"[Eval] Episode {episode}/{max_episodes} | "
                f"Eval Success {success_rate * 100:.1f}% | "
                f"Eval AvgSteps {avg_steps:.1f}"
            )
            detail_text = " | ".join(
                [f"{name}:{sr*100:.0f}%/{st:.0f}" for name, (sr, st) in detail.items()]
            )
            print(f"[EvalDetail] {detail_text}")

            # 将验证失败场景加入训练难例权重（同名映射不存在时忽略）
            for val_name, (sr, _) in detail.items():
                if sr < 1.0:
                    # 尝试用名称关键字匹配训练集（不强依赖同文件名）
                    for train_path in train_scenario_files:
                        train_name = os.path.basename(train_path)
                        if (
                            "start" in val_name and "start" in train_name
                        ) or (
                            "goal" in val_name and "goal" in train_name
                        ) or (
                            "crossfire" in val_name and ("overlap" in train_name or "dense" in train_name)
                        ):
                            hard_scenario_weights[train_path] = min(4.0, hard_scenario_weights[train_path] + 0.3)

            improved = (
                success_rate > best_success_rate
                or (np.isclose(success_rate, best_success_rate) and avg_steps < best_avg_steps)
            )

            if improved:
                best_success_rate = success_rate
                best_avg_steps = avg_steps
                agent.save(DQN_MODEL_PATH)
                print(f"New best model saved to {DQN_MODEL_PATH}")
                no_improve_evals = 0
            else:
                no_improve_evals += 1

            # 若长时间无提升，则回滚到最佳模型再继续训练，抑制后期退化
            if no_improve_evals >= 8 and os.path.exists(DQN_MODEL_PATH):
                try:
                    agent.load(DQN_MODEL_PATH)
                    no_improve_evals = 0
                    print("Eval长期无提升，已回滚到最佳模型继续训练。")
                except Exception as e:
                    print(f"回滚最佳模型失败: {e}")

            if success_rate >= 1.0 and avg_steps < 500:
                print("Early stop: policy already stable and efficient.")
                break

    if best_success_rate == 0.0:
        agent.save(DQN_MODEL_PATH)
        print(f"No checkpoint improved during eval; saved current model to {DQN_MODEL_PATH}")
    else:
        print(f"Training finished. Best success rate: {best_success_rate * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
