import numpy as np

from core.environment import GridEnvironment
from logic.dqn_agent import DQNAgent, build_state_features, risk_aware_action, state_to_pos


SCENARIO_PATH = "config/scenario.json"
DQN_MODEL_PATH = "dqn_model.pth"


def shaped_reward(env, curr_state: int, next_state: int, env_reward: float, terminated: bool) -> float:
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

    # 成功到达目标的主奖励（不能压过长期风险累计）
    if terminated:
        reward += 35.0

    # 裁剪奖励，提升训练稳定性
    reward = float(np.clip(reward, -25.0, 25.0))
    return reward


def evaluate_policy(env, agent: DQNAgent, eval_runs: int = 5, max_steps: int = 1200) -> tuple[float, float]:
    success_count = 0
    total_steps = 0

    for _ in range(eval_runs):
        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = risk_aware_action(agent, env, state)
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
            steps += 1

        if done:
            success_count += 1
        total_steps += steps

    success_rate = success_count / eval_runs
    avg_steps = total_steps / eval_runs
    return success_rate, avg_steps


def main():
    np.random.seed(42)

    env = GridEnvironment(scenario_path=SCENARIO_PATH, render_mode="ansi")

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

    # 默认先跑一个可接受时长的训练配置
    max_episodes = 800
    max_steps_per_episode = 1000

    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay_steps = 200_000

    global_step = 0
    best_success_rate = 0.0
    update_every = 4
    warmup_steps = 2000
    recent_successes = 0

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        updates_in_episode = 0

        for step_in_episode in range(1, max_steps_per_episode + 1):
            epsilon = max(
                epsilon_final,
                epsilon_start - (epsilon_start - epsilon_final) * (global_step / epsilon_decay_steps),
            )

            state_vec = build_state_features(env, state)
            action = agent.select_action(state_vec, epsilon)

            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward = shaped_reward(env, state, next_state, env_reward, terminated)
            next_state_vec = build_state_features(env, next_state)

            agent.replay.push(state_vec, action, reward, next_state_vec, done)
            if global_step >= warmup_steps and global_step % update_every == 0:
                loss = agent.update()
                if loss is not None:
                    updates_in_episode += 1

            state = next_state
            episode_reward += reward
            global_step += 1

            if done:
                break

        if done:
            recent_successes += 1

        if episode % 10 == 0:
            print(
                f"Episode {episode}/{max_episodes} | "
                f"Steps {step_in_episode} | "
                f"Reward {episode_reward:.2f} | "
                f"Epsilon {epsilon:.3f} | "
                f"Updates {updates_in_episode} | "
                f"RecentSuccess(10ep) {recent_successes}/10"
            )
            recent_successes = 0

        if episode % 20 == 0:
            success_rate, avg_steps = evaluate_policy(env, agent, eval_runs=3, max_steps=max_steps_per_episode)
            print(
                f"[Eval] Episode {episode}/{max_episodes} | "
                f"Eval Success {success_rate * 100:.1f}% | "
                f"Eval AvgSteps {avg_steps:.1f}"
            )

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                agent.save(DQN_MODEL_PATH)
                print(f"New best model saved to {DQN_MODEL_PATH}")

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
