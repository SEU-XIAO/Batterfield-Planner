import json
import time
import numpy as np

from core.environment import GridEnvironment
from logic.dqn_agent import DQNAgent, risk_aware_action
from scripts_ml.tkinter_renderer import TkinterRenderer


SCENARIO_PATH = "config/scenario.json"
DQN_MODEL_PATH = "dqn_model.pth"


def main():
    with open(SCENARIO_PATH, "r", encoding="utf-8") as f:
        scenario = json.load(f)

    m = scenario["map"]

    renderer = TkinterRenderer(
        width=m["grid_size"],
        height=m["grid_size"],
        start=m["start_pos"],
        goal=m["goal_pos"],
        obstacles=[],
        enemies=scenario.get("enemies", []),
        cell_size=20,
    )

    env = GridEnvironment(
        scenario_path=SCENARIO_PATH,
        render_mode="human",
        renderer=renderer,
    )

    agent = DQNAgent(state_dim=14, action_dim=env.action_space.n)

    try:
        agent.load(DQN_MODEL_PATH)
    except FileNotFoundError:
        print(f"未找到模型文件 {DQN_MODEL_PATH}，请先运行 entry_dqn.py 训练。")
        env.close()
        return

    max_steps = 2000
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    max_path_risk = 0.0
    unchanged_steps = 0
    visit_counts = {int(state): 1}
    last_state = None
    no_progress_steps = 0

    print("开始执行 DQN 轨迹，可关闭窗口结束。")

    while not done and max_steps > 0:
        try:
            if not renderer.root.winfo_exists():
                print("检测到窗口已关闭，结束可视化。")
                break
        except Exception:
            break

        prev_state = state
        curr_pos = np.array([state // env.grid_width, state % env.grid_width], dtype=int)
        curr_dist = float(np.linalg.norm(curr_pos - env.goal_pos))

        action = risk_aware_action(
            agent,
            env,
            state,
            visit_counts=visit_counts,
            last_state=last_state,
            no_progress_steps=no_progress_steps,
        )
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        last_state = prev_state
        visit_counts[int(state)] = visit_counts.get(int(state), 0) + 1

        pos = np.array([state // env.grid_width, state % env.grid_width], dtype=int)
        next_dist = float(np.linalg.norm(pos - env.goal_pos))
        if next_dist < curr_dist - 1e-6:
            no_progress_steps = 0
        else:
            no_progress_steps += 1

        curr_risk = env.estimate_combined_discovery_probability(pos)
        if curr_risk > max_path_risk:
            max_path_risk = curr_risk

        if state == prev_state:
            unchanged_steps += 1
        else:
            unchanged_steps = 0

        if unchanged_steps > 120:
            print("检测到长时间原地不动，提前结束本次可视化。")
            break

        max_steps -= 1
        time.sleep(0.03)

    if done:
        print(f"DQN轨迹执行结束：已到达目标，总回报 {total_reward:.2f}")
    else:
        print(f"DQN轨迹执行结束：未到达目标（达到步数上限），总回报 {total_reward:.2f}")

    print(f"轨迹最大联合风险概率: {max_path_risk:.4f}")

    print("可视化结束，关闭窗口即可退出。")
    try:
        renderer.root.mainloop()
    except Exception:
        pass
    env.close()


if __name__ == "__main__":
    main()
