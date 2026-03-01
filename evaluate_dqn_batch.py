import argparse
import csv
import glob
import os
from typing import Dict, List

import numpy as np

from core.environment import GridEnvironment
from logic.dqn_agent import DQNAgent, risk_aware_action


def evaluate_one_scenario(model_path: str, scenario_path: str, runs: int, max_steps: int) -> Dict[str, float]:
    env = GridEnvironment(scenario_path=scenario_path, render_mode="ansi")
    agent = DQNAgent(state_dim=14, action_dim=env.action_space.n)
    agent.load(model_path)

    success_count = 0
    step_list: List[int] = []
    max_risk_list: List[float] = []
    avg_risk_list: List[float] = []

    for _ in range(runs):
        state, _ = env.reset()
        done = False
        steps = 0
        risks = []
        visit_counts = {int(state): 1}
        last_state = None
        no_progress_steps = 0

        while not done and steps < max_steps:
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
            prev_state = state
            state, _, terminated, truncated, _ = env.step(action)

            next_pos = np.array([state // env.grid_width, state % env.grid_width], dtype=int)
            next_dist = float(np.linalg.norm(next_pos - env.goal_pos))
            if next_dist < curr_dist - 1e-6:
                no_progress_steps = 0
            else:
                no_progress_steps += 1

            last_state = prev_state
            visit_counts[int(state)] = visit_counts.get(int(state), 0) + 1

            done = terminated or truncated
            steps += 1

            pos = np.array([state // env.grid_width, state % env.grid_width], dtype=int)
            risk = env.estimate_combined_discovery_probability(pos)
            risks.append(float(risk))

        if done:
            success_count += 1

        step_list.append(steps)
        max_risk_list.append(max(risks) if risks else 0.0)
        avg_risk_list.append(float(np.mean(risks)) if risks else 0.0)

    env.close()

    return {
        "success_rate": success_count / runs,
        "avg_steps": float(np.mean(step_list)),
        "avg_max_risk": float(np.mean(max_risk_list)),
        "avg_path_risk": float(np.mean(avg_risk_list)),
    }


def main():
    parser = argparse.ArgumentParser(description="批量评估DQN模型在多场景上的表现")
    parser.add_argument("--model", default="dqn_model.pth", help="模型路径")
    parser.add_argument("--scenario-dir", default="config/generated_scenarios", help="场景目录")
    parser.add_argument("--runs", type=int, default=5, help="每个场景评估次数")
    parser.add_argument("--max-steps", type=int, default=1200, help="单次轨迹最大步数")
    parser.add_argument("--out-csv", default="dqn_batch_eval.csv", help="输出CSV路径")
    args = parser.parse_args()

    scenario_files = sorted(glob.glob(os.path.join(args.scenario_dir, "*.json")))
    if not scenario_files:
        print(f"未在目录中找到场景文件: {args.scenario_dir}")
        return

    rows = []
    for path in scenario_files:
        metrics = evaluate_one_scenario(
            model_path=args.model,
            scenario_path=path,
            runs=args.runs,
            max_steps=args.max_steps,
        )
        row = {
            "scenario": os.path.basename(path),
            "success_rate": metrics["success_rate"],
            "avg_steps": metrics["avg_steps"],
            "avg_max_risk": metrics["avg_max_risk"],
            "avg_path_risk": metrics["avg_path_risk"],
        }
        rows.append(row)
        print(
            f"{row['scenario']}: "
            f"success={row['success_rate']*100:.1f}% | "
            f"steps={row['avg_steps']:.1f} | "
            f"max_risk={row['avg_max_risk']:.4f} | "
            f"path_risk={row['avg_path_risk']:.4f}"
        )

    overall_success = float(np.mean([r["success_rate"] for r in rows]))
    overall_steps = float(np.mean([r["avg_steps"] for r in rows]))
    overall_max_risk = float(np.mean([r["avg_max_risk"] for r in rows]))
    overall_path_risk = float(np.mean([r["avg_path_risk"] for r in rows]))

    print("\n=== Overall ===")
    print(f"mean_success_rate: {overall_success*100:.1f}%")
    print(f"mean_avg_steps: {overall_steps:.1f}")
    print(f"mean_avg_max_risk: {overall_max_risk:.4f}")
    print(f"mean_avg_path_risk: {overall_path_risk:.4f}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "success_rate", "avg_steps", "avg_max_risk", "avg_path_risk"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n评估结果已写入: {args.out_csv}")


if __name__ == "__main__":
    main()
