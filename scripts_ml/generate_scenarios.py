import argparse
import json
import os
import random


def build_enemy(enemy_id: int, grid_size: int):
    margin = max(3, grid_size // 8)
    pos = [
        random.randint(margin, grid_size - 1 - margin),
        random.randint(margin, grid_size - 1 - margin),
    ]

    theta = float(random.randint(0, 359))
    alpha = float(random.choice([60, 80, 90, 100, 120]))

    base_r = random.randint(max(3, grid_size // 10), max(5, grid_size // 6))
    r2 = base_r + random.randint(2, max(3, grid_size // 10))
    r3 = r2 + random.randint(2, max(4, grid_size // 8))

    p1 = random.choice([0.9, 0.95, 0.85])
    p2 = random.choice([0.6, 0.5, 0.4])
    p3 = random.choice([0.3, 0.25, 0.2])

    if p2 > p1:
        p2 = p1
    if p3 > p2:
        p3 = p2

    return {
        "id": enemy_id,
        "pos": pos,
        "theta": theta,
        "alpha": alpha,
        "comment": f"auto-generated enemy {enemy_id}",
        "detection_zones": [
            {"r": int(base_r), "p": float(p1)},
            {"r": int(r2), "p": float(p2)},
            {"r": int(r3), "p": float(p3)},
        ],
    }


def generate_scenarios(
    out_dir: str,
    count: int,
    grid_size: int,
    min_enemies: int,
    max_enemies: int,
    seed: int,
):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    for index in range(1, count + 1):
        enemy_count = random.randint(min_enemies, max_enemies)
        enemies = [build_enemy(i + 1, grid_size) for i in range(enemy_count)]

        scenario = {
            "map": {
                "grid_size": grid_size,
                "start_pos": [0, 0],
                "goal_pos": [grid_size - 1, grid_size - 1],
            },
            "enemies": enemies,
        }

        file_name = f"scenario_{index:03d}.json"
        file_path = os.path.join(out_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(scenario, f, ensure_ascii=False, indent=2)

    print(f"已生成 {count} 组场景到: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="批量生成DQN评估场景")
    parser.add_argument("--out-dir", default="config/generated_scenarios", help="输出目录")
    parser.add_argument("--count", type=int, default=20, help="生成数量")
    parser.add_argument("--grid-size", type=int, default=30, help="网格大小")
    parser.add_argument("--min-enemies", type=int, default=1, help="最少敌人数")
    parser.add_argument("--max-enemies", type=int, default=3, help="最多敌人数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    generate_scenarios(
        out_dir=args.out_dir,
        count=args.count,
        grid_size=args.grid_size,
        min_enemies=args.min_enemies,
        max_enemies=args.max_enemies,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
