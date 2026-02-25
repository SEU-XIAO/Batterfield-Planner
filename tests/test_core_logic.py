import sys
import os
import numpy as np

# 将项目根目录添加到系统路径，确保能导入 core 和 common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.parser import ScenarioParser
from core.physics import DetectionPhysics

def run_test():
    print("=== 开始核心逻辑集成测试 ===")
    
    # 1. 测试参数解析器
    config_path = os.path.join("config", "scenario.json")
    try:
        parser = ScenarioParser(config_path)
        config = parser.parse()
        print(f"成功加载地图，大小为: {config['map']['grid_size']}")
    except Exception as e:
        print(f"测试失败：解析配置文件出错 -> {e}")
        return

    # 2. 测试物理判定逻辑
    enemy = config['enemies'][0] 
    test_points = [
        {"name": "敌人中心点", "pos": np.array([25, 25]), "expected": "高概率"}, # 在位置 P_i 
        {"name": "扇区内近处", "pos": np.array([27, 27]), "expected": "有概率"}, # 处于 theta 朝向附近 
        {"name": "扇区外远处", "pos": np.array([0, 0]), "expected": "零概率"},    # 距离过远或角度不对 
        {"name": "朝向背面的点", "pos": np.array([20, 20]), "expected": "零概率"} # 角度判定超限 
    ]

    physics = DetectionPhysics()
    
    print("\n详细判定测试:")
    for tp in test_points:
        prob = physics.compute_discovery_probability(tp['pos'], enemy) 
        print(f"测试点 [{tp['name']}]: 坐标{tp['pos']} -> 发现概率: {prob:.2f} (预期: {tp['expected']})")

    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    run_test()