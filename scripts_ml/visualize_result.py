import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator
import numpy as np
import os
import sys

# 确保可以导入项目根目录下的模块
# 假设当前脚本位于 scripts_ml 文件夹下，将其父目录加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目中定义的模块
try:
    from entry_astar import AStarPlanner 
    from common.parser import ScenarioParser
except ImportError:
    print("错误：无法导入核心模块。请确保 entry_astar.py 在根目录，且存在 common/parser.py")
    sys.exit(1)

def run_visualization(config_path='config/scenario.json'):
    # 1. 解析配置文件
    if not os.path.exists(config_path):
        # 尝试相对于根目录的路径
        config_path = os.path.join(project_root, 'config', 'scenario.json')
    
    parser = ScenarioParser(config_path)
    data = parser.parse()
    grid_size = data['map']['grid_size']
    
    # 2. 初始化 A* 规划器并执行路径搜索
    planner = AStarPlanner(data)
    path, total_cost = planner.search()
    c_map = planner.c_map
    
    # 3. 创建画布并配置高对比度视觉效果
    fig, ax = plt.subplots(figsize=(12, 11))
    
    # 核心改进：引入 LogNorm 提高区分度
    # vmin=1.0 对应绝对安全区（base_cost），颜色最淡
    # 对数刻度能让较低的风险值在视觉上也呈现出明显的颜色加深
    norm = colors.LogNorm(vmin=1.0, vmax=max(c_map.max(), 1.1))
    
    # 使用 'YlOrRd' (黄-橙-红) 渐变色，热力图渲染
    im = ax.imshow(c_map.T, origin='lower', cmap='YlOrRd', 
                   norm=norm, interpolation='nearest', alpha=0.9)
    
    # --- 增强栅格显示逻辑 ---
    # 设置次刻度位置（每个格子的边缘）
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    
    # 绘制细网格线，明确每一个小格子
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 主刻度每 5 个格显示一个数字标签
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    # ----------------------------

    # 4. 绘制 A* 最优路径
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        # 绘制蓝色路径线
        ax.plot(path_x, path_y, color='#0077FF', linewidth=3, label='A* Safe Path', zorder=3)
        # 绘制路径节点
        ax.scatter(path_x, path_y, color='#0077FF', s=20, edgecolors='white', linewidth=0.5, zorder=4)
    else:
        print("未找到有效路径！")
    
    # 5. 标记起点、终点和敌人信息
    start = data['map']['start_pos']
    goal = data['map']['goal_pos']
    ax.scatter(start[0], start[1], c='lime', s=150, label='Start', marker='P', edgecolors='black', zorder=5)
    ax.scatter(goal[0], goal[1], c='gold', s=200, label='Goal', marker='*', edgecolors='black', zorder=5)
    
    for enemy in data['enemies']:
        ex, ey = enemy['pos']
        ax.scatter(ex, ey, c='black', s=100, marker='x', zorder=6)
        ax.annotate(f"E{enemy['id']}\n({enemy['theta']}°)", (ex, ey), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 6. 图表细节美化
    ax.set_title(f"Battlefield A* Planning (Grid: {grid_size}x{grid_size})\nHigh-Safety Priority Strategy", fontsize=14)
    ax.set_xlabel("X Grid Index")
    ax.set_ylabel("Y Grid Index")
    ax.set_aspect('equal') # 保证栅格为正方形
    
    # 配置右侧颜色条，显示对数代价范围
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cost (Log Scale: Risk Weight * -log(P_surv) + 1)', rotation=270, labelpad=15)
    
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_visualization()