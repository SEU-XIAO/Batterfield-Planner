import heapq
import numpy as np
import math
from common.parser import ScenarioParser
from logic.cost_evaluator import CostEvaluator

class AStarPlanner:
    def __init__(self, scenario_data, risk_weight=100.0):
        self.grid_size = scenario_data['map']['grid_size'] 
        self.enemies = scenario_data['enemies'] 
        self.start_pos = tuple(scenario_data['map']['start_pos'])
        self.goal_pos = tuple(scenario_data['map']['goal_pos'])
        
        # 传入高权重系数
        self.evaluator = CostEvaluator(risk_weight=risk_weight)
        self.c_map = self._precompute_cost_map() 

    def _precompute_cost_map(self):
        """预计算风险权值地图"""
        c_map = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # 这里的 cost 包含 -log(P_survival) + base_cost
                c_map[x, y] = self.evaluator.evaluate_grid_cost([x, y], self.enemies)
        return c_map

    def _heuristic(self, pos):
        """启发项 h(n): 使用欧几里得距离"""
        return math.sqrt((pos[0] - self.goal_pos[0])**2 + (pos[1] - self.goal_pos[1])**2)

    def search(self):
        # 优先级队列存储 (f_score, g_score, pos)
        # f_score = g_score + h_score
        start_h = self._heuristic(self.start_pos)
        pq = [(start_h, 0.0, self.start_pos)]
        
        g_score = {self.start_pos: 0.0}
        parent = {}
        
        while pq:
            f, current_g, curr_pos = heapq.heappop(pq)
            
            if curr_pos == self.goal_pos:
                return self._reconstruct_path(parent), current_g
            
            # 8方向移动规则 [cite: 4]
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = curr_pos[0] + dx, curr_pos[1] + dy
                
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # 关键修改：区分直线移动和对角线移动的物理距离
                    move_dist = math.sqrt(dx**2 + dy**2) 
                    
                    # 这里的代价 = (风险代价 + 基础代价) * 移动距离
                    # 这样可以确保走对角线比走“直角弯”更省代价
                    step_cost = self.c_map[nx, ny] * move_dist
                    tentative_g = current_g + step_cost
                    
                    if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                        continue

                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        parent[(nx, ny)] = curr_pos
                        g_score[(nx, ny)] = tentative_g
                        f_total = tentative_g + self._heuristic((nx, ny))
                        heapq.heappush(pq, (f_total, tentative_g, (nx, ny)))
        
        return None, float('inf')

    def _reconstruct_path(self, parent):
        path = []
        curr = self.goal_pos
        while curr in parent:
            path.append(curr)
            curr = parent[curr]
        path.append(self.start_pos)
        return path[::-1]