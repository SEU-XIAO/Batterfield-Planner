import numpy as np
from core.physics import DetectionPhysics

class CostEvaluator:
    def __init__(self, risk_weight=50.0): # 显著提高默认风险权重
        self.physics = DetectionPhysics()
        self.risk_weight = risk_weight

    def evaluate_grid_cost(self, point, enemies, base_cost=1.0) -> float:
        p_not_discovered_all = 1.0
        for enemy in enemies:
            p_discovery = self.physics.compute_discovery_probability(np.array(point), enemy)
            p_not_discovered_all *= (1 - p_discovery)
        
        p_survival = max(p_not_discovered_all, 1e-10)
        # 放大风险项的影响 [cite: 23, 24]
        risk_cost = self.risk_weight * (-np.log(p_survival)) 
        
        return float(risk_cost + base_cost)