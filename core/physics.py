import numpy as np

class DetectionPhysics:
    """
    意图: 实现文档中的物理判定逻辑，包括距离计算、角度判定和区间匹配。
    """
    @staticmethod
    def compute_discovery_probability(point, enemy) -> float:
        """
        计算点 P_t 被敌人 E_i 发现的概率
        :param point: np.array([x, y])
        :param enemy: 字典，包含 pos, theta, alpha, detection_zones
        :return: 发现概率 p_discovery
        """
        # 1. 距离计算: 欧几里得距离
        enemy_pos = np.array(enemy['pos'])
        d = np.linalg.norm(point - enemy_pos)
        
        # 2. 角度判定 
        # 计算目标点相对于敌人的绝对角度 (度)
        dx, dy = point[0] - enemy_pos[0], point[1] - enemy_pos[1]
        target_angle = np.degrees(np.arctan2(dy, dx))
        
        # 计算角度偏差 angle_diff 
        angle_diff = np.abs(target_angle - enemy['theta'])
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        # 若在视野张角外，发现概率为 0 
        if angle_diff > enemy['alpha'] / 2:
            return 0.0
            
        # 3. 区间匹配 
        # 在数组 D_i 中检索 d 所属的最小距离区间 
        for zone in enemy['detection_zones']:
            if d <= zone['r']:
                return zone['p']
        
        # 若超出最大探测距离 
        return 0.0