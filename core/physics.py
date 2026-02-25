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
        # 统一角度约定（数学常用）：0° 向右，90° 向上
        # 网格坐标是 [row, col]，而屏幕坐标 y 轴向下，故需要对 row 取反映射到 y。
        dr = point[0] - enemy_pos[0]   # row 差
        dc = point[1] - enemy_pos[1]   # col 差
        dx = dc
        dy = -dr
        target_angle = np.degrees(np.arctan2(dy, dx))
        target_angle = (target_angle + 360.0) % 360.0
        enemy_theta = float(enemy['theta']) % 360.0
        
        # 计算角度偏差 angle_diff 
        angle_diff = np.abs(target_angle - enemy_theta)
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