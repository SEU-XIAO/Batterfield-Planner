import numpy as np

class DetectionPhysics:
    """
    意图: 实现文档中的物理判定逻辑，包括距离计算、角度判定和区间匹配。
    """
    @staticmethod
    def compute_discovery_probability(*args) -> float:
        """
        计算在给定坐标处被特定敌人发现的概率。

        兼容两种调用：
        1) compute_discovery_probability(point, enemy)
        2) compute_discovery_probability(x, y, ex, ey, etheta, ealpha, eradius)
        """
        detection_zones = None
        danger_shape = "sector"

        if len(args) == 2:
            point, enemy = args
            x = float(point[0])
            y = float(point[1])
            if 'pos' in enemy and len(enemy['pos']) >= 2:
                ex = float(enemy['pos'][0])
                ey = float(enemy['pos'][1])
            else:
                ex = float(enemy['x'])
                ey = float(enemy['y'])
            etheta = float(enemy.get('theta', 0.0))
            ealpha = float(enemy.get('alpha', 360.0))
            danger_shape = str(enemy.get('danger_shape', 'sector')).lower()
            detection_zones = enemy.get('detection_zones', [])
            if 'radius' in enemy:
                eradius = float(enemy['radius'])
            elif detection_zones:
                eradius = float(max(float(zone['r']) for zone in detection_zones))
            else:
                eradius = 0.0
        elif len(args) == 7:
            x, y, ex, ey, etheta, ealpha, eradius = [float(v) for v in args]
        else:
            raise TypeError("compute_discovery_probability expects 2 or 7 arguments")

        if int(x) == int(ex) and int(y) == int(ey):
            return 1.0

        dr = x - ex
        dc = y - ey
        d = float(np.sqrt(dr ** 2 + dc ** 2))

        if d > eradius:
            return 0.0

        if danger_shape != 'circle':
            # 网格坐标(row, col) 转换到数学坐标(x, y)
            # x 对应 col 差，y 对应 -row 差（因为屏幕 row 向下增大）
            dx = dc
            dy = -dr
            target_angle = float(np.degrees(np.arctan2(dy, dx)))
            target_angle = (target_angle + 360.0) % 360.0
            enemy_theta = etheta % 360.0

            angle_diff = abs(target_angle - enemy_theta)
            if angle_diff > 180.0:
                angle_diff = 360.0 - angle_diff

            if angle_diff > (ealpha / 2.0):
                return 0.0

        if detection_zones:
            for zone in detection_zones:
                if d <= float(zone['r']):
                    return float(zone['p'])

        if eradius <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - d / eradius))