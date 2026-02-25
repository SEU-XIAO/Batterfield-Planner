import json
import os

class ScenarioParser:
    """
    意图: 解析战场态势配置文件，并验证物理约束条件。
    """
    def __init__(self, config_path: str):
        """
        :param config_path: scenario.json 文件的绝对或相对路径
        """
        self.config_path = config_path

    def parse(self) -> dict:
        """
        读取并解析 JSON 文件，执行数学约束验证。
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"未找到配置文件: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 格式错误: {e}")

        # 执行文档要求的物理约束校验
        self._validate(data)
        return data

    def _validate(self, data: dict):
        """
        内部逻辑：验证距离 r 递增和概率 p 递减 
        """
        if 'enemies' not in data:
            return

        for enemy in data['enemies']:
            zones = enemy.get('detection_zones', [])
            if not zones:
                continue
            
            # 文档约束: r1 < r2 < ... < rn 且 p1 >= p2 >= ... >= pn 
            for i in range(len(zones) - 1):
                curr_r, next_r = zones[i]['r'], zones[i+1]['r']
                curr_p, next_p = zones[i]['p'], zones[i+1]['p']
                
                if curr_r >= next_r:
                    raise ValueError(f"敌人 {enemy['id']} 探测距离约束失败: r 必须递增。")
                if curr_p < next_p:
                    raise ValueError(f"敌人 {enemy['id']} 探测概率约束失败: p 必须递减。")

        print(">>> [Parser] 配置文件解析成功，物理约束校验通过。")