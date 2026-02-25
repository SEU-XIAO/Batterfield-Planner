import gym
import numpy as np
import torch
from gym import spaces
from stable_baselines3 import PPO

class BattleFieldEnv(gym.Env):
    def __init__(self):
        super(BattleFieldEnv, self).__init__()
        # 1. 栅格地图与移动规则 [cite: 3, 4]
        self.grid_size = 50
        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([49, 49])
        self.current_pos = self.start_pos.copy()
        
        # 动作空间：8个邻域移动 
        self.action_space = spaces.Discrete(8)
        self.action_map = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # 状态空间：当前坐标 (x, y)
        self.observation_space = spaces.Box(low=0, high=49, shape=(2,), dtype=np.float32)

        # 2. 敌人参数设定 [cite: 6, 7, 10]
        self.enemies = [
            {
                "pos": np.array([25, 25]),
                "theta": 45,        # 朝向角度 [cite: 8]
                "alpha": 90,       # 视野张角 [cite: 9]
                "D": [(10, 0.8), (20, 0.4), (30, 0.1)] # 多段探测数组 (r, p) 
            }
        ]

    def _get_survival_prob(self, pos):
        """计算单点的生存概率 [cite: 11, 16]"""
        p_not_discovered_all = 1.0
        
        for e in self.enemies:
            # 距离计算 [cite: 12]
            d = np.linalg.norm(pos - e["pos"])
            
            # 角度判定 [cite: 13]
            target_angle = np.degrees(np.arctan2(pos[1]-e["pos"][1], pos[0]-e["pos"][0]))
            angle_diff = np.abs(target_angle - e["theta"])
            if angle_diff > 180: angle_diff = 360 - angle_diff
            
            p_discovery = 0
            if angle_diff <= e["alpha"] / 2:
                # 区间匹配 [cite: 14]
                for r, p in e["D"]:
                    if d <= r:
                        p_discovery = p
                        break
            
            p_not_discovered_all *= (1 - p_discovery) # 生存概率融合 
            
        return max(p_not_discovered_all, 1e-6) # 避免log(0)

    def step(self, action):
        # 执行移动 
        move = self.action_map[action]
        new_pos = self.current_pos + np.array(move)
        
        # 边界检查
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)
        self.current_pos = new_pos
        
        # 3. 奖励设计 [cite: 31]
        p_survival = self._get_survival_prob(self.current_pos)
        cost = -np.log(p_survival) # 负对数变换 [cite: 23]
        
        reward = -cost # 每步奖励为负代价 [cite: 31]
        done = False
        
        # 判定终点
        if np.array_equal(self.current_pos, self.goal_pos):
            reward += 100 # 目的地巨大正奖励 [cite: 31]
            done = True
            
        return self.current_pos.astype(np.float32), reward, done, {}

    def reset(self):
        self.current_pos = self.start_pos.copy()
        return self.current_pos.astype(np.float32)

# --- 训练部分 ---
env = BattleFieldEnv()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=100000)

# --- 测试结果 ---
obs = env.reset()
path = [obs.copy()]
for _ in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    path.append(obs.copy())
    if done: break

print(f"规划路径长度: {len(path)}")
print(f"最终位置: {obs}")