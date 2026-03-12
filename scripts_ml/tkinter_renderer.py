import tkinter as tk


class TkinterRenderer:
    def __init__(self, width, height, start, goal, obstacles, enemies=None, cell_size=30):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.enemies = enemies or []

        self.root = tk.Tk()
        self.root.title("Battlefield Planner - DQN")
        
        self.canvas_width = self.width * self.cell_size
        self.canvas_height = self.height * self.cell_size
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        self._draw_grid()
        self._draw_enemy_fov()
        self._draw_fixed_elements()
        
        self.agent_rect = None
        self.root.update()

    def _draw_grid(self):
        for i in range(self.width + 1):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.canvas_height, fill='lightgrey')
        for i in range(self.height + 1):
            y = i * self.cell_size
            self.canvas.create_line(0, y, self.canvas_width, y, fill='lightgrey')

    def _draw_cell(self, pos, color):
        x1 = pos[1] * self.cell_size
        y1 = pos[0] * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='grey')

    def _draw_fixed_elements(self):
        # 起点、终点和硬障碍
        self._draw_cell(self.start, "green")
        self._draw_cell(self.goal, "blue")
        for obs in self.obstacles:
            self._draw_cell(obs, "black")

    def _draw_enemy_fov(self):
        """绘制敌人的扇形探测区，不同概率用不同颜色。"""
        for enemy in self.enemies:
            ex, ey = enemy["pos"]  # [row, col]
            cx = ey * self.cell_size + self.cell_size / 2
            cy = ex * self.cell_size + self.cell_size / 2

            theta = enemy.get("theta", 0.0)
            alpha = enemy.get("alpha", 0.0)
            danger_shape = str(enemy.get("danger_shape", "sector")).lower()
            zones = sorted(enemy.get("detection_zones", []), key=lambda z: float(z["r"]), reverse=True)

            for zone in zones:
                r_cells = zone["r"]  # 以格子为单位的半径
                p = zone["p"]

                # 颜色: 概率越大越偏红
                if p >= 0.8:
                    color = "#ff6666"  # 高风险
                elif p >= 0.5:
                    color = "#ffa500"  # 中风险
                else:
                    color = "#ffff66"  # 低风险

                r_pix = r_cells * self.cell_size
                x1 = cx - r_pix
                y1 = cy - r_pix
                x2 = cx + r_pix
                y2 = cy + r_pix

                if danger_shape == "circle":
                    self.canvas.create_oval(
                        x1,
                        y1,
                        x2,
                        y2,
                        outline="",
                        fill=color,
                    )
                else:
                    # 与 physics 统一：0°向右，90°向上
                    # Tkinter create_arc 也是 0°向右、正角逆时针，因此可直接使用 theta。
                    start_angle = theta - alpha / 2.0
                    extent = alpha

                    self.canvas.create_arc(
                        x1,
                        y1,
                        x2,
                        y2,
                        start=start_angle,
                        extent=extent,
                        style=tk.PIESLICE,
                        outline="",
                        fill=color,
                    )

    def render(self, agent_pos):
        # 窗口可能被用户手动关闭，防止抛出 TclError
        try:
            if not self.root.winfo_exists():
                return
        except Exception:
            return

        try:
            if self.agent_rect:
                self.canvas.delete(self.agent_rect)

            x1 = agent_pos[1] * self.cell_size + self.cell_size * 0.1
            y1 = agent_pos[0] * self.cell_size + self.cell_size * 0.1
            x2 = x1 + self.cell_size * 0.8
            y2 = y1 + self.cell_size * 0.8
            self.agent_rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill='red')

            # Update the display and add a small delay
            self.root.update()
            self.root.after(50) # 50ms delay
        except tk.TclError:
            return

    def close(self):
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except Exception:
            pass
