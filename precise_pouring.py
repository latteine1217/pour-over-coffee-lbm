# precise_pouring.py
"""
精確注水系統 - 0.5cm直徑垂直水流
模擬真實手沖咖啡的注水過程
"""

import taichi as ti
import numpy as np
import config

@ti.data_oriented
class PrecisePouringSystem:
    def __init__(self):
        # 注水參數 (基於實際手沖操作)
        self.POUR_DIAMETER_CM = 0.5           # 水流直徑 0.5cm
        self.POUR_DIAMETER_GRID = self.POUR_DIAMETER_CM / config.GRID_SIZE_CM  # 轉換為網格單位
        
        # 基於真實物理的注水速度 (考慮重力加速度)
        self.POUR_HEIGHT_CM = config.POUR_HEIGHT_CM    # 注水高度 12.5cm
        self.POUR_VELOCITY = config.INLET_VELOCITY_PHYS  # 考慮重力的入水速度
        self.POUR_HEIGHT = config.NZ * 0.9    # 注水高度 (頂部90%位置)
        
        # 注水狀態
        self.pouring_active = ti.field(dtype=ti.i32, shape=())
        self.pour_center_x = ti.field(dtype=ti.f32, shape=())
        self.pour_center_y = ti.field(dtype=ti.f32, shape=())
        self.pour_flow_rate = ti.field(dtype=ti.f32, shape=())
        
        # 動態注水模式
        self.pour_pattern = ti.field(dtype=ti.i32, shape=())  # 0=定點, 1=螺旋, 2=手動
        self.pour_time = ti.field(dtype=ti.f32, shape=())
        
        # 螺旋注水參數
        self.spiral_radius = ti.field(dtype=ti.f32, shape=())
        self.spiral_speed = ti.field(dtype=ti.f32, shape=())
        self.spiral_center_x = ti.field(dtype=ti.f32, shape=())
        self.spiral_center_y = ti.field(dtype=ti.f32, shape=())
        
        print(f"精確注水系統初始化 - 水流直徑: {self.POUR_DIAMETER_CM}cm ({self.POUR_DIAMETER_GRID:.2f}格)")
    
    def start_pouring(self, center_x=None, center_y=None, flow_rate=1.0, pattern='center'):
        """開始注水"""
        if center_x is None:
            center_x = config.NX // 2
        if center_y is None:
            center_y = config.NY // 2
        
        self.pour_center_x[None] = center_x
        self.pour_center_y[None] = center_y
        self.pour_flow_rate[None] = flow_rate
        self.pouring_active[None] = 1
        self.pour_time[None] = 0.0
        
        # 設置注水模式
        if pattern == 'center':
            self.pour_pattern[None] = 0
        elif pattern == 'spiral':
            self.pour_pattern[None] = 1
            self.spiral_center_x[None] = center_x
            self.spiral_center_y[None] = center_y
            self.spiral_radius[None] = 5.0      # 初始螺旋半徑
            self.spiral_speed[None] = 1.0       # 螺旋速度
        
        print(f"開始注水: 位置({center_x:.1f}, {center_y:.1f}), 模式: {pattern}")
    
    def stop_pouring(self):
        """停止注水"""
        self.pouring_active[None] = 0
        print("停止注水")
    
    @ti.kernel
    def apply_pouring(self, lbm_u: ti.template(), lbm_rho: ti.template(), 
                     multiphase_phi: ti.template(), dt: ti.f32):
        """施加精確注水到LBM場"""
        if self.pouring_active[None] == 1:
            # 更新注水時間
            self.pour_time[None] += dt
            
            # 計算當前注水位置
            pour_x, pour_y = self._get_current_pour_position()
            
            # 注水影響區域
            pour_radius = self.POUR_DIAMETER_GRID / 2.0
            pour_z = self.POUR_HEIGHT
            
            # 在注水區域施加水流
            for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
                # 計算到注水中心的距離
                dx = i - pour_x
                dy = j - pour_y
                distance_to_pour = ti.sqrt(dx*dx + dy*dy)
                
                # 只在水流圓柱體內施加影響
                if distance_to_pour <= pour_radius and k >= pour_z:
                    # 高斯分佈的水流強度 (中心最強)
                    intensity = ti.exp(-0.5 * (distance_to_pour / pour_radius)**2)
                    
                    # 設置水相 (phi = 1 表示純水)
                    multiphase_phi[i, j, k] = 1.0
                    
                    # 設置密度為水密度
                    lbm_rho[i, j, k] = config.RHO_WATER
                    
                    # 設置垂直向下的水流速度
                    vertical_velocity = -self.POUR_VELOCITY * intensity * self.pour_flow_rate[None]
                    
                    # 添加輕微的徑向分散 (更真實)
                    radial_vx = 0.0
                    radial_vy = 0.0
                    if distance_to_pour > 0:
                        radial_factor = 0.1 * intensity  # 徑向分散強度
                        radial_vx = radial_factor * dx / distance_to_pour
                        radial_vy = radial_factor * dy / distance_to_pour
                    
                    lbm_u[i, j, k] = ti.Vector([radial_vx, radial_vy, vertical_velocity])
    
    @ti.func
    def _get_current_pour_position(self):
        """獲取當前注水位置 (支持不同注水模式)"""
        x = self.pour_center_x[None]
        y = self.pour_center_y[None]
        
        if self.pour_pattern[None] == 1:  # 螺旋注水
            t = self.pour_time[None] * self.spiral_speed[None]
            current_radius = self.spiral_radius[None] * (1.0 + 0.1 * t)  # 漸大螺旋
            
            x = self.spiral_center_x[None] + current_radius * ti.cos(t)
            y = self.spiral_center_y[None] + current_radius * ti.sin(t)
            
            # 確保在邊界內
            x = ti.max(self.POUR_DIAMETER_GRID, ti.min(config.NX - self.POUR_DIAMETER_GRID, x))
            y = ti.max(self.POUR_DIAMETER_GRID, ti.min(config.NY - self.POUR_DIAMETER_GRID, y))
        
        return x, y
    
    @ti.kernel
    def create_water_impact_force(self, particle_system: ti.template(), 
                                 max_force: ti.f32, dt: ti.f32):
        """計算水流對咖啡粉的衝擊力"""
        if self.pouring_active[None] == 1:
            pour_x, pour_y = self._get_current_pour_position()
            pour_radius = self.POUR_DIAMETER_GRID / 2.0
            impact_strength = 10.0 * self.pour_flow_rate[None]  # 衝擊強度
            
            # 對每個咖啡粒子檢查是否受到水流衝擊
            for p in range(particle_system.particle_count[None]):
                if particle_system.active[p] == 1:
                    pos = particle_system.position[p]
                    
                    # 檢查粒子是否在水流影響範圍內
                    dx = pos.x - pour_x
                    dy = pos.y - pour_y
                    distance_to_pour = ti.sqrt(dx*dx + dy*dy)
                    
                    if distance_to_pour <= pour_radius * 2.0 and pos.z > config.NZ * 0.3:
                        # 計算衝擊力
                        impact_intensity = ti.exp(-0.5 * (distance_to_pour / pour_radius)**2)
                        
                        # 垂直向下的衝擊力
                        impact_force = ti.Vector([0.0, 0.0, -impact_strength * impact_intensity])
                        
                        # 徑向分散力 (模擬水流分散效應)
                        if distance_to_pour > 0.1:
                            radial_force_magnitude = impact_strength * 0.3 * impact_intensity
                            radial_force_x = radial_force_magnitude * dx / distance_to_pour
                            radial_force_y = radial_force_magnitude * dy / distance_to_pour
                            impact_force += ti.Vector([radial_force_x, radial_force_y, 0.0])
                        
                        # 應用力到粒子 (限制最大力)
                        force_magnitude = impact_force.norm()
                        if force_magnitude > max_force:
                            impact_force = impact_force * (max_force / force_magnitude)
                        
                        particle_system.force[p] += impact_force
    
    @ti.kernel
    def visualize_pour_stream(self, vis_field: ti.template()):
        """在可視化場中標記水流位置"""
        if self.pouring_active[None] == 1:
            pour_x, pour_y = self._get_current_pour_position()
            pour_radius = self.POUR_DIAMETER_GRID / 2.0
            
            # 標記整個水流柱
            for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
                dx = i - pour_x
                dy = j - pour_y
                distance = ti.sqrt(dx*dx + dy*dy)
                
                if distance <= pour_radius and k >= self.POUR_HEIGHT:
                    vis_field[i, j, k] = 2.0  # 特殊標記值表示水流
    
    def get_pouring_info(self):
        """獲取當前注水信息"""
        if self.pouring_active[None]:
            return {
                'active': True,
                'center_x': self.pour_center_x[None],
                'center_y': self.pour_center_y[None],
                'flow_rate': self.pour_flow_rate[None],
                'pattern': ['center', 'spiral', 'manual'][self.pour_pattern[None]],
                'time': self.pour_time[None],
                'diameter_cm': self.POUR_DIAMETER_CM
            }
        else:
            return {'active': False}
    
    def adjust_flow_rate(self, new_rate):
        """調整水流速率"""
        self.pour_flow_rate[None] = max(0.1, min(3.0, new_rate))
        print(f"調整水流速率: {self.pour_flow_rate[None]:.2f}")
    
    def switch_to_spiral_pour(self, radius=10.0, speed=1.0):
        """切換到螺旋注水模式"""
        if self.pouring_active[None]:
            self.pour_pattern[None] = 1
            self.spiral_radius[None] = radius
            self.spiral_speed[None] = speed
            print(f"切換到螺旋注水: 半徑={radius:.1f}, 速度={speed:.1f}")
    
    def move_pour_center(self, new_x, new_y):
        """移動注水中心位置"""
        self.pour_center_x[None] = max(5, min(config.NX-5, new_x))
        self.pour_center_y[None] = max(5, min(config.NY-5, new_y))
        print(f"移動注水位置: ({self.pour_center_x[None]:.1f}, {self.pour_center_y[None]:.1f})")