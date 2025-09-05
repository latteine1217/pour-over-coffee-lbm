# precise_pouring.py
"""
精確注水系統 - 0.5cm直徑垂直水流
模擬真實手沖咖啡的注水過程
統一相場邏輯重構版本
"""

import taichi as ti
import numpy as np
import config as config

@ti.data_oriented
class PrecisePouringSystem:
    def __init__(self):
        # 注水參數 (基於實際手沖操作)
        self.POUR_DIAMETER_CM = 0.5           # 水流直徑 0.5cm
        self.POUR_DIAMETER_GRID = self.POUR_DIAMETER_CM / config.GRID_SIZE_CM  # 轉換為網格單位
        
        # 基於真實物理的注水速度 (考慮重力加速度)
        self.POUR_HEIGHT_CM = config.POUR_HEIGHT_CM    # 注水高度 12.5cm
        # 使用格子單位速度（已於config中依流量與截面計算並限幅，確保LBM穩定）
        self.POUR_VELOCITY = config.INLET_VELOCITY  # lu/ts
        # 注水高度：改為靠近V60杯口上方少量間隙，縮短到達時間
        v60_bottom_z = 5.0
        cup_height_lu = int(config.CUP_HEIGHT / config.SCALE_LENGTH)
        v60_top_z = int(v60_bottom_z + cup_height_lu)
        clearance = 2  # 於杯口上方2格
        # 確保不越界且不低於安全下限
        self.POUR_HEIGHT = max(8, min(int(v60_top_z + clearance), config.NZ - 6))
        
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
        
        print(f"開始注水: 位置({center_x:.1f}, {center_y:.1f}), 高度: {self.POUR_HEIGHT}, 模式: {pattern}")
        print(f"注水參數: 直徑={self.POUR_DIAMETER_GRID:.2f}格, 速度={self.POUR_VELOCITY:.3f}lu/ts")
    
    def stop_pouring(self):
        """停止注水"""
        self.pouring_active[None] = 0
        print("停止注水")
    
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

    @ti.func
    def _is_in_pouring_region(self, i: ti.i32, j: ti.i32, k: ti.i32, 
                              pour_x: ti.f32, pour_y: ti.f32) -> ti.f32:
        """統一的倒灌區域識別與強度計算
        
        Args:
            i, j, k: 格點座標
            pour_x, pour_y: 注水中心位置
            
        Returns:
            0.0: 不在倒灌區域
            >0.0: 在倒灌區域，返回總強度值 (高斯徑向 × 指數式垂直衰減)
        """
        dx = i - pour_x
        dy = j - pour_y
        distance_to_pour = ti.sqrt(dx*dx + dy*dy)
        
        pour_radius = self.POUR_DIAMETER_GRID / 2.0
        pour_z = self.POUR_HEIGHT
        pour_stream_height = 4.0
        
        total_intensity = 0.0
        if distance_to_pour <= pour_radius and k <= pour_z and k >= pour_z - pour_stream_height:
            # 高斯徑向分佈 (中心最強)
            intensity = ti.exp(-0.5 * (distance_to_pour / pour_radius)**2)
            # 指數式垂直衰減 (從噴嘴向下)
            vertical_distance = pour_z - k
            vertical_decay = ti.exp(-vertical_distance / 2.0)
            total_intensity = intensity * vertical_decay
        
        return total_intensity

    @ti.kernel
    def apply_pouring_force(self, lbm_body_force: ti.template(), 
                            solid: ti.template(), dt: ti.f32):
        """以體力注入的方式施加注水（配合Guo forcing）

        - 僅在噴嘴附近区域施加向下的加速度，避免被SoA巨觀量重算覆寫
        - 不再直接修改相場，交由 apply_gradual_phase_change 統一處理
        - 加速度標度：approx target_u / dt * intensity * flow_rate
        """
        if self.pouring_active[None] == 1:
            # 更新注水時間（用於螺旋軌跡）
            self.pour_time[None] += dt

            # 當前注水中心
            pour_x, pour_y = self._get_current_pour_position()

            for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
                if solid[i, j, k] == 0:  # 只在流體區域施加體力
                    total_intensity = self._is_in_pouring_region(i, j, k, pour_x, pour_y)
                    
                    if total_intensity > 0.0:
                        # 以目標速度/時間步近似所需加速度，並限幅
                        # 注意：POUR_VELOCITY為lu/ts，dt為本步使用時間步
                        accel_mag = 0.0
                        if dt > 1e-8:
                            accel_mag = self.POUR_VELOCITY * total_intensity * self.pour_flow_rate[None] / dt
                        # 限制過大加速度數值，避免forcing被內核夾制後失真
                        accel_mag = ti.min(accel_mag, 10.0)  # 與重力同級上限

                        # 僅施加向下加速度（z負向）；徑向分散由流場自行演化
                        bf = lbm_body_force[i, j, k]
                        bf += ti.Vector([0.0, 0.0, -accel_mag])
                        lbm_body_force[i, j, k] = bf
    
    @ti.kernel
    def apply_gradual_phase_change(self, multiphase_phi: ti.template(), solid: ti.template(), dt: ti.f32):
        """漸進式相場變化 - 唯一負責注水過程中相場修改的函數"""
        if self.pouring_active[None] == 1:
            pour_x, pour_y = self._get_current_pour_position()
            
            for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
                if solid[i, j, k] == 0:  # 只在流體區域更新相場
                    total_intensity = self._is_in_pouring_region(i, j, k, pour_x, pour_y)
                    
                    if total_intensity > 0.0:
                        # 漸進式相場設置
                        target_phi = 1.0  # 目標：純水相
                        current_phi = multiphase_phi[i, j, k]
                        
                        # 使用時間常數控制相場變化速率
                        tau_phase = 0.05  # 相場變化時間常數
                        change_rate = (target_phi - current_phi) / tau_phase
                        
                        # 限制最大變化率，避免數值震盪
                        max_change_per_dt = 2.0
                        change_rate = ti.max(-max_change_per_dt, ti.min(max_change_per_dt, change_rate))
                        
                        # 考慮注水強度的影響
                        phase_change = change_rate * total_intensity * dt * self.pour_flow_rate[None]
                        new_phi = current_phi + phase_change
                        
                        # 確保相場在物理範圍內
                        multiphase_phi[i, j, k] = ti.max(-1.0, ti.min(1.0, new_phi))

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
    
    def get_pouring_info(self):
        """獲取注水資訊用於診斷"""
        if self.pouring_active[None] == 1:
            # 手動計算當前位置，避免調用 @ti.func
            x = float(self.pour_center_x[None])
            y = float(self.pour_center_y[None])
            
            if self.pour_pattern[None] == 1:  # 螺旋注水
                import math
                t = float(self.pour_time[None] * self.spiral_speed[None])
                current_radius = float(self.spiral_radius[None] * (1.0 + 0.1 * t))
                
                x = float(self.spiral_center_x[None]) + current_radius * math.cos(t)
                y = float(self.spiral_center_y[None]) + current_radius * math.sin(t)
                
                # 確保在邊界內
                x = max(self.POUR_DIAMETER_GRID, min(config.NX - self.POUR_DIAMETER_GRID, x))
                y = max(self.POUR_DIAMETER_GRID, min(config.NY - self.POUR_DIAMETER_GRID, y))
            
            return {
                'active': True,
                'position': (x, y),
                'diameter_grid': float(self.POUR_DIAMETER_GRID),
                'diameter_cm': float(self.POUR_DIAMETER_CM),
                'velocity': float(self.POUR_VELOCITY),
                'flow_rate': float(self.pour_flow_rate[None]),
                'pour_time': float(self.pour_time[None]),
                'pattern': int(self.pour_pattern[None])
            }
        else:
            return {
                'active': False,
                'position': (0, 0),
                'diameter_grid': 0,
                'diameter_cm': 0,
                'velocity': 0,
                'flow_rate': 0,
                'pour_time': 0,
                'pattern': 0
            }
    
    def get_pouring_diagnostics(self):
        """獲取注水系統診斷資訊"""
        diagnostics = {
            'configuration': {
                'diameter_cm': self.POUR_DIAMETER_CM,
                'diameter_grid': self.POUR_DIAMETER_GRID,
                'height': self.POUR_HEIGHT,
                'velocity': self.POUR_VELOCITY,
                'grid_size_cm': config.GRID_SIZE_CM
            },
            'current_state': self.get_pouring_info(),
            'conditions_check': self._check_pouring_conditions()
        }
        return diagnostics

    def get_current_flow_rate(self) -> float:
        """回傳當前注水體積流率（格子單位 lu^3/ts）

        - 以噴嘴截面積（物理）換算為格子單位面積，再乘以格子速度
        - 乘上當前 `flow_rate` 作為比例因子
        """
        try:
            if self.pouring_active[None] == 0:
                return 0.0
            area_lu2 = config.INLET_AREA / (config.SCALE_LENGTH ** 2)
            u_lu = config.INLET_VELOCITY
            return float(self.pour_flow_rate[None] * u_lu * area_lu2)
        except Exception:
            return 0.0

    def get_current_flow_rate_ml_s(self) -> float:
        """回傳當前注水體積流率（ml/s）"""
        try:
            # 基礎流率（ml/s）乘上 flow_rate 作為比例
            return float(config.POUR_RATE_ML_S * max(0.0, self.pour_flow_rate[None]))
        except Exception:
            return 0.0
    
    def _check_pouring_conditions(self):
        """檢查注水條件的有效性"""
        pour_radius = self.POUR_DIAMETER_GRID / 2.0
        pour_z = self.POUR_HEIGHT
        pour_stream_height = 4.0
        center_x = config.NX // 2
        center_y = config.NY // 2
        
        # 統計滿足注水條件的格子數
        affected_cells = 0
        total_checked = 0
        
        # 簡單檢查（避免在Python中嵌套三層循環）
        for i in range(max(0, int(center_x - pour_radius)), 
                      min(config.NX, int(center_x + pour_radius + 1))):
            for j in range(max(0, int(center_y - pour_radius)),
                          min(config.NY, int(center_y + pour_radius + 1))):
                for k in range(max(0, int(pour_z - pour_stream_height)),
                              min(config.NZ, int(pour_z + 1))):
                    total_checked += 1
                    
                    dx = i - center_x
                    dy = j - center_y
                    distance_to_pour = (dx*dx + dy*dy)**0.5
                    
                    if distance_to_pour <= pour_radius and k <= pour_z and k >= pour_z - pour_stream_height:
                        affected_cells += 1
        
        return {
            'center_position': (center_x, center_y),
            'pour_radius': pour_radius,
            'z_range': [pour_z - pour_stream_height, pour_z],
            'affected_cells': affected_cells,
            'total_checked': total_checked,
            'effectiveness': affected_cells / max(1, total_checked)
        }
    
    def diagnose_pouring_system(self):
        """診斷注水系統"""
        print("\n🚿 注水系統診斷")
        print("-" * 30)
        
        diagnostics = self.get_pouring_diagnostics()
        
        # 配置資訊
        config_info = diagnostics['configuration']
        print(f"配置:")
        print(f"  注水直徑: {config_info['diameter_cm']:.2f} cm ({config_info['diameter_grid']:.1f} 格)")
        print(f"  注水高度: {config_info['height']:.1f} 格")
        print(f"  注水速度: {config_info['velocity']:.3f}")
        print(f"  格子尺寸: {config_info['grid_size_cm']:.3f} cm")
        
        # 當前狀態
        state = diagnostics['current_state']
        print(f"\n當前狀態:")
        print(f"  注水活躍: {'是' if state['active'] else '否'}")
        if state['active']:
            print(f"  注水位置: ({state['position'][0]:.1f}, {state['position'][1]:.1f})")
            print(f"  流量率: {state['flow_rate']:.3f}")
            print(f"  注水時間: {state['pour_time']:.2f}s")
        
        # 條件檢查
        conditions = diagnostics['conditions_check']
        print(f"\n條件檢查:")
        print(f"  影響格子數: {conditions['affected_cells']:,}")
        print(f"  檢查格子數: {conditions['total_checked']:,}")
        print(f"  有效性: {conditions['effectiveness']:.1%}")
        print(f"  Z範圍: {conditions['z_range'][0]:.1f} -> {conditions['z_range'][1]:.1f}")
        
        if conditions['affected_cells'] == 0:
            print("⚠️  警告：沒有格子受到注水影響！")
        elif conditions['affected_cells'] < 10:
            print("⚠️  警告：受影響格子數過少！")
    
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