"""
增強版咖啡顆粒系統 - 完整物理與約束 (防護式設計)
包含高斯分布、流體作用力和邊界約束
修復極座標溢出和數值穩定性問題
開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import config
import math
from typing import Dict, Any

@ti.data_oriented
class CoffeeParticleSystem:
    """增強版咖啡顆粒系統 - 包含完整物理與約束 (防護式設計)"""
    
    def __init__(self, max_particles=15000):
        self.max_particles = max_particles
        print(f"☕ 初始化增強顆粒系統 (max: {max_particles:,})...")
        
        # 物理邊界常數 (防止數值溢出)
        self.MAX_COORDINATE = float(max(config.NX, config.NY, config.NZ))
        self.MIN_COORDINATE = 0.0
        self.MAX_VELOCITY = 10.0  # m/s 物理上限
        self.MAX_RADIUS = 0.01    # 1cm 物理上限
        self.MIN_RADIUS = 1e-5    # 10微米 物理下限
        
        # 顆粒基本屬性
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.radius = ti.field(dtype=ti.f32, shape=max_particles)
        self.mass = ti.field(dtype=ti.f32, shape=max_particles)
        self.active = ti.field(dtype=ti.i32, shape=max_particles)
        
        # 增強物理屬性
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.settling_velocity = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.particle_reynolds = ti.field(dtype=ti.f32, shape=max_particles)
        
        # 增強物理屬性
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.settling_velocity = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.particle_reynolds = ti.field(dtype=ti.f32, shape=max_particles)
        
        # 雙向耦合屬性 (Phase 2強耦合) - 新增
        self.drag_coefficient = ti.field(dtype=ti.f32, shape=max_particles)
        self.fluid_velocity_at_particle = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.drag_force = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        
        # 反作用力場（顆粒→流體）- 新增
        self.reaction_force_field = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 亞鬆弛控制 - 新增
        self.drag_force_old = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.drag_force_new = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        
        # 計數器
        self.particle_count = ti.field(dtype=ti.i32, shape=())
        self.active_count = ti.field(dtype=ti.i32, shape=())
        
        # 錯誤統計
        self.boundary_violations = ti.field(dtype=ti.i32, shape=())
        self.coordinate_errors = ti.field(dtype=ti.i32, shape=())
        
        # 物理常數
        self.gravity = 9.81
        self.coffee_density = config.COFFEE_BEAN_DENSITY
        self.water_density = config.WATER_DENSITY_90C
        self.water_viscosity = config.WATER_VISCOSITY_90C * config.WATER_DENSITY_90C
        
        print("✅ 增強顆粒系統初始化完成 (含防護機制)")
    
    @ti.func
    def validate_coordinate(self, x: ti.f32, y: ti.f32, z: ti.f32) -> ti.i32:
        """驗證座標是否在合理範圍內"""
        valid = 1
        if (x < self.MIN_COORDINATE or x > self.MAX_COORDINATE or
            y < self.MIN_COORDINATE or y > self.MAX_COORDINATE or
            z < self.MIN_COORDINATE or z > self.MAX_COORDINATE):
            valid = 0
        
        # 檢查NaN或inf - 使用Taichi的內置檢查
        if not (x == x and y == y and z == z):  # NaN檢查: NaN != NaN
            valid = 0
        
        # 檢查無窮大
        max_val = 1e6
        if (ti.abs(x) > max_val or ti.abs(y) > max_val or ti.abs(z) > max_val):
            valid = 0
        
        return valid
    
    @ti.func 
    def validate_velocity(self, vx: ti.f32, vy: ti.f32, vz: ti.f32) -> ti.i32:
        """驗證速度是否在合理範圍內"""
        valid = 1
        speed_squared = vx*vx + vy*vy + vz*vz
        
        # 檢查NaN
        if not (vx == vx and vy == vy and vz == vz):
            valid = 0
        
        # 檢查速度大小
        if speed_squared > self.MAX_VELOCITY * self.MAX_VELOCITY:
            valid = 0
        
        return valid
    
    @ti.func
    def validate_radius(self, radius: ti.f32) -> ti.i32:
        """驗證半徑是否在合理範圍內"""
        valid = 1
        if (radius < self.MIN_RADIUS or radius > self.MAX_RADIUS or 
            radius != radius):  # NaN檢查
            valid = 0
        return valid
    
    @ti.kernel
    def clear_all_particles(self):
        """清空所有顆粒並重置錯誤統計"""
        for i in range(self.max_particles):
            self.position[i] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
            self.settling_velocity[i] = ti.Vector([0.0, 0.0, 0.0])
            self.radius[i] = 0.0
            self.mass[i] = 0.0
            self.active[i] = 0
            self.particle_reynolds[i] = 0.0
            
            # 雙向耦合字段初始化
            self.drag_coefficient[i] = 0.0
            self.fluid_velocity_at_particle[i] = ti.Vector([0.0, 0.0, 0.0])
            self.drag_force[i] = ti.Vector([0.0, 0.0, 0.0])
            self.drag_force_old[i] = ti.Vector([0.0, 0.0, 0.0])
            self.drag_force_new[i] = ti.Vector([0.0, 0.0, 0.0])
            
        # 清零反作用力場
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            self.reaction_force_field[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            
        self.particle_count[None] = 0
        self.active_count[None] = 0
        self.boundary_violations[None] = 0
        self.coordinate_errors[None] = 0
    
    def generate_gaussian_particle_radius(self, mean_radius=None, std_dev_ratio=0.3):
        """
        生成高斯分佈的咖啡粉顆粒半徑 (防護式設計)
        
        Args:
            mean_radius: 平均半徑 (m)，默認使用config中的值
            std_dev_ratio: 標準差相對於均值的比例 (30%)
        
        Returns:
            半徑值 (m) - 保證在合理物理範圍內
        """
        if mean_radius is None:
            mean_radius = config.COFFEE_PARTICLE_RADIUS
        
        # 確保mean_radius在合理範圍內
        mean_radius = max(self.MIN_RADIUS, min(self.MAX_RADIUS, mean_radius))
        
        std_dev = mean_radius * std_dev_ratio
        
        # 生成高斯分佈半徑 (多次嘗試以確保合理值)
        for attempt in range(10):
            radius = np.random.normal(mean_radius, std_dev)
            
            # 限制半徑範圍：50%-150%的平均值
            min_radius = max(self.MIN_RADIUS, mean_radius * 0.5)
            max_radius = min(self.MAX_RADIUS, mean_radius * 1.5)
            radius = np.clip(radius, min_radius, max_radius)
            
            # 檢查是否為有效數值
            if np.isfinite(radius) and min_radius <= radius <= max_radius:
                return float(radius)
        
        # 如果所有嘗試都失敗，返回安全的默認值
        print(f"⚠️  粒子半徑生成失敗，使用默認值: {mean_radius}")
        return float(mean_radius)
    
    @ti.kernel
    def create_particle_with_physics(self, idx: ti.i32, px: ti.f32, py: ti.f32, pz: ti.f32, 
                                   radius: ti.f32, vx: ti.f32, vy: ti.f32, vz: ti.f32) -> ti.i32:
        """創建帶完整物理屬性的顆粒 (返回成功/失敗)"""
        success = 0
        if idx < self.max_particles:
            # 驗證所有輸入參數
            if (self.validate_coordinate(px, py, pz) and 
                self.validate_velocity(vx, vy, vz) and 
                self.validate_radius(radius)):
                
                self.position[idx] = ti.Vector([px, py, pz])
                self.velocity[idx] = ti.Vector([vx, vy, vz])
                self.radius[idx] = radius
                
                # 計算質量 (球體體積 × 密度)
                volume = (4.0/3.0) * 3.14159 * radius**3
                mass = volume * self.coffee_density
                
                # 驗證質量
                if mass == mass and mass > 0:  # NaN檢查: mass == mass
                    self.mass[idx] = mass
                    self.active[idx] = 1
                    self.force[idx] = ti.Vector([0.0, 0.0, 0.0])
                    self.settling_velocity[idx] = ti.Vector([0.0, 0.0, 0.0])
                    self.particle_reynolds[idx] = 0.0
                    success = 1
                else:
                    # 質量無效
                    self.coordinate_errors[None] += 1
            else:
                # 參數驗證失敗
                self.coordinate_errors[None] += 1
        
        return success
    
    def initialize_coffee_bed_confined(self, filter_paper_system):
        """在濾紙上方生成真實的咖啡床（防護式設計，解決z=1.4e10問題）"""
        print("🔧 生成真實咖啡床（濾紙上方）- 防護式設計...")
        
        # 清空現有顆粒
        self.clear_all_particles()
        
        # 獲取濾紙邊界信息並驗證
        try:
            boundary = filter_paper_system.get_coffee_bed_boundary()
            center_x = float(boundary['center_x'])
            center_y = float(boundary['center_y'])
            bottom_z = float(boundary['bottom_z'])
            top_radius_lu = float(boundary['top_radius_lu'])
            bottom_radius_lu = float(boundary['bottom_radius_lu'])
        except Exception as e:
            print(f"❌ 濾紙邊界獲取失敗: {e}")
            return 0
        
        # 驗證邊界參數
        if not all(np.isfinite([center_x, center_y, bottom_z, top_radius_lu, bottom_radius_lu])):
            print(f"❌ 邊界參數包含非法值!")
            return 0
        
        if (center_x < 0 or center_x > config.NX or
            center_y < 0 or center_y > config.NY or
            bottom_z < 0 or bottom_z > config.NZ):
            print(f"❌ 邊界參數超出網格範圍!")
            print(f"   center: ({center_x:.1f}, {center_y:.1f}), bottom_z: {bottom_z:.1f}")
            print(f"   grid: {config.NX}×{config.NY}×{config.NZ}")
            return 0
        
        # === 關鍵修正：咖啡床位置計算 (安全版本) ===
        # 濾紙表面位置（濾紙底部 + 安全間距）
        filter_surface_z = bottom_z + 2.0  # 濾紙表面
        
        # 使用真實的咖啡床高度並限制在合理範圍
        try:
            coffee_bed_height_phys = getattr(config, 'COFFEE_BED_HEIGHT_PHYS', 0.015)  # 默認1.5cm
            coffee_bed_height_phys = max(0.005, min(0.05, coffee_bed_height_phys))  # 限制0.5-5cm
            coffee_bed_height_lu = coffee_bed_height_phys / config.SCALE_LENGTH
            coffee_bed_height_lu = max(5.0, min(30.0, coffee_bed_height_lu))  # 限制5-30格子單位
        except:
            coffee_bed_height_lu = 15.0  # 安全默認值
            coffee_bed_height_phys = coffee_bed_height_lu * config.SCALE_LENGTH
        
        # 咖啡床範圍：從濾紙表面開始往上（安全檢查）
        coffee_bed_bottom = filter_surface_z
        coffee_bed_top = coffee_bed_bottom + coffee_bed_height_lu
        
        # 確保咖啡床不會超出網格範圍
        if coffee_bed_top >= config.NZ - 5:
            coffee_bed_top = config.NZ - 5
            coffee_bed_height_lu = coffee_bed_top - coffee_bed_bottom
            print(f"⚠️  咖啡床高度已調整以適應網格範圍")
        
        print(f"   └─ V60中心: ({center_x:.1f}, {center_y:.1f})")
        print(f"   └─ 濾紙表面: Z = {filter_surface_z:.1f} lu")
        print(f"   └─ 咖啡床範圍: Z = {coffee_bed_bottom:.1f} -> {coffee_bed_top:.1f} lu")
        print(f"   └─ 咖啡床高度: {coffee_bed_height_lu:.1f} lu ({coffee_bed_height_phys*100:.1f}cm)")
        
        # === 防護式3D分層生成策略 ===
        target_particles = min(2000, self.max_particles - 100)  # 留100個緩衝
        layer_count = max(10, min(int(coffee_bed_height_lu / 2), 30))  # 限制層數
        
        # 確保每層至少有1個顆粒
        if target_particles < layer_count:
            layer_count = max(1, target_particles)
        
        particles_per_layer = max(1, target_particles // layer_count)  # 確保至少1個
        
        created = 0
        successful_placements = 0
        total_attempts = 0
        coordinate_errors = 0
        
        print(f"   └─ 目標顆粒: {target_particles}, 分{layer_count}層, 每層{particles_per_layer}個")
        
        for layer in range(layer_count):
            if created >= target_particles:
                break
                
            # 計算該層在咖啡床內的高度（從濾紙表面開始）
            layer_height_ratio = float(layer) / float(layer_count)
            z = coffee_bed_bottom + layer_height_ratio * coffee_bed_height_lu
            
            # 確保在咖啡床範圍內
            if z > coffee_bed_top:
                break
            
            # 計算該高度處的V60半徑（錐形容器）- 防護式計算
            try:
                cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
                cup_height_lu = max(10.0, cup_height_lu)  # 避免除零
                height_in_v60 = z - bottom_z
                radius_ratio = min(1.0, max(0.0, height_in_v60 / cup_height_lu))
                layer_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * radius_ratio
                layer_radius = max(1.0, min(top_radius_lu * 1.5, layer_radius))  # 安全範圍
            except:
                layer_radius = bottom_radius_lu  # 安全默認值
            
            # 咖啡床有效半徑（考慮錐台形狀）- 防護式計算
            try:
                coffee_bed_radius_ratio = (coffee_bed_height_lu - (z - coffee_bed_bottom)) / coffee_bed_height_lu
                coffee_bed_radius_ratio = max(0.1, min(1.0, coffee_bed_radius_ratio))
                coffee_bed_effective_radius = getattr(config, 'COFFEE_BED_TOP_RADIUS', top_radius_lu * 0.8) / config.SCALE_LENGTH
                current_coffee_radius = coffee_bed_effective_radius * (0.3 + 0.7 * coffee_bed_radius_ratio)
            except:
                current_coffee_radius = layer_radius * 0.5  # 安全默認值
            
            # 取較小的半徑（V60邊界 vs 咖啡床形狀）
            effective_radius = min(layer_radius * 0.85, current_coffee_radius)
            effective_radius = max(2.0, min(effective_radius, 50.0))  # 限制在2-50格子單位
            
            # 在該層生成顆粒 - 防護式生成
            layer_particles = 0
            attempts = 0
            max_attempts = particles_per_layer * 5  # 增加嘗試次數
            
            while layer_particles < particles_per_layer and attempts < max_attempts and created < target_particles:
                attempts += 1
                total_attempts += 1
                
                try:
                    # 防護式隨機極坐標位置
                    angle = np.random.uniform(0, 2*math.pi)
                    # 使用更保守的分布
                    r_normalized = np.random.uniform(0, 1) ** 1.5  # 向中心集中
                    r = r_normalized * effective_radius * 0.9  # 額外安全邊距
                    
                    # 確保r在合理範圍內
                    r = max(0.0, min(effective_radius * 0.9, r))
                    
                    # 計算笛卡爾坐標
                    x = center_x + r * math.cos(angle)
                    y = center_y + r * math.sin(angle)
                    
                    # 添加隨機高度變化（模擬自然堆積）- 防護式
                    z_variation = np.random.uniform(-0.5, 0.5)  # 減小變化範圍
                    z_final = z + z_variation
                    
                    # 嚴格的邊界檢查
                    if not (5.0 <= x <= config.NX-5.0 and 
                           5.0 <= y <= config.NY-5.0 and 
                           coffee_bed_bottom <= z_final <= coffee_bed_top):
                        continue
                    
                    # 確保所有數值都是有限的
                    if not all(np.isfinite([x, y, z_final, r, angle])):
                        coordinate_errors += 1
                        continue
                    
                    # 確保在V60容器內 - 使用安全檢查
                    safe_boundary_check = self._safe_cone_boundary_check(
                        x, y, z_final, center_x, center_y, 
                        bottom_z, bottom_radius_lu, top_radius_lu)
                    
                    if safe_boundary_check:
                        # 生成高斯分佈半徑 - 防護式
                        particle_radius = self.generate_gaussian_particle_radius()
                        
                        # 創建顆粒（靜止狀態）- 檢查成功與否
                        success = self.create_particle_with_physics(created, x, y, z_final, particle_radius, 0, 0, 0)
                        if success:
                            created += 1
                            layer_particles += 1
                            successful_placements += 1
                        else:
                            coordinate_errors += 1
                    
                except Exception as e:
                    coordinate_errors += 1
                    if coordinate_errors < 5:  # 只報告前幾個錯誤
                        print(f"   ⚠️  顆粒生成錯誤: {e}")
            
            if (layer + 1) % 5 == 0:
                print(f"   └─ 完成第 {layer+1}/{layer_count} 層，累計顆粒: {created}")
        
        # 更新計數
        self.particle_count[None] = created
        self.active_count[None] = created
        
        # 獲取錯誤統計
        coord_errors = self.coordinate_errors[None]
        
        print(f"✅ 真實咖啡床生成完成 (防護式設計):")
        print(f"   └─ 成功生成: {created:,} 顆粒")
        print(f"   └─ 成功率: {successful_placements/max(1,total_attempts)*100:.1f}%")
        print(f"   └─ 座標錯誤: {coord_errors + coordinate_errors}")
        print(f"   └─ 咖啡床位置: 濾紙表面上方")
        print(f"   └─ 物理高度: {coffee_bed_height_phys*100:.1f}cm")
        
        if coord_errors + coordinate_errors > 0:
            print(f"   ⚠️  檢測到數值問題，已通過防護機制解決")
        
        return created
    
    def _safe_cone_boundary_check(self, x, y, z, center_x, center_y, bottom_z, 
                                bottom_radius_lu, top_radius_lu):
        """安全的錐形邊界檢查（防止數值溢出）"""
        try:
            # 計算到軸心的距離 - 防護式
            dx = float(x - center_x)
            dy = float(y - center_y)
            
            # 檢查是否有異常值
            if not (np.isfinite(dx) and np.isfinite(dy)):
                return False
            
            distance_from_center = math.sqrt(dx*dx + dy*dy)
            
            # 檢查距離是否合理
            if not np.isfinite(distance_from_center) or distance_from_center > 1000:
                return False
            
            # 計算該高度處的最大允許半徑
            if z <= bottom_z:
                return distance_from_center <= bottom_radius_lu
            
            cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
            if cup_height_lu <= 0:
                return False
                
            height_ratio = min(1.0, max(0.0, (z - bottom_z) / cup_height_lu))
            max_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            
            # 檢查最大半徑是否合理
            if not np.isfinite(max_radius) or max_radius <= 0:
                return False
            
            return distance_from_center <= max_radius * 0.9  # 留10%安全邊距
            
        except Exception:
            # 任何異常都返回False
            return False
    
    @ti.kernel
    def enforce_filter_boundary(self, center_x: ti.f32, center_y: ti.f32, bottom_z: ti.f32,
                               bottom_radius_lu: ti.f32, top_radius_lu: ti.f32) -> ti.i32:
        """強制執行濾紙邊界約束 - 防護式設計，返回違反邊界的顆粒數"""
        violations = 0
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        
        # 防護：確保cup_height_lu不為零
        if cup_height_lu <= 0:
            cup_height_lu = 50.0  # 安全默認值
        
        for i in range(self.max_particles):
            if self.active[i] == 1:
                pos = self.position[i]
                
                # 驗證當前位置是否有效
                if not self.validate_coordinate(pos.x, pos.y, pos.z):
                    # 位置無效，停用該顆粒
                    self.active[i] = 0
                    violations += 1
                    self.coordinate_errors[None] += 1
                    continue
                
                # 計算到軸心的距離 - 防護式計算
                dx = pos.x - center_x
                dy = pos.y - center_y
                distance_squared = dx*dx + dy*dy
                
                # 檢查距離平方是否合理
                if distance_squared > 1e6:  # 距離過大
                    self.active[i] = 0
                    violations += 1
                    self.coordinate_errors[None] += 1
                    continue
                
                distance_from_center = ti.sqrt(distance_squared)
                
                # 計算該高度的最大允許半徑 - 防護式計算
                max_radius = bottom_radius_lu  # 默認底部半徑
                if pos.z > bottom_z:
                    height_diff = pos.z - bottom_z
                    if height_diff < cup_height_lu:  # 防止除以零或負數
                        height_ratio = height_diff / cup_height_lu
                        height_ratio = ti.max(0.0, ti.min(1.0, height_ratio))  # 限制在[0,1]
                        max_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                
                # 邊界違反檢查
                if distance_from_center > max_radius * 0.9:
                    # 推回邊界內 - 防護式修正
                    if distance_from_center > 0:  # 避免除以零
                        scale_factor = max_radius * 0.8 / distance_from_center
                        scale_factor = ti.max(0.1, ti.min(1.0, scale_factor))  # 限制縮放比例
                        
                        new_x = center_x + dx * scale_factor
                        new_y = center_y + dy * scale_factor
                        
                        # 驗證新位置
                        if self.validate_coordinate(new_x, new_y, pos.z):
                            self.position[i].x = new_x
                            self.position[i].y = new_y
                        else:
                            # 新位置無效，停用顆粒
                            self.active[i] = 0
                        
                        # 清零速度（避免進一步移動出邊界）
                        self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
                        violations += 1
                
                # 底部邊界 - 防護式檢查
                if pos.z < bottom_z:
                    new_z = bottom_z + 0.1
                    if self.validate_coordinate(pos.x, pos.y, new_z):
                        self.position[i].z = new_z
                        self.velocity[i].z = 0.0
                    else:
                        self.active[i] = 0
                    violations += 1
                
                # 頂部邊界檢查
                max_z = ti.min(bottom_z + cup_height_lu * 2, ti.f32(config.NZ - 5))
                if pos.z > max_z:
                    new_z = max_z - 0.1
                    if self.validate_coordinate(pos.x, pos.y, new_z):
                        self.position[i].z = new_z
                        self.velocity[i].z = 0.0
                    else:
                        self.active[i] = 0
                    violations += 1
        
        return violations
    
    @ti.kernel
    def apply_fluid_forces(self, fluid_u: ti.template(), fluid_v: ti.template(), 
                          fluid_w: ti.template(), fluid_density: ti.template(), 
                          pressure: ti.template(), dt: ti.f32):
        """應用簡化的流體作用力到顆粒 - 防護式設計"""
        
        # 驗證時間步長 - 創建局部變量
        dt_safe = ti.max(1e-8, ti.min(1e-2, dt))  # 限制dt在合理範圍
        
        for i in range(self.max_particles):
            if self.active[i] == 1:
                pos = self.position[i]
                
                # 驗證顆粒位置
                if not self.validate_coordinate(pos.x, pos.y, pos.z):
                    self.active[i] = 0  # 停用無效顆粒
                    self.coordinate_errors[None] += 1
                    continue
                
                # 獲取網格索引 - 防護式轉換
                grid_i = ti.cast(ti.max(0, ti.min(config.NX-2, pos.x)), ti.i32)
                grid_j = ti.cast(ti.max(0, ti.min(config.NY-2, pos.y)), ti.i32)
                grid_k = ti.cast(ti.max(0, ti.min(config.NZ-2, pos.z)), ti.i32)
                
                # 邊界檢查
                if (grid_i >= 0 and grid_i < config.NX-1 and 
                    grid_j >= 0 and grid_j < config.NY-1 and 
                    grid_k >= 0 and grid_k < config.NZ-1):
                    
                    # 直接使用向量場 - LBM的u場是向量場
                    fluid_velocity = fluid_u[grid_i, grid_j, grid_k]
                    
                    # 驗證流體速度
                    fluid_speed = fluid_velocity.norm()
                    if fluid_speed == fluid_speed and fluid_speed <= 100.0:  # NaN檢查並限制速度
                    
                        particle_vel = self.velocity[i]
                        
                        # 驗證顆粒速度
                        if not self.validate_velocity(particle_vel.x, particle_vel.y, particle_vel.z):
                            # 重置無效速度
                            self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
                            particle_vel = self.velocity[i]
                        
                        relative_vel = fluid_velocity - particle_vel
                        relative_speed = relative_vel.norm()
                        
                        if relative_speed > 1e-6 and relative_speed < 10.0:  # 合理的相對速度範圍
                            radius = self.radius[i]
                            mass = self.mass[i]
                            
                            # 驗證顆粒屬性
                            if self.validate_radius(radius) and mass == mass and mass > 0:
                                
                                # 初始化所有力為零（確保在所有執行路徑中定義）
                                drag_force = ti.Vector([0.0, 0.0, 0.0])
                                buoyancy = ti.Vector([0.0, 0.0, 0.0])
                                gravity_force = ti.Vector([0.0, 0.0, 0.0])
                                
                                # === 1. 簡化阻力 (Stokes) - 防護式計算 ===
                                # 阻力係數
                                Re_p = relative_speed * 2.0 * radius * self.water_density / ti.max(1e-8, self.water_viscosity)
                                Re_p = ti.max(0.01, ti.min(1000.0, Re_p))  # 限制雷諾數範圍
                                
                                C_D = 24.0 / ti.max(0.1, Re_p)
                                C_D = ti.max(0.1, ti.min(10.0, C_D))  # 限制阻力係數
                                
                                # 阻力
                                drag_magnitude = 0.5 * C_D * 3.14159 * radius**2 * self.water_density * relative_speed
                                drag_magnitude = ti.min(drag_magnitude, mass * 100.0)  # 限制阻力不超過100g
                                
                                if relative_speed > 0:
                                    drag_force = drag_magnitude * (relative_vel / relative_speed)
                                
                                # === 2. 浮力 - 防護式計算 ===
                                volume = (4.0/3.0) * 3.14159 * radius**3
                                buoyancy_magnitude = volume * self.water_density * self.gravity
                                buoyancy = ti.min(buoyancy_magnitude, mass * 20.0) * ti.Vector([0, 0, 1])  # 限制浮力
                                
                                # === 3. 重力 - 防護式計算 ===
                                gravity_magnitude = mass * self.gravity
                                gravity_force = gravity_magnitude * ti.Vector([0, 0, -1])
                                
                                # === 總力 - 防護式組合 ===
                                total_force = drag_force + buoyancy + gravity_force
                                
                                # 驗證總力
                                force_magnitude = total_force.norm()
                                if force_magnitude == force_magnitude and force_magnitude < mass * 1000.0:  # NaN檢查並限制力
                                    self.force[i] = total_force
                                else:
                                    # 力過大或無效，只應用重力
                                    self.force[i] = gravity_force
    
    @ti.kernel
    def update_particle_physics(self, dt: ti.f32, center_x: ti.f32, center_y: ti.f32, 
                               bottom_z: ti.f32, bottom_radius_lu: ti.f32, top_radius_lu: ti.f32):
        """更新顆粒物理（集成力、更新位置速度、邊界約束）- 防護式設計"""
        
        # 驗證時間步長 - 創建局部變量
        dt_safe = ti.max(1e-8, ti.min(1e-2, dt))  # 限制在合理範圍
        
        for i in range(self.max_particles):
            if self.active[i] == 1:
                # 驗證當前狀態
                if not self.validate_coordinate(self.position[i].x, self.position[i].y, self.position[i].z):
                    self.active[i] = 0
                    self.coordinate_errors[None] += 1
                    continue
                
                # 力積分更新速度 - 防護式計算
                if self.mass[i] > 1e-10:  # 避免除以零或極小質量
                    acceleration = self.force[i] / self.mass[i]
                    
                    # 限制加速度大小
                    acc_magnitude = acceleration.norm()
                    if acc_magnitude > 1000.0:  # 限制加速度不超過1000 m/s²
                        if acc_magnitude > 0:
                            acceleration = acceleration * (1000.0 / acc_magnitude)
                    
                    # 更新速度
                    new_velocity = self.velocity[i] + acceleration * dt_safe
                    
                    # 驗證新速度
                    if self.validate_velocity(new_velocity.x, new_velocity.y, new_velocity.z):
                        self.velocity[i] = new_velocity
                    else:
                        # 速度無效，重置並記錄錯誤
                        self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
                        self.coordinate_errors[None] += 1
                
                # 位置更新 - 防護式計算
                old_pos = self.position[i]
                displacement = self.velocity[i] * dt_safe
                
                # 限制位移大小
                disp_magnitude = displacement.norm()
                if disp_magnitude > 1.0:  # 限制單步位移不超過1個格子單位
                    if disp_magnitude > 0:
                        displacement = displacement * (1.0 / disp_magnitude)
                
                new_pos = old_pos + displacement
                
                # 邊界約束檢查 - 防護式
                boundary_violation = self.check_particle_boundary_violation_safe(
                    new_pos, center_x, center_y, bottom_z, bottom_radius_lu, top_radius_lu)
                
                if boundary_violation == 1:
                    # 違反邊界，約束回內部
                    constrained_pos = self.constrain_to_boundary_safe(
                        new_pos, center_x, center_y, bottom_z, bottom_radius_lu, top_radius_lu)
                    
                    # 驗證約束後的位置
                    if self.validate_coordinate(constrained_pos.x, constrained_pos.y, constrained_pos.z):
                        new_pos = constrained_pos
                        # 阻尼速度
                        self.velocity[i] *= 0.3
                        self.boundary_violations[None] += 1
                    else:
                        # 約束失敗，保持原位置
                        new_pos = old_pos
                        self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
                        self.coordinate_errors[None] += 1
                
                # 最終位置驗證
                if self.validate_coordinate(new_pos.x, new_pos.y, new_pos.z):
                    self.position[i] = new_pos
                else:
                    # 新位置無效，停用顆粒
                    self.active[i] = 0
                    self.coordinate_errors[None] += 1
                
                # 清零力（為下一步準備）
                self.force[i] = ti.Vector([0.0, 0.0, 0.0])
    
    def update_particles(self, dt: float):
        """公共接口：更新顆粒系統 - 用於基準測試"""
        # V60幾何參數 (格子單位)
        center_x = config.NX // 2
        center_y = config.NY // 2
        bottom_z = config.NZ // 4
        bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
        top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
        
        # 調用核心物理更新
        self.update_particle_physics(dt, center_x, center_y, bottom_z, bottom_radius_lu, top_radius_lu)
    
    @ti.func
    def check_particle_boundary_violation_safe(self, pos, center_x: ti.f32, center_y: ti.f32,
                                             bottom_z: ti.f32, bottom_radius_lu: ti.f32, 
                                             top_radius_lu: ti.f32) -> ti.i32:
        """檢查顆粒是否違反邊界 - 防護式版本"""
        violation = 0
        
        # 驗證輸入位置
        valid_position = self.validate_coordinate(pos.x, pos.y, pos.z)
        if not valid_position:
            violation = 1  # 位置無效就是違規
        else:
            # 底部檢查
            if pos.z < bottom_z - 1.0:  # 留1個格子的緩衝
                violation = 1
            else:
                # 錐形側壁檢查 - 防護式計算
                dx = pos.x - center_x
                dy = pos.y - center_y
                distance_squared = dx*dx + dy*dy
                
                # 防止距離過大
                if distance_squared > 1e6:
                    violation = 1
                else:
                    distance_from_center = ti.sqrt(distance_squared)
                    
                    cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
                    if cup_height_lu <= 0:
                        cup_height_lu = 50.0  # 安全默認值
                    
                    height_diff = pos.z - bottom_z
                    if height_diff >= 0 and height_diff < cup_height_lu:
                        height_ratio = height_diff / cup_height_lu
                        height_ratio = ti.max(0.0, ti.min(1.0, height_ratio))
                        max_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                        
                        if distance_from_center > max_radius * 0.9:
                            violation = 1
                    elif height_diff >= cup_height_lu:
                        # 超出容器頂部
                        if distance_from_center > top_radius_lu * 0.9:
                            violation = 1
        
        return violation
    
    @ti.func
    def constrain_to_boundary_safe(self, pos, center_x: ti.f32, center_y: ti.f32,
                                 bottom_z: ti.f32, bottom_radius_lu: ti.f32, 
                                 top_radius_lu: ti.f32):
        """將位置約束到邊界內 - 防護式版本"""
        constrained_pos = pos
        
        # 底部約束 - 防護式
        if constrained_pos.z < bottom_z:
            constrained_pos.z = bottom_z + 0.1
        
        # 頂部約束
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        if cup_height_lu <= 0:
            cup_height_lu = 50.0
        
        max_z = ti.min(bottom_z + cup_height_lu * 1.5, ti.cast(config.NZ - 5, ti.f32))
        if constrained_pos.z > max_z:
            constrained_pos.z = max_z - 0.1
        
        # 錐形側壁約束 - 防護式計算
        dx = constrained_pos.x - center_x
        dy = constrained_pos.y - center_y
        distance_squared = dx*dx + dy*dy
        
        if distance_squared < 1e6:  # 防止距離過大
            distance_from_center = ti.sqrt(distance_squared)
            
            if distance_from_center > 0.1:  # 避免除以接近零的數
                height_diff = constrained_pos.z - bottom_z
                height_diff = ti.max(0.0, height_diff)
                
                # 初始化max_radius為默認值
                max_radius = top_radius_lu
                
                if height_diff < cup_height_lu:
                    height_ratio = height_diff / cup_height_lu
                    height_ratio = ti.max(0.0, ti.min(1.0, height_ratio))
                    max_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                
                if distance_from_center > max_radius * 0.8:
                    scale_factor = max_radius * 0.8 / distance_from_center
                    scale_factor = ti.max(0.1, ti.min(1.0, scale_factor))
                    
                    constrained_pos.x = center_x + dx * scale_factor
                    constrained_pos.y = center_y + dy * scale_factor
        else:
            # 距離過大，強制回到中心附近
            constrained_pos.x = center_x
            constrained_pos.y = center_y
        
        return constrained_pos
    
    def get_particle_statistics(self):
        """獲取顆粒系統統計信息 - 防護式版本"""
        radii = []
        positions = []
        valid_particles = 0
        invalid_particles = 0
        
        for i in range(self.particle_count[None]):
            if self.active[i] == 1:
                try:
                    radius = self.radius[i]
                    pos = self.position[i]
                    
                    # 驗證數據有效性
                    if (np.isfinite(radius) and np.isfinite(pos[0]) and 
                        np.isfinite(pos[1]) and np.isfinite(pos[2]) and
                        self.MIN_RADIUS <= radius <= self.MAX_RADIUS and
                        0 <= pos[0] <= config.NX and
                        0 <= pos[1] <= config.NY and
                        0 <= pos[2] <= config.NZ):
                        
                        radii.append(radius)
                        positions.append([pos[0], pos[1], pos[2]])
                        valid_particles += 1
                    else:
                        invalid_particles += 1
                        
                except Exception:
                    invalid_particles += 1
        
        radii = np.array(radii)
        positions = np.array(positions)
        
        # 計算統計值，包含錯誤檢查
        if len(radii) > 0:
            mean_radius = np.mean(radii)
            std_radius = np.std(radii)
            min_radius = np.min(radii)
            max_radius = np.max(radii)
        else:
            mean_radius = std_radius = min_radius = max_radius = 0.0
        
        # 獲取錯誤統計
        coord_errors = self.coordinate_errors[None]
        boundary_violations = self.boundary_violations[None]
        
        return {
            'count': valid_particles,
            'invalid_count': invalid_particles,
            'coordinate_errors': coord_errors,
            'boundary_violations': boundary_violations,
            'mean_radius': mean_radius,
            'std_radius': std_radius,
            'min_radius': min_radius,
            'max_radius': max_radius,
            'positions': positions,
            'radii': radii,
            'success_rate': valid_particles / max(1, valid_particles + invalid_particles) * 100
        }
    
    def validate_system_integrity(self):
        """驗證系統完整性並返回診斷報告"""
        stats = self.get_particle_statistics()
        
        issues = []
        warnings = []
        
        # 檢查顆粒數量
        if stats['count'] == 0:
            issues.append("沒有有效顆粒")
        elif stats['count'] < 100:
            warnings.append(f"顆粒數量較少: {stats['count']}")
        
        # 檢查錯誤率
        error_rate = stats['coordinate_errors'] / max(1, stats['count'])
        if error_rate > 0.1:
            issues.append(f"座標錯誤率過高: {error_rate:.1%}")
        elif error_rate > 0.01:
            warnings.append(f"座標錯誤率偏高: {error_rate:.1%}")
        
        # 檢查邊界違規
        violation_rate = stats['boundary_violations'] / max(1, stats['count'])
        if violation_rate > 0.2:
            issues.append(f"邊界違規率過高: {violation_rate:.1%}")
        elif violation_rate > 0.05:
            warnings.append(f"邊界違規率偏高: {violation_rate:.1%}")
        
        # 檢查半徑分佈
        if stats['count'] > 0:
            radius_cv = stats['std_radius'] / max(1e-10, stats['mean_radius'])
            if radius_cv > 1.0:
                warnings.append(f"半徑變異過大: CV={radius_cv:.2f}")
        
        # 檢查位置分佈
        if len(stats['positions']) > 0:
            z_coords = stats['positions'][:, 2]
            if np.max(z_coords) > config.NZ * 0.9:
                warnings.append("部分顆粒接近頂部邊界")
            if np.min(z_coords) < 5:
                warnings.append("部分顆粒接近底部邊界")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'stats': stats,
            'healthy': len(issues) == 0
        }
    
    def emergency_cleanup(self):
        """緊急清理異常顆粒"""
        print("🚨 執行緊急顆粒清理...")
        
        cleaned_count = 0
        
        # CPU端清理（更靈活的檢查）
        for i in range(self.particle_count[None]):
            if self.active[i] == 1:
                try:
                    pos = self.position[i]
                    vel = self.velocity[i] 
                    radius = self.radius[i]
                    mass = self.mass[i]
                    
                    # 檢查各種異常情況
                    invalid = False
                    
                    # 位置檢查
                    if not (np.isfinite(pos[0]) and np.isfinite(pos[1]) and np.isfinite(pos[2])):
                        invalid = True
                    elif not (0 <= pos[0] <= config.NX and 0 <= pos[1] <= config.NY and 0 <= pos[2] <= config.NZ):
                        invalid = True
                    
                    # 速度檢查  
                    if not (np.isfinite(vel[0]) and np.isfinite(vel[1]) and np.isfinite(vel[2])):
                        invalid = True
                    elif np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2) > self.MAX_VELOCITY:
                        invalid = True
                    
                    # 半徑檢查
                    if not (np.isfinite(radius) and self.MIN_RADIUS <= radius <= self.MAX_RADIUS):
                        invalid = True
                    
                    # 質量檢查
                    if not (np.isfinite(mass) and mass > 0):
                        invalid = True
                    
                    if invalid:
                        self.active[i] = 0
                        cleaned_count += 1
                        
                except Exception:
                    self.active[i] = 0
                    cleaned_count += 1
        
        # 重新計算有效顆粒數
        active_count = 0
        for i in range(self.particle_count[None]):
            if self.active[i] == 1:
                active_count += 1
        
        self.active_count[None] = active_count
        
        print(f"✅ 清理完成: 移除了 {cleaned_count} 個異常顆粒")
        print(f"   剩餘有效顆粒: {active_count}")
        
        return cleaned_count
    
    # ======================================================================
    # Phase 2 強耦合系統 - 雙向動量傳遞
    # ======================================================================
    
    @ti.func
    def interpolate_fluid_velocity_trilinear(self, particle_idx: ti.i32, fluid_u: ti.template()) -> ti.template():
        """三線性插值獲取顆粒位置的流體速度 - P0完整實現"""
        pos = self.position[particle_idx]
        
        # 網格索引計算（確保邊界安全）
        i = ti.cast(ti.max(0, ti.min(config.NX-2, pos[0])), ti.i32)
        j = ti.cast(ti.max(0, ti.min(config.NY-2, pos[1])), ti.i32) 
        k = ti.cast(ti.max(0, ti.min(config.NZ-2, pos[2])), ti.i32)
        
        # 插值權重計算
        fx = pos[0] - ti.cast(i, ti.f32)
        fy = pos[1] - ti.cast(j, ti.f32)
        fz = pos[2] - ti.cast(k, ti.f32)
        
        # 限制權重在[0,1]範圍內（防護式設計）
        fx = ti.max(0.0, ti.min(1.0, fx))
        fy = ti.max(0.0, ti.min(1.0, fy))
        fz = ti.max(0.0, ti.min(1.0, fz))
        
        # 8個節點權重
        w000 = (1-fx) * (1-fy) * (1-fz)
        w001 = (1-fx) * (1-fy) * fz
        w010 = (1-fx) * fy * (1-fz)
        w011 = (1-fx) * fy * fz
        w100 = fx * (1-fy) * (1-fz)
        w101 = fx * (1-fy) * fz
        w110 = fx * fy * (1-fz)
        w111 = fx * fy * fz
        
        # 三線性插值計算 - 8點插值
        interpolated_u = (
            w000 * fluid_u[i, j, k] +
            w001 * fluid_u[i, j, k+1] +
            w010 * fluid_u[i, j+1, k] +
            w011 * fluid_u[i, j+1, k+1] +
            w100 * fluid_u[i+1, j, k] +
            w101 * fluid_u[i+1, j, k+1] +
            w110 * fluid_u[i+1, j+1, k] +
            w111 * fluid_u[i+1, j+1, k+1]
        )
        
        return interpolated_u
    
    @ti.func
    def distribute_force_to_grid(self, particle_idx: ti.i32, force: ti.template()):
        """三線性插值分布反作用力到網格 - 路線圖核心算法"""
        pos = self.position[particle_idx]
        
        # 網格索引計算（確保邊界安全）
        i = ti.cast(ti.max(0, ti.min(config.NX-2, pos[0])), ti.i32)
        j = ti.cast(ti.max(0, ti.min(config.NY-2, pos[1])), ti.i32) 
        k = ti.cast(ti.max(0, ti.min(config.NZ-2, pos[2])), ti.i32)
        
        # 插值權重計算
        fx = pos[0] - ti.cast(i, ti.f32)
        fy = pos[1] - ti.cast(j, ti.f32)
        fz = pos[2] - ti.cast(k, ti.f32)
        
        # 限制權重在[0,1]範圍內
        fx = ti.max(0.0, ti.min(1.0, fx))
        fy = ti.max(0.0, ti.min(1.0, fy))
        fz = ti.max(0.0, ti.min(1.0, fz))
        
        # 8個節點權重
        w000 = (1-fx) * (1-fy) * (1-fz)
        w001 = (1-fx) * (1-fy) * fz
        w010 = (1-fx) * fy * (1-fz)
        w011 = (1-fx) * fy * fz
        w100 = fx * (1-fy) * (1-fz)
        w101 = fx * (1-fy) * fz
        w110 = fx * fy * (1-fz)
        w111 = fx * fy * fz
        
        # 原子操作分布力（防止競爭條件）
        ti.atomic_add(self.reaction_force_field[i, j, k], w000 * force)
        ti.atomic_add(self.reaction_force_field[i, j+1, k], w010 * force)
        ti.atomic_add(self.reaction_force_field[i, j, k+1], w001 * force)
        ti.atomic_add(self.reaction_force_field[i, j+1, k+1], w011 * force)
        ti.atomic_add(self.reaction_force_field[i+1, j, k], w100 * force)
        ti.atomic_add(self.reaction_force_field[i+1, j+1, k], w110 * force)
        ti.atomic_add(self.reaction_force_field[i+1, j, k+1], w101 * force)
        ti.atomic_add(self.reaction_force_field[i+1, j+1, k+1], w111 * force)
    
    @ti.func
    def compute_drag_coefficient(self, re_p: ti.f32) -> ti.f32:
        """Reynolds數依賴拖曳係數 - 路線圖完整版（修正Taichi兼容性）"""
        cd = 24.0 / ti.max(0.01, re_p)  # 默認Stokes區域
        
        if re_p >= 0.1 and re_p < 1000.0:
            # Schiller-Naumann修正
            cd = 24.0 / re_p * (1.0 + 0.15 * ti.pow(re_p, 0.687))
        elif re_p >= 1000.0:
            cd = 0.44  # 牛頓阻力區域
            
        return cd
    
    @ti.kernel
    def clear_reaction_forces(self):
        """清零反作用力場"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            self.reaction_force_field[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.kernel
    def compute_two_way_coupling_forces(self, fluid_u: ti.template()):
        """計算雙向耦合力 - 完整版實現"""
        # 清零反作用力場
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            self.reaction_force_field[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        
        # 顆粒循環：計算拖曳力並分布反作用力
        for p in range(self.max_particles):
            if self.active[p]:
                # 1. 插值流體速度到顆粒位置
                u_fluid = self.interpolate_fluid_velocity_from_field(p, fluid_u)
                u_rel = u_fluid - self.velocity[p]
                u_rel_mag = u_rel.norm()
                
                # 存儲流體速度用於診斷
                self.fluid_velocity_at_particle[p] = u_fluid
                
                if u_rel_mag > 1e-8:
                    # 2. 計算顆粒Reynolds數
                    re_p = (self.water_density * u_rel_mag * 2 * self.radius[p] / 
                           ti.max(1e-8, self.water_viscosity))
                    
                    # 存儲Reynolds數用於診斷
                    self.particle_reynolds[p] = re_p
                    
                    # 3. 計算拖曳力
                    cd = self.compute_drag_coefficient(re_p)
                    self.drag_coefficient[p] = cd  # 存儲用於診斷
                    
                    area = 3.14159 * self.radius[p] * self.radius[p]
                    drag_magnitude = 0.5 * self.water_density * cd * area * u_rel_mag
                    
                    # 限制最大拖曳力（數值穩定性）
                    max_drag = self.mass[p] * 100.0  # 100g加速度上限
                    drag_magnitude = ti.min(drag_magnitude, max_drag)
                    
                    # 4. 拖曳力向量
                    self.drag_force_new[p] = drag_magnitude * u_rel / u_rel_mag
                    
                    # 5. 分布反作用力到網格（牛頓第三定律）
                    reaction_force = -self.drag_force_new[p]
                    self.distribute_force_to_grid(p, reaction_force)
                else:
                    # 相對速度太小，清零力
                    self.drag_force_new[p] = ti.Vector([0.0, 0.0, 0.0])
                    self.particle_reynolds[p] = 0.0
                    self.drag_coefficient[p] = 0.0
    
    @ti.func
    def interpolate_fluid_velocity_from_field(self, particle_idx: ti.i32, fluid_u: ti.template()) -> ti.template():
        """從給定的流體速度場插值到顆粒位置"""
        pos = self.position[particle_idx]
        
        # 網格索引計算（確保邊界安全）
        i = ti.cast(ti.max(0, ti.min(config.NX-2, pos[0])), ti.i32)
        j = ti.cast(ti.max(0, ti.min(config.NY-2, pos[1])), ti.i32) 
        k = ti.cast(ti.max(0, ti.min(config.NZ-2, pos[2])), ti.i32)
        
        # 插值權重計算
        fx = pos[0] - ti.cast(i, ti.f32)
        fy = pos[1] - ti.cast(j, ti.f32)
        fz = pos[2] - ti.cast(k, ti.f32)
        
        # 限制權重在[0,1]範圍內
        fx = ti.max(0.0, ti.min(1.0, fx))
        fy = ti.max(0.0, ti.min(1.0, fy))
        fz = ti.max(0.0, ti.min(1.0, fz))
        
        # 8個節點權重
        w000 = (1-fx) * (1-fy) * (1-fz)
        w001 = (1-fx) * (1-fy) * fz
        w010 = (1-fx) * fy * (1-fz)
        w011 = (1-fx) * fy * fz
        w100 = fx * (1-fy) * (1-fz)
        w101 = fx * (1-fy) * fz
        w110 = fx * fy * (1-fz)
        w111 = fx * fy * fz
        
        # 三線性插值計算
        interpolated_u = (
            w000 * fluid_u[i, j, k] +
            w001 * fluid_u[i, j, k+1] +
            w010 * fluid_u[i, j+1, k] +
            w011 * fluid_u[i, j+1, k+1] +
            w100 * fluid_u[i+1, j, k] +
            w101 * fluid_u[i+1, j, k+1] +
            w110 * fluid_u[i+1, j+1, k] +
            w111 * fluid_u[i+1, j+1, k+1]
        )
        
        return interpolated_u
    
    @ti.kernel
    def apply_under_relaxation(self, relaxation_factor: ti.f32):
        """防止數值震蕩的亞鬆弛技術 - 路線圖核心穩定性算法"""
        for p in range(self.max_particles):
            if self.active[p]:
                # 亞鬆弛公式：F_new = α·F_computed + (1-α)·F_old
                self.drag_force[p] = (
                    relaxation_factor * self.drag_force_new[p] + 
                    (1.0 - relaxation_factor) * self.drag_force_old[p]
                )
                
                # 更新歷史值
                self.drag_force_old[p] = self.drag_force[p]
    
    def get_coupling_diagnostics(self) -> Dict[str, Any]:
        """獲取雙向耦合診斷信息"""
        if self.particle_count[None] == 0:
            return {
                'active_particles': 0,
                'avg_reynolds': 0.0,
                'avg_drag_coeff': 0.0,
                'max_reaction_force': 0.0,
                'coupling_quality': 'no_particles'
            }
        
        # 收集診斷數據
        reynolds_values = []
        drag_coeffs = []
        reaction_forces = []
        
        for i in range(self.particle_count[None]):
            if self.active[i] == 1:
                reynolds_values.append(self.particle_reynolds[i])
                drag_coeffs.append(self.drag_coefficient[i])
        
        # 獲取反作用力場統計
        reaction_force_data = self.reaction_force_field.to_numpy()
        max_reaction_force = np.max(np.linalg.norm(reaction_force_data, axis=-1))
        
        return {
            'active_particles': len(reynolds_values),
            'avg_reynolds': np.mean(reynolds_values) if reynolds_values else 0.0,
            'max_reynolds': np.max(reynolds_values) if reynolds_values else 0.0,
            'avg_drag_coeff': np.mean(drag_coeffs) if drag_coeffs else 0.0,
            'max_reaction_force': float(max_reaction_force),
            'coupling_quality': 'active' if reynolds_values else 'inactive'
        }