"""
增強版咖啡顆粒系統 - 完整物理與約束
包含高斯分布、流體作用力和邊界約束
開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import config

@ti.data_oriented
class CoffeeParticleSystem:
    """增強版咖啡顆粒系統 - 包含完整物理與約束"""
    
    def __init__(self, max_particles=15000):
        self.max_particles = max_particles
        print(f"☕ 初始化增強顆粒系統 (max: {max_particles:,})...")
        
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
        
        # 計數器
        self.particle_count = ti.field(dtype=ti.i32, shape=())
        self.active_count = ti.field(dtype=ti.i32, shape=())
        
        # 物理常數
        self.gravity = 9.81
        self.coffee_density = config.COFFEE_BEAN_DENSITY
        self.water_density = config.WATER_DENSITY_90C
        self.water_viscosity = config.WATER_VISCOSITY_90C * config.WATER_DENSITY_90C
        
        print("✅ 增強顆粒系統初始化完成")
    
    @ti.kernel
    def clear_all_particles(self):
        """清空所有顆粒"""
        for i in range(self.max_particles):
            self.position[i] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
            self.settling_velocity[i] = ti.Vector([0.0, 0.0, 0.0])
            self.radius[i] = 0.0
            self.mass[i] = 0.0
            self.active[i] = 0
            self.particle_reynolds[i] = 0.0
        self.particle_count[None] = 0
        self.active_count[None] = 0
    
    def generate_gaussian_particle_radius(self, mean_radius=None, std_dev_ratio=0.3):
        """
        生成高斯分佈的咖啡粉顆粒半徑
        
        Args:
            mean_radius: 平均半徑 (m)，默認使用config中的值
            std_dev_ratio: 標準差相對於均值的比例 (30%)
        
        Returns:
            半徑值 (m)
        """
        if mean_radius is None:
            mean_radius = config.COFFEE_PARTICLE_RADIUS
        
        std_dev = mean_radius * std_dev_ratio
        
        # 生成高斯分佈半徑
        radius = np.random.normal(mean_radius, std_dev)
        
        # 限制半徑範圍：50%-150%的平均值
        min_radius = mean_radius * 0.5
        max_radius = mean_radius * 1.5
        radius = np.clip(radius, min_radius, max_radius)
        
        return radius
    
    @ti.kernel
    def create_particle_with_physics(self, idx: ti.i32, px: ti.f32, py: ti.f32, pz: ti.f32, 
                                   radius: ti.f32, vx: ti.f32, vy: ti.f32, vz: ti.f32):
        """創建帶完整物理屬性的顆粒"""
        if idx < self.max_particles:
            self.position[idx] = ti.Vector([px, py, pz])
            self.velocity[idx] = ti.Vector([vx, vy, vz])
            self.radius[idx] = radius
            
            # 計算質量 (球體體積 × 密度)
            volume = (4.0/3.0) * 3.14159 * radius**3
            self.mass[idx] = volume * self.coffee_density
            
            self.active[idx] = 1
            self.force[idx] = ti.Vector([0.0, 0.0, 0.0])
            self.settling_velocity[idx] = ti.Vector([0.0, 0.0, 0.0])
            self.particle_reynolds[idx] = 0.0
    
    def initialize_coffee_bed_confined(self, filter_paper_system):
        """在錐形濾紙內部生成約束的高斯分布咖啡床"""
        print("🔧 生成錐形約束的高斯分布咖啡床...")
        
        # 清空現有顆粒
        self.clear_all_particles()
        
        # 獲取濾紙邊界信息
        boundary = filter_paper_system.get_coffee_bed_boundary()
        center_x = boundary['center_x']
        center_y = boundary['center_y']
        bottom_z = boundary['bottom_z']
        top_radius_lu = boundary['top_radius_lu']
        bottom_radius_lu = boundary['bottom_radius_lu']
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        
        print(f"   └─ 邊界參數: 中心({center_x:.1f}, {center_y:.1f}), 底部Z={bottom_z:.1f}")
        print(f"   └─ 半徑範圍: {bottom_radius_lu:.1f} -> {top_radius_lu:.1f} lu")
        
        # === 3D分層生成策略 ===
        target_particles = 1000  # 控制顆粒數量便於測試
        coffee_bed_height_lu = 30.0  # 咖啡床高度（格子單位）
        layer_count = 20  # 分成20層
        particles_per_layer = target_particles // layer_count
        
        created = 0
        successful_placements = 0
        
        for layer in range(layer_count):
            if created >= self.max_particles:
                break
                
            # 計算該層的高度和半徑
            layer_height_ratio = layer / layer_count
            z = bottom_z + 1.0 + layer_height_ratio * coffee_bed_height_lu
            
            # 錐形半徑插值
            radius_ratio = min(1.0, (z - bottom_z) / cup_height_lu)
            layer_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * radius_ratio
            effective_radius = layer_radius * 0.8  # 留20%邊距
            
            # 在該層生成顆粒
            for p in range(particles_per_layer):
                if created >= self.max_particles:
                    break
                
                # 隨機極坐標位置
                angle = np.random.uniform(0, 2*np.pi)
                r = np.sqrt(np.random.uniform(0, 1)) * effective_radius  # 使用根號分布確保均勻
                
                x = center_x + r * np.cos(angle)
                y = center_y + r * np.sin(angle)
                
                # 檢查邊界
                if (3 < x < config.NX-3 and 3 < y < config.NY-3 and z < config.NZ-3):
                    # 錐形邊界檢查
                    if self.check_within_cone_boundary(x, y, z, center_x, center_y, 
                                                     bottom_z, bottom_radius_lu, top_radius_lu):
                        # 生成高斯分佈半徑
                        particle_radius = self.generate_gaussian_particle_radius()
                        
                        # 創建顆粒（靜止狀態）
                        self.create_particle_with_physics(created, x, y, z, particle_radius, 0, 0, 0)
                        created += 1
                        successful_placements += 1
            
            if (layer + 1) % 5 == 0:
                print(f"   └─ 完成第 {layer+1}/{layer_count} 層，累計顆粒: {created}")
        
        # 更新計數
        self.particle_count[None] = created
        self.active_count[None] = created
        
        print(f"✅ 咖啡床生成完成:")
        print(f"   └─ 成功生成: {created:,} 顆粒")
        print(f"   └─ 邊界約束成功率: {successful_placements/created*100:.1f}%")
        
        return created
    
    @staticmethod
    def check_within_cone_boundary(x, y, z, center_x, center_y, bottom_z, 
                                 bottom_radius_lu, top_radius_lu):
        """檢查點是否在錐形邊界內"""
        # 計算到軸心的距離
        dx = x - center_x
        dy = y - center_y
        distance_from_center = np.sqrt(dx*dx + dy*dy)
        
        # 計算該高度處的最大允許半徑
        if z <= bottom_z:
            return distance_from_center <= bottom_radius_lu
        
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        height_ratio = min(1.0, (z - bottom_z) / cup_height_lu)
        max_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
        
        return distance_from_center <= max_radius * 0.95  # 留5%安全邊距
    
    @ti.kernel
    def enforce_filter_boundary(self, center_x: ti.f32, center_y: ti.f32, bottom_z: ti.f32,
                               bottom_radius_lu: ti.f32, top_radius_lu: ti.f32) -> ti.i32:
        """強制執行濾紙邊界約束 - 返回違反邊界的顆粒數"""
        violations = 0
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        
        for i in range(self.max_particles):
            if self.active[i] == 1:
                pos = self.position[i]
                
                # 計算到軸心的距離
                dx = pos.x - center_x
                dy = pos.y - center_y
                distance_from_center = ti.sqrt(dx*dx + dy*dy)
                
                # 計算該高度的最大允許半徑
                max_radius = bottom_radius_lu  # 默認底部半徑
                if pos.z > bottom_z:
                    height_ratio = ti.min(1.0, (pos.z - bottom_z) / cup_height_lu)
                    max_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                
                # 邊界違反檢查
                if distance_from_center > max_radius * 0.95:
                    # 推回邊界內
                    scale_factor = max_radius * 0.9 / distance_from_center
                    self.position[i].x = center_x + dx * scale_factor
                    self.position[i].y = center_y + dy * scale_factor
                    
                    # 清零速度（避免進一步移動出邊界）
                    self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
                    violations += 1
                
                # 底部邊界
                if pos.z < bottom_z:
                    self.position[i].z = bottom_z + 0.1
                    self.velocity[i].z = 0.0
                    violations += 1
        
        return violations
    
    @ti.kernel
    def apply_fluid_forces(self, fluid_u: ti.template(), fluid_v: ti.template(), 
                          fluid_w: ti.template(), fluid_density: ti.template(), 
                          pressure: ti.template(), dt: ti.f32):
        """應用簡化的流體作用力到顆粒"""
        for i in range(self.max_particles):
            if self.active[i] == 1:
                pos = self.position[i]
                
                # 獲取網格索引
                grid_i = ti.cast(pos.x, ti.i32)
                grid_j = ti.cast(pos.y, ti.i32)
                grid_k = ti.cast(pos.z, ti.i32)
                
                # 邊界檢查
                if (grid_i >= 0 and grid_i < config.NX-1 and 
                    grid_j >= 0 and grid_j < config.NY-1 and 
                    grid_k >= 0 and grid_k < config.NZ-1):
                    
                    # 直接使用向量場 - LBM的u場是向量場
                    fluid_velocity = fluid_u[grid_i, grid_j, grid_k]
                    
                    particle_vel = self.velocity[i]
                    relative_vel = fluid_velocity - particle_vel
                    relative_speed = relative_vel.norm()
                    
                    if relative_speed > 1e-6:
                        radius = self.radius[i]
                        mass = self.mass[i]
                        
                        # === 1. 簡化阻力 (Stokes) ===
                        # 阻力係數
                        Re_p = relative_speed * 2.0 * radius * self.water_density / self.water_viscosity
                        C_D = 24.0 / ti.max(1.0, Re_p)
                        
                        # 阻力
                        drag_force = 0.5 * C_D * 3.14159 * radius**2 * self.water_density * \
                                    relative_speed * relative_vel
                        
                        # === 2. 浮力 ===
                        volume = (4.0/3.0) * 3.14159 * radius**3
                        buoyancy = volume * self.water_density * self.gravity * ti.Vector([0, 0, 1])
                        
                        # === 3. 重力 ===
                        gravity_force = mass * self.gravity * ti.Vector([0, 0, -1])
                        
                        # === 總力 ===
                        total_force = drag_force + buoyancy + gravity_force
                        
                        # 更新顆粒力
                        self.force[i] = total_force
    
    @ti.kernel
    def update_particle_physics(self, dt: ti.f32, center_x: ti.f32, center_y: ti.f32, 
                               bottom_z: ti.f32, bottom_radius_lu: ti.f32, top_radius_lu: ti.f32):
        """更新顆粒物理（集成力、更新位置速度、邊界約束）"""
        for i in range(self.max_particles):
            if self.active[i] == 1:
                # 力積分更新速度
                if self.mass[i] > 0:
                    acceleration = self.force[i] / self.mass[i]
                    self.velocity[i] += acceleration * dt
                
                # 位置更新
                old_pos = self.position[i]
                new_pos = old_pos + self.velocity[i] * dt
                
                # 邊界約束檢查
                if self.check_particle_boundary_violation(new_pos, center_x, center_y, 
                                                        bottom_z, bottom_radius_lu, top_radius_lu):
                    # 違反邊界，約束回內部
                    new_pos = self.constrain_to_boundary(new_pos, center_x, center_y, 
                                                       bottom_z, bottom_radius_lu, top_radius_lu)
                    # 阻尼速度
                    self.velocity[i] *= 0.5
                
                self.position[i] = new_pos
                
                # 清零力（為下一步準備）
                self.force[i] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.func
    def check_particle_boundary_violation(self, pos, center_x: ti.f32, center_y: ti.f32,
                                        bottom_z: ti.f32, bottom_radius_lu: ti.f32, 
                                        top_radius_lu: ti.f32) -> ti.i32:
        """檢查顆粒是否違反邊界"""
        violation = 0
        
        # 底部檢查
        if pos.z < bottom_z:
            violation = 1
        else:
            # 錐形側壁檢查
            dx = pos.x - center_x
            dy = pos.y - center_y
            distance_from_center = ti.sqrt(dx*dx + dy*dy)
            
            cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
            height_ratio = ti.min(1.0, (pos.z - bottom_z) / cup_height_lu)
            max_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            
            if distance_from_center > max_radius * 0.95:
                violation = 1
        
        return violation
    
    @ti.func
    def constrain_to_boundary(self, pos, center_x: ti.f32, center_y: ti.f32,
                            bottom_z: ti.f32, bottom_radius_lu: ti.f32, 
                            top_radius_lu: ti.f32):
        """將位置約束到邊界內"""
        constrained_pos = pos
        
        # 底部約束
        if constrained_pos.z < bottom_z:
            constrained_pos.z = bottom_z + 0.1
        
        # 錐形側壁約束
        dx = constrained_pos.x - center_x
        dy = constrained_pos.y - center_y
        distance_from_center = ti.sqrt(dx*dx + dy*dy)
        
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        height_ratio = ti.min(1.0, (constrained_pos.z - bottom_z) / cup_height_lu)
        max_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
        
        if distance_from_center > max_radius * 0.9:
            scale_factor = max_radius * 0.9 / distance_from_center
            constrained_pos.x = center_x + dx * scale_factor
            constrained_pos.y = center_y + dy * scale_factor
        
        return constrained_pos
    
    def get_particle_statistics(self):
        """獲取顆粒系統統計信息"""
        radii = []
        positions = []
        
        for i in range(self.particle_count[None]):
            if self.active[i] == 1:
                radii.append(self.radius[i])
                pos = self.position[i]
                positions.append([pos[0], pos[1], pos[2]])
        
        radii = np.array(radii)
        positions = np.array(positions)
        
        return {
            'count': len(radii),
            'mean_radius': np.mean(radii) if len(radii) > 0 else 0,
            'std_radius': np.std(radii) if len(radii) > 0 else 0,
            'min_radius': np.min(radii) if len(radii) > 0 else 0,
            'max_radius': np.max(radii) if len(radii) > 0 else 0,
            'positions': positions,
            'radii': radii
        }