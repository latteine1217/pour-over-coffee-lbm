# coffee_particles.py
"""
可移動咖啡粉粒子系統
實現真實的咖啡粉顆粒在水流中的運動和相互作用
"""

import taichi as ti
import numpy as np
import config

@ti.data_oriented
class CoffeeParticleSystem:
    def __init__(self, max_particles=50000):
        self.max_particles = max_particles
        
        # 粒子基本屬性
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)     # 位置
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)     # 速度
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)        # 受力
        
        # 粒子物理屬性
        self.mass = ti.field(dtype=ti.f32, shape=max_particles)                   # 質量
        self.radius = ti.field(dtype=ti.f32, shape=max_particles)                 # 半徑
        self.density = ti.field(dtype=ti.f32, shape=max_particles)                # 密度
        self.active = ti.field(dtype=ti.i32, shape=max_particles)                 # 是否活躍
        
        # 咖啡萃取屬性
        self.extraction_state = ti.field(dtype=ti.f32, shape=max_particles)       # 萃取程度 [0,1]
        self.surface_area = ti.field(dtype=ti.f32, shape=max_particles)           # 表面積
        
        # 物理常數
        self.COFFEE_DENSITY = 1200.0      # 咖啡粉密度 kg/m³
        self.DRAG_COEFFICIENT = 0.47      # 阻力係數
        self.RESTITUTION = 0.3            # 恢復係數
        self.FRICTION = 0.6               # 摩擦係數
        self.GRAVITY = ti.Vector([0.0, 0.0, -9.81])  # 重力加速度
        
        # 粒子計數
        self.particle_count = ti.field(dtype=ti.i32, shape=())
        
        print(f"咖啡粒子系統初始化完成 - 最大粒子數: {max_particles:,}")
    
    @ti.kernel
    def initialize_coffee_bed(self, bed_height: ti.f32, bed_top_radius: ti.f32, 
                             center_x: ti.f32, center_y: ti.f32, bottom_z: ti.f32):
        """初始化咖啡床 - 在V60錐形內部正確分佈咖啡粉"""
        particle_idx = 0
        
        # 計算需要的粒子數和層數 (性能優化版)
        # 原始: 150粒子/層 = 7,500總粒子
        # 快速: 75粒子/層 = 3,750總粒子 (2倍加速)
        # 測試: 30粒子/層 = 1,500總粒子 (5倍加速)
        particles_per_layer = 75  # 平衡模式
        bed_height_lu = bed_height / config.SCALE_LENGTH
        layers = ti.max(5, ti.min(50, int(bed_height_lu * 10)))  # 限制層數在5-50之間
        
        # V60錐形參數 (物理單位到格子單位轉換)
        bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
        bed_top_radius_lu = bed_top_radius / config.SCALE_LENGTH
        
        for layer in range(layers):
            z_pos = bottom_z + (layer + 0.5) * bed_height_lu / layers
            height_ratio = (layer + 0.5) / layers  # 當前層在咖啡床中的相對高度
            
            # 根據V60錐形計算當前層的允許半徑
            current_radius_lu = bottom_radius_lu + (bed_top_radius_lu - bottom_radius_lu) * height_ratio
            
            # 每層隨機分佈粒子，確保在錐形內部
            for i in range(particles_per_layer):
                if particle_idx < self.max_particles:  # 使用條件而不是break
                    # 在當前層的圓形區域內隨機分佈
                    r = ti.sqrt(ti.random()) * current_radius_lu * 0.9  # 留10%安全邊距
                    theta = ti.random() * 2.0 * 3.14159265
                    
                    x_pos = center_x + r * ti.cos(theta)
                    y_pos = center_y + r * ti.sin(theta)
                    
                    # 檢查是否在有效範圍內
                    if (1 < x_pos < config.NX-1 and 1 < y_pos < config.NY-1 and 
                        1 < z_pos < config.NZ-1):
                        
                        # 設置粒子屬性
                        self.position[particle_idx] = ti.Vector([x_pos, y_pos, z_pos])
                        self.velocity[particle_idx] = ti.Vector([0.0, 0.0, 0.0])
                        self.force[particle_idx] = ti.Vector([0.0, 0.0, 0.0])
                        
                        # 物理屬性 (隨機變化模擬不同大小的咖啡粉)
                        particle_radius = 0.3 + ti.random() * 0.4  # 0.3-0.7 網格單位
                        self.radius[particle_idx] = particle_radius
                        self.density[particle_idx] = self.COFFEE_DENSITY * (0.8 + ti.random() * 0.4)
                        
                        particle_volume = 4.0/3.0 * 3.14159265 * particle_radius**3
                        self.mass[particle_idx] = self.density[particle_idx] * particle_volume
                        self.surface_area[particle_idx] = 4.0 * 3.14159265 * particle_radius**2
                        
                        # 萃取狀態
                        self.extraction_state[particle_idx] = 0.0
                        self.active[particle_idx] = 1
                        
                        particle_idx += 1
        
        self.particle_count[None] = min(particle_idx, self.max_particles)
    
    def initialize_coffee_bed_with_info(self, bed_height, bed_top_radius, center_x, center_y, bottom_z):
        """帶信息打印的咖啡床初始化包裝方法"""
        # 計算層數用於信息顯示
        bed_height_lu = bed_height / config.SCALE_LENGTH
        layers = max(5, min(50, int(bed_height_lu * 10)))  # 限制層數在5-50之間
        particles_per_layer = 75  # 平衡模式
        
        print(f"生成咖啡床: {layers}層, 每層{particles_per_layer}顆粒")
        
        # 調用實際的kernel
        self.initialize_coffee_bed(bed_height, bed_top_radius, center_x, center_y, bottom_z)
        
        # 打印結果
        print(f"咖啡床初始化完成 - 生成粒子數: {self.particle_count[None]}")
        print(f"咖啡床高度: {bed_height*100:.1f}cm, 頂部半徑: {bed_top_radius*100:.1f}cm")
    
    @ti.kernel
    def compute_fluid_forces(self, fluid_u: ti.template(), fluid_rho: ti.template()):
        """計算流體對粒子的作用力"""
        for i in range(self.particle_count[None]):
            if self.active[i] == 0:
                continue
            
            pos = self.position[i]
            vel = self.velocity[i]
            
            # 獲取粒子位置的網格索引
            gi, gj, gk = int(pos.x), int(pos.y), int(pos.z)
            
            if 0 <= gi < config.NX-1 and 0 <= gj < config.NY-1 and 0 <= gk < config.NZ-1:
                # 三線性插值獲取流體速度
                fluid_velocity = self._interpolate_velocity(fluid_u, pos)
                fluid_density = self._interpolate_scalar(fluid_rho, pos)
                
                # 相對速度
                relative_velocity = fluid_velocity - vel
                relative_speed = relative_velocity.norm()
                
                if relative_speed > 1e-6:
                    # 阻力
                    drag_force = (0.5 * fluid_density * self.DRAG_COEFFICIENT * 
                                 self.surface_area[i] * relative_speed * relative_velocity)
                    
                    # 浮力 (阿基米德原理)
                    particle_volume = 4.0/3.0 * 3.14159265 * self.radius[i]**3
                    buoyancy = fluid_density * particle_volume * (-self.GRAVITY)
                    
                    # 更新力
                    self.force[i] = drag_force + buoyancy + self.mass[i] * self.GRAVITY
                else:
                    self.force[i] = self.mass[i] * self.GRAVITY
    
    @ti.func
    def _interpolate_velocity(self, field, pos):
        """三線性插值流體速度"""
        x, y, z = pos.x, pos.y, pos.z
        i, j, k = int(x), int(y), int(z)
        fx, fy, fz = x - i, y - j, z - k
        
        # 邊界檢查
        i = ti.max(0, ti.min(config.NX-2, i))
        j = ti.max(0, ti.min(config.NY-2, j))
        k = ti.max(0, ti.min(config.NZ-2, k))
        
        # 三線性插值
        c000 = field[i, j, k]
        c001 = field[i, j, k+1]
        c010 = field[i, j+1, k]
        c011 = field[i, j+1, k+1]
        c100 = field[i+1, j, k]
        c101 = field[i+1, j, k+1]
        c110 = field[i+1, j+1, k]
        c111 = field[i+1, j+1, k+1]
        
        c00 = c000 * (1-fz) + c001 * fz
        c01 = c010 * (1-fz) + c011 * fz
        c10 = c100 * (1-fz) + c101 * fz
        c11 = c110 * (1-fz) + c111 * fz
        
        c0 = c00 * (1-fy) + c01 * fy
        c1 = c10 * (1-fy) + c11 * fy
        
        return c0 * (1-fx) + c1 * fx
    
    @ti.func
    def _interpolate_scalar(self, field, pos):
        """三線性插值標量場"""
        x, y, z = pos.x, pos.y, pos.z
        i, j, k = int(x), int(y), int(z)
        fx, fy, fz = x - i, y - j, z - k
        
        i = ti.max(0, ti.min(config.NX-2, i))
        j = ti.max(0, ti.min(config.NY-2, j))
        k = ti.max(0, ti.min(config.NZ-2, k))
        
        c000 = field[i, j, k]
        c001 = field[i, j, k+1]
        c010 = field[i, j+1, k]
        c011 = field[i, j+1, k+1]
        c100 = field[i+1, j, k]
        c101 = field[i+1, j, k+1]
        c110 = field[i+1, j+1, k]
        c111 = field[i+1, j+1, k+1]
        
        c00 = c000 * (1-fz) + c001 * fz
        c01 = c010 * (1-fz) + c011 * fz
        c10 = c100 * (1-fz) + c101 * fz
        c11 = c110 * (1-fz) + c111 * fz
        
        c0 = c00 * (1-fy) + c01 * fy
        c1 = c10 * (1-fy) + c11 * fy
        
        return c0 * (1-fx) + c1 * fx
    
    @ti.kernel
    def update_particles(self, dt: ti.f32):
        """更新粒子位置和速度"""
        for i in range(self.particle_count[None]):
            if self.active[i] == 0:
                continue
            
            # 牛頓第二定律 F = ma
            acceleration = self.force[i] / self.mass[i]
            
            # 速度更新 (Velocity Verlet)
            self.velocity[i] += acceleration * dt
            
            # 速度阻尼 (防止不穩定)
            damping = 0.99
            self.velocity[i] *= damping
            
            # 位置更新
            new_position = self.position[i] + self.velocity[i] * dt
            
            # 邊界條件處理
            new_position = self._handle_boundary_collision(new_position, self.velocity[i], i)
            
            self.position[i] = new_position
    
    @ti.func
    def _handle_boundary_collision(self, pos, vel, particle_idx):
        """處理邊界碰撞"""
        new_pos = pos
        new_vel = vel
        radius = self.radius[particle_idx]
        
        # X邊界
        if new_pos.x < radius:
            new_pos.x = radius
            new_vel.x = -new_vel.x * self.RESTITUTION
        elif new_pos.x > config.NX - radius:
            new_pos.x = config.NX - radius
            new_vel.x = -new_vel.x * self.RESTITUTION
        
        # Y邊界
        if new_pos.y < radius:
            new_pos.y = radius
            new_vel.y = -new_vel.y * self.RESTITUTION
        elif new_pos.y > config.NY - radius:
            new_pos.y = config.NY - radius
            new_vel.y = -new_vel.y * self.RESTITUTION
        
        # Z邊界 (底部固體邊界)
        if new_pos.z < radius:
            new_pos.z = radius
            new_vel.z = -new_vel.z * self.RESTITUTION
            # 摩擦力
            friction_force = self.FRICTION * ti.abs(new_vel.z)
            new_vel.x *= ti.max(0.0, 1.0 - friction_force)
            new_vel.y *= ti.max(0.0, 1.0 - friction_force)
        elif new_pos.z > config.NZ - radius:
            new_pos.z = config.NZ - radius
            new_vel.z = -new_vel.z * self.RESTITUTION
        
        # 更新速度
        self.velocity[particle_idx] = new_vel
        
        return new_pos
    
    @ti.kernel
    def update_porosity_field(self, porous_field: ti.template()):
        """根據粒子分佈更新多孔度場"""
        # 先重置為1.0 (完全空隙)
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            porous_field[i, j, k] = 1.0
        
        # 根據粒子位置減少孔隙率
        for p in range(self.particle_count[None]):
            if self.active[p] == 0:
                continue
            
            pos = self.position[p]
            radius = self.radius[p]
            
            # 影響周圍網格點
            min_i = ti.max(0, int(pos.x - radius))
            max_i = ti.min(config.NX-1, int(pos.x + radius) + 1)
            min_j = ti.max(0, int(pos.y - radius))
            max_j = ti.min(config.NY-1, int(pos.y + radius) + 1)
            min_k = ti.max(0, int(pos.z - radius))
            max_k = ti.min(config.NZ-1, int(pos.z + radius) + 1)
            
            for i in range(min_i, max_i):
                for j in range(min_j, max_j):
                    for k in range(min_k, max_k):
                        grid_pos = ti.Vector([i + 0.5, j + 0.5, k + 0.5])
                        distance = (grid_pos - pos).norm()
                        
                        if distance < radius:
                            # 根據距離計算孔隙率減少
                            influence = 1.0 - distance / radius
                            porosity_reduction = influence * 0.8  # 最大減少80%
                            porous_field[i, j, k] = ti.max(0.1, 
                                                         porous_field[i, j, k] - porosity_reduction)
    
    @ti.kernel
    def compute_particle_collisions(self):
        """計算顆粒間碰撞力 - 使用空間哈希優化"""
        # 清零碰撞力
        for i in range(self.particle_count[None]):
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
        
        # O(N²) 碰撞檢測 - 對於合理數量的粒子是可接受的
        for i in range(self.particle_count[None]):
            if self.active[i] == 0:
                continue
                
            for j in range(i + 1, self.particle_count[None]):
                if self.active[j] == 0:
                    continue
                
                pos_i = self.position[i]
                pos_j = self.position[j]
                radius_i = self.radius[i]
                radius_j = self.radius[j]
                
                # 計算距離
                distance_vec = pos_i - pos_j
                distance = distance_vec.norm()
                contact_distance = radius_i + radius_j
                
                # 檢查是否有碰撞
                if distance < contact_distance and distance > 1e-6:
                    # 歸一化距離向量
                    normal = distance_vec / distance
                    
                    # 重疊量
                    overlap = contact_distance - distance
                    
                    # Hertz接觸力模型
                    effective_radius = (radius_i * radius_j) / (radius_i + radius_j)
                    contact_stiffness = 1e3  # 接觸剛度
                    normal_force_mag = contact_stiffness * ti.sqrt(effective_radius * overlap**3)
                    
                    # 阻尼力
                    relative_velocity = self.velocity[i] - self.velocity[j]
                    normal_velocity = relative_velocity.dot(normal)
                    damping_coeff = 0.3
                    damping_force_mag = damping_coeff * normal_velocity
                    
                    # 總法向力
                    total_normal_force = (normal_force_mag - damping_force_mag) * normal
                    
                    # 切向摩擦力
                    tangent_velocity = relative_velocity - normal_velocity * normal
                    tangent_speed = tangent_velocity.norm()
                    
                    # 初始化摩擦力
                    friction_force = ti.Vector([0.0, 0.0, 0.0])
                    if tangent_speed > 1e-6:
                        friction_force = -self.FRICTION * normal_force_mag * (tangent_velocity / tangent_speed)
                    
                    # 總力
                    total_force = total_normal_force + friction_force
                    
                    # 根據質量分配力
                    mass_i = self.mass[i]
                    mass_j = self.mass[j]
                    total_mass = mass_i + mass_j
                    
                    self.force[i] += total_force * (mass_j / total_mass)
                    self.force[j] -= total_force * (mass_i / total_mass)

    @ti.kernel
    def apply_fluid_body_force(self, lbm_u: ti.template(), lbm_rho: ti.template(), 
                              lbm_force: ti.template()):
        """將顆粒對流體的反作用力施加到LBM格點上 (IBM方法)"""
        # 清零流體體力場
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            lbm_force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        
        # 將顆粒力分佈到周圍流體格點
        for p in range(self.particle_count[None]):
            if self.active[p] == 0:
                continue
            
            pos = self.position[p]
            radius = self.radius[p]
            particle_force = self.force[p]
            
            # 影響範圍：顆粒半徑的2倍
            influence_radius = radius * 2.0
            
            min_i = ti.max(0, int(pos.x - influence_radius))
            max_i = ti.min(config.NX-1, int(pos.x + influence_radius) + 1)
            min_j = ti.max(0, int(pos.y - influence_radius))
            max_j = ti.min(config.NY-1, int(pos.y + influence_radius) + 1)
            min_k = ti.max(0, int(pos.z - influence_radius))
            max_k = ti.min(config.NZ-1, int(pos.z + influence_radius) + 1)
            
            for i in range(min_i, max_i):
                for j in range(min_j, max_j):
                    for k in range(min_k, max_k):
                        grid_pos = ti.Vector([i + 0.5, j + 0.5, k + 0.5])
                        distance = (grid_pos - pos).norm()
                        
                        if distance < influence_radius:
                            # 使用高斯分佈函數分佈力
                            weight = ti.exp(-(distance / radius)**2)
                            distributed_force = particle_force * weight / (4.0 * 3.14159 * radius**2)
                            
                            # 添加到流體體力場 (注意：是反向力)
                            lbm_force[i, j, k] -= distributed_force

    @ti.kernel
    def update_dynamic_porosity(self, porosity_field: ti.template(), 
                               permeability_field: ti.template()):
        """動態更新孔隙率和滲透率場"""
        # 初始化為完全多孔
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            porosity_field[i, j, k] = 1.0
            permeability_field[i, j, k] = 1e-3  # 基礎滲透率
        
        # 根據顆粒分佈計算局部孔隙率
        for p in range(self.particle_count[None]):
            if self.active[p] == 0:
                continue
            
            pos = self.position[p]
            radius = self.radius[p]
            
            # 影響範圍
            influence_radius = radius * 1.5
            
            min_i = ti.max(0, int(pos.x - influence_radius))
            max_i = ti.min(config.NX-1, int(pos.x + influence_radius) + 1)
            min_j = ti.max(0, int(pos.y - influence_radius))
            max_j = ti.min(config.NY-1, int(pos.y + influence_radius) + 1)
            min_k = ti.max(0, int(pos.z - influence_radius))
            max_k = ti.min(config.NZ-1, int(pos.z + influence_radius) + 1)
            
            for i in range(min_i, max_i):
                for j in range(min_j, max_j):
                    for k in range(min_k, max_k):
                        grid_pos = ti.Vector([i + 0.5, j + 0.5, k + 0.5])
                        distance = (grid_pos - pos).norm()
                        
                        if distance < radius:
                            # 顆粒內部：孔隙率為0
                            porosity_field[i, j, k] = 0.05
                            permeability_field[i, j, k] = 1e-8  # 幾乎不透水
                        elif distance < influence_radius:
                            # 影響區域：根據距離計算孔隙率
                            influence = 1.0 - (distance - radius) / (influence_radius - radius)
                            porosity_reduction = influence * 0.6  # 最大減少60%
                            porosity_field[i, j, k] = ti.max(0.2, 
                                                            porosity_field[i, j, k] - porosity_reduction)
                            
                            # Kozeny-Carman方程計算滲透率
                            porosity = porosity_field[i, j, k]
                            particle_diameter = radius * 2.0 * config.SCALE_LENGTH  # 轉換為物理單位
                            if porosity > 0.2:
                                kozeny_constant = 180.0
                                permeability = (particle_diameter**2 * porosity**3) / (kozeny_constant * (1-porosity)**2)
                                permeability_field[i, j, k] = ti.max(1e-8, permeability)

    @ti.kernel
    def update_extraction(self, dt: ti.f32, fluid_velocity: ti.template(), 
                         fluid_temperature: ti.f32):
        """更新咖啡萃取過程 - 考慮顆粒移動和變形"""
        for i in range(self.particle_count[None]):
            if self.active[i] == 0:
                continue
            
            pos = self.position[i]
            
            # 獲取流體速度
            fluid_vel = self._interpolate_velocity(fluid_velocity, pos)
            flow_speed = fluid_vel.norm()
            
            # 萃取速率取決於：流速、表面積、溫度、當前萃取狀態
            temperature_factor = 1.0 + (fluid_temperature - 20.0) / 100.0  # 溫度效應
            surface_factor = self.surface_area[i]
            extraction_efficiency = 1.0 - self.extraction_state[i]  # 已萃取部分效率降低
            
            extraction_rate = (0.05 * flow_speed * surface_factor * 
                             temperature_factor * extraction_efficiency * dt)
            
            # 更新萃取狀態
            new_extraction = self.extraction_state[i] + extraction_rate
            self.extraction_state[i] = ti.min(1.0, new_extraction)
            
            # 顆粒物理變化：萃取導致體積減小和密度變化
            if self.extraction_state[i] > 0.1:
                # 體積收縮 (最大收縮30%)
                shrink_factor = 1.0 - 0.3 * self.extraction_state[i]
                new_radius = self.radius[i] * (0.99 + 0.01 * shrink_factor)
                self.radius[i] = ti.max(0.1, new_radius)  # 防止過小
                
                # 重新計算表面積和質量
                self.surface_area[i] = 4.0 * 3.14159265 * self.radius[i]**2
                
                # 密度變化：萃取後變輕
                density_reduction = self.extraction_state[i] * 0.2
                new_density = self.COFFEE_DENSITY * (1.0 - density_reduction)
                self.density[i] = ti.max(800.0, new_density)  # 最低密度限制
                
                # 重新計算質量
                particle_volume = 4.0/3.0 * 3.14159265 * self.radius[i]**3
                self.mass[i] = self.density[i] * particle_volume
            
            # 高度萃取的顆粒更容易被沖散
            if self.extraction_state[i] > 0.8 and flow_speed > 0.01:
                # 增加對流體力的敏感性
                self.mass[i] *= 0.8  # 有效質量減少，更容易移動

    @ti.kernel  
    def particle_clustering_effects(self):
        """模擬顆粒聚集和分散效應"""
        for i in range(self.particle_count[None]):
            if self.active[i] == 0:
                continue
            
            pos_i = self.position[i]
            cluster_force = ti.Vector([0.0, 0.0, 0.0])
            neighbor_count = 0
            
            # 檢查周圍顆粒
            for j in range(self.particle_count[None]):
                if i == j or self.active[j] == 0:
                    continue
                
                pos_j = self.position[j]
                distance_vec = pos_i - pos_j
                distance = distance_vec.norm()
                interaction_range = (self.radius[i] + self.radius[j]) * 3.0
                
                if distance < interaction_range and distance > 1e-6:
                    neighbor_count += 1
                    direction = distance_vec / distance
                    
                    # 范德華吸引力 (短程)
                    if distance < interaction_range * 0.5:
                        attraction_strength = 50.0 / (distance**2 + 1.0)
                        cluster_force -= direction * attraction_strength
                    
                    # 排斥力 (防止重疊)
                    contact_distance = self.radius[i] + self.radius[j]
                    if distance < contact_distance * 1.1:
                        repulsion_strength = 100.0 / (distance + 0.1)
                        cluster_force += direction * repulsion_strength
            
            # 根據鄰居數量調整聚集傾向
            if neighbor_count > 3:
                # 高密度區域：增加穩定性
                self.mass[i] *= 1.05  # 稍微增加有效質量
            elif neighbor_count < 2:
                # 孤立顆粒：更容易被沖散
                self.mass[i] *= 0.95
            
            # 應用聚集力
            self.force[i] += cluster_force * 0.1
    
    def step_particle_physics(self, dt, lbm_solver):
        """執行完整的顆粒物理時間步長"""
        # 1. 計算流體對顆粒的作用力
        self.compute_fluid_forces(lbm_solver.u, lbm_solver.rho)
        
        # 2. 計算顆粒間碰撞力
        self.compute_particle_collisions()
        
        # 3. 計算聚集/分散效應
        self.particle_clustering_effects()
        
        # 4. 更新顆粒位置和速度
        self.update_particles(dt)
        
        # 5. 更新萃取狀態
        self.update_extraction(dt, lbm_solver.u, 90.0)  # 90°C水溫
        
        # 6. 將顆粒力反饋給流體 (雙向耦合，無需多孔介質)
        if hasattr(lbm_solver, 'body_force'):
            self.apply_fluid_body_force(lbm_solver.u, lbm_solver.rho, lbm_solver.body_force)
    
    def get_detailed_statistics(self):
        """獲取詳細的顆粒系統統計信息"""
        stats = {}
        active_particles = []
        extraction_levels = []
        sizes = []
        speeds = []
        
        # 獲取Taichi field數據
        active_data = self.active.to_numpy()
        extraction_data = self.extraction_state.to_numpy()
        radius_data = self.radius.to_numpy()
        velocity_data = self.velocity.to_numpy()
        
        for i in range(min(self.particle_count[None], self.max_particles)):
            if active_data[i] == 1:
                active_particles.append(i)
                extraction_levels.append(extraction_data[i])
                sizes.append(radius_data[i])
                speeds.append(np.linalg.norm(velocity_data[i]))
        
        if active_particles:
            stats = {
                'active_particles': len(active_particles),
                'total_particles': self.particle_count[None],
                'average_extraction': np.mean(extraction_levels),
                'max_extraction': np.max(extraction_levels),
                'min_extraction': np.min(extraction_levels),
                'average_size': np.mean(sizes),
                'average_speed': np.mean(speeds),
                'max_speed': np.max(speeds),
                'extraction_distribution': {
                    'low': len([x for x in extraction_levels if x < 0.3]),
                    'medium': len([x for x in extraction_levels if 0.3 <= x < 0.7]),
                    'high': len([x for x in extraction_levels if x >= 0.7])
                }
            }
        else:
            stats = {
                'active_particles': 0,
                'total_particles': self.particle_count[None],
                'average_extraction': 0.0,
                'max_extraction': 0.0,
                'min_extraction': 0.0,
                'average_size': 0.0,
                'average_speed': 0.0,
                'max_speed': 0.0,
                'extraction_distribution': {'low': 0, 'medium': 0, 'high': 0}
            }
        
        return stats