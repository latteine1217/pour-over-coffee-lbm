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
    def initialize_coffee_bed(self, bed_height: ti.f32, bed_radius: ti.f32, 
                             center_x: ti.f32, center_y: ti.f32, bottom_z: ti.f32):
        """初始化咖啡床 - 在V60底部區域隨機分佈咖啡粉"""
        particle_idx = 0
        
        # 計算需要的粒子數 (基於體積和粒子大小)
        particles_per_layer = 200
        layers = int(bed_height * config.NZ / 10.0)  # 每10個網格點一層
        
        for layer in range(layers):
            z_pos = bottom_z + (layer + 0.5) * bed_height / layers
            
            # 每層隨機分佈粒子
            for i in range(particles_per_layer):
                if particle_idx >= self.max_particles:
                    break
                
                # 在圓形區域內隨機分佈
                r = ti.sqrt(ti.random()) * bed_radius * (1.0 - layer / layers * 0.3)  # 上層半徑更小
                theta = ti.random() * 2.0 * 3.14159265
                
                x_pos = center_x + r * ti.cos(theta)
                y_pos = center_y + r * ti.sin(theta)
                
                # 檢查是否在有效範圍內
                if (0 < x_pos < config.NX and 0 < y_pos < config.NY and 
                    0 < z_pos < config.NZ):
                    
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
        
        self.particle_count[None] = particle_idx
        print(f"咖啡床初始化完成 - 生成粒子數: {particle_idx}")
    
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
    def update_extraction(self, dt: ti.f32, fluid_velocity: ti.template()):
        """更新咖啡萃取過程"""
        for i in range(self.particle_count[None]):
            if self.active[i] == 0:
                continue
            
            pos = self.position[i]
            
            # 獲取流體速度大小
            fluid_vel = self._interpolate_velocity(fluid_velocity, pos)
            flow_speed = fluid_vel.norm()
            
            # 萃取速率與流速和表面積成正比
            extraction_rate = 0.1 * flow_speed * self.surface_area[i] * dt
            
            # 更新萃取狀態
            self.extraction_state[i] = ti.min(1.0, self.extraction_state[i] + extraction_rate)
            
            # 粒子萃取完全後變小 (物理上更真實)
            if self.extraction_state[i] > 0.8:
                shrink_factor = 1.0 - 0.2 * self.extraction_state[i]
                self.radius[i] *= shrink_factor
                self.surface_area[i] = 4.0 * 3.14159265 * self.radius[i]**2
    
    def get_statistics(self):
        """獲取粒子系統統計信息"""
        active_count = 0
        avg_extraction = 0.0
        
        for i in range(self.particle_count[None]):
            if self.active[i]:
                active_count += 1
                avg_extraction += self.extraction_state[i]
        
        if active_count > 0:
            avg_extraction /= active_count
        
        return {
            'active_particles': active_count,
            'total_particles': self.particle_count[None],
            'average_extraction': avg_extraction
        }