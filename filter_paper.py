# filter_paper.py
"""
V60濾紙系統 - 純顆粒相容的濾紙物理模型
實現濾紙的透水性、顆粒阻擋、和動態阻力調節
"""

import taichi as ti
import numpy as np
import config

@ti.data_oriented
class FilterPaperSystem:
    def __init__(self, lbm_solver):
        """
        初始化V60濾紙系統
        
        Args:
            lbm_solver: LBM求解器實例
        """
        self.lbm = lbm_solver
        
        # 濾紙物理參數
        self.PAPER_THICKNESS = 0.0001      # 濾紙厚度 0.1mm (真實V60濾紙)
        self.PAPER_POROSITY = 0.85         # 濾紙孔隙率 85% (紙質多孔)
        self.PAPER_PORE_SIZE = 20e-6       # 濾紙孔徑 20微米 (V60濾紙標準)
        self.PAPER_PERMEABILITY = 1e-12    # 濾紙滲透率 (m²)
        
        # 濾紙區域標記場
        self.filter_zone = ti.field(dtype=ti.i32, shape=(config.NX, config.NY, config.NZ))
        self.filter_resistance = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.filter_blockage = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 濾紙動態狀態
        self.accumulated_particles = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.local_flow_rate = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 濾紙幾何參數
        self.filter_bottom_z = None  # 將在初始化時設置
        self.filter_thickness_lu = None
        
        print("濾紙系統初始化完成")
        print(f"  濾紙厚度: {self.PAPER_THICKNESS*1000:.2f}mm")
        print(f"  濾紙孔隙率: {self.PAPER_POROSITY:.1%}")
        print(f"  濾紙孔徑: {self.PAPER_PORE_SIZE*1e6:.0f}微米")
    
    def initialize_filter_geometry(self):
        """初始化錐形濾紙幾何分佈"""
        # 計算濾紙位置 (從V60底部延伸到整個錐形內表面)
        bottom_z = 5.0  # V60底部位置 (與lbm_solver.py一致)
        self.filter_bottom_z = bottom_z  # 濾紙從底部開始
        self.filter_thickness_lu = max(1, int(self.PAPER_THICKNESS / config.SCALE_LENGTH))
        
        # **關鍵修復**: 先設置V60固體邊界
        self._setup_v60_geometry()
        
        self._setup_filter_zones()
        self._calculate_initial_resistance()
        
        # 計算濾紙覆蓋的錐形表面積
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        filter_coverage_height = cup_height_lu
        
        print(f"完整圓錐形濾紙幾何初始化完成:")
        print(f"  濾紙底部位置: Z = {self.filter_bottom_z:.1f} 格子單位")
        print(f"  濾紙覆蓋高度: {filter_coverage_height:.1f} 格子單位") 
        print(f"  濾紙厚度: {self.filter_thickness_lu} 格子單位")
        print(f"  濾杯-濾紙空隙: 2.0 mm")
        print(f"  設計特點:")
        print(f"    └─ 完整圓錐形濾紙（非平底）")
        print(f"    └─ 濾杯與濾紙間2mm排水/排氣空隙")
        print(f"    └─ V60底部設置為開放大洞（正確設計）")
        print(f"    └─ 底部開口直徑: {config.BOTTOM_RADIUS*2*100:.1f}cm")
        print(f"    └─ 流體通過濾紙後從底部開口流出")
    
    @ti.kernel
    def _setup_v60_geometry(self):
        """設置V60固體邊界幾何 - 修正版：濾杯與濾紙間2mm空隙"""
        center_x = config.NX * 0.5
        center_y = config.NY * 0.5
        top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
        bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        
        # V60幾何範圍
        v60_bottom_z = 5.0
        v60_top_z = v60_bottom_z + cup_height_lu
        wall_thickness = 2.0  # V60壁厚（格子單位）
        
        # 2mm空隙轉換為格子單位
        air_gap_lu = 0.002 / config.SCALE_LENGTH  # 2mm空隙
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            x = ti.cast(i, ti.f32)
            y = ti.cast(j, ti.f32)
            z = ti.cast(k, ti.f32)
            
            radius_from_center = ti.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # 設置固體邊界
            is_solid = False
            
            # 1. V60底部設置為開放大洞（正確的V60設計）
            if z <= v60_bottom_z:
                # V60底部應該是開放的大洞，只有壁厚部分是固體
                if radius_from_center > bottom_radius_lu:
                    is_solid = True  # 底部外圍的支撐結構
                # 底部中心的開口區域保持為流體（is_solid = False）
            elif z <= v60_top_z:
                # 錐形側壁 - 考慮與濾紙的空隙
                height_ratio = (z - v60_bottom_z) / cup_height_lu
                inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                outer_radius = inner_radius + wall_thickness
                
                # V60杯壁：在inner_radius + air_gap之外設為固體
                if radius_from_center > inner_radius + air_gap_lu + wall_thickness:
                    is_solid = True
            else:
                # V60頂部以上的邊界
                if radius_from_center > top_radius_lu + wall_thickness:
                    is_solid = True
            
            # 2. 計算域邊界
            if (i <= 2 or i >= config.NX-3 or 
                j <= 2 or j >= config.NY-3 or 
                k <= 2 or k >= config.NZ-3):
                is_solid = True
                
            # 設置固體場
            if is_solid:
                self.lbm.solid[i, j, k] = ti.u8(1)
            else:
                self.lbm.solid[i, j, k] = ti.u8(0)
    
    @ti.kernel
    def _setup_filter_zones(self):
        """設置完整圓錐形濾紙區域標記 - 修正版：包含錐形底部"""
        center_x = config.NX * 0.5
        center_y = config.NY * 0.5
        top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
        bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        
        # 濾紙覆蓋範圍：完整圓錐形
        filter_top_z = 5.0 + cup_height_lu  # V60頂部
        filter_bottom_z = self.filter_bottom_z  # V60底部
        
        # 濾紙厚度（格子單位）
        paper_thickness_lu = ti.max(1.0, self.PAPER_THICKNESS / config.SCALE_LENGTH)
        
        # 2mm空隙轉換為格子單位
        air_gap_lu = 0.002 / config.SCALE_LENGTH  # 2mm空隙
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            x = ti.cast(i, ti.f32)
            y = ti.cast(j, ti.f32)
            z = ti.cast(k, ti.f32)
            
            radius_from_center = ti.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # 初始化為非濾紙區域
            self.filter_zone[i, j, k] = 0
            
            # 檢查是否在濾紙高度範圍內
            if z >= filter_bottom_z and z <= filter_top_z:
                
                # 計算該高度的V60內表面半徑（減去空隙）
                height_ratio = (z - filter_bottom_z) / cup_height_lu
                height_ratio = ti.max(0.0, ti.min(1.0, height_ratio))  # 限制在[0,1]
                
                # V60內表面半徑（考慮2mm空隙）
                v60_inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                filter_outer_radius = v60_inner_radius - air_gap_lu  # 濾紙外表面
                filter_inner_radius = filter_outer_radius - paper_thickness_lu  # 濾紙內表面
                
                # 圓錐形濾紙條件：在濾紙厚度範圍內
                if filter_inner_radius <= radius_from_center <= filter_outer_radius:
                    self.filter_zone[i, j, k] = 1
            
            # 特殊處理：圓錐形底部濾紙
            elif z >= filter_bottom_z - paper_thickness_lu and z < filter_bottom_z:
                # 底部濾紙：圓錐形底部，不是平底
                transition_radius = bottom_radius_lu - air_gap_lu
                if radius_from_center <= transition_radius:
                    self.filter_zone[i, j, k] = 1
    
    @ti.kernel 
    def _calculate_initial_resistance(self):
        """計算濾紙初始阻力"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.filter_zone[i, j, k] == 1:
                # 基於濾紙物理參數計算阻力
                # Darcy's law: ΔP = (μ * L * v) / K
                # 阻力係數 = μ * L / K
                viscosity = config.WATER_VISCOSITY_90C
                thickness = self.PAPER_THICKNESS
                permeability = self.PAPER_PERMEABILITY
                
                base_resistance = viscosity * thickness / permeability
                # 轉換為格子單位並正規化
                self.filter_resistance[i, j, k] = base_resistance * config.SCALE_TIME / config.SCALE_LENGTH
            else:
                self.filter_resistance[i, j, k] = 0.0
            
            # 初始化其他場
            self.filter_blockage[i, j, k] = 0.0
            self.accumulated_particles[i, j, k] = 0.0
            self.local_flow_rate[i, j, k] = 0.0
    
    @ti.kernel
    def apply_filter_effects(self):
        """對流體場施加濾紙效應"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.filter_zone[i, j, k] == 1 and self.lbm.solid[i, j, k] == 0:
                # 獲取當前流體速度
                u_local = self.lbm.u[i, j, k]
                
                # 計算總阻力 (基礎阻力 + 顆粒堵塞)
                total_resistance = self.filter_resistance[i, j, k] * (1.0 + self.filter_blockage[i, j, k])
                
                # 施加濾紙阻力 (指數衰減模型)
                resistance_factor = ti.exp(-total_resistance * config.DT)
                
                # 更新速度 (主要影響垂直分量，保持水平分量以模擬孔隙結構)
                u_local.z *= resistance_factor
                u_local.x *= (resistance_factor + 1.0) / 2.0  # 水平阻力較小
                u_local.y *= (resistance_factor + 1.0) / 2.0
                
                # 記錄局部流速用於動態調整
                self.local_flow_rate[i, j, k] = u_local.norm()
                
                # 更新LBM場
                self.lbm.u[i, j, k] = u_local
    
    @ti.kernel
    def block_particles_at_filter(self, particle_positions: ti.template(), 
                                 particle_velocities: ti.template(),
                                 particle_radii: ti.template(),
                                 particle_active: ti.template(),
                                 particle_count: ti.template()):
        """阻擋咖啡顆粒通過濾紙"""
        for p in range(particle_count[None]):
            if particle_active[p] == 0:
                continue
                
            pos = particle_positions[p]
            vel = particle_velocities[p]
            radius = particle_radii[p]
            
            # 轉換為格子單位
            grid_x = int(pos.x / config.SCALE_LENGTH)
            grid_y = int(pos.y / config.SCALE_LENGTH) 
            grid_z = int(pos.z / config.SCALE_LENGTH)
            
            # 檢查顆粒是否接近濾紙
            if (grid_x >= 0 and grid_x < config.NX and
                grid_y >= 0 and grid_y < config.NY and
                grid_z >= 0 and grid_z < config.NZ):
                
                # 檢查是否在濾紙區域或即將進入
                particle_radius_lu = radius / config.SCALE_LENGTH
                
                for offset_z in range(-2, 3):  # 檢查附近格點
                    check_z = grid_z + offset_z
                    if (check_z >= 0 and check_z < config.NZ and
                        self.filter_zone[grid_x, grid_y, check_z] == 1):
                        
                        # 顆粒觸碰濾紙，反彈處理
                        if vel.z < 0:  # 向下運動
                            # 彈性碰撞，垂直速度反向並衰減
                            vel.z = -vel.z * 0.3  # 30%的恢復係數
                            
                            # 增加水平隨機擾動（模擬濾紙表面不平）
                            random_x = (ti.random() - 0.5) * 0.01
                            random_y = (ti.random() - 0.5) * 0.01
                            vel.x += random_x
                            vel.y += random_y
                            
                            # 更新粒子速度
                            particle_velocities[p] = vel
                            
                            # 累積顆粒在濾紙的影響（用於動態阻力調整）
                            if (grid_x < config.NX and grid_y < config.NY and 
                                check_z < config.NZ):
                                self.accumulated_particles[grid_x, grid_y, check_z] += 0.01
                        
                        break
    
    @ti.kernel
    def update_dynamic_resistance(self):
        """根據顆粒累積動態更新濾紙阻力"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.filter_zone[i, j, k] == 1:
                # 根據累積顆粒計算額外阻塞
                particle_accumulation = self.accumulated_particles[i, j, k]
                
                # 阻塞模型：指數增長，但有上限
                max_blockage = 0.9  # 最大90%阻塞
                blockage_rate = 0.1
                new_blockage = max_blockage * (1.0 - ti.exp(-blockage_rate * particle_accumulation))
                
                # 平滑更新阻塞程度
                self.filter_blockage[i, j, k] = 0.95 * self.filter_blockage[i, j, k] + 0.05 * new_blockage
                
                # 顆粒累積緩慢衰減（模擬沖刷效果）
                self.accumulated_particles[i, j, k] *= 0.999
    
    def step(self, particle_system):
        """執行一個濾紙系統時間步"""
        # 1. 對流體施加濾紙阻力
        self.apply_filter_effects()
        
        # 2. 阻擋咖啡顆粒
        if particle_system is not None:
            self.block_particles_at_filter(
                particle_system.position,
                particle_system.velocity, 
                particle_system.radius,
                particle_system.active,
                particle_system.particle_count
            )
        
        # 3. 動態更新阻力
        self.update_dynamic_resistance()
    
    def get_filter_statistics(self):
        """獲取濾紙系統統計信息"""
        filter_zone_data = self.filter_zone.to_numpy()
        resistance_data = self.filter_resistance.to_numpy()
        blockage_data = self.filter_blockage.to_numpy()
        flow_data = self.local_flow_rate.to_numpy()
        
        total_filter_nodes = np.sum(filter_zone_data)
        avg_resistance = np.mean(resistance_data[filter_zone_data == 1]) if total_filter_nodes > 0 else 0
        avg_blockage = np.mean(blockage_data[filter_zone_data == 1]) if total_filter_nodes > 0 else 0
        avg_flow = np.mean(flow_data[filter_zone_data == 1]) if total_filter_nodes > 0 else 0
        
        return {
            'total_filter_nodes': int(total_filter_nodes),
            'average_resistance': float(avg_resistance),
            'average_blockage': float(avg_blockage * 100),  # 轉為百分比
            'average_flow_rate': float(avg_flow),
            'max_blockage': float(np.max(blockage_data) * 100) if total_filter_nodes > 0 else 0
        }
    
    @ti.kernel
    def get_filter_inner_radius_at_height(self, z: ti.f32) -> ti.f32:
        """獲取指定高度的濾紙內表面半徑"""
        bottom_z = self.filter_bottom_z
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
        bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
        
        # 計算高度比例
        height_ratio = (z - bottom_z) / cup_height_lu
        height_ratio = ti.max(0.0, ti.min(1.0, height_ratio))
        
        # 錐形內半徑
        inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
        return inner_radius
    
    def get_coffee_bed_boundary(self):
        """獲取咖啡床邊界信息（供顆粒系統使用）"""
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        center_x = config.NX * 0.5
        center_y = config.NY * 0.5
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'bottom_z': self.filter_bottom_z,
            'top_z': self.filter_bottom_z + cup_height_lu,
            'top_radius_lu': config.TOP_RADIUS / config.SCALE_LENGTH,
            'bottom_radius_lu': config.BOTTOM_RADIUS / config.SCALE_LENGTH,
            'get_radius_at_height': self.get_filter_inner_radius_at_height
        }
        
    def print_status(self):
        """打印濾紙系統狀態"""
        stats = self.get_filter_statistics()
        print(f"📄 濾紙系統狀態:")
        print(f"   └─ 濾紙節點數: {stats['total_filter_nodes']:,}")
        print(f"   └─ 平均阻力: {stats['average_resistance']:.2e}")
        print(f"   └─ 平均阻塞度: {stats['average_blockage']:.1f}%")
        print(f"   └─ 最大阻塞度: {stats['max_blockage']:.1f}%")
        print(f"   └─ 平均流速: {stats['average_flow_rate']:.4f} m/s")