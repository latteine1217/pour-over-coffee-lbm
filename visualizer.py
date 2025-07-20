# visualizer.py
"""
統一視覺化系統 - 3D LBM專用
提供密度場、速度場、相場等多種視覺化選項
"""

import taichi as ti
import numpy as np
import config

@ti.data_oriented
class UnifiedVisualizer:
    def __init__(self, lbm_solver, multiphase=None, geometry=None, particle_system=None):
        self.lbm = lbm_solver
        self.multiphase = multiphase
        self.geometry = geometry
        self.particles = particle_system
        self.mode = '3d'  # 固定為3D模式
        
        # 2D顯示場 (用於3D切片顯示)
        self.display_field = ti.field(dtype=ti.f32, shape=(config.NX, config.NY))
        self.color_field = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY))
        
        # 3D視覺化場
        self._init_3d_fields()
        
        # 統計信息
        self.stats = ti.field(dtype=ti.f32, shape=10)
        
        print("統一視覺化系統初始化完成 (3D專用)")
    
    def _init_3d_fields(self):
        """初始化3D視覺化相關場"""
        # 不同方向的切片
        self.xz_slice = ti.field(dtype=ti.f32, shape=(config.NX, config.NZ))
        self.yz_slice = ti.field(dtype=ti.f32, shape=(config.NY, config.NZ))
        self.xz_color = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NZ))
        self.yz_color = ti.Vector.field(3, dtype=ti.f32, shape=(config.NY, config.NZ))
        
        # 體積渲染
        self.volume_texture = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.render_mask = ti.field(dtype=ti.i32, shape=(config.NX, config.NY, config.NZ))
        
        self._init_render_mask()
    
    @ti.kernel
    def _init_render_mask(self):
        """初始化3D渲染遮罩"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 只渲染非固體區域
            if self.lbm.solid[i, j, k] == 0:
                self.render_mask[i, j, k] = 1
            else:
                self.render_mask[i, j, k] = 0
    
    @ti.kernel
    def prepare_density_field(self):
        """準備密度場視覺化 - 3D切片"""
        # 顯示中間層切片
        z_slice = config.NZ // 2
        for i, j in self.display_field:
            self.display_field[i, j] = self.lbm.rho[i, j, z_slice]
    
    @ti.kernel
    def prepare_velocity_field(self):
        """準備速度場視覺化 - 3D切片"""
        z_slice = config.NZ // 2
        for i, j in self.display_field:
            u = self.lbm.u[i, j, z_slice]
            self.display_field[i, j] = u.norm()
    
    @ti.kernel
    def prepare_phase_field(self):
        """準備相場視覺化 - 3D切片"""
        z_slice = config.NZ // 2
        for i, j in self.display_field:
            self.display_field[i, j] = self.lbm.phase[i, j, z_slice]
    
    @ti.kernel
    def compute_composite_field(self):
        """計算綜合場 - 3D切片"""
        z_slice = config.NZ // 2
        for i, j in self.color_field:
            # 獲取各種場的值
            rho = self.lbm.rho[i, j, z_slice]
            u = self.lbm.u[i, j, z_slice]
            phase = self.lbm.phase[i, j, z_slice]
            
            # 計算顏色
            u_mag = u.norm()
            
            # 紅色通道：密度（水相）
            red = ti.min(rho / config.RHO_WATER, 1.0)
            
            # 綠色通道：速度大小
            green = ti.min(u_mag * 10.0, 1.0)
            
            # 藍色通道：相場
            blue = phase
            
            self.color_field[i, j] = ti.Vector([red, green, blue])
    
    @ti.kernel
    def compute_3d_slice(self, direction: ti.i32, slice_idx: ti.i32):
        """計算3D切片"""
        if direction == 0:  # XY切片
            for i, j in self.display_field:
                if slice_idx < config.NZ:
                    self.display_field[i, j] = self.lbm.rho[i, j, slice_idx]
        elif direction == 1:  # XZ切片
            for i, k in self.xz_slice:
                if slice_idx < config.NY:
                    self.xz_slice[i, k] = self.lbm.rho[i, slice_idx, k]
        elif direction == 2:  # YZ切片
            for j, k in self.yz_slice:
                if slice_idx < config.NX:
                    self.yz_slice[j, k] = self.lbm.rho[slice_idx, j, k]
    
    @ti.kernel
    def compute_statistics(self):
        """計算統計信息"""
        total_water_mass = 0.0
        total_air_mass = 0.0
        max_velocity = 0.0
        min_velocity = 1e6
        avg_velocity = 0.0
        total_nodes = 0
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.lbm.solid[i, j, k] == 0:  # 只計算流體區域
                rho = self.lbm.rho[i, j, k]
                phase = self.lbm.phase[i, j, k]
                u_mag = self.lbm.u[i, j, k].norm()
                
                if phase > 0.5:  # 水相
                    total_water_mass += rho
                else:  # 氣相
                    total_air_mass += rho
                
                max_velocity = ti.max(max_velocity, u_mag)
                min_velocity = ti.min(min_velocity, u_mag)
                avg_velocity += u_mag
                total_nodes += 1
        
        if total_nodes > 0:
            avg_velocity /= total_nodes
        
        # 存儲統計結果
        self.stats[0] = total_water_mass
        self.stats[1] = total_air_mass
        self.stats[2] = max_velocity
        self.stats[3] = min_velocity
        self.stats[4] = avg_velocity
        self.stats[5] = total_nodes
    
    def get_statistics(self):
        """獲取統計信息"""
        self.compute_statistics()
        stats_np = self.stats.to_numpy()
        
        return {
            'total_water_mass': stats_np[0],
            'total_air_mass': stats_np[1],
            'max_velocity': stats_np[2],
            'min_velocity': stats_np[3],
            'avg_velocity': stats_np[4],
            'total_nodes': int(stats_np[5])
        }
    
    def display_gui(self, field_type='density', slice_direction='xy', slice_idx=None):
        """顯示GUI"""
        if slice_idx is None:
            slice_idx = config.NZ // 2
        
        gui = ti.GUI("Pour-Over Coffee Simulation", res=(config.NX, config.NY))
        
        while gui.running:
            if field_type == 'density':
                self.prepare_density_field()
                gui.set_image(self.display_field)
            elif field_type == 'velocity':
                self.prepare_velocity_field()
                gui.set_image(self.display_field)
            elif field_type == 'phase':
                self.prepare_phase_field()
                gui.set_image(self.display_field)
            elif field_type == 'composite':
                self.compute_composite_field()
                gui.set_image(self.color_field)
            
            gui.show()
    
    def save_image(self, filename, field_type='density'):
        """保存圖像"""
        if field_type == 'density':
            self.prepare_density_field()
            ti.tools.imwrite(self.display_field, filename)
        elif field_type == 'velocity':
            self.prepare_velocity_field()
            ti.tools.imwrite(self.display_field, filename)
        elif field_type == 'phase':
            self.prepare_phase_field()
            ti.tools.imwrite(self.display_field, filename)
        elif field_type == 'composite':
            self.compute_composite_field()
            ti.tools.imwrite(self.color_field, filename)
        
        print(f"圖像已保存: {filename}")