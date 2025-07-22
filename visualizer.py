# visualizer.py
"""
統一視覺化系統 - 3D LBM專用
提供密度場、速度場、相場等多種視覺化選項
"""

import taichi as ti
import numpy as np
import time
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
        """計算統計信息 - 簡化版本避免潛在問題"""
        total_water_mass = 0.0
        total_air_mass = 0.0
        max_velocity = 0.0
        min_velocity = 1e6
        avg_velocity = 0.0
        total_nodes = 0
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 檢查邊界避免越界
            if 1 <= i < config.NX-1 and 1 <= j < config.NY-1 and 1 <= k < config.NZ-1:
                if self.lbm.solid[i, j, k] == 0:  # 只計算流體區域
                    rho = self.lbm.rho[i, j, k]
                    phase = self.lbm.phase[i, j, k]
                    
                    # 計算速度大小，使用分量來避免norm()的問題
                    ux = self.lbm.u[i, j, k][0]
                    uy = self.lbm.u[i, j, k][1]
                    uz = self.lbm.u[i, j, k][2]
                    u_mag = ti.sqrt(ux*ux + uy*uy + uz*uz)
                    
                    # 修復：使用更寬鬆的水相判定條件
                    if phase > 0.1:  # 降低閾值，phi > 0.1 就算水相
                        total_water_mass += rho
                    else:  # 氣相
                        total_air_mass += rho
                    
                    if u_mag > max_velocity:
                        max_velocity = u_mag
                    if u_mag < min_velocity:
                        min_velocity = u_mag
                    avg_velocity += u_mag
                    total_nodes += 1
        
        if total_nodes > 0:
            avg_velocity /= total_nodes
        else:
            min_velocity = 0.0  # 避免無限大
        
        # 存儲統計結果
        self.stats[0] = total_water_mass
        self.stats[1] = total_air_mass
        self.stats[2] = max_velocity
        self.stats[3] = min_velocity
        self.stats[4] = avg_velocity
        self.stats[5] = total_nodes
    
    def get_statistics(self):
        """獲取統計信息 - 實際計算版本"""
        try:
            # 執行實際的統計計算
            self.compute_statistics()
            
            # 從Taichi場中讀取結果
            return {
                'total_water_mass': float(self.stats[0]),
                'total_air_mass': float(self.stats[1]),
                'max_velocity': float(self.stats[2]),
                'min_velocity': float(self.stats[3]),
                'avg_velocity': float(self.stats[4]),
                'total_nodes': int(self.stats[5])
            }
        except Exception as e:
            print(f"Statistics computation error: {e}")
            return {
                'total_water_mass': 0.0,
                'total_air_mass': 0.0,
                'max_velocity': 0.0,
                'min_velocity': 0.0,
                'avg_velocity': 0.0,
                'total_nodes': 0
            }
    
    def get_detailed_diagnostics(self):
        """獲取詳細診斷資訊"""
        try:
            # 基本統計
            basic_stats = self.get_statistics()
            
            # 額外診斷資訊
            u_data = self.lbm_solver.u.to_numpy()
            rho_data = self.lbm_solver.rho.to_numpy()
            
            # 速度場分析
            u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
            non_zero_velocity = u_magnitude > 1e-8
            
            # 相場分析（如果有）
            phase_stats = {}
            if self.multiphase_solver:
                phi_data = self.multiphase_solver.phi.to_numpy()
                water_phase = phi_data > 0.5
                air_phase = phi_data <= 0.5
                
                phase_stats = {
                    'water_cells': int(np.sum(water_phase)),
                    'air_cells': int(np.sum(air_phase)),
                    'interface_cells': int(np.sum(np.abs(phi_data) < 0.5)),
                    'max_phi': float(np.max(phi_data)),
                    'min_phi': float(np.min(phi_data))
                }
            
            # 流場診斷
            flow_diagnostics = {
                'non_zero_velocity_cells': int(np.sum(non_zero_velocity)),
                'velocity_std': float(np.std(u_magnitude)),
                'max_u_x': float(np.max(u_data[:,:,:,0])),
                'max_u_y': float(np.max(u_data[:,:,:,1])),
                'max_u_z': float(np.max(u_data[:,:,:,2])),
                'density_range': [float(np.min(rho_data)), float(np.max(rho_data))]
            }
            
            return {
                'basic_stats': basic_stats,
                'phase_stats': phase_stats,
                'flow_diagnostics': flow_diagnostics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Detailed diagnostics error: {e}")
            return {
                'basic_stats': self.get_statistics(),
                'phase_stats': {},
                'flow_diagnostics': {},
                'error': str(e)
            }
    
    def diagnose_velocity_field_issue(self):
        """診斷速度場問題"""
        print("\n🔍 速度場問題診斷")
        print("-" * 30)
        
        try:
            u_data = self.lbm_solver.u.to_numpy()
            u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
            
            # 統計分析
            total_cells = u_magnitude.size
            non_zero_cells = np.sum(u_magnitude > 1e-8)
            max_velocity = np.max(u_magnitude)
            
            print(f"總格子數: {total_cells:,}")
            print(f"非零速度格子: {non_zero_cells:,} ({non_zero_cells/total_cells*100:.2f}%)")
            print(f"最大速度: {max_velocity:.8f}")
            
            # 空間分布分析
            if non_zero_cells > 0:
                non_zero_indices = np.where(u_magnitude > 1e-8)
                z_distribution = np.bincount(non_zero_indices[2], minlength=config.NZ)
                
                print(f"垂直分布（Z方向非零點數）:")
                for z in range(0, config.NZ, max(1, config.NZ//10)):
                    if z < len(z_distribution):
                        print(f"  Z={z:3d}: {z_distribution[z]:4d}點")
            
            # 相場檢查
            if self.multiphase_solver:
                phi_data = self.multiphase_solver.phi.to_numpy()
                water_cells = np.sum(phi_data > 0.5)
                print(f"水相格子數: {water_cells:,}")
                
                if water_cells == 0:
                    print("⚠️  警告：沒有水相！")
                elif non_zero_cells == 0:
                    print("⚠️  警告：有水相但無流動！")
            
        except Exception as e:
            print(f"診斷失敗: {e}")
    
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