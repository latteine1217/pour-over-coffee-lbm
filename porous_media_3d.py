# porous_media_3d.py
"""
3D多孔介質處理模組 - 使用Taichi稀疏數據結構優化
高效處理咖啡粉層的Darcy流動和萃取過程
"""

import taichi as ti
import config

@ti.data_oriented
class PorousMedia3D:
    def __init__(self, lbm_solver, particle_system=None):
        self.lbm = lbm_solver
        self.particles = particle_system
        
        # 多孔介質屬性場
        self.porosity = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.permeability = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.drag_force = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 咖啡萃取相關場
        self.extraction_concentration = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.coffee_density = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 稀疏數據結構 - 簡化版本
        if config.USE_SPARSE_MATRIX:
            self.n_porous_nodes = ti.field(dtype=ti.i32, shape=())
        
        # 性能計數器
        self.performance_stats = ti.field(dtype=ti.f32, shape=5)
        
        self.init_porous_properties()
    
    @ti.kernel
    def init_porous_properties(self):
        """初始化多孔介質屬性 - 並行處理"""
        porous_count = 0
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 檢查是否在咖啡床區域
            z_phys = k * config.SCALE_LENGTH
            
            # 計算到中心軸的距離
            cx, cy = config.NX // 2, config.NY // 2
            r_current = ti.sqrt((i - cx)**2 + (j - cy)**2) * config.SCALE_LENGTH
            
            # 在咖啡床高度範圍內且在V60內部
            if (z_phys < config.COFFEE_BED_HEIGHT_PHYS and 
                self.lbm.solid[i, j, k] == 0):
                
                # 設置多孔介質屬性
                self.porosity[i, j, k] = config.PORE_PERC
                self.permeability[i, j, k] = config.DARCY_NUMBER
                self.lbm.porous[i, j, k] = config.PORE_PERC
                
                # 初始化咖啡濃度（隨機分佈模擬粉粒不均）
                noise = ti.sin(i * 0.1) * ti.cos(j * 0.1) * ti.sin(k * 0.1)
                self.coffee_density[i, j, k] = 1.0 + 0.1 * noise
                self.extraction_concentration[i, j, k] = 0.0
                
                porous_count += 1
            else:
                # 自由流區域
                self.porosity[i, j, k] = 1.0
                self.permeability[i, j, k] = 1.0
                self.lbm.porous[i, j, k] = 1.0
                self.coffee_density[i, j, k] = 0.0
                self.extraction_concentration[i, j, k] = 0.0
        
        # 儲存多孔節點數量
        if config.USE_SPARSE_MATRIX:
            self.n_porous_nodes[None] = porous_count
    
    @ti.func
    def get_effective_viscosity(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.f32:
        """計算有效黏度 - 考慮相場和多孔性"""
        phase_local = self.lbm.phase[i, j, k]
        porosity_local = self.porosity[i, j, k]
        
        # 基礎黏度
        tau_base = config.TAU_WATER * phase_local + config.TAU_AIR * (1.0 - phase_local)
        
        # 多孔介質修正（Kozeny-Carman關係）
        tau_eff = tau_base
        if porosity_local < 0.99:
            tortuosity = (1.0 - porosity_local)**2 / porosity_local**3
            tau_eff = tau_base * (1.0 + tortuosity)
        
        return (tau_eff - 0.5) / 3.0
    
    @ti.kernel
    def compute_darcy_drag_sparse(self):
        """計算Darcy阻力 - 稀疏優化版本"""
        # 方法1：標準並行處理
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.porosity[i, j, k] < 0.99:  # 多孔介質
                u_local = self.lbm.u[i, j, k]
                viscosity = self.get_effective_viscosity(i, j, k)
                permeability = self.permeability[i, j, k]
                
                if permeability > 1e-12:
                    # Darcy阻力：F = -μ/K * u
                    drag_coefficient = viscosity / permeability
                    self.drag_force[i, j, k] = -drag_coefficient * u_local
                    
                    # Forchheimer修正（高速流動）
                    u_magnitude = u_local.norm()
                    if u_magnitude > 1e-6:
                        forchheimer_coeff = 1.75 * viscosity / ti.sqrt(permeability)
                        inertial_drag = -forchheimer_coeff * u_magnitude * u_local
                        self.drag_force[i, j, k] += inertial_drag
                else:
                    # 無滲透性：完全阻止流動
                    self.drag_force[i, j, k] = -u_local * 1e6
            else:
                self.drag_force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.kernel
    def apply_porous_effects(self):
        """施加多孔介質效應到LBM - 並行處理"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.porosity[i, j, k] < 0.99:  # 多孔區域
                rho_local = self.lbm.rho[i, j, k]
                
                if rho_local > 1e-10:
                    # 施加阻力加速度
                    acceleration = self.drag_force[i, j, k] / rho_local
                    self.lbm.u[i, j, k] += acceleration * config.DT
                
                # 修正密度以反映孔隙率
                porosity_local = self.porosity[i, j, k]
                effective_density = self.lbm.rho[i, j, k] * porosity_local
                self.lbm.rho[i, j, k] = effective_density
    
    @ti.kernel
    def coffee_extraction_kinetics(self):
        """咖啡萃取動力學 - 並行化學反應"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if (self.porosity[i, j, k] < 0.99 and 
                self.lbm.phase[i, j, k] > 0.5):  # 水在多孔介質中
                
                # 萃取速率與接觸時間和溫度相關
                water_phase = self.lbm.phase[i, j, k]
                coffee_available = self.coffee_density[i, j, k]
                
                if coffee_available > 0.01:
                    # 簡化的萃取動力學：一級反應
                    extraction_rate = 0.001 * water_phase * coffee_available
                    extracted_amount = extraction_rate * config.DT
                    
                    # 更新咖啡濃度和剩餘咖啡量
                    self.extraction_concentration[i, j, k] += extracted_amount
                    self.coffee_density[i, j, k] -= extracted_amount * 0.1
                    
                    # 增加流體密度以反映萃取的咖啡
                    density_increase = extracted_amount * 0.05
                    self.lbm.rho[i, j, k] += density_increase
                    
                    # 限制最大密度
                    max_density = config.RHO_WATER * 1.2
                    self.lbm.rho[i, j, k] = ti.min(self.lbm.rho[i, j, k], max_density)
    
    @ti.kernel
    def apply_filter_paper_resistance(self):
        """施加濾紙阻力效應 - 底部邊界層"""
        filter_zone_height = 3  # 濾紙影響區域
        
        for i, j, k in ti.ndrange(config.NX, config.NY, filter_zone_height):
            if self.lbm.solid[i, j, k] == 0:  # 流體區域
                # 濾紙阻力隨深度增加
                depth_factor = (filter_zone_height - k) / filter_zone_height
                resistance = 0.5 * depth_factor
                
                # 施加阻力
                self.lbm.u[i, j, k] *= (1.0 - resistance * config.DT)
    
    @ti.kernel
    def compute_performance_metrics(self):
        """計算性能指標 - 並行歸約"""
        total_extraction = 0.0
        total_flow_rate = 0.0
        max_concentration = 0.0
        porous_volume = 0.0
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.porosity[i, j, k] < 0.99:
                porous_volume += 1.0
                
                # 萃取總量
                total_extraction += self.extraction_concentration[i, j, k]
                
                # 最大濃度
                max_concentration = ti.max(max_concentration, 
                                         self.extraction_concentration[i, j, k])
                
                # 流速
                if self.lbm.phase[i, j, k] > 0.5:
                    u_magnitude = self.lbm.u[i, j, k].norm()
                    total_flow_rate += u_magnitude
        
        # 儲存統計結果
        if porous_volume > 0:
            self.performance_stats[0] = total_extraction / porous_volume  # 平均萃取濃度
            self.performance_stats[1] = total_flow_rate / porous_volume   # 平均流速
            self.performance_stats[2] = max_concentration                 # 最大濃度
            self.performance_stats[3] = porous_volume                     # 多孔體積
            self.performance_stats[4] = total_extraction                  # 總萃取量
    
    def step(self):
        """執行多孔介質一個時間步長 - 高效並行流水線"""
        self.compute_darcy_drag_sparse()
        self.apply_porous_effects()
        self.coffee_extraction_kinetics()
        self.apply_filter_paper_resistance()
        
        # 定期重新計算性能指標
        if ti.static(True):  # 每步都計算，可以改為週期性
            self.compute_performance_metrics()
    
    def get_performance_stats(self):
        """獲取性能統計數據"""
        return self.performance_stats.to_numpy()