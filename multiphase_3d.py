# multiphase_3d.py
"""
3D多相流體處理模組 - 使用Taichi並行計算優化
實現空氣-水界面追蹤和表面張力效應
"""

import taichi as ti
import config

@ti.data_oriented
class MultiphaseFlow3D:
    def __init__(self, lbm_solver):
        self.lbm = lbm_solver
        
        # 相場相關變數 - 3D
        self.phi = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))        # 相場變數
        self.phi_new = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))    # 新相場變數
        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))  # 界面法向量
        self.curvature = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))  # 曲率場
        
        # 梯度計算用的場
        self.grad_phi = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 表面張力力場
        self.surface_force = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
    
    @ti.kernel
    def init_phase_field(self):
        """初始化3D相場變數 - 並行處理"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 設置初始界面：上半部分為水，下半部分為空氣
            if k > config.NZ * 0.7:
                self.phi[i, j, k] = 1.0  # 水相
            elif k < config.NZ * 0.3:
                self.phi[i, j, k] = -1.0  # 空氣相
            else:
                # 界面區域使用tanh函數平滑過渡
                z_center = config.NZ * 0.5
                interface_width = config.INTERFACE_THICKNESS
                self.phi[i, j, k] = ti.tanh(2.0 * (k - z_center) / interface_width)
    
    @ti.kernel
    def compute_gradients(self):
        """計算相場梯度 - 3D中央差分"""
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            # 3D中央差分計算梯度
            dphi_dx = (self.phi[i+1, j, k] - self.phi[i-1, j, k]) * 0.5
            dphi_dy = (self.phi[i, j+1, k] - self.phi[i, j-1, k]) * 0.5
            dphi_dz = (self.phi[i, j, k+1] - self.phi[i, j, k-1]) * 0.5
            
            self.grad_phi[i, j, k] = ti.Vector([dphi_dx, dphi_dy, dphi_dz])
            
            # 計算界面法向量（歸一化梯度）
            grad_magnitude = self.grad_phi[i, j, k].norm()
            if grad_magnitude > 1e-10:
                self.normal[i, j, k] = self.grad_phi[i, j, k] / grad_magnitude
            else:
                self.normal[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.kernel
    def compute_curvature(self):
        """計算3D界面曲率 - 法向量散度"""
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            # 計算法向量的散度（曲率）
            dnx_dx = (self.normal[i+1, j, k][0] - self.normal[i-1, j, k][0]) * 0.5
            dny_dy = (self.normal[i, j+1, k][1] - self.normal[i, j-1, k][1]) * 0.5
            dnz_dz = (self.normal[i, j, k+1][2] - self.normal[i, j, k-1][2]) * 0.5
            
            self.curvature[i, j, k] = dnx_dx + dny_dy + dnz_dz
    
    @ti.kernel
    def update_phase_field(self):
        """更新相場變數 - 3D對流"""
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            # 基於3D速度場對流相場
            u_local = self.lbm.u[i, j, k]
            
            # 3D上風差分格式
            dphi_dx = 0.0
            dphi_dy = 0.0
            dphi_dz = 0.0
            
            if u_local[0] > 0:
                dphi_dx = self.phi[i, j, k] - self.phi[i-1, j, k]
            else:
                dphi_dx = self.phi[i+1, j, k] - self.phi[i, j, k]
                
            if u_local[1] > 0:
                dphi_dy = self.phi[i, j, k] - self.phi[i, j-1, k]
            else:
                dphi_dy = self.phi[i, j+1, k] - self.phi[i, j, k]
                
            if u_local[2] > 0:
                dphi_dz = self.phi[i, j, k] - self.phi[i, j, k-1]
            else:
                dphi_dz = self.phi[i, j, k+1] - self.phi[i, j, k]
            
            # 3D對流項
            advection = u_local[0] * dphi_dx + u_local[1] * dphi_dy + u_local[2] * dphi_dz
            
            # 更新相場
            self.phi_new[i, j, k] = self.phi[i, j, k] - config.DT * advection
            
            # 限制相場值
            self.phi_new[i, j, k] = ti.max(-1.0, ti.min(1.0, self.phi_new[i, j, k]))
    
    @ti.kernel
    def compute_surface_tension_force(self):
        """計算3D表面張力力 - 並行優化"""
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            phi_local = self.phi[i, j, k]
            
            # 只在界面區域計算表面張力
            if abs(phi_local) < 0.9:
                grad_mag = self.grad_phi[i, j, k].norm()
                
                if grad_mag > 1e-10:
                    # 表面張力力 = σ * κ * n * δ(界面)
                    delta_function = grad_mag  # 界面δ函數的近似
                    force_magnitude = config.SURFACE_TENSION * self.curvature[i, j, k] * delta_function
                    
                    self.surface_force[i, j, k] = force_magnitude * self.normal[i, j, k]
                else:
                    self.surface_force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            else:
                self.surface_force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.kernel
    def apply_phase_separation(self):
        """施加相分離效應 - 防止相混合"""
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            phi_local = self.phi[i, j, k]
            
            # Cahn-Hilliard方程的化學勢項
            if abs(phi_local) < 0.99:
                # 計算拉普拉斯算子
                laplacian = (self.phi[i+1, j, k] + self.phi[i-1, j, k] +
                           self.phi[i, j+1, k] + self.phi[i, j-1, k] +
                           self.phi[i, j, k+1] + self.phi[i, j, k-1] -
                           6.0 * phi_local)
                
                # 化學勢
                chemical_potential = phi_local * (phi_local**2 - 1.0) - 0.01 * laplacian
                
                # 更新相場
                self.phi_new[i, j, k] += -0.001 * chemical_potential * config.DT
    
    @ti.kernel
    def apply_surface_tension(self):
        """施加表面張力到LBM速度場"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.lbm.solid[i, j, k] == 0:  # 流體區域
                rho_local = self.lbm.rho[i, j, k]
                
                if rho_local > 1e-10:
                    acceleration = self.surface_force[i, j, k] / rho_local
                    self.lbm.u[i, j, k] += acceleration * config.DT
    
    @ti.kernel
    def update_density_from_phase(self):
        """根據相場更新密度場 - 並行處理"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            phi_local = self.phi[i, j, k]
            
            # 密度隨相場線性插值
            self.lbm.rho[i, j, k] = (config.RHO_WATER * (1.0 + phi_local) + 
                                   config.RHO_AIR * (1.0 - phi_local)) * 0.5
            
            # 更新相標記
            if phi_local > 0.1:
                self.lbm.phase[i, j, k] = config.PHASE_WATER
            elif phi_local < -0.1:
                self.lbm.phase[i, j, k] = config.PHASE_AIR
            else:
                self.lbm.phase[i, j, k] = (phi_local + 1.0) * 0.5
    
    @ti.kernel
    def copy_phase_field(self):
        """複製相場數據 - 並行內存操作"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            self.phi[i, j, k] = self.phi_new[i, j, k]
    
    def step(self):
        """執行多相流一個時間步長 - 完整並行流水線"""
        self.compute_gradients()
        self.compute_curvature()
        self.compute_surface_tension_force()
        self.apply_surface_tension()
        self.update_phase_field()
        self.apply_phase_separation()
        self.copy_phase_field()
        self.update_density_from_phase()