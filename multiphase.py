# multiphase.py
"""
多相流體處理模組
實現空氣-水界面的追蹤和表面張力效應
使用相場方法（Phase Field Method）結合LBM
"""

import taichi as ti
import numpy as np
import config

@ti.data_oriented
class MultiphaseFlow:
    def __init__(self, lbm_solver):
        self.lbm = lbm_solver
        
        # 相場相關變數
        self.phi = ti.field(dtype=ti.f32, shape=(config.NX, config.NY))        # 相場變數
        self.phi_new = ti.field(dtype=ti.f32, shape=(config.NX, config.NY))    # 新相場變數
        self.normal = ti.Vector.field(2, dtype=ti.f32, shape=(config.NX, config.NY))  # 界面法向量
        self.curvature = ti.field(dtype=ti.f32, shape=(config.NX, config.NY))  # 曲率場
        
        # 梯度計算用的臨時場
        self.grad_phi = ti.Vector.field(2, dtype=ti.f32, shape=(config.NX, config.NY))
    
    @ti.kernel
    def init_phase_field(self):
        """初始化相場變數"""
        for i, j in self.phi:
            # 相場變數：-1為空氣，+1為水，0為界面
            if j > config.NY * 0.8:
                self.phi[i, j] = 1.0  # 水相
            elif j < config.NY * 0.2:
                self.phi[i, j] = -1.0  # 空氣相
            else:
                # 界面區域使用tanh函數平滑過渡
                y_center = config.NY * 0.5
                self.phi[i, j] = ti.tanh(2.0 * (j - y_center) / config.INTERFACE_THICKNESS)
    
    @ti.kernel
    def compute_gradients(self):
        """計算相場梯度"""
        for i, j in self.grad_phi:
            # 中央差分計算梯度
            if 1 <= i < config.NX - 1 and 1 <= j < config.NY - 1:
                dphi_dx = (self.phi[i+1, j] - self.phi[i-1, j]) * 0.5
                dphi_dy = (self.phi[i, j+1] - self.phi[i, j-1]) * 0.5
                self.grad_phi[i, j] = ti.Vector([dphi_dx, dphi_dy])
                
                # 計算界面法向量（歸一化梯度）
                grad_magnitude = ti.sqrt(dphi_dx**2 + dphi_dy**2)
                if grad_magnitude > 1e-10:
                    self.normal[i, j] = self.grad_phi[i, j] / grad_magnitude
                else:
                    self.normal[i, j] = ti.Vector([0.0, 0.0])
    
    @ti.kernel
    def compute_curvature(self):
        """計算界面曲率"""
        for i, j in self.curvature:
            if 1 <= i < config.NX - 1 and 1 <= j < config.NY - 1:
                # 計算法向量的散度來得到曲率
                dnx_dx = (self.normal[i+1, j][0] - self.normal[i-1, j][0]) * 0.5
                dny_dy = (self.normal[i, j+1][1] - self.normal[i, j-1][1]) * 0.5
                self.curvature[i, j] = dnx_dx + dny_dy
    
    @ti.kernel
    def update_phase_field(self):
        """更新相場變數"""
        for i, j in self.phi:
            if 1 <= i < config.NX - 1 and 1 <= j < config.NY - 1:
                # Cahn-Hilliard 方程的簡化形式
                # 基於速度場對流相場
                u_local = self.lbm.u[i, j]
                
                # 上風差分格式
                dphi_dx = 0.0
                dphi_dy = 0.0
                
                if u_local[0] > 0:
                    dphi_dx = self.phi[i, j] - self.phi[i-1, j]
                else:
                    dphi_dx = self.phi[i+1, j] - self.phi[i, j]
                
                if u_local[1] > 0:
                    dphi_dy = self.phi[i, j] - self.phi[i, j-1]
                else:
                    dphi_dy = self.phi[i, j+1] - self.phi[i, j]
                
                # 對流項
                advection = u_local[0] * dphi_dx + u_local[1] * dphi_dy
                
                # 更新相場
                self.phi_new[i, j] = self.phi[i, j] - config.DT * advection
                
                # 限制相場值在[-1, 1]範圍內
                self.phi_new[i, j] = ti.max(-1.0, ti.min(1.0, self.phi_new[i, j]))
    
    @ti.kernel
    def apply_surface_tension(self):
        """施加表面張力力"""
        for i, j in self.lbm.u:
            if 1 <= i < config.NX - 1 and 1 <= j < config.NY - 1:
                # 檢查是否在界面附近
                phi_local = self.phi[i, j]
                if abs(phi_local) < 0.9:  # 界面區域
                    # 計算表面張力力
                    grad_mag = ti.sqrt(self.grad_phi[i, j][0]**2 + self.grad_phi[i, j][1]**2)
                    
                    if grad_mag > 1e-10:
                        # 表面張力力 = σ * κ * n * δ(界面)
                        delta_function = grad_mag  # 界面δ函數的近似
                        surface_force = config.SURFACE_TENSION * self.curvature[i, j] * self.normal[i, j] * delta_function
                        
                        # 施加到速度場
                        rho_local = self.lbm.rho[i, j]
                        if rho_local > 1e-10:
                            acceleration = surface_force / rho_local
                            self.lbm.u[i, j] += acceleration * config.DT
    
    @ti.kernel
    def update_density_from_phase(self):
        """根據相場更新密度場"""
        for i, j in self.lbm.rho:
            phi_local = self.phi[i, j]
            # 密度隨相場線性插值
            self.lbm.rho[i, j] = (config.RHO_WATER * (1.0 + phi_local) + config.RHO_AIR * (1.0 - phi_local)) * 0.5
            
            # 更新相標記
            if phi_local > 0.1:
                self.lbm.phase[i, j] = config.PHASE_WATER
            elif phi_local < -0.1:
                self.lbm.phase[i, j] = config.PHASE_AIR
            else:
                # 界面區域
                self.lbm.phase[i, j] = (phi_local + 1.0) * 0.5
    
    @ti.kernel
    def copy_phase_field(self):
        """複製相場數據"""
        for i, j in self.phi:
            self.phi[i, j] = self.phi_new[i, j]
    
    def step(self):
        """執行多相流一個時間步長"""
        self.compute_gradients()
        self.compute_curvature()
        self.apply_surface_tension()
        self.update_phase_field()
        self.copy_phase_field()
        self.update_density_from_phase()