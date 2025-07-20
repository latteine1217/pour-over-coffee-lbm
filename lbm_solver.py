# lbm_solver.py
"""
3D LBM求解器 - D3Q19模型
專用於手沖咖啡模擬的格子Boltzmann方法實現
"""

import taichi as ti
import numpy as np
import config

@ti.data_oriented
class LBMSolver:
    """3D LBM求解器 - D3Q19模型"""
    
    def __init__(self):
        """初始化3D LBM求解器"""
        print("初始化3D LBM求解器 (D3Q19)...")
        
        # 初始化3D場變數
        self._init_3d_fields()
        self._init_velocity_templates()
        
        print(f"D3Q19模型初始化完成 - 網格: {config.NX}×{config.NY}×{config.NZ}")
    
    def _init_3d_fields(self):
        """初始化3D場變數 (D3Q19)"""
        # 主要場變數
        self.f = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ, config.Q_3D))
        self.f_new = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ, config.Q_3D))
        self.rho = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.u = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.phase = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 幾何和邊界標記
        self.solid = ti.field(dtype=ti.i32, shape=(config.NX, config.NY, config.NZ))
        self.porous = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # D3Q19速度模板
        self.e = ti.Vector.field(3, dtype=ti.i32, shape=config.Q_3D)
        self.w = ti.field(dtype=ti.f32, shape=config.Q_3D)
    
    def _init_velocity_templates(self):
        """初始化3D速度模板"""
        # 將numpy數組拷貝到Taichi field
        self.e.from_numpy(np.column_stack([config.CX_3D, config.CY_3D, config.CZ_3D]))
        self.w.from_numpy(config.WEIGHTS_3D)
    
    def init_fields(self):
        """初始化所有場變數"""
        self.init_3d_fields_kernel()
    
    @ti.kernel
    def init_3d_fields_kernel(self):
        """3D場變數初始化"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            self.rho[i, j, k] = config.RHO_AIR
            self.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.phase[i, j, k] = config.PHASE_AIR
            self.solid[i, j, k] = 0
            self.porous[i, j, k] = 1.0
            
            # 在上部區域初始化水相
            if k > config.NZ * 0.8:
                self.rho[i, j, k] = config.RHO_WATER
                self.phase[i, j, k] = config.PHASE_WATER
            
            # 初始化分佈函數為平衡態
            for q in range(config.Q_3D):
                self.f[i, j, k, q] = self.equilibrium_3d(i, j, k, q)
                self.f_new[i, j, k, q] = self.f[i, j, k, q]
    
    @ti.func
    def equilibrium_3d(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32) -> ti.f32:
        """計算3D平衡分佈函數"""
        rho_local = self.rho[i, j, k]
        u_local = self.u[i, j, k]
        e_q = self.e[q]
        w_q = self.w[q]
        
        eu = e_q.dot(u_local)
        u_sq = u_local.dot(u_local)
        
        return w_q * rho_local * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq)
    
    @ti.kernel
    def collision_3d(self):
        """3D BGK碰撞算子"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.solid[i, j, k] == 0:  # 非固體區域
                # 計算巨觀量
                rho_new = 0.0
                u_new = ti.Vector([0.0, 0.0, 0.0])
                
                for q in range(config.Q_3D):
                    rho_new += self.f[i, j, k, q]
                    u_new += self.f[i, j, k, q] * ti.cast(self.e[q], ti.f32)
                
                # 數值穩定性檢查
                if rho_new < config.RHO_AIR * 0.1:
                    rho_new = config.RHO_AIR
                elif rho_new > config.RHO_WATER * 5.0:
                    rho_new = config.RHO_WATER
                
                self.rho[i, j, k] = rho_new
                if rho_new > 1e-6:
                    u_new = u_new / rho_new
                    # 限制最大速度防止不穩定
                    u_mag = u_new.norm()
                    if u_mag > 0.1:  # 限制最大速度
                        u_new = u_new / u_mag * 0.1
                    self.u[i, j, k] = u_new
                
                # BGK碰撞
                tau = config.TAU_WATER if self.phase[i, j, k] > 0.5 else config.TAU_AIR
                
                for q in range(config.Q_3D):
                    f_eq = self.equilibrium_3d(i, j, k, q)
                    
                    # 添加重力源項 (Guo forcing scheme)
                    gravity_force = ti.Vector([0.0, 0.0, -config.GRAVITY_LU])  # z方向向下重力
                    c_vec = ti.Vector([ti.cast(self.e[q][0], ti.f32), 
                                      ti.cast(self.e[q][1], ti.f32), 
                                      ti.cast(self.e[q][2], ti.f32)])
                    force_term = self.w[q] * rho_new * (1.0 - 1.0/(2.0*tau)) * \
                                (c_vec.dot(gravity_force) / (0.333333) + 
                                 (c_vec.dot(gravity_force) * c_vec.dot(u_new)) / (0.111111) - 
                                 u_new.dot(gravity_force) / (0.333333))
                    
                    # 添加穩定性檢查
                    f_new = self.f[i, j, k, q] - (self.f[i, j, k, q] - f_eq) / tau + force_term * config.DT
                    if f_new < 0.0:
                        f_new = 0.0
                    self.f_new[i, j, k, q] = f_new
    
    @ti.kernel
    def streaming_3d(self):
        """3D流動步驟"""
        for i, j, k, q in ti.ndrange(config.NX, config.NY, config.NZ, config.Q_3D):
            e_q = self.e[q]
            i_new = i - e_q[0]
            j_new = j - e_q[1]
            k_new = k - e_q[2]
            
            if 0 <= i_new < config.NX and 0 <= j_new < config.NY and 0 <= k_new < config.NZ:
                if self.solid[i, j, k] == 0:
                    self.f[i, j, k, q] = self.f_new[i_new, j_new, k_new, q]
    
    @ti.kernel 
    def apply_boundary_conditions_3d(self):
        """應用3D邊界條件"""
        # 邊界反彈條件
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 處理邊界
            if i == 0 or i == config.NX-1 or j == 0 or j == config.NY-1 or k == 0 or k == config.NZ-1:
                self.solid[i, j, k] = 1
                # 邊界處設置為反彈條件
                for q in range(config.Q_3D):
                    self.f[i, j, k, q] = self.f_new[i, j, k, q]
    
    def step(self):
        """執行一個LBM時間步"""
        self.collision_3d()
        self.streaming_3d()
        self.apply_boundary_conditions_3d()
    
    def get_velocity_magnitude(self):
        """獲取3D速度場大小"""
        u_data = self.u.to_numpy()
        return np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)