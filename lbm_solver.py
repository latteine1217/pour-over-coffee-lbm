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
        
        # 顆粒-流體耦合相關場
        self.body_force = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
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
        # 在Python範圍內調用V60幾何設置
        self.apply_v60_geometry()
    
    @ti.kernel
    def init_3d_fields_kernel(self):
        """3D場變數初始化"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            self.rho[i, j, k] = config.RHO_AIR
            self.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.phase[i, j, k] = config.PHASE_AIR
            self.solid[i, j, k] = 0
            
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
        """應用3D邊界條件 - 包含V60濾杯幾何"""
        # 首先設置V60濾杯壁邊界條件
        self.apply_v60_geometry()
        
        # 然後設置計算域外邊界
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 處理計算域邊界 (立方體邊界)
            if i == 0 or i == config.NX-1 or j == 0 or j == config.NY-1 or k == 0 or k == config.NZ-1:
                self.solid[i, j, k] = 1
                # 邊界處設置為反彈條件
                for q in range(config.Q_3D):
                    self.f[i, j, k, q] = self.f_new[i, j, k, q]
    
    @ti.kernel
    def apply_v60_geometry(self):
        """應用V60濾杯錐形邊界條件"""
        center_x = config.NX * 0.5
        center_y = config.NY * 0.5
        bottom_z = 5.0  # 濾杯底部位置（格子單位）
        
        # V60參數（轉換為格子單位）
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
        bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
        outlet_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH  # 出水孔半徑
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            x = ti.cast(i, ti.f32)
            y = ti.cast(j, ti.f32)
            z = ti.cast(k, ti.f32)
            
            # 計算距離中心的半徑
            radius_from_center = ti.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # 檢查是否在V60濾杯範圍內
            if z >= bottom_z and z <= bottom_z + cup_height_lu:
                # 計算在當前高度的V60內半徑
                height_ratio = (z - bottom_z) / cup_height_lu
                current_inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                
                # 如果在V60壁內側，保持流體；如果在壁外側，設為固體
                if radius_from_center > current_inner_radius:
                    self.solid[i, j, k] = 1  # V60壁外側為固體
                else:
                    # V60內部保持為流體區域
                    self.solid[i, j, k] = 0
            
            # V60底部處理 - 設置出水孔
            elif z >= bottom_z - 2 and z < bottom_z:
                if radius_from_center <= outlet_radius_lu:
                    # 出水孔區域保持為流體（允許水流出）
                    self.solid[i, j, k] = 0
                else:
                    # 出水孔外的底部設為固體
                    self.solid[i, j, k] = 1
            
            # V60底部以下區域設為固體
            elif z < bottom_z - 2:
                self.solid[i, j, k] = 1
            
            # V60頂部以上區域保持為流體（空氣區域）
            elif z > bottom_z + cup_height_lu:
                self.solid[i, j, k] = 0
    
    @ti.kernel
    def collision_with_particles(self):
        """碰撞算子 - 包含顆粒-流體耦合效應"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.solid[i, j, k] == 0:  # 流體區域
                rho_local = 0.0
                u_local = ti.Vector([0.0, 0.0, 0.0])
                
                 # 計算宏觀量
                for q in range(config.Q_3D):
                    rho_local += self.f[i, j, k, q]
                    u_local += ti.cast(self.e[q], ti.f32) * self.f[i, j, k, q]
                
                # 防止除零
                if rho_local > 1e-6:
                    u_local /= rho_local
                
                # 儲存宏觀量 (移除所有達西定律相關計算)
                self.rho[i, j, k] = rho_local
                self.u[i, j, k] = u_local
                
                # BGK碰撞 (使用標準鬆弛時間)
                for q in range(config.Q_3D):
                    # 平衡分佈函數
                    eu = ti.cast(self.e[q], ti.f32).dot(u_local)
                    uu = u_local.dot(u_local)
                    f_eq = self.w[q] * rho_local * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*uu)
                    
                    # BGK碰撞步驟 (使用標準水的鬆弛時間)
                    self.f_new[i, j, k, q] = self.f[i, j, k, q] - (self.f[i, j, k, q] - f_eq) / config.TAU_WATER
                    
                    # 確保分佈函數為正值
                    self.f_new[i, j, k, q] = ti.max(0.0, self.f_new[i, j, k, q])

    def step_with_particles(self, particle_system=None):
        """執行一個包含顆粒耦合的LBM時間步"""
        if particle_system is not None:
            # 1. 顆粒物理更新
            particle_system.step_particle_physics(config.DT, self)
            
            # 2. 顆粒-流體耦合碰撞
            self.collision_with_particles()
        else:
            # 3. 普通碰撞 (無顆粒耦合)
            self.collision_3d()
        
        # 4. 流動
        self.streaming_3d()
        
        # 5. 邊界條件
        self.apply_boundary_conditions_3d()
    
    def get_velocity_magnitude(self):
        """獲取3D速度場大小"""
        u_data = self.u.to_numpy()
        return np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)