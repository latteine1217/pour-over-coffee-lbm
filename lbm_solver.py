# lbm_solver.py
"""
3D LBM求解器 - D3Q19模型 + LES湍流建模
專用於手沖咖啡模擬的格子Boltzmann方法實現
集成Smagorinsky LES模型用於高Reynolds數流動

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import config

# 導入LES湍流模型
if config.ENABLE_LES and config.RE_CHAR > config.LES_REYNOLDS_THRESHOLD:
    from les_turbulence import LESTurbulenceModel

@ti.data_oriented
class LBMSolver:
    """3D LBM求解器 - D3Q19模型"""
    
    def __init__(self):
        """初始化3D LBM求解器 + LES湍流建模"""
        print("初始化3D LBM求解器 (D3Q19)...")
        
        # 初始化3D場變數
        self._init_3d_fields()
        self._init_velocity_templates()
        
        # 初始化LES湍流模型
        if config.ENABLE_LES and config.RE_CHAR > config.LES_REYNOLDS_THRESHOLD:
            print("🌀 啟用LES湍流建模...")
            self.les_model = LESTurbulenceModel()
            self.use_les = True
        else:
            print("📐 使用純LBM (層流假設)...")
            self.les_model = None
            self.use_les = False
        
        print(f"D3Q19模型初始化完成 - 網格: {config.NX}×{config.NY}×{config.NZ}")
    
    def _init_3d_fields(self):
        """
        初始化3D場變數 (GPU記憶體優化版)
        使用SoA布局提升GPU coalesced access效率
        """
        # === 分布函數場 - SoA布局最佳化GPU訪問 ===
        self.f = ti.field(dtype=ti.f32, shape=(config.Q_3D, config.NX, config.NY, config.NZ))
        self.f_new = ti.field(dtype=ti.f32, shape=(config.Q_3D, config.NX, config.NY, config.NZ))
        
        # === 巨觀量場 ===
        self.rho = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.u = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.phase = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # === 幾何場 (記憶體對齊優化) ===
        self.solid = ti.field(dtype=ti.u8, shape=(config.NX, config.NY, config.NZ))
        
        # === 力場和耦合 ===
        self.body_force = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # === GPU常數記憶體 (coalesced訪問) ===
        self.cx = ti.field(dtype=ti.i32, shape=config.Q_3D)
        self.cy = ti.field(dtype=ti.i32, shape=config.Q_3D)
        self.cz = ti.field(dtype=ti.i32, shape=config.Q_3D)
        self.w = ti.field(dtype=ti.f32, shape=config.Q_3D)
        
        # === 兼容性速度向量數組 ===
        self.e = ti.Vector.field(3, dtype=ti.i32, shape=config.Q_3D)
        
        # === 性能優化緩存 ===
        self.u_sq = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))  # |u|²緩存
        self.opposite_dir = ti.field(dtype=ti.i32, shape=config.Q_3D)  # 相反方向查找表
        
        print("✅ GPU記憶體優化布局初始化完成")
    
    def _init_velocity_templates(self):
        """
        初始化3D速度模板 - GPU並行優化版
        預計算查找表，避免運行時計算
        """
        # 將numpy數組拷貝到GPU常數記憶體
        self.cx.from_numpy(config.CX_3D)
        self.cy.from_numpy(config.CY_3D) 
        self.cz.from_numpy(config.CZ_3D)
        self.w.from_numpy(config.WEIGHTS_3D)
        
        # 初始化兼容性速度向量數組
        self._init_e_vectors()
        
        # 預計算相反方向查找表 (GPU優化)
        self._compute_opposite_directions()
        
        print("✅ GPU常數記憶體載入完成")
    
    @ti.kernel
    def _init_e_vectors(self):
        """初始化速度向量數組用於兼容性"""
        for q in range(config.Q_3D):
            self.e[q] = ti.Vector([self.cx[q], self.cy[q], self.cz[q]])
    
    @ti.kernel
    def _compute_opposite_directions(self):
        """預計算相反方向查找表 - 避免運行時搜索"""
        for q in range(config.Q_3D):
            # 尋找相反方向 (-ex, -ey, -ez)
            for opp_q in range(config.Q_3D):
                if (self.cx[opp_q] == -self.cx[q] and 
                    self.cy[opp_q] == -self.cy[q] and 
                    self.cz[opp_q] == -self.cz[q]):
                    self.opposite_dir[q] = opp_q
                    break
    
    def init_fields(self):
        """初始化所有場變數"""
        self.init_3d_fields_kernel()
        print("✅ LBM場變數初始化完成")
    
    @ti.kernel
    def init_3d_fields_kernel(self):
        """
        3D場變數初始化 (CFD專家版)
        """
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 初始化密度場
            self.rho[i, j, k] = config.RHO_AIR
            
            # 初始化速度場
            self.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.u_sq[i, j, k] = 0.0
            
            # 初始化相場 (-1: 氣相, +1: 液相)
            self.phase[i, j, k] = -1.0  # 初始全為氣相
            
            # 初始化固體場
            self.solid[i, j, k] = ti.u8(0)
            
            # 初始化體力場
            self.body_force[i, j, k] = ti.Vector([0.0, 0.0, -config.GRAVITY_LU])
            
            # 初始化分布函數為平衡態
            for q in range(config.Q_3D):
                f_eq = self.w[q] * config.RHO_AIR  # 靜止平衡態
                self.f[q, i, j, k] = f_eq  # SoA訪問
                self.f_new[q, i, j, k] = f_eq
    
    def step_optimized(self):
        """
        優化的LBM時間步進 (CFD專家版)
        """
        # 執行融合的collision+streaming
        self._collision_streaming_step()
        
        # 場交換
        self.swap_fields()
        
        # LES湍流更新
        if self.use_les and hasattr(self, 'les_model') and self.les_model is not None:
            self.les_model.update_turbulent_viscosity(self.u)
    
    @ti.kernel
    def _collision_streaming_step(self):
        """融合collision+streaming的內核"""
        # 第一步：計算巨觀量
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.solid[i, j, k] == 0:  # 只處理流體節點
                # 計算密度
                rho_local = 0.0
                for q in range(config.Q_3D):
                    rho_local += self.f[q, i, j, k]  # SoA訪問模式
                self.rho[i, j, k] = rho_local
                
                # 計算速度
                u_local = ti.Vector([0.0, 0.0, 0.0])
                for q in range(config.Q_3D):
                    e_q = ti.cast(self.e[q], ti.f32)
                    u_local += self.f[q, i, j, k] * e_q  # SoA訪問模式
                
                if rho_local > 1e-12:
                    u_local /= rho_local
                    
                    # 包含體力 (Guo scheme) - 限制過大的體力影響
                    force = self.body_force[i, j, k]
                    tau = config.TAU_WATER if self.phase[i, j, k] > 0.5 else config.TAU_AIR
                    if rho_local > 1e-12:  # 雙重檢查
                        force_term = 0.5 * force * tau / rho_local
                        # 限制體力項的大小，避免數值不穩定
                        max_force_impact = 0.1  # 最大允許的速度變化
                        for d in ti.static(range(3)):
                            if ti.abs(force_term[d]) > max_force_impact:
                                sign_val = 1.0 if force_term[d] > 0.0 else -1.0
                                force_term[d] = max_force_impact * sign_val
                        u_local += force_term
                    
                self.u[i, j, k] = u_local
                self.u_sq[i, j, k] = u_local.norm_sqr()
        
        # 第二步：collision + streaming融合
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.solid[i, j, k] == 0:
                rho = self.rho[i, j, k]
                u = self.u[i, j, k]
                force = self.body_force[i, j, k]
                tau = config.TAU_WATER if self.phase[i, j, k] > 0.5 else config.TAU_AIR
                omega = 1.0 / tau
                
                for q in range(config.Q_3D):
                    # 計算平衡分布
                    f_eq = self.equilibrium_3d(i, j, k, q, rho, u)
                    
                    # Guo forcing
                    F_q = self._compute_guo_forcing(q, u, force, tau)
                    
                    # BGK collision with forcing
                    f_post = self.f[q, i, j, k] - omega * (self.f[q, i, j, k] - f_eq) + F_q  # SoA
                    
                    # Streaming到相鄰節點
                    ni = i + self.e[q][0]
                    nj = j + self.e[q][1]
                    nk = k + self.e[q][2]
                    
                    # 邊界檢查
                    if 0 <= ni < config.NX and 0 <= nj < config.NY and 0 <= nk < config.NZ:
                        if self.solid[ni, nj, nk] == 0:  # 流體節點
                            self.f_new[q, ni, nj, nk] = f_post  # SoA
                        else:  # 固體節點：bounce-back
                            opp_q = self.opposite_dir[q]
                            self.f_new[opp_q, i, j, k] = f_post  # SoA
                    # 出界：自然邊界條件
    
    @ti.kernel
    def swap_fields(self):
        """高效場交換 - GPU coalesced訪問優化"""
        for q, i, j, k in ti.ndrange(config.Q_3D, config.NX, config.NY, config.NZ):
            self.f[q, i, j, k], self.f_new[q, i, j, k] = self.f_new[q, i, j, k], self.f[q, i, j, k]
    
    @ti.func
    def equilibrium_3d(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32, 
                      rho: ti.f32, u: ti.template()) -> ti.f32:
        """
        計算D3Q19平衡分布函數 (科研級修正版)
        基於正確的Chapman-Enskog展開
        """
        e_q = ti.cast(self.e[q], ti.f32)
        w_q = self.w[q]
        
        eu = e_q.dot(u)
        u_sq = u.dot(u)
        
        return w_q * rho * (
            1.0 + 
            config.INV_CS2 * eu +
            4.5 * eu * eu -
            1.5 * u_sq
        )
    
    @ti.func
    def _compute_guo_forcing(self, q: ti.i32, u: ti.template(),
                            force: ti.template(), tau: ti.f32) -> ti.f32:
        """
        計算Guo forcing scheme源項
        F_i = w_i * (1 - 1/(2τ)) * (e_i - u)/cs² + (e_i·u)e_i/cs⁴) · F
        """
        e_q = ti.cast(self.e[q], ti.f32)
        w_q = self.w[q]
        
        # 安全檢查tau值
        tau_safe = ti.max(tau, 0.5001)  # 避免tau <= 0.5導致的數值問題
        
        # Guo forcing項
        eu = e_q.dot(u)
        ef = e_q.dot(force)
        uf = u.dot(force)
        
        forcing_term = 0.0
        
        # 限制forcing的大小，避免數值不穩定
        force_magnitude = ti.sqrt(force.dot(force))
        if force_magnitude <= 100.0:  # 只有在合理範圍內才計算
            temp_forcing = w_q * (1.0 - 1.0 / (2.0 * tau_safe)) * (
                config.INV_CS2 * ef + 
                config.INV_CS2 * config.INV_CS2 * eu * uf
            )
            
            # 檢查結果是否合理，如果合理就使用
            if ti.abs(temp_forcing) <= 10.0:
                forcing_term = temp_forcing
            
        return forcing_term    
         
    @ti.kernel
    def streaming_3d(self):
        """3D格子波茲曼流動步驟 (科研級)"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            for q in range(config.Q_3D):
                # 計算目標位置
                ni = i + self.e[q][0]
                nj = j + self.e[q][1] 
                nk = k + self.e[q][2]
                
                # 邊界檢查 - 僅對有效範圍進行流動
                if 0 <= ni < config.NX and 0 <= nj < config.NY and 0 <= nk < config.NZ:
                    self.f[ni, nj, nk, q] = self.f_new[i, j, k, q]
        
        # 交換緩衝區
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            for q in range(config.Q_3D):
                self.f_new[i, j, k, q] = self.f[i, j, k, q]
    
    def step(self):
        """執行一個LBM時間步 (科研級版本 + LES湍流建模)"""
        # 如果啟用LES，更新湍流黏性場
        if self.use_les and self.les_model is not None:
            self.les_model.update_turbulent_viscosity(self.u)
        
        # 使用融合的collision+streaming
        self.collision_streaming_fused()
        self.apply_boundary_conditions()  # 使用新的科研級邊界條件
    
    
    def step_with_particles(self, particle_system):
        """執行一個包含顆粒耦合的LBM時間步 (CFD專家版)"""
        # 執行優化的LBM時間步
        self.step_optimized()
        
        # 顆粒系統更新 (如果提供)
        if particle_system and hasattr(particle_system, 'update'):
            particle_system.update(self)
    
    def step(self):
        """標準LBM時間步"""
        self.step_optimized()
    
    @ti.kernel
    def apply_boundary_conditions(self):
        """應用邊界條件 - CFD專家版"""
        # 1. 固體邊界 - bounce-back邊界條件
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.solid[i, j, k] == 1:  # 固體節點
                # Bounce-back邊界條件
                for q in range(config.Q_3D):
                    # 找到相反方向
                    opp_q = self._get_opposite_direction(q)
                    # 交換分佈函數
                    temp = self.f[q, i, j, k]
                    self.f[q, i, j, k] = self.f[opp_q, i, j, k]
                    self.f[opp_q, i, j, k] = temp
        
        # 2. 頂部邊界 - 開放邊界 (自由流出)
        for i, j in ti.ndrange(config.NX, config.NY):
            k = config.NZ - 1  # 頂部
            if self.solid[i, j, k] == 0:  # 流體節點
                # 複製內部節點的速度和密度
                if k > 0 and self.solid[i, j, k-1] == 0:
                    self.rho[i, j, k] = self.rho[i, j, k-1]
                    self.u[i, j, k] = self.u[i, j, k-1]
                    
                    # 設置平衡分佈
                    for q in range(config.Q_3D):
                        self.f[q, i, j, k] = self._compute_equilibrium(
                            self.rho[i, j, k], self.u[i, j, k], q)
        
        # 3. 底部邊界 - 無滑移邊界條件
        for i, j in ti.ndrange(config.NX, config.NY):
            k = 0  # 底部
            if self.solid[i, j, k] == 0:  # 流體節點
                # 無滑移邊界條件：u = 0
                self.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                
                # 設置平衡分佈
                for q in range(config.Q_3D):
                    self.f[q, i, j, k] = self._compute_equilibrium(
                        self.rho[i, j, k], self.u[i, j, k], q)
        
        # 4. 側面邊界 - 週期性或無滑移
        # X邊界
        for j, k in ti.ndrange(config.NY, config.NZ):
            # 左邊界
            i = 0
            if self.solid[i, j, k] == 0:
                self.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            # 右邊界  
            i = config.NX - 1
            if self.solid[i, j, k] == 0:
                self.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        
        # Y邊界
        for i, k in ti.ndrange(config.NX, config.NZ):
            # 前邊界
            j = 0
            if self.solid[i, j, k] == 0:
                self.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            # 後邊界
            j = config.NY - 1
            if self.solid[i, j, k] == 0:
                self.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.func
    def _get_opposite_direction(self, q: ti.i32) -> ti.i32:
        """獲取相反方向的索引"""
        # D3Q19模型的相反方向映射
        opposite = ti.Vector([0, 2, 1, 4, 3, 6, 5, 8, 7, 
                             10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
        return opposite[q]
    
    @ti.func
    def _compute_equilibrium(self, rho: ti.f32, u: ti.template(), q: ti.i32) -> ti.f32:
        """計算平衡分佈函數"""
        w = self.weights[q]
        e_q = ti.cast(self.e[q], ti.f32)
        
        # 點積計算
        eu = e_q[0] * u[0] + e_q[1] * u[1] + e_q[2] * u[2]
        uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
        
        # Maxwell-Boltzmann平衡分佈
        f_eq = w * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu)
        return f_eq
    
    def get_velocity_magnitude(self):
        """獲取3D速度場大小"""
        u_data = self.u.to_numpy()
        return np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)