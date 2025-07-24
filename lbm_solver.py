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
        3D場變數初始化 (CFD專家版) - 策略5修復版
        完全穩定的初始化，避免任何數值不穩定
        """
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 初始化密度場 - 使用穩定的初始密度
            self.rho[i, j, k] = 1.0  # 統一初始密度，避免密度跳躍
            
            # 初始化速度場 - 嚴格零初始化
            self.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.u_sq[i, j, k] = 0.0
            
            # 初始化相場 - 平滑過渡，避免劇烈界面
            self.phase[i, j, k] = 0.0  # 中性初始化，避免極端值
            
            # 初始化固體場
            self.solid[i, j, k] = ti.u8(0)
            
            # 初始化體力場 - 策略5：完全零初始化，避免任何體力
            self.body_force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            
            # 關鍵修復：穩定的平衡分布函數初始化
            rho_init = 1.0
            u_init = ti.Vector([0.0, 0.0, 0.0])
            
            for q in range(config.Q_3D):
                # 使用標準的平衡分布函數，避免任何偏差
                f_eq = self._compute_stable_equilibrium(q, rho_init, u_init)
                self.f[q, i, j, k] = f_eq  # SoA訪問
                self.f_new[q, i, j, k] = f_eq

    @ti.func
    def _compute_stable_equilibrium(self, q: ti.i32, rho: ti.f32, u: ti.template()) -> ti.f32:
        """
        計算穩定的平衡分布函數 - 策略5
        使用嚴格的數值穩定版本，避免任何計算偏差
        """
        w_q = self.w[q]
        
        # 計算平衡分布
        result = w_q * rho  # 預設為靜止態
        
        # 對於非靜止情況，計算完整平衡分布
        u_norm = u.norm()
        if u_norm >= 1e-15:
            e_q = ti.cast(self.e[q], ti.f32)
            eu = e_q.dot(u)
            u_sq = u.dot(u)
            
            # 使用安全的數值計算
            term1 = 1.0
            term2 = config.INV_CS2 * eu
            term3 = 4.5 * eu * eu  # = (3/2) * (eu/cs)^2
            term4 = -1.5 * u_sq    # = -(3/2) * u^2/cs^2
            
            result = w_q * rho * (term1 + term2 + term3 + term4)
        
        return result
    
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

    # 為了保持兼容性，添加普通step方法
    def step(self):
        """標準LBM步進方法"""
        return self.step_with_cfl_control()
    
    @ti.kernel
    def _collision_streaming_step(self):
        """融合collision+streaming的內核 - 修復版"""
        # 第一步：計算巨觀量
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.solid[i, j, k] == 0:  # 只處理流體節點
                # 計算密度
                rho_local = 0.0
                for q in range(config.Q_3D):
                    rho_local += self.f[q, i, j, k]  # SoA訪問模式
                self.rho[i, j, k] = rho_local
                
                # 計算速度 - 純粹從分布函數計算，不加重力
                u_local = ti.Vector([0.0, 0.0, 0.0])
                for q in range(config.Q_3D):
                    e_q = ti.cast(self.e[q], ti.f32)
                    u_local += self.f[q, i, j, k] * e_q  # SoA訪問模式
                
                if rho_local > 1e-12:
                    u_local /= rho_local
                    
                    # 注意：重力在collision步驟中應用，不在macroscopic量計算中
                    
                self.u[i, j, k] = u_local
                self.u_sq[i, j, k] = u_local.norm_sqr()
        
        # 第二步：collision + streaming融合
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.solid[i, j, k] == 0:
                rho = self.rho[i, j, k]
                u = self.u[i, j, k]
                phase_val = self.phase[i, j, k]
                
                # 只在水相區域計算體力 - 策略5：穩定重力計算
                force = ti.Vector([0.0, 0.0, 0.0])
                if phase_val > 0.01:  # 降低閾值，包含更多水相區域
                    # 使用修正後的重力，config.GRAVITY_LU已設為20%強度
                    gravity_strength = config.GRAVITY_LU * phase_val  # 移除10倍削弱
                    # 保守的重力限制，確保數值穩定性
                    max_gravity = 10.0  # 降低一個數量級
                    gravity_strength = ti.min(gravity_strength, max_gravity)
                    force = ti.Vector([0.0, 0.0, -gravity_strength])
                
                tau = config.TAU_WATER if phase_val > 0.5 else config.TAU_AIR
                omega = 1.0 / tau
                
                for q in range(config.Q_3D):
                    # 計算平衡分布
                    f_eq = self.equilibrium_3d(i, j, k, q, rho, u)
                    
                    # Guo forcing - 策略5：安全的forcing計算
                    F_q = 0.0
                    if force.norm() > 1e-15:  # 更嚴格的閾值
                        F_q = self._compute_stable_guo_forcing(q, u, force, tau)
                        # 保守限制forcing項，確保數值穩定性
                        max_forcing = 0.01  # 降低一個數量級
                        F_q = ti.max(-max_forcing, ti.min(max_forcing, F_q))
                    
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
    def _compute_stable_guo_forcing(self, q: ti.i32, u: ti.template(),
                                  force: ti.template(), tau: ti.f32) -> ti.f32:
        """
        策略5：穩定的Guo forcing計算
        使用保守的數值方法避免發散
        """
        e_q = ti.cast(self.e[q], ti.f32)
        w_q = self.w[q]
        
        # 安全檢查tau值 - 更嚴格的限制
        tau_safe = ti.max(tau, 0.6)  # 更保守的tau下限
        tau_safe = ti.min(tau_safe, 1.5)  # 添加tau上限
        
        # 安全檢查輸入大小
        force_norm = force.norm()
        u_norm = u.norm()
        
        forcing_result = 0.0
        
        # 保守的forcing計算範圍，確保數值穩定性
        if force_norm <= 10.0 and u_norm <= 0.1:  # 降低一個數量級
            # Guo forcing項計算
            eu = e_q.dot(u)
            ef = e_q.dot(force)
            uf = u.dot(force)
            
            # 分步計算，避免數值溢出
            coeff = w_q * (1.0 - 0.5 / tau_safe)
            term1 = config.INV_CS2 * ef
            term2 = config.INV_CS2 * config.INV_CS2 * eu * uf
            
            temp_result = coeff * (term1 + term2)
            
            # 最終安全檢查
            if ti.abs(temp_result) <= 1e-6:
                forcing_result = temp_result
        
        return forcing_result    
         
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
        self._collision_streaming_step()
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
        """應用邊界條件 - CFD專家版 (修復版)"""
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
        
        # 2. 頂部邊界 - 開放邊界 (自由流出) - 修復版
        for i, j in ti.ndrange(config.NX, config.NY):
            k = config.NZ - 1  # 頂部
            if self.solid[i, j, k] == 0:  # 流體節點
                # 只複製內部節點的密度，速度保持當前計算值
                if k > 0 and self.solid[i, j, k-1] == 0:
                    self.rho[i, j, k] = self.rho[i, j, k-1]
                    # 移除有問題的速度複製：不要複製速度，讓LBM自然演化
                    # self.u[i, j, k] = self.u[i, j, k-1]  # 刪除這行
                    
                    # 基於當前速度重新計算平衡分佈
                    for q in range(config.Q_3D):
                        self.f[q, i, j, k] = self._compute_equilibrium(
                            self.rho[i, j, k], self.u[i, j, k], q)
        
        # 3. 底部邊界 - 完全固體邊界（無outlet）
        for i, j in ti.ndrange(config.NX, config.NY):
            k = 0  # 底部
            if self.solid[i, j, k] == 0:  # 如果是流體節點，改為固體
                # 底部完全封閉，設為bounce-back邊界
                self.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                for q in range(config.Q_3D):
                    opp_q = self.opposite_dir[q]
                    # Bounce-back邊界條件
                    temp = self.f[q, i, j, k]
                    self.f[q, i, j, k] = self.f[opp_q, i, j, k]
                    self.f[opp_q, i, j, k] = temp
        
        # 4. 計算域邊界outlet條件 - 自由流出邊界
        # X邊界 - 改為outlet邊界
        for j, k in ti.ndrange(config.NY, config.NZ):
            # 左邊界 - outlet邊界條件
            i = 0
            if self.solid[i, j, k] == 0:
                # 外推邊界條件：從內部節點外推密度和速度
                if i + 1 < config.NX and self.solid[i+1, j, k] == 0:
                    self.rho[i, j, k] = self.rho[i+1, j, k]
                    self.u[i, j, k] = self.u[i+1, j, k]
                # 更新分佈函數
                for q in range(config.Q_3D):
                    self.f[q, i, j, k] = self._compute_equilibrium(
                        self.rho[i, j, k], self.u[i, j, k], q)
            
            # 右邊界 - outlet邊界條件
            i = config.NX - 1
            if self.solid[i, j, k] == 0:
                # 外推邊界條件
                if i - 1 >= 0 and self.solid[i-1, j, k] == 0:
                    self.rho[i, j, k] = self.rho[i-1, j, k]
                    self.u[i, j, k] = self.u[i-1, j, k]
                # 更新分佈函數
                for q in range(config.Q_3D):
                    self.f[q, i, j, k] = self._compute_equilibrium(
                        self.rho[i, j, k], self.u[i, j, k], q)
        
        # Y邊界 - outlet邊界條件
        for i, k in ti.ndrange(config.NX, config.NZ):
            # 前邊界 - outlet邊界條件
            j = 0
            if self.solid[i, j, k] == 0:
                # 外推邊界條件
                if j + 1 < config.NY and self.solid[i, j+1, k] == 0:
                    self.rho[i, j, k] = self.rho[i, j+1, k]
                    self.u[i, j, k] = self.u[i, j+1, k]
                # 更新分佈函數
                for q in range(config.Q_3D):
                    self.f[q, i, j, k] = self._compute_equilibrium(
                        self.rho[i, j, k], self.u[i, j, k], q)
            
            # 後邊界 - outlet邊界條件
            j = config.NY - 1
            if self.solid[i, j, k] == 0:
                # 外推邊界條件
                if j - 1 >= 0 and self.solid[i, j-1, k] == 0:
                    self.rho[i, j, k] = self.rho[i, j-1, k]
                    self.u[i, j, k] = self.u[i, j-1, k]
                # 更新分佈函數
                for q in range(config.Q_3D):
                    self.f[q, i, j, k] = self._compute_equilibrium(
                        self.rho[i, j, k], self.u[i, j, k], q)
    
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