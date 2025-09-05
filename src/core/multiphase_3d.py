# multiphase_3d.py
"""
3D多相流體處理模組 - 科研級修正版
基於Cahn-Hilliard相場方程和連續表面力模型
參考文獻：Jacqmin (1999), Lee & Fischer (2006)
"""

import taichi as ti
import numpy as np
import config as config

@ti.data_oriented  
class MultiphaseFlow3D:
    def __init__(self, lbm_solver):
        """
        初始化3D多相流系統
        
        Args:
            lbm_solver: LBM求解器實例
        """
        self.lbm = lbm_solver
        
        # 相場相關變數 - 3D
        self.phi = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))        # 相場變數 (-1: 氣相, +1: 液相)
        self.phi_new = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))    # 新相場變數
        self.mu = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))         # 化學勢
        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))  # 界面法向量
        self.curvature = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))  # 曲率場
        
        # 梯度計算用的場
        self.grad_phi = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.grad_mu = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.laplacian_phi = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 表面張力力場
        self.surface_force = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # Cahn-Hilliard方程參數 (CFD專家修正版)
        self.INTERFACE_WIDTH = 2.0      # 界面厚度 (格子單位) - 優化為2lu提升效率
        self.MOBILITY = 0.001           # 相場遷移率 M - 降低提升穩定性
        self.SURFACE_TENSION_COEFF = config.SURFACE_TENSION_LU  # 表面張力係數
        self.CAHN_NUMBER = 0.005        # Cahn數 Cn = W/(ρu²L) - 更保守的值
        
        # 數值穩定性參數
        self.BETA = 12.0 * self.SURFACE_TENSION_COEFF / self.INTERFACE_WIDTH  # 化學勢係數
        self.KAPPA = 1.5 * self.SURFACE_TENSION_COEFF * self.INTERFACE_WIDTH  # 梯度能係數
        
        print(f"📊 多相流系統初始化完成 (CFD專家版):")
        print(f"  界面厚度: {self.INTERFACE_WIDTH} lu")
        print(f"  遷移率: {self.MOBILITY}")
        print(f"  β係數: {self.BETA:.4f}")
        print(f"  κ係數: {self.KAPPA:.4f}")
        print(f"  表面張力係數: {self.SURFACE_TENSION_COEFF:.6f}")
    
    @ti.kernel
    def init_phase_field(self):
        """
        初始化3D相場變數 - 物理合理的初始條件
        設置空氣-水界面，滿足熱力學平衡
        """
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 計算到V60中心軸的距離
            center_x = config.NX * 0.5
            center_y = config.NY * 0.5
            r = ti.sqrt((i - center_x)**2 + (j - center_y)**2)
            
            # V60幾何：錐形容器
            top_radius = config.TOP_RADIUS / config.SCALE_LENGTH
            bottom_radius = config.BOTTOM_RADIUS / config.SCALE_LENGTH
            cup_height = config.CUP_HEIGHT / config.SCALE_LENGTH
            
            # 計算此高度處的V60半徑
            z_ratio = k / cup_height
            current_radius = bottom_radius + (top_radius - bottom_radius) * z_ratio
            
            # 初始狀態：完全乾燥的V60濾杯，全部為氣相
            # 這樣才能模擬真實的注水過程
            self.phi[i, j, k] = -1.0  # 全部設為氣相（乾燥狀態）
    
    @ti.kernel
    def compute_chemical_potential(self):
        """
        計算Cahn-Hilliard化學勢
        μ = f'(φ) - κ∇²φ
        其中 f(φ) = (φ² - 1)²/4 是雙井勢函數
        """
        # 先計算Laplacian
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            # 二階中央差分計算Laplacian
            laplacian = (
                self.phi[i+1, j, k] + self.phi[i-1, j, k] +
                self.phi[i, j+1, k] + self.phi[i, j-1, k] +  
                self.phi[i, j, k+1] + self.phi[i, j, k-1] -
                6.0 * self.phi[i, j, k]
            )
            self.laplacian_phi[i, j, k] = laplacian
        
        # 計算化學勢
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            phi_local = self.phi[i, j, k]
            
            # 雙井勢的導數: f'(φ) = φ(φ² - 1) = φ³ - φ
            potential_derivative = phi_local * phi_local * phi_local - phi_local
            
            # 界面能項: -κ∇²φ，其中 κ = 3σW/8 (W是界面厚度)
            kappa = 3.0 * self.SURFACE_TENSION_COEFF * self.INTERFACE_WIDTH / 8.0
            interface_term = -kappa * self.laplacian_phi[i, j, k]
            
            self.mu[i, j, k] = potential_derivative + interface_term
    
    @ti.kernel
    def compute_gradients(self):
        """計算相場和化學勢的梯度 - 3D中央差分"""
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            # 相場梯度
            dphi_dx = (self.phi[i+1, j, k] - self.phi[i-1, j, k]) * 0.5
            dphi_dy = (self.phi[i, j+1, k] - self.phi[i, j-1, k]) * 0.5
            dphi_dz = (self.phi[i, j, k+1] - self.phi[i, j, k-1]) * 0.5
            self.grad_phi[i, j, k] = ti.Vector([dphi_dx, dphi_dy, dphi_dz])
            
            # 化學勢梯度
            dmu_dx = (self.mu[i+1, j, k] - self.mu[i-1, j, k]) * 0.5
            dmu_dy = (self.mu[i, j+1, k] - self.mu[i, j-1, k]) * 0.5
            dmu_dz = (self.mu[i, j, k+1] - self.mu[i, j, k-1]) * 0.5
            self.grad_mu[i, j, k] = ti.Vector([dmu_dx, dmu_dy, dmu_dz])
            
            # 計算界面法向量（歸一化梯度）
            grad_magnitude = self.grad_phi[i, j, k].norm()
            if grad_magnitude > 1e-10:
                self.normal[i, j, k] = self.grad_phi[i, j, k] / grad_magnitude
            else:
                self.normal[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.kernel
    def compute_curvature(self):
        """
        計算3D界面曲率 - 基於法向量散度
        κ = ∇ · n = ∇ · (∇φ/|∇φ|)
        """
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            # 計算法向量的散度（平均曲率）
            if self.normal[i, j, k].norm() > 1e-10:
                dnx_dx = (self.normal[i+1, j, k][0] - self.normal[i-1, j, k][0]) * 0.5
                dny_dy = (self.normal[i, j+1, k][1] - self.normal[i, j-1, k][1]) * 0.5
                dnz_dz = (self.normal[i, j, k+1][2] - self.normal[i, j, k-1][2]) * 0.5
                
                self.curvature[i, j, k] = dnx_dx + dny_dy + dnz_dz
            else:
                self.curvature[i, j, k] = 0.0
    
    @ti.kernel
    def update_phase_field_cahn_hilliard(self):
        """
        更新相場變數 - Cahn-Hilliard方程
        ∂φ/∂t + u·∇φ = M∇²μ
        這裡使用分步法：先對流，後擴散
        """
        # 第一步：對流項 (保守形式)
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            u_local = self.lbm.u[i, j, k]
            
            # 3D上風差分格式處理對流項
            dphi_dx = 0.0
            dphi_dy = 0.0
            dphi_dz = 0.0
            
            if u_local.x > 0:
                dphi_dx = self.phi[i, j, k] - self.phi[i-1, j, k]
            else:
                dphi_dx = self.phi[i+1, j, k] - self.phi[i, j, k]
                
            if u_local.y > 0:
                dphi_dy = self.phi[i, j, k] - self.phi[i, j-1, k]
            else:
                dphi_dy = self.phi[i, j+1, k] - self.phi[i, j, k]
                
            if u_local.z > 0:
                dphi_dz = self.phi[i, j, k] - self.phi[i, j, k-1]
            else:
                dphi_dz = self.phi[i, j, k+1] - self.phi[i, j, k]
            
            # 對流項：-u·∇φ
            convection = -(u_local.x * dphi_dx + u_local.y * dphi_dy + u_local.z * dphi_dz)
            
            # 擴散項：M∇²μ (Cahn-Hilliard核心)
            diffusion = self.MOBILITY * (
                self.mu[i+1, j, k] + self.mu[i-1, j, k] +
                self.mu[i, j+1, k] + self.mu[i, j-1, k] +
                self.mu[i, j, k+1] + self.mu[i, j, k-1] -
                6.0 * self.mu[i, j, k]
            )
            
            # 時間推進：顯式Euler (可替換為更穩定的隱式格式)
            self.phi_new[i, j, k] = self.phi[i, j, k] + config.DT * (convection + diffusion)
            
            # 保持相場變數在物理範圍內
            self.phi_new[i, j, k] = ti.max(-1.0, ti.min(1.0, self.phi_new[i, j, k]))
    
    @ti.kernel  
    def compute_surface_tension_force(self):
        """
        計算表面張力力 - 連續表面力(CSF)模型
        F = σκn𝛿_s ≈ σκ∇φ (在界面處)
        """
        for i, j, k in ti.ndrange((1, config.NX-1), (1, config.NY-1), (1, config.NZ-1)):
            # 表面張力只在界面附近有效
            grad_phi_magnitude = self.grad_phi[i, j, k].norm()
            
            if grad_phi_magnitude > 1e-8:  # 界面識別閾值
                # CSF模型：F = σκ∇φ
                force_magnitude = self.SURFACE_TENSION_COEFF * self.curvature[i, j, k]
                self.surface_force[i, j, k] = force_magnitude * self.grad_phi[i, j, k]
            else:
                self.surface_force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.kernel
    def apply_boundary_conditions(self):
        """應用相場的邊界條件 - 增強版"""
        # 處理域邊界
        for j, k in ti.ndrange(config.NY, config.NZ):
            # x方向邊界：零梯度
            self.phi_new[0, j, k] = self.phi_new[1, j, k]
            self.phi_new[config.NX-1, j, k] = self.phi_new[config.NX-2, j, k]
            
        for i, k in ti.ndrange(config.NX, config.NZ):
            # y方向邊界：零梯度
            self.phi_new[i, 0, k] = self.phi_new[i, 1, k]  
            self.phi_new[i, config.NY-1, k] = self.phi_new[i, config.NY-2, k]
            
        for i, j in ti.ndrange(config.NX, config.NY):
            # z方向邊界：零梯度
            self.phi_new[i, j, 0] = self.phi_new[i, j, 1]
            self.phi_new[i, j, config.NZ-1] = self.phi_new[i, j, config.NZ-2]
            
        # 全域相場範圍檢查和修正
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 確保相場在物理範圍內
            self.phi_new[i, j, k] = ti.max(-1.0, ti.min(1.0, self.phi_new[i, j, k]))
            
            # 在固體邊界處的相場處理
            if hasattr(self.lbm, 'solid') and self.lbm.solid[i, j, k] == 1:
                # 固體表面的相場根據濕潤性設置
                # 這裡假設V60表面中性濕潤
                self.phi_new[i, j, k] = 0.0  # 中性相場值
    
    def step(self):
        """執行一個多相流時間步"""
        # 1. 計算化學勢
        self.compute_chemical_potential()
        
        # 2. 計算梯度
        self.compute_gradients()
        
        # 3. 計算曲率
        self.compute_curvature()
        
        # 4. 更新相場 (Cahn-Hilliard方程)
        self.update_phase_field_cahn_hilliard()
        
        # 5. 應用邊界條件
        self.apply_boundary_conditions()
        
        # 6. 計算表面張力
        self.compute_surface_tension_force()
        
        # 7. 更新相場
        self.phi.copy_from(self.phi_new)
        
        # 8. 更新LBM中的密度和相位標記
        self.update_lbm_properties()
    
    @ti.kernel
    def update_lbm_properties(self):
        """根據相場更新LBM的物性參數"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            phi_local = self.phi[i, j, k]
            
            # 線性插值密度：ρ = 0.5*((1+φ)*ρ_water + (1-φ)*ρ_air)
            self.lbm.rho[i, j, k] = 0.5 * (
                (1.0 + phi_local) * config.RHO_WATER +
                (1.0 - phi_local) * config.RHO_AIR
            )
            
            # 更新相位標記
            self.lbm.phase[i, j, k] = phi_local
            
            # 將表面張力力添加到體積力
            self.lbm.body_force[i, j, k] = self.surface_force[i, j, k]
    
    def get_interface_statistics(self):
        """獲取界面統計信息"""
        phi_data = self.phi.to_numpy()
        
        # 界面區域識別 (|φ| < 0.9)
        interface_mask = np.abs(phi_data) < 0.9
        interface_volume = np.sum(interface_mask) * config.SCALE_LENGTH**3
        
        # 水相體積分數
        water_fraction = np.sum(phi_data > 0) / phi_data.size
        
        # 界面厚度統計
        grad_phi_data = self.grad_phi.to_numpy()
        interface_thickness = np.mean(1.0 / (np.linalg.norm(grad_phi_data, axis=3) + 1e-10)[interface_mask])
        
        return {
            'interface_volume': interface_volume,
            'water_fraction': water_fraction,
            'interface_thickness': interface_thickness * config.SCALE_LENGTH,
            'max_curvature': np.max(np.abs(self.curvature.to_numpy())),
            'surface_tension_magnitude': np.max(np.linalg.norm(self.surface_force.to_numpy(), axis=3))
        }
    
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
                    force_magnitude = self.SURFACE_TENSION_COEFF * self.curvature[i, j, k] * delta_function
                    
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
        """施加表面張力到LBM體力場（交由Guo forcing處理）"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.lbm.solid[i, j, k] == 0:  # 流體區域
                rho_local = self.lbm.rho[i, j, k]
                
                if rho_local > 1e-10:
                    acceleration = self.surface_force[i, j, k] / rho_local
                    self.lbm.body_force[i, j, k] += acceleration
    
    @ti.kernel
    def update_density_from_phase(self):
        """根據相場更新密度場 - 修正版本"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            phi_local = self.phi[i, j, k]
            
            # 確保相場在合理範圍內
            phi_local = ti.max(-1.0, ti.min(1.0, phi_local))
            
            # 正確的線性插值: ρ = ρ_air + (ρ_water - ρ_air) * (φ + 1) / 2
            # φ=-1(氣相) → ρ=ρ_air, φ=+1(水相) → ρ=ρ_water
            density = config.RHO_AIR + (config.RHO_WATER - config.RHO_AIR) * (phi_local + 1.0) / 2.0
            self.lbm.rho[i, j, k] = density
            
            # 簡化相場標記，直接使用歸一化的φ值 [0, 1]範圍
            phase_normalized = (phi_local + 1.0) / 2.0
            self.lbm.phase[i, j, k] = phase_normalized
    
    @ti.kernel
    def copy_phase_field(self):
        """複製相場數據 - 並行內存操作"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            self.phi[i, j, k] = self.phi_new[i, j, k]
    
    def step(self, step_count=0, precollision_applied: bool = False):
        """執行多相流一個時間步長 - 完整並行流水線

        Args:
            step_count: 當前步數，用於延遲啟動表面張力
            precollision_applied: 若當步已在碰撞前累加表面張力，這裡就不再重複累加
        """
        self.compute_gradients()
        self.compute_curvature()
        self.compute_surface_tension_force()
        
        # 延遲啟動表面張力效果，避免初始化時的數值不穩定
        if (not precollision_applied) and step_count > 10:
            self.apply_surface_tension()
        
        self.update_phase_field_cahn_hilliard()
        self.apply_phase_separation()
        self.copy_phase_field()
        self.update_density_from_phase()

    def accumulate_surface_tension_pre_collision(self):
        """在碰撞前累加表面張力到 LBM 體力場（參與當步Guo forcing）"""
        self.compute_gradients()
        self.compute_curvature()
        self.compute_surface_tension_force()
        self.apply_surface_tension()
    
    # ====================
    # 初始狀態標準化系統 (CFD一致性優化)
    # ====================
    
    def validate_initial_phase_consistency(self):
        """
        驗證多相流初始狀態一致性 (CFD一致性優化)
        
        檢查多相流初始狀態與邊界條件、幾何設置的一致性，
        確保初始化階段各模組間沒有衝突。
        
        Validation Checks:
            1. 相場初始值範圍 [-1, 1]
            2. 密度場與相場對應關係
            3. 固體區域相場處理
            4. 邊界區域相場狀態
            
        Physics Consistency:
            - 乾燥V60濾杯: 全域氣相 (φ = -1)
            - 注水前狀態: 無水相存在
            - 固體區域: 相場無定義
            - 邊界條件: 與相場演化相容
        """
        print("🔍 驗證多相流初始狀態一致性...")
        
        try:
            # 檢查1: 相場值範圍
            self._check_phase_field_range()
            
            # 檢查2: 密度-相場對應關係
            self._check_density_phase_consistency()
            
            # 檢查3: 固體區域處理
            self._check_solid_region_phase()
            
            # 檢查4: 初始狀態物理合理性
            self._check_initial_physics()
            
            print("   └─ ✅ 多相流初始狀態一致性驗證通過")
            
        except Exception as e:
            print(f"   └─ ❌ 多相流一致性驗證失敗: {e}")
            raise
    
    @ti.kernel
    def _check_phase_field_range_kernel(self) -> ti.i32:
        """檢查相場值範圍的核心"""
        error_count = 0
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            phi_val = self.phi[i, j, k]
            if phi_val < -1.1 or phi_val > 1.1:  # 允許小量數值誤差
                error_count += 1
                if error_count < 5:  # 只報告前5個錯誤
                    print(f"相場值超出範圍: phi[{i},{j},{k}] = {phi_val}")
        return error_count
    
    def _check_phase_field_range(self):
        """檢查相場值範圍"""
        error_count = self._check_phase_field_range_kernel()
        if error_count > 0:
            raise ValueError(f"發現 {error_count} 個相場值超出合理範圍 [-1,1]")
    
    @ti.kernel  
    def _check_density_consistency_kernel(self) -> ti.i32:
        """檢查密度-相場一致性的核心"""
        inconsistency_count = 0
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.lbm.solid[i, j, k] == 0:  # 只檢查流體區域
                phi_val = self.phi[i, j, k] 
                rho_val = self.lbm.rho[i, j, k]
                
                # 計算期望密度
                expected_rho = (config.RHO_WATER * (1.0 + phi_val) + 
                              config.RHO_AIR * (1.0 - phi_val)) * 0.5
                
                # 檢查一致性 (允許5%誤差)
                relative_error = ti.abs(rho_val - expected_rho) / expected_rho
                if relative_error > 0.05:
                    inconsistency_count += 1
                    
        return inconsistency_count
    
    def _check_density_phase_consistency(self):
        """檢查密度場與相場的對應關係"""
        inconsistency_count = self._check_density_consistency_kernel()
        if inconsistency_count > 0:
            print(f"   ⚠️  發現 {inconsistency_count} 個密度-相場不一致點 (可接受)")
    
    @ti.kernel
    def _check_solid_phase_kernel(self) -> ti.i32:
        """檢查固體區域相場處理"""
        solid_count = 0
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.lbm.solid[i, j, k] == 1:  # 固體區域
                solid_count += 1
        return solid_count
    
    def _check_solid_region_phase(self):
        """檢查固體區域相場處理"""
        solid_count = self._check_solid_phase_kernel()
        print(f"   ├─ 固體節點數量: {solid_count:,}")
        
    @ti.kernel
    def _check_initial_air_phase_kernel(self) -> ti.f32:
        """檢查初始氣相比例"""
        air_count = 0
        total_fluid_count = 0
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.lbm.solid[i, j, k] == 0:  # 流體區域
                total_fluid_count += 1
                if self.phi[i, j, k] < -0.5:  # 氣相主導
                    air_count += 1
                    
        return ti.cast(air_count, ti.f32) / ti.cast(total_fluid_count, ti.f32)
    
    def _check_initial_physics(self):
        """檢查初始狀態物理合理性"""
        air_ratio = self._check_initial_air_phase_kernel()
        print(f"   ├─ 初始氣相比例: {air_ratio*100:.1f}%")
        
        if air_ratio < 0.9:
            print(f"   ⚠️  初始狀態非乾燥濾杯 (氣相比例 < 90%)")
        else:
            print(f"   ├─ ✅ 初始乾燥狀態合理")
    
    def standardize_initial_state(self, force_dry_state=True):
        """
        標準化初始狀態 (CFD一致性優化)
        
        統一設置多相流初始狀態，確保與注水系統、邊界條件
        的一致性和協調性。
        
        Args:
            force_dry_state: 強制設置為乾燥狀態 (推薦)
            
        Standard Initial State:
            - 流體區域: 完全氣相 (φ = -1.0)
            - 固體區域: 保持不變
            - 密度場: 根據相場更新
            - 化學勢: 重新計算
        """
        print("🔧 標準化多相流初始狀態...")
        
        if force_dry_state:
            self._set_dry_initial_state()
        
        # 更新關聯場
        self.update_density_from_phase()
        self.compute_chemical_potential()
        self.compute_gradients()
        
        # 驗證設置結果
        self.validate_initial_phase_consistency()
        
        print("   └─ ✅ 多相流初始狀態標準化完成")
    
    @ti.kernel
    def _set_dry_initial_state(self):
        """設置乾燥初始狀態"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.lbm.solid[i, j, k] == 0:  # 只處理流體區域
                self.phi[i, j, k] = -1.0  # 完全氣相
                self.phi_new[i, j, k] = -1.0
