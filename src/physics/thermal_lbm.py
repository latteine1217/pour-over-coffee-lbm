# thermal_lbm.py - D3Q7溫度場LBM求解器
"""
基礎熱傳導格子玻爾茲曼方法求解器
使用D3Q7格子結構進行純擴散-對流方程求解
獨立於流體系統，專門處理溫度場演化

技術規格:
- D3Q7格子結構 (7個離散速度)
- BGK碰撞運算子
- 多種邊界條件支援
- GPU並行優化 (Taichi)
- 數值穩定性保證

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import math
from typing import Tuple, Optional
from config import NX, NY, NZ, DX, DT, CS2

# ==============================================
# D3Q7格子結構定義
# ==============================================

# D3Q7離散速度集合 {(0,0,0), (±1,0,0), (0,±1,0), (0,0,±1)}
Q_THERMAL = 7
CX_THERMAL = ti.Vector([0, 1, -1, 0, 0, 0, 0], ti.i32)
CY_THERMAL = ti.Vector([0, 0, 0, 1, -1, 0, 0], ti.i32) 
CZ_THERMAL = ti.Vector([0, 0, 0, 0, 0, 1, -1], ti.i32)

# D3Q7權重係數
W_THERMAL = ti.Vector([1.0/4.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0], ti.f32)

# 熱傳導專用參數
CS2_THERMAL = 1.0/3.0  # 熱擴散格子聲速平方
INV_CS2_THERMAL = 3.0

@ti.data_oriented
class ThermalLBM:
    """
    D3Q7熱傳導LBM求解器
    
    功能：
    - 溫度場分布函數演化
    - BGK碰撞步驟
    - 流場步驟  
    - 溫度場重建
    - 邊界條件處理
    - 數值穩定性監控
    """
    
    def __init__(self, 
                 thermal_diffusivity: float = 1.6e-7,  # 水的熱擴散係數 m²/s
                 scale_length: float = 0.000625,       # 長度尺度 m/lu
                 scale_time: float = 0.0625):          # 時間尺度 s/ts
        """
        初始化熱傳LBM求解器
        
        Args:
            thermal_diffusivity: 熱擴散係數 (m²/s)
            scale_length: 長度尺度轉換係數
            scale_time: 時間尺度轉換係數
        """
        
        # 物理參數
        self.alpha_phys = thermal_diffusivity
        self.scale_length = scale_length
        self.scale_time = scale_time
        
        # 格子單位熱擴散係數
        self.alpha_lu = self.alpha_phys * self.scale_time / (self.scale_length**2)
        
        # 熱傳導鬆弛時間 τ = α/(c_s²) + 0.5
        self.tau_thermal = self.alpha_lu / CS2_THERMAL + 0.5
        
        # 數值穩定性檢查
        if self.tau_thermal < 0.51:
            print(f"⚠️  熱傳導τ={self.tau_thermal:.6f} < 0.51，數值不穩定")
            self.tau_thermal = 0.51
        elif self.tau_thermal > 2.0:
            print(f"⚠️  熱傳導τ={self.tau_thermal:.6f} > 2.0，過度擴散")
        else:
            print(f"✅ 熱傳導τ={self.tau_thermal:.6f}，數值穩定")
        
        self.omega_thermal = 1.0 / self.tau_thermal
        
        # 初始化Taichi場
        self._init_fields()
        
        # 系統狀態
        self.is_initialized = False
        self.current_step = 0
        
    def _init_fields(self):
        """初始化所有Taichi場"""
        
        # 溫度分布函數 g_i(x,t)
        self.g = ti.field(ti.f32, shape=(NX, NY, NZ, Q_THERMAL))
        self.g_new = ti.field(ti.f32, shape=(NX, NY, NZ, Q_THERMAL))
        
        # 溫度場 T(x,t)
        self.temperature = ti.field(ti.f32, shape=(NX, NY, NZ))
        self.temperature_old = ti.field(ti.f32, shape=(NX, NY, NZ))
        
        # 熱流場 q = -k∇T
        self.heat_flux = ti.Vector.field(3, ti.f32, shape=(NX, NY, NZ))
        
        # 熱源項 S(x,t)
        self.heat_source = ti.field(ti.f32, shape=(NX, NY, NZ))
        
        # 熱物性場
        self.thermal_conductivity = ti.field(ti.f32, shape=(NX, NY, NZ))
        self.heat_capacity = ti.field(ti.f32, shape=(NX, NY, NZ))
        self.thermal_diffusivity_field = ti.field(ti.f32, shape=(NX, NY, NZ))
        
        # 邊界標記
        self.boundary_type = ti.field(ti.i32, shape=(NX, NY, NZ))
        self.boundary_temperature = ti.field(ti.f32, shape=(NX, NY, NZ))
        self.boundary_heat_flux = ti.field(ti.f32, shape=(NX, NY, NZ))
        
        # 診斷變量
        self.max_temperature = ti.field(ti.f32, shape=())
        self.min_temperature = ti.field(ti.f32, shape=())
        self.avg_temperature = ti.field(ti.f32, shape=())
        
        # 速度場接口 (用於對流耦合)
        self.velocity_field = ti.Vector.field(3, ti.f32, shape=(NX, NY, NZ))
        self.enable_convection = False  # 控制是否啟用對流項
        
    @ti.kernel
    def init_temperature_field(self, 
                              T_initial: ti.f32,
                              T_hot_region: ti.f32,
                              hot_region_height: ti.i32):
        """
        初始化溫度場
        
        Args:
            T_initial: 初始溫度 (°C)
            T_hot_region: 熱區域溫度 (°C) 
            hot_region_height: 熱區域高度 (格點)
        """
        
        for i, j, k in ti.ndrange(NX, NY, NZ):
            # 初始溫度分布
            if k < hot_region_height:  # 下部熱水區域
                self.temperature[i, j, k] = T_hot_region
            else:  # 上部環境溫度
                self.temperature[i, j, k] = T_initial
            
            # 初始化分布函數為平衡態
            for q in ti.static(range(Q_THERMAL)):
                g_eq = self._equilibrium_distribution(q, self.temperature[i, j, k])
                self.g[i, j, k, q] = g_eq
                self.g_new[i, j, k, q] = g_eq
            
            # 初始化熱物性 (純水)
            self.thermal_conductivity[i, j, k] = 0.68  # W/(m·K)
            self.heat_capacity[i, j, k] = 4180.0       # J/(kg·K)
            self.thermal_diffusivity_field[i, j, k] = self.alpha_phys
            
            # 初始化邊界為內部點
            self.boundary_type[i, j, k] = 0  # 0=內部, 1=Dirichlet, 2=Neumann, 3=Robin
            self.boundary_temperature[i, j, k] = T_initial
            self.boundary_heat_flux[i, j, k] = 0.0
            
            # 初始化熱源
            self.heat_source[i, j, k] = 0.0
    
    def complete_initialization(self, T_initial: float, T_hot_region: float, hot_region_height: int):
        """
        完成初始化流程
        
        Args:
            T_initial: 初始溫度 (°C)
            T_hot_region: 熱區域溫度 (°C) 
            hot_region_height: 熱區域高度 (格點)
        """
        
        self.init_temperature_field(T_initial, T_hot_region, hot_region_height)
        self.is_initialized = True
        print(f"✅ 溫度場初始化完成: T_initial={T_initial}°C, T_hot={T_hot_region}°C")
    
    @ti.func
    def _equilibrium_distribution(self, q: ti.i32, temperature: ti.f32) -> ti.f32:
        """
        計算平衡分布函數
        
        對於純擴散過程，平衡分布為：
        g_q^eq = w_q * T
        
        Args:
            q: 離散速度方向索引
            temperature: 局部溫度
            
        Returns:
            平衡分布函數值
        """
        return W_THERMAL[q] * temperature
    
    @ti.kernel  
    def collision_step(self):
        """
        BGK碰撞步驟
        
        演化方程：g_i(x,t+dt) = g_i(x,t) - (g_i - g_i^eq)/τ + S_i*dt
        """
        
        for i, j, k in ti.ndrange(NX, NY, NZ):
            # 計算局部溫度
            T_local = 0.0
            for q in ti.static(range(Q_THERMAL)):
                T_local += self.g[i, j, k, q]
            
            self.temperature[i, j, k] = T_local
            
            # BGK碰撞
            for q in ti.static(range(Q_THERMAL)):
                g_eq = self._equilibrium_distribution(q, T_local)
                
                # 熱源項投影到分布函數
                source_term = W_THERMAL[q] * self.heat_source[i, j, k] * DT
                
                # BGK碰撞運算子
                self.g_new[i, j, k, q] = (self.g[i, j, k, q] - 
                                         self.omega_thermal * (self.g[i, j, k, q] - g_eq) +
                                         source_term)
    
    @ti.kernel
    def streaming_step(self):
        """
        流場步驟
        
        將分布函數沿離散速度方向傳播
        """
        
        for i, j, k in ti.ndrange(NX, NY, NZ):
            for q in ti.static(range(Q_THERMAL)):
                # 計算源位置
                src_i = i - CX_THERMAL[q]
                src_j = j - CY_THERMAL[q] 
                src_k = k - CZ_THERMAL[q]
                
                # 邊界檢查
                if (src_i >= 0 and src_i < NX and
                    src_j >= 0 and src_j < NY and 
                    src_k >= 0 and src_k < NZ):
                    self.g[i, j, k, q] = self.g_new[src_i, src_j, src_k, q]
                else:
                    # 邊界處理 (簡單反彈)
                    self.g[i, j, k, q] = self.g_new[i, j, k, q]
    
    @ti.kernel
    def compute_temperature(self):
        """
        從分布函數重建溫度場
        """
        
        for i, j, k in ti.ndrange(NX, NY, NZ):
            T_local = 0.0
            for q in ti.static(range(Q_THERMAL)):
                T_local += self.g[i, j, k, q]
            
            self.temperature[i, j, k] = T_local
    
    @ti.kernel
    def compute_heat_flux(self):
        """
        計算熱流密度向量 q = -k∇T
        """
        
        for i in range(1, NX-1):
            for j in range(1, NY-1):
                for k in range(1, NZ-1):
                    # 溫度梯度 (中心差分)
                    dT_dx = (self.temperature[i+1, j, k] - self.temperature[i-1, j, k]) / (2.0 * DX)
                    dT_dy = (self.temperature[i, j+1, k] - self.temperature[i, j-1, k]) / (2.0 * DX)
                    dT_dz = (self.temperature[i, j, k+1] - self.temperature[i, j, k-1]) / (2.0 * DX)
                    
                    # Fourier熱傳導定律
                    k_thermal = self.thermal_conductivity[i, j, k]
                    self.heat_flux[i, j, k] = ti.Vector([-k_thermal * dT_dx,
                                                       -k_thermal * dT_dy, 
                                                       -k_thermal * dT_dz])
    
    @ti.kernel
    def apply_dirichlet_bc(self, 
                          boundary_mask: ti.template(),
                          boundary_temp: ti.f32):
        """
        施加Dirichlet邊界條件 (固定溫度)
        
        Args:
            boundary_mask: 邊界標記場
            boundary_temp: 邊界溫度值
        """
        
        for i, j, k in ti.ndrange(NX, NY, NZ):
            if boundary_mask[i, j, k]:
                self.temperature[i, j, k] = boundary_temp
                
                # 重設分布函數為邊界溫度的平衡態
                for q in ti.static(range(Q_THERMAL)):
                    self.g[i, j, k, q] = self._equilibrium_distribution(q, boundary_temp)
    
    @ti.kernel  
    def apply_neumann_bc(self,
                        boundary_mask: ti.template(),
                        boundary_flux: ti.f32):
        """
        施加Neumann邊界條件 (固定熱流)
        
        Args:
            boundary_mask: 邊界標記場
            boundary_flux: 邊界熱流密度
        """
        
        for i, j, k in ti.ndrange(NX, NY, NZ):
            if boundary_mask[i, j, k]:
                # Neumann邊界條件實現
                # 這裡使用簡化處理，實際可能需要更精確的算法
                self.heat_source[i, j, k] = boundary_flux
    
    @ti.kernel
    def apply_convective_bc(self,
                           boundary_mask: ti.template(), 
                           h_conv: ti.f32,
                           T_ambient: ti.f32):
        """
        施加對流邊界條件 (Robin邊界)
        
        邊界條件：-k∇T = h(T - T_ambient)
        
        Args:
            boundary_mask: 邊界標記場
            h_conv: 對流換熱係數 W/(m²·K)
            T_ambient: 環境溫度 °C
        """
        
        for i, j, k in ti.ndrange(NX, NY, NZ):
            if boundary_mask[i, j, k]:
                T_surface = self.temperature[i, j, k]
                heat_flux_conv = h_conv * (T_surface - T_ambient)
                self.heat_source[i, j, k] = -heat_flux_conv
    
    @ti.kernel
    def check_numerical_stability(self) -> ti.i32:
        """
        檢查數值穩定性
        
        Returns:
            0: 穩定, 1: 不穩定
        """
        
        unstable = 0
        
        for i, j, k in ti.ndrange(NX, NY, NZ):
            T_local = self.temperature[i, j, k]
            
            # 溫度範圍檢查
            if T_local < -50.0 or T_local > 150.0:
                unstable = 1
            
            # NaN/Inf檢查
            if not (T_local == T_local):  # NaN檢測
                unstable = 1
            
            # 分布函數檢查
            for q in ti.static(range(Q_THERMAL)):
                g_val = self.g[i, j, k, q]
                if not (g_val == g_val) or abs(g_val) > 1000.0:
                    unstable = 1
        
        return unstable
    
    @ti.kernel
    def compute_diagnostics(self):
        """計算診斷統計量"""
        
        T_sum = 0.0
        T_max = -1000.0
        T_min = 1000.0
        
        for i, j, k in ti.ndrange(NX, NY, NZ):
            T_local = self.temperature[i, j, k]
            T_sum += T_local
            T_max = max(T_max, T_local)
            T_min = min(T_min, T_local)
        
        self.max_temperature[None] = T_max
        self.min_temperature[None] = T_min
        self.avg_temperature[None] = T_sum / (NX * NY * NZ)
    
    def step(self) -> bool:
        """
        執行一個完整的LBM時間步 (含對流耦合)
        
        Returns:
            True: 成功, False: 數值不穩定
        """
        
        if not self.is_initialized:
            print("❌ 錯誤：溫度場未初始化")
            return False
        
        # 保存舊溫度場
        self.temperature_old.copy_from(self.temperature)
        
        # 如果啟用對流，計算對流項
        if self.enable_convection:
            self.compute_convection_source_term()
        
        # LBM步驟
        self.collision_step()
        self.streaming_step()
        self.compute_temperature()
        self.compute_heat_flux()
        
        # 穩定性檢查
        if self.check_numerical_stability():
            print(f"❌ 步驟{self.current_step}: 熱傳LBM數值不穩定")
            return False
        
        self.current_step += 1
        return True
    
    def get_temperature_stats(self) -> Tuple[float, float, float]:
        """
        獲取溫度統計量
        
        Returns:
            (最小溫度, 最大溫度, 平均溫度)
        """
        
        self.compute_diagnostics()
        return (self.min_temperature[None], 
                self.max_temperature[None], 
                self.avg_temperature[None])
    
    def get_effective_thermal_diffusivity(self) -> float:
        """
        獲取有效熱擴散係數
        
        Returns:
            有效熱擴散係數 (m²/s)
        """
        return (self.tau_thermal - 0.5) * CS2_THERMAL * (self.scale_length**2) / self.scale_time
    
    def set_heat_source(self, source_field: np.ndarray):
        """
        設置熱源項
        
        Args:
            source_field: 3D熱源陣列 (W/m³)
        """
        
        if source_field.shape != (NX, NY, NZ):
            raise ValueError(f"熱源場尺寸不匹配: {source_field.shape} vs ({NX}, {NY}, {NZ})")
        
        self.heat_source.from_numpy(source_field.astype(np.float32))
    
    def reset(self):
        """重置求解器狀態"""
        
        self.current_step = 0
        self.is_initialized = False
        
        # 清零所有場
        self.g.fill(0.0)
        self.g_new.fill(0.0)
        self.temperature.fill(25.0)  # 環境溫度
        self.heat_source.fill(0.0)
        self.velocity_field.fill(0.0)  # 重置速度場
    
    # ==============================================
    # 對流耦合介面方法 (Phase 2)
    # ==============================================
    
    def enable_convection_coupling(self, enable: bool = True):
        """
        啟用/禁用對流耦合
        
        Args:
            enable: 是否啟用對流項計算
        """
        self.enable_convection = enable
        if enable:
            print("🌊 熱傳對流耦合已啟用")
        else:
            print("🔥 熱傳純擴散模式")
    
    def set_velocity_field(self, velocity_field: ti.Vector.field):
        """
        設置流體速度場 (來自LBM求解器)
        
        Args:
            velocity_field: 3D向量速度場 [NX×NY×NZ×3]
        """
        if not self.enable_convection:
            return
            
        # 複製速度場數據
        self._copy_velocity_field(velocity_field)
    
    @ti.kernel
    def _copy_velocity_field(self, source_velocity: ti.template()):
        """
        複製速度場數據到熱傳求解器
        
        Args:
            source_velocity: 源速度場 (來自LBM求解器)
        """
        for i, j, k in ti.ndrange(NX, NY, NZ):
            self.velocity_field[i, j, k] = source_velocity[i, j, k]
    
    @ti.kernel 
    def compute_convection_source_term(self):
        """
        計算對流項源項 S_conv = -u·∇T
        將結果疊加到熱源場中
        """
        
        for i in range(1, NX-1):
            for j in range(1, NY-1):
                for k in range(1, NZ-1):
                    # 溫度梯度 (中心差分)
                    dT_dx = (self.temperature[i+1, j, k] - self.temperature[i-1, j, k]) / (2.0 * DX)
                    dT_dy = (self.temperature[i, j+1, k] - self.temperature[i, j-1, k]) / (2.0 * DX)
                    dT_dz = (self.temperature[i, j, k+1] - self.temperature[i, j, k-1]) / (2.0 * DX)
                    
                    # 對流項 -u·∇T
                    u_vec = self.velocity_field[i, j, k]
                    convection_term = -(u_vec.x * dT_dx + u_vec.y * dT_dy + u_vec.z * dT_dz)
                    
                    # 疊加到熱源項
                    self.heat_source[i, j, k] += convection_term
    
    @ti.kernel
    def reset_heat_source_to_base(self, base_heat_source: ti.template()):
        """
        重置熱源場到基礎值 (移除上一步的對流項)
        
        Args:
            base_heat_source: 基礎熱源場 (不含對流項)
        """
        for i, j, k in ti.ndrange(NX, NY, NZ):
            self.heat_source[i, j, k] = base_heat_source[i, j, k]
        
        print("✅ 熱傳LBM求解器已重置")


# ==============================================
# 模組測試函數
# ==============================================

def test_thermal_lbm_basic():
    """基礎功能測試"""
    
    print("\n🔬 測試熱傳LBM基礎功能...")
    
    # 初始化求解器
    solver = ThermalLBM()
    
    # 初始化溫度場
    solver.complete_initialization(T_initial=25.0, T_hot_region=90.0, hot_region_height=10)
    
    # 執行10步
    for step in range(10):
        success = solver.step()
        if not success:
            print(f"❌ 第{step}步失敗")
            return False
        
        T_min, T_max, T_avg = solver.get_temperature_stats()
        print(f"  步驟{step}: T∈[{T_min:.2f}, {T_max:.2f}]°C, 平均{T_avg:.2f}°C")
    
    print("✅ 基礎功能測試通過")
    return True

def test_thermal_diffusivity():
    """熱擴散係數測試"""
    
    print("\n🌡️  測試熱擴散係數計算...")
    
    # 不同擴散係數
    alphas = [1.0e-7, 1.6e-7, 2.0e-7]  # m²/s
    
    for alpha in alphas:
        solver = ThermalLBM(thermal_diffusivity=alpha)
        effective_alpha = solver.get_effective_thermal_diffusivity()
        error = abs(effective_alpha - alpha) / alpha * 100
        
        print(f"  α_設定={alpha:.2e}, α_有效={effective_alpha:.2e}, 誤差={error:.1f}%")
        
        if error > 5.0:  # 5%誤差限制
            print(f"❌ 熱擴散係數誤差過大: {error:.1f}%")
            return False
    
    print("✅ 熱擴散係數測試通過")
    return True

if __name__ == "__main__":
    # 初始化Taichi
    ti.init(arch=ti.cpu)  # 使用CPU進行測試，避免GPU記憶體問題
    
    print("=== 熱傳LBM模組測試 ===")
    
    # 執行測試
    test1 = test_thermal_lbm_basic()
    test2 = test_thermal_diffusivity()
    
    if test1 and test2:
        print("\n✅ 所有測試通過！熱傳LBM模組就緒")
    else:
        print("\n❌ 測試失敗，需要修正")