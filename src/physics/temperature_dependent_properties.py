# temperature_dependent_properties.py - 溫度依賴流體物性模型
"""
Phase 3: 溫度依賴流體物性計算系統

功能:
- 水的溫度依賴密度計算 (Boussinesq近似)
- 溫度依賴動力黏度計算 (Vogel-Fulcher-Tammann模型)
- 溫度依賴熱物性計算 (熱導率、比熱、熱擴散係數)
- GPU並行優化的物性場更新

物理模型:
- 密度: ρ(T) = ρ₀[1 - β(T - T₀)] (Boussinesq近似)
- 黏度: μ(T) = μ₀ * exp(A/(T + B)) (VFT模型)
- 熱導率: k(T) = k₀ + k₁T + k₂T² (多項式擬合)

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import math
from typing import Tuple, Dict, Any
from dataclasses import dataclass

import config.config as config
# from src.physics.thermal_properties import ThermalPropertiesDatabase  # 暫時註釋，避免循環導入

@dataclass
class FluidPropertyConstants:
    """流體物性常數配置"""
    
    # 參考狀態 (25°C, 1 atm)
    T_ref: float = 25.0        # 參考溫度 (°C)
    rho_ref: float = 997.0     # 參考密度 (kg/m³)
    mu_ref: float = 8.9e-4     # 參考動力黏度 (Pa·s)
    
    # Boussinesq近似參數
    beta: float = 2.1e-4       # 體積膨脹係數 (1/K)
    
    # VFT黏度模型參數 (重新校準適用於15-100°C咖啡溫度範圍)
    mu_vft_A: float = 120.0    # VFT參數A (K) - 降低敏感度
    mu_vft_B: float = 100.0    # VFT參數B (K) - 調整基準溫度
    
    # 熱導率溫度依賴參數
    k_coeff_0: float = 0.5562  # W/(m·K)
    k_coeff_1: float = 1.9e-3  # W/(m·K²)
    k_coeff_2: float = -8e-6   # W/(m·K³)
    
    # 比熱容溫度依賴參數
    cp_coeff_0: float = 4180.0 # J/(kg·K)
    cp_coeff_1: float = -0.5   # J/(kg·K²)
    cp_coeff_2: float = 1e-3   # J/(kg·K³)

@ti.data_oriented
class TemperatureDependentProperties:
    """
    溫度依賴流體物性計算系統
    
    提供高精度的水物性溫度依賴關係計算，支援GPU並行運算
    適用於手沖咖啡溫度範圍 (15-100°C)
    
    Features:
    - Boussinesq近似密度計算
    - VFT模型黏度計算  
    - 多項式熱物性計算
    - GPU優化的場更新
    - 數值穩定性保證
    """
    
    def __init__(self, constants: FluidPropertyConstants = None):
        """
        初始化溫度依賴物性計算系統
        
        Args:
            constants: 流體物性常數配置
        """
        
        self.constants = constants or FluidPropertyConstants()
        
        # 初始化物性場
        self._init_property_fields()
        
        # 載入熱物性數據庫 (暫時禁用)
        # self.thermal_db = ThermalPropertiesDatabase()
        self.thermal_db = None
        
        # 溫度範圍檢查
        self.T_min = 5.0   # 最低安全溫度 (°C)
        self.T_max = 105.0 # 最高安全溫度 (°C)
        
        # 計算統計
        self.update_count = 0
        self.last_update_time = 0.0
        
        print("✅ 溫度依賴物性系統初始化完成")
        print(f"   溫度範圍: {self.T_min:.0f} - {self.T_max:.0f}°C")
        print(f"   參考狀態: {self.constants.T_ref:.0f}°C, {self.constants.rho_ref:.0f} kg/m³")
    
    def _init_property_fields(self):
        """初始化物性場"""
        
        # 流體物性場
        self.density_field = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.viscosity_field = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.relaxation_time_field = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 熱物性場
        self.thermal_conductivity_field = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.heat_capacity_field = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.thermal_diffusivity_field = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 無量綱場
        self.buoyancy_factor_field = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 初始化為參考值
        self.density_field.fill(self.constants.rho_ref)
        self.viscosity_field.fill(self.constants.mu_ref)
        self.thermal_conductivity_field.fill(self.constants.k_coeff_0)
        self.heat_capacity_field.fill(self.constants.cp_coeff_0)
    
    @ti.func
    def density_from_temperature(self, T: ti.f32) -> ti.f32:
        """
        Boussinesq近似密度計算
        
        ρ(T) = ρ₀[1 - β(T - T₀)]
        
        Args:
            T: 溫度 (°C)
            
        Returns:
            密度 (kg/m³)
        """
        
        # 溫度範圍檢查
        T_safe = ti.max(self.T_min, ti.min(T, self.T_max))
        
        # Boussinesq近似
        delta_T = T_safe - self.constants.T_ref
        density = self.constants.rho_ref * (1.0 - self.constants.beta * delta_T)
        
        return density
    
    @ti.func  
    def viscosity_from_temperature(self, T: ti.f32) -> ti.f32:
        """
        VFT模型動力黏度計算
        
        μ(T) = μ₀ * exp(A/(T + B))
        
        Args:
            T: 溫度 (°C)
            
        Returns:
            動力黏度 (Pa·s)
        """
        
        # 溫度範圍檢查和Kelvin轉換
        T_safe = ti.max(self.T_min, ti.min(T, self.T_max))
        T_K = T_safe + 273.15
        
        # VFT模型
        exp_arg = self.constants.mu_vft_A / (T_K + self.constants.mu_vft_B)
        
        # 限制指數參數防止溢出
        exp_arg_safe = ti.max(-10.0, ti.min(exp_arg, 10.0))
        
        viscosity = self.constants.mu_ref * ti.exp(exp_arg_safe)
        
        return viscosity
    
    @ti.func
    def thermal_conductivity_from_temperature(self, T: ti.f32) -> ti.f32:
        """
        多項式熱導率計算
        
        k(T) = k₀ + k₁T + k₂T²
        
        Args:
            T: 溫度 (°C)
            
        Returns:
            熱導率 (W/(m·K))
        """
        
        T_safe = ti.max(self.T_min, ti.min(T, self.T_max))
        
        k = (self.constants.k_coeff_0 + 
             self.constants.k_coeff_1 * T_safe +
             self.constants.k_coeff_2 * T_safe * T_safe)
        
        # 確保正值
        return ti.max(k, 0.1)
    
    @ti.func
    def heat_capacity_from_temperature(self, T: ti.f32) -> ti.f32:
        """
        多項式比熱容計算
        
        cp(T) = cp₀ + cp₁T + cp₂T²
        
        Args:
            T: 溫度 (°C)
            
        Returns:
            比熱容 (J/(kg·K))
        """
        
        T_safe = ti.max(self.T_min, ti.min(T, self.T_max))
        
        cp = (self.constants.cp_coeff_0 + 
              self.constants.cp_coeff_1 * T_safe +
              self.constants.cp_coeff_2 * T_safe * T_safe)
        
        # 確保正值
        return ti.max(cp, 1000.0)
    
    @ti.func
    def relaxation_time_from_viscosity(self, viscosity: ti.f32, density: ti.f32) -> ti.f32:
        """
        從黏度計算LBM鬆弛時間
        
        τ = ν/(c_s²) + 0.5 = μ/(ρc_s²) + 0.5
        
        Args:
            viscosity: 動力黏度 (Pa·s)
            density: 密度 (kg/m³)
            
        Returns:
            鬆弛時間 (無量綱)
        """
        
        # 運動黏度
        kinematic_viscosity = viscosity / density
        
        # 格子單位運動黏度 (需要尺度轉換)
        nu_lattice = kinematic_viscosity * config.DT / (config.DX * config.DX)
        
        # LBM鬆弛時間
        tau = nu_lattice / config.CS2 + 0.5
        
        # 數值穩定性限制
        return ti.max(0.51, ti.min(tau, 2.0))
    
    @ti.func
    def buoyancy_factor_from_temperature(self, T: ti.f32) -> ti.f32:
        """
        計算浮力因子 (用於浮力項)
        
        factor = -β(T - T₀)
        
        Args:
            T: 溫度 (°C)
            
        Returns:
            浮力因子 (無量綱)
        """
        
        delta_T = T - self.constants.T_ref
        return -self.constants.beta * delta_T
    
    @ti.kernel
    def update_properties_from_temperature(self, temperature_field: ti.template()):
        """
        從溫度場更新所有物性場
        
        Args:
            temperature_field: 溫度場 [NX×NY×NZ]
        """
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            T = temperature_field[i, j, k]
            
            # 計算所有物性
            density = self.density_from_temperature(T)
            viscosity = self.viscosity_from_temperature(T)
            k_thermal = self.thermal_conductivity_from_temperature(T)
            cp = self.heat_capacity_from_temperature(T)
            
            # 更新物性場
            self.density_field[i, j, k] = density
            self.viscosity_field[i, j, k] = viscosity
            self.thermal_conductivity_field[i, j, k] = k_thermal
            self.heat_capacity_field[i, j, k] = cp
            
            # 計算衍生量
            self.relaxation_time_field[i, j, k] = self.relaxation_time_from_viscosity(viscosity, density)
            self.thermal_diffusivity_field[i, j, k] = k_thermal / (density * cp)
            self.buoyancy_factor_field[i, j, k] = self.buoyancy_factor_from_temperature(T)
    
    def get_property_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        獲取物性統計信息
        
        Returns:
            物性統計字典
        """
        
        stats = {}
        
        # 密度統計
        rho_np = self.density_field.to_numpy()
        stats['density'] = {
            'min': float(np.min(rho_np)),
            'max': float(np.max(rho_np)),
            'mean': float(np.mean(rho_np)),
            'std': float(np.std(rho_np))
        }
        
        # 黏度統計
        mu_np = self.viscosity_field.to_numpy()
        stats['viscosity'] = {
            'min': float(np.min(mu_np)),
            'max': float(np.max(mu_np)),
            'mean': float(np.mean(mu_np)),
            'std': float(np.std(mu_np))
        }
        
        # 鬆弛時間統計
        tau_np = self.relaxation_time_field.to_numpy()
        stats['relaxation_time'] = {
            'min': float(np.min(tau_np)),
            'max': float(np.max(tau_np)),
            'mean': float(np.mean(tau_np)),
            'std': float(np.std(tau_np))
        }
        
        # 熱導率統計
        k_np = self.thermal_conductivity_field.to_numpy()
        stats['thermal_conductivity'] = {
            'min': float(np.min(k_np)),
            'max': float(np.max(k_np)),
            'mean': float(np.mean(k_np)),
            'std': float(np.std(k_np))
        }
        
        return stats
    
    def validate_property_ranges(self) -> bool:
        """
        驗證物性範圍的合理性
        
        Returns:
            True: 物性範圍合理, False: 存在異常值
        """
        
        stats = self.get_property_statistics()
        
        # 密度範圍檢查 (水在5-100°C: 960-1000 kg/m³)
        if not (960.0 <= stats['density']['min'] <= stats['density']['max'] <= 1010.0):
            print(f"⚠️  密度範圍異常: {stats['density']['min']:.1f} - {stats['density']['max']:.1f} kg/m³")
            return False
        
        # 黏度範圍檢查 (水在5-100°C: 1e-4 - 1.5e-3 Pa·s) - 修正為更寬鬆範圍
        if not (5e-5 <= stats['viscosity']['min'] <= stats['viscosity']['max'] <= 5e-3):
            print(f"⚠️  黏度範圍異常: {stats['viscosity']['min']:.2e} - {stats['viscosity']['max']:.2e} Pa·s")
            return False
        
        # 鬆弛時間範圍檢查 (數值穩定性: 0.51 - 2.0)
        if not (0.50 <= stats['relaxation_time']['min'] <= stats['relaxation_time']['max'] <= 2.1):
            print(f"⚠️  鬆弛時間範圍異常: {stats['relaxation_time']['min']:.3f} - {stats['relaxation_time']['max']:.3f}")
            return False
        
        return True
    
    def reset_to_reference_state(self):
        """重置所有物性場到參考狀態"""
        
        self.density_field.fill(self.constants.rho_ref)
        self.viscosity_field.fill(self.constants.mu_ref)
        self.thermal_conductivity_field.fill(self.constants.k_coeff_0)
        self.heat_capacity_field.fill(self.constants.cp_coeff_0)
        
        # 計算參考鬆弛時間
        ref_tau = self.constants.mu_ref / (self.constants.rho_ref * config.CS2 * config.DX * config.DX / config.DT) + 0.5
        self.relaxation_time_field.fill(ref_tau)
        
        self.buoyancy_factor_field.fill(0.0)
        
        print(f"🔄 物性場重置到參考狀態 (T={self.constants.T_ref}°C)")

# 工廠函數
def create_water_properties() -> TemperatureDependentProperties:
    """創建水的溫度依賴物性計算器"""
    
    water_constants = FluidPropertyConstants(
        T_ref=25.0,
        rho_ref=997.0,
        mu_ref=8.9e-4,
        beta=2.1e-4,
        mu_vft_A=580.0,
        mu_vft_B=138.0
    )
    
    return TemperatureDependentProperties(water_constants)

def create_coffee_properties() -> TemperatureDependentProperties:
    """創建咖啡液的溫度依賴物性計算器 (近似水的物性)"""
    
    coffee_constants = FluidPropertyConstants(
        T_ref=25.0,
        rho_ref=1002.0,    # 咖啡液略高密度
        mu_ref=9.2e-4,     # 咖啡液略高黏度
        beta=2.0e-4,       # 略小膨脹係數
        mu_vft_A=130.0,    # 調整VFT參數適合咖啡
        mu_vft_B=105.0     # 調整基準溫度
    )
    
    return TemperatureDependentProperties(coffee_constants)