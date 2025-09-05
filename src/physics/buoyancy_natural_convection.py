# buoyancy_natural_convection.py - 浮力驅動自然對流機制
"""
Phase 3: 浮力驅動自然對流實現

物理原理:
- Boussinesq近似: 密度僅在浮力項中變化
- 浮力項: F_b = ρ₀gβ(T - T₀)
- 自然對流: 溫度梯度 → 密度梯度 → 浮力 → 流動 → 對流傳熱

LBM實現:
- Guo forcing scheme浮力項添加
- 格子單位尺度轉換
- 數值穩定性控制
- GPU並行優化

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import math
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass

import config
from src.physics.temperature_dependent_properties import TemperatureDependentProperties

@dataclass  
class BuoyancyParameters:
    """浮力參數配置"""
    
    # 重力場
    gravity_magnitude: float = 9.81    # 重力加速度 (m/s²)
    gravity_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)  # 重力方向 (單位向量)
    
    # Boussinesq近似參數
    reference_temperature: float = 25.0  # 參考溫度 (°C)
    reference_density: float = 997.0     # 參考密度 (kg/m³)
    thermal_expansion: float = 2.1e-4    # 體積膨脹係數 (1/K)
    
    # 格子單位轉換
    buoyancy_scaling: float = 1.0        # 浮力強度調節係數
    max_buoyancy_force: float = 0.1      # 最大浮力 (格子單位)
    
    # Rayleigh數控制
    target_rayleigh: float = 1e4         # 目標Rayleigh數
    adaptive_scaling: bool = True        # 自適應浮力強度

@ti.data_oriented
class BuoyancyNaturalConvection:
    """
    浮力驅動自然對流系統
    
    實現基於Boussinesq近似的浮力驅動自然對流
    適用於手沖咖啡的溫度驅動流動模擬
    
    Features:
    - Boussinesq浮力項計算
    - Guo forcing scheme集成
    - 自適應浮力強度控制
    - Rayleigh數自動調節
    - GPU並行優化
    """
    
    def __init__(self, 
                 buoyancy_params: BuoyancyParameters = None,
                 properties_calculator: TemperatureDependentProperties = None):
        """
        初始化浮力自然對流系統
        
        Args:
            buoyancy_params: 浮力參數配置
            properties_calculator: 溫度依賴物性計算器
        """
        
        self.params = buoyancy_params or BuoyancyParameters()
        self.properties = properties_calculator
        
        # 初始化浮力場
        self._init_buoyancy_fields()
        
        # 計算格子單位參數
        self._compute_lattice_parameters()
        
        # 初始化診斷量
        self.rayleigh_number = 0.0
        self.nusselt_number = 0.0
        self.max_velocity_magnitude = 0.0
        
        # 統計信息
        self.total_buoyancy_force = 0.0
        self.update_count = 0
        
        print("✅ 浮力自然對流系統初始化完成")
        print(f"   重力方向: {self.params.gravity_direction}")
        print(f"   參考溫度: {self.params.reference_temperature:.1f}°C")
        print(f"   體積膨脹係數: {self.params.thermal_expansion:.2e} 1/K")
    
    def _init_buoyancy_fields(self):
        """初始化浮力相關場變數"""
        
        # 浮力場 F_b = ρ₀gβ(T - T₀)
        self.buoyancy_force = ti.Vector.field(3, ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 浮力強度場 |F_b|
        self.buoyancy_magnitude = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 溫度差場 ΔT = T - T₀
        self.temperature_difference = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 重力向量 (格子單位)
        self.gravity_lattice = ti.Vector([
            self.params.gravity_direction[0],
            self.params.gravity_direction[1], 
            self.params.gravity_direction[2]
        ])
        
        # 初始化為零
        self.buoyancy_force.fill(0.0)
        self.buoyancy_magnitude.fill(0.0)
        self.temperature_difference.fill(0.0)
    
    def _compute_lattice_parameters(self):
        """計算格子單位浮力參數"""
        
        # 物理單位浮力: F_phys = ρ₀gβΔT (N/m³)
        # 格子單位浮力: F_lu = F_phys * scale_factor
        
        # 修正：重力格子單位轉換  
        # g_lu = g_phys × SCALE_TIME² / SCALE_LENGTH
        # 其中 SCALE_TIME = 75ms, SCALE_LENGTH = 0.625mm
        self.gravity_lattice_magnitude = (self.params.gravity_magnitude * 
                                         config.SCALE_TIME * config.SCALE_TIME / config.SCALE_LENGTH)
        
        # 浮力前置係數 (格子單位) 
        # F_b = ρ₀ * g_lu * β * ΔT * scaling
        # 量綱檢查: [kg/m³] * [LT⁻²] * [K⁻¹] * [K] = [kg/(m²T²)] = [N/m³] ✓
        self.buoyancy_coefficient = (self.params.reference_density * 
                                   self.gravity_lattice_magnitude * 
                                   self.params.thermal_expansion *
                                   self.params.buoyancy_scaling)
        
        # 為了數值穩定性，進一步縮放到合理範圍
        # 典型LBM中體力應該 << 1.0 格子單位
        lattice_density = 1.0  # LBM參考密度
        scaling_factor = lattice_density / self.params.reference_density
        self.buoyancy_coefficient *= scaling_factor
        
        print(f"🔧 修正後格子單位浮力參數:")
        print(f"   重力格子單位: {self.gravity_lattice_magnitude:.6f}")
        print(f"   原始浮力係數: {self.buoyancy_coefficient/scaling_factor:.6f}")
        print(f"   縮放後浮力係數: {self.buoyancy_coefficient:.6f}")
        print(f"   密度縮放因子: {scaling_factor:.6f}")
    
    @ti.kernel
    def compute_buoyancy_force(self, temperature_field: ti.template()):
        """
        計算浮力場
        
        F_b(x) = ρ₀gβ(T(x) - T₀) * ĝ
        
        Args:
            temperature_field: 溫度場 [NX×NY×NZ]
        """
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 溫度差
            delta_T = temperature_field[i, j, k] - self.params.reference_temperature
            self.temperature_difference[i, j, k] = delta_T
            
            # 浮力強度
            buoyancy_strength = self.buoyancy_coefficient * delta_T
            
            # 限制最大浮力 (數值穩定性)
            buoyancy_strength = ti.max(-self.params.max_buoyancy_force,
                                     ti.min(buoyancy_strength, self.params.max_buoyancy_force))
            
            # 浮力向量
            buoyancy_vec = buoyancy_strength * self.gravity_lattice
            
            # 更新場
            self.buoyancy_force[i, j, k] = buoyancy_vec
            self.buoyancy_magnitude[i, j, k] = buoyancy_vec.norm()
    
    @ti.kernel
    def apply_buoyancy_to_distribution(self, 
                                     f_field: ti.template(),
                                     f_new_field: ti.template(),
                                     density_field: ti.template(),
                                     velocity_field: ti.template(),
                                     cx: ti.template(),
                                     cy: ti.template(), 
                                     cz: ti.template(),
                                     w: ti.template()):
        """
        將浮力項應用到分布函數 (Guo forcing scheme)
        
        f_i^{n+1} = f_i^* + Δt * S_i
        S_i = w_i * (1 - 1/(2τ)) * [(e_i - u)/c_s² + (e_i·u)(e_i)/c_s⁴] · F
        
        Args:
            f_field: 當前分布函數 [q, i, j, k] (SoA格式)
            f_new_field: 更新後分布函數 [q, i, j, k] (SoA格式)
            density_field: 密度場
            velocity_field: 速度場
            cx, cy, cz: LBM離散速度常數 
            w: LBM權重常數
        """
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 局部量
            rho = density_field[i, j, k]
            u = velocity_field[i, j, k]
            F_b = self.buoyancy_force[i, j, k]
            
            # 避免除零和小密度區域
            if rho < 1e-10:
                continue
            
            # Guo forcing項 (修正版)
            for q in ti.static(range(config.Q_3D)):  # 使用config.Q_3D確保一致性
                # 離散速度 e_q (從傳入參數獲取)
                e_q = ti.Vector([cx[q], cy[q], cz[q]], ti.f32)
                
                # 權重 (從傳入參數獲取)
                w_q = w[q]
                
                # Guo forcing係數 (使用正確鬆弛時間)
                tau = config.TAU_WATER  # 水相鬆弛時間
                guo_coeff = w_q * (1.0 - 1.0/(2.0 * tau))
                
                # 速度相關項
                e_dot_u = e_q.dot(u)
                e_dot_F = e_q.dot(F_b)
                
                # 修正的Guo forcing項計算
                # S_q = w_q * (1 - 1/2τ) * [e_q·F/cs² + (e_q·u)(e_q·F)/cs⁴]
                cs2_inv = config.INV_CS2  # 1/cs² = 3.0
                term1 = e_dot_F * cs2_inv
                term2 = e_dot_u * e_dot_F * cs2_inv * cs2_inv
                
                S_q = guo_coeff * (term1 + term2)
                
                # 數值穩定性限制
                S_q = ti.max(-0.01, ti.min(0.01, S_q))
                
                # 應用forcing到分布函數 (SoA格式)
                f_new_field[q, i, j, k] += config.DT * S_q
    
    @ti.kernel
    def compute_rayleigh_number(self, 
                              temperature_field: ti.template(),
                              velocity_field: ti.template()) -> ti.f32:
        """
        計算局部Rayleigh數
        
        Ra = gβΔTL³/(νᾱ)
        
        Args:
            temperature_field: 溫度場
            velocity_field: 速度場
            
        Returns:
            平均Rayleigh數
        """
        
        total_ra = 0.0
        count = 0.0
        
        for i in range(1, config.NX-1):
            for j in range(1, config.NY-1):
                for k in range(1, config.NZ-1):
                    # 局部溫度梯度
                    dT_dx = (temperature_field[i+1, j, k] - temperature_field[i-1, j, k]) / (2.0 * config.DX)
                    dT_dy = (temperature_field[i, j+1, k] - temperature_field[i, j-1, k]) / (2.0 * config.DX)
                    dT_dz = (temperature_field[i, j, k+1] - temperature_field[i, j, k-1]) / (2.0 * config.DX)
                    
                    # 溫度梯度量級
                    grad_T_mag = ti.sqrt(dT_dx*dT_dx + dT_dy*dT_dy + dT_dz*dT_dz)
                    
                    if grad_T_mag > 1e-6:  # 避免除零
                        # 特徵長度 (假設為格子間距)
                        L_char = config.DX
                        
                        # 局部Rayleigh數估算
                        ra_local = (self.params.gravity_magnitude * 
                                   self.params.thermal_expansion * 
                                   grad_T_mag * L_char * L_char * L_char) / (1.0e-6 * 1.5e-7)
                        
                        total_ra += ra_local
                        count += 1.0
        
        # 修正：避免在kernel中使用條件return
        result = 0.0
        if count > 0.5:  # 使用數值比較代替 count > 0
            result = total_ra / count
        
        return result
    
    def update_buoyancy_system(self, 
                             temperature_field: ti.field,
                             density_field: ti.field,
                             velocity_field: ti.field) -> Dict[str, float]:
        """
        更新浮力系統
        
        Args:
            temperature_field: 溫度場
            density_field: 密度場  
            velocity_field: 速度場
            
        Returns:
            系統診斷信息
        """
        
        # 計算浮力場
        self.compute_buoyancy_force(temperature_field)
        
        # 計算Rayleigh數
        self.rayleigh_number = self.compute_rayleigh_number(temperature_field, velocity_field)
        
        # 統計信息
        buoyancy_np = self.buoyancy_magnitude.to_numpy()
        self.total_buoyancy_force = float(np.sum(buoyancy_np))
        self.max_velocity_magnitude = float(np.max(velocity_field.to_numpy()))
        
        # 自適應浮力強度調節
        if self.params.adaptive_scaling and self.rayleigh_number > 0:
            target_ratio = self.params.target_rayleigh / self.rayleigh_number
            if 0.1 < target_ratio < 10.0:  # 合理調節範圍
                self.params.buoyancy_scaling *= min(1.1, max(0.9, target_ratio ** 0.1))
                self._compute_lattice_parameters()  # 重新計算係數
        
        self.update_count += 1
        
        # 返回診斷信息
        return {
            'rayleigh_number': self.rayleigh_number,
            'total_buoyancy_force': self.total_buoyancy_force,
            'max_buoyancy_magnitude': float(np.max(buoyancy_np)),
            'mean_temperature_difference': float(np.mean(self.temperature_difference.to_numpy())),
            'buoyancy_scaling': self.params.buoyancy_scaling,
            'max_velocity_magnitude': self.max_velocity_magnitude
        }
    
    def get_natural_convection_diagnostics(self) -> Dict[str, Any]:
        """
        獲取自然對流診斷信息
        
        Returns:
            詳細診斷字典
        """
        
        # 浮力統計
        buoyancy_stats = {}
        buoyancy_np = self.buoyancy_magnitude.to_numpy()
        buoyancy_stats['magnitude'] = {
            'min': float(np.min(buoyancy_np)),
            'max': float(np.max(buoyancy_np)),
            'mean': float(np.mean(buoyancy_np)),
            'std': float(np.std(buoyancy_np))
        }
        
        # 溫度差統計
        temp_diff_np = self.temperature_difference.to_numpy()
        temp_diff_stats = {
            'min': float(np.min(temp_diff_np)),
            'max': float(np.max(temp_diff_np)),
            'mean': float(np.mean(temp_diff_np)),
            'std': float(np.std(temp_diff_np))
        }
        
        return {
            'rayleigh_number': self.rayleigh_number,
            'nusselt_number': self.nusselt_number,
            'buoyancy_statistics': buoyancy_stats,
            'temperature_difference_statistics': temp_diff_stats,
            'total_buoyancy_force': self.total_buoyancy_force,
            'max_velocity_magnitude': self.max_velocity_magnitude,
            'update_count': self.update_count,
            'parameters': {
                'gravity_magnitude': self.params.gravity_magnitude,
                'thermal_expansion': self.params.thermal_expansion,
                'buoyancy_scaling': self.params.buoyancy_scaling,
                'reference_temperature': self.params.reference_temperature
            }
        }
    
    def reset_buoyancy_system(self):
        """重置浮力系統"""
        
        self.buoyancy_force.fill(0.0)
        self.buoyancy_magnitude.fill(0.0)
        self.temperature_difference.fill(0.0)
        
        self.rayleigh_number = 0.0
        self.nusselt_number = 0.0
        self.total_buoyancy_force = 0.0
        self.update_count = 0
        
        print("🔄 浮力自然對流系統已重置")

# 工廠函數
def create_coffee_buoyancy_system(properties_calculator: TemperatureDependentProperties = None) -> BuoyancyNaturalConvection:
    """創建適用於手沖咖啡的浮力系統"""
    
    coffee_params = BuoyancyParameters(
        gravity_magnitude=9.81,
        gravity_direction=(0.0, 0.0, -1.0),  # Z軸向下
        reference_temperature=25.0,
        reference_density=997.0,
        thermal_expansion=2.1e-4,
        buoyancy_scaling=0.1,  # 保守的初始值
        max_buoyancy_force=0.05,
        target_rayleigh=5e3,   # 適中的Rayleigh數
        adaptive_scaling=True
    )
    
    return BuoyancyNaturalConvection(coffee_params, properties_calculator)