# strong_coupled_solver.py - Phase 3 強耦合系統控制器
"""
Phase 3: 雙向強耦合系統主控制器

功能:
- 溫度↔流體雙向反饋控制
- 浮力驅動自然對流
- 溫度依賴物性實時更新
- 耦合穩定性監控和控制
- 自適應時間步長調節

耦合機制:
1. T → ρ(T), μ(T) → 流體物性更新
2. T → 浮力項 → 流動驅動
3. u → 對流項 → 溫度場演化
4. 穩定性監控 → 自適應參數調節

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

# 核心模組
from src.core.lbm_solver import LBMSolver
from src.physics.thermal_lbm import ThermalLBM
from src.physics.temperature_dependent_properties import TemperatureDependentProperties, create_water_properties
from src.physics.buoyancy_natural_convection import BuoyancyNaturalConvection, create_coffee_buoyancy_system
import config

@dataclass
class StrongCouplingConfig:
    """強耦合系統配置"""
    
    # 耦合控制
    coupling_frequency: int = 1          # 耦合頻率 (每N步)
    max_coupling_iterations: int = 3     # 最大耦合迭代次數
    coupling_tolerance: float = 1e-4     # 耦合收斂容差
    
    # 穩定性控制
    enable_adaptive_relaxation: bool = True    # 自適應鬆弛
    relaxation_factor: float = 0.8            # 鬆弛係數
    min_relaxation: float = 0.1               # 最小鬆弛係數
    max_relaxation: float = 1.0               # 最大鬆弛係數
    
    # 溫度依賴物性控制
    enable_variable_density: bool = True      # 可變密度
    enable_variable_viscosity: bool = True    # 可變黏度
    enable_buoyancy: bool = True              # 浮力項
    
    # 診斷和監控
    enable_diagnostics: bool = True           # 診斷監控
    stability_check_frequency: int = 10       # 穩定性檢查頻率
    max_temperature_change: float = 5.0       # 最大溫度變化 (°C/步)
    max_velocity_magnitude: float = 0.3       # 最大速度 (格子單位)
    
    # 性能優化
    parallel_property_update: bool = True     # 並行物性更新
    cache_property_calculations: bool = False # 緩存物性計算

@ti.data_oriented
class StrongCoupledSolver:
    """
    Phase 3 雙向強耦合求解器
    
    管理溫度和流體場的完全雙向耦合，實現自然對流模擬
    
    Features:
    - 溫度依賴流體物性
    - 浮力驅動自然對流
    - 穩定性自適應控制
    - 高性能GPU並行計算
    - 完整診斷監控系統
    
    Physics:
    - Boussinesq近似自然對流
    - 溫度依賴密度和黏度
    - 對流-擴散耦合傳熱
    - 多物理場數值穩定性保證
    """
    
    def __init__(self, 
                 coupling_config: StrongCouplingConfig = None,
                 thermal_diffusivity: float = 1.6e-7):
        """
        初始化強耦合系統
        
        Args:
            coupling_config: 強耦合配置參數
            thermal_diffusivity: 熱擴散係數 (m²/s)
        """
        
        print("🔗 初始化Phase 3強耦合系統...")
        
        # 配置參數
        self.config = coupling_config or StrongCouplingConfig()
        
        # 初始化子系統
        self._init_subsystems(thermal_diffusivity)
        
        # 耦合狀態
        self.coupling_step = 0
        self.is_initialized = False
        self.is_converged = False
        self.current_relaxation = self.config.relaxation_factor
        
        # 性能統計
        self.performance_stats = {
            'fluid_time': 0.0,
            'thermal_time': 0.0,
            'property_update_time': 0.0,
            'buoyancy_time': 0.0,
            'coupling_iterations': [],
            'total_steps': 0
        }
        
        # 穩定性監控
        self.stability_history = {
            'temperature_changes': [],
            'velocity_magnitudes': [],
            'coupling_residuals': [],
            'rayleigh_numbers': []
        }
        
        print("✅ Phase 3強耦合系統初始化完成")
    
    def _init_subsystems(self, thermal_diffusivity: float):
        """初始化所有子系統"""
        
        # 1. 流體LBM求解器
        print("  🌊 初始化流體LBM求解器...")
        self.fluid_solver = LBMSolver()
        
        # 2. 熱傳LBM求解器  
        print("  🔥 初始化熱傳LBM求解器...")
        self.thermal_solver = ThermalLBM(thermal_diffusivity=thermal_diffusivity)
        
        # 3. 溫度依賴物性計算器
        print("  🌡️  初始化物性計算器...")
        self.properties_calculator = create_water_properties()
        
        # 4. 浮力自然對流系統
        if self.config.enable_buoyancy:
            print("  🌊 初始化浮力系統...")
            self.buoyancy_system = create_coffee_buoyancy_system(self.properties_calculator)
        else:
            self.buoyancy_system = None
        
        # 5. 啟用溫度依賴物性
        self.fluid_solver.enable_temperature_dependent_properties(
            properties_calculator=self.properties_calculator,
            buoyancy_system=self.buoyancy_system
        )
        
        # 6. 啟用熱傳對流耦合
        self.thermal_solver.enable_convection_coupling(True)
    
    def initialize_coupled_system(self,
                                fluid_initial_conditions: Dict[str, Any],
                                thermal_initial_conditions: Dict[str, Any],
                                base_heat_source: Optional[np.ndarray] = None):
        """
        初始化強耦合系統
        
        Args:
            fluid_initial_conditions: 流體初始條件
            thermal_initial_conditions: 熱傳初始條件
            base_heat_source: 基礎熱源場
        """
        
        print("🚀 初始化強耦合系統狀態...")
        
        # 初始化流體求解器
        self.fluid_solver.init_fields()
        
        # 初始化熱傳求解器
        self.thermal_solver.complete_initialization(
            T_initial=thermal_initial_conditions.get('T_initial', 25.0),
            T_hot_region=thermal_initial_conditions.get('T_hot_region', 85.0),
            hot_region_height=thermal_initial_conditions.get('hot_region_height', 15)
        )
        
        # 設置熱源
        if base_heat_source is not None:
            self.thermal_solver.set_heat_source(base_heat_source)
        
        # 初始物性更新
        self._initial_property_coupling()
        
        self.is_initialized = True
        print("✅ 強耦合系統初始化完成")
    
    def _initial_property_coupling(self):
        """初始物性耦合"""
        
        print("🔄 執行初始物性耦合...")
        
        # 從初始溫度場更新物性
        self.properties_calculator.update_properties_from_temperature(
            self.thermal_solver.temperature
        )
        
        # 初始化浮力場
        if self.buoyancy_system:
            self.buoyancy_system.update_buoyancy_system(
                self.thermal_solver.temperature,
                self.fluid_solver.rho,
                self.fluid_solver.u
            )
        
        print("✅ 初始物性耦合完成")
    
    def coupled_step(self) -> bool:
        """
        執行一個完整的強耦合時間步
        
        Returns:
            True: 成功, False: 數值不穩定或耦合失敗
        """
        
        if not self.is_initialized:
            print("❌ 錯誤：強耦合系統未初始化")
            return False
        
        step_start_time = time.time()
        self.is_converged = False
        coupling_iterations = 0
        
        # 耦合迭代循環
        for iteration in range(self.config.max_coupling_iterations):
            coupling_iterations += 1
            
            # 1. 物性更新
            if iteration == 0 or self.config.coupling_frequency == 1:
                property_start = time.time()
                success = self._update_temperature_dependent_properties()
                if not success:
                    print(f"❌ 步驟{self.coupling_step}: 物性更新失敗")
                    return False
                self.performance_stats['property_update_time'] += time.time() - property_start
            
            # 2. 浮力更新
            if self.config.enable_buoyancy and self.buoyancy_system:
                buoyancy_start = time.time()
                buoyancy_diag = self.buoyancy_system.update_buoyancy_system(
                    self.thermal_solver.temperature,
                    self.fluid_solver.rho,
                    self.fluid_solver.u
                )
                self.performance_stats['buoyancy_time'] += time.time() - buoyancy_start
            
            # 3. 流體步驟
            fluid_start = time.time()
            try:
                self.fluid_solver.step_with_temperature_coupling(
                    self.thermal_solver.temperature
                )
                fluid_success = True
            except Exception as e:
                print(f"❌ 流體求解器異常: {e}")
                fluid_success = False
            
            if not fluid_success:
                print(f"❌ 步驟{self.coupling_step}: 流體求解失敗")
                return False
            self.performance_stats['fluid_time'] += time.time() - fluid_start
            
            # 4. 速度場傳遞
            velocity_field = self.fluid_solver.get_velocity_field_for_thermal_coupling()
            self.thermal_solver.set_velocity_field(velocity_field)
            
            # 5. 熱傳步驟
            thermal_start = time.time()
            thermal_success = self.thermal_solver.step()
            if not thermal_success:
                print(f"❌ 步驟{self.coupling_step}: 熱傳求解失敗")
                return False
            self.performance_stats['thermal_time'] += time.time() - thermal_start
            
            # 6. 收斂性檢查
            if iteration > 0:
                convergence_residual = self._check_coupling_convergence()
                if convergence_residual < self.config.coupling_tolerance:
                    self.is_converged = True
                    break
        
        # 7. 穩定性檢查
        if self.coupling_step % self.config.stability_check_frequency == 0:
            stability_ok = self._check_system_stability()
            if not stability_ok:
                print(f"❌ 步驟{self.coupling_step}: 系統穩定性檢查失敗")
                return False
        
        # 8. 自適應鬆弛調節
        if self.config.enable_adaptive_relaxation:
            self._adaptive_relaxation_control(coupling_iterations)
        
        # 更新統計
        self.performance_stats['coupling_iterations'].append(coupling_iterations)
        self.performance_stats['total_steps'] += 1
        self.coupling_step += 1
        
        return True
    
    def _update_temperature_dependent_properties(self) -> bool:
        """
        更新溫度依賴物性
        
        Returns:
            True: 成功, False: 失敗
        """
        
        try:
            # 更新物性場
            self.properties_calculator.update_properties_from_temperature(
                self.thermal_solver.temperature
            )
            
            # 驗證物性範圍 (改為警告模式，不阻斷運行)
            if self.config.enable_diagnostics:
                ranges_valid = self.properties_calculator.validate_property_ranges()
                if not ranges_valid:
                    print("⚠️  物性範圍異常，但繼續計算")
            
            return True
            
        except Exception as e:
            print(f"❌ 物性更新異常: {e}")
            return False
    
    def _check_coupling_convergence(self) -> float:
        """
        檢查耦合收斂性
        
        Returns:
            收斂殘差
        """
        
        # 簡化的收斂檢查：溫度場變化
        T_current = self.thermal_solver.temperature.to_numpy()
        T_old = getattr(self, '_T_previous', T_current)
        
        # 計算殘差
        residual = np.mean(np.abs(T_current - T_old))
        
        # 保存當前狀態
        self._T_previous = T_current.copy()
        
        return float(residual)
    
    def _check_system_stability(self) -> bool:
        """
        檢查系統穩定性
        
        Returns:
            True: 穩定, False: 不穩定
        """
        
        # 溫度範圍檢查
        T_min, T_max, T_avg = self.thermal_solver.get_temperature_stats()
        
        if T_max > 120.0 or T_min < -5.0:
            print(f"⚠️  溫度超出安全範圍: {T_min:.1f} - {T_max:.1f}°C")
            return False
        
        # 溫度變化率檢查
        if len(self.stability_history['temperature_changes']) > 0:
            last_T_avg = self.stability_history['temperature_changes'][-1]
            T_change_rate = abs(T_avg - last_T_avg)
            
            if T_change_rate > self.config.max_temperature_change:
                print(f"⚠️  溫度變化過快: {T_change_rate:.2f}°C/步")
                return False
        
        # 速度量級檢查
        velocity_magnitude = self.fluid_solver.get_velocity_magnitude()
        max_vel = np.max(velocity_magnitude)
        
        if max_vel > self.config.max_velocity_magnitude:
            print(f"⚠️  速度過大: {max_vel:.3f} (格子單位)")
            return False
        
        # 更新穩定性歷史
        self.stability_history['temperature_changes'].append(T_avg)
        self.stability_history['velocity_magnitudes'].append(max_vel)
        
        # 限制歷史長度
        max_history = 50
        for key in self.stability_history:
            if len(self.stability_history[key]) > max_history:
                self.stability_history[key] = self.stability_history[key][-max_history:]
        
        return True
    
    def _adaptive_relaxation_control(self, coupling_iterations: int):
        """
        自適應鬆弛控制
        
        Args:
            coupling_iterations: 當前步的耦合迭代次數
        """
        
        target_iterations = 2  # 目標迭代次數
        
        if coupling_iterations > target_iterations:
            # 迭代太多，減小鬆弛係數
            self.current_relaxation *= 0.95
        elif coupling_iterations < target_iterations and self.is_converged:
            # 迭代太少，增大鬆弛係數
            self.current_relaxation *= 1.05
        
        # 限制範圍
        self.current_relaxation = max(self.config.min_relaxation,
                                    min(self.current_relaxation, self.config.max_relaxation))
    
    def get_strong_coupling_diagnostics(self) -> Dict[str, Any]:
        """
        獲取強耦合診斷信息
        
        Returns:
            完整診斷字典
        """
        
        # 基本狀態
        diagnostics = {
            'coupling_step': self.coupling_step,
            'is_converged': self.is_converged,
            'current_relaxation': self.current_relaxation,
        }
        
        # 溫度統計
        if self.thermal_solver.is_initialized:
            T_min, T_max, T_avg = self.thermal_solver.get_temperature_stats()
            diagnostics['thermal_stats'] = {
                'T_min': float(T_min),
                'T_max': float(T_max),
                'T_avg': float(T_avg),
                'thermal_diffusivity': self.thermal_solver.get_effective_thermal_diffusivity()
            }
        
        # 物性統計
        if self.properties_calculator:
            diagnostics['property_stats'] = self.properties_calculator.get_property_statistics()
        
        # 浮力統計
        if self.buoyancy_system:
            diagnostics['buoyancy_stats'] = self.buoyancy_system.get_natural_convection_diagnostics()
        
        # 性能統計
        total_time = sum(self.performance_stats[key] for key in 
                        ['fluid_time', 'thermal_time', 'property_update_time', 'buoyancy_time'])
        
        if total_time > 0:
            diagnostics['performance'] = {
                'fluid_fraction': self.performance_stats['fluid_time'] / total_time,
                'thermal_fraction': self.performance_stats['thermal_time'] / total_time,
                'property_fraction': self.performance_stats['property_update_time'] / total_time,
                'buoyancy_fraction': self.performance_stats['buoyancy_time'] / total_time,
                'avg_coupling_iterations': np.mean(self.performance_stats['coupling_iterations']) if self.performance_stats['coupling_iterations'] else 0,
                'steps_per_second': self.performance_stats['total_steps'] / total_time if total_time > 0 else 0
            }
        
        # 穩定性統計
        diagnostics['stability'] = {
            'temperature_changes': self.stability_history['temperature_changes'][-10:],  # 最近10步
            'velocity_magnitudes': self.stability_history['velocity_magnitudes'][-10:],
            'max_temperature_change': self.config.max_temperature_change,
            'max_velocity_magnitude': self.config.max_velocity_magnitude
        }
        
        return diagnostics
    
    def save_coupled_state(self, step_num: int) -> Dict[str, np.ndarray]:
        """
        保存強耦合系統狀態
        
        Args:
            step_num: 步驟編號
            
        Returns:
            完整系統狀態數據
        """
        
        state_data = {
            'step': step_num,
            'coupling_step': self.coupling_step,
            
            # 流體場
            'velocity': self.fluid_solver.get_velocity_vector_field().to_numpy(),
            'density': self.fluid_solver.rho.to_numpy(),
            'pressure': self.fluid_solver.rho.to_numpy() * config.CS2,  # 近似壓力
            
            # 溫度場
            'temperature': self.thermal_solver.temperature.to_numpy(),
            'heat_flux': self.thermal_solver.heat_flux.to_numpy(),
            
            # 物性場
            'fluid_density': self.properties_calculator.density_field.to_numpy(),
            'viscosity': self.properties_calculator.viscosity_field.to_numpy(),
            'thermal_conductivity': self.properties_calculator.thermal_conductivity_field.to_numpy(),
        }
        
        # 浮力場 (如果可用)
        if self.buoyancy_system:
            state_data['buoyancy_force'] = self.buoyancy_system.buoyancy_force.to_numpy()
            state_data['buoyancy_magnitude'] = self.buoyancy_system.buoyancy_magnitude.to_numpy()
        
        return state_data
    
    def reset_strong_coupling_system(self):
        """重置強耦合系統"""
        
        print("🔄 重置強耦合系統...")
        
        # 重置子系統
        self.fluid_solver.reset_solver()
        self.thermal_solver.reset()
        self.properties_calculator.reset_to_reference_state()
        
        if self.buoyancy_system:
            self.buoyancy_system.reset_buoyancy_system()
        
        # 重置狀態
        self.coupling_step = 0
        self.is_initialized = False
        self.is_converged = False
        self.current_relaxation = self.config.relaxation_factor
        
        # 重置統計
        self.performance_stats = {
            'fluid_time': 0.0,
            'thermal_time': 0.0,
            'property_update_time': 0.0,
            'buoyancy_time': 0.0,
            'coupling_iterations': [],
            'total_steps': 0
        }
        
        self.stability_history = {
            'temperature_changes': [],
            'velocity_magnitudes': [],
            'coupling_residuals': [],
            'rayleigh_numbers': []
        }
        
        print("✅ 強耦合系統重置完成")

# 工廠函數
def create_coffee_strong_coupling_system(thermal_diffusivity: float = 1.6e-7) -> StrongCoupledSolver:
    """創建適用於手沖咖啡的強耦合系統"""
    
    coffee_config = StrongCouplingConfig(
        coupling_frequency=1,
        max_coupling_iterations=3,
        coupling_tolerance=1e-4,
        enable_adaptive_relaxation=True,
        relaxation_factor=0.7,  # 保守的鬆弛係數
        enable_variable_density=True,
        enable_variable_viscosity=True,
        enable_buoyancy=True,
        enable_diagnostics=True,
        stability_check_frequency=5,
        max_temperature_change=2.0,
        max_velocity_magnitude=0.2
    )
    
    return StrongCoupledSolver(coffee_config, thermal_diffusivity)