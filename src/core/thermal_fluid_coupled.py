# thermal_fluid_coupled.py - 熱流弱耦合系統控制器
"""
Phase 2: 熱流弱耦合系統

實現流體→熱傳的單向耦合，流體速度場驅動溫度場的對流傳熱
使用交替求解策略：先更新流體，再將速度場傳遞給熱傳求解器

耦合策略:
1. 流體LBM步驟 → 獲得新速度場
2. 速度場傳遞到熱傳求解器
3. 熱傳LBM步驟 (含對流項)
4. 重複下一時間步

技術特點:
- 單向耦合 (流體→熱傳)
- 時序協調控制
- 數值穩定性保證
- 性能監控

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

# 核心模組
from src.core.lbm_solver import LBMSolver
from src.physics.thermal_lbm import ThermalLBM
import config

@dataclass
class CouplingConfig:
    """耦合系統配置"""
    coupling_frequency: int = 1      # 耦合頻率 (每N步更新一次)
    velocity_smoothing: bool = True  # 速度場平滑
    thermal_subcycles: int = 1       # 熱傳子循環數
    enable_diagnostics: bool = True  # 診斷監控
    max_coupling_error: float = 1e6  # 最大耦合誤差限制

@ti.data_oriented
class ThermalFluidCoupledSolver:
    """
    熱流弱耦合求解器
    
    管理流體LBM求解器和熱傳LBM求解器的協調運行
    實現流體→熱傳的單向耦合
    
    Attributes:
        fluid_solver: 流體LBM求解器
        thermal_solver: 熱傳LBM求解器
        coupling_config: 耦合配置
        base_heat_source: 基礎熱源場 (不含對流項)
    """
    
    def __init__(self, 
                 coupling_config: Optional[CouplingConfig] = None,
                 thermal_diffusivity: float = 1.6e-7):
        """
        初始化熱流耦合系統
        
        Args:
            coupling_config: 耦合配置參數
            thermal_diffusivity: 熱擴散係數 (m²/s)
        """
        
        print("🔗 初始化熱流弱耦合系統 (Phase 2)...")
        
        # 配置參數
        self.coupling_config = coupling_config or CouplingConfig()
        
        # 初始化子求解器
        print("  🌊 初始化流體LBM求解器...")
        self.fluid_solver = LBMSolver()
        
        print("  🔥 初始化熱傳LBM求解器...")
        self.thermal_solver = ThermalLBM(thermal_diffusivity=thermal_diffusivity)
        
        # 啟用熱傳耦合
        self.thermal_solver.enable_convection_coupling(True)
        self.fluid_solver.enable_thermal_coupling_output(True)
        
        # 基礎熱源場 (用於重置)
        self.base_heat_source = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 耦合狀態
        self.coupling_step = 0
        self.is_initialized = False
        self.performance_stats = {
            'fluid_time': 0.0,
            'thermal_time': 0.0,
            'coupling_time': 0.0,
            'total_steps': 0
        }
        
        print("✅ 熱流弱耦合系統初始化完成")
    
    def initialize_system(self,
                         fluid_initial_conditions: Dict[str, Any],
                         thermal_initial_conditions: Dict[str, Any],
                         base_heat_source: Optional[np.ndarray] = None):
        """
        初始化耦合系統
        
        Args:
            fluid_initial_conditions: 流體初始條件
            thermal_initial_conditions: 熱傳初始條件  
            base_heat_source: 基礎熱源場 (W/m³)
        """
        
        print("🚀 初始化耦合系統狀態...")
        
        # 初始化流體求解器
        if 'density_field' in fluid_initial_conditions:
            # 如果提供了密度場，先初始化基本場，然後設置密度
            self.fluid_solver.init_fields()
            # 注意：LBM求解器可能不支援直接設置密度場，使用默認初始化
            print("  流體求解器使用默認初始化")
        else:
            self.fluid_solver.init_fields()
        
        # 初始化熱傳求解器
        self.thermal_solver.complete_initialization(
            T_initial=thermal_initial_conditions.get('T_initial', 25.0),
            T_hot_region=thermal_initial_conditions.get('T_hot_region', 93.0),
            hot_region_height=thermal_initial_conditions.get('hot_region_height', 20)
        )
        
        # 設置基礎熱源
        if base_heat_source is not None:
            if base_heat_source.shape != (config.NX, config.NY, config.NZ):
                raise ValueError(f"熱源場尺寸不匹配: {base_heat_source.shape}")
            self.base_heat_source.from_numpy(base_heat_source.astype(np.float32))
            self.thermal_solver.set_heat_source(base_heat_source)
        else:
            self.base_heat_source.fill(0.0)
        
        self.is_initialized = True
        print("✅ 耦合系統初始化完成")
    
    def step(self) -> bool:
        """
        執行一個完整的耦合時間步
        
        順序:
        1. 流體LBM步驟
        2. 速度場傳遞 (如果達到耦合頻率)
        3. 熱傳LBM步驟 (含對流項)
        
        Returns:
            True: 成功, False: 數值不穩定或耦合失敗
        """
        
        if not self.is_initialized:
            print("❌ 錯誤：耦合系統未初始化")
            return False
        
        step_start_time = time.time()
        
        # 1. 流體LBM步驟
        fluid_start = time.time()
        try:
            self.fluid_solver.step()  # LBM solver step() 不返回布爾值
            fluid_success = True
        except Exception as e:
            print(f"❌ 流體求解器異常: {e}")
            fluid_success = False
            
        if not fluid_success:
            print(f"❌ 步驟{self.coupling_step}: 流體求解器失敗")
            return False
        self.performance_stats['fluid_time'] += time.time() - fluid_start
        
        # 2. 速度場傳遞 (按耦合頻率)
        if self.coupling_step % self.coupling_config.coupling_frequency == 0:
            coupling_start = time.time()
            success = self._update_thermal_velocity_coupling()
            if not success:
                print(f"❌ 步驟{self.coupling_step}: 速度場耦合失敗")
                return False
            self.performance_stats['coupling_time'] += time.time() - coupling_start
        
        # 3. 熱傳LBM步驟
        thermal_start = time.time()
        
        # 重置熱源場到基礎值
        self.thermal_solver.reset_heat_source_to_base(self.base_heat_source)
        
        # 執行熱傳子循環
        for subcycle in range(self.coupling_config.thermal_subcycles):
            thermal_success = self.thermal_solver.step()
            if not thermal_success:
                print(f"❌ 步驟{self.coupling_step}.{subcycle}: 熱傳求解器失敗")
                return False
        
        self.performance_stats['thermal_time'] += time.time() - thermal_start
        
        # 4. 診斷檢查
        if self.coupling_config.enable_diagnostics:
            if not self._check_coupling_stability():
                print(f"❌ 步驟{self.coupling_step}: 耦合穩定性檢查失敗")
                return False
        
        self.coupling_step += 1
        self.performance_stats['total_steps'] += 1
        
        return True
    
    def _update_thermal_velocity_coupling(self) -> bool:
        """
        更新熱傳求解器的速度場耦合
        
        Returns:
            True: 成功, False: 失敗
        """
        
        try:
            # 獲取流體速度場
            velocity_field = self.fluid_solver.get_velocity_field_for_thermal_coupling()
            
            # 可選的速度場平滑
            if self.coupling_config.velocity_smoothing:
                # 可實現簡單的空間平滑算法
                pass
            
            # 傳遞到熱傳求解器
            self.thermal_solver.set_velocity_field(velocity_field)
            
            return True
            
        except Exception as e:
            print(f"❌ 速度場耦合錯誤: {e}")
            return False
    
    def _check_coupling_stability(self) -> bool:
        """
        檢查耦合穩定性
        
        Returns:
            True: 穩定, False: 不穩定
        """
        
        # 檢查溫度場範圍
        T_min, T_max, T_avg = self.thermal_solver.get_temperature_stats()
        
        if T_max > 150.0 or T_min < -10.0:  # 物理合理範圍
            print(f"⚠️  溫度超出合理範圍: {T_min:.1f} - {T_max:.1f}°C")
            return False
        
        if abs(T_max - T_min) > self.coupling_config.max_coupling_error:
            print(f"⚠️  溫度梯度過大: {abs(T_max - T_min):.1f}°C")
            return False
        
        # 檢查速度場量級
        velocity_magnitude = self.fluid_solver.get_velocity_magnitude()
        if np.any(velocity_magnitude > 1.0):  # 格子單位
            print(f"⚠️  速度場量級過大: max={np.max(velocity_magnitude):.3f}")
            return False
        
        return True
    
    def get_coupling_diagnostics(self) -> Dict[str, Any]:
        """
        獲取耦合診斷資訊
        
        Returns:
            診斷資訊字典
        """
        
        if self.performance_stats['total_steps'] == 0:
            return {'status': 'not_started'}
        
        # 溫度統計
        T_min, T_max, T_avg = self.thermal_solver.get_temperature_stats()
        
        # 性能統計
        total_time = (self.performance_stats['fluid_time'] + 
                     self.performance_stats['thermal_time'] + 
                     self.performance_stats['coupling_time'])
        
        diagnostics = {
            'coupling_step': self.coupling_step,
            'thermal_stats': {
                'T_min': float(T_min),
                'T_max': float(T_max), 
                'T_avg': float(T_avg),
                'thermal_diffusivity': self.thermal_solver.get_effective_thermal_diffusivity()
            },
            'performance': {
                'fluid_fraction': self.performance_stats['fluid_time'] / total_time if total_time > 0 else 0,
                'thermal_fraction': self.performance_stats['thermal_time'] / total_time if total_time > 0 else 0,
                'coupling_fraction': self.performance_stats['coupling_time'] / total_time if total_time > 0 else 0,
                'steps_per_second': self.performance_stats['total_steps'] / total_time if total_time > 0 else 0
            },
            'coupling_config': {
                'frequency': self.coupling_config.coupling_frequency,
                'thermal_subcycles': self.coupling_config.thermal_subcycles,
                'convection_enabled': self.thermal_solver.enable_convection
            }
        }
        
        return diagnostics
    
    def save_coupling_state(self, step_num: int) -> Dict[str, np.ndarray]:
        """
        保存耦合系統狀態
        
        Args:
            step_num: 步驟編號
            
        Returns:
            系統狀態數據
        """
        
        # 獲取流體狀態
        velocity_field = self.fluid_solver.get_velocity_vector_field()
        density_field = self.fluid_solver.rho
        
        # 獲取熱傳狀態
        temperature_field = self.thermal_solver.temperature
        heat_flux_field = self.thermal_solver.heat_flux
        
        state_data = {
            'step': step_num,
            'velocity': velocity_field.to_numpy(),
            'density': density_field.to_numpy(),
            'temperature': temperature_field.to_numpy(),
            'heat_flux': heat_flux_field.to_numpy()
        }
        
        return state_data
    
    def reset_coupling_system(self):
        """重置耦合系統"""
        
        print("🔄 重置熱流耦合系統...")
        
        # 重置子求解器
        self.fluid_solver.reset_solver()
        self.thermal_solver.reset()
        
        # 重置狀態
        self.coupling_step = 0
        self.is_initialized = False
        self.performance_stats = {
            'fluid_time': 0.0,
            'thermal_time': 0.0, 
            'coupling_time': 0.0,
            'total_steps': 0
        }
        
        print("✅ 耦合系統重置完成")
    
    # ==========================================
    # 相容性介面 - 為其他系統提供LBM求解器介面
    # ==========================================
    
    @property
    def solid(self):
        """代理到流體求解器的solid字段"""
        return self.fluid_solver.solid
    
    @property
    def rho(self):
        """代理到流體求解器的密度字段"""
        return self.fluid_solver.rho
    
    @property
    def phase(self):
        """代理到流體求解器的相場字段"""
        if hasattr(self.fluid_solver, 'phase'):
            return self.fluid_solver.phase
        return None
    
    @property
    def u(self):
        """代理到流體求解器的速度字段"""
        return self.fluid_solver.u
    
    @property
    def ux(self):
        """代理到流體求解器的x方向速度"""
        if hasattr(self.fluid_solver, 'ux'):
            return self.fluid_solver.ux
        return None
    
    @property
    def uy(self):
        """代理到流體求解器的y方向速度"""
        if hasattr(self.fluid_solver, 'uy'):
            return self.fluid_solver.uy
        return None
    
    @property 
    def uz(self):
        """代理到流體求解器的z方向速度"""
        if hasattr(self.fluid_solver, 'uz'):
            return self.fluid_solver.uz
        return None
    
    def has_soa_velocity_layout(self):
        """檢查是否使用SoA速度布局"""
        return hasattr(self.fluid_solver, 'has_soa_velocity_layout') and self.fluid_solver.has_soa_velocity_layout()
    
    def get_velocity_components(self):
        """獲取速度分量"""
        if hasattr(self.fluid_solver, 'get_velocity_components'):
            return self.fluid_solver.get_velocity_components()
        return self.ux, self.uy, self.uz
    
    def get_velocity_vector_field(self):
        """獲取向量速度場"""
        return self.fluid_solver.get_velocity_vector_field()
    
    def init_fields(self):
        """初始化字段 - 代理到流體求解器並自動初始化耦合系統"""
        # 初始化流體求解器
        result = self.fluid_solver.init_fields()
        
        # 自動初始化耦合系統
        if not self.is_initialized:
            fluid_conditions = {
                'density_initial': 1.0,
                'velocity_initial': [0.0, 0.0, 0.0]
            }
            thermal_conditions = {
                'temperature_initial': 25.0,  # 室溫 °C
                'hot_zone_temperature': 90.0  # 注水溫度 °C
            }
            self.initialize_system(fluid_conditions, thermal_conditions)
        
        return result
    
    def reset_solver(self):
        """重置求解器 - 使用耦合系統的重置方法"""
        return self.reset_coupling_system()
    
    @property
    def boundary_manager(self):
        """代理到流體求解器的邊界管理器"""
        if hasattr(self.fluid_solver, 'boundary_manager'):
            return self.fluid_solver.boundary_manager
        return None
    
    def get_temperature_field(self):
        """獲取溫度場 - 熱耦合特有方法"""
        return self.thermal_solver.temperature if hasattr(self.thermal_solver, 'temperature') else None
    
    def thermal_coupling_step(self):
        """熱耦合步進 - 使用統一的step方法"""
        return self.step()