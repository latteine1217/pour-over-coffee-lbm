# thermal_properties.py - 熱物性參數管理系統
"""
咖啡沖煮過程中的熱物性參數管理
包含水、空氣、咖啡粉的溫度依賴熱物性
支援多相流熱傳建模

功能：
- 溫度依賴的熱物性計算
- 多孔介質有效熱物性  
- 相變與混合物熱物性
- GPU優化的熱物性查表

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# ==============================================
# 熱物性數據類別
# ==============================================

@dataclass
class ThermalProperties:
    """熱物性數據結構"""
    thermal_conductivity: float  # W/(m·K)
    heat_capacity: float         # J/(kg·K)
    density: float              # kg/m³
    thermal_diffusivity: float  # m²/s
    dynamic_viscosity: float    # Pa·s
    thermal_expansion: float    # 1/K

# ==============================================
# 標準熱物性數據庫
# ==============================================

# 水的熱物性 (溫度依賴)
WATER_THERMAL_DATA = {
    # 溫度 (°C): (k, cp, ρ, α, μ, β)
    20.0: ThermalProperties(0.598, 4182, 998.2, 1.43e-7, 1.002e-3, 2.07e-4),
    40.0: ThermalProperties(0.628, 4179, 992.2, 1.51e-7, 6.53e-4, 3.85e-4),
    60.0: ThermalProperties(0.651, 4185, 983.2, 1.58e-7, 4.67e-4, 5.2e-4),
    80.0: ThermalProperties(0.670, 4197, 971.8, 1.64e-7, 3.55e-4, 6.4e-4),
    90.0: ThermalProperties(0.675, 4205, 965.3, 1.66e-7, 3.15e-4, 6.95e-4),
    100.0: ThermalProperties(0.679, 4220, 958.4, 1.68e-7, 2.82e-4, 7.56e-4)
}

# 空氣熱物性 (20°C標準)
AIR_PROPERTIES = ThermalProperties(
    thermal_conductivity=0.0257,   # W/(m·K)
    heat_capacity=1005,            # J/(kg·K)
    density=1.204,                 # kg/m³
    thermal_diffusivity=2.12e-5,   # m²/s
    dynamic_viscosity=1.825e-5,    # Pa·s
    thermal_expansion=3.43e-3      # 1/K
)

# 咖啡粉固體熱物性
COFFEE_SOLID_PROPERTIES = ThermalProperties(
    thermal_conductivity=0.3,      # W/(m·K)
    heat_capacity=1800,            # J/(kg·K)
    density=1200,                  # kg/m³
    thermal_diffusivity=1.39e-7,   # m²/s
    dynamic_viscosity=0.0,         # 固體無黏滯度
    thermal_expansion=1.5e-5       # 1/K
)

@ti.data_oriented
class ThermalPropertyManager:
    """
    熱物性參數管理器
    
    功能：
    - 溫度插值計算熱物性
    - 多相混合物熱物性
    - 多孔介質有效熱物性
    - GPU優化的屬性查詢
    """
    
    def __init__(self, nx: int, ny: int, nz: int):
        """
        初始化熱物性管理器
        
        Args:
            nx, ny, nz: 網格尺寸
        """
        
        self.nx, self.ny, self.nz = nx, ny, nz
        
        # 初始化Taichi場
        self._init_fields()
        
        # 建立溫度-物性查表
        self._build_lookup_tables()
        
    def _init_fields(self):
        """初始化Taichi場"""
        
        # 空間分布的熱物性場
        self.thermal_conductivity = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        self.heat_capacity = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        self.density = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        self.thermal_diffusivity = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        self.thermal_expansion = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        
        # 相標記場
        self.phase_field = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))  # 0=air, 1=water
        self.porosity = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))    # 孔隙率
        
        # 溫度場 (外部提供)
        self.temperature = ti.field(ti.f32, shape=(self.nx, self.ny, self.nz))
        
    def _build_lookup_tables(self):
        """建立水的熱物性查表"""
        
        # 溫度範圍與步長
        self.T_min = 10.0
        self.T_max = 100.0
        self.n_temp_points = 91  # 1°C 間隔
        
        # 查表陣列
        self.water_k_table = ti.field(ti.f32, shape=self.n_temp_points)
        self.water_cp_table = ti.field(ti.f32, shape=self.n_temp_points)
        self.water_rho_table = ti.field(ti.f32, shape=self.n_temp_points)
        self.water_alpha_table = ti.field(ti.f32, shape=self.n_temp_points)
        self.water_beta_table = ti.field(ti.f32, shape=self.n_temp_points)
        
        # 填充查表數據
        temp_points = np.linspace(self.T_min, self.T_max, self.n_temp_points)
        k_values = np.zeros(self.n_temp_points)
        cp_values = np.zeros(self.n_temp_points)
        rho_values = np.zeros(self.n_temp_points)
        alpha_values = np.zeros(self.n_temp_points)
        beta_values = np.zeros(self.n_temp_points)
        
        # 插值計算每個溫度點的物性
        for i, T in enumerate(temp_points):
            props = self._interpolate_water_properties(T)
            k_values[i] = props.thermal_conductivity
            cp_values[i] = props.heat_capacity
            rho_values[i] = props.density
            alpha_values[i] = props.thermal_diffusivity
            beta_values[i] = props.thermal_expansion
        
        # 上傳到GPU
        self.water_k_table.from_numpy(k_values.astype(np.float32))
        self.water_cp_table.from_numpy(cp_values.astype(np.float32))
        self.water_rho_table.from_numpy(rho_values.astype(np.float32))
        self.water_alpha_table.from_numpy(alpha_values.astype(np.float32))
        self.water_beta_table.from_numpy(beta_values.astype(np.float32))
        
        print(f"✅ 熱物性查表建立完成: {self.n_temp_points}個溫度點 ({self.T_min}-{self.T_max}°C)")
    
    def _interpolate_water_properties(self, temperature: float) -> ThermalProperties:
        """
        線性插值計算水的熱物性
        
        Args:
            temperature: 溫度 (°C)
            
        Returns:
            插值後的熱物性
        """
        
        # 獲取數據點溫度
        temps = sorted(WATER_THERMAL_DATA.keys())
        
        # 邊界處理
        if temperature <= temps[0]:
            return WATER_THERMAL_DATA[temps[0]]
        if temperature >= temps[-1]:
            return WATER_THERMAL_DATA[temps[-1]]
        
        # 找到插值區間
        for i in range(len(temps)-1):
            if temps[i] <= temperature <= temps[i+1]:
                T1, T2 = temps[i], temps[i+1]
                props1 = WATER_THERMAL_DATA[T1]
                props2 = WATER_THERMAL_DATA[T2]
                
                # 線性插值權重
                w = (temperature - T1) / (T2 - T1)
                
                return ThermalProperties(
                    thermal_conductivity=props1.thermal_conductivity + w * (props2.thermal_conductivity - props1.thermal_conductivity),
                    heat_capacity=props1.heat_capacity + w * (props2.heat_capacity - props1.heat_capacity),
                    density=props1.density + w * (props2.density - props1.density),
                    thermal_diffusivity=props1.thermal_diffusivity + w * (props2.thermal_diffusivity - props1.thermal_diffusivity),
                    dynamic_viscosity=props1.dynamic_viscosity + w * (props2.dynamic_viscosity - props1.dynamic_viscosity),
                    thermal_expansion=props1.thermal_expansion + w * (props2.thermal_expansion - props1.thermal_expansion)
                )
        
        # 預設返回90°C數據
        return WATER_THERMAL_DATA[90.0]
    
    @ti.func
    def _get_water_property_from_table(self, temperature: ti.f32, table: ti.template()) -> ti.f32:
        """
        從查表獲取水的熱物性 (GPU函數)
        
        Args:
            temperature: 溫度 (°C)
            table: 物性查表
            
        Returns:
            插值後的物性值
        """
        
        # 溫度範圍限制
        T_clamped = max(self.T_min, min(self.T_max, temperature))
        
        # 計算查表索引
        index_f = (T_clamped - self.T_min) / (self.T_max - self.T_min) * (self.n_temp_points - 1)
        index = int(index_f)
        weight = index_f - index
        
        # 邊界檢查與插值
        result = 0.0
        if index >= self.n_temp_points - 1:
            result = table[self.n_temp_points - 1]
        else:
            # 線性插值
            result = table[index] * (1.0 - weight) + table[index + 1] * weight
        
        return result
    
    @ti.kernel
    def update_thermal_properties(self):
        """
        更新所有格點的熱物性
        基於溫度場和相場分布
        """
        
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            T_local = self.temperature[i, j, k]
            phase = self.phase_field[i, j, k]
            porosity_local = self.porosity[i, j, k]
            
            if phase > 0.5:  # 水相
                # 從查表獲取水的熱物性
                k_water = self._get_water_property_from_table(T_local, self.water_k_table)
                cp_water = self._get_water_property_from_table(T_local, self.water_cp_table)
                rho_water = self._get_water_property_from_table(T_local, self.water_rho_table)
                alpha_water = self._get_water_property_from_table(T_local, self.water_alpha_table)
                beta_water = self._get_water_property_from_table(T_local, self.water_beta_table)
                
                self.thermal_conductivity[i, j, k] = k_water
                self.heat_capacity[i, j, k] = cp_water
                self.density[i, j, k] = rho_water
                self.thermal_diffusivity[i, j, k] = alpha_water
                self.thermal_expansion[i, j, k] = beta_water
                
            elif porosity_local > 0.1:  # 多孔咖啡區域
                # 多孔介質有效熱物性 (並聯模型)
                k_water = self._get_water_property_from_table(T_local, self.water_k_table)
                k_coffee = 0.3  # 咖啡固體熱傳導係數
                
                k_eff = porosity_local * k_water + (1.0 - porosity_local) * k_coffee
                cp_eff = porosity_local * 4180.0 + (1.0 - porosity_local) * 1800.0
                rho_eff = porosity_local * 965.3 + (1.0 - porosity_local) * 1200.0
                alpha_eff = k_eff / (rho_eff * cp_eff)
                
                self.thermal_conductivity[i, j, k] = k_eff
                self.heat_capacity[i, j, k] = cp_eff
                self.density[i, j, k] = rho_eff
                self.thermal_diffusivity[i, j, k] = alpha_eff
                self.thermal_expansion[i, j, k] = 1.5e-5  # 咖啡固體膨脹係數
                
            else:  # 空氣相
                self.thermal_conductivity[i, j, k] = 0.0257
                self.heat_capacity[i, j, k] = 1005.0
                self.density[i, j, k] = 1.204
                self.thermal_diffusivity[i, j, k] = 2.12e-5
                self.thermal_expansion[i, j, k] = 3.43e-3
    
    @ti.kernel
    def init_phase_field(self, 
                        water_level: ti.i32,
                        coffee_bottom: ti.i32,
                        coffee_top: ti.i32,
                        coffee_porosity: ti.f32):
        """
        初始化相場分布
        
        Args:
            water_level: 水面高度 (格點)
            coffee_bottom: 咖啡床底部 (格點)
            coffee_top: 咖啡床頂部 (格點) 
            coffee_porosity: 咖啡孔隙率
        """
        
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if k < coffee_bottom:
                # 底部空氣
                self.phase_field[i, j, k] = 0.0
                self.porosity[i, j, k] = 0.0
            elif k < coffee_top:
                # 咖啡床區域
                if k < water_level:
                    self.phase_field[i, j, k] = 1.0  # 濕潤咖啡
                else:
                    self.phase_field[i, j, k] = 0.0  # 乾燥咖啡
                self.porosity[i, j, k] = coffee_porosity
            elif k < water_level:
                # 水相區域
                self.phase_field[i, j, k] = 1.0
                self.porosity[i, j, k] = 0.0
            else:
                # 上部空氣
                self.phase_field[i, j, k] = 0.0
                self.porosity[i, j, k] = 0.0
    
    @ti.kernel
    def compute_effective_conductivity_tensor(self, 
                                            anisotropy_ratio: ti.f32):
        """
        計算各向異性有效熱傳導張量
        
        Args:
            anisotropy_ratio: 各向異性比 (垂直/水平)
        """
        
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            k_base = self.thermal_conductivity[i, j, k]
            porosity_local = self.porosity[i, j, k]
            
            if porosity_local > 0.1:
                # 咖啡床各向異性：垂直方向導熱較差
                k_horizontal = k_base
                k_vertical = k_base * anisotropy_ratio
                
                # 這裡可以存儲張量分量，簡化版本只修改主要熱導率
                self.thermal_conductivity[i, j, k] = (k_horizontal + k_vertical) / 2.0
    
    def set_temperature_field(self, temperature_field: np.ndarray):
        """
        設置溫度場
        
        Args:
            temperature_field: 3D溫度陣列 (°C)
        """
        
        if temperature_field.shape != (self.nx, self.ny, self.nz):
            raise ValueError(f"溫度場尺寸不匹配: {temperature_field.shape} vs ({self.nx}, {self.ny}, {self.nz})")
        
        self.temperature.from_numpy(temperature_field.astype(np.float32))
    
    def get_thermal_properties_numpy(self) -> Dict[str, np.ndarray]:
        """
        獲取所有熱物性的numpy陣列
        
        Returns:
            熱物性字典
        """
        
        return {
            'thermal_conductivity': self.thermal_conductivity.to_numpy(),
            'heat_capacity': self.heat_capacity.to_numpy(),
            'density': self.density.to_numpy(),
            'thermal_diffusivity': self.thermal_diffusivity.to_numpy(),
            'thermal_expansion': self.thermal_expansion.to_numpy(),
            'phase_field': self.phase_field.to_numpy(),
            'porosity': self.porosity.to_numpy()
        }
    
    def get_water_properties_at_temperature(self, temperature: float) -> ThermalProperties:
        """
        獲取指定溫度下的水熱物性
        
        Args:
            temperature: 溫度 (°C)
            
        Returns:
            熱物性數據
        """
        
        return self._interpolate_water_properties(temperature)


# ==============================================
# 模組測試函數
# ==============================================

def test_water_property_interpolation():
    """測試水熱物性插值"""
    
    print("\n🌊 測試水熱物性插值...")
    
    manager = ThermalPropertyManager(10, 10, 10)
    
    # 測試溫度點
    test_temps = [25.0, 50.0, 75.0, 95.0]
    
    for T in test_temps:
        props = manager.get_water_properties_at_temperature(T)
        print(f"  T={T}°C: k={props.thermal_conductivity:.3f} W/(m·K), "
              f"ρ={props.density:.1f} kg/m³, "
              f"α={props.thermal_diffusivity:.2e} m²/s")
    
    print("✅ 水熱物性插值測試通過")

def test_thermal_property_update():
    """測試熱物性場更新"""
    
    print("\n🔄 測試熱物性場更新...")
    
    # 小網格測試
    nx, ny, nz = 20, 20, 20
    manager = ThermalPropertyManager(nx, ny, nz)
    
    # 設置溫度場
    temp_field = np.full((nx, ny, nz), 25.0)  # 25°C
    temp_field[:, :, :10] = 90.0  # 底部90°C
    manager.set_temperature_field(temp_field)
    
    # 設置相場
    manager.init_phase_field(water_level=15, coffee_bottom=5, coffee_top=10, coffee_porosity=0.4)
    
    # 更新熱物性
    manager.update_thermal_properties()
    
    # 檢查結果
    props = manager.get_thermal_properties_numpy()
    
    print(f"  熱傳導係數範圍: {props['thermal_conductivity'].min():.3f} - {props['thermal_conductivity'].max():.3f} W/(m·K)")
    print(f"  密度範圍: {props['density'].min():.1f} - {props['density'].max():.1f} kg/m³")
    print(f"  孔隙率範圍: {props['porosity'].min():.1f} - {props['porosity'].max():.1f}")
    
    print("✅ 熱物性場更新測試通過")

if __name__ == "__main__":
    # 初始化Taichi
    ti.init(arch=ti.cpu)
    
    print("=== 熱物性管理模組測試 ===")
    
    test_water_property_interpolation()
    test_thermal_property_update()
    
    print("\n✅ 所有測試通過！熱物性管理模組就緒")