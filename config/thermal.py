# config/thermal.py - 熱傳參數統一配置
"""
熱傳導LBM系統的專用配置參數 - 統一版本
解決了原始的溫度參數衝突，統一以90°C為標準

與核心配置統一依賴，確保參數一致性
消除了D3Q7參數重複定義

開發：opencode + GitHub Copilot
"""

import math
import numpy as np

# 從統一核心配置導入D3Q7參數（避免重複定義）
from config.core import (
    NX, NY, NZ, SCALE_LENGTH, SCALE_TIME, PHYSICAL_DOMAIN_SIZE,
    Q_THERMAL, CS2_THERMAL, INV_CS2_THERMAL,
    CX_THERMAL, CY_THERMAL, CZ_THERMAL, W_THERMAL,
    MIN_TAU_THERMAL, MAX_TAU_THERMAL, MAX_CFL_THERMAL
)

# 從物理配置導入基礎溫度設定（統一標準）
from config.physics import (
    WATER_TEMP_C, AMBIENT_TEMP_C, COFFEE_INITIAL_TEMP_C,
    WATER_THERMAL_CONDUCTIVITY, WATER_HEAT_CAPACITY, WATER_DENSITY_90C,
    WATER_THERMAL_DIFFUSIVITY, WATER_THERMAL_EXPANSION,
    AIR_THERMAL_CONDUCTIVITY, AIR_HEAT_CAPACITY, AIR_DENSITY_20C,
    AIR_THERMAL_DIFFUSIVITY, AIR_THERMAL_EXPANSION,
    COFFEE_THERMAL_CONDUCTIVITY, COFFEE_HEAT_CAPACITY, COFFEE_BEAN_DENSITY,
    COFFEE_THERMAL_DIFFUSIVITY, COFFEE_THERMAL_EXPANSION, COFFEE_POROSITY
)

# ==============================================
# 熱傳邊界條件 - 統一溫度標準
# ==============================================

# 溫度設定 (°C) - 解決溫度衝突，統一為90°C標準
T_INITIAL = AMBIENT_TEMP_C          # 25°C 初始環境溫度
T_INLET = WATER_TEMP_C              # 90°C 注水溫度 (統一標準)
T_AMBIENT = AMBIENT_TEMP_C          # 25°C 環境溫度
T_COFFEE_INITIAL = COFFEE_INITIAL_TEMP_C  # 25°C 咖啡粉初始溫度

# 溫度範圍限制
T_MIN_PHYSICAL = 10.0      # 最低物理溫度
T_MAX_PHYSICAL = 100.0     # 最高物理溫度
T_MIN_STABLE = 15.0        # 數值穩定下限
T_MAX_STABLE = 98.0        # 數值穩定上限

# 溫度轉換為格子單位
T_CHAR = T_INLET - T_AMBIENT         # 特徵溫度差 (90°C - 25°C = 65°C)
T_INITIAL_LU = (T_INITIAL - T_AMBIENT) / T_CHAR  # 0.0
T_INLET_LU = (T_INLET - T_AMBIENT) / T_CHAR      # 1.0  
T_COFFEE_LU = (T_COFFEE_INITIAL - T_AMBIENT) / T_CHAR  # 0.0

# ==============================================
# 熱傳導LBM數值參數
# ==============================================

# 格子單位熱擴散係數轉換
ALPHA_WATER_LU = WATER_THERMAL_DIFFUSIVITY * SCALE_TIME / (SCALE_LENGTH**2)
ALPHA_AIR_LU = AIR_THERMAL_DIFFUSIVITY * SCALE_TIME / (SCALE_LENGTH**2)
ALPHA_COFFEE_LU = COFFEE_THERMAL_DIFFUSIVITY * SCALE_TIME / (SCALE_LENGTH**2)

# 熱傳導鬆弛時間 τ = α/(c_s²) + 0.5
TAU_THERMAL_WATER = ALPHA_WATER_LU / CS2_THERMAL + 0.5
TAU_THERMAL_AIR = ALPHA_AIR_LU / CS2_THERMAL + 0.5
TAU_THERMAL_COFFEE = ALPHA_COFFEE_LU / CS2_THERMAL + 0.5

# 安全範圍檢查
def validate_tau_thermal(tau, name):
    """驗證熱傳鬆弛時間安全範圍"""
    if tau < MIN_TAU_THERMAL:
        print(f"⚠️  {name} 鬆弛時間過低: {tau:.3f} < {MIN_TAU_THERMAL}")
        return MIN_TAU_THERMAL
    elif tau > MAX_TAU_THERMAL:
        print(f"⚠️  {name} 鬆弛時間過高: {tau:.3f} > {MAX_TAU_THERMAL}")
        return MAX_TAU_THERMAL
    return tau

# 應用安全限制
TAU_THERMAL_WATER = validate_tau_thermal(TAU_THERMAL_WATER, "水相熱傳")
TAU_THERMAL_AIR = validate_tau_thermal(TAU_THERMAL_AIR, "氣相熱傳")
TAU_THERMAL_COFFEE = validate_tau_thermal(TAU_THERMAL_COFFEE, "咖啡相熱傳")

# ==============================================
# 多孔介質熱物性 (咖啡床)
# ==============================================

# 有效熱導率 (體積平均)
COFFEE_EFFECTIVE_CONDUCTIVITY = (COFFEE_POROSITY * WATER_THERMAL_CONDUCTIVITY + 
                                (1 - COFFEE_POROSITY) * COFFEE_THERMAL_CONDUCTIVITY)

# 有效熱容 (質量平均)
COFFEE_EFFECTIVE_HEAT_CAPACITY = (COFFEE_POROSITY * WATER_HEAT_CAPACITY + 
                                 (1 - COFFEE_POROSITY) * COFFEE_HEAT_CAPACITY)

# 有效密度 (體積平均)
COFFEE_EFFECTIVE_DENSITY = (COFFEE_POROSITY * WATER_DENSITY_90C + 
                           (1 - COFFEE_POROSITY) * COFFEE_BEAN_DENSITY)

# 有效熱擴散率
COFFEE_EFFECTIVE_DIFFUSIVITY = (COFFEE_EFFECTIVE_CONDUCTIVITY / 
                               (COFFEE_EFFECTIVE_DENSITY * COFFEE_EFFECTIVE_HEAT_CAPACITY))

# ==============================================
# 熱邊界條件參數
# ==============================================

# 對流換熱係數 (W/(m²·K))
H_CONVECTION_AIR = 5.0             # 自然對流
H_CONVECTION_FORCED = 20.0         # 強制對流

# 輻射換熱係數 (W/(m²·K))
EMISSIVITY_COFFEE = 0.8            # 咖啡表面發射率
STEFAN_BOLTZMANN = 5.67e-8         # W/(m²·K⁴)

# Biot數計算 (對流與導熱比)
BIOT_NUMBER_CONVECTION = (H_CONVECTION_AIR * SCALE_LENGTH) / WATER_THERMAL_CONDUCTIVITY

# ==============================================
# 熱流耦合控制參數
# ==============================================

# 耦合時間步控制
THERMAL_COUPLING_FREQ = 1          # 每步耦合
THERMAL_SUB_STEPS = 1              # 熱傳子步數

# 溫度相依性控制
ENABLE_TEMP_DEPENDENT_VISCOSITY = True
ENABLE_BUOYANCY = True
ENABLE_THERMAL_EXPANSION = True

# 浮力係數 (基於Rayleigh數)
RAYLEIGH_NUMBER = (WATER_THERMAL_EXPANSION * 9.81 * T_CHAR * (SCALE_LENGTH**3)) / \
                  (3.15e-7 * WATER_THERMAL_DIFFUSIVITY)  # 使用90°C水的黏滯度

BUOYANCY_COEFFICIENT = WATER_THERMAL_EXPANSION * 9.81 * SCALE_TIME**2 / SCALE_LENGTH

# ==============================================
# 診斷與驗證函數
# ==============================================

def print_thermal_diagnostics():
    """輸出熱傳參數診斷信息"""
    
    print("\n=== 熱傳參數診斷 ===")
    
    print(f"🌡️  溫度設定:")
    print(f"  入水溫度: {T_INLET:.0f}°C")
    print(f"  環境溫度: {T_AMBIENT:.0f}°C") 
    print(f"  溫度差: {T_CHAR:.0f}°C")
    
    print(f"\n⚡ 熱傳鬆弛時間:")
    print(f"  水相: {TAU_THERMAL_WATER:.3f}")
    print(f"  氣相: {TAU_THERMAL_AIR:.3f}")
    print(f"  咖啡相: {TAU_THERMAL_COFFEE:.3f}")
    
    print(f"\n🔄 熱擴散係數 (LU):")
    print(f"  水: {ALPHA_WATER_LU:.2e}")
    print(f"  空氣: {ALPHA_AIR_LU:.2e}")
    print(f"  咖啡: {ALPHA_COFFEE_LU:.2e}")
    
    print(f"\n☕ 咖啡床熱物性:")
    print(f"  有效熱導率: {COFFEE_EFFECTIVE_CONDUCTIVITY:.3f} W/(m·K)")
    print(f"  有效熱容: {COFFEE_EFFECTIVE_HEAT_CAPACITY:.0f} J/(kg·K)")
    print(f"  有效密度: {COFFEE_EFFECTIVE_DENSITY:.1f} kg/m³")
    
    print(f"\n🌊 無量綱數:")
    print(f"  Rayleigh數: {RAYLEIGH_NUMBER:.1e}")
    print(f"  Biot數: {BIOT_NUMBER_CONVECTION:.3f}")

def validate_thermal_config():
    """驗證熱傳配置的合理性"""
    
    errors = []
    warnings = []
    
    # 溫度範圍檢查
    if T_INLET < T_MIN_STABLE or T_INLET > T_MAX_STABLE:
        warnings.append(f"入水溫度超出穩定範圍: {T_INLET}°C")
    
    # 鬆弛時間檢查
    for tau, name in [(TAU_THERMAL_WATER, "水相"), 
                      (TAU_THERMAL_AIR, "氣相"), 
                      (TAU_THERMAL_COFFEE, "咖啡相")]:
        if tau < MIN_TAU_THERMAL:
            errors.append(f"{name}熱傳鬆弛時間過低: {tau:.3f}")
        elif tau > MAX_TAU_THERMAL:
            warnings.append(f"{name}熱傳鬆弛時間偏高: {tau:.3f}")
    
    # 熱擴散穩定性檢查
    thermal_cfl = max(ALPHA_WATER_LU, ALPHA_AIR_LU, ALPHA_COFFEE_LU) / CS2_THERMAL
    if thermal_cfl > MAX_CFL_THERMAL:
        warnings.append(f"熱擴散CFL過高: {thermal_cfl:.3f} > {MAX_CFL_THERMAL}")
    
    # Rayleigh數檢查
    if RAYLEIGH_NUMBER > 1e6:
        warnings.append(f"Rayleigh數過高，可能出現湍流對流: {RAYLEIGH_NUMBER:.1e}")
    
    # 報告結果
    if errors:
        print(f"\n❌ 熱傳配置錯誤:")
        for error in errors:
            print(f"  • {error}")
        return False
    
    if warnings:
        print(f"\n⚠️  熱傳配置警告:")
        for warning in warnings:
            print(f"  • {warning}")
    
    print(f"✅ 熱傳配置驗證通過")
    return True

def get_thermal_summary():
    """獲取熱傳參數摘要"""
    
    thermal_cfl = max(ALPHA_WATER_LU, ALPHA_AIR_LU, ALPHA_COFFEE_LU) / CS2_THERMAL
    
    return {
        'temperatures': {
            'inlet_temp': T_INLET,
            'ambient_temp': T_AMBIENT,
            'temperature_difference': T_CHAR,
            'inlet_lu': T_INLET_LU,
            'initial_lu': T_INITIAL_LU
        },
        'relaxation_times': {
            'water': TAU_THERMAL_WATER,
            'air': TAU_THERMAL_AIR,
            'coffee': TAU_THERMAL_COFFEE
        },
        'diffusivities': {
            'water_lu': ALPHA_WATER_LU,
            'air_lu': ALPHA_AIR_LU,
            'coffee_lu': ALPHA_COFFEE_LU
        },
        'stability': {
            'thermal_cfl': thermal_cfl,
            'rayleigh_number': RAYLEIGH_NUMBER,
            'biot_number': BIOT_NUMBER_CONVECTION
        },
        'coffee_bed': {
            'effective_conductivity': COFFEE_EFFECTIVE_CONDUCTIVITY,
            'effective_heat_capacity': COFFEE_EFFECTIVE_HEAT_CAPACITY,
            'effective_density': COFFEE_EFFECTIVE_DENSITY
        }
    }

# 模組導入時執行基本驗證
if __name__ == "__main__":
    print_thermal_diagnostics()
    validate_thermal_config()
else:
    # 簡化驗證
    thermal_cfl = max(ALPHA_WATER_LU, ALPHA_AIR_LU, ALPHA_COFFEE_LU) / CS2_THERMAL
    if thermal_cfl > MAX_CFL_THERMAL:
        print(f"⚠️  熱傳配置警告: 熱擴散CFL過高 ({thermal_cfl:.3f})")
    else:
        print(f"✅ 熱傳配置載入成功 (T_入水={T_INLET:.0f}°C, CFL_熱={thermal_cfl:.3f})")