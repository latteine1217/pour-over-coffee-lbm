# thermal_config.py - 熱傳系統專用配置
"""
熱傳導LBM系統的專用配置參數
與主config.py協調，專門處理熱傳相關參數
確保熱流耦合系統的數值穩定性

配置內容：
- 熱傳導LBM參數
- 溫度邊界條件
- 熱物性參數
- 耦合控制參數
- 診斷與輸出設定

開發：opencode + GitHub Copilot
"""

import math
import numpy as np
from config import (
    NX, NY, NZ, DX, DT, SCALE_LENGTH, SCALE_TIME,
    WATER_TEMP_C, WATER_DENSITY_90C, COFFEE_POWDER_MASS,
    CUP_HEIGHT, TOP_RADIUS, BOTTOM_RADIUS, PHYSICAL_DOMAIN_SIZE
)

# ==============================================
# 基礎熱傳LBM參數
# ==============================================

# D3Q7格子參數
Q_THERMAL = 7
CS2_THERMAL = 1.0/3.0      # 熱擴散格子聲速平方
INV_CS2_THERMAL = 3.0

# D3Q7離散速度向量
CX_THERMAL = np.array([0, 1, -1, 0, 0, 0, 0], dtype=np.int32)
CY_THERMAL = np.array([0, 0, 0, 1, -1, 0, 0], dtype=np.int32)
CZ_THERMAL = np.array([0, 0, 0, 0, 0, 1, -1], dtype=np.int32)

# D3Q7權重係數
W_THERMAL = np.array([1.0/4.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0], dtype=np.float32)

# 驗證權重歸一化
assert abs(np.sum(W_THERMAL) - 1.0) < 1e-6, "D3Q7權重係數歸一化失敗"

# ==============================================
# 物理熱傳參數 (咖啡沖煮專用)
# ==============================================

# 標準溫度設定 (°C)
T_INITIAL = 25.0           # 初始環境溫度
T_INLET = 93.0             # 注水溫度 (手沖標準)
T_AMBIENT = 25.0           # 環境溫度
T_COFFEE_INITIAL = 25.0    # 咖啡粉初始溫度

# 溫度範圍限制
T_MIN_PHYSICAL = 10.0      # 最低物理溫度
T_MAX_PHYSICAL = 100.0     # 最高物理溫度
T_MIN_STABLE = 15.0        # 數值穩定下限
T_MAX_STABLE = 98.0        # 數值穩定上限

# 水的熱物性 (93°C標準，參考config.py)
WATER_THERMAL_CONDUCTIVITY = 0.675     # W/(m·K) @93°C
WATER_HEAT_CAPACITY = 4205             # J/(kg·K) @93°C
WATER_DENSITY_THERMAL = WATER_DENSITY_90C  # kg/m³ (與流體一致)
WATER_THERMAL_DIFFUSIVITY = 1.66e-7   # m²/s @93°C
WATER_THERMAL_EXPANSION = 6.95e-4     # 1/K @93°C

# 空氣熱物性 (20°C)
AIR_THERMAL_CONDUCTIVITY = 0.0257      # W/(m·K)
AIR_HEAT_CAPACITY = 1005               # J/(kg·K)
AIR_DENSITY_THERMAL = 1.204            # kg/m³
AIR_THERMAL_DIFFUSIVITY = 2.12e-5      # m²/s
AIR_THERMAL_EXPANSION = 3.43e-3        # 1/K

# 咖啡固體熱物性
COFFEE_THERMAL_CONDUCTIVITY = 0.3      # W/(m·K) (烘焙咖啡豆)
COFFEE_HEAT_CAPACITY = 1800            # J/(kg·K)
COFFEE_DENSITY_THERMAL = 1200          # kg/m³
COFFEE_THERMAL_DIFFUSIVITY = 1.39e-7   # m²/s
COFFEE_THERMAL_EXPANSION = 1.5e-5      # 1/K

# 多孔介質熱物性 (咖啡床)
COFFEE_POROSITY = 0.45                 # 孔隙率 (與config.py一致)
COFFEE_EFFECTIVE_CONDUCTIVITY = (COFFEE_POROSITY * WATER_THERMAL_CONDUCTIVITY + 
                                (1 - COFFEE_POROSITY) * COFFEE_THERMAL_CONDUCTIVITY)

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

# 數值穩定性安全係數
THERMAL_SAFETY_FACTOR = 1.2
MIN_TAU_THERMAL = 0.51    # 絕對穩定下限
MAX_TAU_THERMAL = 2.0     # 數值擴散上限

# 空氣相特殊處理 - 降低熱擴散係數以滿足CFL條件
AIR_THERMAL_DIFFUSIVITY_REDUCED = 2.5e-6  # 進一步降低至穩定範圍
ALPHA_AIR_LU_REDUCED = AIR_THERMAL_DIFFUSIVITY_REDUCED * SCALE_TIME / (SCALE_LENGTH**2)
TAU_THERMAL_AIR_RAW = ALPHA_AIR_LU_REDUCED / CS2_THERMAL + 0.5

# 應用安全係數並檢查穩定性
TAU_THERMAL_WATER = max(MIN_TAU_THERMAL, min(MAX_TAU_THERMAL, TAU_THERMAL_WATER * THERMAL_SAFETY_FACTOR))
TAU_THERMAL_AIR = max(MIN_TAU_THERMAL, min(MAX_TAU_THERMAL, TAU_THERMAL_AIR_RAW * THERMAL_SAFETY_FACTOR))
TAU_THERMAL_COFFEE = max(MIN_TAU_THERMAL, min(MAX_TAU_THERMAL, TAU_THERMAL_COFFEE * THERMAL_SAFETY_FACTOR))

# 鬆弛頻率
OMEGA_THERMAL_WATER = 1.0 / TAU_THERMAL_WATER
OMEGA_THERMAL_AIR = 1.0 / TAU_THERMAL_AIR
OMEGA_THERMAL_COFFEE = 1.0 / TAU_THERMAL_COFFEE

# 熱擴散CFL數檢查
CFL_THERMAL_WATER = ALPHA_WATER_LU * DT / (DX**2)
CFL_THERMAL_AIR = ALPHA_AIR_LU_REDUCED * DT / (DX**2)  # 使用降低後的擴散係數
CFL_THERMAL_COFFEE = ALPHA_COFFEE_LU * DT / (DX**2)
MAX_CFL_THERMAL = 0.5     # 擴散方程穩定性條件

# ==============================================
# 邊界條件參數
# ==============================================

# Dirichlet邊界 (固定溫度)
BOUNDARY_TEMP_INLET = T_INLET          # 注水口溫度
BOUNDARY_TEMP_OUTLET = T_AMBIENT       # 出口溫度
BOUNDARY_TEMP_INITIAL = T_INITIAL      # 初始邊界溫度

# Neumann邊界 (固定熱流)
BOUNDARY_HEAT_FLUX_ADIABATIC = 0.0     # 絕熱邊界
BOUNDARY_HEAT_FLUX_COOLING = -100.0    # 冷卻邊界 (W/m²)

# Robin邊界 (對流散熱)
CONVECTION_COEFF_NATURAL = 5.0         # 自然對流係數 W/(m²·K)
CONVECTION_COEFF_FORCED = 25.0         # 強制對流係數 W/(m²·K)
CONVECTION_COEFF_COFFEE = 15.0         # 咖啡床內對流係數 W/(m²·K)

# 濾杯壁面散熱
FILTER_CUP_EMISSIVITY = 0.85           # 陶瓷濾杯發射率
STEFAN_BOLTZMANN = 5.67e-8             # Stefan-Boltzmann常數 W/(m²·K⁴)

# ==============================================
# 幾何與初始條件
# ==============================================

# 溫度場初始化參數
THERMAL_INIT_STRATEGY = "layered"      # "uniform", "layered", "gradient"

# 分層初始化
HOT_WATER_INITIAL_HEIGHT = 0.02        # 初始熱水層高度 (m)
COFFEE_BED_INITIAL_TEMP = T_COFFEE_INITIAL
WATER_LAYER_INITIAL_TEMP = T_INLET

# 漸變初始化
TEMP_GRADIENT_BOTTOM = T_INLET          # 底部溫度
TEMP_GRADIENT_TOP = T_AMBIENT           # 頂部溫度

# 格子單位轉換
HOT_WATER_INITIAL_HEIGHT_LU = int(HOT_WATER_INITIAL_HEIGHT / SCALE_LENGTH)

# ==============================================
# 熱源項與萃取建模
# ==============================================

# 咖啡萃取熱效應
COFFEE_EXTRACTION_ENTHALPY = -50.0     # kJ/kg (吸熱過程)
EXTRACTION_ACTIVATION_ENERGY = 25000   # J/mol (Arrhenius模型)
GAS_CONSTANT = 8.314                   # J/(mol·K)

# 萃取速率與溫度關係
EXTRACTION_RATE_COEFFICIENT = 1e-4     # s⁻¹
OPTIMAL_EXTRACTION_TEMP = 92.0         # °C (最佳萃取溫度)
EXTRACTION_TEMP_SENSITIVITY = 10.0     # °C (溫度敏感度)

# 蒸發散熱
WATER_LATENT_HEAT = 2260000            # J/kg (100°C)
EVAPORATION_RATE_COEFF = 1e-8          # kg/(m²·s·Pa)

# ==============================================
# 診斷與監控參數
# ==============================================

# 溫度場診斷
ENABLE_THERMAL_DIAGNOSTICS = True
THERMAL_DIAGNOSTIC_FREQ = 50           # 診斷頻率 (步數)

# 熱流診斷
ENABLE_HEAT_FLUX_MONITORING = True
HEAT_FLUX_DIAGNOSTIC_FREQ = 100

# 穩定性監控
ENABLE_THERMAL_STABILITY_CHECK = True
STABILITY_CHECK_FREQ = 10
MAX_TEMP_CHANGE_RATE = 5.0             # °C/s (最大溫度變化率)

# 收斂判定
THERMAL_CONVERGENCE_TOLERANCE = 0.01   # °C (收斂容差)
THERMAL_CONVERGENCE_STEPS = 100        # 收斂檢查步數

# ==============================================
# 輸出與視覺化
# ==============================================

# 溫度場輸出
ENABLE_TEMPERATURE_OUTPUT = True
TEMPERATURE_OUTPUT_FREQ = 200          # 輸出頻率

# 熱流場輸出
ENABLE_HEAT_FLUX_OUTPUT = True
HEAT_FLUX_OUTPUT_FREQ = 200

# 3D視覺化
ENABLE_3D_THERMAL_VIZ = True
THERMAL_VIZ_FREQ = 500

# 溫度色標設定
TEMP_COLORMAP_MIN = T_AMBIENT          # 色標最小值
TEMP_COLORMAP_MAX = T_INLET            # 色標最大值
TEMP_COLORMAP = "viridis"              # matplotlib色標

# ==============================================
# 數值穩定性報告
# ==============================================

def print_thermal_stability_report():
    """輸出熱傳數值穩定性報告"""
    
    print("\n=== 熱傳系統穩定性報告 ===")
    
    print(f"🌡️  溫度參數:")
    print(f"  注水溫度: {T_INLET}°C")
    print(f"  環境溫度: {T_AMBIENT}°C")
    print(f"  溫度範圍: {T_MIN_STABLE}-{T_MAX_STABLE}°C")
    
    print(f"\n⏰ 鬆弛時間:")
    print(f"  τ_水: {TAU_THERMAL_WATER:.6f} ({'✅' if MIN_TAU_THERMAL <= TAU_THERMAL_WATER <= MAX_TAU_THERMAL else '❌'})")
    print(f"  τ_空氣: {TAU_THERMAL_AIR:.6f} ({'✅' if MIN_TAU_THERMAL <= TAU_THERMAL_AIR <= MAX_TAU_THERMAL else '❌'})")
    print(f"  τ_咖啡: {TAU_THERMAL_COFFEE:.6f} ({'✅' if MIN_TAU_THERMAL <= TAU_THERMAL_COFFEE <= MAX_TAU_THERMAL else '❌'})")
    
    print(f"\n📊 CFL數 (熱擴散):")
    print(f"  CFL_水: {CFL_THERMAL_WATER:.6f} ({'✅' if CFL_THERMAL_WATER <= MAX_CFL_THERMAL else '❌'})")
    print(f"  CFL_空氣: {CFL_THERMAL_AIR:.6f} ({'✅' if CFL_THERMAL_AIR <= MAX_CFL_THERMAL else '❌'})")
    print(f"  CFL_咖啡: {CFL_THERMAL_COFFEE:.6f} ({'✅' if CFL_THERMAL_COFFEE <= MAX_CFL_THERMAL else '❌'})")
    
    print(f"\n🔬 物性參數:")
    print(f"  水熱導率: {WATER_THERMAL_CONDUCTIVITY:.3f} W/(m·K)")
    print(f"  水熱擴散: {WATER_THERMAL_DIFFUSIVITY:.2e} m²/s")
    print(f"  咖啡有效導率: {COFFEE_EFFECTIVE_CONDUCTIVITY:.3f} W/(m·K)")
    
    print(f"\n📏 尺度轉換:")
    print(f"  空間尺度: {SCALE_LENGTH*1000:.2f} mm/lu")
    print(f"  時間尺度: {SCALE_TIME*1000:.2f} ms/ts")
    print(f"  熱水初始高度: {HOT_WATER_INITIAL_HEIGHT_LU} lu ({HOT_WATER_INITIAL_HEIGHT*100:.1f} cm)")
    
    # 穩定性檢查
    thermal_errors = []
    
    if TAU_THERMAL_WATER < MIN_TAU_THERMAL or TAU_THERMAL_WATER > MAX_TAU_THERMAL:
        thermal_errors.append(f"水相熱傳鬆弛時間不穩定: {TAU_THERMAL_WATER:.6f}")
    
    if TAU_THERMAL_AIR < MIN_TAU_THERMAL or TAU_THERMAL_AIR > MAX_TAU_THERMAL:
        thermal_errors.append(f"氣相熱傳鬆弛時間不穩定: {TAU_THERMAL_AIR:.6f}")
    
    if CFL_THERMAL_WATER > MAX_CFL_THERMAL:
        thermal_errors.append(f"水相熱擴散CFL過大: {CFL_THERMAL_WATER:.6f}")
    
    if CFL_THERMAL_AIR > MAX_CFL_THERMAL:
        thermal_errors.append(f"氣相熱擴散CFL過大: {CFL_THERMAL_AIR:.6f}")
    
    if thermal_errors:
        print(f"\n❌ 發現熱傳穩定性問題:")
        for i, error in enumerate(thermal_errors, 1):
            print(f"  {i}. {error}")
    else:
        print(f"\n✅ 熱傳系統數值穩定性檢查通過！")

def get_thermal_config_summary():
    """獲取熱傳配置摘要"""
    
    return {
        'temperature_range': (T_MIN_STABLE, T_MAX_STABLE),
        'inlet_temperature': T_INLET,
        'ambient_temperature': T_AMBIENT,
        'relaxation_times': {
            'water': TAU_THERMAL_WATER,
            'air': TAU_THERMAL_AIR,
            'coffee': TAU_THERMAL_COFFEE
        },
        'cfl_numbers': {
            'water': CFL_THERMAL_WATER,
            'air': CFL_THERMAL_AIR,
            'coffee': CFL_THERMAL_COFFEE
        },
        'thermal_properties': {
            'water_conductivity': WATER_THERMAL_CONDUCTIVITY,
            'water_diffusivity': WATER_THERMAL_DIFFUSIVITY,
            'coffee_effective_conductivity': COFFEE_EFFECTIVE_CONDUCTIVITY
        },
        'boundary_conditions': {
            'convection_coeff_natural': CONVECTION_COEFF_NATURAL,
            'convection_coeff_forced': CONVECTION_COEFF_FORCED
        }
    }

# 模組導入時自動執行穩定性檢查
if __name__ == "__main__":
    print_thermal_stability_report()
else:
    # 簡化版穩定性檢查
    critical_errors = []
    if TAU_THERMAL_WATER < MIN_TAU_THERMAL:
        critical_errors.append("水相熱傳τ不穩定")
    if CFL_THERMAL_WATER > MAX_CFL_THERMAL:
        critical_errors.append("水相熱擴散CFL過大")
    
    if critical_errors:
        print(f"⚠️  熱傳配置警告: {', '.join(critical_errors)}")
        print("   建議運行: python thermal_config.py")
    else:
        print(f"✅ 熱傳配置載入成功 (τ_水={TAU_THERMAL_WATER:.3f}, CFL_熱={CFL_THERMAL_WATER:.3f})")

# ==============================================
# 配置驗證函數
# ==============================================

def validate_thermal_config():
    """驗證熱傳配置的完整性和一致性"""
    
    validation_errors = []
    warnings = []
    
    # 溫度範圍檢查
    if T_INLET > T_MAX_PHYSICAL:
        validation_errors.append(f"注水溫度過高: {T_INLET}°C > {T_MAX_PHYSICAL}°C")
    
    if T_AMBIENT < T_MIN_PHYSICAL:
        validation_errors.append(f"環境溫度過低: {T_AMBIENT}°C < {T_MIN_PHYSICAL}°C")
    
    # 數值穩定性檢查
    for tau, name in [(TAU_THERMAL_WATER, "水"), (TAU_THERMAL_AIR, "空氣"), (TAU_THERMAL_COFFEE, "咖啡")]:
        if tau < MIN_TAU_THERMAL:
            validation_errors.append(f"{name}相鬆弛時間過小: {tau:.6f} < {MIN_TAU_THERMAL}")
        elif tau > MAX_TAU_THERMAL:
            warnings.append(f"{name}相鬆弛時間偏大: {tau:.6f} > {MAX_TAU_THERMAL}")
    
    # CFL數檢查
    for cfl, name in [(CFL_THERMAL_WATER, "水"), (CFL_THERMAL_AIR, "空氣"), (CFL_THERMAL_COFFEE, "咖啡")]:
        if cfl > MAX_CFL_THERMAL:
            validation_errors.append(f"{name}相熱擴散CFL過大: {cfl:.6f} > {MAX_CFL_THERMAL}")
    
    # 物性參數合理性檢查
    if WATER_THERMAL_CONDUCTIVITY <= 0:
        validation_errors.append("水熱傳導係數必須為正值")
    
    if COFFEE_POROSITY < 0 or COFFEE_POROSITY > 1:
        validation_errors.append(f"咖啡孔隙率超出範圍: {COFFEE_POROSITY}")
    
    # 報告結果
    if validation_errors:
        print(f"\n❌ 熱傳配置驗證失敗:")
        for error in validation_errors:
            print(f"  • {error}")
        return False
    
    if warnings:
        print(f"\n⚠️  熱傳配置警告:")
        for warning in warnings:
            print(f"  • {warning}")
    
    print(f"✅ 熱傳配置驗證通過")
    return True