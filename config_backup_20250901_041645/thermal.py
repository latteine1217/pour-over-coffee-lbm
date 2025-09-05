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

# ==============================================
# 安全的熱擴散係數計算 - Phase 1 穩定性修正
# ==============================================

# 熱擴散穩定性限制
MAX_ALPHA_LU_SAFE = MAX_CFL_THERMAL * CS2_THERMAL * 0.8  # 20%安全裕度
MIN_ALPHA_LU_SAFE = 1e-6  # 最小有效值

def compute_safe_thermal_diffusivity(physical_diffusivity, name):
    """計算安全的熱擴散係數，確保CFL穩定性"""
    
    # 物理計算值
    alpha_physical = physical_diffusivity * SCALE_TIME / (SCALE_LENGTH**2)
    
    # 應用安全限制
    alpha_safe = max(MIN_ALPHA_LU_SAFE, min(MAX_ALPHA_LU_SAFE, alpha_physical))
    
    # 計算對應的CFL數
    thermal_cfl = alpha_safe / CS2_THERMAL
    
    # 記錄修正信息
    if abs(alpha_physical - alpha_safe) > 1e-8:
        print(f"🔧 {name}熱擴散修正: 物理值{alpha_physical:.2e} → 穩定值{alpha_safe:.2e} (CFL: {thermal_cfl:.3f})")
    
    return alpha_safe

# 安全的熱擴散係數 (確保CFL < 0.5)
ALPHA_WATER_LU = compute_safe_thermal_diffusivity(WATER_THERMAL_DIFFUSIVITY, "水相")
ALPHA_AIR_LU = compute_safe_thermal_diffusivity(AIR_THERMAL_DIFFUSIVITY, "氣相")
ALPHA_COFFEE_LU = compute_safe_thermal_diffusivity(COFFEE_THERMAL_DIFFUSIVITY, "咖啡相")

# 熱傳導鬆弛時間 τ = α/(c_s²) + 0.5 (現在保證 τ < 2.0)
TAU_THERMAL_WATER = ALPHA_WATER_LU / CS2_THERMAL + 0.5
TAU_THERMAL_AIR = ALPHA_AIR_LU / CS2_THERMAL + 0.5
TAU_THERMAL_COFFEE = ALPHA_COFFEE_LU / CS2_THERMAL + 0.5

# 更嚴格的安全範圍檢查 - Phase 1 穩定性修正
def validate_tau_thermal(tau, name):
    """驗證熱傳鬆弛時間安全範圍 - 強制穩定性限制"""
    
    # 更嚴格的穩定性上限 (原本2.0 → 1.8)
    STRICT_MAX_TAU = 1.8
    
    if tau < MIN_TAU_THERMAL:
        print(f"🔧 {name} 鬆弛時間修正: {tau:.3f} → {MIN_TAU_THERMAL:.3f} (低於下限)")
        return MIN_TAU_THERMAL
    elif tau > STRICT_MAX_TAU:
        print(f"🔧 {name} 鬆弛時間修正: {tau:.3f} → {STRICT_MAX_TAU:.3f} (超出穩定上限)")
        return STRICT_MAX_TAU
    else:
        print(f"✅ {name} 鬆弛時間安全: {tau:.3f}")
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
    """驗證熱傳配置的合理性 - Phase 1 嚴格穩定性檢查"""
    
    errors = []
    warnings = []
    stability_fixes = []
    
    # 溫度範圍檢查
    if T_INLET < T_MIN_STABLE or T_INLET > T_MAX_STABLE:
        warnings.append(f"入水溫度超出穩定範圍: {T_INLET}°C")
    
    # 更嚴格的鬆弛時間檢查 (1.8 上限)
    STRICT_MAX_TAU = 1.8
    for tau, name in [(TAU_THERMAL_WATER, "水相"), 
                      (TAU_THERMAL_AIR, "氣相"), 
                      (TAU_THERMAL_COFFEE, "咖啡相")]:
        if tau < MIN_TAU_THERMAL:
            errors.append(f"{name}熱傳鬆弛時間過低: {tau:.3f}")
        elif tau > STRICT_MAX_TAU:
            stability_fixes.append(f"{name}鬆弛時間已自動修正: {tau:.3f} → ≤{STRICT_MAX_TAU}")
    
    # 熱擴散穩定性檢查 (更嚴格)
    thermal_cfl = max(ALPHA_WATER_LU, ALPHA_AIR_LU, ALPHA_COFFEE_LU) / CS2_THERMAL
    if thermal_cfl > MAX_CFL_THERMAL:
        stability_fixes.append(f"熱擴散CFL已自動修正: {thermal_cfl:.3f} → ≤{MAX_CFL_THERMAL}")
    
    # 數值穩定性總體評估
    stability_score = 100
    if thermal_cfl > 0.3:
        stability_score -= 20
    if any(tau > 1.5 for tau in [TAU_THERMAL_WATER, TAU_THERMAL_AIR, TAU_THERMAL_COFFEE]):
        stability_score -= 15
    
    # Rayleigh數檢查
    if RAYLEIGH_NUMBER > 1e6:
        warnings.append(f"Rayleigh數過高，可能出現湍流對流: {RAYLEIGH_NUMBER:.1e}")
    
    # 報告結果
    print(f"\n🛡️  Phase 1 熱傳穩定性檢查:")
    print(f"   穩定性評分: {stability_score}/100")
    
    if errors:
        print(f"\n❌ 熱傳配置錯誤:")
        for error in errors:
            print(f"   • {error}")
        return False
    
    if stability_fixes:
        print(f"\n🔧 自動穩定性修正:")
        for fix in stability_fixes:
            print(f"   • {fix}")
    
    if warnings:
        print(f"\n⚠️  熱傳配置警告:")
        for warning in warnings:
            print(f"   • {warning}")
    
    if stability_score >= 85:
        print(f"✅ 熱傳配置穩定性驗證通過 (評分: {stability_score}/100)")
    else:
        print(f"⚠️  熱傳配置穩定性需要關注 (評分: {stability_score}/100)")
    
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