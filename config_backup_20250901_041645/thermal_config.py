"""
thermal_config.py - 向後相容層（轉發至統一的 config.thermal）

用途：
- 保持舊程式碼 `from config.thermal_config import ...` 可用。
- 所有定義均轉發自 `config.thermal`（Phase 1 統一熱傳配置）。
- 並提供歷史名稱別名：`get_thermal_config_summary` 對應 `get_thermal_summary`。

開發：opencode + GitHub Copilot
"""

# 轉發所有熱傳配置與函數
from .thermal import *  # noqa: F401,F403

# 單次告警，提示使用新入口
_printed_notice = False
def _print_deprecation_once():
    global _printed_notice
    if not _printed_notice:
        try:
            print("⚠️  Deprecation: 請改用 `from config.thermal import ...`，thermal_config 已轉為相容層。")
        except Exception:
            pass
        _printed_notice = True

_print_deprecation_once()

# 歷史名稱別名：
try:
    get_thermal_config_summary  # type: ignore
except NameError:
    try:
        from .thermal import get_thermal_summary as get_thermal_config_summary  # type: ignore
    except Exception:
        pass
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
