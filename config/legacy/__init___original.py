# config/__init__.py - 統一配置匯出介面
"""
Pour-Over CFD 統一配置系統
分層架構設計，確保參數一致性和可維護性

配置層級:
1. core_config.py     - 核心LBM參數 (絕對不可修改)
2. physics_config.py  - 物理常數與材料參數  
3. thermal_config.py  - 熱傳系統專用參數
4. config.py         - 歷史兼容性 (逐步棄用)

統一匯出所有必要參數，確保向後兼容性
開發：opencode + GitHub Copilot
"""

# 核心LBM參數 (最高優先級)
from .core_config import (
    # 網格參數
    NX, NY, NZ, DX, DT,
    
    # LBM理論常數
    Q_3D, CS2, CS4, INV_CS2,
    CX_3D, CY_3D, CZ_3D, WEIGHTS_3D,
    
    # 穩定性關鍵參數
    CFL_NUMBER, MAX_VELOCITY_LU, SCALE_VELOCITY,
    MIN_TAU_STABLE, MAX_TAU_STABLE, TAU_FLUID, TAU_AIR,
    TIME_SCALE_OPTIMIZATION_FACTOR,
    
    # 尺度轉換
    PHYSICAL_DOMAIN_SIZE, SCALE_LENGTH, SCALE_TIME,
    GRID_SIZE_CM,
    
    # LES參數
    SMAGORINSKY_CONSTANT, LES_FILTER_WIDTH, ENABLE_LES, LES_REYNOLDS_THRESHOLD,
    
    # 熱傳LBM參數
    Q_THERMAL, CS2_THERMAL, INV_CS2_THERMAL,
    CX_THERMAL, CY_THERMAL, CZ_THERMAL, W_THERMAL,
    MIN_TAU_THERMAL, MAX_TAU_THERMAL, MAX_CFL_THERMAL,
    
    # 驗證函數
    validate_core_parameters
)

# 嘗試導入物理參數 (需要核心參數)
try:
    # 手動添加路徑以避免循環導入
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # 現在可以導入
    from . import physics_config as _physics_config
    
    # 物理常數與材料參數
    WATER_TEMP_C = _physics_config.WATER_TEMP_C
    AMBIENT_TEMP_C = _physics_config.AMBIENT_TEMP_C
    COFFEE_INITIAL_TEMP_C = _physics_config.COFFEE_INITIAL_TEMP_C
    
    # 流體物性
    WATER_DENSITY_90C = _physics_config.WATER_DENSITY_90C
    WATER_VISCOSITY_90C = _physics_config.WATER_VISCOSITY_90C
    WATER_THERMAL_CONDUCTIVITY = _physics_config.WATER_THERMAL_CONDUCTIVITY
    WATER_HEAT_CAPACITY = _physics_config.WATER_HEAT_CAPACITY
    WATER_THERMAL_DIFFUSIVITY = _physics_config.WATER_THERMAL_DIFFUSIVITY
    WATER_SURFACE_TENSION = _physics_config.WATER_SURFACE_TENSION
    
    AIR_DENSITY_20C = _physics_config.AIR_DENSITY_20C
    AIR_VISCOSITY_20C = _physics_config.AIR_VISCOSITY_20C
    
    COFFEE_BEAN_DENSITY = _physics_config.COFFEE_BEAN_DENSITY
    COFFEE_PARTICLE_DIAMETER = _physics_config.COFFEE_PARTICLE_DIAMETER
    
    # V60幾何
    CUP_HEIGHT = _physics_config.CUP_HEIGHT
    TOP_RADIUS = _physics_config.TOP_RADIUS
    BOTTOM_RADIUS = _physics_config.BOTTOM_RADIUS
    
    # 咖啡參數
    COFFEE_POWDER_MASS = _physics_config.COFFEE_POWDER_MASS
    COFFEE_POROSITY = _physics_config.COFFEE_POROSITY
    COFFEE_BED_HEIGHT_PHYS = _physics_config.COFFEE_BED_HEIGHT_PHYS
    COFFEE_BED_HEIGHT_LU = _physics_config.COFFEE_BED_HEIGHT_LU
    
    # 注水參數
    POUR_RATE_ML_S = _physics_config.POUR_RATE_ML_S
    INLET_VELOCITY = _physics_config.INLET_VELOCITY
    TOTAL_WATER_ML = _physics_config.TOTAL_WATER_ML
    BREWING_TIME_SECONDS = _physics_config.BREWING_TIME_SECONDS
    
    # 密度比
    RHO_WATER = _physics_config.RHO_WATER
    RHO_AIR = _physics_config.RHO_AIR
    
    # 重力與表面張力
    GRAVITY_LU = _physics_config.GRAVITY_LU
    SURFACE_TENSION_LU = _physics_config.SURFACE_TENSION_LU
    
    # 無量綱數
    RE_CHAR = _physics_config.RE_CHAR
    RE_LATTICE = _physics_config.RE_LATTICE
    FR_CHAR = _physics_config.FR_CHAR
    MACH_NUMBER = _physics_config.MACH_NUMBER
    
    # 模擬控制
    MAX_STEPS = _physics_config.MAX_STEPS
    POURING_STEPS = _physics_config.POURING_STEPS
    OUTPUT_FREQ = _physics_config.OUTPUT_FREQ
    
    # 多相流
    PHASE_WATER = _physics_config.PHASE_WATER
    PHASE_AIR = _physics_config.PHASE_AIR
    INTERFACE_THICKNESS = _physics_config.INTERFACE_THICKNESS
    
    # 物理參數驗證函數
    print_physics_diagnostics = _physics_config.print_physics_diagnostics
    validate_physics_parameters = _physics_config.validate_physics_parameters
    get_physics_summary = _physics_config.get_physics_summary
    
    # 標記物理配置載入成功
    _PHYSICS_CONFIG_LOADED = True
    
except ImportError as e:
    print(f"⚠️  物理配置載入失敗: {e}")
    print("   使用核心參數和默認值")
    _PHYSICS_CONFIG_LOADED = False
    
    # 使用默認值確保系統可運行
    WATER_TEMP_C = 90.0
    RHO_WATER = 1.0
    RHO_AIR = 0.00125

# 嘗試導入熱傳配置
try:
    from .thermal_config import (
        T_INITIAL, T_INLET, T_AMBIENT,
        TAU_THERMAL_WATER, TAU_THERMAL_AIR, TAU_THERMAL_COFFEE,
        ENABLE_THERMAL_DIAGNOSTICS,
        print_thermal_stability_report,
        validate_thermal_config,
        get_thermal_config_summary
    )
    _THERMAL_CONFIG_LOADED = True
    
except ImportError as e:
    print(f"⚠️  熱傳配置載入失敗: {e}")
    _THERMAL_CONFIG_LOADED = False

# ==============================================
# 統一配置驗證與診斷
# ==============================================

def validate_unified_config():
    """統一配置驗證"""
    
    print("\n🔍 統一配置系統驗證...")
    
    errors = []
    warnings = []
    
    # 核心參數驗證
    try:
        validate_core_parameters()
        print("   ✅ 核心LBM參數驗證通過")
    except Exception as e:
        errors.append(f"核心參數驗證失敗: {e}")
    
    # 物理參數驗證
    if _PHYSICS_CONFIG_LOADED:
        try:
            validate_physics_parameters()
            print("   ✅ 物理參數驗證通過")
        except Exception as e:
            errors.append(f"物理參數驗證失敗: {e}")
    else:
        warnings.append("物理配置未載入")
    
    # 熱傳參數驗證
    if _THERMAL_CONFIG_LOADED:
        try:
            validate_thermal_config()
            print("   ✅ 熱傳參數驗證通過")
        except Exception as e:
            errors.append(f"熱傳參數驗證失敗: {e}")
    else:
        warnings.append("熱傳配置未載入")
    
    # 跨模組一致性檢查
    if _PHYSICS_CONFIG_LOADED:
        # 溫度一致性檢查
        if abs(WATER_TEMP_C - 90.0) > 1.0:
            warnings.append(f"水溫度偏離標準值: {WATER_TEMP_C}°C vs 90°C")
        
        # Mach數檢查
        if MACH_NUMBER > 0.3:
            errors.append(f"Mach數過高: {MACH_NUMBER}")
    
    # 報告結果
    if errors:
        print(f"\n❌ 統一配置驗證失敗:")
        for error in errors:
            print(f"  • {error}")
        return False
    
    if warnings:
        print(f"\n⚠️  統一配置警告:")
        for warning in warnings:
            print(f"  • {warning}")
    
    print("   ✅ 統一配置系統驗證通過")
    return True

def print_unified_config_summary():
    """輸出統一配置摘要"""
    
    print("\n=== Pour-Over CFD 統一配置摘要 ===")
    
    print(f"📋 系統狀態:")
    print(f"  核心配置: ✅ 已載入")
    print(f"  物理配置: {'✅ 已載入' if _PHYSICS_CONFIG_LOADED else '❌ 未載入'}")
    print(f"  熱傳配置: {'✅ 已載入' if _THERMAL_CONFIG_LOADED else '❌ 未載入'}")
    
    print(f"\n🏗️ 核心LBM參數:")
    print(f"  網格: {NX}×{NY}×{NZ}")
    print(f"  CFL數: {CFL_NUMBER}")
    print(f"  尺度: {SCALE_LENGTH*1000:.2f} mm/lu, {SCALE_TIME*1000:.1f} ms/ts")
    
    if _PHYSICS_CONFIG_LOADED:
        print(f"\n🌊 物理參數:")
        print(f"  Reynolds數: {RE_CHAR:.0f} (物理), {RE_LATTICE:.0f} (格子)")
        print(f"  Mach數: {MACH_NUMBER:.3f}")
        print(f"  咖啡粉: {COFFEE_POWDER_MASS*1000:.0f}g")
        print(f"  注水速度: {POUR_RATE_ML_S:.1f} ml/s")
    
    if _THERMAL_CONFIG_LOADED:
        print(f"\n🌡️ 熱傳參數:")
        print(f"  注水溫度: {T_INLET}°C")
        print(f"  環境溫度: {T_AMBIENT}°C")

def get_config_status():
    """獲取配置載入狀態"""
    
    return {
        'core_loaded': True,
        'physics_loaded': _PHYSICS_CONFIG_LOADED,
        'thermal_loaded': _THERMAL_CONFIG_LOADED,
        'unified_validation': validate_unified_config()
    }

# ==============================================
# 向後兼容性維護
# ==============================================

# 舊config.py的重要參數別名
SURFACE_TENSION_PHYS = WATER_SURFACE_TENSION if _PHYSICS_CONFIG_LOADED else 0.0728
CAHN_HILLIARD_MOBILITY = 0.01
DP = COFFEE_PARTICLE_DIAMETER if _PHYSICS_CONFIG_LOADED else 6.5e-4

# ==============================================
# 模組初始化
# ==============================================

# 在模組導入時執行基本驗證
if __name__ != "__main__":
    try:
        # 簡化驗證避免循環導入
        print(f"📦 Pour-Over CFD 配置系統載入")
        print(f"   核心參數: ✅")
        print(f"   物理參數: {'✅' if _PHYSICS_CONFIG_LOADED else '⚠️'}")
        print(f"   熱傳參數: {'✅' if _THERMAL_CONFIG_LOADED else '⚠️'}")
        
        if not _PHYSICS_CONFIG_LOADED:
            print("   建議檢查physics_config.py導入問題")
            
    except Exception as e:
        print(f"⚠️  配置初始化警告: {e}")

if __name__ == "__main__":
    # 完整驗證和診斷
    validate_unified_config()
    print_unified_config_summary()