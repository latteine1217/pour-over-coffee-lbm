"""
physics_config.py - 向後相容層（轉發至統一的 config.physics）

用途：
- 保持舊程式碼 `from config.physics_config import ...` 可用。
- 所有定義均轉發自 `config.physics`（統一物理參數的單一來源）。
- 提供與 thermal_config 同風格的棄用提醒。

開發：opencode + GitHub Copilot
"""

# 轉發所有物理參數與函數
from .physics import *  # noqa: F401,F403

# 單次告警，提示使用新入口
_printed_notice = False

def _print_deprecation_once():
    global _printed_notice
    if not _printed_notice:
        try:
            print("⚠️  Deprecation: 請改用 `from config.physics import ...`，physics_config 已轉為相容層。")
        except Exception:
            pass
        _printed_notice = True

_print_deprecation_once()
GRAVITY_LU_FULL = GRAVITY_PHYS * (SCALE_TIME**2) / SCALE_LENGTH
GRAVITY_STRENGTH_FACTOR = 0.5          # 50%重力強度 (穩定性與物理效果平衡)
GRAVITY_LU = GRAVITY_LU_FULL * GRAVITY_STRENGTH_FACTOR

# 表面張力 (基於Weber數)
WEBER_NUMBER = 1.0                     # 目標Weber數
SURFACE_TENSION_LU = (RHO_WATER * (U_CHAR * SCALE_TIME / SCALE_LENGTH)**2 * SCALE_LENGTH) / WEBER_NUMBER

# ==============================================
# 無量綱數計算
# ==============================================

# Reynolds數
RE_CHAR = U_CHAR * L_CHAR / NU_CHAR                # 物理特徵Re
RE_LATTICE = SCALE_VELOCITY * NZ / NU_WATER_LU     # 格子Re

# Froude數 (重力與慣性力比)
FR_CHAR = U_CHAR / np.sqrt(GRAVITY_PHYS * L_CHAR)

# Mach數 (可壓縮性檢查)
MACH_NUMBER = SCALE_VELOCITY / np.sqrt(CS2)

# Capillary數 (表面張力效應)
CA_NUMBER = (WATER_VISCOSITY_90C * U_CHAR) / WATER_SURFACE_TENSION

# Bond數 (重力與表面張力比)
BOND_NUMBER = (WATER_DENSITY_90C * GRAVITY_PHYS * L_CHAR**2) / WATER_SURFACE_TENSION

# Péclet數 (對流與擴散比)
PE_THERMAL = (U_CHAR * L_CHAR) / WATER_THERMAL_DIFFUSIVITY

# ==============================================
# 咖啡床幾何計算
# ==============================================

def solve_coffee_bed_height():
    """基於錐台幾何精確計算咖啡床高度"""
    
    # V60內部錐台體積
    v60_volume = (math.pi * CUP_HEIGHT / 3) * (TOP_RADIUS**2 + TOP_RADIUS * BOTTOM_RADIUS + BOTTOM_RADIUS**2)
    coffee_fill_ratio = 0.15               # 填充V60的15%體積
    coffee_bed_volume = v60_volume * coffee_fill_ratio
    
    # 錐台高度求解
    cone_slope = (TOP_RADIUS - BOTTOM_RADIUS) / CUP_HEIGHT
    
    # 二分法求解
    h_min, h_max = 0.001, CUP_HEIGHT * 0.6
    for _ in range(100):
        h_test = (h_min + h_max) / 2
        r_top = BOTTOM_RADIUS + h_test * cone_slope
        volume_test = (math.pi * h_test / 3) * (BOTTOM_RADIUS**2 + BOTTOM_RADIUS * r_top + r_top**2)
        
        if abs(volume_test - coffee_bed_volume) < 1e-8:
            return h_test, r_top
        elif volume_test < coffee_bed_volume:
            h_min = h_test
        else:
            h_max = h_test
    
    return h_max, BOTTOM_RADIUS + h_max * cone_slope

# 計算咖啡床幾何
COFFEE_BED_HEIGHT_PHYS, COFFEE_BED_TOP_RADIUS = solve_coffee_bed_height()
COFFEE_BED_HEIGHT_LU = int(COFFEE_BED_HEIGHT_PHYS / SCALE_LENGTH)

# 咖啡床物性
COFFEE_SOLID_VOLUME = COFFEE_POWDER_MASS / COFFEE_BEAN_DENSITY
COFFEE_BED_VOLUME_PHYS = (math.pi * COFFEE_BED_HEIGHT_PHYS / 3) * (
    BOTTOM_RADIUS**2 + BOTTOM_RADIUS * COFFEE_BED_TOP_RADIUS + COFFEE_BED_TOP_RADIUS**2
)
ACTUAL_POROSITY = 1 - (COFFEE_SOLID_VOLUME / COFFEE_BED_VOLUME_PHYS)

# ==============================================
# 注水參數計算
# ==============================================

# 重力修正係數
GRAVITY_CORRECTION = 0.05              # 5%重力修正

def compute_inlet_velocity():
    """計算入水速度"""
    
    # 基於流量和截面積的基礎速度
    inlet_area = math.pi * (INLET_DIAMETER_M / 2.0)**2
    inlet_velocity_base = POUR_RATE_M3_S / inlet_area
    
    # 重力修正
    inlet_velocity_phys = inlet_velocity_base * (1.0 + GRAVITY_CORRECTION)
    
    # 轉換為格子單位
    inlet_velocity_lu = inlet_velocity_phys * SCALE_TIME / SCALE_LENGTH
    
    # 安全限制 (避免高Mach數)
    inlet_velocity = min(0.05, inlet_velocity_lu)
    
    return inlet_velocity

INLET_VELOCITY = compute_inlet_velocity()

# ==============================================
# 模擬控制參數
# ==============================================

MAX_STEPS = int(BREWING_TIME_SECONDS / SCALE_TIME)
POURING_STEPS = int(80 / SCALE_TIME)      # 80秒注水時間
OUTPUT_FREQ = max(100, MAX_STEPS // 1000)

# ==============================================
# 多相流參數
# ==============================================

PHASE_WATER = 1.0
PHASE_AIR = 0.0
INTERFACE_THICKNESS = 1.5              # 格子單位
CAHN_HILLIARD_MOBILITY = 0.01

# ==============================================
# 診斷與驗證函數
# ==============================================

def print_physics_diagnostics():
    """輸出物理參數診斷信息"""
    
    print("\n=== 物理參數診斷 ===")
    
    print(f"📏 尺度轉換:")
    print(f"  長度: {SCALE_LENGTH*1000:.2f} mm/lu")
    print(f"  時間: {SCALE_TIME*1000:.2f} ms/ts") 
    print(f"  速度: {SCALE_VELOCITY:.3f} lu/ts")
    
    print(f"\n🌊 流動特性:")
    print(f"  物理Re: {RE_CHAR:.1f}")
    print(f"  格子Re: {RE_LATTICE:.1f}")
    print(f"  Froude數: {FR_CHAR:.3f}")
    print(f"  Mach數: {MACH_NUMBER:.3f}")
    
    print(f"\n📊 無量綱數:")
    print(f"  Capillary數: {CA_NUMBER:.2e}")
    print(f"  Bond數: {BOND_NUMBER:.1f}")
    print(f"  Péclet數: {PE_THERMAL:.1f}")
    print(f"  Weber數: {WEBER_NUMBER:.1f}")
    
    print(f"\n☕ 咖啡參數:")
    print(f"  咖啡粉: {COFFEE_POWDER_MASS*1000:.0f}g")
    print(f"  顆粒直徑: {COFFEE_PARTICLE_DIAMETER*1000:.2f}mm")
    print(f"  咖啡床高度: {COFFEE_BED_HEIGHT_PHYS*100:.1f}cm")
    print(f"  實際孔隙率: {ACTUAL_POROSITY:.3f}")
    
    print(f"\n💧 注水參數:")
    print(f"  注水速度: {POUR_RATE_ML_S:.1f} ml/s")
    print(f"  入水速度: {INLET_VELOCITY:.4f} lu/ts")
    print(f"  總注水量: {TOTAL_WATER_ML:.0f}ml")

def validate_physics_parameters():
    """驗證物理參數的合理性"""
    
    errors = []
    warnings = []
    
    # 無量綱數檢查
    if MACH_NUMBER > 0.3:
        errors.append(f"Mach數過高: {MACH_NUMBER:.3f} > 0.3")
    elif MACH_NUMBER > 0.1:
        warnings.append(f"Mach數建議降低: {MACH_NUMBER:.3f} > 0.1")
    
    # 咖啡床合理性
    max_coffee_height = CUP_HEIGHT * 2/3
    if COFFEE_BED_HEIGHT_PHYS > max_coffee_height:
        warnings.append(f"咖啡床高度偏高: {COFFEE_BED_HEIGHT_PHYS*100:.1f}cm")
    
    # 孔隙率檢查
    if ACTUAL_POROSITY < 0.3 or ACTUAL_POROSITY > 0.7:
        warnings.append(f"咖啡床孔隙率異常: {ACTUAL_POROSITY:.3f}")
    
    # 注水速度檢查
    if INLET_VELOCITY > 0.1:
        warnings.append(f"注水速度偏高: {INLET_VELOCITY:.4f} > 0.1 lu/ts")
    
    # 密度比檢查
    if RHO_AIR > 0.01:
        warnings.append(f"空氣密度比偏高: {RHO_AIR:.6f}")
    
    # 報告結果
    if errors:
        print(f"\n❌ 物理參數錯誤:")
        for error in errors:
            print(f"  • {error}")
        return False
    
    if warnings:
        print(f"\n⚠️  物理參數警告:")
        for warning in warnings:
            print(f"  • {warning}")
    
    print(f"✅ 物理參數驗證通過")
    return True

def get_physics_summary():
    """獲取物理參數摘要"""
    
    return {
        'dimensionless_numbers': {
            'reynolds_physical': RE_CHAR,
            'reynolds_lattice': RE_LATTICE,
            'froude': FR_CHAR,
            'mach': MACH_NUMBER,
            'capillary': CA_NUMBER,
            'bond': BOND_NUMBER,
            'peclet': PE_THERMAL,
            'weber': WEBER_NUMBER
        },
        'fluid_properties': {
            'water_density': WATER_DENSITY_90C,
            'water_viscosity': WATER_VISCOSITY_90C,
            'air_density': AIR_DENSITY_20C,
            'air_viscosity': AIR_VISCOSITY_20C,
            'density_ratio': RHO_AIR
        },
        'geometry': {
            'cup_height': CUP_HEIGHT,
            'top_radius': TOP_RADIUS,
            'bottom_radius': BOTTOM_RADIUS,
            'coffee_bed_height': COFFEE_BED_HEIGHT_PHYS,
            'coffee_porosity': ACTUAL_POROSITY
        },
        'pouring': {
            'pour_rate': POUR_RATE_ML_S,
            'inlet_velocity': INLET_VELOCITY,
            'total_water': TOTAL_WATER_ML,
            'brewing_time': BREWING_TIME_SECONDS
        }
    }

# 模組導入時執行基本驗證
if __name__ == "__main__":
    print_physics_diagnostics()
    validate_physics_parameters()
else:
    # 簡化驗證
    if MACH_NUMBER > 0.3:
        print(f"⚠️  物理配置警告: Mach數過高 ({MACH_NUMBER:.3f})")
    else:
        print(f"✅ 物理配置載入成功 (Re={RE_CHAR:.0f}, Ma={MACH_NUMBER:.3f})")