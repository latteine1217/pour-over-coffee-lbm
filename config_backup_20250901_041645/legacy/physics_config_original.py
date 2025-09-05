# physics_config.py - 物理常數與材料參數配置
"""
物理常數與材料參數配置
包含所有流體物性、幾何參數、無量綱數計算

與core_config.py分離，專注於物理模型參數
確保物理參數的一致性和可追溯性

開發：opencode + GitHub Copilot
"""

import math
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.core_config import (
    NX, NY, NZ, SCALE_LENGTH, SCALE_TIME, SCALE_VELOCITY,
    CS2, TAU_FLUID, TAU_AIR, CFL_NUMBER
)

# ==============================================
# 標準溫度與物理常數
# ==============================================

# 標準工作溫度 (°C)
WATER_TEMP_C = 90.0            # 90°C熱水 (標準手沖溫度)
AMBIENT_TEMP_C = 25.0          # 環境溫度
COFFEE_INITIAL_TEMP_C = 25.0   # 咖啡粉初始溫度

# 物理常數
GRAVITY_PHYS = 9.81            # m/s² (重力加速度)
STEFAN_BOLTZMANN = 5.67e-8     # W/(m²·K⁴) (Stefan-Boltzmann常數)

# ==============================================
# 流體物性參數 (90°C水 & 20°C空氣)
# ==============================================

# 90°C熱水物理性質 (標準參考)
WATER_DENSITY_90C = 965.3              # kg/m³
WATER_VISCOSITY_90C = 3.15e-7          # m²/s (運動黏滯度)
WATER_THERMAL_CONDUCTIVITY = 0.675     # W/(m·K)
WATER_HEAT_CAPACITY = 4205             # J/(kg·K)
WATER_THERMAL_DIFFUSIVITY = 1.66e-7    # m²/s
WATER_THERMAL_EXPANSION = 6.95e-4      # 1/K
WATER_SURFACE_TENSION = 0.0728         # N/m

# 20°C空氣物理性質
AIR_DENSITY_20C = 1.204                # kg/m³
AIR_VISCOSITY_20C = 1.516e-5           # m²/s
AIR_THERMAL_CONDUCTIVITY = 0.0257      # W/(m·K)
AIR_HEAT_CAPACITY = 1005               # J/(kg·K)
AIR_THERMAL_DIFFUSIVITY = 2.12e-5      # m²/s
AIR_THERMAL_EXPANSION = 3.43e-3        # 1/K

# 咖啡固體物理性質
COFFEE_BEAN_DENSITY = 1200             # kg/m³
COFFEE_THERMAL_CONDUCTIVITY = 0.3      # W/(m·K)
COFFEE_HEAT_CAPACITY = 1800            # J/(kg·K)
COFFEE_THERMAL_DIFFUSIVITY = 1.39e-7   # m²/s
COFFEE_THERMAL_EXPANSION = 1.5e-5      # 1/K

# ==============================================
# V60幾何參數 (真實規格)
# ==============================================

# V60濾杯尺寸 (Hario V60-02標準)
CUP_HEIGHT = 0.085                     # m (8.5 cm)
TOP_RADIUS = 0.058                     # m (11.6 cm直徑)
BOTTOM_RADIUS = 0.010                  # m (2 cm出水孔)

# 咖啡床參數
COFFEE_POWDER_MASS = 0.02              # 20g
COFFEE_POROSITY = 0.45                 # 孔隙率
COFFEE_PARTICLE_DIAMETER = 6.5e-4      # 0.65mm
COFFEE_PARTICLE_RADIUS = COFFEE_PARTICLE_DIAMETER / 2

# 注水參數
POUR_RATE_ML_S = 4.0                   # 4 ml/s
POUR_RATE_M3_S = POUR_RATE_ML_S * 1e-6
TOTAL_WATER_ML = 320                   # 320ml
BREWING_TIME_SECONDS = 140             # 2:20

# 注水幾何
POUR_HEIGHT_CM = 12.5                  # 注水高度
INLET_DIAMETER_M = 0.005               # 入水直徑 (0.5 cm)
NOZZLE_DIAMETER_M = 0.005              # 噴嘴直徑上限

# ==============================================
# 尺度轉換與無量綱化
# ==============================================

# 特徵尺度選擇
L_CHAR = CUP_HEIGHT                    # 特徵長度 (V60高度)
U_CHAR = 0.02                          # 特徵速度 (2 cm/s)
T_CHAR = L_CHAR / U_CHAR               # 特徵時間
RHO_CHAR = WATER_DENSITY_90C           # 特徵密度
NU_CHAR = WATER_VISCOSITY_90C          # 特徵黏滯度

# 密度比 (相對於水)
RHO_WATER = 1.0                        # 參考密度
RHO_AIR = AIR_DENSITY_20C / WATER_DENSITY_90C    # 真實密度比 (~0.00125)

# 格子單位下的物性參數
NU_WATER_LU = WATER_VISCOSITY_90C * SCALE_TIME / (SCALE_LENGTH**2)
NU_AIR_LU = AIR_VISCOSITY_20C * SCALE_TIME / (SCALE_LENGTH**2)

# 驗證鬆弛時間一致性 (與core_config一致)
TAU_WATER_CALCULATED = NU_WATER_LU / CS2 + 0.5
TAU_AIR_CALCULATED = NU_AIR_LU / CS2 + 0.5

# 使用core_config的安全值，但驗證一致性
if abs(TAU_WATER_CALCULATED - TAU_FLUID) > 0.1:
    print(f"⚠️  水相鬆弛時間不一致: 計算值{TAU_WATER_CALCULATED:.3f} vs 核心值{TAU_FLUID:.3f}")

if abs(TAU_AIR_CALCULATED - TAU_AIR) > 0.1:
    print(f"⚠️  氣相鬆弛時間不一致: 計算值{TAU_AIR_CALCULATED:.3f} vs 核心值{TAU_AIR:.3f}")

# ==============================================
# 重力與表面張力
# ==============================================

# 重力轉換
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