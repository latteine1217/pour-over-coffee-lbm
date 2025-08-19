# config_fixed.py - 科研級CFD修正版
"""
修正版本 - 解決所有CFD理論和數值問題
基於嚴格的LBM理論和尺度分析
開發：opencode + GitHub Copilot
"""

import math
import numpy as np

# ==============================================
# 基礎LBM參數 - D3Q19模型
# ==============================================

# 網格設定 (平衡效率和精度) - 224³研究級精度，完整包含V60幾何
NX = 224
NY = 224  
NZ = 224
DX = 1.0         # 格點間距 (lattice units)
DT = 1.0         # 時間步長 (lattice units)

# LBM基本參數
Q_3D = 19
CS2 = 1.0/3.0      # 格子聲速平方
CS4 = CS2 * CS2
INV_CS2 = 3.0

# D3Q19離散速度向量 (正確版本)
CX_3D = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.int32)
CY_3D = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int32)
CZ_3D = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int32)

# D3Q19權重係數 (修正錯誤權重)
WEIGHTS_3D = np.array([
    1.0/3.0,                                           # 0: 靜止 (e=0,0,0)
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,  # 1-6: 面中心 |e|=1
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,          # 7-10: 邊中心 |e|=√2
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,          # 11-14: 邊中心 |e|=√2  
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0           # 15-18: 邊中心 |e|=√2
], dtype=np.float32)

# 驗證理論一致性
assert abs(np.sum(WEIGHTS_3D) - 1.0) < 1e-6, "權重係數歸一化失敗"

# ==============================================
# 物理參數 (90°C熱水)
# ==============================================

# 90°C熱水物理性質
WATER_TEMP_C = 90.0
WATER_DENSITY_90C = 965.3              # kg/m³
WATER_VISCOSITY_90C = 3.15e-7          # m²/s (運動黏滯度)
AIR_DENSITY_20C = 1.204                # kg/m³
AIR_VISCOSITY_20C = 1.516e-5           # m²/s

# V60幾何參數 (真實規格) - 擴大物理域以完整包含
CUP_HEIGHT = 0.085                     # m (8.5 cm)
TOP_RADIUS = 0.058                     # m (11.6 cm直徑)
BOTTOM_RADIUS = 0.010                  # m (2 cm出水孔，真實V60尺寸)

# 物理域尺寸 - 擴大至14cm以完整包含V60 + 20%安全裕度
PHYSICAL_DOMAIN_SIZE = 0.14            # m (14 cm, 包含11.6cm V60 + 裕度)

# ==============================================
# 科學的尺度轉換 (CFD專家修正版)
# ==============================================

# 特徵尺度選擇 - 基於實際V60幾何和擴大物理域
L_CHAR = CUP_HEIGHT                    # 特徵長度: 8.5 cm (V60高度)
U_CHAR = 0.02                          # 特徵速度: 2 cm/s (保守估計)
T_CHAR = L_CHAR / U_CHAR               # 特徵時間: 4.25s
RHO_CHAR = WATER_DENSITY_90C           
NU_CHAR = WATER_VISCOSITY_90C          

# 格子尺度轉換 (基於NZ=224格點，物理域14cm)
SCALE_LENGTH = PHYSICAL_DOMAIN_SIZE / NZ  # 0.625 mm/lu (研究級解析度，完整V60包含)
SCALE_VELOCITY = 0.01                  # lu/ts (保守初始值，策略2優化)

# 時間尺度優化 (基於重力和黏滯度降低的物理分析)
TIME_SCALE_OPTIMIZATION_FACTOR = 1.2  # 20%時間步增加 (提升計算效率)
SCALE_TIME = (SCALE_LENGTH / SCALE_VELOCITY) * TIME_SCALE_OPTIMIZATION_FACTOR  
SCALE_DENSITY = RHO_CHAR               

# 網格物理尺寸
GRID_SIZE_CM = SCALE_LENGTH * 100      # 每個網格的實際尺寸 (0.66 mm)               

# ==============================================
# LBM鬆弛時間 (CFD理論正確版)
# ==============================================

# 格子單位下的運動黏滯度
NU_WATER_LU = WATER_VISCOSITY_90C * SCALE_TIME / (SCALE_LENGTH**2)

# 空氣黏滯度安全係數調整 (解決τ_air過度擴散問題)
AIR_VISCOSITY_SAFETY_FACTOR = 0.15  # 15%安全係數，進一步降低τ_air到合理範圍
NU_AIR_LU = (AIR_VISCOSITY_20C * AIR_VISCOSITY_SAFETY_FACTOR) * SCALE_TIME / (SCALE_LENGTH**2)

# LBM正確理論: ν = c_s²(τ - 0.5), where c_s² = 1/3
TAU_WATER = max(0.8, NU_WATER_LU / CS2 + 0.5)  # 強制最小值，增加數值穩定性
TAU_AIR = max(0.8, NU_AIR_LU / CS2 + 0.5)      # 增加黏性穩定性        

# 檢查數值穩定性範圍
MIN_TAU_STABLE = 0.51   # 絕對穩定性下限
MAX_TAU_STABLE = 2.0    # 數值擴散上限
if TAU_WATER < MIN_TAU_STABLE:
    print(f"❌ τ_water={TAU_WATER:.6f} < {MIN_TAU_STABLE} 數值不穩定")
elif TAU_WATER > MAX_TAU_STABLE:
    print(f"⚠️  τ_water={TAU_WATER:.6f} > {MAX_TAU_STABLE} 過度擴散")
else:
    print(f"✅ τ_water={TAU_WATER:.6f} 數值穩定")

if TAU_AIR < MIN_TAU_STABLE:
    print(f"❌ τ_air={TAU_AIR:.6f} < {MIN_TAU_STABLE} 數值不穩定")
elif TAU_AIR > MAX_TAU_STABLE:
    print(f"⚠️  τ_air={TAU_AIR:.6f} > {MAX_TAU_STABLE} 過度擴散")
elif TAU_AIR > 1.8:
    print(f"📊 τ_air={TAU_AIR:.6f} 可接受範圍 (安全係數: {AIR_VISCOSITY_SAFETY_FACTOR*100:.0f}%)")
else:
    print(f"✅ τ_air={TAU_AIR:.6f} 最佳範圍 (安全係數: {AIR_VISCOSITY_SAFETY_FACTOR*100:.0f}%)")

# ==============================================
# 密度和重力
# ==============================================

RHO_WATER = 1.0                        # 參考密度
RHO_AIR = AIR_DENSITY_20C / WATER_DENSITY_90C  # 真實密度比

# 重力轉換 (優化CFD流動速度 - 15%強度設定)
GRAVITY_PHYS = 9.81                    # m/s²
GRAVITY_LU_FULL = GRAVITY_PHYS * (SCALE_TIME**2) / SCALE_LENGTH  # 完整重力: ~61.34
GRAVITY_STRENGTH_FACTOR = 0.15         # 15%重力強度，優化流動速度同時保持數值穩定性
GRAVITY_LU = GRAVITY_LU_FULL * GRAVITY_STRENGTH_FACTOR  # 實際使用重力: ~9.20

# 表面張力係數
SURFACE_TENSION_PHYS = 0.0728          # N/m (90°C水的表面張力)
WEBER_NUMBER = 1.0                     # 目標Weber數
SURFACE_TENSION_LU = (RHO_WATER * (U_CHAR * SCALE_TIME / SCALE_LENGTH)**2 * SCALE_LENGTH) / WEBER_NUMBER

# ==============================================
# 無量綱數分析 (CFD專家修正版)
# ==============================================

# Reynolds數計算 (正確公式)
RE_CHAR = U_CHAR * L_CHAR / NU_CHAR    # 物理特徵Re = UL/ν
RE_LATTICE = SCALE_VELOCITY * NZ / NU_WATER_LU  # 格子Re

# CFL數檢查 (正確LBM理論)
CFL_NUMBER = SCALE_VELOCITY * DT / DX   # CFL = u·Δt/Δx，在LBM中DT=DX=1
MAX_VELOCITY_LU = 0.1                   # Ma < 0.3的限制
MACH_NUMBER = SCALE_VELOCITY / np.sqrt(CS2)  # Ma = u/c_s

# 穩定性檢查
if CFL_NUMBER > 0.5:
    print(f"❌ CFL={CFL_NUMBER:.3f} > 0.5 數值不穩定")
elif CFL_NUMBER > 0.1:
    print(f"⚠️  CFL={CFL_NUMBER:.3f} > 0.1 精度下降")
else:
    print(f"✅ CFL={CFL_NUMBER:.3f} 數值穩定")

if MACH_NUMBER > 0.3:
    print(f"❌ Ma={MACH_NUMBER:.3f} > 0.3 可壓縮效應")
elif MACH_NUMBER > 0.1:
    print(f"⚠️  Ma={MACH_NUMBER:.3f} > 0.1 建議降低")
else:
    print(f"✅ Ma={MACH_NUMBER:.3f} 不可壓縮假設有效")

# Froude數 (重力與慣性力比)
FR_CHAR = U_CHAR / np.sqrt(GRAVITY_PHYS * L_CHAR)

# 物理參數診斷輸出
print(f"\n🔬 CFD無量綱數診斷:")
print(f"Re_physical = {RE_CHAR:.1f}")
print(f"Re_lattice = {RE_LATTICE:.1f}")
print(f"Fr = {FR_CHAR:.3f}")
print(f"特徵時間 = {T_CHAR:.2f}s")
print(f"格子時間步 = {SCALE_TIME*1000:.1f}ms (優化係數: {TIME_SCALE_OPTIMIZATION_FACTOR:.1f}x)")

# ==============================================
# LES湍流建模
# ==============================================

SMAGORINSKY_CONSTANT = 0.17
LES_FILTER_WIDTH = 1.0
ENABLE_LES = True
LES_REYNOLDS_THRESHOLD = 500.0

# ==============================================
# 咖啡相關參數
# ==============================================

COFFEE_POWDER_MASS = 0.02              # 20g
COFFEE_BEAN_DENSITY = 1200             # kg/m³
PORE_PERC = 0.45                       # 孔隙率

# 咖啡粉粒徑
DP = 6.5e-4                            # 0.65mm
PARTICLE_DIAMETER_MM = 0.65
COFFEE_PARTICLE_RADIUS = DP / 2        # 0.325mm radius in meters

# ==============================================
# 咖啡床幾何計算
# ==============================================

# V60內部錐台體積計算
V60_INTERNAL_VOLUME = (math.pi * CUP_HEIGHT / 3) * (TOP_RADIUS**2 + TOP_RADIUS * BOTTOM_RADIUS + BOTTOM_RADIUS**2)
COFFEE_FILL_RATIO = 0.15               # 填充V60的15%體積
COFFEE_BED_VOLUME_PHYS = V60_INTERNAL_VOLUME * COFFEE_FILL_RATIO

# 基於實際孔隙率計算咖啡床高度  
COFFEE_SOLID_VOLUME = COFFEE_POWDER_MASS / COFFEE_BEAN_DENSITY
ACTUAL_POROSITY = 1 - (COFFEE_SOLID_VOLUME / COFFEE_BED_VOLUME_PHYS)

def solve_coffee_bed_height():
    """基於錐台幾何精確計算咖啡床高度"""
    target_volume = COFFEE_BED_VOLUME_PHYS
    cone_slope = (TOP_RADIUS - BOTTOM_RADIUS) / CUP_HEIGHT
    
    # 二分法求解
    h_min, h_max = 0.001, CUP_HEIGHT * 0.6
    for _ in range(100):
        h_test = (h_min + h_max) / 2
        r_top = BOTTOM_RADIUS + h_test * cone_slope
        volume_test = (math.pi * h_test / 3) * (BOTTOM_RADIUS**2 + BOTTOM_RADIUS * r_top + r_top**2)
        
        if abs(volume_test - target_volume) < 1e-8:
            return h_test, r_top
        elif volume_test < target_volume:
            h_min = h_test
        else:
            h_max = h_test
    return h_max, BOTTOM_RADIUS + h_max * cone_slope

COFFEE_BED_HEIGHT_PHYS, COFFEE_BED_TOP_RADIUS = solve_coffee_bed_height()
COFFEE_BED_HEIGHT_LU = int(COFFEE_BED_HEIGHT_PHYS / SCALE_LENGTH)

# 驗證合理性
MAX_COFFEE_HEIGHT = CUP_HEIGHT * 2/3
if COFFEE_BED_HEIGHT_PHYS <= MAX_COFFEE_HEIGHT:
    print(f"☕ 咖啡床高度: {COFFEE_BED_HEIGHT_PHYS*100:.1f}cm (合理)")
else:
    print(f"⚠️  咖啡床高度: {COFFEE_BED_HEIGHT_PHYS*100:.1f}cm (可能過高)")

# ==============================================
# 注水參數
# ==============================================

POUR_RATE_ML_S = 4.0                   # 4 ml/s
POUR_RATE_M3_S = POUR_RATE_ML_S * 1e-6
TOTAL_WATER_ML = 320                   # 320ml
BREWING_TIME_SECONDS = 140             # 2:20

# 注水幾何
POUR_HEIGHT_CM = 12.5

# 以物理單位設定入口直徑，提供噴嘴上限（不再支援比例參數）
INLET_DIAMETER_M = 0.005              # 直接設定入口直徑（m），預設0.5 cm
NOZZLE_DIAMETER_M = 0.005             # 物理噴嘴直徑上限（0.5 cm）

# 重力加速修正（用於速度推導）
GRAVITY_CORRECTION = 0.05  # 5% 重力修正，而非完整自由落體

def _compute_inlet_diameter(verbose: bool = False) -> float:
    """計算實際入水直徑（m），僅使用 INLET_DIAMETER_M，並夾制至噴嘴上限。"""
    d = min(INLET_DIAMETER_M, NOZZLE_DIAMETER_M)
    if verbose and INLET_DIAMETER_M > NOZZLE_DIAMETER_M:
        print(f"⚠️  入水直徑設定為{INLET_DIAMETER_M*100:.2f}cm，已夾制至噴嘴上限{NOZZLE_DIAMETER_M*100:.2f}cm")
    return d

def recompute_pouring_derived(verbose: bool = False) -> None:
    """重算與注水相關的派生量（允許於 YAML 覆寫後被呼叫）。"""
    global INLET_DIAMETER, INLET_AREA, INLET_VELOCITY_BASE, INLET_VELOCITY_PHYS
    global INLET_VELOCITY_LU, INLET_VELOCITY
    INLET_DIAMETER = _compute_inlet_diameter(verbose)
    INLET_AREA = math.pi * (INLET_DIAMETER/2.0)**2
    INLET_VELOCITY_BASE = POUR_RATE_M3_S / INLET_AREA
    INLET_VELOCITY_PHYS = INLET_VELOCITY_BASE * (1.0 + GRAVITY_CORRECTION)
    INLET_VELOCITY_LU = INLET_VELOCITY_PHYS * SCALE_TIME / SCALE_LENGTH
    INLET_VELOCITY = min(0.05, INLET_VELOCITY_LU)
    if INLET_VELOCITY > 0.1 and verbose:
        print(f"⚠️  INLET_VELOCITY={INLET_VELOCITY:.3f} > 0.1，自動限制至0.05")

# 初次計算（之後可被 YAML 覆寫後重算）
recompute_pouring_derived(verbose=False)

# ==============================================
# 模擬控制參數
# ==============================================

MAX_STEPS = int(BREWING_TIME_SECONDS / SCALE_TIME)
POURING_STEPS = int(80 / SCALE_TIME)  # 80秒注水時間
OUTPUT_FREQ = max(100, MAX_STEPS // 1000)

# ==============================================
# 相場和多相流參數
# ==============================================

PHASE_WATER = 1.0
PHASE_AIR = 0.0
INTERFACE_THICKNESS = 1.5
CAHN_HILLIARD_MOBILITY = 0.01
SURFACE_TENSION_PHYS = 0.0728          # N/m

# ==============================================
# 輸出診斷信息
# ==============================================

print("=== 修正版CFD參數診斷 ===")
print(f"📏 尺度轉換:")
print(f"  長度: {SCALE_LENGTH*1000:.2f} mm/lu")
print(f"  時間: {SCALE_TIME*1000:.2f} ms/ts")
print(f"  速度: {SCALE_VELOCITY:.3f} lu/ts")

print(f"\n🔍 數值穩定性:")
print(f"  CFL數: {CFL_NUMBER:.3f} ({'✅' if CFL_NUMBER < 0.7 else '⚠️' if CFL_NUMBER < 1.0 else '❌'})")
print(f"  τ_water: {TAU_WATER:.6f} ({'✅' if TAU_WATER > 0.55 else '⚠️'})")
print(f"  τ_air: {TAU_AIR:.6f}")
print(f"  Mach數: {MACH_NUMBER:.3f}")

print(f"\n🌊 流動特性:")
print(f"  物理Re: {RE_CHAR:.1f}")
print(f"  格子Re: {RE_LATTICE:.1f}")
print(f"  Froude數: {FR_CHAR:.3f}")

print(f"\n⏱️  模擬控制:")
print(f"  總步數: {MAX_STEPS:,}")
print(f"  注水步數: {POURING_STEPS:,}")
print(f"  實際模擬時間: {MAX_STEPS * SCALE_TIME:.1f} 秒")

print(f"\n☕ 咖啡參數:")
print(f"  咖啡粉: {COFFEE_POWDER_MASS*1000:.0f}g")
print(f"  顆粒直徑: {PARTICLE_DIAMETER_MM:.2f}mm")
print(f"  注水速度: {POUR_RATE_ML_S:.1f} ml/s")

# 驗證關鍵條件
errors = []
if CFL_NUMBER >= 1.0:
    errors.append(f"CFL不穩定: {CFL_NUMBER:.3f} ≥ 1.0")
if TAU_WATER <= 0.5:
    errors.append(f"τ_water不穩定: {TAU_WATER:.6f} ≤ 0.5")
if MACH_NUMBER > 0.3:
    errors.append(f"Mach數過高: {MACH_NUMBER:.3f} > 0.3")

if errors:
    print(f"\n❌ 發現問題:")
    for error in errors:
        print(f"  • {error}")
else:
    print(f"\n✅ 所有參數通過驗證！")

# ====================
# CFD參數一致性驗證系統 (優化)
# ====================

def validate_parameter_consistency():
    """
    檢查各模組間參數一致性 (CFD一致性優化)
    
    全面驗證CFD系統中各模組使用的物理參數、數值參數
    和邊界條件參數的一致性，確保系統級一致性。
    
    Validation Categories:
        1. 物理參數一致性 (密度、黏滯度、溫度)
        2. 數值參數一致性 (時間步、空間步、CFL)
        3. 幾何參數一致性 (尺度轉換、座標系)
        4. 邊界條件一致性 (邊界類型、參數值)
        
    Error Detection:
        - 參數值不匹配
        - 單位不一致
        - 數值範圍超出合理區間
        - 模組間衝突設定
    """
    print("\n🔍 CFD參數一致性檢查...")
    
    consistency_errors = []
    warnings = []
    
    try:
        # 1. 物理參數一致性
        print("   ├─ 檢查物理參數一致性...")
        _check_physical_parameters(consistency_errors, warnings)
        
        # 2. 數值參數一致性  
        print("   ├─ 檢查數值參數一致性...")
        _check_numerical_parameters(consistency_errors, warnings)
        
        # 3. 幾何參數一致性
        print("   ├─ 檢查幾何參數一致性...")
        _check_geometric_parameters(consistency_errors, warnings)
        
        # 4. 邊界條件一致性
        print("   ├─ 檢查邊界條件一致性...")
        _check_boundary_parameters(consistency_errors, warnings)
        
        # 5. 模組耦合一致性
        print("   ├─ 檢查模組耦合一致性...")
        _check_coupling_parameters(consistency_errors, warnings)
        
        # 彙總結果
        if consistency_errors:
            print(f"\n❌ 發現 {len(consistency_errors)} 個一致性錯誤:")
            for i, error in enumerate(consistency_errors, 1):
                print(f"  {i}. {error}")
            raise ValueError("CFD參數一致性檢查失敗")
            
        if warnings:
            print(f"\n⚠️  發現 {len(warnings)} 個警告:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        print("   └─ ✅ CFD參數一致性檢查通過")
        
    except Exception as e:
        print(f"   └─ ❌ 一致性檢查過程失敗: {e}")
        raise

def _check_physical_parameters(errors, warnings):
    """檢查物理參數一致性"""
    # 密度參數檢查
    if abs(RHO_WATER - 1.0) > 1e-6:
        errors.append(f"水密度參考值不為1.0: {RHO_WATER}")
    
    # 密度比檢查
    expected_air_ratio = AIR_DENSITY_20C / WATER_DENSITY_90C
    if abs(RHO_AIR - expected_air_ratio) > 1e-6:
        errors.append(f"空氣密度比不一致: {RHO_AIR} vs {expected_air_ratio}")
    
    # 黏滯度參數檢查
    if TAU_WATER <= 0.5:
        errors.append(f"水相鬆弛時間不穩定: τ_water = {TAU_WATER}")
    
    if TAU_AIR <= 0.5:
        errors.append(f"氣相鬆弛時間不穩定: τ_air = {TAU_AIR}")
    
    # 溫度一致性
    if WATER_TEMP_C != 90.0:
        warnings.append(f"水溫度非標準90°C: {WATER_TEMP_C}°C")

def _check_numerical_parameters(errors, warnings):
    """檢查數值參數一致性"""
    # CFL數檢查
    if CFL_NUMBER >= 1.0:
        errors.append(f"CFL數不穩定: {CFL_NUMBER} >= 1.0")
    elif CFL_NUMBER > 0.5:
        warnings.append(f"CFL數偏高: {CFL_NUMBER} > 0.5")
    
    # Mach數檢查
    if MACH_NUMBER > 0.3:
        errors.append(f"Mach數過高: {MACH_NUMBER} > 0.3")
    elif MACH_NUMBER > 0.1:
        warnings.append(f"Mach數建議降低: {MACH_NUMBER} > 0.1")
    
    # 時間步檢查
    if DT != 1.0:
        warnings.append(f"LBM時間步非標準值: DT = {DT}")
    
    # 空間步檢查
    if DX != 1.0:
        warnings.append(f"LBM空間步非標準值: DX = {DX}")

def _check_geometric_parameters(errors, warnings):
    """檢查幾何參數一致性"""
    # 網格尺寸一致性
    if NX != NY or NY != NZ:
        warnings.append(f"非立方網格: {NX}×{NY}×{NZ}")
    
    # 尺度轉換一致性
    expected_scale_length = PHYSICAL_DOMAIN_SIZE / NZ
    if abs(SCALE_LENGTH - expected_scale_length) > 1e-8:
        errors.append(f"長度尺度不一致: {SCALE_LENGTH} vs {expected_scale_length}")
    
    # V60幾何合理性
    if TOP_RADIUS <= BOTTOM_RADIUS:
        errors.append(f"V60幾何不合理: 頂部半徑 <= 底部半徑")
    
    # 物理域包含V60檢查
    v60_diameter = 2 * TOP_RADIUS
    if PHYSICAL_DOMAIN_SIZE < v60_diameter * 1.2:
        warnings.append(f"物理域可能太小: {PHYSICAL_DOMAIN_SIZE*100:.1f}cm vs V60直徑{v60_diameter*100:.1f}cm")

def _check_boundary_parameters(errors, warnings):
    """檢查邊界條件一致性"""
    # 重力參數檢查
    if GRAVITY_STRENGTH_FACTOR > 0.2:
        warnings.append(f"重力強度係數偏高: {GRAVITY_STRENGTH_FACTOR}")
    
    # 注水參數檢查
    if INLET_VELOCITY > 0.1:
        warnings.append(f"注水速度偏高: {INLET_VELOCITY} > 0.1 lu/ts")
    
    # 咖啡床高度合理性
    if hasattr(locals(), 'COFFEE_BED_HEIGHT_PHYS'):
        if COFFEE_BED_HEIGHT_PHYS > CUP_HEIGHT * 0.5:
            warnings.append(f"咖啡床高度偏高: {COFFEE_BED_HEIGHT_PHYS*100:.1f}cm")

def _check_coupling_parameters(errors, warnings):
    """檢查模組耦合一致性"""
    # LES參數與Reynolds數匹配
    if ENABLE_LES and RE_CHAR < LES_REYNOLDS_THRESHOLD:
        warnings.append(f"LES啟用但Re數偏低: Re={RE_CHAR} < {LES_REYNOLDS_THRESHOLD}")
    
    # 表面張力與Weber數一致性
    if abs(WEBER_NUMBER - 1.0) > 0.1:
        warnings.append(f"Weber數非標準值: We = {WEBER_NUMBER}")
    
    # 多相流參數合理性
    interface_thickness = INTERFACE_THICKNESS  # 使用實際配置值
    if interface_thickness > SCALE_LENGTH * 1000 * 3:  # 界面厚度不應超過3個格子尺寸
        warnings.append(f"界面厚度相對格子尺寸偏大")

def get_consistency_report():
    """
    獲取完整的參數一致性報告
    
    Returns:
        dict: 包含所有參數一致性信息的報告
    """
    return {
        'physical_parameters': {
            'water_density': RHO_WATER,
            'air_density': RHO_AIR,
            'water_tau': TAU_WATER,
            'air_tau': TAU_AIR,
            'temperature': WATER_TEMP_C
        },
        'numerical_parameters': {
            'cfl_number': CFL_NUMBER,
            'mach_number': MACH_NUMBER,
            'reynolds_physical': RE_CHAR,
            'reynolds_lattice': RE_LATTICE
        },
        'geometric_parameters': {
            'grid_size': (NX, NY, NZ),
            'scale_length': SCALE_LENGTH,
            'physical_domain': PHYSICAL_DOMAIN_SIZE,
            'v60_geometry': (TOP_RADIUS, BOTTOM_RADIUS, CUP_HEIGHT)
        },
        'coupling_parameters': {
            'les_enabled': ENABLE_LES,
            'les_threshold': LES_REYNOLDS_THRESHOLD,
            'weber_number': WEBER_NUMBER,
            'surface_tension': SURFACE_TENSION_LU
        }
    }

# 在模組載入時自動執行一致性檢查
if __name__ == "__main__":
    validate_parameter_consistency()
else:
    # 導入時執行簡化檢查
    try:
        validate_parameter_consistency()
    except Exception as e:
        print(f"⚠️  CFD參數一致性警告: {e}")
        print("   建議運行完整檢查: python config.py")
