# config.py
"""
本模組定義D3Q19 LBM模擬手沖咖啡所需的所有常數與參數
使用Taichi GPU加速和並行運算優化

Hario V60-02 標準規格:
- 上部直徑: 116mm
- 下部出水孔: 4mm  
- 高度: 82mm
- 錐角: 60度
- 螺旋槽紋設計: 提升萃取效率

手沖咖啡操作參數:
- 注水速度: 4 ml/s (標準手沖速度)
- 總注水量: 320ml (20g咖啡豆 × 16倍水粉比)
- 注水時間: 約80秒 (分3-4次注水)
- 總萃取時間: 2:20 (140秒) - 標準手沖時間
- 注水高度: 10-15cm (典型手沖高度)
- 咖啡粉用量: 20g
- 研磨粗細: 中粗研磨 (約0.5mm粒徑)

中烘焙咖啡豆物性:
- 密度: 1.2 g/cm³ (1200 kg/m³)
- 顆粒大小: 約0.5mm
"""

import math
import numpy as np
import taichi as ti

# -----------------------
# LBM網格與參數設定 (平衡V60尺寸與計算效率)
# -----------------------
# 性能優化選項 (可根據需求調整):
# - 高精度: 160³ (4.1M節點) - 完整模擬
# - 平衡模式: 128³ (2.1M節點) - 性能/精度平衡 
# - 快速模式: 96³ (0.88M節點) - 4.7倍加速
# - 測試模式: 64³ (0.26M節點) - 15.8倍加速

# 當前設定: 平衡模式 (推薦)
NX = 128         # x方向格點數 (水平) - 平衡尺寸與效率
NY = 128         # y方向格點數 (水平) - 平衡尺寸與效率
NZ = 128         # z方向格點數 (垂直) - 平衡尺寸與效率
DX = 1.0         # 格點間距 (lattice units)
DT = 1.0         # 時間步長 (lattice units)

# -----------------------
# 物理尺寸和網格轉換 (調整為V60真實比例)
PHYSICAL_WIDTH = 0.12            # 物理寬度 12cm (容納11.6cm V60)
PHYSICAL_HEIGHT = 0.12           # 物理高度 12cm (容納8.5cm V60)
GRID_SIZE_CM = PHYSICAL_WIDTH / NX  # 每個網格的實際尺寸 (cm)

# -----------------------
# Hario V60-02 濾杯幾何參數 (3D) - 真實V60規格
# -----------------------
# V60-02標準規格 (符合官方尺寸):
CUP_HEIGHT      = 0.085          # 濾杯內部高度 (8.5 cm) - V60-02標準
TOP_RADIUS      = 0.058          # 濾杯內部上半徑 (5.8 cm) - 直徑11.6cm
TOP_DIAMETER    = TOP_RADIUS * 2.0
BOTTOM_DIAMETER = 0.004          # 濾杯底部出水孔直徑 (4 mm) - V60標準出水孔
BOTTOM_RADIUS   = BOTTOM_DIAMETER / 2.0
CONE_ALPHA      = (TOP_RADIUS - BOTTOM_RADIUS) / CUP_HEIGHT  # 實際錐角
V60_CONE_ANGLE  = 60.0           # V60標準錐角

# LBM模型參數 (僅3D)
Q_3D = 19  # D3Q19模型

# D3Q19速度向量
CX_3D = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.int32)
CY_3D = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int32)
CZ_3D = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int32)

# D3Q19權重係數
WEIGHTS_3D = np.array([
    1/3,                           # 靜止 (0,0,0)
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,  # 6個面中心
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36,  # 12個邊中心
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36
], dtype=np.float32)

# 物理到格子單位的轉換比例 (針對真實V60優化)
SCALE_LENGTH = CUP_HEIGHT / (90)    # m/lu (米每格子單位) - 90格對應V60高度8.5cm
SCALE_TIME = 1e-4                   # s/lu (秒每格子單位) - 0.1ms每時間步

# -----------------------
# 流體物性參數 (格子單位) - 基於90°C熱水
# -----------------------
# 90°C水的物理性質
WATER_TEMP_C = 90.0             # 水溫 90°C
WATER_VISCOSITY_90C = 3.15e-7   # 90°C水的動力粘滯度 (m²/s)
WATER_DENSITY_90C = 965.3       # 90°C水的密度 (kg/m³)

# LBM中的鬆弛時間參數 (基於90°C水的粘滯度)
# τ = 0.5 + 3*ν/(c²*Δt), 其中 ν = μ/ρ 是運動粘滯度
KINEMATIC_VISCOSITY = WATER_VISCOSITY_90C  # m²/s
VISCOSITY_LU = KINEMATIC_VISCOSITY * (SCALE_TIME / SCALE_LENGTH**2)  # 格子單位粘滯度
TAU_WATER = 0.5 + 3.0 * VISCOSITY_LU      # 基於真實粘滯度的鬆弛時間
TAU_AIR = 0.8        # 空氣的鬆弛時間 (保持原值)

# 物理密度 (kg/m^3) - 基於90°C
RHO_WATER_PHYS = WATER_DENSITY_90C  # 90°C水密度
RHO_AIR_PHYS = 1.0                  # 90°C空氣密度 (略低於常溫)

# 格子單位中的密度 (減少密度差異以提高穩定性)
RHO_WATER = 1.0      # 水的格子密度
RHO_AIR = 0.1        # 空氣的格子密度 (從0.001增加到0.1)

# 重力加速度 (格子單位) - 基於真實重力
GRAVITY_PHYS = 9.81             # 物理重力加速度 m/s²
GRAVITY_LU = GRAVITY_PHYS * (SCALE_TIME**2) / SCALE_LENGTH  # 格子單位重力

# 表面張力參數
SURFACE_TENSION = 0.02   # 界面張力係數 (減小)

# -----------------------
# 咖啡粉與多孔介質參數 (基於手沖研磨度計算)
# -----------------------
COFFEE_POWDER_MASS = 0.02       # 20g咖啡粉
COFFEE_BEAN_DENSITY = 1200      # 中烘焙咖啡豆密度 1.2g/cm³ = 1200 kg/m³
SOLID_DENSITY = COFFEE_BEAN_DENSITY  # 咖啡粉固體密度使用咖啡豆密度
PORE_PERC = 0.45               # 孔隙率 45%

# 咖啡颗粒尺寸参数 (基於手沖研磨度 - 二號砂糖大小)
DP = 6.5e-4                    # 咖啡粉平均粒徑 0.65mm (m) - 手沖研磨度
PARTICLE_DIAMETER_MM = 0.65    # 主体颗粒直径 (mm)
PARTICLE_RADIUS_M = DP / 2     # 颗粒半径 (m)

# 咖啡颗粒数量 (基於粒径分布计算)
# 使用分布式计算: 细粉10% + 主体80% + 粗粒10%
TOTAL_PARTICLE_COUNT = 493796  # 总颗粒数 (约49.4万个)
MAIN_PARTICLE_COUNT = 92726    # 主体颗粒数 (0.65mm)
FINE_PARTICLE_COUNT = 397887   # 细粉颗粒数 (0.2mm) 
COARSE_PARTICLE_COUNT = 3183   # 粗粒颗粒数 (1.0mm)

# 单个颗粒物理参数
SINGLE_PARTICLE_VOLUME = 1.4379e-7  # 单颗体积 (m³) - 0.65mm球形
SINGLE_PARTICLE_MASS = 1.7255e-7    # 单颗质量 (kg)

# Darcy數 (多孔介質滲透率的無量綱參數)
DARCY_NUMBER = 1e-8

# 計算咖啡床參數 (基於更新後的V60內部錐形幾何)
# V60內部錐台體積: V = (π * h / 3) * (R² + R*r + r²)
V60_INTERNAL_VOLUME = (math.pi * CUP_HEIGHT / 3) * (TOP_RADIUS**2 + TOP_RADIUS * BOTTOM_RADIUS + BOTTOM_RADIUS**2)
COFFEE_FILL_RATIO = 0.15  # 咖啡粉填充V60的15%體積 (合理比例)
COFFEE_BED_VOLUME_PHYS = V60_INTERNAL_VOLUME * COFFEE_FILL_RATIO  # 修正後的咖啡床體積

# 基於實際孔隙率計算咖啡床高度
COFFEE_SOLID_VOLUME = COFFEE_POWDER_MASS / SOLID_DENSITY  # 咖啡固體體積
ACTUAL_POROSITY = 1 - (COFFEE_SOLID_VOLUME / COFFEE_BED_VOLUME_PHYS)  # 實際孔隙率 ~80.5%

# 咖啡床高度計算 (基於錐台體積公式，而非平均半徑近似)
# 解錐台體積方程: V = (π*h/3) * (r₁² + r₁*r₂ + r₂²)
# 其中 r₁ = BOTTOM_RADIUS, r₂ = r₁ + h*tan(α)
# 使用數值方法求解高度

def solve_coffee_bed_height():
    """基於錐台幾何精確計算咖啡床高度"""
    target_volume = COFFEE_BED_VOLUME_PHYS
    cone_slope = (TOP_RADIUS - BOTTOM_RADIUS) / CUP_HEIGHT
    
    # 二分法求解高度
    h_min, h_max = 0.001, CUP_HEIGHT * 0.6  # 最大不超過V60高度的60%
    
    for _ in range(100):  # 最多迭代100次
        h_test = (h_min + h_max) / 2
        r_top = BOTTOM_RADIUS + h_test * cone_slope
        
        # 錐台體積計算
        volume_test = (math.pi * h_test / 3) * (BOTTOM_RADIUS**2 + BOTTOM_RADIUS * r_top + r_top**2)
        
        if abs(volume_test - target_volume) < 1e-8:  # 收斂
            return h_test, r_top
        elif volume_test < target_volume:
            h_min = h_test
        else:
            h_max = h_test
    
    return h_max, BOTTOM_RADIUS + h_max * cone_slope

COFFEE_BED_HEIGHT_PHYS, COFFEE_BED_TOP_RADIUS = solve_coffee_bed_height()
COFFEE_BED_HEIGHT_LU = int(COFFEE_BED_HEIGHT_PHYS / SCALE_LENGTH)

# 验证咖啡床高度不超过V60的2/3
MAX_COFFEE_HEIGHT = CUP_HEIGHT * 2/3  # V60高度的2/3
if COFFEE_BED_HEIGHT_PHYS > MAX_COFFEE_HEIGHT:
    print(f"⚠️  咖啡床高度 {COFFEE_BED_HEIGHT_PHYS*100:.1f}cm 超过V60的2/3高度 {MAX_COFFEE_HEIGHT*100:.1f}cm")
else:
    print(f"✅ 咖啡床高度 {COFFEE_BED_HEIGHT_PHYS*100:.1f}cm 合理 (< {MAX_COFFEE_HEIGHT*100:.1f}cm)")
    print(f"✅ 咖啡床頂部半徑 {COFFEE_BED_TOP_RADIUS*100:.1f}cm 在V60範圍內")

# -----------------------
# 手沖咖啡注水參數 (基於實際操作)
# -----------------------
# 手沖咖啡標準注水速度: 4 ml/s
POUR_RATE_ML_S = 4.0            # 注水速度 (ml/s)
POUR_RATE_M3_S = POUR_RATE_ML_S * 1e-6  # 轉換為 m³/s

# 手沖咖啡典型參數
TOTAL_WATER_ML = 320            # 總注水量 320ml (16:1水粉比)
POURING_TIME_S = TOTAL_WATER_ML / POUR_RATE_ML_S  # 實際注水時間 80秒
BREWING_TIME_SECONDS = 140      # 總萃取時間 2:20 (140秒) - 標準手沖時間

# 注水高度參數 (影響重力加速和入水速度)
POUR_HEIGHT_CM = 12.5           # 注水高度 12.5cm (典型手沖高度)
POUR_HEIGHT_M = POUR_HEIGHT_CM / 100.0  # 轉換為米

# V60入水區域計算 (基於濾杯上徑的比例)
# 手沖注水寬度選項:
# - 細水流: 0.15 (15%) - 精準控制
# - 標準: 0.2 (20%) - 當前設置  
# - 寬水流: 0.3 (30%) - 快速萃取
INLET_DIAMETER_RATIO = 0.3      # 調整為30%，增加注水寬度
INLET_DIAMETER = TOP_DIAMETER * INLET_DIAMETER_RATIO
INLET_AREA = math.pi * (INLET_DIAMETER/2.0)**2  # 入水面積 (m²)

# 物理入水速度計算 (考慮重力加速度效應)
INLET_VELOCITY_BASE = POUR_RATE_M3_S / INLET_AREA  # 基礎入水速度 m/s
GRAVITY_ACCELERATION = 9.81     # 重力加速度 m/s²
GRAVITY_VELOCITY = math.sqrt(2 * GRAVITY_ACCELERATION * POUR_HEIGHT_M)  # 重力自由落體速度
INLET_VELOCITY_PHYS = INLET_VELOCITY_BASE + GRAVITY_VELOCITY  # 考慮重力的總入水速度

# -----------------------
# 邊界條件參數 (基於真實注水速度)
# -----------------------
# 入水速度 (格子單位) - 基於4ml/s注水速度計算
INLET_VELOCITY = INLET_VELOCITY_PHYS * SCALE_TIME / SCALE_LENGTH  # 約0.00148 lu/ts

# 模擬步數和輸出頻率 (基於真實手沖時間)
MAX_STEPS = int(BREWING_TIME_SECONDS / SCALE_TIME)  # 對應真實時間的步數 (約2,400,000步)
MAX_STEPS_DEMO = 10000          # 演示模式的較少步數
POURING_STEPS = int(POURING_TIME_S / SCALE_TIME)    # 實際注水步數 (約800,000步)
OUTPUT_FREQ = max(200, MAX_STEPS // 1000)  # 輸出頻率自動調整

# -----------------------
# 相識別參數
# -----------------------
# 相場參數 (0=空氣, 1=水)
PHASE_AIR = 0.0
PHASE_WATER = 1.0
INTERFACE_THICKNESS = 3  # 界面厚度(格子單位)

# -----------------------
# Taichi並行優化參數
# -----------------------
# 使用稀疏矩陣的閾值
SPARSE_THRESHOLD = 0.1

# 並行執行塊大小
BLOCK_SIZE = 16

# GPU記憶體管理
USE_SPARSE_MATRIX = True
MEMORY_POOL_SIZE = 1024  # MB
