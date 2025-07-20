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
# LBM網格與參數設定 (調整為較小的網格以提高穩定性)
# -----------------------
NX = 64          # x方向格點數 (水平) - 減小網格
NY = 64          # y方向格點數 (水平) - 減小網格
NZ = 128         # z方向格點數 (垂直) - 減小網格
DX = 1.0         # 格點間距 (lattice units)
DT = 1.0         # 時間步長 (lattice units)

# -----------------------
# 物理尺寸和網格轉換 (基於V60-02規格)
# -----------------------
PHYSICAL_WIDTH = 0.12           # 物理寬度 12cm (略大於V60上徑11.6cm)
PHYSICAL_HEIGHT = 0.15          # 物理高度 15cm (包含上方注水空間)
GRID_SIZE_CM = PHYSICAL_WIDTH / NX  # 每個網格的實際尺寸 (cm)

# -----------------------
# Hario V60 濾杯幾何參數 (3D) - 標準規格
# -----------------------
# V60-02規格: 上徑116mm, 下徑出水孔4mm, 高度約82mm, 60度錐角
CUP_HEIGHT      = 0.082          # 濾杯高度 (82 mm) - V60-02標準高度
TOP_DIAMETER    = 0.116          # 濾杯上部直徑 (116 mm) - V60-02標準上徑
TOP_RADIUS      = TOP_DIAMETER / 2.0
BOTTOM_DIAMETER = 0.004          # 濾杯底部出水孔直徑 (4 mm) - V60標準出水孔
BOTTOM_RADIUS   = BOTTOM_DIAMETER / 2.0
CONE_ALPHA      = (TOP_RADIUS - BOTTOM_RADIUS) / CUP_HEIGHT  # 約60度錐角
V60_CONE_ANGLE  = 60.0           # V60特有的60度錐角

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

# 物理到格子單位的轉換比例 (針對手沖咖啡優化)
SCALE_LENGTH = CUP_HEIGHT / NZ   # m/lu (米每格子單位)
SCALE_TIME = 1e-4               # s/lu (秒每格子單位) - 0.1ms每時間步

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
# 咖啡粉與多孔介質參數
# -----------------------
COFFEE_POWDER_MASS = 0.02       # 20g咖啡粉
COFFEE_BEAN_DENSITY = 1200      # 中烘焙咖啡豆密度 1.2g/cm³ = 1200 kg/m³
SOLID_DENSITY = COFFEE_BEAN_DENSITY  # 咖啡粉固體密度使用咖啡豆密度
PORE_PERC = 0.45               # 孔隙率 45%
DP = 5e-4                      # 咖啡粉平均粒徑 (m)

# Darcy數 (多孔介質滲透率的無量綱參數)
DARCY_NUMBER = 1e-8

# 計算咖啡床高度 (格子單位)
COFFEE_BED_HEIGHT_PHYS = COFFEE_POWDER_MASS / (SOLID_DENSITY * (1 - PORE_PERC) * (math.pi * (BOTTOM_RADIUS ** 2)))
COFFEE_BED_HEIGHT_LU = int(COFFEE_BED_HEIGHT_PHYS / SCALE_LENGTH)

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

# V60入水區域計算 (基於濾杯上徑的20%區域)
INLET_DIAMETER_RATIO = 0.2      # 入水區域佔上徑的比例
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
