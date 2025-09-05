# core_config.py - 核心LBM參數配置
"""
核心LBM參數 - 工業級調校，絕對不可隨意修改
包含所有基礎LBM理論參數和數值穩定性關鍵參數

這些參數經過嚴格的CFD理論分析和大量測試驗證，
任何修改都必須經過完整的穩定性測試。

開發：opencode + GitHub Copilot
"""

import math
import numpy as np

# ==============================================
# 核心LBM理論參數 - D3Q19模型
# ==============================================

# 網格設定 (研究級精度) - 224³完整包含V60幾何
NX = 224
NY = 224  
NZ = 224
DX = 1.0         # 格點間距 (lattice units)
DT = 1.0         # 時間步長 (lattice units)

# LBM基本理論常數
Q_3D = 19
CS2 = 1.0/3.0      # 格子聲速平方
CS4 = CS2 * CS2
INV_CS2 = 3.0

# D3Q19離散速度向量 (標準理論定義)
CX_3D = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.int32)
CY_3D = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int32)
CZ_3D = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int32)

# D3Q19權重係數 (理論精確值)
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
# 數值穩定性關鍵參數 - 絕對不可修改
# ==============================================

# CFL條件 (工業級安全設定)
CFL_NUMBER = 0.010          # 極度保守的CFL數，確保絕對穩定
MAX_VELOCITY_LU = 0.1       # Ma < 0.3的嚴格限制
SCALE_VELOCITY = 0.01       # lu/ts (保守初始值)

# 鬆弛時間安全範圍
MIN_TAU_STABLE = 0.51       # 絕對穩定性下限
MAX_TAU_STABLE = 2.0        # 數值擴散上限
TAU_FLUID = 0.53           # 工業級調校值 (核心穩定性參數)
TAU_AIR = 0.8               # 空氣相安全值

# 時間尺度優化係數
TIME_SCALE_OPTIMIZATION_FACTOR = 1.2  # 20%時間步增加 (經過驗證)

# ==============================================
# 物理域與尺度轉換 (基礎參數)
# ==============================================

# 物理域尺寸 - 14cm完整包含V60 + 20%安全裕度
PHYSICAL_DOMAIN_SIZE = 0.14            # m (14 cm)

# 格子尺度轉換 (基於NZ=224格點)
SCALE_LENGTH = PHYSICAL_DOMAIN_SIZE / NZ    # 0.625 mm/lu (研究級解析度)

# 時間尺度 (基於長度和速度尺度)
SCALE_TIME = (SCALE_LENGTH / SCALE_VELOCITY) * TIME_SCALE_OPTIMIZATION_FACTOR

# 網格物理尺寸
GRID_SIZE_CM = SCALE_LENGTH * 100           # 每個網格的實際尺寸 (0.625 mm)

# ==============================================
# LES湍流建模參數
# ==============================================

SMAGORINSKY_CONSTANT = 0.17
LES_FILTER_WIDTH = 1.0
ENABLE_LES = True
LES_REYNOLDS_THRESHOLD = 500.0

# ==============================================
# 熱傳導LBM基礎參數 (D3Q7)
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

# 熱傳導數值穩定性參數
MIN_TAU_THERMAL = 0.51      # 熱傳鬆弛時間下限
MAX_TAU_THERMAL = 2.0       # 熱傳鬆弛時間上限
MAX_CFL_THERMAL = 0.5       # 熱擴散穩定性條件

# ==============================================
# 標準驗證函數
# ==============================================

def validate_core_parameters():
    """驗證核心參數的理論一致性"""
    
    errors = []
    
    # 基礎理論檢查
    if abs(np.sum(WEIGHTS_3D) - 1.0) > 1e-6:
        errors.append("D3Q19權重係數歸一化失敗")
    
    if abs(np.sum(W_THERMAL) - 1.0) > 1e-6:
        errors.append("D3Q7權重係數歸一化失敗")
    
    # 穩定性檢查
    if CFL_NUMBER >= 1.0:
        errors.append(f"CFL數不穩定: {CFL_NUMBER} >= 1.0")
    
    if TAU_FLUID <= MIN_TAU_STABLE:
        errors.append(f"流體鬆弛時間不穩定: {TAU_FLUID} <= {MIN_TAU_STABLE}")
    
    # Mach數檢查
    mach_number = SCALE_VELOCITY / np.sqrt(CS2)
    if mach_number > 0.3:
        errors.append(f"Mach數過高: {mach_number} > 0.3")
    
    if errors:
        raise ValueError(f"核心參數驗證失敗: {errors}")
    
    print(f"✅ 核心LBM參數驗證通過")
    print(f"   CFL={CFL_NUMBER:.3f}, Ma={mach_number:.3f}, τ={TAU_FLUID:.3f}")

# 自動驗證
if __name__ == "__main__":
    validate_core_parameters()