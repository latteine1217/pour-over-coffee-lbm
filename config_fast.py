# config_fast.py
"""
快速執行模式配置 - 大幅提升性能用於開發和測試
針對速度優化，犧牲部分精度
"""

# 導入基礎配置
from config import *

# ===========================
# 性能優化覆蓋設定
# ===========================

# 1. 大幅減少網格尺寸 (15.8倍加速)
NX = 64          # 從160降到64 
NY = 64          # 從160降到64
NZ = 64          # 從160降到64

# 2. 減少粒子數量 (10倍加速)
FAST_PARTICLES_PER_LAYER = 15  # 從150降到15

# 3. 減少最大步數
MAX_STEPS = 2000       # 從10000降到2000
MAX_STEPS_DEMO = 1000  # 演示模式

# 4. 增加輸出頻率 (減少I/O開銷)
OUTPUT_FREQ = 100      # 從200降到100

# 5. 簡化物理計算
SURFACE_TENSION = 0.01  # 降低表面張力計算複雜度
INTERFACE_THICKNESS = 2 # 從3降到2

# ===========================
# 重新計算相關參數
# ===========================

# 更新長度比例
SCALE_LENGTH = CUP_HEIGHT / (48)  # 調整到64格中的48格有效區域

# 更新咖啡床高度
COFFEE_BED_HEIGHT_LU = int(COFFEE_BED_HEIGHT_PHYS / SCALE_LENGTH)

print("🚀 快速模式配置載入:")
print(f"   網格: {NX}³ = {NX*NY*NZ:,} 節點 (vs 160³ = 4,096,000)")
print(f"   粒子: ~{FAST_PARTICLES_PER_LAYER*50:,} 個 (vs ~7,500)")
print(f"   預期加速: ~25倍")
print(f"   記憶體需求: ~0.8GB (vs 4GB)")