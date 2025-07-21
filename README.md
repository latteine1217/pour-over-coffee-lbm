# ☕ 手沖咖啡3D流體模擬 - LBM方法

> 使用3D Lattice Boltzmann Method (LBM) 模擬真實手沖咖啡過程的完整流體動力學  
> 🤖 本專案由 **opencode + GitHub Copilot** 開發

## 📋 專案簡介

這個專案使用3D計算流體力學(CFD)方法來完整模擬V60手沖咖啡過程，重點關注：
- 💧 3D多相流體（空氣-水界面）
- 🌊 D3Q19 LBM核心算法  
- 🔺 Hario V60濾杯真實幾何
- ☕ 咖啡粉層3D多孔介質
- 🌡️ 90°C熱水物理特性
- 📊 即時3D視覺化

## 🚀 快速開始

### 運行模擬
```bash
python main.py                    # 完整3D模擬
python main.py interactive        # 交互式模式
python test_simple.py             # 基本功能測試
```

### 使用增強版可視化
```python
from enhanced_visualizer import EnhancedVisualizer
from lbm_solver import LBMSolver

lbm = LBMSolver()
enhanced_viz = EnhancedVisualizer(lbm)

# 生成專業分析圖
enhanced_viz.save_longitudinal_analysis(time, step)  # 縱向剖面
enhanced_viz.save_velocity_analysis(time, step)      # 流速分析  
enhanced_viz.save_combined_analysis(time, step)      # 綜合分析
```

### 基本視覺化選項
```python
from visualizer import UnifiedVisualizer
from lbm_solver import LBMSolver

lbm = LBMSolver()
vis = UnifiedVisualizer(lbm)

# 不同視覺化模式
vis.display_gui('composite')    # 綜合視圖（推薦）
vis.display_gui('density')      # 密度場
vis.display_gui('velocity')     # 速度場  
vis.display_gui('phase')        # 相場
```

## 🏗️ 技術架構

### 核心模組
- **`lbm_solver.py`** - D3Q19 3D LBM求解器
- **`multiphase_3d.py`** - 3D多相流處理（相場方法）
- **`porous_media_3d.py`** - 3D多孔介質（咖啡粉層）
- **`coffee_particles.py`** - 咖啡粒子追蹤系統
- **`precise_pouring.py`** - 精確注水控制
- **`visualizer.py`** - 3D即時視覺化
- **`enhanced_visualizer.py`** - 🆕 增強版可視化分析
- **`config.py`** - 所有模擬參數

### 物理模型
- **LBM**: D3Q19格子模型，BGK碰撞算子，Guo重力源項
- **多相流**: 3D相場方法追蹤空氣-水界面
- **表面張力**: 3D Laplace壓力模型
- **多孔介質**: 3D Darcy流動 + 修正碰撞過程
- **邊界條件**: 3D反彈邊界、入水/出水邊界
- **粒子系統**: 50,000個可移動咖啡粒子

## ⚙️ 真實物理參數

### Hario V60實際規格 (基於網路數據)
| 參數 | 值 | 說明 |
|------|----|----|
| 外部尺寸 | 138×115×95mm | 長×寬×高 (含外壁) |
| 內部高度 | 85mm | 扣除壁厚的有效高度 |
| 內部上徑 | 111mm | 扣除壁厚的有效直徑 |
| 下部出水孔 | 4mm | V60標準出水孔 |
| 實際錐角 | 64.4度 | 基於幾何計算 |
| 內部體積 | 284.4cm³ | 有效沖泡空間 |

### 90°C熱水物理特性
| 參數 | 值 | 說明 |
|------|----|----|
| 水溫 | 90°C | 手沖標準溫度 |
| 密度 | 965.3 kg/m³ | 比常溫水輕3.5% |
| 運動粘滯度 | 3.15×10⁻⁷ m²/s | 約為常溫水的1/3 |
| 注水速度 | 4 ml/s | 標準手沖速度 |
| 注水高度 | 12.5 cm | 典型手沖高度 |

### 手沖咖啡參數 (更新)
| 參數 | 值 | 說明 |
|------|----|----|
| 網格尺寸 | 64×64×128 | 3D格子單位 |
| 總萃取時間 | 2:20 | 140秒標準時間 |
| 咖啡豆密度 | 1.2 g/cm³ | 中烘焙密度 |
| 咖啡粉用量 | 20g | 標準手沖份量 |
| 咖啡顆粒數 | 493,796個 | 基於手沖研磨度分布 |
| 主體粒徑 | 0.65mm | 二號砂糖大小 |
| 咖啡床高度 | 3.3cm | 合理堆積高度 |
| 咖啡床體積 | 85.3cm³ | V60的30%填充 |
| 孔隙率 | 80.5% | 實際手沖研磨孔隙率 |
| 水粉比 | 16:1 | 320ml總注水量 |

## 📊 輸出說明

### 🆕 增強版可視化分析
程式現在支持三種專業級分析圖：

#### 1. 縱向剖面圖 (Longitudinal Cross-Section) 🆕 增強V60幾何
- **目的**: 顯示水流從上到下的完整過程，**正確呈現V60錐形濾杯**
- **內容**: 4個子圖展示密度分布、流速大小、垂直流速分量、流線圖
- **🔺 V60幾何特色**: 
  - 清晰的錐形濾杯輪廓 (60度錐角)
  - 濾紙邊界線標識
  - 咖啡床區域顯示 (棕色虛線)
  - 出水口位置 (紅色粗線)
  - 注水區域指示 (青色箭頭)
  - 重力方向箭頭
- **檔名**: `v60_longitudinal_analysis_step_XXXX.png`

#### 2. 流速分析圖 (Velocity Analysis)  
- **目的**: 深入分析水在濾杯中的流速特性
- **內容**: 不同高度流速分布、統計柱狀圖、節點分析、性能總結
- **特色**: 類似`test_velocity_field.png`的高質量可視化
- **檔名**: `velocity_analysis_step_XXXX.png`

#### 3. 綜合分析圖 (Combined Analysis) 🆕 V60完整視圖
- **目的**: 結合密度和速度的完整概覽，**突出V60錐形幾何**
- **內容**: 大幅縱向剖面圖 + 速度統計 + 流域分析 + 總結信息
- **🔺 V60視覺化**: 主圖清晰顯示錐形濾杯、咖啡床、濾紙、出水口
- **特色**: 一頁展示所有關鍵信息，圖像中心就是V60濾杯
- **檔名**: `combined_analysis_step_XXXX.png`

### 3D視覺化顏色編碼
- **紅色**: 水相密度（越紅水越多）
- **綠色**: 流速大小（越亮速度越快）  
- **藍色**: 相場（氣水界面）

### 數據輸出
- **🆕 專業分析圖**: `*_analysis_step_XXXX.png` (300 DPI高分辨率)
- 自動保存快照: `coffee_sim_3d_XXXXXX.png`
- 實時進度監控  
- 性能統計信息
- 3D切片視覺化

## 🔧 調整參數

編輯 `config.py` 來修改模擬參數：

```python
# 網格解析度
NX = 64               # 增加以提高精度 (需要更多記憶體)
NY = 64
NZ = 128

# 注水參數  
POUR_RATE_ML_S = 4.0  # 注水速度 ml/s
BREWING_TIME_SECONDS = 140  # 萃取時間

# 物理參數
TAU_WATER = 0.500230  # 基於90°C水粘滯度
GRAVITY_LU = 0.000153 # 真實重力轉換
```

## 📁 檔案結構

```
pour-over/
├── main.py              # 主程序 (3D專用)
├── config.py            # 參數配置
├── lbm_solver.py        # 3D LBM核心 (D3Q19)
├── multiphase_3d.py     # 3D多相流
├── porous_media_3d.py   # 3D多孔介質
├── coffee_particles.py  # 咖啡粒子系統
├── precise_pouring.py   # 精確注水
├── visualizer.py        # 3D視覺化
├── enhanced_visualizer.py # 🆕 增強版分析圖
├── init.py              # 初始化工具
├── test_simple.py       # 基本測試
├── test_enhanced_visualization.py # 🆕 增強版測試
├── test_v60_visualization.py # 🆕 V60幾何測試
├── demo_enhanced_output.py # 🆕 功能演示
└── AGENTS.md           # AI代理指南
```

## 🎯 未來發展

- [ ] 熱傳導和溫度場模擬
- [ ] 咖啡萃取動力學模型
- [ ] 實驗數據驗證
- [ ] 不同濾杯形狀支援
- [ ] GPU性能進一步優化
- [ ] 機器學習優化萃取參數

## 🚀 技術特色

- **🆕 真實V60幾何**: 正確顯示錐形濾杯輪廓、咖啡床、濾紙邊界、出水口
- **🆕 專業可視化**: 縱向剖面圖和流速分析，採用科學出版品質標準
- **真實物理**: 基於90°C熱水的真實物理參數
- **高性能**: Taichi GPU加速，支援Apple Silicon
- **數值穩定**: 經過優化的D3Q19 LBM算法
- **可視化**: 實時3D切片和統計監控
- **完整模擬**: 從注水到萃取的完整流程

## 🤝 貢獻

歡迎提交Issue和Pull Request！

## 📝 授權

MIT License

---

*"好咖啡來自於對細節的極致追求" - 包含3D流體力學* ☕✨