# ☕ Pour-Over CFD Simulation - Agent Guide

## 🎯 Agent Role
請扮演一位專業的CFD科研人員，並同時擅長python、GPU並行運算

## 🔬 專案概述
工業級3D計算流體力學模擬系統，專門用於V60手沖咖啡沖煮過程的科學模擬。使用D3Q19格子玻爾茲曼方法(LBM)實現多相流動、咖啡顆粒追蹤、LES湍流建模等複雜物理現象。已達成100%數值穩定性，支援224³網格(0.625mm解析度)的研究級精度運算。

## 🏗️ 核心技術架構
- **LBM求解器**: D3Q19 3D格子玻爾茲曼方法，GPU並行優化
- **多相流模擬**: 水-空氣-咖啡顆粒三相流動建模
- **湍流建模**: 大渦模擬(LES)技術，Smagorinsky模型
- **顆粒追蹤**: 1,890顆粒穩定運行，拉格朗日追蹤法
- **幾何建模**: Hario V60真實濾杯形狀，完整濾紙系統
- **GPU加速**: Taichi並行框架，Metal後端，8GB記憶體優化

## 🛠️ Build/Test Commands
```bash
# Run full simulation
python main.py

# Quick stability test (10 steps)  
python main.py debug 10

# Single test file
python test_lbm_diagnostics.py

# Geometry visualization
python geometry_visualizer.py

# Research-grade analysis
python enhanced_visualizer.py
```

## 🛡️ 數值穩定性守則 (絕對遵守)
- **NEVER modify**: `SCALE_VELOCITY`, `TAU_*`, `CFL_NUMBER` in config.py
- **Always test**: Run `python main.py debug 10` after any changes
- **Stability first**: Any modification must pass stability verification
- **Core parameters**: 已經過工業級調校，隨意修改將導致數值發散

## 📊 關鍵物理參數
- **計算域**: 224×224×224 (11.2M格點)
- **物理域**: 14.0×14.0×14.0 cm 
- **格子解析度**: 0.625 mm/格點
- **CFL數**: 0.010 (極穩定)
- **Reynolds數**: 基於咖啡沖泡實際條件
- **記憶體需求**: ~2.09 GB

## 📝 Code Style & Standards
- **Imports**: Standard library → third-party → local modules
- **Taichi GPU**: `@ti.data_oriented` classes, `@ti.kernel` functions
- **Documentation**: 中文註解，詳細物理意義說明
- **Type hints**: 關鍵函數簽名必須標註
- **Naming**: snake_case變數/函數, PascalCase類別
- **Error handling**: 數值異常檢測，NaN/Inf防護

## 🏗️ Project Structure & Dependencies  
### 核心執行模組
- `main.py` - 主模擬程式，統一控制入口
- `config.py` - 科學級穩定參數配置 (工業級調校)
- `lbm_solver.py` - D3Q19 LBM求解器 (GPU優化核心)
- `multiphase_3d.py` - 3D多相流動系統
- `coffee_particles.py` - 拉格朗日顆粒追蹤系統
- `les_turbulence.py` - LES湍流模擬模組

### 視覺化與分析
- `visualizer.py` - 即時3D視覺化 (Taichi GUI)
- `enhanced_visualizer.py` - 科研級分析 (matplotlib)
- `geometry_visualizer.py` - 幾何驗證工具

### 專業系統模組
- `precise_pouring.py` - V60注水模式精確控制
- `filter_paper.py` - 濾紙多孔介質建模
- `lbm_diagnostics.py` - 即時診斷與監控

Developed with [opencode](https://opencode.ai) + GitHub Copilot
## Git 規則
- 不要主動git
- 在被告知要建立github repository時，建立.gitignore文件

## markdwon檔案原則（此處不包含AGENTS.md）
- README.md 中必須要標示本專案使用opencode+Github Copilot開發
- 避免建立過多的markdown文件來描述專案
- markdown文件可以多使用emoji來增加豐富度

## 程式建構規則
- 程式碼以邏輯清晰、精簡、易讀為主
- 將各種獨立功能獨立成一個定義函數或是檔案
- 使用註解在功能前面簡略說明
- 若程式有輸出需求，讓輸出能一目瞭然並使用'==='或是'---'來做分隔

## 檔案參考
重要： 當您遇到檔案參考 (例如 @rules/general.md)，請使用你的read工具，依需要載入。它們與當前的 SPECIFIC 任務相關。

### 說明：

- 請勿預先載入所有參考資料 - 根據實際需要使用懶惰載入。
- 載入時，將內容視為覆寫預設值的強制指示
- 需要時，以遞迴方式跟蹤參照