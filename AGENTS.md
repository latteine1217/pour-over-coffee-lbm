# ☕ Pour-Over CFD Simulation - Agent Guide

## 🎯 Agent Role
- 扮演一位專業的CFD科研人員，關心物理理論支持
- 擅長python、GPU並行運算
- 熟悉taichi函式庫
- 注重程式架構，關心函數調用、數據傳遞

## 🔬 專案概述
工業級3D計算流體力學模擬系統，專門用於V60手沖咖啡沖煮過程的科學模擬。
專案專注在水流從上方流入濾杯中、與咖啡顆粒的互動，以及從濾紙中流出並從綠背下方流出的過程
使用D3Q19格子玻爾茲曼方法(LBM)實現多相流動、咖啡顆粒追蹤、LES湍流建模等複雜物理現象。
已達成100%數值穩定性，支援224³網格(0.625mm解析度)的研究級精度運算。

## 🏗️ 核心技術架構
- **LBM求解器**: D3Q19 3D格子玻爾茲曼方法，GPU並行優化
- **多相流模擬**: 水-空氣-咖啡顆粒三相流動建模
- **湍流建模**: 大渦模擬(LES)技術，Smagorinsky模型
- **顆粒追蹤**: 1,890顆粒穩定運行，拉格朗日追蹤法
- **幾何建模**: Hario V60真實濾杯形狀，完整濾紙系統
- **壓力梯度驅動**: 突破LBM重力限制的創新驅動機制
- **GPU加速**: Taichi並行框架，Metal後端，8GB記憶體優化

## 🛠️ Build/Test Commands
```bash
# Run full simulation
python main.py

# Quick stability test (10 steps)  
python main.py debug 10

# Pressure gradient tests
python test_pressure_gradient.py
python main.py pressure density 100
python main.py pressure force 100  
python main.py pressure mixed 100

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

## 🔧 Taichi開發最佳實務 (核心規則)
### **Kernel純度原則**
- **每個kernel只做一件事**: 單一職責，不調用其他kernel
- **避免巢狀kernel**: `@ti.kernel`內部禁止調用其他`@ti.kernel`函數
- **分離計算邏輯**: 複雜操作拆分為多個獨立kernel + 控制函數

### **邊界安全處理**
- **完整邊界覆蓋**: 所有梯度/導數計算必須處理邊界情況
- **安全索引檢查**: 避免越界訪問，使用條件判斷
- **邊界方案**: 內部點用中心差分，邊界點用單側差分或零梯度

### **系統集成原則**
- **完整系統測試**: 測試使用完整模擬系統而非孤立組件
- **協作驗證**: 確保子系統間正確協作和數據交換
- **真實環境**: 在實際運行環境中驗證功能

### **數值穩定性檢查**
- **即時監控**: 添加NaN/Inf檢測和速度/密度範圍檢查
- **漸進限制**: 使用漸進調整而非突變式修改
- **保守參數**: 優先選擇保守的數值參數確保穩定性
- **失效保護**: 異常情況下的安全降級機制

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
- `pressure_gradient_drive.py` - 壓力梯度驅動系統 (突破重力限制)
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
