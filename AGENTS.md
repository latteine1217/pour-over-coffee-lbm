# ☕ Pour-Over CFD Simulation - Agent Guide

## 🎯 Agent Role
- 扮演一位專業的CFD科研人員，關心物理理論支持
- 擅長python、GPU並行運算、科研級視覺化分析
- 熟悉taichi函式庫、matplotlib專業繪圖
- 注重程式架構，關心函數調用、數據傳遞、報告生成

## 🔬 專案概述
工業級3D計算流體力學模擬系統，專門用於V60手沖咖啡沖煮過程的科學模擬。
專案專注在水流從上方流入濾杯中、與咖啡顆粒的互動，以及從濾紙中流出並從濾杯下方流出的過程。
使用D3Q19格子玻爾茲曼方法(LBM)實現多相流動、咖啡顆粒追蹤、LES湍流建模等複雜物理現象。
已達成100%數值穩定性，支援224³網格(0.625mm解析度)的研究級精度運算。

## 🏗️ 核心技術架構
- **LBM求解器**: D3Q19 3D格子玻爾茲曼方法，GPU並行優化
- **多相流模擬**: 水-空氣-咖啡顆粒三相流動建模
- **湍流建模**: 大渦模擬(LES)技術，Smagorinsky模型
- **顆粒追蹤**: 1,995顆粒穩定運行，拉格朗日追蹤法
- **幾何建模**: Hario V60真實濾杯形狀，完整濾紙系統
- **壓力梯度驅動**: 突破LBM重力限制的創新驅動機制
- **GPU加速**: Taichi並行框架，Metal後端，8GB記憶體優化
- **🆕 CFD工程師級分析**: 壓力場、湍流特徵、無量綱數、邊界層專業分析
- **🆕 智能報告系統**: 自動時間戳目錄管理 (`report/{timestamp}/`)

## 🛠️ Build/Test Commands
```bash
# 完整模擬運行
python main.py

# 快速穩定性測試 (10步，生成完整CFD報告)
python main.py debug 10

# CFD工程師級專業分析 (5步，快速預覽)
python main.py debug 5

# 簡化版測試運行
python working_main.py

# JAX高性能引擎測試
python jax_hybrid_core.py

# 壓力梯度專業測試
python main.py pressure density 100
python main.py pressure force 100  
python main.py pressure mixed 100

# 單獨測試模組
python tests/test_lbm_diagnostics.py
python tests/test_enhanced_viz.py

# 幾何與視覺化
python examples/conservative_coupling_demo.py
python src/visualization/enhanced_visualizer.py

# 報告系統測試
python tests/test_enhanced_viz.py

# Phase 3強耦合測試
python tests/integration/test_phase3_strong_coupling.py
python tests/integration/test_thermal_integration.py

# 性能基準測試
python benchmarks/benchmark_suite.py
python benchmarks/ultimate_benchmark_suite.py

# 開發工具
python tools/update_imports.py        # 更新import路徑
python tools/fix_config_imports.py    # 修正config導入
python tools/fix_test_imports.py      # 修正測試路徑
```

## 📊 CFD工程師級專業輸出 (新功能)
系統現在生成完整的CFD專業分析報告，包含：

### 🔬 專業分析圖表
- **壓力場分析** (`cfd_pressure_analysis_step_XXXX.png`)
  - 壓力分佈、壓力梯度、壓力係數分析
  - 壓力損失與流動阻力計算
- **湍流特徵分析** (`cfd_turbulence_analysis_step_XXXX.png`)
  - Q-criterion渦流識別、λ2-criterion分析
  - 湍流動能、耗散率計算
- **無量綱數分析** (`cfd_dimensionless_analysis_step_XXXX.png`)
  - Reynolds、Capillary、Bond、Péclet數時序分析
- **邊界層分析** (`cfd_boundary_layer_analysis_step_XXXX.png`)
  - 邊界層厚度、壁面剪應力分析
- **速度場分析** (`velocity_analysis_step_XXXX.png`)
- **V60縱向分析** (`v60_longitudinal_analysis_step_XXXX.png`)
- **綜合多物理場分析** (`combined_analysis_step_XXXX.png`)

### 📁 智能報告目錄結構
```
report/YYYYMMDD_HHMMSS/
├── images/           # CFD專業分析圖片
├── data/            # 數值數據輸出
└── analysis/        # 詳細分析報告
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

### 📁 企業級項目架構
```
pour-over/                    # 🏗️ 根目錄
├── 🚀 main.py               # 主模擬程式，統一控制入口
├── 🔧 working_main.py       # 簡化版主程式 (測試用)
├── ⚡ jax_hybrid_core.py    # JAX混合計算引擎 (高性能)
├── 📄 README.md             # 主要說明文檔
├── 🛠️ requirements.txt     # Python依賴列表
├── src/                     # 📦 核心模組
│   ├── core/                # 🧠 計算引擎 (10個檔案)
│   │   ├── lbm_solver.py          # D3Q19 LBM求解器 (GPU優化核心)
│   │   ├── multiphase_3d.py       # 3D多相流動系統
│   │   ├── strong_coupled_solver.py # Phase 3強耦合求解器
│   │   ├── ultra_optimized_lbm.py # 極致優化LBM核心
│   │   ├── thermal_fluid_coupled.py # 熱流耦合系統
│   │   └── ultimate_cfd_system.py # 終極CFD集成系統
│   ├── physics/             # 🔬 物理模型 (10個檔案)
│   │   ├── coffee_particles.py    # 拉格朗日顆粒追蹤系統
│   │   ├── filter_paper.py        # 濾紙多孔介質建模
│   │   ├── boundary_conditions.py # V60幾何邊界處理
│   │   ├── les_turbulence.py      # LES湍流模擬模組
│   │   ├── precise_pouring.py     # V60注水模式精確控制
│   │   ├── pressure_gradient_drive.py # 壓力梯度驅動系統 (突破重力限制)
│   │   └── thermal_properties.py  # 熱物性參數模組
│   ├── visualization/       # 📊 視覺化系統 (4個檔案)
│   │   ├── visualizer.py          # 即時3D視覺化 (Taichi GUI)
│   │   ├── enhanced_visualizer.py # CFD工程師級科研分析系統 (1,669行)
│   │   ├── lbm_diagnostics.py     # 即時診斷與監控
│   │   └── geometry_visualizer.py # 幾何模型視覺化
│   └── utils/               # 🛠️ 工具函數 (4個檔案)
│       ├── config_validator.py    # 配置驗證工具
│       ├── error_handling.py      # 錯誤處理系統
│       └── data_structure_analysis.py # 數據結構分析
├── config/                  # ⚙️ 配置模組 (3個檔案)
│   ├── config.py            # 科學級穩定參數配置 (工業級調校)
│   ├── thermal_config.py    # 熱流參數配置
│   └── init.py              # 系統初始化設定
├── tests/                   # 🧪 測試系統 (重組)
│   ├── unit/                # 單元測試
│   ├── integration/         # 整合測試 (8個測試)
│   └── benchmarks/          # 性能測試
├── examples/                # 📚 示例演示 (3個檔案)
│   ├── conservative_coupling_demo.py # 保守耦合演示
│   ├── convection_effect_demo.py     # 對流效應演示
│   └── detailed_coupling_demo.py     # 詳細耦合演示
├── benchmarks/              # 🏃 性能基準 (2個檔案)
│   ├── benchmark_suite.py   # 基準測試套件
│   └── ultimate_benchmark_suite.py # 終極性能測試
├── docs/                    # 📖 技術文檔系統
│   ├── mathematical/        # 數學模型推導
│   ├── physics/             # 物理建模細節
│   ├── performance/         # 性能分析報告
│   ├── technical/           # 技術論文
│   ├── THERMAL_DEVELOPMENT_PLAN.md # 熱流發展計畫
│   ├── THERMAL_PHASE3_ANALYSIS.md  # Phase 3分析報告
│   ├── DEPENDENCY_ANALYSIS.md      # 依賴分析
│   └── 技術文檔_完整物理建模.md     # 中文技術文檔
├── tools/                   # 🔧 開發工具
│   ├── update_imports.py    # Import路徑批量更新工具
│   ├── fix_config_imports.py # Config導入修正工具
│   └── fix_test_imports.py  # 測試路徑修正工具
└── backups/                 # 💾 備份檔案
    ├── config_backup_20250724_155949.py # 配置檔案備份
    └── boundary_conditions_backup.py    # 邊界條件備份
```

### 🧠 核心計算引擎 (src/core/)
- `lbm_solver.py` - D3Q19 LBM求解器 (GPU優化核心)
- `multiphase_3d.py` - 3D多相流動系統
- `strong_coupled_solver.py` - Phase 3強耦合求解器
- `ultra_optimized_lbm.py` - 極致優化LBM核心
- `thermal_fluid_coupled.py` - 熱流耦合系統
- `ultimate_cfd_system.py` - 終極CFD集成系統

### 🔬 物理建模系統 (src/physics/)
- `coffee_particles.py` - 拉格朗日顆粒追蹤系統
- `filter_paper.py` - 濾紙多孔介質建模
- `boundary_conditions.py` - V60幾何邊界處理
- `les_turbulence.py` - LES湍流模擬模組
- `precise_pouring.py` - V60注水模式精確控制
- `pressure_gradient_drive.py` - 壓力梯度驅動系統 (突破重力限制)
- `thermal_properties.py` - 熱物性參數模組

### 📊 視覺化與分析 (src/visualization/)
- `visualizer.py` - 即時3D視覺化 (Taichi GUI)
- `enhanced_visualizer.py` - 🆕 CFD工程師級科研分析系統 (1,669行)
  - 壓力場專業分析 (壓力梯度、壓力係數)
  - 湍流特徵分析 (Q-criterion、λ2-criterion)
  - 無量綱數分析 (Reynolds、Capillary、Bond、Péclet)
  - 邊界層分析 (厚度、剪應力)
  - 流動拓撲分析 (分離點、臨界點)
  - 智能報告目錄管理 (`report/{timestamp}/`)
- `lbm_diagnostics.py` - 即時診斷與監控
- `geometry_visualizer.py` - 🆕 幾何模型視覺化工具

### 🧪 專業測試系統 (tests/)
- `unit/` - 單元測試套件
- `integration/` - 整合測試 (包含8個Phase 3測試)
- `benchmarks/` - 性能基準測試

### 📖 企業級文檔系統 (docs/)
- `mathematical/` - 完整數學模型推導
- `physics/` - 物理現象建模細節
- `performance/` - 性能分析與基準測試
- `technical/` - 技術論文與驗證程序

Developed with [opencode](https://opencode.ai) + GitHub Copilot

## 🗂️ 項目組織原則

### 📁 目錄管理規則
- **`src/`** - 所有核心源碼，按功能分模組
- **`config/`** - 統一配置管理，分離關注點
- **`tests/`** - 完整測試體系，unit/integration/benchmarks
- **`docs/`** - 技術文檔集中管理
- **`tools/`** - 開發維護工具集
- **`backups/`** - 重要配置備份，版本控制
- **`examples/`** - 示例代碼與演示
- **根目錄** - 僅保留核心執行檔案和重要文檔

### 🔧 開發工具使用
- **Import更新**: 使用 `tools/update_imports.py` 批量處理
- **配置修正**: 使用 `tools/fix_config_imports.py` 修復引用
- **測試修正**: 使用 `tools/fix_test_imports.py` 處理測試路徑
- **備份管理**: `backups/` 目錄保存重要版本

### 🧪 測試策略
- **Unit Tests**: `tests/unit/` 單元測試
- **Integration Tests**: `tests/integration/` 系統整合測試
- **Performance Tests**: `tests/benchmarks/` 性能基準測試
- **Phase 3測試**: 專門的強耦合系統測試套件

## 角色扮演
在執行專案時，請扮演一個專業CFD工程師的視角來分析程式碼，並給出階段性計畫的建議。熟悉企業級項目架構，理解模組化設計原則。

## Git 規則
- 不要主動git
- 檢查是否存在.gitignore文件
- 被告知上傳至github時先執行```git status```查看狀況
- 上傳至github前請先更新 README.md 文檔

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
