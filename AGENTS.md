# ☕ Pour-Over CFD Simulation - Agent Guide

## 🎯 Agent Role
- 扮演一位專業且資深的CFD科研人員，注重物理理論支持
- 擅長python、GPU並行運算、科研級視覺化分析
- 熟悉taichi函式庫、matplotlib專業繪圖
- 注重程式架構，關心函數調用、數據傳遞、報告生成
仔細思考，只執行我給你的具體任務，用最簡潔優雅的解決方案，盡可能少的修改程式碼

## 語言使用規則
- 平時回覆以及註解撰寫：中文
- 作圖標題、label：英文

## tools使用規則
- 當需要搜尋文件內容時，在shell中使用"ripgrep" (https://github.com/BurntSushi/ripgrep)指令取代grep指令
- 當我使用"@"指名文件時，使用read工具閱讀
- 當需要搜尋文件位置＆名字時，在shell中使用"fd" (https://github.com/sharkdp/fd)指令取代find指令
- 當需要查看專案檔案結構時，在shell中使用"tree"指令

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

## 程式建構準則
**以下順序為建構程式時需要遵循及考慮的優先度**
1. **理論完整度（Theoretical Soundness）**
- 確保數學模型、控制方程式、邊界條件、數值方法都嚴謹且合理。
- 優先驗證模型假設與理論一致性，避免模型本身就偏離物理實際。

2. **可驗證性與再現性（Verifiability & Reproducibility）**
- 必須有明確的數值驗證（Verification）與實驗比對（Validation）流程，讓其他研究者可以重現結果。
- 資料、代碼、參數設定要清楚公開或可存取。

3. **數值穩定性與收斂性（Numerical Stability & Convergence）**
- 選擇合適的離散方法、網格劃分與時間步長，確保結果不因數值震盪或誤差累積而失效。

4. **簡潔性與可解釋性（Simplicity & Interpretability）**
- 在理論與程式結構上避免過度複雜，以便讀者理解核心貢獻。

5. **效能與可擴展性（Performance & Scalability）**
- 如果研究包含大規模計算，需確保程式能在高效能運算環境中平穩運行

## 🛠️ Build/Test Commands
```bash
# 完整模擬運行
python main.py

# 快速穩定性測試 (5步，生成完整CFD報告)
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

## 📊 CFD工程師級專業輸出
系統要能夠生成完整的CFD專業分析報告，格式如論文風格，包含：

### 🔬 分析圖表
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

### 📁 報告目錄結構
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
- **請注意**:在 Taichi kernel 中不能調用其他 kernel

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
pour-over/                                                      # 🏗️ 根目錄
├── 🚀 main.py                                                  # 統一主模擬程式 - 支援熱耦合與壓力驅動模式 (1254行)
├── 🔧 lightweight_test.py                                       # 輕量級測試程式 - 快速驗證與開發測試 (174行)
├── ⚡ jax_hybrid_core.py                                       # JAX-Taichi混合超級計算引擎 - XLA編譯器最佳化 (308行)
├── 📄 README.md                                               # 主要說明文檔 - 完整專案介紹與使用指南
├── 🛠️ requirements.txt                                        # Python依賴套件清單
├── ⚙️ pytest.ini                                              # 測試框架配置檔案
├── 🔍 codecov.yml                                             # 代碼覆蓋率配置
├── 📋 .flake8                                                 # Python代碼風格檢查配置
├── 🎯 .coveragerc                                            # 測試覆蓋率配置
├── src/                                                       # 📦 核心模組目錄
│   ├── 🧠 core/                                               # 計算引擎核心
│   │   ├── lbm_solver.py                                      # D3Q19格子玻爾茲曼求解器 - 3D流體力學核心
│   │   ├── ultra_optimized_lbm.py                            # 超級優化版LBM - Apple Silicon深度優化 (SoA布局)
│   │   ├── multiphase_3d.py                                  # 3D多相流系統 - 水-空氣界面動力學 (Cahn-Hilliard方程)
│   │   ├── thermal_fluid_coupled.py                          # 🌡️ 熱流弱耦合求解器 - Phase 2流體→熱傳單向耦合
│   │   ├── strong_coupled_solver.py                          # Phase 3強耦合求解器 - 完全雙向熱流耦合
│   │   ├── ultimate_cfd_system.py                            # 終極CFD集成系統 - 企業級整合解決方案
│   │   ├── apple_silicon_optimizations.py                    # Apple Silicon專用優化 - Metal GPU加速
│   │   ├── cuda_dual_gpu_lbm.py                              # NVIDIA雙GPU並行LBM - P2P記憶體最佳化
│   │   ├── memory_optimizer.py                               # 記憶體管理最佳化器 - 大規模網格支援
│   │   ├── numerical_stability.py                            # 數值穩定性控制器 - 100%收斂保證
│   │   ├── lbm_protocol.py                                   # LBM協議定義 - 統一介面標準
│   │   └── __init__.py                                       # 核心模組初始化
│   ├── 🔬 physics/                                           # 物理模型系統
│   │   ├── coffee_particles.py                               # 咖啡顆粒拉格朗日追蹤 - 1,995顆粒穩定運行
│   │   ├── filter_paper.py                                   # V60濾紙多孔介質 - Darcy定律與動態阻力
│   │   ├── boundary_conditions.py                            # V60幾何邊界處理 - 完整濾杯建模
│   │   ├── precise_pouring.py                                # 精確注水系統 - 0.5cm直徑垂直水流
│   │   ├── pressure_gradient_drive.py                        # 壓力梯度驅動系統 - 突破LBM重力限制
│   │   ├── les_turbulence.py                                 # LES大渦模擬 - Smagorinsky湍流模型
│   │   ├── thermal_lbm.py                                    # 熱傳導LBM - 溫度場求解器
│   │   ├── thermal_properties.py                             # 熱物性參數管理 - 溫度相依性質
│   │   ├── temperature_dependent_properties.py               # 動態熱物性 - 實時溫度耦合
│   │   ├── buoyancy_natural_convection.py                    # 浮力自然對流 - 溫差驅動流動
│   │   └── __init__.py                                       # 物理模組初始化
│   ├── 📊 visualization/                                     # 視覺化與分析系統  
│   │   ├── enhanced_visualizer.py                            # CFD工程師級科研分析 - 7種專業視覺化模式 (1669行)
│   │   ├── visualizer.py                                     # 統一視覺化管理器 - 即時3D監控
│   │   ├── lbm_diagnostics.py                                # LBM診斷監控系統 - 數值品質與守恆定律檢查
│   │   ├── geometry_visualizer.py                            # 幾何模型視覺化 - V60濾杯專業製圖
│   │   └── __init__.py                                       # 視覺化模組初始化
│   ├── 🛠️ utils/                                             # 工具函數庫
│   │   ├── config_validator.py                               # 配置參數驗證器 - CFD參數一致性檢查
│   │   ├── error_handling.py                                 # 企業級錯誤處理 - 異常捕獲與恢復
│   │   ├── data_structure_analysis.py                        # 資料結構分析工具 - 記憶體布局最佳化
│   │   ├── physics_plugin_system.py                          # 物理外掛系統 - 模組化擴展框架
│   │   └── __init__.py                                       # 工具模組初始化
│   └── __init__.py                                           # 源碼模組初始化
├── ⚙️ config/                                                # 配置管理系統
│   ├── config.py                                             # 核心CFD參數配置 - 科學級穩定參數 (工業級調校)
│   ├── thermal_config.py                                     # 🌡️ 熱流系統參數 - 溫度邊界條件與物性
│   ├── init.py                                               # Taichi系統初始化 - GPU後端配置
│   └── __init__.py                                           # 配置模組初始化
├── 🧪 tests/                                                 # 全面測試系統 (85%+ 覆蓋率)
│   ├── unit/                                                 # 🔬 單元測試套件
│   ├── integration/                                          # 🔧 整合測試套件 (8個專業測試)
│   │   ├── test_thermal_integration.py                       # 熱耦合系統整合測試
│   │   ├── test_phase3_strong_coupling.py                    # Phase 3強耦合完整測試  
│   │   ├── test_weak_coupling.py                             # 弱耦合系統測試
│   │   └── enhanced_pressure_test.py                         # 增強壓力梯度測試
│   ├── benchmarks/                                           # 📈 性能基準測試
│   ├── test_lbm_solver_unit.py                               # LBM求解器單元測試
│   ├── test_multiphase_flow.py                               # 多相流測試
│   ├── test_coffee_particles_extended.py                     # 擴展顆粒系統測試
│   ├── test_filter_paper.py                                  # 濾紙系統測試
│   ├── test_precise_pouring.py                               # 注水系統測試
│   ├── test_boundary_conditions.py                           # 邊界條件測試
│   ├── test_pressure_gradient.py                             # 壓力梯度測試
│   ├── test_les_turbulence.py                                # LES湍流測試
│   ├── test_numerical_stability.py                           # 數值穩定性測試
│   ├── test_enhanced_viz.py                                  # 增強視覺化測試 (報告系統)
│   ├── test_lbm_diagnostics.py                               # 診斷系統測試
│   ├── test_visualizer.py                                    # 視覺化系統測試
│   ├── README.md                                             # 測試系統使用說明
│   └── __init__.py                                           # 測試模組初始化
├── 📚 examples/                                              # 示例演示程式
│   ├── conservative_coupling_demo.py                         # 保守耦合演示 - 專業幾何分析視覺化
│   ├── convection_effect_demo.py                             # 對流效應演示 - 熱傳導效應
│   ├── detailed_coupling_demo.py                             # 詳細耦合演示 - 完整物理耦合
│   └── __init__.py                                           # 示例模組初始化
├── 🏃 benchmarks/                                            # 性能基準測試
│   ├── benchmark_suite.py                                    # 標準性能測試套件 - 系統性能評估
│   ├── ultimate_benchmark_suite.py                          # 終極性能測試 - 極限負載測試
│   └── __init__.py                                           # 基準測試模組初始化
├── 📖 docs/                                                  # 技術文檔系統 (53,000+字)
│   ├── 📊 mathematical/                                      # 數學模型完整推導
│   │   └── mathematical_models.md                           # 完整數學方程式推導 (255+方程)
│   ├── 🔬 physics/                                           # 物理建模詳細說明
│   │   └── physics_modeling.md                              # 物理現象建模細節
│   ├── 📈 performance/                                       # 性能分析報告
│   │   └── performance_analysis.md                          # 詳細基準測試結果
│   ├── 📄 technical/                                         # 技術論文
│   │   └── technical_paper.md                               # 期刊級研究論文
│   ├── THERMAL_DEVELOPMENT_PLAN.md                          # 🌡️ 熱流系統開發路線圖
│   ├── THERMAL_PHASE3_ANALYSIS.md                           # Phase 3強耦合系統分析報告
│   ├── DEPENDENCY_ANALYSIS.md                               # 系統依賴關係分析
│   ├── CI_CD_GUIDE.md                                       # 持續整合/部署指南
│   ├── README.md                                            # 文檔系統導覽
│   └── 技術文檔_完整物理建模.md                               # 中文版完整技術文檔
├── 🔧 tools/                                                 # 開發維護工具
│   ├── update_imports.py                                     # Import路徑批量更新工具
│   ├── fix_config_imports.py                                # 配置導入修正工具
│   └── fix_test_imports.py                                   # 測試路徑修正工具
├── 💾 backups/                                               # 重要檔案備份
│   ├── config_backup_20250724_155949.py                     # 配置檔案時間戳備份
│   └── boundary_conditions_backup.py                        # 邊界條件備份
└── .github/                                                  # GitHub工作流程
    └── workflows/
        └── ci.yml                                            # CI/CD自動化配置
```

### 🧠 核心計算模組 (src/core/)

**主要求解器架構**
- **`lbm_solver.py`** - D3Q19格子玻爾茲曼基礎求解器 (1400+行)
  - 🔬 3D流體力學核心引擎，採用D3Q19離散速度模型
  - 🌀 集成LES湍流建模 (Smagorinsky模型)，支援高Reynolds數流動
  - 💧 多相流支援，處理水-空氣界面動力學
  - ⚡ GPU並行優化，企業級錯誤監控
  
- **`ultra_optimized_lbm.py`** - 超級優化版求解器 (800+行)
  - 🚀 針對Apple Silicon的終極優化，採用真正SoA布局
  - 🍎 Metal GPU深度優化，cache-line對齊設計  
  - ⚡ 統一記憶體零拷貝技術，SIMD vectorization友好
  - 📈 極致性能：159M+ 格點/秒運算能力

- **`multiphase_3d.py`** - 3D多相流系統 (600+行)
  - 🌊 基於Cahn-Hilliard相場方程的科學級實現
  - 💧 完整水-空氣界面動力學，表面張力精確計算
  - 📐 連續表面力模型，界面法向量與曲率場計算
  - 🔬 參考Jacqmin(1999)、Lee & Fischer(2006)理論

**熱流耦合求解器**
- **`thermal_fluid_coupled.py`** - Phase 2弱耦合系統 (400+行)
  - 🌡️ 流體→熱傳單向耦合，交替求解策略
  - 📊 時序協調控制，數值穩定性保證
  - ⚡ 性能監控與診斷功能
  
- **`strong_coupled_solver.py`** - Phase 3強耦合系統 (600+行)
  - 🔥 完全雙向熱流耦合，溫度-流動實時交互
  - 🌡️ 溫度相依流體性質，浮力驅動自然對流
  - 📈 數值穩定性控制，企業級錯誤處理

**系統優化與支援**
- **`ultimate_cfd_system.py`** - 終極CFD集成系統
  - 🏗️ 企業級整合解決方案，統一多物理場管理
  - 📊 智能負載平衡，自適應計算資源分配
  
- **`apple_silicon_optimizations.py`** - Apple Silicon專用優化
  - 🍎 M1/M2/M3晶片檢測與配置，Metal GPU最大化利用
  - ⚡ 記憶體帶寬最佳化，統一記憶體架構充分發揮
  
- **`cuda_dual_gpu_lbm.py`** - NVIDIA雙GPU並行
  - 🔥 P100雙GPU直接記憶體存取，P2P高速通訊
  - 📈 負載動態平衡，最大化並行效能

### 📊 視覺化與分析系統 (src/visualization/)

**CFD工程師級科研視覺化**
- **`enhanced_visualizer.py`** - 科研級增強視覺化系統 (1669行)
  - 🔬 7種專業CFD分析模式：壓力場、湍流特徵、無量綱數、邊界層、流動拓撲
  - 📈 **壓力場分析**：壓力梯度場、壓力係數分布、損失計算與可視化
  - 🌀 **湍流特徵分析**：Q-準則渦流識別、λ2-準則分析、湍流動能時空分布
  - 📐 **無量綱數分析**：Reynolds、Capillary、Bond、Péclet數即時追蹤與時序圖
  - 🌊 **邊界層分析**：厚度計算、壁面剪應力分布、流動分離點檢測
  - 🎯 **流動拓撲分析**：臨界點識別、分離線追蹤、渦流結構可視化
  - 🗂️ **智能報告管理**：自動時間戳目錄 `report/{timestamp}/` 結構化輸出
  - 📊 **多物理場綜合分析**：溫度-流動-壓力-相場統一視覺化
  
**專業診斷與監控**
- **`lbm_diagnostics.py`** - LBM診斷監控系統 (500+行)
  - 🔍 **數值品質監控**：時間穩定性、守恆定律檢驗、收斂性分析
  - 📊 **統計數據追蹤**：循環緩衝區歷史數據管理，多層次監控頻率
  - ⚡ **即時性能分析**：計算效率、記憶體使用、GPU利用率監控
  - 🛡️ **穩定性預警**：異常檢測、發散預測、自動降級機制
  - 📈 **專業CFD指標**：CFL數、Mach數、Reynolds數動態追蹤

**統一視覺化管理**
- **`visualizer.py`** - 統一視覺化管理器 (400+行)
  - 🎮 即時3D監控界面，Taichi GUI直接渲染
  - 🔄 多場統計數據獲取，標準化視覺化接口
  - 📊 基礎視覺化功能：密度場、速度場、相場切面顯示

**幾何模型專業製圖**
- **`geometry_visualizer.py`** - 幾何模型視覺化工具 (300+行)
  - 📐 V60濾杯專業製圖：尺寸標註、間隙分析、流體路徑
  - ☕ 咖啡顆粒3D分布：散點圖、密度熱圖、統計分析
  - 📊 顆粒大小分佈：直方圖、累積分佈、正態性檢驗
  - 🏗️ 工程製圖標準：專業標註、材料特性、裝配關係

### 🧪 全面測試系統 (tests/ - 85%+覆蓋率)

**單元測試套件 (tests/unit/)**
- 🔬 獨立模組功能驗證，確保每個組件正確性
- ⚡ 快速反饋機制，開發過程即時品質保證

**整合測試套件 (tests/integration/ - 4個核心測試)**
- **`test_thermal_integration.py`** - 熱耦合系統完整測試
- **`test_phase3_strong_coupling.py`** - Phase 3強耦合端到端驗證
- **`test_weak_coupling.py`** - 弱耦合系統穩定性測試
- **`enhanced_pressure_test.py`** - 增強壓力梯度系統驗證

**專業模組測試**
- **`test_lbm_solver_unit.py`** - LBM求解器單元測試，數值方法驗證
- **`test_multiphase_flow.py`** - 多相流系統測試，界面動力學驗證
- **`test_coffee_particles_extended.py`** - 擴展顆粒系統測試
- **`test_filter_paper.py`** - 濾紙系統多孔介質流動測試
- **`test_precise_pouring.py`** - 注水系統精度測試
- **`test_boundary_conditions.py`** - 邊界條件正確性測試
- **`test_pressure_gradient.py`** - 壓力梯度驅動效能測試
- **`test_les_turbulence.py`** - LES湍流模型驗證
- **`test_numerical_stability.py`** - 數值穩定性系統性測試
- **`test_enhanced_viz.py`** - 增強視覺化報告系統測試
- **`test_lbm_diagnostics.py`** - 診斷系統功能測試
- **`test_visualizer.py`** - 基礎視覺化系統測試

**性能基準測試 (tests/benchmarks/)**
- 📈 系統性能基準建立，回歸檢測機制
- ⚡ 多平台性能對比，優化效果量化評估

### 📚 示例演示與基準測試

**專業示例程式 (examples/)**
- **`conservative_coupling_demo.py`** - 保守耦合演示 (400+行)
  - 🎨 專業幾何分析視覺化：V60濾杯完整建模與尺寸驗證
  - ☕ 咖啡顆粒分布分析：3D散點圖、密度熱圖、統計特性
  - 📊 顆粒大小分佈：正態性檢驗、累積分佈、分層統計
  - 🏗️ 工程製圖展示：專業標註、間隙分析、流體路徑
  
- **`convection_effect_demo.py`** - 對流效應演示
  - 🌡️ 熱傳導效應可視化，溫度梯度驅動流動
  - 🔥 自然對流現象展示，浮力效應分析
  
- **`detailed_coupling_demo.py`** - 詳細耦合演示  
  - 🔗 完整物理耦合過程展示
  - 📊 多物理場交互分析，耦合效應量化

**性能基準測試 (benchmarks/)**
- **`benchmark_suite.py`** - 標準性能測試套件 (300+行)
  - 📈 系統性能評估：計算速度、記憶體效率、數值精度
  - 🎯 多場景基準：不同網格尺寸、顆粒數量、物理複雜度
  - 📊 跨平台對比：CPU vs GPU、不同硬體配置效能
  
- **`ultimate_benchmark_suite.py`** - 終極性能測試 (400+行)
  - 🚀 極限負載測試：最大網格、最多顆粒、最複雜物理
  - 💪 穩定性壓力測試：長時間運行、記憶體洩漏檢測
  - 🔥 性能最佳化驗證：Apple Silicon vs NVIDIA vs CPU對比

### 📖 技術文檔系統 (docs/ - 53,000+字)
- **`mathematical/`** - 完整數學模型推導 (255+方程式)
- **`physics/`** - 物理建模細節與理論基礎
- **`performance/`** - 性能分析與基準測試報告
- **`technical/`** - 期刊級研究論文與驗證程序
- **`THERMAL_DEVELOPMENT_PLAN.md`** - 🌡️ 熱流系統開發路線圖
- **`THERMAL_PHASE3_ANALYSIS.md`** - Phase 3強耦合系統分析報告
- **`技術文檔_完整物理建模.md`** - 中文版完整技術文檔

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
## 開發者指引 👨‍💻

### 📋 任務執行流程
1. **📖 需求分析**: 仔細理解用戶需求，識別技術關鍵點
2. **🏗️ 架構設計**: 優先制定階段性實現方案，考慮擴展性和維護性
3. **分析步驟**：分析實現方案所需之具體步驟，確定執行方式
4. **👨‍💻 編碼實現**: 遵循專案規範，撰寫高品質程式碼
5. **🧪 測試驗證**: 撰寫單元測試，確保功能正確性
6. **📝 文檔更新**: 更新相關文檔，包括 README、API 文檔等
7. **🔍 程式碼審查**: 自我檢查程式碼品質，確保符合專案標準

### ⚠️ 重要提醒
- **🚫 避免破壞性變更**: 保持向後相容性，漸進式重構
- **📁 檔案參考**: 遇到 `@filename` 時使用 Read 工具載入內容
- **🔄 懶惰載入**: 按需載入參考資料，避免預先載入所有檔案
- **💬 回應方式**: 優先提供計畫和建議，除非用戶明確要求立即實作


## 程式構建指引
### Git 規則
- 不要主動git
- 檢查是否存在.gitignore文件
- 被告知上傳至github時先執行```git status```查看狀況
- 上傳至github前請先更新 @README.md 文檔


### markdwon檔案原則（此處不包含AGENTS.md）
- README.md 中必須要標示本專案使用opencode+Github Copilot開發
- 說明檔案請盡可能簡潔明瞭
- 避免建立過多的markdown文件來描述專案
- markdown文件可以多使用emoji以及豐富排版來增加豐富度
- 用字請客觀且中性，不要過度樂觀以及膚淺

### 程式規則
- 程式碼以邏輯清晰、精簡、易讀、高效這四點為主
- 將各種獨立功能獨立成一個定義函數或是api檔案，並提供api文檔
- 各api檔案需要有獨立性，避免循環嵌套
- 盡量避免大於3層的迴圈以免程式效率低下
- 使用註解在功能前面簡略說明
- 若程式有輸出需求，讓輸出能一目瞭然並使用'==='或是'---'來做分隔

## 說明：

- 請勿預先載入所有參考資料 - 根據實際需要使用懶惰載入。
- 載入時，將內容視為覆寫預設值的強制指示
- 需要時，以遞迴方式跟蹤參照
