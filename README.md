# ☕ Pour-Over CFD Simulation

> **V60手沖咖啡3D計算流體力學模擬系統**  
> 🔬 基於D3Q19格子玻爾茲曼方法的研究型實現  
> 🛠️ 開發工具：[OpenCode](https://github.com/sst/opencode) + GitHub Copilot  
> ⚗️ **實驗性專案** - 持續開發與驗證中

[![Python](https://img.shields.io/badge/Python-3.10.12-blue.svg)](https://python.org)
[![Taichi](https://img.shields.io/badge/Taichi-1.7.4-green.svg)](https://taichi-lang.org)
[![Development Status](https://img.shields.io/badge/Status-Experimental-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey.svg)]()

## 🎯 專案概述

本專案是一個研究性的3D計算流體力學模擬系統，專注於V60手沖咖啡沖煮過程的物理建模。採用D3Q19格子玻爾茲曼方法，嘗試實現多相流動、咖啡顆粒追蹤、湍流建模等複雜物理現象的數值模擬。

⚠️ **重要提醒**：本專案目前處於實驗開發階段，部分功能可能不穩定或需要進一步驗證。

### 🔬 主要研究目標
- **流體力學建模**: 基於LBM的3D流場計算
- **多相流模擬**: 水-空氣界面動力學研究  
- **顆粒追蹤**: 咖啡顆粒運動軌跡分析
- **湍流現象**: LES大渦模擬實驗
- **熱傳導耦合**: 溫度場與流場的相互作用
- **GPU並行化**: Taichi框架的性能優化探索

## 🚀 使用指南

### 環境要求
- **Python**: 3.10+ (已測試: 3.10.12)
- **Taichi**: 1.7+ (已驗證: 1.7.4)  
- **平台**: macOS (主要開發平台)，Linux (理論支援)
- **記憶體**: 建議 8GB+ RAM
- **GPU**: 可選，Apple Metal 或 NVIDIA CUDA

### 安裝步驟
```bash
# 克隆專案
git clone <repository-url>
cd pour-over

# 安裝依賴
pip install -r requirements.txt
```

### 基本測試
```bash
# 檢查環境
python -c "import taichi as ti; print(f'Taichi {ti.__version__} ready')"

# 基本功能測試 (如果可用)
python main.py debug 5

# 個別模組測試
python lightweight_test.py
```

⚠️ **注意**: 由於專案正在開發中，部分功能可能需要額外配置或除錯。

## 📁 專案架構

```
pour-over/                                                      # 🏗️ 根目錄
├── 📄 README.md                                               # 專案說明文檔
├── 🔧 requirements.txt                                        # Python依賴套件清單
├── ⚙️ pytest.ini                                              # 測試框架配置
├── 🎯 codecov.yml                                             # 代碼覆蓋率配置
├── 📋 .flake8                                                 # Python代碼風格檢查
├── 🔍 .coveragerc                                             # 測試覆蓋率配置
├── 🚀 main.py                                                  # 統一主模擬程式 - 支援熱耦合與壓力驅動
├── ⚡ jax_hybrid_core.py                                       # JAX-Taichi混合計算引擎
├── 🔧 lightweight_test.py                                       # 輕量級測試程式
├── 📝 AGENTS.md                                               # Agent開發指南
├── 📊 GEMINI.md                                               # Gemini模型集成說明
├── 📋 REFACTORING_PLAN.md                                      # 重構計劃文檔
├── 💾 backups/                                                # 重要檔案備份
├── src/                                                       # 📦 核心模組目錄
│   ├── 🧠 core/                                               # 計算引擎核心
│   │   ├── lbm_unified.py                                     # 統一LBM求解器系統
│   │   ├── multiphase_3d.py                                  # 3D多相流系統
│   │   ├── thermal_fluid_coupled.py                          # 🌡️ 熱流弱耦合求解器
│   │   ├── strong_coupled_solver.py                          # Phase 3強耦合求解器
│   │   ├── ultimate_cfd_system.py                            # 集成CFD系統
│   │   ├── apple_silicon_optimizations.py                    # Apple Silicon專用優化
│   │   ├── memory_optimizer.py                               # 記憶體管理最佳化器
│   │   ├── numerical_stability.py                            # 數值穩定性控制器
│   │   ├── lbm_algorithms.py                                 # LBM算法庫
│   │   ├── lbm_protocol.py                                   # LBM協議定義
│   │   ├── backends/                                         # 計算後端系統
│   │   │   ├── compute_backends.py                          # 後端工廠管理器
│   │   │   ├── apple_backend.py                             # Apple Metal後端
│   │   │   ├── cuda_backend.py                              # NVIDIA CUDA後端
│   │   │   └── cpu_backend.py                               # CPU參考後端
│   │   ├── adapters/                                         # 記憶體布局適配器
│   │   │   ├── memory_layouts.py                            # 記憶體布局管理
│   │   │   ├── soa_adapter.py                               # SoA布局適配器
│   │   │   ├── standard_adapter.py                          # 標準布局適配器
│   │   │   └── gpu_adapter.py                               # GPU優化適配器
│   │   └── legacy/                                           # 遺留系統(參考用)
│   │       ├── lbm_solver.py                                # 原始LBM求解器
│   │       ├── ultra_optimized_lbm.py                       # 優化版LBM求解器
│   │       └── cuda_dual_gpu_lbm.py                         # 雙GPU並行LBM
│   ├── 🔬 physics/                                           # 物理模型系統
│   │   ├── coffee_particles.py                               # 咖啡顆粒拉格朗日追蹤
│   │   ├── filter_paper.py                                   # V60濾紙多孔介質
│   │   ├── boundary_conditions.py                            # V60幾何邊界處理
│   │   ├── precise_pouring.py                                # 精確注水系統
│   │   ├── pressure_gradient_drive.py                        # 壓力梯度驅動系統
│   │   ├── les_turbulence.py                                 # LES大渦模擬
│   │   ├── thermal_lbm.py                                    # 熱傳導LBM
│   │   ├── thermal_properties.py                             # 熱物性參數管理
│   │   ├── temperature_dependent_properties.py               # 動態熱物性
│   │   └── buoyancy_natural_convection.py                    # 浮力自然對流
│   ├── 📊 visualization/                                     # 視覺化與分析系統  
│   │   ├── enhanced_visualizer.py                            # CFD工程師級科研分析
│   │   ├── visualizer.py                                     # 統一視覺化管理器
│   │   ├── lbm_diagnostics.py                                # LBM診斷監控系統
│   │   └── geometry_visualizer.py                            # 幾何模型視覺化
│   └── 🛠️ utils/                                             # 工具函數庫
│       ├── config_validator.py                               # 配置參數驗證器
│       ├── error_handling.py                                 # 錯誤處理工具
│       ├── data_structure_analysis.py                        # 資料結構分析工具
│       └── physics_plugin_system.py                          # 物理外掛系統
├── ⚙️ config/                                                # 配置管理系統
│   ├── config.py                                             # 核心CFD參數配置
│   ├── core.py                                               # 核心配置參數
│   ├── physics.py                                            # 物理系統參數
│   ├── thermal.py                                            # 🌡️ 熱流系統參數
│   ├── init.py                                               # Taichi系統初始化
│   ├── config.yaml                                           # YAML配置檔案
│   └── legacy/                                               # 配置系統備份
│       ├── config_original.py                               # 原始配置備份
│       ├── core_config_original.py                          # 核心配置備份
│       ├── physics_config_original.py                       # 物理配置備份
│       └── thermal_config_original.py                       # 熱流配置備份
├── 🧪 tests/                                                 # 全面測試系統 (85%+ 覆蓋率)
│   ├── unit/                                                 # 🔬 單元測試套件
│   ├── integration/                                          # 🔧 整合測試套件
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
│   ├── test_enhanced_viz.py                                  # 增強視覺化測試
│   ├── test_lbm_diagnostics.py                               # 診斷系統測試
│   └── test_visualizer.py                                    # 視覺化系統測試
├── 📚 examples/                                              # 示例演示程式
│   ├── conservative_coupling_demo.py                         # 保守耦合演示
│   ├── convection_effect_demo.py                             # 對流效應演示
│   └── detailed_coupling_demo.py                             # 詳細耦合演示
├── 🏃 benchmarks/                                            # 性能基準測試
│   ├── benchmark_suite.py                                    # 標準性能測試套件
│   └── ultimate_benchmark_suite.py                          # 進階性能測試
├── 📖 docs/                                                  # 技術文檔系統 (53,000+字)
│   ├── 📊 mathematical/                                      # 數學模型完整推導
│   │   └── mathematical_models.md                           # 完整數學方程式推導
│   ├── 🔬 physics/                                           # 物理建模詳細說明
│   │   └── physics_modeling.md                              # 物理現象建模細節
│   ├── 📈 performance/                                       # 性能分析報告
│   │   └── performance_analysis.md                          # 詳細基準測試結果
│   ├── 📄 technical/                                         # 技術論文
│   │   └── technical_paper.md                               # 技術研究論文草稿
│   ├── CI_CD_GUIDE.md                                       # 持續整合/部署指南
│   ├── README.md                                            # 文檔系統導覽
│   └── 技術文檔_完整物理建模.md                               # 中文版完整技術文檔
├── 🔧 tools/                                                 # 開發維護工具
│   ├── update_imports.py                                     # Import路徑批量更新工具
│   ├── fix_config_imports.py                                # 配置導入修正工具
│   └── fix_test_imports.py                                   # 測試路徑修正工具
├── 📊 benchmark_results/                                      # 性能基準測試結果
│   ├── benchmark_results.json                            # 標準基準測試數據
│   └── ultimate_optimization_results.json               # 進階優化測試結果
├── 💾 data/                                               # CFD數據輸出
│   ├── cfd_data_export_step_XXXX.json                   # CFD完整數據導出 (JSON格式)
│   └── cfd_data_export_step_XXXX.npz                    # CFD數據緊湊格式 (NumPy格式)
├── 📊 results/                                            # 模擬結果存檔
│   └── simulation_YYYYMMDD_HHMMSS/                      # 時間戳結果目錄
│       └── statistics_step_XXXXXX.json                  # 統計數據檔案
├── 📋 report/                                             # 專業CFD報告系統
│   ├── YYYYMMDD_HHMMSS/                                 # 時間戳報告目錄
│   │   ├── images/                                      # CFD分析圖片
│   │   │   ├── cfd_pressure_analysis_step_XXXX.png     # 壓力場分析
│   │   │   ├── cfd_turbulence_analysis_step_XXXX.png   # 湍流特徵分析
│   │   │   ├── cfd_dimensionless_analysis_step_XXXX.png # 無量綱數分析
│   │   │   ├── cfd_boundary_layer_analysis_step_XXXX.png # 邊界層分析
│   │   │   ├── velocity_analysis_step_XXXX.png          # 速度場分析
│   │   │   ├── v60_longitudinal_analysis_step_XXXX.png  # V60縱向分析
│   │   │   ├── combined_analysis_step_XXXX.png          # 綜合多物理場分析
│   │   │   └── time_series_analysis_step_XXXX.png       # 時序參數分析
│   │   ├── data/                                        # 數值數據輸出
│   │   │   ├── time_series_data_step_XXXX.json         # 時序數據
│   │   │   └── cfd_data_export_step_XXXX.npz           # CFD數據導出
│   │   └── analysis/                                    # 詳細分析報告
│   └── geometry/                                        # 幾何模型分析報告
│       ├── professional_cross_section_analysis.png      # 專業橫截面分析
│       ├── professional_3d_geometry_model.png           # 3D工程模型
│       ├── engineering_drawings.png                     # 工程製圖
│       ├── coffee_particle_distribution.png             # 咖啡顆粒分布分析
│       └── particle_size_distribution.png               # 顆粒大小分佈分析
└── .github/                                                  # GitHub工作流程
    └── workflows/
        └── ci.yml                                            # CI/CD自動化配置
```

## 🛠️ 開發指引

### 🎯 Build/Test Commands
```bash
# 完整模擬運行
python main.py

# 快速穩定性測試
python main.py debug 5

# 壓力梯度測試
python main.py pressure density 100
python main.py pressure force 100  
python main.py pressure mixed 100

# 單獨測試模組
python tests/test_lbm_diagnostics.py
python tests/test_enhanced_viz.py

# 幾何與視覺化
python examples/conservative_coupling_demo.py
python src/visualization/enhanced_visualizer.py

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

## 🛡️ 開發狀態與限制

### ✅ 已實現功能
- 基本LBM框架搭建
- Taichi GPU並行基礎設施
- 專案架構和模組化設計
- 配置管理系統
- 基礎視覺化功能

### 🚧 開發中功能
- 數值穩定性優化
- 多相流界面處理
- 顆粒-流體耦合算法
- 熱傳導耦合實現
- 湍流模型驗證

### ⚠️ 已知限制
- 部分測試模組需要路徑修正
- 數值穩定性需進一步驗證
- 性能最佳化尚未完成
- 文檔可能與實際實現有差異
- 跨平台兼容性未充分測試

### 🔬 研究方向
- 數值方法穩定性分析
- GPU記憶體布局最佳化
- 物理模型準確性驗證
- 計算效率改進

## 🔬 技術特色

### 🌊 流體力學建模
- **D3Q19 LBM**: 基於格子玻爾茲曼方法的流場計算
- **多相流研究**: 嘗試模擬水-空氣界面動力學
- **邊界條件**: V60濾杯幾何形狀的數值處理

### 🌀 湍流模擬探索
- **LES建模**: 實驗性大渦模擬實現
- **Smagorinsky模型**: 亞格子應力建模嘗試
- **渦流識別**: Q-準則和λ2-準則的數值實驗

### ☕ 顆粒系統
- **拉格朗日追蹤**: 咖啡顆粒運動軌跡計算
- **顆粒-流體耦合**: 雙向作用力模型研究
- **統計分析**: 顆粒分布和運動特性分析

### 🌡️ 熱傳導耦合 (實驗性)
- **溫度場計算**: 熱傳導方程數值求解
- **流熱耦合**: 溫度對流動性質影響的研究
- **自然對流**: 浮力驅動流動的模擬嘗試

## 🍎 平台支援與性能

### 開發環境
- **主要平台**: macOS (Taichi Metal後端)
- **理論支援**: Linux (CUDA), Windows (CUDA/CPU)
- **GPU加速**: Apple Metal, NVIDIA CUDA (未充分測試)
- **記憶體管理**: Taichi統一記憶體模型

### 計算參數 (理論設計)
- **計算域**: 224×224×224 格點
- **物理域**: 14.0×14.0×14.0 cm 
- **解析度**: 0.625 mm/格點
- **時間步長**: 依CFL條件自適應
- **記憶體需求**: 約2-4 GB (視配置而定)

⚠️ **性能聲明**: 實際性能表現需根據具體硬體配置和參數設定進行驗證。

## 📚 參考文獻

1. **格子玻爾茲曼方法**: Chen & Doolen (1998), Kruger et al. (2017)
2. **多相流模型**: Jacqmin (1999), Lee & Fischer (2006)  
3. **湍流建模**: Smagorinsky (1963), Germano et al. (1991)
4. **數值穩定性**: He & Luo (1997), Lallemand & Luo (2000)

## 🤝 貢獻指南

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權協議

本專案採用 MIT 授權協議 - 詳見 [LICENSE](LICENSE) 檔案

## 🙏 致謝

- **OpenCode**: 提供強大的AI輔助開發環境
- **GitHub Copilot**: 智能代碼生成與優化建議
- **Taichi**: 高性能並行計算框架
- **Apple**: Metal GPU計算平台
- **咖啡社群**: 提供V60沖煮技術參考

---

**🔬 專業CFD模擬，☕ 精確咖啡科學 | Powered by OpenCode + GitHub Copilot**