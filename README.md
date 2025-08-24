├── 📊 benchmark_results/                                      # 性能基準測試結果
│   ├── benchmark_results.json                            # 標準基準測試數據
│   └── ultimate_optimization_results.json               # 極致優化測試結果
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
│       └── particle_size_distribution.png               # 顆粒大小分佈分析├── 🔧 tools/                                                 # 開發維護工具
│   ├── update_imports.py                                     # Import路徑批量更新工具
│   ├── fix_config_imports.py                                # 配置導入修正工具
│   └── fix_test_imports.py                                   # 測試路徑修正工具
└── .github/                                                  # GitHub工作流程
    └── workflows/
        └── ci.yml                                            # CI/CD自動化配置
```
