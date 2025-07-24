# 📚 Pour-Over CFD Simulation - 論文級技術文檔

## 🎯 文檔架構概覽

本目錄包含V60手沖咖啡LBM模擬系統的完整學術級技術文檔，適用於同行評議、學術發表和深度技術研究。

### 📁 文檔組織結構

```
docs/
├── technical/          # 核心技術論文
├── mathematical/       # 數學模型與公式
├── physics/            # 物理建模詳述  
├── numerical/          # 數值方法實現
├── performance/        # 性能分析報告
├── validation/         # 驗證與測試
├── tutorials/          # 使用教程指南
├── figures/            # 圖表與視覺化
├── references/         # 參考文獻
└── 🌡️ thermal/        # 熱流耦合系統文檔 (新增)
```

### 📖 主要文檔列表

#### 🔬 **核心技術文檔**
- [`technical_paper.md`](technical/technical_paper.md) - 主要技術論文
- [`mathematical_models.md`](mathematical/mathematical_models.md) - 數學模型詳述
- [`physics_modeling.md`](physics/physics_modeling.md) - 物理建模原理

#### 🌡️ **熱流耦合系統** (新增)
- [`THERMAL_DEVELOPMENT_PLAN.md`](THERMAL_DEVELOPMENT_PLAN.md) - 熱流開發計畫與完成狀態
- [`THERMAL_PHASE3_ANALYSIS.md`](THERMAL_PHASE3_ANALYSIS.md) - Phase 3強耦合技術分析
- 熱流耦合數學模型與實現細節

#### 📊 **實現與分析**
- [`numerical_methods.md`](numerical/numerical_methods.md) - 數值方法實現
- [`performance_analysis.md`](performance/performance_analysis.md) - 性能分析報告
- [`validation_testing.md`](validation/validation_testing.md) - 驗證與測試
- [`DEPENDENCY_ANALYSIS.md`](DEPENDENCY_ANALYSIS.md) - 系統依賴關係分析

#### 🔧 **開發維護**
- [`CI_CD_GUIDE.md`](CI_CD_GUIDE.md) - CI/CD配置與故障排除指南

#### 📚 **使用指南**
- [`user_guide.md`](tutorials/user_guide.md) - 完整使用手冊
- [`quick_start.md`](tutorials/quick_start.md) - 快速開始指南
- [`advanced_usage.md`](tutorials/advanced_usage.md) - 高級使用技巧

### 🏆 文檔品質標準

- **學術嚴謹性**: 符合同行評議期刊標準
- **數學精確性**: 完整的推導過程和公式驗證
- **實驗可重現**: 詳細的實驗步驟和參數設定
- **工程實用性**: 實際應用指導和最佳實踐
- **🌡️ 熱流耦合**: 完整的溫度-流動雙向耦合建模
- **🔧 CI/CD完備**: 全自動化測試與部署流程

### 🔗 相關資源

- [主專案README](../README.md)
- [API文檔](api/)
- [基準測試報告](../benchmark_results/)
- [GitHub代碼庫](https://github.com/user/pour-over)

### 📧 學術合作

如需學術合作或引用本研究，請聯繫：
- 技術問題：參考代碼註釋和測試用例
- 學術討論：通過GitHub Issues
- 商業合作：參考LICENSE文件

---
**開發工具**: 使用 [opencode](https://opencode.ai) + GitHub Copilot 開發
**最後更新**: 2025年7月25日 - 熱流耦合系統完成
**重大成就**: ✅ Phase 3強耦合實現 | ✅ CI/CD完全修復 | ✅ 85%+測試覆蓋率