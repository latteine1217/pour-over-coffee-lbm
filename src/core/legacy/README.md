# 🗂️ Legacy LBM Solvers

## 📋 目錄說明

此目錄包含Phase 2重構前的原始LBM求解器，已被統一求解器系統取代。保留作為：
- 📚 **歷史參考**：原始實現的技術文檔
- 🔄 **數值驗證**：新舊系統對比驗證
- 🛡️ **緊急回退**：必要時的備用選項

## 📦 Legacy檔案清單

### 1. `lbm_solver.py` (60,206行)
- **用途**：原始D3Q19 LBM求解器
- **特點**：混合記憶體布局，部分SoA + 4D
- **狀態**：功能完整，已被 `UnifiedLBMSolver` 取代
- **遷移日期**：2025-08-24

### 2. `ultra_optimized_lbm.py` (36,222行)  
- **用途**：Apple Silicon優化版LBM求解器
- **特點**：完全SoA布局，Metal GPU優化
- **狀態**：高性能版本，已整合至 `AppleBackend`
- **遷移日期**：2025-08-24

### 3. `cuda_dual_gpu_lbm.py` (19,462行)
- **用途**：NVIDIA雙GPU並行LBM求解器
- **特點**：GPU分域並行，P2P記憶體最佳化
- **狀態**：GPU特化版本，已整合至 `CUDABackend`
- **遷移日期**：2025-08-24

## 🔄 新舊系統對應關係

| Legacy檔案 | 統一系統對應 | 說明 |
|------------|-------------|------|
| `lbm_solver.py` | `UnifiedLBMSolver` + `StandardAdapter` | 標準4D記憶體布局 |
| `ultra_optimized_lbm.py` | `UnifiedLBMSolver` + `SoAAdapter` + `AppleBackend` | Apple Silicon優化 |
| `cuda_dual_gpu_lbm.py` | `UnifiedLBMSolver` + `GPUAdapter` + `CUDABackend` | NVIDIA GPU並行 |

## ⚠️ 使用警告

- **已棄用**：這些求解器不再主動維護
- **依賴過時**：可能與新配置系統不兼容
- **僅供參考**：建議使用統一求解器系統
- **數值驗證**：可用於驗證新系統的正確性

## 🔧 緊急使用指南

如需臨時使用legacy求解器：

```python
# 僅供緊急情況
from src.core.legacy.ultra_optimized_lbm import UltraOptimizedLBMSolver

# 注意：需要手動處理配置兼容性
solver = UltraOptimizedLBMSolver()
```

## 📊 重構效益統計

- **代碼減少**：115,890行 → 統一架構
- **重複消除**：70%功能重複 → 統一API
- **平台支援**：3套獨立系統 → 自動後端選擇
- **維護成本**：3倍維護 → 統一維護

---

*此目錄由Phase 2重構自動生成 | 開發：opencode + GitHub Copilot*