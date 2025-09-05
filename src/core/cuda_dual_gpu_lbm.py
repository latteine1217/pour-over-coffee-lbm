"""
src/core/cuda_dual_gpu_lbm.py - 向後相容包裝器

說明：
- 提供舊引用 `from src.core.cuda_dual_gpu_lbm import CUDADualGPULBMSolver` 的相容性
- 統一路徑為 `from src.core.cuda_dual_gpu_lbm import CUDADualGPULBMSolver`
- 內部轉發至 legacy 實作，未改變任何計算邏輯

開發：opencode + GitHub Copilot
"""

from .legacy.cuda_dual_gpu_lbm import CUDADualGPULBMSolver  # noqa: F401

__all__ = [
    "CUDADualGPULBMSolver",
]

