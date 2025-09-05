"""
src/core/ultra_optimized_lbm.py - 向後相容包裝器

說明：
- 提供舊引用 `from src.core.ultra_optimized_lbm import UltraOptimizedLBMSolver`
- 轉發至 `src.core.legacy.ultra_optimized_lbm`

開發：opencode + GitHub Copilot
"""

from .legacy.ultra_optimized_lbm import UltraOptimizedLBMSolver  # noqa: F401

__all__ = [
    "UltraOptimizedLBMSolver",
]

