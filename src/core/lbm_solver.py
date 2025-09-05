"""
src/core/lbm_solver.py - 向後相容包裝器

說明：
- 提供舊引用 `from src.core.lbm_solver import LBMSolver` 的相容性
- 轉發至 `src.core.legacy.lbm_solver` 中的實作
- 未來可切換為統一求解器實作而不影響上層模組

開發：opencode + GitHub Copilot
"""

# 盡量從 legacy 匯入，若無 3D 別名則提供相容別名
try:  # pragma: no cover
    from .legacy.lbm_solver import LBMSolver, LBMSolver3D  # type: ignore # noqa: F401
except Exception:  # pragma: no cover
    from .legacy.lbm_solver import LBMSolver  # type: ignore # noqa: F401
    LBMSolver3D = LBMSolver  # 相容別名

__all__ = [
    "LBMSolver",
    "LBMSolver3D",
]

