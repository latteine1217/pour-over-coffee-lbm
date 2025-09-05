"""
core_config.py - 向後相容層（轉發至統一的 config.core）

用途：
- 保持舊程式碼 `from config.core_config import ...` 可用。
- 所有定義均轉發自 `config.core`（統一核心LBM參數的單一來源）。
- 提供與 thermal_config 同風格的棄用提醒。

開發：opencode + GitHub Copilot
"""

# 轉發所有核心參數與函數
from .core import *  # noqa: F401,F403

# 單次告警，提示使用新入口
_printed_notice = False

def _print_deprecation_once():
    global _printed_notice
    if not _printed_notice:
        try:
            print("⚠️  Deprecation: 請改用 `from config.core import ...`，core_config 已轉為相容層。")
        except Exception:
            pass
        _printed_notice = True

_print_deprecation_once()
