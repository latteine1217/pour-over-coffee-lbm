"""
config/config.py - Backward-compatibility shim to the unified config system

說明：
- 本檔案提供向後相容層，將舊的 `import config.config as config` 導向 Phase 1 的統一配置系統。
- 所有參數與函數皆由 `config` 套件的統一出口提供（core/physics/thermal）。
- 這是最小變更方案，不改動數值穩定性關鍵參數，只做轉發與別名。

開發：opencode + GitHub Copilot
"""

# 導入統一配置的所有符號（核心/物理/熱傳 + 驗證/診斷）
from .__init__ import *  # noqa: F401,F403

# 單次告警：提示使用者改為 `import config as config`
_printed_notice = False
def _print_deprecation_once():
    global _printed_notice
    if not _printed_notice:
        try:
            print("⚠️  Deprecation: 請改用 `import config as config`，`import config.config` 已轉為相容層。")
        except Exception:
            pass
        _printed_notice = True

_print_deprecation_once()

# 與舊版名稱保持相容的別名（若有歷史名稱差異，於此補齊）
try:
    # thermal: 提供 get_thermal_config_summary 舊名
    get_thermal_config_summary  # type: ignore
except NameError:
    try:
        # 統一系統中名稱為 get_thermal_summary
        get_thermal_config_summary = get_thermal_summary  # type: ignore
    except Exception:
        pass

# 兼容檢查函數舊名（若外部仍呼叫）
try:
    validate_parameter_consistency  # type: ignore
except NameError:
    # 使用統一的一致性檢查
    try:
        from . import check_parameter_consistency as validate_parameter_consistency  # type: ignore
    except ImportError:
        # 如果沒有檢查函數，創建一個空的版本
        def validate_parameter_consistency():
            """向後相容的參數檢查函數"""
            pass