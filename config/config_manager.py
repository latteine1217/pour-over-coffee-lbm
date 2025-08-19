"""
config_manager.py

統一設定載入器：從 YAML 讀取使用者設定，安全地覆蓋允許的參數，
並避免修改數值穩定性核心（SCALE_VELOCITY, TAU_*, CFL_NUMBER）。

使用方式（於主程式最早期呼叫，載入求解器前）：

    import config.config as config
    from config.config_manager import apply_overrides
    apply_overrides(config)

可用環境變數：
- POUR_OVER_CONFIG: 指定 YAML 路徑（預設: config/config.yaml）
"""

from __future__ import annotations

import os
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


DEFAULT_CONFIG_PATH = os.environ.get("POUR_OVER_CONFIG", os.path.join(os.path.dirname(__file__), "config.yaml"))


# 禁止覆寫的關鍵參數（數值穩定性守則）
PROHIBITED = {
    "SCALE_VELOCITY",
    "TAU_WATER",
    "TAU_AIR",
    "CFL_NUMBER",
}

# YAML -> config 模組屬性映射
MAPPING: Dict[str, str] = {
    # simulation
    "simulation.max_steps": "MAX_STEPS",
    "simulation.output_freq": "OUTPUT_FREQ",
    # domain
    "domain.nx": "NX",
    "domain.ny": "NY",
    "domain.nz": "NZ",
    "domain.physical_domain_size": "PHYSICAL_DOMAIN_SIZE",
    # pouring
    "pouring.pour_rate_ml_s": "POUR_RATE_ML_S",
    # 直接設定入水直徑（米）
    "pouring.inlet_diameter_m": "INLET_DIAMETER_M",
    # les
    "les.enable_les": "ENABLE_LES",
    "les.re_threshold": "LES_REYNOLDS_THRESHOLD",
    "les.update_interval": "LES_UPDATE_INTERVAL",
    # filter paper
    # 注意：濾紙孔隙率屬於濾紙系統內部參數，config 模組中 PORE_PERC 代表咖啡床孔隙率，
    # 過去錯誤地將 filter_paper.porosity → PORE_PERC 會把咖啡床誤設為 0.85，
    # 因此此映射移除以避免物理錯置。若需支援濾紙孔隙率覆寫，應在濾紙系統內提供對應鉤子。
    # "filter_paper.porosity": "PAPER_POROSITY",  # 尚未在 config 中定義，先禁用
    # "filter_paper.thickness_mm": "PAPER_THICKNESS_MM",  # 尚未在 config 中定義，先禁用
    # coffee
    "coffee.powder_mass_g": "COFFEE_POWDER_MASS_G",
    "coffee.particle_diameter_mm": "PARTICLE_DIAMETER_MM",
    # simulation ui/diagnostics
    "simulation.interactive": "INTERACTIVE",
    "simulation.save_output": "SAVE_OUTPUT",
    "simulation.show_progress": "SHOW_PROGRESS",
    "simulation.viz": "VIZ_MODE",
    "simulation.diag_freq": "DIAG_FREQ",
    # boundaries and kernel
    "boundaries.soa_direct": "BOUNDARY_SOA_DIRECT",
    "kernel.block_dim": "APPLE_BLOCK_DIM",
    # visualization controls
    "visualization.enable_heavy_plots": "VIZ_HEAVY",
    "visualization.dpi": "VIZ_DPI",
}


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except Exception:
        return {}


def apply_overrides(config_module) -> None:
    """讀取YAML並套用允許的覆寫到 config 模組。

    必須在任何求解器/場初始化之前呼叫（即在 main.py 開頭）。
    """
    data = _load_yaml(DEFAULT_CONFIG_PATH)
    if not data:
        print(f"⚙️  使用預設設定（未找到或未讀取: {DEFAULT_CONFIG_PATH}）")
        return

    flat = _flatten(data)
    applied = []
    skipped = []

    for ykey, attr in MAPPING.items():
        if ykey not in flat:
            continue
        if attr in PROHIBITED:
            skipped.append((ykey, attr, "prohibited"))
            continue
        if not hasattr(config_module, attr):
            # 屬性不存在，跳過
            skipped.append((ykey, attr, "unknown"))
            continue
        try:
            old = getattr(config_module, attr)
            new = flat[ykey]
            # 嘗試基本型別轉換
            if isinstance(old, bool):
                new_val = bool(new)
            elif isinstance(old, int):
                new_val = int(new)
            elif isinstance(old, float):
                new_val = float(new)
            else:
                new_val = new
            setattr(config_module, attr, new_val)
            applied.append((ykey, attr, old, new_val))
        except Exception:
            skipped.append((ykey, attr, "error"))

    if applied:
        print("🧩 套用YAML覆寫：")
        for ykey, attr, old, new in applied:
            print(f"   - {ykey} → {attr}: {old} → {new}")
    # 在覆寫後重算注水派生量（安全範圍內，不觸及穩定性核心）
    try:
        if hasattr(config_module, 'recompute_pouring_derived'):
            config_module.recompute_pouring_derived(verbose=True)
    except Exception:
        pass
    if skipped:
        print("ℹ️  略過的設定：")
        for ykey, attr, reason in skipped:
            print(f"   - {ykey} → {attr} ({reason})")
