"""
配置參數驗證系統
為config.py提供schema驗證和參數合理性檢查
開發：opencode + GitHub Copilot
"""

# 標準庫導入
import math
from typing import Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field

# 第三方庫導入
import numpy as np


@dataclass
class ParameterRange:
    """參數範圍定義"""
    min_val: float
    max_val: float
    description: str
    critical: bool = True  # 是否為關鍵參數


@dataclass
class ConfigSchema:
    """配置參數驗證Schema"""
    
    # 網格參數範圍
    grid_params: Dict[str, ParameterRange] = field(default_factory=lambda: {
        'NX': ParameterRange(32, 512, "網格X向節點數", True),
        'NY': ParameterRange(32, 512, "網格Y向節點數", True), 
        'NZ': ParameterRange(32, 512, "網格Z向節點數", True),
        'DX': ParameterRange(0.1, 2.0, "格點間距", True),
        'DT': ParameterRange(0.1, 2.0, "時間步長", True)
    })
    
    # LBM數值參數範圍
    lbm_params: Dict[str, ParameterRange] = field(default_factory=lambda: {
        'TAU_WATER': ParameterRange(0.501, 2.0, "水相鬆弛時間", True),
        'TAU_AIR': ParameterRange(0.501, 3.0, "空氣相鬆弛時間", True),
        'CFL_NUMBER': ParameterRange(0.001, 0.1, "CFL數", True),
        'MACH_NUMBER': ParameterRange(0.001, 0.3, "馬赫數", True)
    })
    
    # 物理參數範圍
    physical_params: Dict[str, ParameterRange] = field(default_factory=lambda: {
        'WATER_TEMP_C': ParameterRange(70.0, 100.0, "水溫(°C)", False),
        'GRAVITY': ParameterRange(8.0, 12.0, "重力加速度", False),
        'COFFEE_MASS_G': ParameterRange(10.0, 50.0, "咖啡質量(g)", False),
        'POUR_RATE_ML_S': ParameterRange(1.0, 10.0, "注水速率(ml/s)", False)
    })
    
    # 數值穩定性參數
    stability_params: Dict[str, ParameterRange] = field(default_factory=lambda: {
        'MAX_VELOCITY_LU': ParameterRange(0.001, 0.3, "最大速度(格子單位)", True),
        'MIN_DENSITY': ParameterRange(0.1, 0.9, "最小密度", True),
        'MAX_DENSITY': ParameterRange(1.1, 10.0, "最大密度", True)
    })


class ConfigValidator:
    """配置參數驗證器"""
    
    def __init__(self):
        self.schema = ConfigSchema()
        self.validation_results: List[Dict[str, Any]] = []
        self.critical_errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config_module) -> bool:
        """
        驗證配置模組的所有參數
        
        Args:
            config_module: 配置模組對象
            
        Returns:
            True if 驗證通過, False if 有關鍵錯誤
        """
        self.validation_results.clear()
        self.critical_errors.clear()
        self.warnings.clear()
        
        # 驗證各類參數
        self._validate_grid_parameters(config_module)
        self._validate_lbm_parameters(config_module)
        self._validate_physical_parameters(config_module)
        self._validate_stability_parameters(config_module)
        
        # 驗證參數間的一致性
        self._validate_parameter_consistency(config_module)
        
        # 驗證D3Q19模型參數
        self._validate_d3q19_parameters(config_module)
        
        return len(self.critical_errors) == 0
    
    def _validate_grid_parameters(self, config_module) -> None:
        """驗證網格參數"""
        for param_name, param_range in self.schema.grid_params.items():
            if hasattr(config_module, param_name):
                value = getattr(config_module, param_name)
                self._check_parameter_range(param_name, value, param_range)
            else:
                self.critical_errors.append(f"缺少必要網格參數: {param_name}")
    
    def _validate_lbm_parameters(self, config_module) -> None:
        """驗證LBM數值參數"""
        for param_name, param_range in self.schema.lbm_params.items():
            if hasattr(config_module, param_name):
                value = getattr(config_module, param_name)
                self._check_parameter_range(param_name, value, param_range)
            else:
                if param_range.critical:
                    self.critical_errors.append(f"缺少關鍵LBM參數: {param_name}")
                else:
                    self.warnings.append(f"缺少LBM參數: {param_name}")
    
    def _validate_physical_parameters(self, config_module) -> None:
        """驗證物理參數"""
        for param_name, param_range in self.schema.physical_params.items():
            if hasattr(config_module, param_name):
                value = getattr(config_module, param_name)
                self._check_parameter_range(param_name, value, param_range)
    
    def _validate_stability_parameters(self, config_module) -> None:
        """驗證數值穩定性參數"""
        for param_name, param_range in self.schema.stability_params.items():
            if hasattr(config_module, param_name):
                value = getattr(config_module, param_name)
                self._check_parameter_range(param_name, value, param_range)
    
    def _validate_parameter_consistency(self, config_module) -> None:
        """驗證參數間一致性"""
        # 檢查CFL條件
        if hasattr(config_module, 'CFL_NUMBER') and hasattr(config_module, 'MAX_VELOCITY_LU'):
            cfl = config_module.CFL_NUMBER
            max_vel = config_module.MAX_VELOCITY_LU
            if max_vel > cfl:
                self.critical_errors.append(
                    f"CFL條件違反: MAX_VELOCITY_LU ({max_vel}) > CFL_NUMBER ({cfl})"
                )
        
        # 檢查鬆弛時間穩定性
        if hasattr(config_module, 'TAU_WATER') and config_module.TAU_WATER <= 0.5:
            self.critical_errors.append(
                f"水相鬆弛時間過小: TAU_WATER = {config_module.TAU_WATER} <= 0.5"
            )
        
        # 檢查網格解析度合理性
        if all(hasattr(config_module, param) for param in ['NX', 'NY', 'NZ']):
            total_nodes = config_module.NX * config_module.NY * config_module.NZ
            if total_nodes > 50_000_000:  # 50M nodes
                self.warnings.append(
                    f"網格節點數過大: {total_nodes:,} 可能導致記憶體不足"
                )
    
    def _validate_d3q19_parameters(self, config_module) -> None:
        """驗證D3Q19模型參數"""
        # 檢查離散速度向量
        if hasattr(config_module, 'CX_3D') and hasattr(config_module, 'CY_3D') and hasattr(config_module, 'CZ_3D'):
            cx, cy, cz = config_module.CX_3D, config_module.CY_3D, config_module.CZ_3D
            if not (len(cx) == len(cy) == len(cz) == 19):
                self.critical_errors.append(
                    f"D3Q19離散速度向量長度錯誤: {len(cx)}, {len(cy)}, {len(cz)} != 19"
                )
        
        # 檢查權重係數
        if hasattr(config_module, 'WEIGHTS_3D'):
            weights = config_module.WEIGHTS_3D
            if len(weights) != 19:
                self.critical_errors.append(f"D3Q19權重係數長度錯誤: {len(weights)} != 19")
            
            weight_sum = np.sum(weights)
            if abs(weight_sum - 1.0) > 1e-6:
                self.critical_errors.append(
                    f"D3Q19權重係數歸一化失敗: sum = {weight_sum} != 1.0"
                )
        
        # 檢查格子聲速
        if hasattr(config_module, 'CS2'):
            cs2 = config_module.CS2
            expected_cs2 = 1.0/3.0
            if abs(cs2 - expected_cs2) > 1e-6:
                self.critical_errors.append(
                    f"格子聲速錯誤: CS2 = {cs2} != {expected_cs2}"
                )
    
    def _check_parameter_range(self, name: str, value: float, param_range: ParameterRange) -> None:
        """檢查單個參數範圍"""
        if not (param_range.min_val <= value <= param_range.max_val):
            error_msg = (
                f"{name} = {value} 超出範圍 [{param_range.min_val}, {param_range.max_val}] "
                f"({param_range.description})"
            )
            
            if param_range.critical:
                self.critical_errors.append(error_msg)
            else:
                self.warnings.append(error_msg)
        
        self.validation_results.append({
            'parameter': name,
            'value': value,
            'range': (param_range.min_val, param_range.max_val),
            'valid': param_range.min_val <= value <= param_range.max_val,
            'critical': param_range.critical,
            'description': param_range.description
        })
    
    def get_validation_report(self) -> Dict[str, Any]:
        """獲取完整驗證報告"""
        return {
            'validation_passed': len(self.critical_errors) == 0,
            'critical_errors': self.critical_errors,
            'warnings': self.warnings,
            'total_parameters_checked': len(self.validation_results),
            'parameters_valid': sum(1 for r in self.validation_results if r['valid']),
            'parameters_invalid': sum(1 for r in self.validation_results if not r['valid']),
            'detailed_results': self.validation_results
        }
    
    def print_validation_report(self) -> None:
        """輸出驗證報告"""
        report = self.get_validation_report()
        
        print("=" * 60)
        print("🔍 配置參數驗證報告")
        print("=" * 60)
        
        if report['validation_passed']:
            print("✅ 配置驗證通過")
        else:
            print("❌ 配置驗證失敗")
        
        print(f"📊 統計: {report['parameters_valid']}/{report['total_parameters_checked']} 參數有效")
        
        if report['critical_errors']:
            print("\n🚨 關鍵錯誤:")
            for error in report['critical_errors']:
                print(f"  • {error}")
        
        if report['warnings']:
            print("\n⚠️  警告:")
            for warning in report['warnings']:
                print(f"  • {warning}")
        
        print("=" * 60)


def validate_config_module(config_module) -> bool:
    """
    驗證配置模組的便捷函數
    
    Args:
        config_module: 配置模組對象
        
    Returns:
        True if 驗證通過, False otherwise
        
    Example:
        >>> import config
        >>> from config_validator import validate_config_module
        >>> if validate_config_module(config):
        ...     print("配置驗證通過")
    """
    validator = ConfigValidator()
    success = validator.validate_config(config_module)
    validator.print_validation_report()
    return success


# 自動驗證功能
def auto_validate():
    """自動驗證當前配置"""
    try:
        import config
        return validate_config_module(config)
    except ImportError:
        print("❌ 無法導入config模組")
        return False


if __name__ == "__main__":
    # 執行自動驗證
    auto_validate()