"""
物理模型插件架構系統
為新物理模型提供標準化的插件接口和動態載入機制
開發：opencode + GitHub Copilot
"""

# 標準庫導入
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Optional, Callable, Protocol
from dataclasses import dataclass
from pathlib import Path

# 第三方庫導入
import taichi as ti
import numpy as np

# 本地模組導入
from src.core.lbm_protocol import LBMSolverProtocol


@dataclass
class PluginMetadata:
    """插件元數據"""
    name: str
    version: str
    author: str
    description: str
    dependencies: List[str]
    physics_type: str  # "single_phase", "multiphase", "particle_coupled", etc.
    gpu_required: bool = True
    apple_silicon_optimized: bool = False


class PhysicsModelPlugin(ABC):
    """
    物理模型插件抽象基類
    
    所有物理模型插件必須繼承此類並實現所有抽象方法。
    提供標準化接口確保插件間的一致性和互操作性。
    """
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """返回插件元數據"""
        pass
    
    @abstractmethod
    def initialize(self, config: Any) -> None:
        """
        初始化物理模型
        
        Args:
            config: 配置參數對象
        """
        pass
    
    @abstractmethod
    def setup_fields(self, solver: LBMSolverProtocol) -> None:
        """
        設置所需的場變數
        
        Args:
            solver: LBM求解器實例
        """
        pass
    
    @abstractmethod
    def compute_forces(self, solver: LBMSolverProtocol) -> None:
        """
        計算物理力項
        
        Args:
            solver: LBM求解器實例
        """
        pass
    
    @abstractmethod
    def update_properties(self, solver: LBMSolverProtocol, step: int) -> None:
        """
        更新物理性質
        
        Args:
            solver: LBM求解器實例
            step: 當前時間步
        """
        pass
    
    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """返回診斷信息"""
        pass
    
    def validate_compatibility(self, solver: LBMSolverProtocol) -> bool:
        """
        驗證與求解器的兼容性
        
        Args:
            solver: LBM求解器實例
            
        Returns:
            True if 兼容, False otherwise
        """
        return True
    
    def cleanup(self) -> None:
        """清理資源"""
        pass


class MultiphasePlugin(PhysicsModelPlugin):
    """多相流物理模型插件接口"""
    
    @abstractmethod
    def compute_interface_tension(self, solver: LBMSolverProtocol) -> None:
        """計算界面張力"""
        pass
    
    @abstractmethod
    def update_phase_field(self, solver: LBMSolverProtocol) -> None:
        """更新相場"""
        pass


class ParticlePlugin(PhysicsModelPlugin):
    """顆粒耦合物理模型插件接口"""
    
    @abstractmethod
    def update_particle_dynamics(self, solver: LBMSolverProtocol, particles: Any) -> None:
        """更新顆粒動力學"""
        pass
    
    @abstractmethod
    def compute_fluid_particle_interaction(self, solver: LBMSolverProtocol, particles: Any) -> None:
        """計算流固耦合"""
        pass


class TurbulencePlugin(PhysicsModelPlugin):
    """湍流模型插件接口"""
    
    @abstractmethod
    def compute_turbulent_viscosity(self, solver: LBMSolverProtocol) -> None:
        """計算湍流黏性"""
        pass
    
    @abstractmethod
    def apply_turbulent_forcing(self, solver: LBMSolverProtocol) -> None:
        """應用湍流強迫項"""
        pass


class PluginRegistry:
    """插件註冊中心"""
    
    def __init__(self):
        self._plugins: Dict[str, Type[PhysicsModelPlugin]] = {}
        self._active_plugins: Dict[str, PhysicsModelPlugin] = {}
        self._plugin_metadata: Dict[str, PluginMetadata] = {}
    
    def register_plugin(self, plugin_class: Type[PhysicsModelPlugin]) -> None:
        """
        註冊插件類
        
        Args:
            plugin_class: 插件類
        """
        # 創建臨時實例獲取元數據
        temp_instance = plugin_class()
        metadata = temp_instance.metadata
        
        self._plugins[metadata.name] = plugin_class
        self._plugin_metadata[metadata.name] = metadata
        
        print(f"✅ 註冊插件: {metadata.name} v{metadata.version}")
    
    def load_plugin(self, name: str, config: Any) -> PhysicsModelPlugin:
        """
        載入並初始化插件
        
        Args:
            name: 插件名稱
            config: 配置參數
            
        Returns:
            插件實例
            
        Raises:
            ValueError: 當插件不存在時
        """
        if name not in self._plugins:
            raise ValueError(f"插件 '{name}' 未註冊")
        
        plugin_class = self._plugins[name]
        plugin_instance = plugin_class()
        
        # 檢查依賴
        metadata = self._plugin_metadata[name]
        for dep in metadata.dependencies:
            if not self._check_dependency(dep):
                raise RuntimeError(f"插件 '{name}' 依賴項 '{dep}' 不滿足")
        
        # 初始化插件
        plugin_instance.initialize(config)
        self._active_plugins[name] = plugin_instance
        
        print(f"🔌 載入插件: {name}")
        return plugin_instance
    
    def unload_plugin(self, name: str) -> None:
        """
        卸載插件
        
        Args:
            name: 插件名稱
        """
        if name in self._active_plugins:
            self._active_plugins[name].cleanup()
            del self._active_plugins[name]
            print(f"🔌 卸載插件: {name}")
    
    def get_active_plugins(self) -> Dict[str, PhysicsModelPlugin]:
        """獲取所有活躍插件"""
        return self._active_plugins.copy()
    
    def get_plugin_list(self) -> List[str]:
        """獲取所有註冊插件名稱"""
        return list(self._plugins.keys())
    
    def get_plugin_metadata(self, name: str) -> Optional[PluginMetadata]:
        """獲取插件元數據"""
        return self._plugin_metadata.get(name)
    
    def _check_dependency(self, dependency: str) -> bool:
        """檢查依賴項是否滿足"""
        try:
            importlib.import_module(dependency)
            return True
        except ImportError:
            return False
    
    def discover_plugins(self, plugin_dir: str = "plugins") -> None:
        """
        自動發現插件目錄中的插件
        
        Args:
            plugin_dir: 插件目錄路徑
        """
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            return
        
        for py_file in plugin_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            try:
                module_name = f"{plugin_dir}.{py_file.stem}"
                module = importlib.import_module(module_name)
                
                # 尋找插件類
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, PhysicsModelPlugin) and 
                        obj != PhysicsModelPlugin):
                        self.register_plugin(obj)
                        
            except ImportError as e:
                print(f"⚠️  載入插件 {py_file.name} 失敗: {e}")


class PluginManager:
    """插件管理器 - 高級插件編排和生命週期管理"""
    
    def __init__(self, solver: LBMSolverProtocol):
        self.solver = solver
        self.registry = PluginRegistry()
        self.execution_order: List[str] = []
        self.plugin_hooks: Dict[str, List[Callable]] = {
            'before_step': [],
            'after_collision': [],
            'after_streaming': [],
            'after_step': []
        }
    
    def setup_physics_pipeline(self, plugin_configs: List[Dict[str, Any]]) -> None:
        """
        設置物理計算管道
        
        Args:
            plugin_configs: 插件配置列表
        """
        for plugin_config in plugin_configs:
            plugin_name = plugin_config['name']
            plugin = self.registry.load_plugin(plugin_name, plugin_config)
            
            # 驗證兼容性
            if not plugin.validate_compatibility(self.solver):
                raise RuntimeError(f"插件 '{plugin_name}' 與求解器不兼容")
            
            # 設置場變數
            plugin.setup_fields(self.solver)
            
            # 添加到執行順序
            if plugin_name not in self.execution_order:
                self.execution_order.append(plugin_name)
        
        print(f"🔧 物理管道設置完成: {', '.join(self.execution_order)}")
    
    def execute_physics_step(self, step: int) -> None:
        """
        執行物理計算步驟
        
        Args:
            step: 當前時間步
        """
        # Before step hooks
        for hook in self.plugin_hooks['before_step']:
            hook(self.solver, step)
        
        # 執行插件物理計算
        active_plugins = self.registry.get_active_plugins()
        for plugin_name in self.execution_order:
            if plugin_name in active_plugins:
                plugin = active_plugins[plugin_name]
                plugin.compute_forces(self.solver)
                plugin.update_properties(self.solver, step)
        
        # After step hooks
        for hook in self.plugin_hooks['after_step']:
            hook(self.solver, step)
    
    def add_hook(self, event: str, callback: Callable) -> None:
        """
        添加事件回調
        
        Args:
            event: 事件名稱
            callback: 回調函數
        """
        if event in self.plugin_hooks:
            self.plugin_hooks[event].append(callback)
        else:
            raise ValueError(f"未知事件: {event}")
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """獲取系統診斷信息"""
        diagnostics = {
            'active_plugins': list(self.registry.get_active_plugins().keys()),
            'execution_order': self.execution_order,
            'plugin_diagnostics': {}
        }
        
        for name, plugin in self.registry.get_active_plugins().items():
            diagnostics['plugin_diagnostics'][name] = plugin.get_diagnostics()
        
        return diagnostics


# 全域插件註冊中心
plugin_registry = PluginRegistry()
plugin_registry.discover_plugins()


def register_physics_plugin(plugin_class: Type[PhysicsModelPlugin]) -> None:
    """
    插件註冊裝飾器
    
    Example:
        >>> @register_physics_plugin
        ... class MyPhysicsModel(PhysicsModelPlugin):
        ...     # 實現抽象方法
        ...     pass
    """
    plugin_registry.register_plugin(plugin_class)
    return plugin_class


# 內建插件示例
@register_physics_plugin  
class SurfaceTensionPlugin(MultiphasePlugin):
    """表面張力插件示例"""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="surface_tension",
            version="1.0.0", 
            author="CFD Team",
            description="表面張力計算模型",
            dependencies=["numpy", "taichi"],
            physics_type="multiphase"
        )
    
    def initialize(self, config: Any) -> None:
        self.surface_tension_coefficient = getattr(config, 'SURFACE_TENSION', 0.0728)
    
    def setup_fields(self, solver: LBMSolverProtocol) -> None:
        # 設置表面張力相關場變數
        pass
    
    def compute_forces(self, solver: LBMSolverProtocol) -> None:
        # 計算表面張力
        pass
    
    def update_properties(self, solver: LBMSolverProtocol, step: int) -> None:
        # 更新性質
        pass
    
    def get_diagnostics(self) -> Dict[str, Any]:
        return {"surface_tension_coefficient": self.surface_tension_coefficient}
    
    def compute_interface_tension(self, solver: LBMSolverProtocol) -> None:
        # 實現界面張力計算
        pass
    
    def update_phase_field(self, solver: LBMSolverProtocol) -> None:
        # 實現相場更新
        pass


if __name__ == "__main__":
    # 插件系統測試
    print("🔌 插件系統測試")
    print(f"註冊插件數量: {len(plugin_registry.get_plugin_list())}")
    for plugin_name in plugin_registry.get_plugin_list():
        metadata = plugin_registry.get_plugin_metadata(plugin_name)
        print(f"  • {plugin_name} v{metadata.version} - {metadata.description}")