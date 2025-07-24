"""
LBM求解器抽象介面協議
定義所有LBM求解器必須實現的標準介面，確保代碼的可擴展性和一致性
開發：opencode + GitHub Copilot
"""

from typing import Protocol, runtime_checkable, Optional, Tuple, Any
import taichi as ti
import numpy as np


@runtime_checkable
class LBMSolverProtocol(Protocol):
    """
    LBM求解器標準協議
    
    定義所有LBM求解器實現必須遵循的介面標準，包括：
    - 基礎場變數 (分布函數、巨觀量、幾何)
    - 核心運算方法 (collision、streaming、邊界處理)
    - 診斷和監控能力
    - 記憶體管理和最佳化
    
    這個協議確保不同LBM實現間的互換性和一致性。
    """
    
    # ==================== 必要屬性 ====================
    
    # 分布函數場
    f: ti.field
    f_new: ti.field
    
    # 巨觀量場
    rho: ti.field  # 密度場
    u: ti.Vector.field  # 速度場
    
    # 幾何場
    solid: ti.field  # 固體標記場
    phase: ti.field  # 相場 (多相流)
    
    # ==================== 核心運算方法 ====================
    
    def step(self) -> None:
        """
        執行一個時間步的LBM運算
        
        包含完整的collision-streaming循環，邊界條件處理，
        以及必要的診斷檢查。這是LBM求解器的核心方法。
        """
        ...
    
    def collision_step(self) -> None:
        """
        執行collision運算子
        
        實現BGK或其他collision模型，計算平衡態分布並更新
        分布函數。必須支援多相流和LES湍流建模。
        """
        ...
    
    def streaming_step(self) -> None:
        """
        執行streaming運算子
        
        將分布函數沿各離散速度方向進行傳播，處理邊界條件
        和固體反彈。
        """
        ...
    
    def compute_macroscopic_quantities(self) -> None:
        """
        計算巨觀量 (密度、速度)
        
        從分布函數計算流體的巨觀性質，包括密度和動量。
        必須處理數值穩定性和除零保護。
        """
        ...
    
    def apply_boundary_conditions(self) -> None:
        """
        應用邊界條件
        
        處理各種邊界類型：入流、出流、固體壁面、週期性邊界等。
        必須保證數值穩定性和物理正確性。
        """
        ...
    
    # ==================== 初始化和設置 ====================
    
    def initialize_fields(self, initial_density: float = 1.0, 
                         initial_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        """
        初始化所有場變數
        
        Args:
            initial_density: 初始密度值
            initial_velocity: 初始速度向量 (ux, uy, uz)
        """
        ...
    
    def set_geometry(self, geometry_function: Any) -> None:
        """
        設置幾何配置
        
        Args:
            geometry_function: 定義固體區域的函數或數據
        """
        ...
    
    # ==================== 診斷和監控 ====================
    
    def get_diagnostics(self) -> dict:
        """
        獲取診斷信息
        
        Returns:
            包含數值穩定性、物理參數、性能指標的字典
        """
        ...
    
    def check_stability(self) -> bool:
        """
        檢查數值穩定性
        
        Returns:
            True if 穩定, False if 發現數值問題
        """
        ...
    
    def get_kinetic_energy(self) -> float:
        """
        計算系統總動能
        
        Returns:
            系統的總動能值
        """
        ...
    
    def get_mass_conservation_error(self) -> float:
        """
        計算質量守恆誤差
        
        Returns:
            質量守恆的相對誤差
        """
        ...
    
    # ==================== 記憶體和性能 ====================
    
    def get_memory_usage(self) -> dict:
        """
        獲取記憶體使用情況
        
        Returns:
            包含各場變數記憶體使用的字典
        """
        ...
    
    def optimize_memory_layout(self) -> None:
        """
        優化記憶體布局
        
        根據硬體特性調整記憶體配置，提升快取效率
        """
        ...
    
    # ==================== 可選進階功能 ====================
    
    def enable_les_turbulence(self, smagorinsky_constant: float = 0.1) -> None:
        """
        啟用LES湍流建模 (可選)
        
        Args:
            smagorinsky_constant: Smagorinsky模型常數
        """
        ...
    
    def add_force_term(self, force_field: ti.field) -> None:
        """
        添加體積力項 (可選)
        
        Args:
            force_field: 體積力場
        """
        ...
    
    def export_vtk(self, filename: str) -> None:
        """
        導出VTK格式數據 (可選)
        
        Args:
            filename: 輸出檔案名
        """
        ...


@runtime_checkable
class MultiphaseLBMProtocol(LBMSolverProtocol, Protocol):
    """
    多相流LBM求解器協議
    
    擴展基礎LBM協議，增加多相流特有的功能：
    - 相間作用力
    - 表面張力
    - 潤濕性
    - 相分離
    """
    
    def compute_phase_field(self) -> None:
        """計算相場分布"""
        ...
    
    def apply_surface_tension(self) -> None:
        """應用表面張力效應"""
        ...
    
    def set_contact_angle(self, angle: float) -> None:
        """設置接觸角 (潤濕性)"""
        ...


@runtime_checkable
class ParticleCoupledLBMProtocol(LBMSolverProtocol, Protocol):
    """
    顆粒耦合LBM求解器協議
    
    支援流體-顆粒耦合的LBM求解器介面：
    - 顆粒追蹤
    - 流固耦合力
    - 顆粒碰撞
    """
    
    def update_particle_coupling(self, particle_system: Any) -> None:
        """更新流體-顆粒耦合"""
        ...
    
    def compute_drag_force(self) -> ti.field:
        """計算顆粒拖曳力"""
        ...


def validate_lbm_solver(solver: Any) -> bool:
    """
    驗證對象是否符合LBM求解器協議
    
    Args:
        solver: 待驗證的求解器對象
        
    Returns:
        True if 符合協議, False otherwise
        
    Example:
        >>> from src.core.lbm_solver import LBMSolver
        >>> solver = LBMSolver()
        >>> if validate_lbm_solver(solver):
        ...     print("求解器符合標準協議")
    """
    return isinstance(solver, LBMSolverProtocol)


def get_solver_capabilities(solver: LBMSolverProtocol) -> dict:
    """
    獲取求解器功能清單
    
    Args:
        solver: LBM求解器實例
        
    Returns:
        功能清單字典
    """
    capabilities = {
        'basic_lbm': True,
        'multiphase': isinstance(solver, MultiphaseLBMProtocol),
        'particle_coupling': isinstance(solver, ParticleCoupledLBMProtocol),
        'les_turbulence': hasattr(solver, 'enable_les_turbulence'),
        'vtk_export': hasattr(solver, 'export_vtk'),
        'apple_silicon_optimized': hasattr(solver, 'apple_config'),
        'jax_hybrid': hasattr(solver, 'jax_enabled')
    }
    
    return capabilities


# 類型別名，方便使用
LBMSolver = LBMSolverProtocol
MultiphaseLBM = MultiphaseLBMProtocol
ParticleCoupledLBM = ParticleCoupledLBMProtocol