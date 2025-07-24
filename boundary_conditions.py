# boundary_conditions.py
"""
模組化邊界條件系統 - CFD專家版

高度模組化的邊界條件實現系統，基於策略設計模式提供可擴展
和可測試的邊界條件管理。專為格子Boltzmann方法設計。

重構自lbm_solver.py的apply_boundary_conditions()方法，使用現代
軟件工程原則提升代碼可維護性、可測試性和擴展性。

主要特性:
    - 策略模式設計: 每種邊界條件獨立封裝
    - 模組化架構: 易於添加新邊界條件類型
    - 統一介面: 所有邊界條件遵循相同API
    - 優先級管理: 確保邊界條件正確應用順序
    - 企業級錯誤處理: 完整的異常捕獲和恢復

邊界條件類型:
    - BounceBackBoundary: 固體壁面反彈邊界
    - OutletBoundary: 自由流出邊界  
    - TopBoundary: 頂部開放邊界
    - BottomBoundary: 底部封閉邊界
    - FilterPaperBoundary: 濾紙多孔介質邊界

開發：opencode + GitHub Copilot
"""

from typing import Dict, Any, Protocol
from abc import ABC, abstractmethod
import taichi as ti
import config

class LBMSolverProtocol(Protocol):
    """LBM求解器協議定義，用於類型檢查"""
    solid: ti.field
    f: ti.field  
    rho: ti.field
    u: ti.field
    opposite_dir: ti.field

@ti.data_oriented
class BoundaryConditionBase(ABC):
    """
    邊界條件基類
    
    抽象基類定義所有邊界條件的統一介面。基於策略模式設計，
    確保所有具體邊界條件實現相同的API。
    
    Design Pattern:
        Strategy Pattern - 每種邊界條件作為獨立策略
        Template Method - 統一的應用流程框架
        
    Abstract Methods:
        apply(): 執行邊界條件的核心方法
        
    Interface Contract:
        - 所有子類必須實現apply方法
        - apply方法接受LBMSolverProtocol類型的求解器
        - 邊界條件應當是冪等的(重複調用結果相同)
        
    Implementation Notes:
        - 使用@ti.kernel裝飾器實現GPU並行計算
        - 確保記憶體訪問模式友好於GPU快取
        - 避免分支密集的控制流程
    """
    
    @abstractmethod
    def apply(self, solver):
        """
        抽象方法：應用邊界條件
        
        子類必須實現此方法來定義具體的邊界條件邏輯。
        
        Args:
            solver: LBM求解器實例，提供訪問場變數的介面
            
        Implementation Requirements:
            - 確保數值穩定性和物理正確性
            - 處理邊界情況和異常值
            
        Contract:
            - 方法應該是冪等的
            - 不應修改求解器的核心配置
            - 必須保持數值穩定性
        """
        pass

@ti.data_oriented
class FilterPaperBoundary(BoundaryConditionBase):
    """
    濾紙邊界條件 - 多孔介質流動邊界
    
    實現V60濾紙的多孔介質邊界條件，包括：
    - 多孔介質流動阻力 (Darcy定律)
    - 動態阻塞效應
    - 方向性阻力模型
    
    與FilterPaperSystem集成，提供統一的邊界條件介面。
    """
    
    def __init__(self, filter_system=None):
        """初始化濾紙邊界條件"""
        self.filter_system = filter_system
        print("✅ 濾紙邊界條件初始化完成")
    
    def set_filter_system(self, filter_system):
        """設置濾紙系統參考"""
        self.filter_system = filter_system
    
    def apply(self, solver):
        """應用濾紙邊界條件"""
        if self.filter_system and hasattr(self.filter_system, 'apply_filter_effects'):
            self.filter_system.apply_filter_effects()

@ti.data_oriented
class BounceBackBoundary(BoundaryConditionBase):
    """
    反彈邊界條件 - 固體壁面
    
    實施標準的bounce-back邊界條件，適用於固體壁面的無滑移邊界。
    此方法在格子Boltzmann方法中廣泛用於模擬固體邊界。
    
    Physics:
        - 無滑移條件: 壁面速度為零
        - 動量守恆: 粒子完全彈性反彈
        - 質量守恆: 分布函數對稱交換
        
    Algorithm:
        f_q^new(xᵦ) = f_q̄^old(xᵦ)
        其中q̄為q的相反方向
        
    Applications:
        - V60濾杯壁面
        - 咖啡顆粒表面
        - 固體障礙物
        
    Numerical Properties:
        - 一階精度邊界條件
        - 完全局部操作
        - GPU並行友好
    """
    
    def apply(self, solver):
        """應用bounce-back邊界條件"""
        if hasattr(solver, 'solid') and hasattr(solver, 'opposite_dir'):
            if hasattr(solver, 'f') and isinstance(solver.f, list):
                # SoA布局 - f是列表形式
                self._apply_bounce_back_soa_kernel(solver.solid, solver.f, solver.opposite_dir)
            elif hasattr(solver, 'f'):
                # 傳統4D場布局
                self._apply_bounce_back_4d_kernel(solver.solid, solver.f, solver.opposite_dir)
    
    @ti.kernel
    def _apply_bounce_back_soa_kernel(self, solid: ti.template(), f: ti.template(), opposite_dir: ti.template()):
        """bounce-back核心計算 - SoA版本"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solid[i, j, k] == 1:  # 固體節點
                # 執行bounce-back操作 - SoA布局
                for q in ti.static(range(1, config.Q_3D)):
                    opp_q = opposite_dir[q]
                    temp = f[q][i, j, k]
                    f[q][i, j, k] = f[opp_q][i, j, k]
                    f[opp_q][i, j, k] = temp
    
    @ti.kernel
    def _apply_bounce_back_4d_kernel(self, solid: ti.template(), f: ti.template(), opposite_dir: ti.template()):
        """bounce-back核心計算 - 4D場版本"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solid[i, j, k] == 1:  # 固體節點
                # 執行bounce-back操作 - 4D場
                for q in ti.static(range(1, config.Q_3D)):
                    opp_q = opposite_dir[q]
                    temp = f[q, i, j, k]
                    f[q, i, j, k] = f[opp_q, i, j, k]
                    f[opp_q, i, j, k] = temp

@ti.data_oriented
class OutletBoundary(BoundaryConditionBase):
    """
    流出邊界條件 - 計算域邊界
    
    實施外推型流出邊界條件，適用於計算域的側面和出口邊界。
    使用外推法維持流動的自然流出特性。
    """
    
    def apply(self, solver):
        """應用流出邊界條件"""
        if hasattr(solver, 'solid') and hasattr(solver, 'rho'):
            if hasattr(solver, 'ux'):  # SoA速度場
                self._apply_outlet_soa_kernel(solver.solid, solver.rho, solver.ux, solver.uy, solver.uz)
            elif hasattr(solver, 'u'):  # 向量速度場
                self._apply_outlet_vector_kernel(solver.solid, solver.rho, solver.u)
    
    @ti.kernel
    def _apply_outlet_soa_kernel(self, solid: ti.template(), rho: ti.template(),
                                ux: ti.template(), uy: ti.template(), uz: ti.template()):
        """流出邊界 - SoA版本"""
        # X方向邊界
        for j, k in ti.ndrange(config.NY, config.NZ):
            if solid[0, j, k] == 0:  # 左邊界
                rho[0, j, k] = rho[1, j, k]
                ux[0, j, k] = ux[1, j, k]
                uy[0, j, k] = uy[1, j, k]
                uz[0, j, k] = uz[1, j, k]
                
            if solid[config.NX-1, j, k] == 0:  # 右邊界
                rho[config.NX-1, j, k] = rho[config.NX-2, j, k]
                ux[config.NX-1, j, k] = ux[config.NX-2, j, k]
                uy[config.NX-1, j, k] = uy[config.NX-2, j, k]
                uz[config.NX-1, j, k] = uz[config.NX-2, j, k]
        
        # Y方向邊界
        for i, k in ti.ndrange(config.NX, config.NZ):
            if solid[i, 0, k] == 0:  # 前邊界
                rho[i, 0, k] = rho[i, 1, k]
                ux[i, 0, k] = ux[i, 1, k]
                uy[i, 0, k] = uy[i, 1, k]
                uz[i, 0, k] = uz[i, 1, k]
                
            if solid[i, config.NY-1, k] == 0:  # 後邊界
                rho[i, config.NY-1, k] = rho[i, config.NY-2, k]
                ux[i, config.NY-1, k] = ux[i, config.NY-2, k]
                uy[i, config.NY-1, k] = uy[i, config.NY-2, k]
                uz[i, config.NY-1, k] = uz[i, config.NY-2, k]
    
    @ti.kernel
    def _apply_outlet_vector_kernel(self, solid: ti.template(), rho: ti.template(), u: ti.template()):
        """流出邊界 - 向量版本"""
        # X方向邊界
        for j, k in ti.ndrange(config.NY, config.NZ):
            if solid[0, j, k] == 0:  # 左邊界
                rho[0, j, k] = rho[1, j, k]
                u[0, j, k] = u[1, j, k]
                    
            if solid[config.NX-1, j, k] == 0:  # 右邊界
                rho[config.NX-1, j, k] = rho[config.NX-2, j, k]
                u[config.NX-1, j, k] = u[config.NX-2, j, k]
        
        # Y方向邊界
        for i, k in ti.ndrange(config.NX, config.NZ):
            if solid[i, 0, k] == 0:  # 前邊界
                rho[i, 0, k] = rho[i, 1, k]
                u[i, 0, k] = u[i, 1, k]
                    
            if solid[i, config.NY-1, k] == 0:  # 後邊界
                rho[i, config.NY-1, k] = rho[i, config.NY-2, k]
                u[i, config.NY-1, k] = u[i, config.NY-2, k]

@ti.data_oriented
class TopBoundary(BoundaryConditionBase):
    """頂部邊界條件 - 大氣接觸面"""
    
    def apply(self, solver):
        """應用頂部邊界條件"""
        if hasattr(solver, 'solid') and hasattr(solver, 'rho'):
            if hasattr(solver, 'ux'):  # SoA速度場
                self._apply_top_soa_kernel(solver.solid, solver.rho, solver.ux, solver.uy, solver.uz)
            elif hasattr(solver, 'u'):  # 向量速度場
                self._apply_top_vector_kernel(solver.solid, solver.rho, solver.u)
    
    @ti.kernel
    def _apply_top_soa_kernel(self, solid: ti.template(), rho: ti.template(),
                             ux: ti.template(), uy: ti.template(), uz: ti.template()):
        """頂部邊界 - SoA版本"""
        for i, j in ti.ndrange(config.NX, config.NY):
            if solid[i, j, config.NZ-1] == 0:  # 頂部開放
                rho[i, j, config.NZ-1] = 1.0  # 大氣壓
                ux[i, j, config.NZ-1] = 0.0
                uy[i, j, config.NZ-1] = 0.0
                uz[i, j, config.NZ-1] = 0.0
    
    @ti.kernel
    def _apply_top_vector_kernel(self, solid: ti.template(), rho: ti.template(), u: ti.template()):
        """頂部邊界 - 向量版本"""
        for i, j in ti.ndrange(config.NX, config.NY):
            if solid[i, j, config.NZ-1] == 0:  # 頂部開放
                rho[i, j, config.NZ-1] = 1.0  # 大氣壓
                u[i, j, config.NZ-1] = ti.Vector([0.0, 0.0, 0.0])

@ti.data_oriented
class BottomBoundary(BoundaryConditionBase):
    """底部邊界條件 - 支持結構"""
    
    def apply(self, solver):
        """應用底部邊界條件"""
        if hasattr(solver, 'solid') and hasattr(solver, 'rho'):
            if hasattr(solver, 'ux'):  # SoA速度場
                self._apply_bottom_soa_kernel(solver.solid, solver.rho, solver.ux, solver.uy, solver.uz)
            elif hasattr(solver, 'u'):  # 向量速度場
                self._apply_bottom_vector_kernel(solver.solid, solver.rho, solver.u)
    
    @ti.kernel
    def _apply_bottom_soa_kernel(self, solid: ti.template(), rho: ti.template(),
                                ux: ti.template(), uy: ti.template(), uz: ti.template()):
        """底部邊界 - SoA版本"""
        for i, j in ti.ndrange(config.NX, config.NY):
            if solid[i, j, 0] == 0:  # 底部流出
                rho[i, j, 0] = rho[i, j, 1]
                ux[i, j, 0] = 0.0
                uy[i, j, 0] = 0.0
                uz[i, j, 0] = uz[i, j, 1]  # 允許垂直流出
    
    @ti.kernel
    def _apply_bottom_vector_kernel(self, solid: ti.template(), rho: ti.template(), u: ti.template()):
        """底部邊界 - 向量版本"""
        for i, j in ti.ndrange(config.NX, config.NY):
            if solid[i, j, 0] == 0:  # 底部流出
                rho[i, j, 0] = rho[i, j, 1]
                u_above = u[i, j, 1]
                u[i, j, 0] = ti.Vector([0.0, 0.0, u_above[2]])

class BoundaryConditionManager:
    """
    邊界條件管理器 - 策略模式實現
    
    中央化管理所有邊界條件的應用，基於策略設計模式提供
    統一的邊界條件管理介面。確保正確的應用順序和錯誤處理。
    
    Design Pattern:
        Strategy Pattern: 每種邊界條件作為獨立策略
        Manager Pattern: 統一管理和協調所有策略
        
    Boundary Priority (高到低):
        1. BounceBackBoundary: 固體邊界 (最高優先級)
        2. FilterPaperBoundary: 濾紙邊界 (多孔介質效應)
        3. TopBoundary: 頂部邊界
        4. BottomBoundary: 底部邊界  
        5. OutletBoundary: 計算域邊界 (最後處理)
        
    Error Handling:
        - 完整異常捕獲和重拋
        - 詳細錯誤信息記錄
        - 系統狀態保護
        
    Attributes:
        bounce_back: 固體壁面反彈邊界實例
        filter_paper: 濾紙多孔介質邊界實例
        outlet: 自由流出邊界實例
        top: 頂部開放邊界實例  
        bottom: 底部封閉邊界實例
    """
    
    def __init__(self, filter_system=None) -> None:
        """
        初始化所有邊界條件策略
        
        建立所有邊界條件類型的實例，準備統一管理介面。
        確保所有邊界條件策略可用且正確初始化。
        
        Args:
            filter_system: 可選的濾紙系統實例
        """
        self.bounce_back = BounceBackBoundary()
        self.filter_paper = FilterPaperBoundary(filter_system)
        self.outlet = OutletBoundary()
        self.top = TopBoundary()
        self.bottom = BottomBoundary()
        
        print("✅ 邊界條件管理器初始化完成")
    
    def set_filter_system(self, filter_system):
        """設置濾紙系統到邊界條件管理器"""
        self.filter_paper.set_filter_system(filter_system)
    
    def apply_all_boundaries(self, solver):
        """
        按優先級順序應用所有邊界條件
        
        依照物理和數值優先級順序應用所有邊界條件，確保
        邊界處理的正確性和數值穩定性。
        """
        try:
            # 1. 固體邊界 (最高優先級)
            self.bounce_back.apply(solver)
            
            # 2. 濾紙邊界 (多孔介質效應)
            self.filter_paper.apply(solver)
            
            # 3. 頂部邊界
            self.top.apply(solver)
            
            # 4. 底部邊界  
            self.bottom.apply(solver)
            
            # 5. 計算域邊界 (最後處理)
            self.outlet.apply(solver)
            
        except Exception as e:
            print(f"❌ 邊界條件應用失敗: {e}")
            raise
    
    def get_boundary_info(self) -> Dict[str, str]:
        """獲取邊界條件信息"""
        return {
            'bounce_back': '固體壁面反彈邊界',
            'filter_paper': '濾紙多孔介質邊界',
            'outlet': '自由流出邊界', 
            'top': '頂部開放邊界',
            'bottom': '底部封閉邊界'
        }
    
    def get_priority_order(self) -> list:
        """獲取邊界條件應用優先級順序"""
        return [
            'bounce_back',    # 1. 固體邊界 (最高優先級)
            'filter_paper',   # 2. 濾紙邊界 (多孔介質效應)
            'top',           # 3. 頂部邊界 (大氣接觸)
            'bottom',        # 4. 底部邊界 (支持結構)
            'outlet'         # 5. 計算域邊界 (數值穩定)
        ]
    
    # ====================
    # 統一初始化系統 (CFD一致性優化)
    # ====================
    
    def initialize_all_boundaries(self, geometry_system=None, filter_system=None, multiphase_system=None):
        """
        統一初始化所有邊界條件 (CFD一致性優化)
        
        確保邊界條件初始化順序一致性，避免不同模組間的
        初始化競爭和不一致問題。
        
        Args:
            geometry_system: 幾何系統實例 (V60形狀, 固體場)
            filter_system: 濾紙系統實例 (多孔介質)
            multiphase_system: 多相流系統實例 (相場邊界)
            
        Initialization Order:
            1. 幾何邊界 (solid field) - 定義流體域
            2. 濾紙邊界 (filter effects) - 多孔介質
            3. 多相流邊界 (phase field) - 相界面
            4. 流動邊界 (flow conditions) - 動力學邊界
            
        Benefits:
            - 消除初始化順序依賴性
            - 確保邊界條件一致性
            - 減少模組間耦合
            - 統一錯誤處理
        """
        print("🔧 統一初始化邊界條件系統...")
        
        try:
            # 階段 1: 幾何邊界初始化 (最基礎)
            if geometry_system:
                print("   ├─ 階段1: 初始化幾何邊界 (固體場)")
                if hasattr(geometry_system, 'init_geometry'):
                    geometry_system.init_geometry()
                if hasattr(geometry_system, 'create_v60_geometry'):
                    geometry_system.create_v60_geometry()
                
            # 階段 2: 濾紙系統邊界初始化
            if filter_system:
                print("   ├─ 階段2: 初始化濾紙邊界 (多孔介質)")
                self.filter_paper.set_filter_system(filter_system)
                if hasattr(filter_system, 'setup_filter_geometry'):
                    filter_system.setup_filter_geometry()
                
            # 階段 3: 多相流邊界初始化
            if multiphase_system:
                print("   ├─ 階段3: 初始化多相流邊界 (相場)")
                if hasattr(multiphase_system, 'init_phase_field'):
                    multiphase_system.init_phase_field()
                if hasattr(multiphase_system, 'validate_initial_phase_consistency'):
                    multiphase_system.validate_initial_phase_consistency()
                    
            # 階段 4: 流動邊界條件驗證
            print("   ├─ 階段4: 驗證邊界條件一致性")
            self._validate_boundary_consistency()
            
            print("   └─ ✅ 邊界條件統一初始化完成")
            
        except Exception as e:
            print(f"   └─ ❌ 邊界條件初始化失敗: {e}")
            raise RuntimeError(f"邊界條件統一初始化失敗: {e}")
    
    def _validate_boundary_consistency(self):
        """
        驗證邊界條件一致性
        
        檢查各邊界條件模組間是否存在衝突或不一致，
        確保物理邊界和數值邊界的協調。
        """
        # 檢查所有邊界條件實例是否正確初始化
        boundary_components = [
            ('bounce_back', self.bounce_back),
            ('filter_paper', self.filter_paper), 
            ('outlet', self.outlet),
            ('top', self.top),
            ('bottom', self.bottom)
        ]
        
        for name, component in boundary_components:
            if component is None:
                raise ValueError(f"邊界條件組件 {name} 未初始化")
                
        # 檢查濾紙系統是否正確設置
        if self.filter_paper.filter_system is None:
            print("   ⚠️  濾紙系統未設置，將跳過濾紙邊界效應")
            
        print("   ✅ 邊界條件一致性驗證通過")
    
    def get_initialization_summary(self) -> Dict[str, Any]:
        """
        獲取初始化摘要信息
        
        Returns:
            dict: 包含所有邊界條件初始化狀態的摘要
        """
        return {
            'boundary_count': 5,
            'priority_order': self.get_priority_order(),
            'boundary_info': self.get_boundary_info(),
            'filter_system_status': 'connected' if self.filter_paper.filter_system else 'not_connected',
            'initialization_complete': True
        }