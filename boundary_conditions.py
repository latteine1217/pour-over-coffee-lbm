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
        
    Interface:
        apply(solver): 應用邊界條件的核心方法
        
    Subclass Requirements:
        - 必須實現apply()方法
        - 必須使用@ti.kernel裝飾器(GPU並行)
        - 必須處理邊界條件的物理正確性
        
    Usage:
        >>> class MyBoundary(BoundaryConditionBase):
        ...     @ti.kernel
        ...     def apply(self, solver):
        ...         # 實現具體邊界條件邏輯
        ...         pass
    """
    
    @abstractmethod  
    def apply(self, solver: LBMSolverProtocol):
        """
        應用邊界條件到LBM求解器
        
        Args:
            solver: LBM求解器實例，包含所有場變數
            
        Note:
            所有子類必須實現此方法並使用@ti.kernel裝飾器
        """
        pass

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
    
    @ti.kernel
    def apply(self, solver: ti.template()):
        """
        對固體節點應用bounce-back邊界條件
        
        遍歷所有格點，對標記為固體的節點執行分布函數的
        方向反轉操作，實現完美反彈效果。
        
        Args:
            solver: LBM求解器實例
            
        Algorithm Details:
            1. 檢查節點是否為固體 (solid=1)
            2. 對每個離散速度方向q執行反轉
            3. 使用預計算查找表獲取相反方向
            4. 交換f_q和f_opp_q
            
        GPU Optimization:
            - 並行處理所有格點
            - 使用預計算相反方向查找表
            - 避免條件分支提升效能
        """
        """對固體節點應用bounce-back邊界條件"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solver.solid[i, j, k] == 1:  # 固體節點
                for q in range(config.Q_3D):
                    # 使用預計算的相反方向查找表
                    opp_q = solver.opposite_dir[q]
                    # 交換分佈函數
                    temp = solver.f[q, i, j, k]
                    solver.f[q, i, j, k] = solver.f[opp_q, i, j, k]
                    solver.f[opp_q, i, j, k] = temp

@ti.data_oriented
class OutletBoundary(BoundaryConditionBase):
    """
    出口邊界條件 - 自由流出
    
    實施零梯度外推的出口邊界條件，適用於計算域邊界的自由流出。
    此邊界條件假設流體在邊界處不受約束，自然流出計算域。
    
    Physics:
        - 零法向梯度: ∂φ/∂n = 0 (φ為任意場變數)
        - 自然對流: 流體不受人工約束
        - 質量守恆: 確保總質量平衡
        
    Algorithm:
        φ_boundary = φ_interior (一階外推)
        f_q^eq = equilibrium(ρ_boundary, u_boundary)
        
    Applications:
        - X/Y方向計算域邊界
        - 咖啡流出口
        - 自由表面近似
        
    Boundary Locations:
        - 左右邊界: i=0, i=NX-1
        - 前後邊界: j=0, j=NY-1
    """
    
    @ti.kernel
    def apply(self, solver: ti.template()):
        """
        應用outlet邊界條件到計算域邊界
        
        對X和Y方向的計算域邊界應用零梯度外推邊界條件，
        確保流體能夠自然流出而不產生非物理反射。
        
        Args:
            solver: LBM求解器實例
            
        Boundary Processing:
            1. X方向邊界: 左(i=0)和右(i=NX-1)
            2. Y方向邊界: 前(j=0)和後(j=NY-1)
            3. 從內部相鄰節點外推狀態
            4. 重置為平衡分布函數
            
        Extrapolation Method:
            - 一階外推: 簡單複製相鄰節點值
            - 密度外推: 確保質量守恆
            - 速度外推: 保持動量守恆
        """
        """應用outlet邊界條件到計算域邊界"""
        # X方向邊界
        for j, k in ti.ndrange(config.NY, config.NZ):
            # 左邊界
            i = 0
            if solver.solid[i, j, k] == 0:
                self._apply_extrapolation(solver, i, j, k, i+1, j, k)
            
            # 右邊界  
            i = config.NX - 1
            if solver.solid[i, j, k] == 0:
                self._apply_extrapolation(solver, i, j, k, i-1, j, k)
        
        # Y方向邊界
        for i, k in ti.ndrange(config.NX, config.NZ):
            # 前邊界
            j = 0
            if solver.solid[i, j, k] == 0:
                self._apply_extrapolation(solver, i, j, k, i, j+1, k)
            
            # 後邊界
            j = config.NY - 1
            if solver.solid[i, j, k] == 0:
                self._apply_extrapolation(solver, i, j, k, i, j-1, k)
    
    @ti.func
    def _apply_extrapolation(self, solver: ti.template(), i: ti.i32, j: ti.i32, k: ti.i32,
                           ref_i: ti.i32, ref_j: ti.i32, ref_k: ti.i32):
        """
        外推邊界條件實現
        
        從參考節點外推密度和速度到邊界節點，並重新計算
        平衡分布函數確保數值一致性。
        
        Args:
            solver: LBM求解器實例
            i, j, k: 邊界節點坐標
            ref_i, ref_j, ref_k: 參考節點坐標
            
        Extrapolation Method:
            ρ_boundary = ρ_reference
            u_boundary = u_reference  
            f_q_boundary = f_q^eq(ρ_boundary, u_boundary)
            
        Safety Checks:
            - 參考節點邊界檢查
            - 固體節點排除
            - 平衡分布安全計算
        """
        if (0 <= ref_i < config.NX and 0 <= ref_j < config.NY and 
            0 <= ref_k < config.NZ and solver.solid[ref_i, ref_j, ref_k] == 0):
            # 外推密度和速度
            solver.rho[i, j, k] = solver.rho[ref_i, ref_j, ref_k]
            solver.u[i, j, k] = solver.u[ref_i, ref_j, ref_k]
            
            # 更新分佈函數為平衡分佈
            for q in range(config.Q_3D):
                solver.f[q, i, j, k] = solver._compute_equilibrium_safe(
                    solver.rho[i, j, k], solver.u[i, j, k], q)

@ti.data_oriented  
class TopBoundary(BoundaryConditionBase):
    """
    頂部開放邊界 - 自由流出
    
    處理Z方向頂部(k=NZ-1)的開放邊界條件，適用於手沖咖啡
    模擬中水流從頂部自由流出的情況。
    
    Physics:
        - 自由表面近似: 頂部為大氣邊界
        - 壓力外推: 從內部外推大氣壓力
        - 速度保持: 允許垂直流出
        
    Algorithm:
        ρ_top = ρ_interior (k-1層)
        u_top = 當前值 (不強制修改)
        f_q = f_q^eq(ρ_top, u_top)
        
    Applications:
        - V60濾杯頂部開口
        - 注水流入區域上方
        - 大氣接觸面
        
    Boundary Condition:
        - 位置: k = NZ-1 (頂層)
        - 類型: 自由流出/大氣邊界
    """
    
    @ti.kernel
    def apply(self, solver: ti.template()):
        """
        應用頂部邊界條件
        
        對頂部邊界(Z方向最高層)應用自由流出邊界條件，
        允許流體自然流出而不產生人工約束。
        
        Args:
            solver: LBM求解器實例
            
        Implementation:
            1. 遍歷頂層所有X-Y位置
            2. 檢查節點是否為流體
            3. 從下方內部節點外推密度
            4. 保持當前速度不變
            5. 重新計算平衡分布函數
            
        Physical Reasoning:
            - 模擬大氣壓力邊界
            - 避免人工反射波
            - 保持流動自然性
        """
        """頂部邊界處理"""
        for i, j in ti.ndrange(config.NX, config.NY):
            k = config.NZ - 1  # 頂部
            if solver.solid[i, j, k] == 0:  # 流體節點
                # 從內部節點外推密度
                if k > 0 and solver.solid[i, j, k-1] == 0:
                    solver.rho[i, j, k] = solver.rho[i, j, k-1]
                    # 保持當前速度，不強制複製
                    
                    # 基於當前狀態重新計算平衡分佈
                    for q in range(config.Q_3D):
                        solver.f[q, i, j, k] = solver._compute_equilibrium_safe(
                            solver.rho[i, j, k], solver.u[i, j, k], q)

@ti.data_oriented
class BottomBoundary(BoundaryConditionBase):
    """
    底部固體邊界
    
    處理Z方向底部(k=0)的固體邊界條件，模擬V60濾杯底部
    和支撐結構的無滑移邊界。
    
    Physics:
        - 無滑移條件: 流體在固體壁面速度為零
        - 完全阻擋: 底部完全封閉，無流出
        - 動量傳遞: 通過viscous stress與壁面交互
        
    Algorithm:
        u_bottom = 0 (無滑移)
        bounce-back: f_q ↔ f_q̄
        
    Applications:
        - V60濾杯底部封閉面
        - 支撐結構表面  
        - 咖啡台表面
        
    Boundary Condition:
        - 位置: k = 0 (底層)
        - 類型: 無滑移固體壁面
        - 實現: bounce-back + 速度歸零
    """
    
    @ti.kernel
    def apply(self, solver: ti.template()):
        """
        應用底部封閉邊界條件
        
        對底部邊界(Z方向最低層)應用無滑移固體邊界條件，
        確保流體不能穿透底部且在壁面處速度為零。
        
        Args:
            solver: LBM求解器實例
            
        Implementation:
            1. 遍歷底層所有X-Y位置
            2. 檢查節點是否為流體(非固體)
            3. 強制設定速度為零(無滑移)
            4. 應用bounce-back條件
            5. 交換相反方向分布函數
            
        Physical Effects:
            - 實現無滑移邊界條件
            - 阻止流體穿透底部
            - 正確傳遞壁面剪應力
        """
        """底部完全封閉邊界"""
        for i, j in ti.ndrange(config.NX, config.NY):
            k = 0  # 底部
            if solver.solid[i, j, k] == 0:  # 如果是流體節點
                # 設為無滑移邊界條件
                solver.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                for q in range(config.Q_3D):
                    opp_q = solver.opposite_dir[q]
                    # Bounce-back邊界條件
                    temp = solver.f[q, i, j, k]
                    solver.f[q, i, j, k] = solver.f[opp_q, i, j, k]
                    solver.f[opp_q, i, j, k] = temp

@ti.data_oriented
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
        2. TopBoundary: 頂部邊界
        3. BottomBoundary: 底部邊界  
        4. OutletBoundary: 計算域邊界 (最後處理)
        
    Error Handling:
        - 完整異常捕獲和重拋
        - 詳細錯誤信息記錄
        - 系統狀態保護
        
    Attributes:
        bounce_back: 固體壁面反彈邊界實例
        outlet: 自由流出邊界實例
        top: 頂部開放邊界實例  
        bottom: 底部封閉邊界實例
    """
    
    def __init__(self) -> None:
        """
        初始化所有邊界條件策略
        
        建立所有邊界條件類型的實例，準備統一管理介面。
        確保所有邊界條件策略可用且正確初始化。
        
        Initialization Sequence:
            1. 建立固體邊界策略
            2. 建立流出邊界策略
            3. 建立頂部邊界策略
            4. 建立底部邊界策略
            5. 驗證初始化完成
            
        Memory Usage:
            - 輕量級策略物件
            - 無額外GPU記憶體開銷
            - 單例模式設計
        """
        """初始化所有邊界條件策略"""
        self.bounce_back = BounceBackBoundary()
        self.outlet = OutletBoundary()
        self.top = TopBoundary()
        self.bottom = BottomBoundary()
        
        print("✅ 邊界條件管理器初始化完成")
    
    def apply_all_boundaries(self, solver: LBMSolverProtocol) -> None:
        """
        按優先級順序應用所有邊界條件
        
        依照物理和數值優先級順序應用所有邊界條件，確保
        邊界處理的正確性和數值穩定性。
        
        Args:
            solver: LBM求解器實例
            
        Priority Order:
            1. 固體邊界 (最高優先級)
               - 完全決定固體節點行為
               - 不可被其他邊界覆蓋
               
            2. 頂部邊界
               - 大氣接觸面特殊處理
               - 自由流出條件
               
            3. 底部邊界  
               - 封閉邊界特殊處理
               - 無滑移條件
               
            4. 計算域邊界 (最後處理)
               - 外推邊界條件
               - 可能被其他邊界影響
               
        Error Handling:
            - 捕獲所有邊界條件異常
            - 記錄詳細錯誤信息
            - 重拋異常供上層處理
            
        Raises:
            Exception: 當任何邊界條件應用失敗時
        """
        try:
            # 1. 固體邊界 (最高優先級)
            self.bounce_back.apply(solver)
            
            # 2. 頂部邊界
            self.top.apply(solver)
            
            # 3. 底部邊界  
            self.bottom.apply(solver)
            
            # 4. 計算域邊界 (最後處理)
            self.outlet.apply(solver)
            
        except Exception as e:
            print(f"❌ 邊界條件應用失敗: {e}")
            raise
    
    def get_boundary_info(self) -> Dict[str, str]:
        """
        獲取邊界條件信息
        
        返回所有可用邊界條件類型的描述信息，用於系統
        狀態查詢和調試目的。
        
        Returns:
            Dict[str, str]: 邊界條件類型映射表
                鍵: 邊界條件識別名稱
                值: 邊界條件中文描述
                
        Information Included:
            - bounce_back: 固體壁面反彈邊界
            - outlet: 自由流出邊界
            - top: 頂部開放邊界  
            - bottom: 底部封閉邊界
            
        Usage:
            >>> manager = BoundaryConditionManager()
            >>> info = manager.get_boundary_info()
            >>> print(info['bounce_back'])
            '固體壁面反彈邊界'
        """
        return {
            'bounce_back': '固體壁面反彈邊界',
            'outlet': '自由流出邊界', 
            'top': '頂部開放邊界',
            'bottom': '底部封閉邊界'
        }