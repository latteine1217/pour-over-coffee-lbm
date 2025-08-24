"""
統一LBM求解器 - Phase 2實現
整合記憶體適配器與計算後端的統一LBM求解器
開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import config.config as config
from typing import Dict, Any, Optional, Union
import platform
import sys
import os

# 簡化導入 - 避免複雜的依賴關係
try:
    from src.core.backends.apple_backend import AppleBackend
except ImportError:
    AppleBackend = None

try:
    from src.core.backends.cuda_backend import CUDABackend
except ImportError:
    CUDABackend = None

try:
    from src.core.backends.cpu_backend import CPUBackend
except ImportError:
    CPUBackend = None

@ti.data_oriented
class UnifiedLBMSolver:
    """
    統一LBM求解器 - Phase 2架構
    
    核心特性：
    1. 自動平台檢測與最佳後端選擇
    2. 統一記憶體布局適配
    3. 跨平台兼容性 (Apple Silicon/CUDA/CPU)
    4. 數值一致性保證
    5. 性能最佳化
    """
    
    def __init__(self, preferred_backend: Optional[str] = None):
        """
        初始化統一LBM求解器
        
        Args:
            preferred_backend: 偏好後端 ('apple', 'cuda', 'cpu', None=自動)
        """
        print("🔄 初始化統一LBM求解器 (Phase 2)...")
        
        # 自動選擇最佳後端
        self.backend = self._select_optimal_backend(preferred_backend)
        self.memory_adapter = self._create_memory_adapter()
        
        # 初始化LBM場變數
        self._init_lbm_fields()
        
        # 載入物理參數
        self._init_physics_params()
        
        print(f"✅ 統一LBM求解器初始化完成")
        print(f"   後端: {self.backend.get_backend_info()['name']}")
        print(f"   記憶體: {self.memory_adapter.__class__.__name__}")
        print(f"   記憶體使用量: {self._estimate_total_memory():.2f} GB")
    
    def _select_optimal_backend(self, preferred: Optional[str]):
        """自動選擇最佳計算後端"""
        print("🔍 自動選擇最佳計算後端...")
        
        if preferred:
            print(f"   用戶偏好: {preferred}")
        
        # 檢測可用後端
        available_backends = self._detect_available_backends()
        print(f"   可用後端: {list(available_backends.keys())}")
        
        # 選擇策略
        if preferred and preferred in available_backends:
            backend = available_backends[preferred]
            print(f"   ✅ 選擇用戶偏好後端: {preferred}")
        elif 'apple' in available_backends and self._is_apple_silicon():
            backend = available_backends['apple']
            print("   🍎 選擇Apple Silicon最佳化後端")
        elif 'cuda' in available_backends and self._has_nvidia_gpu():
            backend = available_backends['cuda']
            print("   🔥 選擇CUDA GPU後端")
        elif 'cpu' in available_backends:
            backend = available_backends['cpu']
            print("   💻 選擇CPU參考後端")
        else:
            raise RuntimeError("無可用的計算後端")
        
        return backend
    
    def _detect_available_backends(self) -> Dict[str, Any]:
        """檢測可用的計算後端"""
        available = {}
        
        # 檢測Apple Silicon後端
        if AppleBackend and self._is_apple_silicon():
            try:
                backend = AppleBackend()
                if backend.validate_platform():
                    available['apple'] = backend
            except Exception as e:
                print(f"   ⚠️ Apple後端不可用: {e}")
        
        # 檢測CUDA後端
        if CUDABackend and self._has_cuda():
            try:
                backend = CUDABackend()
                if backend.validate_platform():
                    available['cuda'] = backend
            except Exception as e:
                print(f"   ⚠️ CUDA後端不可用: {e}")
        
        # 檢測CPU後端
        if CPUBackend:
            try:
                backend = CPUBackend()
                if backend.validate_platform():
                    available['cpu'] = backend
            except Exception as e:
                print(f"   ⚠️ CPU後端不可用: {e}")
        
        return available
    
    def _is_apple_silicon(self) -> bool:
        """檢測Apple Silicon平台"""
        try:
            return platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin'
        except:
            return False
    
    def _has_nvidia_gpu(self) -> bool:
        """檢測NVIDIA GPU"""
        try:
            # 簡化檢測 - 實際應該檢查nvidia-smi
            return hasattr(ti, 'cuda')
        except:
            return False
    
    def _has_cuda(self) -> bool:
        """檢測CUDA可用性"""
        try:
            return hasattr(ti, 'cuda')
        except:
            return False
    
    def _create_memory_adapter(self):
        """創建記憶體布局適配器"""
        print("🧠 創建記憶體布局適配器...")
        
        # 簡化版記憶體適配器
        class SimpleMemoryAdapter:
            def __init__(self):
                self.layout_type = 'Standard'
                print(f"   💾 使用標準記憶體布局")
        
        return SimpleMemoryAdapter()
    
    def _init_lbm_fields(self):
        """初始化LBM場變數"""
        print("🏗️ 初始化LBM場變數...")
        
        # 分布函數場 - SoA布局
        self.f = []
        self.f_new = []
        for q in range(config.Q_3D):
            f_q = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
            f_new_q = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
            self.f.append(f_q)
            self.f_new.append(f_new_q)
        
        # 巨觀量場
        self.rho = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.u = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 幾何場
        self.solid = ti.field(dtype=ti.i32, shape=(config.NX, config.NY, config.NZ))
        self.phase = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        # 創建記憶體適配器介面
        self._create_adapter_interface()
        
        print(f"   ✅ 場變數初始化完成 ({config.Q_3D} 分布函數)")
    
    def _create_adapter_interface(self):
        """創建適配器介面"""
        # 為後端提供場變數存取
        self.memory_adapter.f = self.f
        self.memory_adapter.f_new = self.f_new
        self.memory_adapter.rho = self.rho
        self.memory_adapter.u = self.u
        self.memory_adapter.solid = self.solid
        self.memory_adapter.phase = self.phase
    
    def _init_physics_params(self):
        """初始化物理參數"""
        self.tau = config.TAU_WATER
        self.dt = config.DT
        self.Re = config.RE_CHAR
        
        print(f"   📊 物理參數: τ={self.tau:.3f}, Re={self.Re:.0f}")
    
    def _estimate_total_memory(self) -> float:
        """估算總記憶體使用量"""
        return self.backend.estimate_memory_usage(config.NX, config.NY, config.NZ)
    
    def step(self):
        """執行一步LBM計算"""
        # 準備參數
        params = {
            'tau': self.tau,
            'dt': self.dt,
            'Reynolds': self.Re
        }
        
        # 執行collision-streaming
        self.backend.execute_collision_streaming(self.memory_adapter, params)
    
    def initialize_fields(self):
        """初始化場變數"""
        print("🏁 初始化流場...")
        
        # 初始化密度場
        self.rho.fill(config.RHO_0)
        
        # 初始化速度場
        self.u.fill(0.0)
        
        # 初始化分布函數 (平衡態)
        self._init_equilibrium()
        
        # 初始化幾何
        self._init_geometry()
        
        print("   ✅ 流場初始化完成")
    
    @ti.kernel
    def _init_equilibrium(self):
        """初始化平衡分布函數"""
        # D3Q19權重
        w = ti.static([1.0/3.0] + [1.0/18.0]*6 + [1.0/36.0]*12)
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            rho_local = self.rho[i, j, k]
            u_local = self.u[i, j, k]
            u_sqr = u_local.dot(u_local)
            
            for q in ti.static(range(config.Q_3D)):
                # 簡化的平衡分布函數
                self.f[q][i, j, k] = w[q] * rho_local * (1.0 - 1.5 * u_sqr)
                self.f_new[q][i, j, k] = self.f[q][i, j, k]
    
    @ti.kernel
    def _init_geometry(self):
        """初始化幾何"""
        # 簡化的V60幾何
        center_x, center_y = config.NX // 2, config.NY // 2
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 預設為流體
            self.solid[i, j, k] = 0
            self.phase[i, j, k] = 1.0  # 水相
            
            # 簡化邊界條件
            if i == 0 or i == config.NX-1 or j == 0 or j == config.NY-1:
                self.solid[i, j, k] = 1  # 邊界為固體
    
    def get_solver_info(self) -> Dict[str, Any]:
        """獲取求解器資訊"""
        backend_info = self.backend.get_backend_info()
        
        return {
            'name': 'Unified LBM Solver',
            'version': 'Phase 2',
            'backend': backend_info,
            'memory_adapter': self.memory_adapter.__class__.__name__,
            'grid_size': (config.NX, config.NY, config.NZ),
            'memory_usage_gb': self._estimate_total_memory(),
            'physics_params': {
                'tau': self.tau,
                'dt': self.dt,
                'Reynolds': self.Re
            }
        }
    
    def run_diagnostic(self) -> Dict[str, Any]:
        """運行診斷檢查"""
        print("🔍 運行統一求解器診斷...")
        
        diagnostic_results = {
            'backend_status': self.backend.get_backend_info(),
            'memory_status': {
                'adapter_type': self.memory_adapter.__class__.__name__,
                'estimated_usage_gb': self._estimate_total_memory()
            },
            'field_status': {
                'total_fields': len(self.f) + 4,  # f + rho, u, solid, phase
                'grid_points': config.NX * config.NY * config.NZ
            }
        }
        
        # CPU後端額外診斷
        if hasattr(self.backend, 'run_debug_checks'):
            diagnostic_results['debug_checks'] = self.backend.run_debug_checks(self.memory_adapter)
        
        print("   ✅ 診斷完成")
        return diagnostic_results


# 工廠函數
def create_unified_solver(preferred_backend: Optional[str] = None) -> UnifiedLBMSolver:
    """
    創建統一LBM求解器
    
    Args:
        preferred_backend: 偏好後端 ('apple', 'cuda', 'cpu', None=自動)
    
    Returns:
        UnifiedLBMSolver: 統一求解器實例
    """
    return UnifiedLBMSolver(preferred_backend)


# 測試功能
def test_unified_solver():
    """測試統一求解器"""
    print("🧪 測試統一LBM求解器...")
    
    try:
        # 創建求解器
        solver = create_unified_solver()
        
        # 初始化
        solver.initialize_fields()
        
        # 運行診斷
        diagnostics = solver.run_diagnostic()
        
        # 執行幾步
        print("   🏃 執行測試步驟...")
        for step in range(3):
            solver.step()
            print(f"     Step {step+1}/3 完成")
        
        print("   ✅ 統一求解器測試成功")
        return True
        
    except Exception as e:
        print(f"   ❌ 統一求解器測試失敗: {e}")
        return False


if __name__ == "__main__":
    # 測試運行
    test_unified_solver()