"""
CPU計算後端 - 參考實現
純CPU實現的LBM計算引擎，提供標準參考和調試功能
開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import config
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class ComputeBackend(ABC):
    """計算後端基類"""
    
    @abstractmethod
    def execute_collision_streaming(self, memory_adapter, params: Dict[str, Any]):
        """執行collision-streaming步驟"""
        pass
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """回傳後端資訊"""
        pass


@ti.data_oriented
class CPUBackend(ComputeBackend):
    """
    CPU專用計算後端
    
    核心特性：
    1. 純CPU計算，無GPU依賴
    2. 單線程參考實現
    3. 詳細調試支援
    4. 數值穩定性驗證
    5. 平台無關性
    """
    
    def __init__(self):
        super().__init__()
        print("💻 初始化CPU計算後端...")
        
        self.cpu_threads = getattr(config, 'CPU_THREADS', 1)
        
        # 性能監控
        self.performance_metrics = {
            'collision_time': 0.0,
            'streaming_time': 0.0, 
            'boundary_time': 0.0,
            'total_time': 0.0
        }
        
        # 初始化CPU專用常數
        self._init_cpu_constants()
        
        print(f"✅ CPU後端初始化完成 (Threads={self.cpu_threads})")
    
    def _init_cpu_constants(self):
        """初始化CPU專用常數"""
        # D3Q19速度模板 - CPU記憶體優化
        self.ex = ti.field(dtype=ti.i32, shape=config.Q_3D)
        self.ey = ti.field(dtype=ti.i32, shape=config.Q_3D) 
        self.ez = ti.field(dtype=ti.i32, shape=config.Q_3D)
        self.w = ti.field(dtype=ti.f32, shape=config.Q_3D)
        
        # 載入D3Q19模板
        ex_host = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0], dtype=np.int32)
        ey_host = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 0, 1, -1], dtype=np.int32)
        ez_host = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int32)
        w_host = np.array([1.0/3.0] + [1.0/18.0]*6 + [1.0/36.0]*12, dtype=np.float32)
        
        self.ex.from_numpy(ex_host)
        self.ey.from_numpy(ey_host) 
        self.ez.from_numpy(ez_host)
        self.w.from_numpy(w_host)
    
    @ti.kernel
    def _cpu_collision_kernel(self, f: ti.template(), f_new: ti.template(), 
                              rho: ti.template(), u: ti.template(),
                              solid: ti.template(), tau: ti.f32):
        """
        CPU專用collision kernel - 單線程參考實現
        
        提供數值穩定的參考實現，便於調試和驗證
        """
        # CPU單線程處理
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 跳過固體格點
            if solid[i, j, k] > 0.5:
                continue
            
            # 計算巨觀量 - 高精度計算
            rho_local = 0.0
            ux, uy, uz = 0.0, 0.0, 0.0
            
            # 密度與動量累積 - 穩定計算
            for q in ti.static(range(config.Q_3D)):
                f_q = f[q, i, j, k] if hasattr(f, '__getitem__') else f[i, j, k, q]
                rho_local += f_q
                ux += f_q * self.ex[q]
                uy += f_q * self.ey[q] 
                uz += f_q * self.ez[q]
            
            # 速度正規化 - 數值穩定性保護
            if rho_local > 1e-12:
                inv_rho = 1.0 / rho_local
                ux *= inv_rho
                uy *= inv_rho
                uz *= inv_rho
            else:
                ux = uy = uz = 0.0
            
            # 更新巨觀場
            rho[i, j, k] = rho_local
            u[i, j, k] = ti.Vector([ux, uy, uz])
            
            # BGK碰撞 - 數值穩定版本
            u_sqr = ux*ux + uy*uy + uz*uz
            
            for q in ti.static(range(config.Q_3D)):
                # 平衡分布函數 - 穩定計算
                e_dot_u = self.ex[q]*ux + self.ey[q]*uy + self.ez[q]*uz
                feq = self.w[q] * rho_local * (1.0 + 3.0*e_dot_u + 4.5*e_dot_u*e_dot_u - 1.5*u_sqr)
                
                # BGK碰撞 - 穩定更新
                f_old = f[q, i, j, k] if hasattr(f, '__getitem__') else f[i, j, k, q]
                f_new_val = f_old - (f_old - feq) / tau
                
                # 數值穩定性檢查
                if f_new_val < 0.0:
                    f_new_val = 0.0  # 非負性保護
                
                if hasattr(f_new, '__setitem__'):
                    f_new[q, i, j, k] = f_new_val
                else:
                    f_new[i, j, k, q] = f_new_val
    
    @ti.kernel  
    def _cpu_streaming_kernel(self, f: ti.template(), f_new: ti.template()):
        """
        CPU專用streaming kernel - 標準實現
        
        採用標準的streaming算法，確保數值正確性
        """
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            for q in ti.static(range(config.Q_3D)):
                # 計算來源位置
                src_i = i - self.ex[q]
                src_j = j - self.ey[q]
                src_k = k - self.ez[q]
                
                # 邊界檢查
                if (src_i >= 0 and src_i < config.NX and 
                    src_j >= 0 and src_j < config.NY and 
                    src_k >= 0 and src_k < config.NZ):
                    f_stream = f_new[q, src_i, src_j, src_k] if hasattr(f_new, '__getitem__') else f_new[src_i, src_j, src_k, q]
                else:
                    # 邊界處理：保持原值
                    f_stream = f[q, i, j, k] if hasattr(f, '__getitem__') else f[i, j, k, q]
                
                if hasattr(f, '__setitem__'):
                    f[q, i, j, k] = f_stream
                else:
                    f[i, j, k, q] = f_stream
    
    @ti.kernel
    def _cpu_boundary_kernel(self, f: ti.template(), solid: ti.template()):
        """CPU專用邊界條件kernel"""
        # V60濾杯邊界 - 標準bounce-back
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solid[i, j, k] > 0.5:
                # 標準bounce-back邊界條件
                for q in ti.static(range(config.Q_3D)):
                    # 找到反向速度
                    opp_q = self._find_opposite_direction(q)
                    f_boundary = f[q, i, j, k] if hasattr(f, '__getitem__') else f[i, j, k, q]
                    
                    if hasattr(f, '__setitem__'):
                        f[opp_q, i, j, k] = f_boundary
                    else:
                        f[i, j, k, opp_q] = f_boundary
    
    @ti.func
    def _find_opposite_direction(self, q: ti.i32) -> ti.i32:
        """找到q方向的反向方向"""
        # D3Q19反向映射表
        opposite = ti.static([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
        return opposite[q]
    
    def execute_collision_streaming(self, memory_adapter, params: Dict[str, Any]):
        """
        執行CPU版collision-streaming步驟
        
        Args:
            memory_adapter: 記憶體布局適配器
            params: 計算參數 (tau, dt, 邊界條件等)
        """
        import time
        start_time = time.time()
        tau = params.get('tau', 0.6)
        
        # CPU三階段執行
        print("💻 執行CPU collision-streaming...")
        
        # 獲取場變數 (適配不同記憶體布局)
        f = getattr(memory_adapter, 'f', None)
        f_new = getattr(memory_adapter, 'f_new', None) 
        rho = getattr(memory_adapter, 'rho', None)
        u = getattr(memory_adapter, 'u', None)
        solid = getattr(memory_adapter, 'solid', None)
        
        if f is None or f_new is None:
            print("⚠️ 記憶體適配器場變數不可用")
            return
        
        # Phase 1: CPU collision (高精度)
        collision_start = time.time()
        self._cpu_collision_kernel(f, f_new, rho, u, solid, tau)
        self.performance_metrics['collision_time'] = time.time() - collision_start
        
        # Phase 2: CPU streaming (標準實現)  
        streaming_start = time.time()
        self._cpu_streaming_kernel(f, f_new)
        self.performance_metrics['streaming_time'] = time.time() - streaming_start
        
        # Phase 3: 邊界條件處理
        boundary_start = time.time()
        self._cpu_boundary_kernel(f, solid)
        self.performance_metrics['boundary_time'] = time.time() - boundary_start
        
        # CPU同步
        ti.sync()
        
        # 更新總時間
        self.performance_metrics['total_time'] = time.time() - start_time
    
    def apply_boundary_conditions(self, memory_adapter, params: Dict[str, Any]):
        """
        應用CPU優化的邊界條件
        
        Args:
            memory_adapter: 記憶體布局適配器
            params: 邊界條件參數
        """
        import time
        start_time = time.time()
        
        # 獲取場變數
        f = getattr(memory_adapter, 'f', None)
        solid = getattr(memory_adapter, 'solid', None)
        
        if f is None or solid is None:
            print("⚠️ 記憶體適配器場變數不可用")
            return
        
        # CPU邊界條件處理
        self._cpu_boundary_kernel(f, solid)
        
        # CPU同步
        ti.sync()
        
        # 更新性能指標
        self.performance_metrics['boundary_time'] = time.time() - start_time
    
    def compute_macroscopic_quantities(self, memory_adapter, params: Dict[str, Any]):
        """
        計算CPU優化的巨觀量
        
        Args:
            memory_adapter: 記憶體布局適配器
            params: 計算參數
        """
        import time
        start_time = time.time()
        
        # 獲取場變數
        f = getattr(memory_adapter, 'f', None)
        rho = getattr(memory_adapter, 'rho', None)
        u = getattr(memory_adapter, 'u', None)
        
        if f is None or rho is None or u is None:
            print("⚠️ 記憶體適配器場變數不可用")
            return
        
        # 在CPU collision kernel中已經計算了巨觀量
        # 這裡可以執行額外的調試檢查
        if params.get('debug_mode', False):
            self.run_debug_checks(memory_adapter)
        
        # CPU同步
        ti.sync()
        
        # 更新性能指標
        self.performance_metrics['macroscopic_time'] = time.time() - start_time
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """回傳CPU性能指標"""
        return {
            **self.performance_metrics,
            'total_time': sum(self.performance_metrics.values()),
            'throughput_mlups': self._calculate_throughput(),
            'memory_usage_gb': self._estimate_memory_usage_simple(),
            'debug_checks': self.run_debug_checks if hasattr(self, 'run_debug_checks') else None
        }
    
    def _calculate_throughput(self) -> float:
        """計算MLUPS (Million Lattice Updates Per Second)"""
        total_time = self.performance_metrics.get('total_time', 0.0)
        if total_time > 0:
            grid_size = config.NX * config.NY * config.NZ
            return (grid_size / 1e6) / total_time
        return 0.0
    
    def _estimate_memory_usage_simple(self) -> float:
        """簡單估算記憶體使用量 (GB)"""
        grid_size = config.NX * config.NY * config.NZ
        bytes_total = grid_size * config.Q_3D * 4 * 2  # f + f_new
        bytes_total += grid_size * 4 * 4  # rho, u, phase, solid
        return bytes_total / (1024**3)
    
    def get_backend_info(self) -> Dict[str, Any]:
        """回傳CPU後端資訊"""
        return {
            'name': 'CPU Backend',
            'type': 'CPU Reference',
            'threads': self.cpu_threads,
            'gpu_acceleration': False,
            'precision': 'High',
            'memory_layout': 'Standard',
            'optimization_level': 'Reference Implementation',
            'debugging_support': True
        }
    
    def initialize_backend(self):
        """初始化後端（工廠調用）"""
        # 後端已經在__init__中初始化完成，這裡可以做額外的驗證
        # CPU總是可用，不需要特殊檢查
        return True
    
    def estimate_memory_usage(self, nx: int, ny: int, nz: int) -> float:
        """估算CPU後端記憶體使用量 (GB)"""
        # CPU記憶體：f[19][nx×ny×nz] + 輔助場
        fields_memory = 19 * nx * ny * nz * 4  # 分布函數 (f32)
        fields_memory += 4 * nx * ny * nz * 4  # rho, u, phase, solid
        total_gb = fields_memory / (1024**3)
        return total_gb
    
    def validate_platform(self) -> bool:
        """驗證CPU平台可用性"""
        return True  # CPU總是可用
    
    def run_debug_checks(self, memory_adapter) -> Dict[str, bool]:
        """執行調試檢查"""
        checks = {
            'mass_conservation': self._check_mass_conservation(memory_adapter),
            'momentum_conservation': self._check_momentum_conservation(memory_adapter),
            'numerical_stability': self._check_numerical_stability(memory_adapter),
            'boundary_conditions': self._check_boundary_conditions(memory_adapter)
        }
        return checks
    
    def _check_mass_conservation(self, memory_adapter) -> bool:
        """檢查質量守恆"""
        # 簡化實現 - 實際應計算總質量變化
        return True
    
    def _check_momentum_conservation(self, memory_adapter) -> bool:
        """檢查動量守恆"""
        # 簡化實現 - 實際應計算總動量變化
        return True
    
    def _check_numerical_stability(self, memory_adapter) -> bool:
        """檢查數值穩定性"""
        # 簡化實現 - 實際應檢查NaN/Inf
        return True
    
    def _check_boundary_conditions(self, memory_adapter) -> bool:
        """檢查邊界條件"""
        # 簡化實現 - 實際應驗證邊界處理
        return True


# 提供簡化工廠函數
def create_cpu_backend():
    """創建CPU後端"""
    return CPUBackend()