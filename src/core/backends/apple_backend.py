"""
Apple Silicon計算後端 - Metal GPU深度優化
專為M1/M2/M3晶片設計的LBM計算引擎，採用SoA記憶體布局與統一記憶體架構
開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import config
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import time


class ComputeBackend(ABC):
    """計算後端基類"""
    
    @abstractmethod
    def execute_collision_streaming(self, memory_adapter, params: Dict[str, Any]):
        """執行collision-streaming步驟"""
        pass
    
    @abstractmethod
    def apply_boundary_conditions(self, memory_adapter, params: Dict[str, Any]):
        """應用邊界條件"""
        pass
    
    @abstractmethod
    def compute_macroscopic_quantities(self, memory_adapter, params: Dict[str, Any]):
        """計算巨觀量"""
        pass
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """回傳後端資訊"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """回傳性能指標"""
        pass


@ti.data_oriented
class AppleBackend(ComputeBackend):
    """
    Apple Silicon專用計算後端
    
    核心優化技術：
    1. Metal GPU專用kernel優化
    2. 統一記憶體零拷貝技術
    3. SoA記憶體布局最大化快取效率
    4. Apple Silicon cache-line對齊
    5. SIMD vectorization友好設計
    """
    
    def __init__(self):
        super().__init__()
        print("🍎 初始化Apple Silicon計算後端...")
        
        # 檢測Apple Silicon平台
        self.is_apple_silicon = self._detect_apple_silicon()
        self.block_dim = getattr(config, 'APPLE_BLOCK_DIM', 128)
        
        # 詳細輸出控制 - 只在初始化時顯示詳細信息
        self.verbose_mode = True  # 初始化時啟用詳細輸出
        self._first_execution = True  # 第一次執行標誌
        
        # 性能監控
        self.performance_metrics = {
            'collision_time': 0.0,
            'streaming_time': 0.0, 
            'boundary_time': 0.0,
            'total_time': 0.0
        }
        
        # 初始化Metal專用常數
        self._init_metal_constants()
        
        print(f"✅ Apple Silicon後端初始化完成 (Metal GPU, Block={self.block_dim})")
    
    def _detect_apple_silicon(self) -> bool:
        """檢測Apple Silicon平台"""
        try:
            import platform
            return platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin'
        except:
            return False
    
    def _init_metal_constants(self):
        """初始化Metal GPU專用常數"""
        # D3Q19速度模板 - cache對齊
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
    def _apple_collision_kernel(self, f: ti.template(), f_new: ti.template(), 
                                rho: ti.template(), u: ti.template(),
                                solid: ti.template(), tau: ti.f32):
        """
        Apple Silicon專用collision kernel - Metal GPU優化
        
        採用SoA記憶體布局，針對Apple統一記憶體架構優化
        """
        # Metal GPU友好的並行化策略
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 跳過固體格點
            if solid[i, j, k] > 0.5:
                continue
            
            # 計算巨觀量 - SIMD友好
            rho_local = 0.0
            ux, uy, uz = 0.0, 0.0, 0.0
            
            # 密度與動量累積 - 向量化友好
            for q in ti.static(range(config.Q_3D)):
                f_q = f[q, i, j, k] if hasattr(f, '__getitem__') else f[i, j, k, q]
                rho_local += f_q
                ux += f_q * self.ex[q]
                uy += f_q * self.ey[q] 
                uz += f_q * self.ez[q]
            
            # 速度正規化
            inv_rho = 1.0 / (rho_local + 1e-12)
            ux *= inv_rho
            uy *= inv_rho
            uz *= inv_rho
            
            # 更新巨觀場
            rho[i, j, k] = rho_local
            u[i, j, k] = ti.Vector([ux, uy, uz])
            
            # BGK碰撞 - Metal GPU優化版本
            u_sqr = ux*ux + uy*uy + uz*uz
            
            for q in ti.static(range(config.Q_3D)):
                # 平衡分布函數
                e_dot_u = self.ex[q]*ux + self.ey[q]*uy + self.ez[q]*uz
                feq = self.w[q] * rho_local * (1.0 + 3.0*e_dot_u + 4.5*e_dot_u*e_dot_u - 1.5*u_sqr)
                
                # BGK碰撞
                f_old = f[q, i, j, k] if hasattr(f, '__getitem__') else f[i, j, k, q]
                f_new_val = f_old - (f_old - feq) / tau
                
                if hasattr(f_new, '__setitem__'):
                    f_new[q, i, j, k] = f_new_val
                else:
                    f_new[i, j, k, q] = f_new_val
    
    @ti.kernel  
    def _apple_streaming_kernel(self, f: ti.template(), f_new: ti.template()):
        """
        Apple Silicon專用streaming kernel - 統一記憶體優化
        
        採用target方式streaming，避免記憶體競爭
        """
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            for q in ti.static(range(config.Q_3D)):
                # 計算來源位置
                src_i = i - self.ex[q]
                src_j = j - self.ey[q]
                src_k = k - self.ez[q]
                
                # 邊界檢查 - Metal GPU分支優化
                if (src_i >= 0 and src_i < config.NX and 
                    src_j >= 0 and src_j < config.NY and 
                    src_k >= 0 and src_k < config.NZ):
                    f_stream = f_new[q, src_i, src_j, src_k] if hasattr(f_new, '__getitem__') else f_new[src_i, src_j, src_k, q]
                else:
                    f_stream = f[q, i, j, k] if hasattr(f, '__getitem__') else f[i, j, k, q]  # 邊界處理
                
                if hasattr(f, '__setitem__'):
                    f[q, i, j, k] = f_stream
                else:
                    f[i, j, k, q] = f_stream
    
    @ti.kernel
    def _apple_macroscopic_kernel(self, f: ti.template(), rho: ti.template(), u: ti.template()):
        """
        Apple Silicon專用巨觀量計算kernel
        
        計算密度和速度場，針對Metal GPU優化
        """
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 計算巨觀量 - SIMD友好
            rho_local = 0.0
            ux, uy, uz = 0.0, 0.0, 0.0
            
            # 密度與動量累積 - 向量化友好
            for q in ti.static(range(config.Q_3D)):
                f_q = f[q, i, j, k] if hasattr(f, '__getitem__') else f[i, j, k, q]
                rho_local += f_q
                ux += f_q * self.ex[q]
                uy += f_q * self.ey[q] 
                uz += f_q * self.ez[q]
            
            # 速度正規化
            inv_rho = 1.0 / (rho_local + 1e-12)
            ux *= inv_rho
            uy *= inv_rho
            uz *= inv_rho
            
            # 更新巨觀場
            rho[i, j, k] = rho_local
            u[i, j, k] = ti.Vector([ux, uy, uz])
    
    @ti.kernel
    def _apple_boundary_kernel(self, f: ti.template(), solid: ti.template()):
        """Apple Silicon專用邊界條件kernel"""
        # V60濾杯邊界 - bounce-back處理
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solid[i, j, k] > 0.5:
                # 簡化bounce-back
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
        執行Apple Silicon優化的collision-streaming步驟
        
        Args:
            memory_adapter: 記憶體布局適配器
            params: 計算參數 (tau, dt, 邊界條件等)
        """
        import time
        start_time = time.time()
        
        tau = params.get('tau', 0.6)
        
        # Apple Silicon三階段優化執行
        if self._first_execution:
            print("🍎 執行Apple Silicon collision-streaming...")
        
        # 獲取場變數 (適配不同記憶體布局)
        f = getattr(memory_adapter, 'f', None)
        f_new = getattr(memory_adapter, 'f_new', None) 
        rho = getattr(memory_adapter, 'rho', None)
        u = getattr(memory_adapter, 'u', None)
        solid = getattr(memory_adapter, 'solid', None)
        
        if f is None or f_new is None:
            print("⚠️ 記憶體適配器場變數不可用")
            return
        
        # 檢查場變數類型 - 修正Taichi field識別邏輯
        # Taichi field 具有 shape 屬性，Python list 沒有
        is_taichi_field = hasattr(f, 'shape') and hasattr(f_new, 'shape')
        is_soa_format = isinstance(f, list) and isinstance(f_new, list)
        
        if is_soa_format and self._first_execution:
            print("🍎 檢測到SoA格式，使用逐個field處理...")
            self._first_execution = False
            # Phase 1: SoA collision - 逐個處理每個方向
            collision_start = time.time()
            self._process_soa_collision(f, f_new, rho, u, solid, tau)
            self.performance_metrics['collision_time'] = time.time() - collision_start
            
            # Phase 2: SoA streaming - 逐個處理每個方向
            streaming_start = time.time()
            self._process_soa_streaming(f, f_new)
            self.performance_metrics['streaming_time'] = time.time() - streaming_start
        elif is_taichi_field:
            if self._first_execution:
                print("🍎 檢測到標準Taichi場，使用標準kernel...")
                self._first_execution = False
                
            # Phase 1: Metal GPU collision (直接使用Taichi field)
            collision_start = time.time()
            self._apple_collision_kernel(f, f_new, rho, u, solid, tau)
            self.performance_metrics['collision_time'] = time.time() - collision_start
            
            # Phase 2: 統一記憶體streaming  
            streaming_start = time.time()
            self._apple_streaming_kernel(f, f_new)
            self.performance_metrics['streaming_time'] = time.time() - streaming_start
        else:
            print(f"⚠️ 未知的場變數格式: f類型={type(f)}, f_new類型={type(f_new)}")
            return
        
        # Metal GPU同步
        ti.sync()
        
        # 更新總時間
        self.performance_metrics['total_time'] = time.time() - start_time
    
    def apply_boundary_conditions(self, memory_adapter, params: Dict[str, Any]):
        """
        應用Apple Silicon優化的邊界條件
        
        Args:
            memory_adapter: 記憶體布局適配器
            params: 邊界條件參數
        """
        start_time = time.time()
        
        # 獲取場變數
        f = getattr(memory_adapter, 'f', None)
        solid = getattr(memory_adapter, 'solid', None)
        
        if f is None or solid is None:
            print("⚠️ 記憶體適配器場變數不可用")
            return
        
        # Apple Silicon邊界條件處理
        self._apple_boundary_kernel(f, solid)
        
        # Metal GPU同步
        ti.sync()
        
        # 更新性能指標
        self.performance_metrics['boundary_time'] = time.time() - start_time
    
    def compute_macroscopic_quantities(self, memory_adapter, params: Dict[str, Any]):
        """
        計算Apple Silicon優化的巨觀量
        
        Args:
            memory_adapter: 記憶體布局適配器
            params: 計算參數
        """
        start_time = time.time()
        
        # 獲取場變數
        f = getattr(memory_adapter, 'f', None)
        rho = getattr(memory_adapter, 'rho', None)
        u = getattr(memory_adapter, 'u', None)
        
        if f is None or rho is None or u is None:
            print("⚠️ 記憶體適配器場變數不可用")
            return
        
        # Apple Silicon巨觀量計算
        self._apple_macroscopic_kernel(f, rho, u)
        
        # Metal GPU同步
        ti.sync()
        
        # 更新性能指標
        self.performance_metrics['macroscopic_time'] = time.time() - start_time
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """回傳Apple Silicon性能指標"""
        return {
            **self.performance_metrics,
            'total_time': sum(self.performance_metrics.values()),
            'throughput_mlups': self._calculate_throughput(),
            'memory_bandwidth': self._estimate_memory_bandwidth()
        }
    
    def _calculate_throughput(self) -> float:
        """計算MLUPS (Million Lattice Updates Per Second)"""
        total_time = self.performance_metrics.get('total_time', 0.0)
        if total_time > 0:
            grid_size = config.NX * config.NY * config.NZ
            return (grid_size / 1e6) / total_time
        return 0.0
    
    def _estimate_memory_bandwidth(self) -> float:
        """估算記憶體帶寬 (GB/s)"""
        total_time = self.performance_metrics.get('total_time', 0.0)
        if total_time > 0:
            # 估算每個時間步的記憶體訪問量
            grid_size = config.NX * config.NY * config.NZ
            bytes_per_step = grid_size * config.Q_3D * 4 * 2  # f + f_new
            return (bytes_per_step / 1e9) / total_time
        return 0.0
    
    def get_backend_info(self) -> Dict[str, Any]:
        """回傳Apple Silicon後端資訊"""
        return {
            'name': 'Apple Silicon Backend',
            'type': 'Metal GPU',
            'is_apple_silicon': self.is_apple_silicon,
            'unified_memory': True,
            'soa_optimized': True,
            'block_dim': self.block_dim,
            'memory_layout': 'SoA',
            'optimization_level': 'Maximum'
        }
    
    def initialize_backend(self):
        """初始化後端（工廠調用）"""
        # 後端已經在__init__中初始化完成，這裡可以做額外的驗證
        if not self.is_apple_silicon:
            raise RuntimeError("當前平台不是Apple Silicon，無法使用Apple後端")
        return True
    
    def _process_soa_collision(self, f_list, f_new_list, rho, u, solid, tau):
        """處理SoA格式的collision步驟"""
        # 第一步：重置巨觀量
        self._reset_macroscopic(rho, u, solid)
        
        # 第二步：累積密度和動量
        for q in range(len(f_list)):
            self._accumulate_density_momentum(f_list[q], rho, u, solid, q)
        
        # 第三步：正規化速度
        self._normalize_velocity(rho, u, solid)
        
        # 第四步：執行collision
        for q in range(len(f_list)):
            self._single_direction_collision(f_list[q], f_new_list[q], rho, u, solid, tau, q)
    
    def _process_soa_streaming(self, f_list, f_new_list):
        """處理SoA格式的streaming步驟"""
        for q in range(len(f_list)):
            self._single_direction_streaming(f_list[q], f_new_list[q], q)
    
    @ti.kernel
    def _reset_macroscopic(self, rho: ti.template(), u: ti.template(), solid: ti.template()):
        """重置巨觀量"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solid[i, j, k] > 0.5:
                continue
                
            rho[i, j, k] = 0.0
            u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            
    @ti.kernel  
    def _accumulate_density_momentum(self, f_q: ti.template(), rho: ti.template(),
                                   u: ti.template(), solid: ti.template(), 
                                   q: ti.i32):
        """累積密度和動量 - 單個方向"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solid[i, j, k] > 0.5:
                continue
                
            f_val = f_q[i, j, k]
            rho[i, j, k] += f_val
            u[i, j, k][0] += f_val * self.ex[q]
            u[i, j, k][1] += f_val * self.ey[q]  
            u[i, j, k][2] += f_val * self.ez[q]
    
    @ti.kernel
    def _normalize_velocity(self, rho: ti.template(), u: ti.template(), solid: ti.template()):
        """正規化速度場"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solid[i, j, k] > 0.5:
                continue
                
            rho_val = rho[i, j, k]
            if rho_val > 1e-12:
                inv_rho = 1.0 / rho_val
                u[i, j, k] *= inv_rho
    
    @ti.kernel
    def _single_direction_collision(self, f_q: ti.template(), f_new_q: ti.template(),
                                  rho: ti.template(), u: ti.template(),
                                  solid: ti.template(), tau: ti.f32, q: ti.i32):
        """單個方向的collision計算"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solid[i, j, k] > 0.5:
                f_new_q[i, j, k] = f_q[i, j, k]  # 固體邊界
                continue
                
            rho_val = rho[i, j, k]
            ux = u[i, j, k][0]
            uy = u[i, j, k][1] 
            uz = u[i, j, k][2]
            
            # 計算平衡分布函數
            e_dot_u = self.ex[q]*ux + self.ey[q]*uy + self.ez[q]*uz
            u_sqr = ux*ux + uy*uy + uz*uz
            feq = self.w[q] * rho_val * (1.0 + 3.0*e_dot_u + 4.5*e_dot_u*e_dot_u - 1.5*u_sqr)
            
            # BGK collision
            f_old = f_q[i, j, k]
            f_new_q[i, j, k] = f_old - (f_old - feq) / tau
    
    @ti.kernel
    def _single_direction_streaming(self, f_q: ti.template(), f_new_q: ti.template(), q: ti.i32):
        """單個方向的streaming計算"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 計算來源位置
            src_i = i - self.ex[q]
            src_j = j - self.ey[q]
            src_k = k - self.ez[q]
            
            # 邊界檢查
            if (0 <= src_i < config.NX and 
                0 <= src_j < config.NY and 
                0 <= src_k < config.NZ):
                f_q[i, j, k] = f_new_q[src_i, src_j, src_k]
            else:
                # 邊界處理 - 保持原值
                f_q[i, j, k] = f_new_q[i, j, k]
    
    def estimate_memory_usage(self, nx: int, ny: int, nz: int) -> float:
        """估算Apple Silicon後端記憶體使用量 (GB)"""
        # SoA布局：f[19][nx×ny×nz] + 輔助場
        fields_memory = 19 * nx * ny * nz * 4  # 分布函數 (f32)
        fields_memory += 4 * nx * ny * nz * 4  # rho, u, phase, solid
        total_gb = fields_memory / (1024**3)
        return total_gb
    
    def validate_platform(self) -> bool:
        """驗證Apple Silicon平台可用性"""
        return self.is_apple_silicon


# 提供簡化工廠函數
def create_apple_backend():
    """創建Apple Silicon後端"""
    return AppleBackend()