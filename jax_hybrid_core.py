"""
JAX混合超級計算核心 - XLA編譯器 + Apple Silicon Metal最佳化
結合JAX的極致編譯優化與Taichi的GPU並行能力
開發：opencode + GitHub Copilot
"""

# 檢查JAX是否可用，如果沒有則提供fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, device_put
    from jax import config as jax_config
    JAX_AVAILABLE = True
    
    # 設定Apple Silicon Metal後端
    try:
        jax_config.update('jax_platform_name', 'metal')
        print("🍎 JAX Metal後端已啟用")
    except Exception:
        print("⚠️  JAX Metal後端不可用，使用CPU")
        
except ImportError:
    print("⚠️  JAX未安裝，使用Taichi純實現")
    JAX_AVAILABLE = False
    import numpy as jnp  # fallback

import taichi as ti
import numpy as np
import config as config  # 使用統一入口，避免相容層警告
from typing import Optional, Union, Tuple

@ti.data_oriented
class JAXHybridSuperCore:
    """
    JAX-Taichi混合超級計算核心
    
    結合優勢:
    1. JAX XLA編譯器極致優化
    2. Taichi GPU並行執行效率
    3. Apple Silicon Metal專用路徑
    4. 自動微分能力 (為未來AI增強準備)
    5. 函數式編程數值穩定性
    """
    
    def __init__(self):
        print("🚀 初始化JAX混合超級計算核心...")
        
        self.jax_enabled = JAX_AVAILABLE
        self.device_info = self._detect_optimal_backend()
        
        if self.jax_enabled:
            print(f"✅ JAX {jax.__version__} + XLA編譯器已啟用")
            print(f"   後端: {jax.default_backend()}")
            self._init_jax_optimized_functions()
        else:
            print("📐 使用Taichi純實現")
            
        self._init_hybrid_constants()
        print("✅ JAX混合核心初始化完成")
    
    def _detect_optimal_backend(self):
        """檢測最佳計算後端"""
        device_info = {
            'platform': 'cpu',
            'has_metal': False,
            'has_cuda': False,
            'memory_gb': 16
        }
        
        if self.jax_enabled:
            try:
                devices = jax.devices()
                if any('metal' in str(device).lower() for device in devices):
                    device_info['platform'] = 'metal'
                    device_info['has_metal'] = True
                elif any('gpu' in str(device).lower() for device in devices):
                    device_info['platform'] = 'gpu' 
                    device_info['has_cuda'] = True
            except:
                pass
                
        print(f"🔍 檢測到最佳後端: {device_info['platform']}")
        return device_info
    
    def _init_jax_optimized_functions(self):
        """初始化JAX XLA優化函數"""
        if not self.jax_enabled:
            return
            
        print("  🔧 編譯JAX XLA優化函數...")
        
        # JIT編譯的平衡態計算
        @jit
        def equilibrium_jax(rho, ux, uy, uz, cx, cy, cz, w):
            """XLA優化的平衡態分布計算"""
            cu = ux * cx + uy * cy + uz * cz
            u_sqr = ux*ux + uy*uy + uz*uz
            
            # 優化的平衡態公式
            feq = w * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sqr)
            return feq
        
        @jit  
        def collision_step_jax(f, rho, ux, uy, uz, tau, cx, cy, cz, w):
            """XLA優化的collision步驟"""
            # 向量化平衡態計算
            feq = equilibrium_jax(rho, ux, uy, uz, cx, cy, cz, w)
            
            # BGK collision
            f_new = f - (f - feq) / tau
            return f_new
        
        @jit
        def macroscopic_moments_jax(f, cx, cy, cz):
            """XLA優化的巨觀量計算"""
            # 密度
            rho = jnp.sum(f, axis=0)
            
            # 動量
            momentum_x = jnp.sum(f * cx[:, None, None, None], axis=0)
            momentum_y = jnp.sum(f * cy[:, None, None, None], axis=0) 
            momentum_z = jnp.sum(f * cz[:, None, None, None], axis=0)
            
            # 避免除零
            rho_safe = jnp.where(rho > 1e-12, rho, 1.0)
            ux = momentum_x / rho_safe
            uy = momentum_y / rho_safe
            uz = momentum_z / rho_safe
            
            return rho, ux, uy, uz
        
        # 向量化版本 (並行處理多個節點)
        self.equilibrium_vectorized = vmap(equilibrium_jax, in_axes=(0, 0, 0, 0, None, None, None, None))
        self.collision_vectorized = vmap(collision_step_jax, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None))
        
        # 儲存編譯後的函數
        self.equilibrium_jax = equilibrium_jax
        self.collision_jax = collision_step_jax  
        self.macroscopic_jax = macroscopic_moments_jax
        
        print("    ✅ JAX XLA函數編譯完成")
    
    def _init_hybrid_constants(self):
        """初始化混合計算常數"""
        # JAX設備常數 (如果可用)
        if self.jax_enabled:
            self.cx_jax = device_put(jnp.array(config.CX_3D, dtype=jnp.float32))
            self.cy_jax = device_put(jnp.array(config.CY_3D, dtype=jnp.float32))
            self.cz_jax = device_put(jnp.array(config.CZ_3D, dtype=jnp.float32))
            self.w_jax = device_put(jnp.array(config.WEIGHTS_3D, dtype=jnp.float32))
        
        # Taichi GPU常數
        self.cx_ti = ti.field(dtype=ti.f32, shape=config.Q_3D)
        self.cy_ti = ti.field(dtype=ti.f32, shape=config.Q_3D)
        self.cz_ti = ti.field(dtype=ti.f32, shape=config.Q_3D)
        self.w_ti = ti.field(dtype=ti.f32, shape=config.Q_3D)
        
        self.cx_ti.from_numpy(config.CX_3D.astype(np.float32))
        self.cy_ti.from_numpy(config.CY_3D.astype(np.float32))
        self.cz_ti.from_numpy(config.CZ_3D.astype(np.float32))
        self.w_ti.from_numpy(config.WEIGHTS_3D.astype(np.float32))
    
    def compute_equilibrium_hybrid(self, rho, ux, uy, uz, method='auto'):
        """
        混合平衡態計算
        
        Args:
            rho, ux, uy, uz: 巨觀量場
            method: 'jax', 'taichi', 'auto'
        
        Returns:
            平衡態分布函數
        """
        if method == 'auto':
            method = 'jax' if self.jax_enabled else 'taichi'
        
        if method == 'jax' and self.jax_enabled:
            return self._compute_equilibrium_jax(rho, ux, uy, uz)
        else:
            return self._compute_equilibrium_taichi(rho, ux, uy, uz)
    
    def _compute_equilibrium_jax(self, rho, ux, uy, uz):
        """JAX XLA優化平衡態計算"""
        if not self.jax_enabled:
            raise RuntimeError("JAX not available")
        
        # 轉換為JAX數組
        rho_jax = device_put(jnp.array(rho))
        ux_jax = device_put(jnp.array(ux))
        uy_jax = device_put(jnp.array(uy))
        uz_jax = device_put(jnp.array(uz))
        
        # XLA編譯優化計算
        feq_list = []
        for q in range(config.Q_3D):
            feq_q = self.equilibrium_jax(
                rho_jax, ux_jax, uy_jax, uz_jax,
                self.cx_jax[q], self.cy_jax[q], self.cz_jax[q], self.w_jax[q]
            )
            feq_list.append(np.array(feq_q))
        
        return feq_list
    
    @ti.kernel
    def _compute_equilibrium_taichi(self, rho: ti.template(), 
                                   ux: ti.template(), uy: ti.template(), uz: ti.template()) -> ti.template():
        """Taichi GPU優化平衡態計算"""
        feq = ti.field(dtype=ti.f32, shape=(config.Q_3D, config.NX, config.NY, config.NZ))
        
        # Apple GPU最佳配置
        ti.loop_config(block_dim=128)
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            rho_local = rho[i, j, k]
            ux_local = ux[i, j, k]
            uy_local = uy[i, j, k]
            uz_local = uz[i, j, k]
            
            u_sqr = ux_local*ux_local + uy_local*uy_local + uz_local*uz_local
            
            for q in ti.static(range(config.Q_3D)):
                cu = ux_local * self.cx_ti[q] + uy_local * self.cy_ti[q] + uz_local * self.cz_ti[q]
                
                feq[q, i, j, k] = self.w_ti[q] * rho_local * (
                    1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sqr
                )
        
        return feq
    
    def benchmark_hybrid_performance(self, iterations=100):
        """基準測試混合性能"""
        print("🧪 JAX混合性能基準測試...")
        
        # 創建測試數據
        nx, ny, nz = 64, 64, 64
        rho = np.ones((nx, ny, nz), dtype=np.float32)
        ux = np.random.random((nx, ny, nz)).astype(np.float32) * 0.01
        uy = np.random.random((nx, ny, nz)).astype(np.float32) * 0.01
        uz = np.random.random((nx, ny, nz)).astype(np.float32) * 0.01
        
        results = {}
        
        # JAX測試
        if self.jax_enabled:
            import time
            start_time = time.time()
            
            for i in range(iterations):
                feq_jax = self._compute_equilibrium_jax(rho, ux, uy, uz)
            
            jax_time = time.time() - start_time
            results['jax'] = {
                'time': jax_time,
                'throughput': (nx * ny * nz * iterations) / jax_time
            }
            print(f"  JAX XLA: {jax_time:.3f}s, {results['jax']['throughput']:.0f} 格點/秒")
        
        # Taichi測試
        rho_ti = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        ux_ti = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        uy_ti = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        uz_ti = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        
        rho_ti.from_numpy(rho)
        ux_ti.from_numpy(ux)
        uy_ti.from_numpy(uy)
        uz_ti.from_numpy(uz)
        
        import time
        start_time = time.time()
        
        for i in range(iterations):
            feq_ti = self._compute_equilibrium_taichi(rho_ti, ux_ti, uy_ti, uz_ti)
        
        taichi_time = time.time() - start_time
        results['taichi'] = {
            'time': taichi_time,
            'throughput': (nx * ny * nz * iterations) / taichi_time
        }
        print(f"  Taichi: {taichi_time:.3f}s, {results['taichi']['throughput']:.0f} 格點/秒")
        
        # 比較
        if self.jax_enabled:
            speedup = taichi_time / jax_time
            print(f"  JAX加速比: {speedup:.2f}x")
        
        return results

# 延遲初始化的全域混合核心
_jax_hybrid_core = None

def get_hybrid_core():
    """獲取JAX混合計算核心 (延遲初始化)"""
    global _jax_hybrid_core
    if _jax_hybrid_core is None:
        _jax_hybrid_core = JAXHybridSuperCore()
    return _jax_hybrid_core

if __name__ == "__main__":
    # 測試混合核心
    core = JAXHybridSuperCore()
    
    if core.jax_enabled:
        print("🧪 執行混合性能測試...")
        results = core.benchmark_hybrid_performance(50)
        print("✅ 混合核心測試完成")
    else:
        print("📐 JAX未啟用，使用Taichi實現")