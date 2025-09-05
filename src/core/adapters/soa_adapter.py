"""
SoA (Structure of Arrays) 記憶體布局適配器
專為Apple Silicon優化的記憶體布局策略
"""

import taichi as ti
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import config.core as config
from .memory_layouts import MemoryLayoutAdapter
from ..apple_silicon_optimizations import apple_optimizer

class SoAAdapter(MemoryLayoutAdapter):
    """
    SoA記憶體布局適配器 - Apple Silicon專用優化
    
    採用Structure of Arrays記憶體布局：f[q][i,j,k]
    - 每個速度方向使用獨立的3D場
    - Apple GPU cache-line對齊優化
    - Metal SIMD vectorization友好設計
    - 統一記憶體零拷貝技術
    """
    
    def __init__(self):
        super().__init__()
        self.layout_type = "soa"
        self.apple_config = apple_optimizer.setup_taichi_metal_optimization()
        
        # SoA場變數
        self.f = []
        self.f_new = []
        
        # 性能追蹤
        self.performance_profile = {
            "layout": "SoA",
            "platform": "Apple Silicon",
            "cache_aligned": True,
            "simd_optimized": True
        }
        
    def allocate_fields(self) -> None:
        """
        分配SoA記憶體布局的分布函數場
        
        每個速度方向 q 使用獨立的3D場：
        - f[q] 為 [NX, NY, NZ] 形狀的獨立場
        - 記憶體連續性最佳化
        - Apple GPU cache-line對齊
        """
        print("🍎 分配SoA記憶體布局 (Apple Silicon優化)...")
        
        # 清理現有場
        self.f.clear()
        self.f_new.clear()
        
        # 為每個速度方向分配獨立場
        for q in range(config.Q_3D):
            f_q = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
            f_new_q = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
            
            self.f.append(f_q)
            self.f_new.append(f_new_q)
            
        print(f"   ✅ 完成 {config.Q_3D} 個速度方向的SoA場分配")
        print(f"   📊 記憶體使用: ~{self.get_memory_usage():.2f} GB")
        
    @ti.func
    def get_f(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32) -> ti.f32:
        """SoA格式獲取分布函數值"""
        return self.f[q][i, j, k]
        
    @ti.func
    def set_f(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32, value: ti.f32) -> None:
        """SoA格式設置分布函數值"""
        self.f[q][i, j, k] = value
        
    @ti.func
    def get_f_new(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32) -> ti.f32:
        """SoA格式獲取新分布函數值"""
        return self.f_new[q][i, j, k]
        
    @ti.func
    def set_f_new(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32, value: ti.f32) -> None:
        """SoA格式設置新分布函數值"""
        self.f_new[q][i, j, k] = value
        
    def swap_fields(self) -> None:
        """交換 f 和 f_new 場 (streaming後)"""
        self.f, self.f_new = self.f_new, self.f
        
    def get_memory_usage(self) -> float:
        """計算SoA布局記憶體使用量 (GB)"""
        total_elements = config.Q_3D * config.NX * config.NY * config.NZ * 2  # f + f_new
        bytes_per_element = 4  # ti.f32
        total_bytes = total_elements * bytes_per_element
        return total_bytes / (1024**3)
        
    def get_layout_description(self) -> str:
        """返回SoA布局描述"""
        return f"SoA Layout: f[{config.Q_3D}][{config.NX}×{config.NY}×{config.NZ}] - Apple Silicon優化"