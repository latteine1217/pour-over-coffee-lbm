"""
標準4D記憶體布局適配器
傳統的f[q,i,j,k]記憶體布局策略
"""

import taichi as ti
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import config.core as config
from .memory_layouts import MemoryLayoutAdapter

class Standard4DAdapter(MemoryLayoutAdapter):
    """
    標準4D記憶體布局適配器
    
    採用傳統4D記憶體布局：f[q,i,j,k]
    - 兼容性最佳，通用性強
    - CPU和GPU都能有效運行
    """
    
    def __init__(self):
        super().__init__()
        self.layout_type = "standard_4d"
        
        # 4D場變數
        self.f = None
        self.f_new = None
        
    def allocate_fields(self) -> None:
        """分配標準4D記憶體布局"""
        print("🖥️  分配標準4D記憶體布局...")
        
        # 分配4D場
        self.f = ti.field(dtype=ti.f32, shape=(config.Q_3D, config.NX, config.NY, config.NZ))
        self.f_new = ti.field(dtype=ti.f32, shape=(config.Q_3D, config.NX, config.NY, config.NZ))
        
        print(f"   ✅ 完成4D場分配 [{config.Q_3D}×{config.NX}×{config.NY}×{config.NZ}]")
        
    @ti.func
    def get_f(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32) -> ti.f32:
        """4D格式獲取分布函數值"""
        return self.f[q, i, j, k]
        
    @ti.func
    def set_f(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32, value: ti.f32) -> None:
        """4D格式設置分布函數值"""
        self.f[q, i, j, k] = value
        
    @ti.func
    def get_f_new(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32) -> ti.f32:
        """4D格式獲取新分布函數值"""
        return self.f_new[q, i, j, k]
        
    @ti.func
    def set_f_new(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32, value: ti.f32) -> None:
        """4D格式設置新分布函數值"""
        self.f_new[q, i, j, k] = value
        
    def swap_fields(self) -> None:
        """交換4D場"""
        self.f, self.f_new = self.f_new, self.f
        
    def get_memory_usage(self) -> float:
        """計算4D布局記憶體使用量"""
        total_elements = config.Q_3D * config.NX * config.NY * config.NZ * 2
        bytes_per_element = 4
        total_bytes = total_elements * bytes_per_element
        return total_bytes / (1024**3)
        
    def get_layout_description(self) -> str:
        """返回4D布局描述"""
        return f"Standard 4D Layout: f[{config.Q_3D},{config.NX},{config.NY},{config.NZ}]"