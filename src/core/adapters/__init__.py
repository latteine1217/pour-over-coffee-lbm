"""
LBM記憶體布局適配器系統
支援不同平台的最佳記憶體布局策略
"""

from .memory_layouts import MemoryLayoutAdapter, MemoryLayoutFactory
from .soa_adapter import SoAAdapter  
from .standard_adapter import Standard4DAdapter
from .gpu_adapter import GPUDomainAdapter

__all__ = [
    'MemoryLayoutAdapter',
    'MemoryLayoutFactory', 
    'SoAAdapter',
    'Standard4DAdapter',
    'GPUDomainAdapter'
]