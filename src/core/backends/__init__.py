"""
LBM計算後端系統
支援不同平台的最佳計算策略
"""

from .compute_backends import ComputeBackend, ComputeBackendFactory
from .apple_backend import AppleBackend
from .cuda_backend import CUDABackend  
from .cpu_backend import CPUBackend

__all__ = [
    'ComputeBackend',
    'ComputeBackendFactory',
    'AppleBackend', 
    'CUDABackend',
    'CPUBackend'
]