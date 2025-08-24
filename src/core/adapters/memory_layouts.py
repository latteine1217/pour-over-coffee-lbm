"""
記憶體布局適配器基類與工廠模式
提供統一的記憶體存取接口，支援不同平台的最佳布局策略
"""

import taichi as ti
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import config.core as config

class MemoryLayoutAdapter(ABC):
    """
    記憶體布局適配器基類
    
    定義統一的記憶體存取接口，支援不同的記憶體布局策略：
    - SoA (Structure of Arrays): f[q][i,j,k] - Apple Silicon優化
    - 4D Standard: f[q,i,j,k] - 傳統標準布局  
    - GPU Domain: 分域管理 - CUDA雙GPU優化
    """
    
    def __init__(self):
        self.layout_type = "base"
        self.performance_profile = {}
        
    @abstractmethod
    def allocate_fields(self) -> None:
        """分配記憶體場變數"""
        pass
        
    @ti.func  
    def get_f(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32) -> ti.f32:
        """獲取分布函數值 f[q][i,j,k]"""
        pass
        
    @ti.func
    def set_f(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32, value: ti.f32) -> None:
        """設置分布函數值 f[q][i,j,k] = value"""
        pass
        
    @ti.func
    def get_f_new(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32) -> ti.f32:
        """獲取新分布函數值 f_new[q][i,j,k]"""
        pass
        
    @ti.func
    def set_f_new(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32, value: ti.f32) -> None:
        """設置新分布函數值 f_new[q][i,j,k] = value"""
        pass
        
    @abstractmethod
    def get_memory_usage(self) -> float:
        """返回記憶體使用量 (GB)"""
        pass
        
    @abstractmethod
    def get_layout_description(self) -> str:
        """返回布局描述"""
        pass
        
    def get_performance_profile(self) -> Dict[str, Any]:
        """返回性能剖析數據"""
        return self.performance_profile
        
    def swap_fields(self) -> None:
        """交換 f 和 f_new 場 (streaming後)"""
        pass

class MemoryLayoutFactory:
    """
    記憶體布局工廠類
    
    根據平台和用戶配置自動選擇最佳的記憶體布局策略
    """
    
    @staticmethod
    def create_optimal_adapter() -> MemoryLayoutAdapter:
        """
        自動檢測並創建最佳記憶體布局適配器
        
        Returns:
            MemoryLayoutAdapter: 最佳適配器實例
        """
        # 檢測Apple Silicon
        try:
            import platform
            if platform.processor() in ['arm', 'arm64'] and platform.system() == 'Darwin':
                from .soa_adapter import SoAAdapter
                print("🍎 檢測到Apple Silicon，使用SoA優化布局")
                return SoAAdapter()
        except ImportError:
            pass
            
        # 檢測CUDA
        try:
            import pycuda.driver as cuda
            cuda.init()
            if cuda.Device.count() >= 2:
                from .gpu_adapter import GPUDomainAdapter
                print("🚀 檢測到多GPU CUDA，使用分域布局")
                return GPUDomainAdapter()
        except ImportError:
            pass
            
        # 默認標準4D布局
        from .standard_adapter import Standard4DAdapter
        print("🖥️  使用標準4D記憶體布局")
        return Standard4DAdapter()
        
    @staticmethod
    def create_adapter(layout_type: str) -> MemoryLayoutAdapter:
        """
        根據指定類型創建適配器
        
        Args:
            layout_type: 'soa', 'standard', 'gpu_domain'
            
        Returns:
            MemoryLayoutAdapter: 指定類型的適配器實例
        """
        if layout_type == 'soa':
            from .soa_adapter import SoAAdapter
            return SoAAdapter()
        elif layout_type == 'standard':
            from .standard_adapter import Standard4DAdapter  
            return Standard4DAdapter()
        elif layout_type == 'gpu_domain':
            from .gpu_adapter import GPUDomainAdapter
            return GPUDomainAdapter()
        else:
            raise ValueError(f"未知的記憶體布局類型: {layout_type}")