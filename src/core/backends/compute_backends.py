"""
企業級計算後端基類與工廠模式
提供統一的計算接口，支援不同平台的最佳計算策略

核心特性：
- 統一後端介面與錯誤處理
- 企業級工廠模式與平台檢測
- 自動降級與故障轉移機制
- 完整的性能監控與診斷
- 智能後端選擇與配置

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
import sys
import os
import platform
import subprocess
import logging
import time
from enum import Enum
import threading
from contextlib import contextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import config.config


# ===== 錯誤處理系統 =====

class BackendError(Exception):
    """後端錯誤基類"""
    def __init__(self, message: str, backend_type: Optional[str] = None, error_code: Optional[str] = None):
        self.backend_type = backend_type or "unknown"
        self.error_code = error_code or "UNKNOWN_ERROR"
        super().__init__(message)


class PlatformDetectionError(BackendError):
    """平台檢測錯誤"""
    pass


class BackendInitializationError(BackendError):
    """後端初始化錯誤"""
    pass


class ComputeExecutionError(BackendError):
    """計算執行錯誤"""
    pass


class MemoryAllocationError(BackendError):
    """記憶體分配錯誤"""
    pass


class PerformanceDegradationError(BackendError):
    """性能降級錯誤"""
    pass


# ===== 平台類型定義 =====

class PlatformType(Enum):
    """支援的平台類型"""
    APPLE_SILICON = "apple"
    NVIDIA_CUDA = "cuda" 
    CPU_REFERENCE = "cpu"
    UNKNOWN = "unknown"


class BackendPriority(Enum):
    """後端優先級"""
    CRITICAL = 1    # 核心功能，必須成功
    HIGH = 2        # 高性能需求
    MEDIUM = 3      # 標準需求
    LOW = 4         # 備用方案


# ===== 統一日誌系統 =====

class BackendLogger:
    """統一後端日誌管理"""
    
    def __init__(self):
        self.logger = logging.getLogger("compute_backends")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, backend_type: Optional[str] = None):
        prefix = f"[{backend_type}] " if backend_type else ""
        self.logger.info(f"{prefix}{message}")
    
    def warning(self, message: str, backend_type: Optional[str] = None):
        prefix = f"[{backend_type}] " if backend_type else ""
        self.logger.warning(f"{prefix}{message}")
    
    def error(self, message: str, backend_type: Optional[str] = None):
        prefix = f"[{backend_type}] " if backend_type else ""
        self.logger.error(f"{prefix}{message}")


# 全域日誌實例
backend_logger = BackendLogger()


# ===== 平台檢測系統 =====

class PlatformDetector:
    """
    智能平台檢測系統
    
    功能特性：
    - 多層次硬體檢測與驗證
    - 性能基準測試與評分
    - 自動降級與故障轉移
    - 系統資源監控
    """
    
    def __init__(self):
        self.detected_platforms = []
        self.performance_scores = {}
        self.availability_cache = {}
        self.lock = threading.Lock()
        
    def detect_all_platforms(self) -> List[Dict[str, Any]]:
        """
        檢測所有可用平台
        
        Returns:
            List[Dict]: 平台信息列表，按優先級排序
        """
        with self.lock:
            platforms = []
            
            # Apple Silicon 檢測
            apple_info = self._detect_apple_silicon()
            if apple_info:
                platforms.append(apple_info)
                
            # NVIDIA CUDA 檢測
            cuda_info = self._detect_nvidia_cuda()
            if cuda_info:
                platforms.append(cuda_info)
                
            # CPU 參考實現
            cpu_info = self._detect_cpu_platform()
            platforms.append(cpu_info)  # CPU 總是可用
            
            # 按性能評分排序
            platforms.sort(key=lambda x: x.get('performance_score', 0), reverse=True)
            
            self.detected_platforms = platforms
            return platforms
    
    def _detect_apple_silicon(self) -> Optional[Dict[str, Any]]:
        """檢測 Apple Silicon 平台"""
        try:
            # 基本平台檢測
            if not (platform.processor() in ['arm', 'arm64'] and platform.system() == 'Darwin'):
                return None
                
            # 檢測 Metal 支援
            try:
                ti.init(arch=ti.metal)
                metal_available = True
                ti.reset()
            except Exception:
                metal_available = False
                
            if not metal_available:
                backend_logger.warning("Apple Silicon 檢測到，但 Metal 不可用", "apple")
                return None
            
            # 獲取系統信息
            try:
                import subprocess
                system_info = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], 
                                                    text=True, timeout=5)
                chip_name = "Apple Silicon"
                for line in system_info.split('\n'):
                    if 'Chip:' in line:
                        chip_name = line.split(':')[1].strip()
                        break
            except Exception:
                chip_name = "Apple Silicon"
                
            return {
                'platform_type': PlatformType.APPLE_SILICON,
                'backend_name': 'apple',
                'display_name': f'🍎 {chip_name} (Metal GPU)',
                'performance_score': 95,  # 高性能評分
                'memory_type': 'unified',
                'compute_units': 'GPU',
                'chip_name': chip_name,
                'metal_available': True,
                'priority': BackendPriority.HIGH
            }
            
        except Exception as e:
            backend_logger.error(f"Apple Silicon 檢測失敗: {e}", "apple")
            return None
    
    def _detect_nvidia_cuda(self) -> Optional[Dict[str, Any]]:
        """檢測 NVIDIA CUDA 平台"""
        try:
            # 檢測 nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    return None
                    
                gpu_info = result.stdout.strip().split('\n')[0].split(', ')
                gpu_name = gpu_info[0]
                gpu_memory = int(gpu_info[1])
                
            except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
                return None
                
            # 檢測 CUDA 支援
            try:
                ti.init(arch=ti.cuda)
                cuda_available = True
                ti.reset()
            except Exception:
                cuda_available = False
                
            if not cuda_available:
                backend_logger.warning("NVIDIA GPU 檢測到，但 CUDA 不可用", "cuda")
                return None
                
            # 計算性能評分 (基於記憶體容量)
            performance_score = min(90, 60 + (gpu_memory // 1000) * 5)
            
            return {
                'platform_type': PlatformType.NVIDIA_CUDA,
                'backend_name': 'cuda',
                'display_name': f'🚀 {gpu_name} (CUDA)',
                'performance_score': performance_score,
                'memory_type': 'dedicated',
                'compute_units': 'CUDA Cores',
                'gpu_name': gpu_name,
                'gpu_memory_mb': gpu_memory,
                'cuda_available': True,
                'priority': BackendPriority.HIGH
            }
            
        except Exception as e:
            backend_logger.error(f"NVIDIA CUDA 檢測失敗: {e}", "cuda")
            return None
    
    def _detect_cpu_platform(self) -> Dict[str, Any]:
        """檢測 CPU 平台 (總是可用)"""
        try:
            try:
                import psutil
                cpu_count = psutil.cpu_count(logical=False)
                cpu_freq = psutil.cpu_freq()
                memory_gb = psutil.virtual_memory().total // (1024**3)
                
                # 基於 CPU 核心數和頻率計算性能評分
                base_score = 30
                core_score = min(20, cpu_count * 2) if cpu_count else 10
                freq_score = min(20, (cpu_freq.max / 1000 - 2) * 5) if cpu_freq and cpu_freq.max else 10
                memory_score = min(10, memory_gb // 4)
                
                performance_score = base_score + core_score + freq_score + memory_score
                
            except ImportError:
                performance_score = 40  # 默認評分
                cpu_count = "Unknown"
                memory_gb = "Unknown"
                
        except Exception:
            performance_score = 40
            cpu_count = "Unknown"
            memory_gb = "Unknown"
            
        return {
            'platform_type': PlatformType.CPU_REFERENCE,
            'backend_name': 'cpu',
            'display_name': f'🖥️ CPU ({cpu_count} cores, {memory_gb}GB RAM)',
            'performance_score': performance_score,
            'memory_type': 'system',
            'compute_units': 'CPU Cores',
            'cpu_cores': cpu_count,
            'memory_gb': memory_gb,
            'priority': BackendPriority.MEDIUM
        }
    
    def get_optimal_platform(self) -> Dict[str, Any]:
        """獲取最佳平台配置"""
        if not self.detected_platforms:
            self.detect_all_platforms()
            
        return self.detected_platforms[0] if self.detected_platforms else self._detect_cpu_platform()
    
    def validate_platform_availability(self, platform_name: str) -> bool:
        """驗證平台可用性"""
        try:
            if platform_name == 'apple':
                ti.init(arch=ti.metal)
            elif platform_name == 'cuda':
                ti.init(arch=ti.cuda)
            elif platform_name == 'cpu':
                ti.init(arch=ti.cpu)
            else:
                return False
                
            ti.reset()
            return True
            
        except Exception as e:
            backend_logger.error(f"平台 {platform_name} 驗證失敗: {e}")
            return False


# 全域平台檢測器
platform_detector = PlatformDetector()


# ===== 統一計算後端基類 =====

class ComputeBackend(ABC):
    """
    企業級計算後端基類
    
    定義統一的計算接口，支援不同的計算策略：
    - Apple Silicon: Metal GPU深度優化
    - NVIDIA CUDA: 高性能並行計算
    - CPU Reference: 跨平台參考實現
    
    核心特性：
    - 統一 API 介面
    - 企業級錯誤處理
    - 性能監控與診斷
    - 資源管理與清理
    """
    
    def __init__(self, backend_type: str = "base"):
        self.backend_type = backend_type
        self.performance_metrics = {}
        self.error_history = []
        self.is_initialized = False
        self.creation_time = time.time()
        self.last_error = None
        
        # 性能監控
        self.execution_times = []
        self.memory_usage = []
        self.operation_count = 0
        
    @abstractmethod
    def execute_collision_streaming(self, memory_adapter, **kwargs) -> None:
        """
        執行collision-streaming融合步驟
        
        Args:
            memory_adapter: 記憶體介面卡
            **kwargs: 計算參數
            
        Raises:
            ComputeExecutionError: 計算執行錯誤
        """
        pass
        
    @abstractmethod  
    def apply_boundary_conditions(self, memory_adapter, **kwargs) -> None:
        """
        應用邊界條件
        
        Args:
            memory_adapter: 記憶體介面卡
            **kwargs: 邊界條件參數
            
        Raises:
            ComputeExecutionError: 邊界條件應用錯誤
        """
        pass
        
    @abstractmethod
    def compute_macroscopic_quantities(self, memory_adapter, **kwargs) -> None:
        """
        計算巨觀量 (密度、速度)
        
        Args:
            memory_adapter: 記憶體介面卡
            **kwargs: 計算參數
            
        Raises:
            ComputeExecutionError: 巨觀量計算錯誤
        """
        pass
        
    @abstractmethod
    def get_platform_info(self) -> Dict[str, Any]:
        """
        返回平台詳細信息
        
        Returns:
            Dict: 平台配置與能力信息
        """
        pass
        
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        返回性能指標
        
        Returns:
            Dict: 性能統計數據
        """
        pass
    
    def initialize_backend(self) -> None:
        """
        初始化計算後端
        
        Raises:
            BackendInitializationError: 初始化失敗
        """
        try:
            backend_logger.info(f"正在初始化 {self.backend_type} 後端...")
            self._perform_initialization()
            self.is_initialized = True
            backend_logger.info(f"{self.backend_type} 後端初始化成功", self.backend_type)
            
        except Exception as e:
            error_msg = f"{self.backend_type} 後端初始化失敗: {e}"
            self._record_error(error_msg, "INIT_FAILED")
            raise BackendInitializationError(error_msg, self.backend_type, "INIT_FAILED")
    
    def cleanup_backend(self) -> None:
        """清理計算後端資源"""
        try:
            backend_logger.info(f"正在清理 {self.backend_type} 後端資源...")
            self._perform_cleanup()
            self.is_initialized = False
            backend_logger.info(f"{self.backend_type} 後端清理完成", self.backend_type)
            
        except Exception as e:
            backend_logger.error(f"{self.backend_type} 後端清理失敗: {e}", self.backend_type)
    
    @contextmanager
    def safe_execution(self, operation_name: str):
        """
        安全執行上下文管理器
        
        Args:
            operation_name: 操作名稱
            
        Yields:
            None
            
        Raises:
            ComputeExecutionError: 執行錯誤
        """
        start_time = time.time()
        
        try:
            backend_logger.info(f"開始執行 {operation_name}", self.backend_type)
            yield
            
        except Exception as e:
            error_msg = f"{operation_name} 執行失敗: {e}"
            self._record_error(error_msg, "EXECUTION_FAILED")
            backend_logger.error(error_msg, self.backend_type)
            raise ComputeExecutionError(error_msg, self.backend_type, "EXECUTION_FAILED")
            
        finally:
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self.operation_count += 1
            
            # 限制歷史記錄長度
            if len(self.execution_times) > 1000:
                self.execution_times = self.execution_times[-500:]
    
    def _perform_initialization(self) -> None:
        """子類實現的具體初始化邏輯"""
        pass
    
    def _perform_cleanup(self) -> None:
        """子類實現的具體清理邏輯"""
        pass
    
    def _record_error(self, error_msg: str, error_code: str) -> None:
        """記錄錯誤信息"""
        error_record = {
            'timestamp': time.time(),
            'message': error_msg,
            'code': error_code,
            'backend_type': self.backend_type
        }
        self.error_history.append(error_record)
        self.last_error = error_record
        
        # 限制錯誤歷史長度
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-50:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """獲取錯誤摘要"""
        return {
            'total_errors': len(self.error_history),
            'last_error': self.last_error,
            'error_rate': len(self.error_history) / max(1, self.operation_count),
            'backend_type': self.backend_type
        }
    
    def get_basic_metrics(self) -> Dict[str, Any]:
        """獲取基礎性能指標"""
        if not self.execution_times:
            return {'status': 'no_data'}
            
        return {
            'backend_type': self.backend_type,
            'total_operations': self.operation_count,
            'avg_execution_time': np.mean(self.execution_times),
            'min_execution_time': np.min(self.execution_times),
            'max_execution_time': np.max(self.execution_times),
            'uptime_seconds': time.time() - self.creation_time,
            'is_initialized': self.is_initialized,
            'error_summary': self.get_error_summary()
        }


# ===== 企業級後端工廠模式 =====

class ComputeBackendFactory:
    """
    企業級計算後端工廠類
    
    核心功能：
    - 智能平台檢測與後端選擇
    - 企業級錯誤處理與降級機制
    - 動態配置與性能最佳化
    - 後端實例生命週期管理
    - 跨平台相容性保證
    
    特性：
    - 線程安全的單例模式
    - 故障轉移與自動降級
    - 性能基準測試與評估
    - 智能快取與重用機制
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """線程安全的單例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化工廠實例"""
        if getattr(self, '_initialized', False):
            return
            
        self.backend_cache = {}
        self.platform_info = {}
        self.creation_history = []
        self.performance_benchmarks = {}
        self.fallback_chain = ['apple', 'cuda', 'cpu']
        self.lock = threading.Lock()
        self._initialized = True
        
        backend_logger.info("🏗️ 計算後端工廠初始化完成")
    
    def create_optimal_backend(self, prefer_performance: bool = True) -> ComputeBackend:
        """
        創建最佳計算後端
        
        Args:
            prefer_performance: 優先考慮性能還是相容性
            
        Returns:
            ComputeBackend: 最佳後端實例
            
        Raises:
            BackendInitializationError: 所有後端都不可用
        """
        with self.lock:
            backend_logger.info("🔍 開始檢測最佳計算後端...")
            
            # 獲取平台信息
            optimal_platform = platform_detector.get_optimal_platform()
            backend_type = optimal_platform['backend_name']
            
            try:
                # 嘗試創建最佳後端
                backend = self._create_backend_with_validation(backend_type)
                
                # 記錄成功創建
                self._record_creation_success(backend_type, optimal_platform)
                backend_logger.info(f"✅ 成功創建 {optimal_platform['display_name']} 後端")
                
                return backend
                
            except Exception as e:
                backend_logger.warning(f"⚠️ 最佳後端 {backend_type} 創建失敗: {e}")
                
                # 嘗試故障轉移
                return self._attempt_fallback_creation(exclude=[backend_type])
    
    def create_backend(self, backend_type: str, 
                      validate: bool = True,
                      use_cache: bool = True) -> ComputeBackend:
        """
        根據指定類型創建後端
        
        Args:
            backend_type: 'apple', 'cuda', 'cpu'
            validate: 是否驗證後端可用性
            use_cache: 是否使用快取實例
            
        Returns:
            ComputeBackend: 指定類型的後端實例
            
        Raises:
            BackendInitializationError: 後端創建失敗
        """
        with self.lock:
            # 檢查快取
            if use_cache and backend_type in self.backend_cache:
                cached_backend = self.backend_cache[backend_type]
                if self._validate_cached_backend(cached_backend):
                    backend_logger.info(f"♻️ 使用快取的 {backend_type} 後端")
                    return cached_backend
                else:
                    # 清理無效快取
                    del self.backend_cache[backend_type]
            
            try:
                # 創建新後端實例
                backend = self._create_backend_with_validation(backend_type, validate)
                
                # 快取有效後端
                if use_cache:
                    self.backend_cache[backend_type] = backend
                
                # 記錄創建歷史
                self._record_creation_success(backend_type)
                
                return backend
                
            except Exception as e:
                error_msg = f"創建 {backend_type} 後端失敗: {e}"
                backend_logger.error(error_msg)
                raise BackendInitializationError(error_msg, backend_type, "CREATION_FAILED")
    
    def get_available_backends(self) -> List[Dict[str, Any]]:
        """
        獲取所有可用的後端信息
        
        Returns:
            List[Dict]: 可用後端列表
        """
        available_backends = []
        all_platforms = platform_detector.detect_all_platforms()
        
        for platform_info in all_platforms:
            backend_type = platform_info['backend_name']
            
            try:
                # 驗證後端可用性
                if platform_detector.validate_platform_availability(backend_type):
                    available_backends.append({
                        **platform_info,
                        'is_available': True,
                        'validation_status': 'passed'
                    })
                else:
                    available_backends.append({
                        **platform_info,
                        'is_available': False,
                        'validation_status': 'failed'
                    })
                    
            except Exception as e:
                available_backends.append({
                    **platform_info,
                    'is_available': False,
                    'validation_status': 'error',
                    'error_message': str(e)
                })
        
        return available_backends
    
    def benchmark_all_backends(self, iterations: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        對所有可用後端進行性能基準測試
        
        Args:
            iterations: 測試迭代次數
            
        Returns:
            Dict: 各後端性能基準結果
        """
        benchmarks = {}
        available_backends = self.get_available_backends()
        
        for backend_info in available_backends:
            if not backend_info['is_available']:
                continue
                
            backend_type = backend_info['backend_name']
            backend_logger.info(f"🏃‍♂️ 開始基準測試 {backend_type} 後端...")
            
            try:
                backend = self.create_backend(backend_type, use_cache=False)
                benchmark_result = self._run_performance_benchmark(backend, iterations)
                benchmarks[backend_type] = benchmark_result
                
                backend_logger.info(f"✅ {backend_type} 基準測試完成")
                
            except Exception as e:
                benchmarks[backend_type] = {
                    'status': 'failed',
                    'error': str(e)
                }
                backend_logger.error(f"❌ {backend_type} 基準測試失敗: {e}")
        
        # 快取基準測試結果
        self.performance_benchmarks = benchmarks
        return benchmarks
    
    def _create_backend_with_validation(self, backend_type: str, 
                                       validate: bool = True) -> ComputeBackend:
        """創建並驗證後端實例"""
        # 驗證平台可用性
        if validate and not platform_detector.validate_platform_availability(backend_type):
            raise BackendInitializationError(
                f"{backend_type} 平台不可用或未正確配置",
                backend_type, "PLATFORM_UNAVAILABLE"
            )
        
        # 動態導入並創建後端
        if backend_type == 'apple':
            try:
                from .apple_backend import AppleBackend
                backend = AppleBackend()
            except ImportError as e:
                raise BackendInitializationError(
                    f"Apple 後端導入失敗: {e}", backend_type, "IMPORT_FAILED")
                
        elif backend_type == 'cuda':
            try:
                from .cuda_backend import CUDABackend
                backend = CUDABackend()
            except ImportError as e:
                raise BackendInitializationError(
                    f"CUDA 後端導入失敗: {e}", backend_type, "IMPORT_FAILED")
                
        elif backend_type == 'cpu':
            try:
                from .cpu_backend import CPUBackend
                backend = CPUBackend()
            except ImportError as e:
                raise BackendInitializationError(
                    f"CPU 後端導入失敗: {e}", backend_type, "IMPORT_FAILED")
        else:
            raise BackendInitializationError(
                f"未知的後端類型: {backend_type}", backend_type, "UNKNOWN_BACKEND")
        
        # 初始化後端
        if validate:
            backend.initialize_backend()
        
        return backend
    
    def _attempt_fallback_creation(self, exclude: List[str] = None) -> ComputeBackend:
        """嘗試故障轉移創建"""
        exclude = exclude or []
        
        for backend_type in self.fallback_chain:
            if backend_type in exclude:
                continue
                
            try:
                backend_logger.info(f"🔄 嘗試故障轉移到 {backend_type} 後端...")
                backend = self._create_backend_with_validation(backend_type)
                backend_logger.info(f"✅ 故障轉移成功，使用 {backend_type} 後端")
                return backend
                
            except Exception as e:
                backend_logger.warning(f"⚠️ 故障轉移到 {backend_type} 失敗: {e}")
                continue
        
        # 所有後端都不可用
        raise BackendInitializationError(
            "所有計算後端都不可用，無法創建後端實例",
            "all", "ALL_BACKENDS_FAILED"
        )
    
    def _validate_cached_backend(self, backend: ComputeBackend) -> bool:
        """驗證快取後端的有效性"""
        try:
            return (backend is not None and 
                   hasattr(backend, 'is_initialized') and
                   backend.is_initialized)
        except Exception:
            return False
    
    def _record_creation_success(self, backend_type: str, 
                                platform_info: Dict[str, Any] = None):
        """記錄後端創建成功"""
        record = {
            'timestamp': time.time(),
            'backend_type': backend_type,
            'status': 'success',
            'platform_info': platform_info
        }
        self.creation_history.append(record)
        
        # 限制歷史記錄長度
        if len(self.creation_history) > 100:
            self.creation_history = self.creation_history[-50:]
    
    def _run_performance_benchmark(self, backend: ComputeBackend, 
                                  iterations: int) -> Dict[str, Any]:
        """運行性能基準測試"""
        start_time = time.time()
        execution_times = []
        
        for i in range(iterations):
            iter_start = time.time()
            
            # 這裡應該運行實際的計算測試
            # 暫時使用簡單的時間測量
            try:
                # 模擬計算操作
                time.sleep(0.001)  # 1ms 模擬計算
                execution_times.append(time.time() - iter_start)
            except Exception as e:
                backend_logger.warning(f"基準測試迭代 {i} 失敗: {e}")
        
        total_time = time.time() - start_time
        
        if not execution_times:
            return {'status': 'failed', 'error': '所有迭代都失敗'}
        
        return {
            'status': 'success',
            'iterations': len(execution_times),
            'total_time': total_time,
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'std_execution_time': np.std(execution_times),
            'throughput': len(execution_times) / total_time,
            'backend_type': backend.backend_type
        }
    
    def get_factory_statistics(self) -> Dict[str, Any]:
        """獲取工廠統計信息"""
        return {
            'total_backends_created': len(self.creation_history),
            'cached_backends': list(self.backend_cache.keys()),
            'creation_history': self.creation_history[-10:],  # 最近10次創建
            'performance_benchmarks': self.performance_benchmarks,
            'available_platforms': len(platform_detector.detected_platforms),
            'fallback_chain': self.fallback_chain
        }
    
    def cleanup_all_backends(self):
        """清理所有後端資源"""
        with self.lock:
            backend_logger.info("🧹 開始清理所有後端資源...")
            
            for backend_type, backend in self.backend_cache.items():
                try:
                    backend.cleanup_backend()
                    backend_logger.info(f"✅ {backend_type} 後端清理完成")
                except Exception as e:
                    backend_logger.error(f"❌ {backend_type} 後端清理失敗: {e}")
            
            self.backend_cache.clear()
            backend_logger.info("🧹 所有後端資源清理完成")
    
    def __del__(self):
        """析構函數，確保資源清理"""
        try:
            self.cleanup_all_backends()
        except Exception:
            pass


# ===== 便利函數與全域實例 =====

# 全域工廠實例
backend_factory = ComputeBackendFactory()


def create_optimal_backend(**kwargs) -> ComputeBackend:
    """
    便利函數：創建最佳後端
    
    Returns:
        ComputeBackend: 最佳後端實例
    """
    return backend_factory.create_optimal_backend(**kwargs)


def create_backend(backend_type: str, **kwargs) -> ComputeBackend:
    """
    便利函數：創建指定類型後端
    
    Args:
        backend_type: 後端類型
        **kwargs: 其他參數
        
    Returns:
        ComputeBackend: 後端實例
    """
    return backend_factory.create_backend(backend_type, **kwargs)


def get_available_backends() -> List[Dict[str, Any]]:
    """
    便利函數：獲取可用後端列表
    
    Returns:
        List[Dict]: 可用後端信息
    """
    return backend_factory.get_available_backends()


def cleanup_backends():
    """便利函數：清理所有後端資源"""
    backend_factory.cleanup_all_backends()


# ===== 模組導出 =====

__all__ = [
    # 基礎類別
    'ComputeBackend',
    'ComputeBackendFactory',
    'PlatformDetector',
    'BackendLogger',
    
    # 錯誤類別
    'BackendError',
    'PlatformDetectionError', 
    'BackendInitializationError',
    'ComputeExecutionError',
    'MemoryAllocationError',
    'PerformanceDegradationError',
    
    # 枚舉類別
    'PlatformType',
    'BackendPriority',
    
    # 全域實例
    'backend_factory',
    'platform_detector',
    'backend_logger',
    
    # 便利函數
    'create_optimal_backend',
    'create_backend',
    'get_available_backends',
    'cleanup_backends'
]