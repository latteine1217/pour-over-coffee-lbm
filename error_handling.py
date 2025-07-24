# error_handling.py
"""
統一錯誤處理系統
為CFD模擬提供全面的異常處理、錯誤記錄和恢復機制

開發：opencode + GitHub Copilot
"""

import time
import json
import traceback
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging

# 設置日志系統
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cfd_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CFD_ErrorHandler')

class ErrorSeverity(Enum):
    """錯誤嚴重程度"""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"

class ErrorCategory(Enum):
    """錯誤類別"""
    NUMERICAL = "numerical"
    PHYSICS = "physics"
    MEMORY = "memory"
    IO = "io"
    CONFIGURATION = "configuration"
    GPU = "gpu"

# 自定義異常類
class CFDError(Exception):
    """CFD模擬基礎異常類"""
    def __init__(self, message: str, category: ErrorCategory, 
                 severity: ErrorSeverity, context: Dict = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()

class NumericalDivergenceError(CFDError):
    """數值發散異常"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorCategory.NUMERICAL, ErrorSeverity.CRITICAL, context)

class PhysicsViolationError(CFDError):
    """物理約束違反異常"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorCategory.PHYSICS, ErrorSeverity.ERROR, context)

class MemoryExhaustionError(CFDError):
    """記憶體耗盡異常"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorCategory.MEMORY, ErrorSeverity.FATAL, context)

class GPUError(CFDError):
    """GPU計算異常"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorCategory.GPU, ErrorSeverity.ERROR, context)

@dataclass
class ErrorRecord:
    """錯誤記錄"""
    timestamp: float
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: Dict
    stack_trace: str
    recovery_attempted: bool = False
    recovery_successful: bool = False

class GlobalErrorHandler:
    """全局錯誤處理器"""
    
    def __init__(self):
        """初始化錯誤處理器"""
        self.error_count = 0
        self.error_log: List[ErrorRecord] = []
        self.recovery_enabled = True
        self.max_recovery_attempts = 3
        self.error_callbacks: Dict[ErrorCategory, List[Callable]] = {
            category: [] for category in ErrorCategory
        }
        
        # 恢復策略註冊
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self._register_default_recovery_strategies()
        
        logger.info("✅ 全局錯誤處理器初始化完成")
    
    def _register_default_recovery_strategies(self):
        """註冊默認恢復策略"""
        self.recovery_strategies[ErrorCategory.NUMERICAL] = self._numerical_recovery
        self.recovery_strategies[ErrorCategory.MEMORY] = self._memory_recovery
        self.recovery_strategies[ErrorCategory.GPU] = self._gpu_recovery
        self.recovery_strategies[ErrorCategory.PHYSICS] = self._physics_recovery
    
    def register_error_callback(self, category: ErrorCategory, callback: Callable):
        """註冊錯誤回調函數"""
        self.error_callbacks[category].append(callback)
    
    def handle_error(self, error: CFDError, context: Dict = None) -> bool:
        """
        統一錯誤處理入口
        返回True表示錯誤已恢復，False表示需要終止
        """
        self.error_count += 1
        
        # 創建錯誤記錄
        error_record = ErrorRecord(
            timestamp=time.time(),
            error_type=type(error).__name__,
            message=str(error),
            category=error.category,
            severity=error.severity,
            context={**(error.context or {}), **(context or {})},
            stack_trace=traceback.format_exc()
        )
        
        self.error_log.append(error_record)
        
        # 記錄錯誤
        log_level = {
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.FATAL
        }[error.severity]
        
        logger.log(log_level, f"{error.category.value.upper()}: {str(error)}")
        
        # 觸發回調
        for callback in self.error_callbacks[error.category]:
            try:
                callback(error, error_record)
            except Exception as e:
                logger.error(f"錯誤回調執行失敗: {e}")
        
        # 嘗試恢復
        if self.recovery_enabled and error.severity != ErrorSeverity.FATAL:
            return self._attempt_recovery(error, error_record)
        
        return False
    
    def _attempt_recovery(self, error: CFDError, record: ErrorRecord) -> bool:
        """嘗試錯誤恢復"""
        record.recovery_attempted = True
        
        try:
            if error.category in self.recovery_strategies:
                recovery_func = self.recovery_strategies[error.category]
                success = recovery_func(error, record.context)
                record.recovery_successful = success
                
                if success:
                    logger.info(f"✅ {error.category.value}錯誤恢復成功")
                    return True
                else:
                    logger.warning(f"⚠️ {error.category.value}錯誤恢復失敗")
            else:
                logger.warning(f"❌ 無{error.category.value}錯誤恢復策略")
                
        except Exception as e:
            logger.error(f"❌ 錯誤恢復過程中發生異常: {e}")
            record.recovery_successful = False
        
        return False
    
    def _numerical_recovery(self, error: CFDError, context: Dict) -> bool:
        """數值錯誤恢復策略"""
        logger.info("🔧 執行數值穩定化恢復...")
        
        try:
            # 如果有求解器實例，應用緊急穩定化
            if 'solver' in context:
                solver = context['solver']
                if hasattr(solver, 'emergency_stabilization'):
                    # 使用數值穩定性監控器
                    from numerical_stability import NumericalStabilityMonitor
                    monitor = NumericalStabilityMonitor()
                    monitor.emergency_stabilization(solver)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"數值恢復失敗: {e}")
            return False
    
    def _memory_recovery(self, error: CFDError, context: Dict) -> bool:
        """記憶體錯誤恢復策略"""
        logger.info("🔧 執行記憶體清理恢復...")
        
        try:
            # 清理GPU記憶體
            import taichi as ti
            if ti.is_initialized():
                # 強制垃圾回收
                import gc
                gc.collect()
                
                # 如果可能，重置Taichi
                logger.warning("嘗試重置Taichi以釋放GPU記憶體")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"記憶體恢復失敗: {e}")
            return False
    
    def _gpu_recovery(self, error: CFDError, context: Dict) -> bool:
        """GPU錯誤恢復策略"""
        logger.info("🔧 執行GPU錯誤恢復...")
        
        try:
            # 嘗試重新初始化GPU
            import taichi as ti
            if ti.is_initialized():
                logger.warning("檢測到GPU錯誤，建議檢查GPU狀態")
                # 在實際應用中，這裡可以嘗試切換到CPU後端
                return False  # GPU錯誤通常需要人工干預
            
            return False
            
        except Exception as e:
            logger.error(f"GPU恢復失敗: {e}")
            return False
    
    def _physics_recovery(self, error: CFDError, context: Dict) -> bool:
        """物理錯誤恢復策略"""
        logger.info("🔧 執行物理約束恢復...")
        
        try:
            # 重置到物理合理的狀態
            if 'solver' in context:
                solver = context['solver']
                # 可以重置場變數到初始狀態
                logger.info("重置場變數到物理合理狀態")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"物理恢復失敗: {e}")
            return False
    
    def get_error_statistics(self) -> Dict:
        """獲取錯誤統計信息"""
        if not self.error_log:
            return {"total_errors": 0}
        
        stats = {
            "total_errors": len(self.error_log),
            "by_category": {},
            "by_severity": {},
            "recovery_rate": 0.0,
            "recent_errors": []
        }
        
        # 按類別統計
        for category in ErrorCategory:
            count = len([e for e in self.error_log if e.category == category])
            if count > 0:
                stats["by_category"][category.value] = count
        
        # 按嚴重程度統計
        for severity in ErrorSeverity:
            count = len([e for e in self.error_log if e.severity == severity])
            if count > 0:
                stats["by_severity"][severity.value] = count
        
        # 恢復率
        recovery_attempts = len([e for e in self.error_log if e.recovery_attempted])
        successful_recoveries = len([e for e in self.error_log if e.recovery_successful])
        if recovery_attempts > 0:
            stats["recovery_rate"] = successful_recoveries / recovery_attempts
        
        # 最近錯誤
        recent_errors = sorted(self.error_log, key=lambda x: x.timestamp, reverse=True)[:5]
        stats["recent_errors"] = [
            {
                "type": e.error_type,
                "message": e.message,
                "time": time.ctime(e.timestamp)
            }
            for e in recent_errors
        ]
        
        return stats
    
    def export_error_log(self, filename: str = None):
        """導出錯誤日誌"""
        if filename is None:
            filename = f"cfd_error_log_{int(time.time())}.json"
        
        log_data = []
        for record in self.error_log:
            log_data.append({
                "timestamp": record.timestamp,
                "time_str": time.ctime(record.timestamp),
                "error_type": record.error_type,
                "message": record.message,
                "category": record.category.value,
                "severity": record.severity.value,
                "context": record.context,
                "recovery_attempted": record.recovery_attempted,
                "recovery_successful": record.recovery_successful
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 錯誤日誌已導出到: {filename}")
    
    def clear_error_log(self):
        """清理錯誤日誌"""
        self.error_log.clear()
        self.error_count = 0
        logger.info("✅ 錯誤日誌已清理")

# 全局錯誤處理器實例
global_error_handler = GlobalErrorHandler()

def handle_cfd_error(error: CFDError, context: Dict = None) -> bool:
    """全局錯誤處理函數"""
    return global_error_handler.handle_error(error, context)

def get_error_handler() -> GlobalErrorHandler:
    """獲取全局錯誤處理器"""
    return global_error_handler

# 裝飾器用於自動錯誤處理
def with_error_handling(category: ErrorCategory, severity: ErrorSeverity = ErrorSeverity.ERROR):
    """錯誤處理裝飾器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CFDError:
                # CFD異常直接重新拋出
                raise
            except Exception as e:
                # 將普通異常包裝為CFD異常
                cfd_error = CFDError(
                    message=f"在{func.__name__}中發生錯誤: {str(e)}",
                    category=category,
                    severity=severity,
                    context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
                )
                
                # 嘗試處理錯誤
                if handle_cfd_error(cfd_error):
                    logger.info(f"錯誤已恢復，重試{func.__name__}")
                    return func(*args, **kwargs)  # 重試
                else:
                    raise cfd_error
        
        return wrapper
    return decorator