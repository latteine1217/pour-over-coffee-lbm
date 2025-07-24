# numerical_stability.py
"""
數值穩定性監控和異常處理系統
為LBM模擬提供實時數值檢測和錯誤恢復機制

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import config
from init import initialize_taichi_once

# 確保Taichi已正確初始化
initialize_taichi_once()

@ti.data_oriented
class NumericalStabilityMonitor:
    """數值穩定性監控器"""
    
    def __init__(self):
        """初始化穩定性監控器"""
        # Taichi已經在外部初始化，跳過檢查
        # if not ti.is_initialized():  # Taichi 1.7版本不支持
        #     ti.init(arch=ti.cpu)
            
        # 監控統計場
        self.max_velocity = ti.field(dtype=ti.f32, shape=())
        self.min_density = ti.field(dtype=ti.f32, shape=())
        self.max_density = ti.field(dtype=ti.f32, shape=())
        self.nan_count = ti.field(dtype=ti.i32, shape=())
        self.inf_count = ti.field(dtype=ti.i32, shape=())
        
        # 穩定性閾值
        self.MAX_VELOCITY_THRESHOLD = 0.3  # Mach數限制
        self.MIN_DENSITY_THRESHOLD = 0.1
        self.MAX_DENSITY_THRESHOLD = 5.0
        self.MAX_NAN_TOLERANCE = 10  # 最大容許NaN數量
        
        # 錯誤計數器
        self.error_history = []
        self.consecutive_errors = 0
        
        print("✅ 數值穩定性監控器初始化完成")
    
    @ti.kernel  
    def check_field_stability(self, solver: ti.template()) -> ti.i32:
        """
        檢查LBM場的數值穩定性
        返回值: 0=正常, 1=警告, 2=嚴重錯誤
        """
        # 初始化統計值
        self.max_velocity[None] = 0.0
        self.min_density[None] = 1e10
        self.max_density[None] = 0.0
        self.nan_count[None] = 0
        self.inf_count[None] = 0
        
        status = 0  # 0=正常
        
        # 掃描所有流體節點
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solver.solid[i, j, k] == 0:  # 流體節點
                # 檢查密度
                rho = solver.rho[i, j, k]
                if ti.math.isnan(rho):
                    self.nan_count[None] += 1
                elif ti.math.isinf(rho):
                    self.inf_count[None] += 1
                else:
                    ti.atomic_min(self.min_density[None], rho)
                    ti.atomic_max(self.max_density[None], rho)
                
                # 檢查速度
                u = solver.u[i, j, k]
                u_mag = u.norm()
                if ti.math.isnan(u_mag):
                    self.nan_count[None] += 1
                elif ti.math.isinf(u_mag):
                    self.inf_count[None] += 1
                else:
                    ti.atomic_max(self.max_velocity[None], u_mag)
                
                # 檢查分佈函數
                for q in range(config.Q_3D):
                    f_val = solver.f[q, i, j, k]
                    if ti.math.isnan(f_val):
                        self.nan_count[None] += 1
                    elif ti.math.isinf(f_val):
                        self.inf_count[None] += 1
        
        # 評估穩定性狀態
        if (self.nan_count[None] > 0 or self.inf_count[None] > 0 or
            self.max_velocity[None] > self.MAX_VELOCITY_THRESHOLD or
            self.min_density[None] < self.MIN_DENSITY_THRESHOLD or
            self.max_density[None] > self.MAX_DENSITY_THRESHOLD):
            
            if (self.nan_count[None] > self.MAX_NAN_TOLERANCE or
                self.max_velocity[None] > 0.5):
                status = 2  # 嚴重錯誤
            else:
                status = 1  # 警告
        
        return status
    
    def diagnose_stability(self, solver, step: int) -> dict:
        """
        執行完整的穩定性診斷
        返回診斷報告字典
        """
        try:
            status = self.check_field_stability(solver)
            
            report = {
                'step': step,
                'status': status,
                'max_velocity': float(self.max_velocity[None]),
                'min_density': float(self.min_density[None]), 
                'max_density': float(self.max_density[None]),
                'nan_count': int(self.nan_count[None]),
                'inf_count': int(self.inf_count[None]),
                'mach_number': float(self.max_velocity[None]) / np.sqrt(config.CS2),
                'is_stable': status == 0
            }
            
            # 記錄錯誤歷史
            if status > 0:
                self.consecutive_errors += 1
                self.error_history.append((step, status))
                if len(self.error_history) > 100:  # 保持最近100個錯誤
                    self.error_history.pop(0)
            else:
                self.consecutive_errors = 0
            
            return report
            
        except Exception as e:
            print(f"❌ 穩定性檢查失敗: {e}")
            return {
                'step': step,
                'status': 2,
                'error': str(e),
                'is_stable': False
            }
    
    @ti.kernel
    def emergency_stabilization(self, solver: ti.template()):
        """
        緊急數值穩定化 - 將異常值重置為安全值
        """
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if solver.solid[i, j, k] == 0:  # 流體節點
                # 修復密度
                rho = solver.rho[i, j, k]
                if ti.math.isnan(rho) or ti.math.isinf(rho) or rho <= 0:
                    solver.rho[i, j, k] = 1.0  # 重置為參考密度
                elif rho > self.MAX_DENSITY_THRESHOLD:
                    solver.rho[i, j, k] = self.MAX_DENSITY_THRESHOLD
                elif rho < self.MIN_DENSITY_THRESHOLD:
                    solver.rho[i, j, k] = self.MIN_DENSITY_THRESHOLD
                
                # 修復速度
                u = solver.u[i, j, k]
                u_mag = u.norm()
                if ti.math.isnan(u_mag) or ti.math.isinf(u_mag):
                    solver.u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                elif u_mag > self.MAX_VELOCITY_THRESHOLD:
                    solver.u[i, j, k] = u * (self.MAX_VELOCITY_THRESHOLD / u_mag)
                
                # 修復分佈函數
                rho_safe = solver.rho[i, j, k]
                u_safe = solver.u[i, j, k]
                
                for q in range(config.Q_3D):
                    f_val = solver.f[q, i, j, k]
                    if ti.math.isnan(f_val) or ti.math.isinf(f_val):
                        # 重置為平衡分佈
                        solver.f[q, i, j, k] = solver._compute_stable_equilibrium(
                            q, rho_safe, u_safe)
    
    def print_stability_report(self, report: dict):
        """打印穩定性報告"""
        step = report['step']
        status = report['status']
        
        if status == 0:
            print(f"✅ Step {step}: 數值穩定")
        elif status == 1:
            print(f"⚠️  Step {step}: 數值警告")
            print(f"   Max velocity: {report['max_velocity']:.6f}")
            print(f"   Density range: [{report['min_density']:.3f}, {report['max_density']:.3f}]")
            if report['nan_count'] > 0:
                print(f"   NaN count: {report['nan_count']}")
        else:
            print(f"❌ Step {step}: 數值發散!")
            print(f"   Max velocity: {report['max_velocity']:.6f}")
            print(f"   NaN/Inf: {report['nan_count']}/{report['inf_count']}")
            print(f"   連續錯誤: {self.consecutive_errors}")
    
    def should_abort_simulation(self) -> bool:
        """判斷是否應該中止模擬"""
        return self.consecutive_errors > 10  # 連續10步錯誤則中止

@ti.data_oriented  
class ErrorRecoverySystem:
    """錯誤恢復系統"""
    
    def __init__(self):
        self.monitor = NumericalStabilityMonitor()
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
    def step_with_monitoring(self, solver, step: int):
        """
        帶監控的LBM步進
        自動檢測和恢復數值問題
        """
        try:
            # 執行正常的LBM步進
            solver.step()
            
            # 檢查數值穩定性
            report = self.monitor.diagnose_stability(solver, step)
            
            if report['status'] == 2:  # 嚴重錯誤
                print(f"🚨 Step {step}: 檢測到數值發散，嘗試恢復...")
                self._attempt_recovery(solver, step)
                self.recovery_attempts += 1
            elif report['status'] == 1:  # 警告
                if step % 100 == 0:  # 每100步報告一次警告
                    self.monitor.print_stability_report(report)
            else:
                self.recovery_attempts = 0  # 重置恢復計數
            
            # 檢查是否應該中止
            if self.monitor.should_abort_simulation():
                raise RuntimeError(f"模擬在Step {step}中止：連續數值發散")
                
            return report
            
        except Exception as e:
            print(f"❌ Step {step} 執行失敗: {e}")
            if self.recovery_attempts < self.max_recovery_attempts:
                self._attempt_recovery(solver, step)
                return self.step_with_monitoring(solver, step)  # 重試
            else:
                raise RuntimeError(f"Step {step}: 恢復失敗，模擬中止")
    
    def _attempt_recovery(self, solver, step: int):
        """嘗試恢復數值穩定性"""
        print(f"🔧 Step {step}: 執行緊急穩定化...")
        
        # 應用緊急穩定化
        self.monitor.emergency_stabilization(solver)
        
        # 重新初始化部分場
        solver.swap_fields()  # 交換分佈函數場
        
        print(f"✅ Step {step}: 穩定化完成")