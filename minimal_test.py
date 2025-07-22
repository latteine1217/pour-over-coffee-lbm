# minimal_test.py
"""
最小化測試 - 檢查CFD模擬是否能正常運行
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

import sys
sys.path.append('.')

# 導入模組
from init import initialize_taichi_once
import config
from lbm_solver import LBMSolver

class MinimalSimulation:
    def __init__(self):
        print("🔧 最小化模擬初始化...")
        self.step_count = 0
        self.lbm = LBMSolver()
        print("✅ LBM求解器創建成功")
        
        # 只初始化基本場
        self.lbm.init_fields()
        print("✅ 場變數初始化完成")
        
    def step(self):
        """執行一個模擬步驟"""
        try:
            self.lbm.step()
            self.step_count += 1
            return True
        except Exception as e:
            print(f"❌ 步驟失敗: {e}")
            return False
    
    def get_stats(self):
        """獲取統計數據"""
        import numpy as np
        try:
            if hasattr(self.lbm, 'u') and hasattr(self.lbm, 'rho'):
                u_data = self.lbm.u.to_numpy()
                rho_data = self.lbm.rho.to_numpy()
                
                max_u = np.max(np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2))
                avg_rho = np.mean(rho_data)
                
                return {
                    'max_velocity': max_u,
                    'avg_density': avg_rho,
                    'step_count': self.step_count
                }
        except Exception as e:
            print(f"⚠️ 統計獲取失敗: {e}")
        
        return {'step_count': self.step_count}

def main():
    print("🧪 最小化CFD測試")
    print("="*50)
    
    # 創建最小化模擬
    sim = MinimalSimulation()
    
    print("\n🔄 運行5步模擬...")
    for step in range(5):
        success = sim.step()
        stats = sim.get_stats()
        
        if success:
            print(f"✅ 步驟 {step+1}: 成功")
            if 'max_velocity' in stats:
                print(f"   └─ 最大速度: {stats['max_velocity']:.6f}")
                print(f"   └─ 平均密度: {stats['avg_density']:.6f}")
        else:
            print(f"❌ 步驟 {step+1}: 失敗")
            break
    
    print("\n📊 最終統計:")
    final_stats = sim.get_stats()
    for key, value in final_stats.items():
        print(f"   └─ {key}: {value}")
    
    print("\n🎉 最小化測試完成")

if __name__ == "__main__":
    main()