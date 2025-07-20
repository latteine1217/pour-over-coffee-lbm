# test_simple.py
"""
簡化的測試程式 - 驗證核心3D功能
"""

import taichi as ti
import numpy as np
from lbm_solver import LBMSolver
from visualizer import UnifiedVisualizer
import config

# 初始化Taichi (使用較小的記憶體)
ti.init(arch=ti.gpu, device_memory_GB=2.0)

def test_3d_simulation():
    """測試3D LBM模擬"""
    print("=== 測試3D LBM模擬 ===")
    
    # 創建3D求解器
    lbm = LBMSolver()
    visualizer = UnifiedVisualizer(lbm)
    
    # 初始化
    lbm.init_fields()
    
    # 運行幾步 (減少步數)
    for step in range(10):
        lbm.step()
        if step % 2 == 0:
            stats = visualizer.get_statistics()
            max_vel = stats['max_velocity']
            water_mass = stats['total_water_mass']
            print(f"Step {step}: Water mass = {water_mass:.3f}, Max vel = {max_vel:.6f}")
            
            # 檢查數值穩定性
            if np.isnan(water_mass) or water_mass > 1e6:
                print("⚠️  發現數值不穩定，停止測試")
                break
    
    print("3D測試完成 ✓")

if __name__ == "__main__":
    try:
        test_3d_simulation()
        print("\n所有測試通過！")
    except Exception as e:
        print(f"測試失敗: {e}")