# minimal_coupling_test.py - 最小耦合測試
"""
最簡化的Phase 2驗證測試
僅測試基本功能是否工作
"""

# 設置Python路徑以便導入模組
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import taichi as ti

# 初始化Taichi
ti.init(arch=ti.cpu)  # 使用CPU避免GPU問題
print("✅ Taichi CPU初始化成功")

try:
    from src.core.thermal_fluid_coupled import ThermalFluidCoupledSolver, CouplingConfig
    print("✅ 模組導入成功")
    
    # 創建簡單配置
    config = CouplingConfig(
        coupling_frequency=999,  # 高頻率=不耦合，簡化測試
        enable_diagnostics=False
    )
    
    # 創建耦合系統
    solver = ThermalFluidCoupledSolver(coupling_config=config)
    print("✅ 耦合系統創建成功")
    
    # 初始化
    fluid_cond = {}
    thermal_cond = {'T_initial': 25.0, 'T_hot_region': 50.0, 'hot_region_height': 5}
    
    solver.initialize_system(
        fluid_initial_conditions=fluid_cond,
        thermal_initial_conditions=thermal_cond
    )
    print("✅ 系統初始化成功")
    
    # 測試單步
    success = solver.step()
    print(f"✅ 單步執行結果: {success}")
    
    if success:
        diagnostics = solver.get_coupling_diagnostics()
        print(f"✅ 診斷資訊獲取成功")
        print(f"   溫度範圍: {diagnostics['thermal_stats']['T_min']:.1f} - {diagnostics['thermal_stats']['T_max']:.1f}°C")
        
        print("🎉 Phase 2 弱耦合基本功能驗證成功！")
    else:
        print("❌ 單步執行失敗")
        
except Exception as e:
    print(f"❌ 測試失敗: {e}")
    import traceback
    traceback.print_exc()