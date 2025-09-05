#!/usr/bin/env python3
"""
重力修正驗證測試
測試純重力模式下是否能產生向下流動
"""

import os
import sys
import time
import numpy as np

# 添加src路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 配置imports
import config
from config.init import init_taichi
from src.core.lbm_solver import LBMSolver3D
from src.physics.boundary_conditions import BoundaryConditionManager
from src.physics.precise_pouring import PrecisePouring
from src.physics.coffee_particles import CoffeeParticleSystem

def test_gravity_flow():
    """測試純重力驅動下的水流動"""
    print("🧪 測試純重力模式下的水流動效果...")
    print(f"📊 重力參數: {config.GRAVITY_LU:.3f} lu/ts² (無限制)")
    print(f"🔧 相場閾值: 0.001 (極低)")
    print(f"⚡ Forcing限制: ±0.5 (放寬10倍)")
    
    # 初始化Taichi
    init_taichi()
    
    # 創建求解器
    solver = LBMSolver3D()
    boundary_manager = BoundaryConditionManager()
    pouring = PrecisePouring()
    
    # 初始化系統
    solver.init_fields()
    boundary_manager.apply_all(solver)
    
    print("\n🚀 開始測試...")
    step = 0
    test_steps = 20
    
    initial_total_water = 0.0
    final_total_water = 0.0
    
    while step < test_steps:
        # 注水 (前10步)
        if step < 10:
            pouring.apply_inlet_conditions(solver, step)
        
        # LBM步進
        solver.step()
        boundary_manager.apply_all(solver)
        
        # 統計水量
        if hasattr(solver, 'rho'):
            rho_data = solver.rho.to_numpy()
            total_water = np.sum(rho_data[rho_data > 1.01])  # 超過參考密度的水
            
            if step == 5:
                initial_total_water = total_water
            if step == test_steps - 1:
                final_total_water = total_water
        
        # 檢查速度場
        if hasattr(solver, 'u'):
            u_data = solver.u.to_numpy()
            avg_speed = np.mean(np.linalg.norm(u_data, axis=-1))
            max_speed = np.max(np.linalg.norm(u_data, axis=-1))
            
            # 重點檢查Z方向速度 (向下為負)
            avg_uz = np.mean(u_data[:, :, :, 2])
            min_uz = np.min(u_data[:, :, :, 2])  # 最負值 = 最大向下速度
            
            print(f"步驟 {step:2d}: 平均速度={avg_speed:.6f}, 最大速度={max_speed:.6f}, "
                  f"平均uz={avg_uz:.6f}, 最大向下速度={-min_uz:.6f}")
            
            # 關鍵檢查：是否有向下流動
            if avg_speed > 1e-6:
                print(f"  ✅ 檢測到流動！平均速度 = {avg_speed:.6f} lu/ts")
            if min_uz < -1e-6:
                print(f"  ⬇️  檢測到向下流動！最大向下速度 = {-min_uz:.6f} lu/ts")
        
        step += 1
    
    print(f"\n📊 測試結果:")
    print(f"  初始水量: {initial_total_water:.3f}")
    print(f"  最終水量: {final_total_water:.3f}")
    if final_total_water > 0:
        print(f"  水量保持: {final_total_water/max(initial_total_water, 1e-10)*100:.1f}%")
    
    if avg_speed > 1e-6:
        print(f"  ✅ 成功！重力修正生效，水開始流動")
        print(f"  💧 平均流動速度: {avg_speed:.6f} lu/ts")
        return True
    else:
        print(f"  ❌ 失敗！水仍然停滯不動")
        print(f"  🔍 可能需要進一步檢查咖啡顆粒阻塞問題")
        return False

if __name__ == "__main__":
    success = test_gravity_flow()
    if success:
        print(f"\n🎉 重力修正測試通過！現在可以測試完整模擬。")
    else:
        print(f"\n⚠️  重力修正未完全解決問題，可能需要額外調整。")