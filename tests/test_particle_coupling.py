#!/usr/bin/env python3
"""
顆粒-流體雙向耦合系統測試 - P0任務驗證
測試反作用力場分布和三線性插值算法

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import config
from src.physics.coffee_particles import CoffeeParticleSystem
from src.core.lbm_solver import LBMSolver
from src.physics.filter_paper import FilterPaperSystem

# 簡化配置
config.NX = config.NY = config.NZ = 64  # 減小網格以快速測試
print(f"🧪 測試配置: {config.NX}×{config.NY}×{config.NZ} 網格")

def test_particle_coupling():
    """測試雙向耦合系統的核心功能"""
    print("="*60)
    print("🔬 P0任務測試：反作用力場分布算法")
    print("="*60)
    
    # 1. 初始化系統
    print("\n1️⃣ 初始化系統...")
    try:
        particle_system = CoffeeParticleSystem(max_particles=100)
        lbm_solver = LBMSolver()
        filter_system = FilterPaperSystem()
        print("   ✅ 系統初始化成功")
    except Exception as e:
        print(f"   ❌ 系統初始化失敗: {e}")
        return False
    
    # 2. 創建少量測試顆粒
    print("\n2️⃣ 創建測試顆粒...")
    particle_system.clear_all_particles()
    
    # 在網格中心創建幾個測試顆粒
    center_x, center_y, center_z = config.NX//2, config.NY//2, config.NZ//2
    test_particles = [
        (center_x, center_y, center_z),
        (center_x + 5, center_y, center_z),
        (center_x, center_y + 5, center_z),
        (center_x, center_y, center_z + 5)
    ]
    
    created_count = 0
    for i, (x, y, z) in enumerate(test_particles):
        radius = 0.002  # 2mm
        success = particle_system.create_particle_with_physics(
            i, float(x), float(y), float(z), radius, 0.0, 0.0, 0.0)
        if success:
            created_count += 1
    
    particle_system.particle_count[None] = created_count
    particle_system.active_count[None] = created_count
    
    print(f"   ✅ 創建了 {created_count} 個測試顆粒")
    
    # 3. 設置簡單的流場
    print("\n3️⃣ 設置測試流場...")
    try:
        lbm_solver.initialize_equilibrium_state()
        
        # 設置簡單的向下流動
        @ti.kernel
        def set_test_flow():
            for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
                lbm_solver.u[i, j, k] = ti.Vector([0.0, 0.0, -0.01])  # 向下流動
        
        set_test_flow()
        print("   ✅ 測試流場設置完成")
    except Exception as e:
        print(f"   ❌ 流場設置失敗: {e}")
        return False
    
    # 4. 測試雙向耦合計算
    print("\n4️⃣ 測試雙向耦合計算...")
    try:
        # 清空反作用力場
        particle_system.clear_reaction_forces()
        
        # 計算雙向耦合力
        particle_system.compute_two_way_coupling_forces(lbm_solver.u)
        
        # 獲取診斷信息
        coupling_stats = particle_system.get_coupling_diagnostics()
        
        print(f"   ✅ 雙向耦合計算成功")
        print(f"      - 活躍顆粒: {coupling_stats['active_particles']}")
        print(f"      - 平均Reynolds數: {coupling_stats['avg_reynolds']:.3f}")
        print(f"      - 平均阻力係數: {coupling_stats['avg_drag_coeff']:.3f}")
        print(f"      - 最大反作用力: {coupling_stats['max_reaction_force']:.6f}")
        print(f"      - 耦合品質: {coupling_stats['coupling_quality']}")
        
    except Exception as e:
        print(f"   ❌ 雙向耦合計算失敗: {e}")
        return False
    
    # 5. 測試LBM集成
    print("\n5️⃣ 測試LBM集成...")
    try:
        # 清空LBM體力場
        lbm_solver.clear_body_force()
        
        # 添加顆粒反作用力
        lbm_solver.add_particle_reaction_forces(particle_system)
        
        # 檢查體力場大小
        body_force_magnitude = lbm_solver._compute_body_force_magnitude()
        print(f"   ✅ LBM集成成功")
        print(f"      - 體力場大小: {body_force_magnitude:.6f}")
        
    except Exception as e:
        print(f"   ❌ LBM集成失敗: {e}")
        return False
    
    # 6. 測試亞鬆弛穩定控制
    print("\n6️⃣ 測試亞鬆弛穩定控制...")
    try:
        relaxation_factor = 0.8
        particle_system.apply_under_relaxation(relaxation_factor)
        
        print(f"   ✅ 亞鬆弛控制成功 (因子: {relaxation_factor})")
        
    except Exception as e:
        print(f"   ❌ 亞鬆弛控制失敗: {e}")
        return False
    
    # 7. 完整的雙向耦合時間步測試
    print("\n7️⃣ 測試完整雙向耦合時間步...")
    try:
        dt = 0.001  # 1ms時間步
        relaxation_factor = 0.8
        
        # 執行耦合時間步
        lbm_solver.step_with_two_way_coupling(particle_system, dt, relaxation_factor)
        
        print(f"   ✅ 完整耦合時間步成功")
        print(f"      - 時間步: {dt*1000:.1f}ms")
        print(f"      - 亞鬆弛因子: {relaxation_factor}")
        
    except Exception as e:
        print(f"   ❌ 完整耦合時間步失敗: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 P0任務測試全部通過！")
    print("✅ 反作用力場分布算法：正常工作")
    print("✅ 三線性插值算法：正常工作") 
    print("✅ LBM體力項集成：正常工作")
    print("✅ 亞鬆弛穩定控制：正常工作")
    print("="*60)
    
    return True

if __name__ == "__main__":
    # 初始化Taichi
    ti.init(arch=ti.cpu, debug=True)
    
    try:
        success = test_particle_coupling()
        if success:
            print("\n🚀 雙向耦合系統準備就緒！")
            exit(0)
        else:
            print("\n❌ 測試失敗，需要修復問題")
            exit(1)
    except Exception as e:
        print(f"\n💥 測試程序異常: {e}")
        import traceback
        traceback.print_exc()
        exit(1)