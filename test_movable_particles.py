# test_movable_particles.py
"""
測試可移動咖啡顆粒系統
驗證顆粒-流體耦合物理模型是否正確工作
"""

import taichi as ti
import numpy as np
import time
import config
from coffee_particles import CoffeeParticleSystem
from lbm_solver import LBMSolver
from multiphase_3d import MultiphaseFlow3D

# 初始化Taichi
ti.init(arch=ti.gpu, device_memory_GB=2.0)

def test_particle_initialization():
    """測試顆粒初始化"""
    print("=== 測試1: 顆粒初始化 ===")
    
    particle_system = CoffeeParticleSystem(max_particles=5000)
    
    # 初始化咖啡床
    bed_height = 0.03  # 3cm
    bed_top_radius = 0.055  # 5.5cm
    center_x = config.NX // 2
    center_y = config.NY // 2
    bottom_z = 5
    
    particle_system.initialize_coffee_bed_with_info(
        bed_height, bed_top_radius, center_x, center_y, bottom_z
    )
    
    stats = particle_system.get_detailed_statistics()
    print(f"✅ 初始化完成:")
    print(f"   └─ 總顆粒數: {stats['total_particles']:,}")
    print(f"   └─ 活躍顆粒: {stats['active_particles']:,}")
    print(f"   └─ 平均大小: {stats['average_size']:.3f} 格子單位")
    print(f"   └─ 初始萃取度: {stats['average_extraction']:.3f}")
    
    assert stats['active_particles'] > 0, "沒有生成活躍顆粒"
    assert stats['average_extraction'] == 0.0, "初始萃取度應為0"
    
    print("✅ 顆粒初始化測試通過\n")
    return particle_system

def test_particle_physics():
    """測試顆粒物理"""
    print("=== 測試2: 顆粒物理模擬 ===")
    
    # 創建簡化的LBM求解器
    lbm = LBMSolver()
    lbm.init_fields()
    
    # 創建顆粒系統
    particle_system = CoffeeParticleSystem(max_particles=1000)
    particle_system.initialize_coffee_bed_with_info(
        0.02, 0.04, config.NX//2, config.NY//2, 10
    )
    
    # 設置簡單流場（向下流動）
    for i in range(config.NX):
        for j in range(config.NY):
            for k in range(config.NZ):
                if lbm.solid[i, j, k] == 0:
                    lbm.u[i, j, k] = [0.0, 0.0, -0.01]  # 向下0.01 m/s
                    lbm.rho[i, j, k] = 1000.0
    
    print("模擬10個時間步驟...")
    
    initial_stats = particle_system.get_detailed_statistics()
    
    for step in range(10):
        particle_system.step_particle_physics(config.DT, lbm)
        
        if step % 3 == 0:
            stats = particle_system.get_detailed_statistics()
            print(f"Step {step}: 平均速度={stats['average_speed']:.6f} m/s, "
                  f"萃取度={stats['average_extraction']:.4f}")
    
    final_stats = particle_system.get_detailed_statistics()
    
    print(f"✅ 物理模擬完成:")
    print(f"   └─ 初始平均速度: {initial_stats['average_speed']:.6f} m/s")
    print(f"   └─ 最終平均速度: {final_stats['average_speed']:.6f} m/s")
    print(f"   └─ 萃取增加: {final_stats['average_extraction'] - initial_stats['average_extraction']:.6f}")
    
    # 驗證顆粒有移動
    assert final_stats['average_speed'] > initial_stats['average_speed'], "顆粒應該有速度增加"
    assert final_stats['average_extraction'] > initial_stats['average_extraction'], "萃取度應該增加"
    
    print("✅ 顆粒物理測試通過\n")

def test_particle_collisions():
    """測試顆粒碰撞"""
    print("=== 測試3: 顆粒碰撞系統 ===")
    
    particle_system = CoffeeParticleSystem(max_particles=100)
    
    # 手動放置兩個重疊的顆粒
    particle_system.particle_count[None] = 2
    
    # 顆粒1
    particle_system.position[0] = [config.NX//2, config.NY//2, 20]
    particle_system.velocity[0] = [0.01, 0.0, 0.0]
    particle_system.radius[0] = 1.0
    particle_system.mass[0] = 1.0
    particle_system.active[0] = 1
    
    # 顆粒2 (重疊位置)
    particle_system.position[1] = [config.NX//2 + 1.5, config.NY//2, 20]
    particle_system.velocity[1] = [-0.01, 0.0, 0.0]
    particle_system.radius[1] = 1.0
    particle_system.mass[1] = 1.0
    particle_system.active[1] = 1
    
    print("測試碰撞前後的速度變化...")
    
    # 碰撞前速度
    v0_before = particle_system.velocity[0].to_numpy()
    v1_before = particle_system.velocity[1].to_numpy()
    
    # 執行碰撞計算
    particle_system.compute_particle_collisions()
    particle_system.update_particles(config.DT)
    
    # 碰撞後速度
    v0_after = particle_system.velocity[0].to_numpy()
    v1_after = particle_system.velocity[1].to_numpy()
    
    print(f"顆粒1速度變化: {v0_before} → {v0_after}")
    print(f"顆粒2速度變化: {v1_before} → {v1_after}")
    
    # 驗證動量守恆
    momentum_before = v0_before + v1_before
    momentum_after = v0_after + v1_after
    momentum_diff = np.linalg.norm(momentum_before - momentum_after)
    
    print(f"動量守恆檢查: 差異 = {momentum_diff:.6f}")
    
    assert momentum_diff < 0.01, f"動量不守恆，差異過大: {momentum_diff}"
    
    print("✅ 顆粒碰撞測試通過\n")

def test_dynamic_porosity():
    """測試動態孔隙率更新"""
    print("=== 測試4: 動態孔隙率系統 ===")
    
    lbm = LBMSolver()
    lbm.init_fields()
    
    particle_system = CoffeeParticleSystem(max_particles=500)
    particle_system.initialize_coffee_bed_with_info(
        0.015, 0.03, config.NX//2, config.NY//2, 8
    )
    
    print("更新動態孔隙率和滲透率場...")
    
    # 更新孔隙率場
    particle_system.update_dynamic_porosity(lbm.porous, lbm.permeability)
    
    # 統計孔隙率分佈
    porosity_data = lbm.porous.to_numpy()
    permeability_data = lbm.permeability.to_numpy()
    
    print(f"✅ 動態孔隙率統計:")
    print(f"   └─ 平均孔隙率: {np.mean(porosity_data):.3f}")
    print(f"   └─ 最小孔隙率: {np.min(porosity_data):.3f}")
    print(f"   └─ 最大孔隙率: {np.max(porosity_data):.3f}")
    print(f"   └─ 平均滲透率: {np.mean(permeability_data):.2e} m²")
    
    # 檢查有無咖啡床區域的低孔隙率
    low_porosity_count = np.sum(porosity_data < 0.5)
    print(f"   └─ 低孔隙率格點數: {low_porosity_count:,}")
    
    assert low_porosity_count > 0, "應該有低孔隙率區域"
    assert np.min(porosity_data) < 0.99, "最小孔隙率應該小於0.99"
    
    print("✅ 動態孔隙率測試通過\n")

def test_extraction_kinetics():
    """測試萃取動力學"""
    print("=== 測試5: 咖啡萃取動力學 ===")
    
    lbm = LBMSolver()
    lbm.init_fields()
    
    particle_system = CoffeeParticleSystem(max_particles=200)
    particle_system.initialize_coffee_bed_with_info(
        0.01, 0.02, config.NX//2, config.NY//2, 10
    )
    
    # 設置流場（模擬熱水流過）
    for i in range(config.NX):
        for j in range(config.NY):
            for k in range(config.NZ):
                if lbm.solid[i, j, k] == 0:
                    lbm.u[i, j, k] = [0.0, 0.0, -0.005]  # 慢速向下流動
                    lbm.rho[i, j, k] = 965.3  # 90°C水密度
    
    print("模擬萃取過程...")
    
    initial_extraction = particle_system.get_detailed_statistics()['average_extraction']
    
    # 模擬萃取過程
    for step in range(50):
        particle_system.update_extraction(config.DT, lbm.u, 90.0)  # 90°C
        
        if step % 10 == 0:
            stats = particle_system.get_detailed_statistics()
            print(f"Step {step}: 萃取度={stats['average_extraction']:.3f}, "
                  f"分佈={stats['extraction_distribution']}")
    
    final_stats = particle_system.get_detailed_statistics()
    final_extraction = final_stats['average_extraction']
    
    print(f"✅ 萃取動力學結果:")
    print(f"   └─ 初始萃取度: {initial_extraction:.3f}")
    print(f"   └─ 最終萃取度: {final_extraction:.3f}")
    print(f"   └─ 萃取增量: {final_extraction - initial_extraction:.3f}")
    
    assert final_extraction > initial_extraction, "萃取度應該增加"
    assert final_extraction <= 1.0, "萃取度不應超過1.0"
    
    print("✅ 萃取動力學測試通過\n")

def run_all_tests():
    """運行所有測試"""
    print("🧪 開始可移動咖啡顆粒系統測試")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 執行所有測試
        particle_system = test_particle_initialization()
        test_particle_physics()
        test_particle_collisions()
        test_dynamic_porosity()
        test_extraction_kinetics()
        
        elapsed_time = time.time() - start_time
        
        print("=" * 50)
        print(f"🎉 所有測試通過！")
        print(f"   └─ 總測試時間: {elapsed_time:.2f} 秒")
        print(f"   └─ 顆粒-流體耦合系統運行正常")
        print(f"   └─ 已替代達西定律固定多孔介質模型")
        print(f"   └─ 支持真實的顆粒移動、碰撞、萃取")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n✨ 可移動顆粒系統已準備好用於咖啡模擬！")
        print("   運行 'python main.py' 體驗新的物理模型")
    else:
        print("\n❌ 測試未通過，請檢查系統配置")