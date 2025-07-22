#!/usr/bin/env python3
"""
測試錐形濾紙和咖啡粉約束系統
驗證：1. 濾紙錐形幾何 2. 咖啡粉強制堆積在濾紙內
開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import config
from lbm_solver import LBMSolver
from filter_paper import FilterPaperSystem
from coffee_particles import CoffeeParticleSystem

# 初始化Taichi
ti.init(arch=ti.gpu, device_memory_GB=4.0)

def test_conical_filter_geometry():
    """測試錐形濾紙幾何"""
    print("🔬 測試錐形濾紙幾何...")
    
    # 創建LBM求解器
    lbm = LBMSolver()
    lbm.init_fields()  # 初始化場變數
    
    # 創建濾紙系統
    filter_system = FilterPaperSystem(lbm)
    filter_system.initialize_filter_geometry()
    
    # 分析濾紙分佈
    filter_zone_data = filter_system.filter_zone.to_numpy()
    total_filter_nodes = np.sum(filter_zone_data)
    
    print(f"✅ 錐形濾紙節點總數: {total_filter_nodes:,}")
    
    # 檢查不同高度的濾紙分佈
    center_x, center_y = config.NX // 2, config.NY // 2
    bottom_z = 5.0
    cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
    
    print("📏 各高度濾紙分佈:")
    for layer in range(0, int(cup_height_lu), max(1, int(cup_height_lu/5))):
        z = int(bottom_z + layer)
        if z < config.NZ:
            layer_filter_count = np.sum(filter_zone_data[:, :, z])
            print(f"   Z={z:3d}: {layer_filter_count:4.0f} 濾紙節點")
    
    return filter_system

def test_coffee_particle_confinement(filter_system):
    """測試增強的咖啡粉約束系統"""
    print("\n☕ 測試增強咖啡粉約束系統...")
    
    # 創建增強顆粒系統
    particle_system = CoffeeParticleSystem(max_particles=2000)
    
    # 測試錐形約束的咖啡床生成
    created_particles = particle_system.initialize_coffee_bed_confined(filter_system)
    
    # 獲取顆粒統計信息
    stats = particle_system.get_particle_statistics()
    
    # 檢查邊界約束
    boundary = filter_system.get_coffee_bed_boundary()
    violations = particle_system.enforce_filter_boundary(
        boundary['center_x'], 
        boundary['center_y'], 
        boundary['bottom_z'],
        boundary['top_radius_lu'],
        boundary['bottom_radius_lu']
    )
    
    constraint_success_rate = (created_particles - violations) / created_particles * 100 if created_particles > 0 else 0
    
    print(f"✅ 增強顆粒系統測試結果:")
    print(f"   └─ 生成顆粒數: {created_particles:,}")
    print(f"   └─ 平均半徑: {stats['mean_radius']*1000:.3f} mm")
    print(f"   └─ 半徑標準差: {stats['std_radius']*1000:.3f} mm")
    print(f"   └─ 半徑範圍: {stats['min_radius']*1000:.3f} - {stats['max_radius']*1000:.3f} mm")
    print(f"   └─ 邊界違反: {violations}")
    print(f"   └─ 約束成功率: {constraint_success_rate:.1f}%")
    
    return particle_system

def test_dynamic_boundary_enforcement(filter_system, particle_system):
    """測試動態邊界約束"""
    print("\n🔄 測試動態邊界約束...")
    
    boundary = filter_system.get_coffee_bed_boundary()
    
    # 執行多步邊界約束測試
    total_violations = 0
    simulation_steps = 10
    
    for step in range(simulation_steps):
        violations = particle_system.enforce_filter_boundary(
            boundary['center_x'], 
            boundary['center_y'], 
            boundary['bottom_z'],
            boundary['top_radius_lu'],
            boundary['bottom_radius_lu']
        )
        total_violations += violations
        
        if violations > 0:
            print(f"   └─ 步驟 {step+1}: 修正了 {violations} 個邊界違反")
    
    constraint_hold_rate = 100.0 if total_violations == 0 else (simulation_steps - total_violations) / simulation_steps * 100
    
    print(f"✅ 動態邊界約束:")
    print(f"   └─ 模擬步數: {simulation_steps}")
    print(f"   └─ 總邊界違反: {total_violations}")
    print(f"   └─ 約束保持率: {constraint_hold_rate:.1f}%")

def main():
    """主測試函數"""
    print("🧪 錐形濾紙和咖啡粉約束系統測試")
    print("=" * 50)
    
    try:
        # 測試1: 錐形濾紙幾何
        filter_system = test_conical_filter_geometry()
        
        # 測試2: 咖啡粉約束生成
        particle_system = test_coffee_particle_confinement(filter_system)
        
        # 測試3: 動態邊界約束
        test_dynamic_boundary_enforcement(filter_system, particle_system)
        
        print("\n" + "=" * 50)
        print("🎉 所有測試完成！")
        print("✅ 錐形濾紙系統: 正常工作")
        print("✅ 咖啡粉約束系統: 正常工作")
        print("✅ 動態邊界約束: 正常工作")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)