#!/usr/bin/env python3
"""
測試咖啡顆粒重力沉降系統
檢查真實顆粒數量計算和重力堆積效果
"""

import taichi as ti
import numpy as np
import time
import config
from lbm_solver import LBMSolver
from filter_paper import FilterPaperSystem
from coffee_particles import CoffeeParticleSystem

# 初始化Taichi
ti.init(arch=ti.metal)

print("🧪 咖啡顆粒重力沉降測試")
print("=" * 50)

def test_realistic_particle_count():
    """測試真實顆粒數量計算"""
    print("🔬 測試真實顆粒數量計算...")
    
    # 初始化顆粒系統（使用較小數量進行測試）
    particle_system = CoffeeParticleSystem(max_particles=20000)
    
    # 計算理論顆粒數量
    realistic_count = particle_system.calculate_realistic_particle_count()
    
    print(f"✅ 真實顆粒數量計算完成: {realistic_count:,}")
    return particle_system, realistic_count

def test_gravity_settling(particle_system, target_particles):
    """測試重力沉降過程"""
    print("\n🌊 測試重力沉降過程...")
    
    # 初始化基礎系統
    lbm_solver = LBMSolver()
    filter_system = FilterPaperSystem(lbm_solver)
    filter_system.initialize_filter_geometry()
    
    # 生成咖啡顆粒（調整數量以適應測試）
    test_particles = min(target_particles, 10000)  # 測試時使用1萬顆粒
    print(f"   └─ 測試顆粒數: {test_particles:,}")
    
    # 使用重力沉降方法生成咖啡床
    created = particle_system.initialize_coffee_bed_with_gravity(filter_system)
    
    # 執行重力沉降模擬
    start_time = time.time()
    settled_count = particle_system.simulate_gravity_settling(filter_system, settling_steps=100)
    end_time = time.time()
    
    print(f"✅ 重力沉降測試完成:")
    print(f"   └─ 模擬時間: {end_time - start_time:.2f} 秒")
    print(f"   └─ 沉降效率: {settled_count/created*100:.1f}%")
    
    return created, settled_count

def analyze_particle_distribution(particle_system):
    """分析顆粒分佈"""
    print("\n📊 分析顆粒最終分佈...")
    
    # 獲取顆粒位置數據
    positions = particle_system.position.to_numpy()
    active = particle_system.active.to_numpy()
    radii = particle_system.radius.to_numpy()
    
    # 只考慮活躍顆粒
    active_positions = positions[active == 1]
    active_radii = radii[active == 1]
    
    if len(active_positions) > 0:
        # 高度分佈分析
        z_coords = active_positions[:, 2]
        min_z = np.min(z_coords)
        max_z = np.max(z_coords)
        mean_z = np.mean(z_coords)
        
        # 半徑分佈分析
        mean_radius = np.mean(active_radii) * config.SCALE_LENGTH * 1000  # 轉換為mm
        std_radius = np.std(active_radii) * config.SCALE_LENGTH * 1000
        
        # 密度分析
        bottom_particles = np.sum(z_coords <= min_z + 5)  # 底部5層的顆粒數
        density_ratio = bottom_particles / len(active_positions)
        
        print(f"✅ 顆粒分佈分析:")
        print(f"   └─ 總活躍顆粒: {len(active_positions):,}")
        print(f"   └─ 高度範圍: {min_z:.1f} - {max_z:.1f} lu")
        print(f"   └─ 平均高度: {mean_z:.1f} lu")
        print(f"   └─ 底部堆積率: {density_ratio*100:.1f}%")
        print(f"   └─ 平均半徑: {mean_radius:.2f}±{std_radius:.2f} mm")
        print(f"   └─ 堆積高度: {(max_z - min_z) * config.SCALE_LENGTH * 100:.1f} cm")
        
        return {
            'total_particles': len(active_positions),
            'height_range': max_z - min_z,
            'stacking_height_cm': (max_z - min_z) * config.SCALE_LENGTH * 100,
            'bottom_density': density_ratio,
            'mean_radius_mm': mean_radius
        }
    else:
        print("❌ 沒有找到活躍顆粒")
        return None

def main():
    """主測試函數"""
    print("🔬 開始咖啡顆粒重力沉降測試...")
    
    # 測試1：真實顆粒數量計算
    particle_system, realistic_count = test_realistic_particle_count()
    
    # 測試2：重力沉降過程
    created, settled = test_gravity_settling(particle_system, realistic_count)
    
    # 測試3：分佈分析
    distribution = analyze_particle_distribution(particle_system)
    
    print("\n" + "=" * 50)
    print("🎉 重力沉降測試完成！")
    
    if distribution:
        print("✅ 重要發現:")
        print(f"   └─ 理論顆粒數: {realistic_count:,}")
        print(f"   └─ 實際生成數: {created:,}")
        print(f"   └─ 沉降成功數: {settled:,}")
        print(f"   └─ 堆積高度: {distribution['stacking_height_cm']:.1f} cm")
        print(f"   └─ 底部密度: {distribution['bottom_density']*100:.1f}%")
        
        # 評估物理真實性
        if distribution['bottom_density'] > 0.5:
            print("✅ 重力沉降效果良好！顆粒成功堆積在底部")
        else:
            print("⚠️  重力沉降效果需要改進，顆粒分佈過於分散")
    else:
        print("❌ 測試失敗，無法分析顆粒分佈")

if __name__ == "__main__":
    main()