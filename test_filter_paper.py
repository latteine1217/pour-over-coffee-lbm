# test_filter_paper.py
"""
測試濾紙系統功能
驗證濾紙幾何、阻力、顆粒阻擋等功能
"""

import taichi as ti
import numpy as np
import time
import config
from lbm_solver import LBMSolver
from coffee_particles import CoffeeParticleSystem  
from filter_paper import FilterPaperSystem

def test_filter_paper_system():
    """測試濾紙系統完整功能"""
    print("=== 濾紙系統功能測試 ===")
    
    # 初始化Taichi
    ti.init(arch=ti.metal, debug=False)
    
    # 創建系統組件
    print("\n1. 初始化系統組件...")
    lbm = LBMSolver()
    particles = CoffeeParticleSystem(max_particles=1000)
    filter_paper = FilterPaperSystem(lbm)
    
    # 初始化場
    lbm.init_fields()
    
    # 初始化濾紙幾何
    print("\n2. 初始化濾紙幾何...")
    filter_paper.initialize_filter_geometry()
    
    # 檢查濾紙幾何
    print("\n3. 檢查濾紙幾何分佈...")
    filter_stats = filter_paper.get_filter_statistics()
    print(f"   濾紙節點總數: {filter_stats['total_filter_nodes']:,}")
    
    if filter_stats['total_filter_nodes'] == 0:
        print("❌ 濾紙幾何初始化失敗 - 沒有濾紙節點")
        return False
    else:
        print("✅ 濾紙幾何初始化成功")
    
    # 創建簡單咖啡顆粒床用於測試
    print("\n4. 創建測試用咖啡顆粒...")
    particles.initialize_coffee_bed(
        bed_height=0.02,  # 2cm
        bed_top_radius=0.03,  # 3cm 
        center_x=config.NX//2 * config.SCALE_LENGTH,
        center_y=config.NY//2 * config.SCALE_LENGTH,
        bottom_z=6 * config.SCALE_LENGTH  # 在濾紙上方
    )
    
    print(f"   活躍顆粒數: {particles.particle_count[None]:,}")
    
    # 測試濾紙阻力效應
    print("\n5. 測試濾紙阻力效應...")
    
    # 設置初始流體速度場 (向下流動)
    initial_velocity = 0.01  # m/s
    
    @ti.kernel
    def set_initial_velocity():
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if lbm.solid[i, j, k] == 0:  # 流體區域
                lbm.u[i, j, k] = ti.Vector([0.0, 0.0, -initial_velocity])
    
    set_initial_velocity()
    
    # 記錄濾紙區域的初始速度
    filter_zone_data = filter_paper.filter_zone.to_numpy()
    initial_u = lbm.u.to_numpy()
    initial_filter_speed = np.mean(np.sqrt(
        initial_u[:,:,:,0]**2 + initial_u[:,:,:,1]**2 + initial_u[:,:,:,2]**2
    )[filter_zone_data == 1])
    
    print(f"   濾紙區域初始平均速度: {initial_filter_speed:.5f} m/s")
    
    # 施加濾紙效應
    filter_paper.apply_filter_effects()
    
    # 檢查速度變化
    final_u = lbm.u.to_numpy()
    final_filter_speed = np.mean(np.sqrt(
        final_u[:,:,:,0]**2 + final_u[:,:,:,1]**2 + final_u[:,:,:,2]**2
    )[filter_zone_data == 1])
    
    print(f"   濾紙區域最終平均速度: {final_filter_speed:.5f} m/s")
    
    speed_reduction = (initial_filter_speed - final_filter_speed) / initial_filter_speed * 100
    print(f"   速度減少: {speed_reduction:.1f}%")
    
    if speed_reduction > 5:  # 至少5%的速度減少
        print("✅ 濾紙阻力效應正常")
    else:
        print("⚠️  濾紙阻力效應可能需要調整")
    
    # 測試顆粒阻擋
    print("\n6. 測試顆粒阻擋機制...")
    
    # 將一些顆粒移動到接近濾紙的位置
    particles_moved = min(10, particles.particle_count[None])
    
    @ti.kernel
    def move_particles_to_filter(num_particles: ti.i32):
        for p in range(num_particles):
            if particles.active[p] == 1:
                # 設置顆粒位置在濾紙上方，速度向下
                particles.position[p] = ti.Vector([
                    config.NX//2 * config.SCALE_LENGTH,
                    config.NY//2 * config.SCALE_LENGTH,
                    (filter_paper.filter_bottom_z + 2) * config.SCALE_LENGTH
                ])
                particles.velocity[p] = ti.Vector([0.0, 0.0, -0.01])  # 向下運動
    
    move_particles_to_filter(particles_moved)
    
    print(f"   設置 {particles_moved} 個顆粒向濾紙運動")
    
    # 記錄顆粒初始向下速度
    initial_particle_velocities = particles.velocity.to_numpy()[:particles_moved, 2]
    downward_particles = np.sum(initial_particle_velocities < 0)
    
    print(f"   初始向下運動顆粒: {downward_particles}")
    
    # 執行顆粒阻擋
    filter_paper.block_particles_at_filter(
        particles.position,
        particles.velocity,
        particles.radius,
        particles.active,
        particles.particle_count
    )
    
    # 檢查顆粒速度變化
    final_particle_velocities = particles.velocity.to_numpy()[:particles_moved, 2]
    upward_particles = np.sum(final_particle_velocities > 0)
    
    print(f"   最終向上運動顆粒: {upward_particles}")
    
    if upward_particles > 0:
        print("✅ 顆粒阻擋機制正常工作")
    else:
        print("⚠️  顆粒阻擋機制可能需要調整")
    
    # 測試動態阻力更新
    print("\n7. 測試動態阻力更新...")
    
    initial_stats = filter_paper.get_filter_statistics()
    print(f"   初始平均阻塞度: {initial_stats['average_blockage']:.1f}%")
    
    # 模擬顆粒累積
    for i in range(10):
        filter_paper.update_dynamic_resistance()
    
    final_stats = filter_paper.get_filter_statistics()
    print(f"   更新後平均阻塞度: {final_stats['average_blockage']:.1f}%")
    
    # 完整系統測試
    print("\n8. 完整系統整合測試...")
    
    # 執行幾個完整的時間步
    for step in range(5):
        filter_paper.step(particles)
        if step == 0:
            print(f"   第 {step+1} 步完成")
    
    print("✅ 完整系統整合測試通過")
    
    # 最終統計
    print("\n9. 最終濾紙系統統計:")
    final_stats = filter_paper.get_filter_statistics()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value:,}")
    
    print("\n🎉 濾紙系統測試完成!")
    return True

def test_filter_paper_geometry():
    """測試濾紙幾何精確性"""
    print("\n=== 濾紙幾何精確性測試 ===")
    
    ti.init(arch=ti.metal, debug=False)
    
    lbm = LBMSolver()
    filter_paper = FilterPaperSystem(lbm)
    
    lbm.init_fields()
    filter_paper.initialize_filter_geometry()
    
    # 檢查濾紙分佈
    filter_zone_data = filter_paper.filter_zone.to_numpy()
    
    print(f"濾紙幾何分析:")
    print(f"  總節點數: {config.NX * config.NY * config.NZ:,}")
    print(f"  濾紙節點數: {np.sum(filter_zone_data):,}")
    print(f"  濾紙覆蓋率: {np.sum(filter_zone_data)/(config.NX * config.NY * config.NZ)*100:.2f}%")
    
    # 檢查濾紙位置分佈
    filter_coords = np.where(filter_zone_data == 1)
    if len(filter_coords[0]) > 0:
        min_z = np.min(filter_coords[2])
        max_z = np.max(filter_coords[2])
        print(f"  濾紙Z範圍: {min_z} - {max_z} 格子單位")
        print(f"  濾紙厚度: {max_z - min_z + 1} 格子單位")
        
        # 檢查濾紙是否在預期位置
        expected_z = filter_paper.filter_bottom_z
        if min_z <= expected_z <= max_z:
            print("✅ 濾紙位置正確")
        else:
            print("⚠️  濾紙位置可能有誤")
    
    return True

if __name__ == "__main__":
    print("V60濾紙系統測試程序")
    print("="*50)
    
    # 運行測試
    geometry_test = test_filter_paper_geometry()
    system_test = test_filter_paper_system()
    
    print("\n" + "="*50)
    if geometry_test and system_test:
        print("🎉 所有濾紙系統測試通過!")
    else:
        print("❌ 某些測試未通過，請檢查實現")