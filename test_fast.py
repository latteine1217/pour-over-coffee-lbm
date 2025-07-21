# test_fast.py
"""
快速性能測試 - 驗證優化效果
"""

import time
import sys

def test_fast_mode():
    """測試快速模式性能"""
    
    # 使用快速配置
    sys.path.insert(0, '.')
    import config_fast as config
    
    # 設置快速模式參數
    import coffee_particles
    
    print("🚀 啟動快速模式測試...")
    print("="*50)
    
    # 測試粒子系統初始化
    start_time = time.time()
    particle_system = coffee_particles.CoffeeParticleSystem(max_particles=2000)
    init_time = time.time() - start_time
    
    print(f"✅ 粒子系統初始化: {init_time:.3f}s")
    
    # 測試咖啡床初始化  
    start_time = time.time()
    particle_system.initialize_coffee_bed(
        bed_height=config.COFFEE_BED_HEIGHT_PHYS,
        bed_top_radius=config.COFFEE_BED_TOP_RADIUS,
        center_x=config.NX/2 * config.SCALE_LENGTH,
        center_y=config.NY/2 * config.SCALE_LENGTH,
        bottom_z=5 * config.SCALE_LENGTH
    )
    bed_init_time = time.time() - start_time
    
    print(f"✅ 咖啡床初始化: {bed_init_time:.3f}s")
    print(f"✅ 活躍粒子數: {particle_system.particle_count[None]:,}")
    
    # 創建測試用的流體速度場和密度場
    import taichi as ti
    fluid_velocity = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
    fluid_density = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
    
    # 初始化流體場
    fluid_density.fill(1.0)
    fluid_velocity.fill([0.0, 0.0, 0.0])
    
    # 測試基本運算性能
    start_time = time.time()
    for i in range(10):  # 減少測試步數
        particle_system.update_particles(0.01)
    compute_time = time.time() - start_time
    
    print(f"✅ 10步粒子計算: {compute_time:.3f}s")
    print(f"✅ 平均每步: {compute_time/10*1000:.2f}ms")
    
    print("="*50)
    print("🎯 快速模式效能總結:")
    print(f"   網格尺寸: {config.NX}³ = {config.NX**3:,} 節點")
    print(f"   粒子數量: {particle_system.particle_count[None]:,}")
    print(f"   初始化時間: {init_time + bed_init_time:.3f}s")
    print(f"   每步計算時間: {compute_time/10*1000:.1f}ms")
    
    return particle_system

if __name__ == "__main__":
    import taichi as ti
    ti.init(arch=ti.metal, fast_math=True, debug=False)
    
    test_fast_mode()