#!/usr/bin/env python3
"""
簡化的顆粒系統測試
驗證基本的顆粒創建和訪問功能
"""

import taichi as ti
import numpy as np

# 簡化初始化
ti.init(arch=ti.cpu, debug=False)

# 最小配置
NX = NY = NZ = 32
Q_3D = 19

@ti.data_oriented
class SimpleParticleTest:
    def __init__(self, max_particles=10):
        self.max_particles = max_particles
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.active = ti.field(dtype=ti.i32, shape=max_particles)
        self.reaction_force_field = ti.Vector.field(3, dtype=ti.f32, shape=(NX, NY, NZ))
        
    @ti.kernel
    def clear_forces(self):
        for i, j, k in ti.ndrange(NX, NY, NZ):
            self.reaction_force_field[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.func
    def distribute_force_simple(self, particle_idx: ti.i32, force: ti.math.vec3):
        """簡化的力分布測試"""
        pos = self.position[particle_idx]
        
        # 簡單的最近鄰分布
        i = ti.cast(ti.max(0, ti.min(pos[0], NX-1)), ti.i32)
        j = ti.cast(ti.max(0, ti.min(pos[1], NY-1)), ti.i32)
        k = ti.cast(ti.max(0, ti.min(pos[2], NZ-1)), ti.i32)
        
        ti.atomic_add(self.reaction_force_field[i, j, k], force)
    
    @ti.kernel
    def test_force_distribution(self):
        for p in range(self.max_particles):
            if self.active[p] == 1:
                test_force = ti.Vector([0.0, 0.0, -1.0])  # 向下的力
                self.distribute_force_simple(p, test_force)

def test_basic_functionality():
    print("🧪 簡化顆粒系統測試")
    
    # 1. 創建系統
    particle_system = SimpleParticleTest(max_particles=5)
    
    # 2. 手動設置顆粒
    particle_system.position[0] = [16.0, 16.0, 16.0]  # 網格中心
    particle_system.active[0] = 1
    
    print("✅ 顆粒系統創建成功")
    
    # 3. 測試力分布
    particle_system.clear_forces()
    particle_system.test_force_distribution()
    
    # 4. 檢查結果
    force_data = particle_system.reaction_force_field.to_numpy()
    max_force = np.max(np.linalg.norm(force_data, axis=-1))
    
    print(f"✅ 最大反作用力: {max_force:.6f}")
    
    if max_force > 0:
        print("🎉 力分布測試成功！")
        return True
    else:
        print("❌ 力分布測試失敗")
        return False

if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        if success:
            print("✅ 基本功能測試通過")
        else:
            print("❌ 基本功能測試失敗")
    except Exception as e:
        print(f"💥 測試異常: {e}")
        import traceback
        traceback.print_exc()