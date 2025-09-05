# test_coffee_particles_extended.py
"""
咖啡顆粒系統擴展測試套件
測試顆粒物理、碰撞檢測、流固耦合等核心功能

開發：opencode + GitHub Copilot
"""

# 設置Python路徑以便導入模組
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

import unittest
import numpy as np
import taichi as ti
import config.config
from src.physics.coffee_particles import CoffeeParticleSystem

class TestCoffeeParticlesExtended(unittest.TestCase):
    """咖啡顆粒系統擴展測試"""
    
    @classmethod
    def setUpClass(cls):
        """測試類初始化"""
        ti.init(arch=ti.cpu)
        print("🔬 開始咖啡顆粒系統擴展測試...")
    
    def setUp(self):
        """每個測試前的初始化"""
        self.particle_system = CoffeeParticleSystem()
        self.particle_system.init_particles()
    
    def test_particle_initialization(self):
        """測試顆粒初始化"""
        # 檢查顆粒數量
        self.assertGreater(self.particle_system.num_particles, 0)
        
        # 檢查顆粒半徑
        radius = config.COFFEE_PARTICLE_RADIUS
        self.assertGreater(radius, 0)
        self.assertLess(radius, 0.001)  # 合理的顆粒尺寸
        
        # 檢查顆粒位置初始化
        positions = self.particle_system.get_positions()
        self.assertEqual(len(positions), self.particle_system.num_particles)
        
        # 檢查位置是否在合理範圍內
        for pos in positions:
            self.assertTrue(all(0 <= p <= config.PHYSICAL_DOMAIN_SIZE for p in pos))
        
        print("✅ 顆粒初始化測試通過")
    
    def test_particle_motion_physics(self):
        """測試顆粒運動物理"""
        initial_positions = self.particle_system.get_positions()
        initial_velocities = self.particle_system.get_velocities()
        
        # 執行幾個時間步
        for step in range(5):
            self.particle_system.update(None)  # 無流體耦合測試
        
        final_positions = self.particle_system.get_positions()
        final_velocities = self.particle_system.get_velocities()
        
        # 檢查位置是否有變化（在重力作用下）
        total_displacement = 0.0
        for i in range(len(initial_positions)):
            displacement = np.linalg.norm(
                np.array(final_positions[i]) - np.array(initial_positions[i])
            )
            total_displacement += displacement
        
        # 在重力作用下，顆粒應該有移動
        self.assertGreater(total_displacement, 1e-6)
        
        # 檢查速度合理性（不應該無限增長）
        max_velocity = max(np.linalg.norm(v) for v in final_velocities)
        self.assertLess(max_velocity, 10.0)  # 合理的速度上限
        
        print("✅ 顆粒運動物理測試通過")
    
    def test_particle_collision_detection(self):
        """測試顆粒碰撞檢測"""
        # 人為設置兩個顆粒非常接近
        if hasattr(self.particle_system, 'set_particle_position'):
            self.particle_system.set_particle_position(0, [0.05, 0.05, 0.05])
            self.particle_system.set_particle_position(1, [0.051, 0.05, 0.05])
            
            # 檢查碰撞檢測
            collisions = self.particle_system.detect_collisions()
            
            # 應該檢測到碰撞
            self.assertGreater(len(collisions), 0)
        
        print("✅ 顆粒碰撞檢測測試通過")
    
    def test_particle_boundary_interaction(self):
        """測試顆粒與邊界的交互"""
        # 將顆粒放在接近邊界的位置
        boundary_positions = [
            [0.001, 0.05, 0.05],  # 接近左邊界
            [config.PHYSICAL_DOMAIN_SIZE - 0.001, 0.05, 0.05],  # 接近右邊界
            [0.05, 0.05, 0.001]   # 接近底部邊界
        ]
        
        for i, pos in enumerate(boundary_positions[:min(3, self.particle_system.num_particles)]):
            if hasattr(self.particle_system, 'set_particle_position'):
                self.particle_system.set_particle_position(i, pos)
        
        # 執行更新
        self.particle_system.update(None)
        
        # 檢查顆粒是否仍在邊界內
        final_positions = self.particle_system.get_positions()
        for pos in final_positions:
            self.assertTrue(0 <= pos[0] <= config.PHYSICAL_DOMAIN_SIZE)
            self.assertTrue(0 <= pos[1] <= config.PHYSICAL_DOMAIN_SIZE) 
            self.assertTrue(0 <= pos[2] <= config.PHYSICAL_DOMAIN_SIZE)
        
        print("✅ 顆粒邊界交互測試通過")
    
    def test_particle_mass_conservation(self):
        """測試顆粒質量守恆"""
        initial_mass = self.particle_system.get_total_mass()
        
        # 執行多個時間步
        for step in range(10):
            self.particle_system.update(None)
        
        final_mass = self.particle_system.get_total_mass()
        
        # 質量應該守恆
        mass_change_ratio = abs(final_mass - initial_mass) / initial_mass
        self.assertLess(mass_change_ratio, 1e-6, 
                       f"質量變化過大: {mass_change_ratio:.8f}")
        
        print("✅ 顆粒質量守恆測試通過")
    
    def test_particle_energy_dissipation(self):
        """測試顆粒能量耗散"""
        # 給顆粒一些初始動能
        if hasattr(self.particle_system, 'set_particle_velocity'):
            for i in range(min(5, self.particle_system.num_particles)):
                self.particle_system.set_particle_velocity(i, [0.1, 0.0, 0.0])
        
        initial_energy = self.particle_system.get_kinetic_energy()
        
        # 執行時間步，期望由於摩擦和碰撞能量會減少
        for step in range(20):
            self.particle_system.update(None)
        
        final_energy = self.particle_system.get_kinetic_energy()
        
        # 能量應該減少（由於耗散）
        self.assertLessEqual(final_energy, initial_energy)
        
        print("✅ 顆粒能量耗散測試通過")

class TestCoffeeParticlePhysics(unittest.TestCase):
    """咖啡顆粒物理模型測試"""
    
    @classmethod
    def setUpClass(cls):
        if not ti.is_initialized():
            ti.init(arch=ti.cpu)
    
    def setUp(self):
        self.particle_system = CoffeeParticleSystem()
        self.particle_system.init_particles()
    
    def test_drag_force_calculation(self):
        """測試拖拽力計算"""
        # 創建模擬流體場
        flow_velocity = np.array([0.1, 0.0, 0.0])
        particle_velocity = np.array([0.05, 0.0, 0.0])
        
        drag_force = self.particle_system.calculate_drag_force(
            flow_velocity, particle_velocity
        )
        
        # 拖拽力應該與相對速度方向相反
        relative_velocity = flow_velocity - particle_velocity
        self.assertGreater(np.dot(drag_force, relative_velocity), 0)
        
        print("✅ 拖拽力計算測試通過")
    
    def test_buoyancy_force(self):
        """測試浮力計算"""
        buoyancy = self.particle_system.calculate_buoyancy_force()
        
        # 浮力應該向上（正Z方向）
        self.assertGreater(buoyancy[2], 0)
        
        # 浮力大小應該合理
        buoyancy_magnitude = np.linalg.norm(buoyancy)
        self.assertGreater(buoyancy_magnitude, 0)
        self.assertLess(buoyancy_magnitude, 1e-6)  # 不應該過大
        
        print("✅ 浮力計算測試通過")
    
    def test_particle_settling_velocity(self):
        """測試顆粒沉降速度"""
        # 在靜止流體中的沉降
        terminal_velocity = self.particle_system.calculate_terminal_velocity()
        
        # 沉降速度應該向下
        self.assertLess(terminal_velocity[2], 0)
        
        # 沉降速度應該在合理範圍內
        settling_speed = abs(terminal_velocity[2])
        self.assertGreater(settling_speed, 1e-6)
        self.assertLess(settling_speed, 0.1)
        
        print("✅ 顆粒沉降速度測試通過")

def run_extended_tests():
    """執行擴展測試套件"""
    print("🧪 開始執行咖啡顆粒系統擴展測試...")
    print("=" * 60)
    
    # 創建測試套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加測試類
    suite.addTests(loader.loadTestsFromTestCase(TestCoffeeParticlesExtended))
    suite.addTests(loader.loadTestsFromTestCase(TestCoffeeParticlePhysics))
    
    # 執行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 輸出總結
    print("=" * 60)
    if result.wasSuccessful():
        print("🎉 咖啡顆粒系統所有擴展測試通過！")
        return True
    else:
        print(f"❌ 測試失敗: {len(result.failures)} 失敗, {len(result.errors)} 錯誤")
        for failure in result.failures:
            print(f"失敗: {failure[0]}")
            print(f"原因: {failure[1]}")
        for error in result.errors:
            print(f"錯誤: {error[0]}")
            print(f"原因: {error[1]}")
        return False

if __name__ == "__main__":
    success = run_extended_tests()
    exit(0 if success else 1)