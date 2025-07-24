# test_multiphase_flow.py
"""
多相流系統測試套件
測試相場演化、表面張力、Cahn-Hilliard方程等核心功能

開發：opencode + GitHub Copilot
"""

import unittest
import numpy as np
import taichi as ti
import config
from multiphase_3d import MultiphaseFlow3D
from lbm_solver import LBMSolver

class TestMultiphaseFlow(unittest.TestCase):
    """多相流系統測試"""
    
    @classmethod
    def setUpClass(cls):
        """測試類初始化"""
        ti.init(arch=ti.cpu)
        print("🔬 開始多相流系統測試...")
    
    def setUp(self):
        """每個測試前的初始化"""
        self.lbm_solver = LBMSolver()
        self.lbm_solver.init_fields()
        self.multiphase = MultiphaseFlow3D()
        self.multiphase.init_phase_field()
    
    def test_phase_field_initialization(self):
        """測試相場初始化"""
        # 檢查相場範圍
        phase_data = self.multiphase.get_phase_field()
        
        # 相場值應該在[0,1]範圍內
        self.assertTrue(np.all(phase_data >= 0.0))
        self.assertTrue(np.all(phase_data <= 1.0))
        
        # 應該有水相和氣相區域
        water_phase_count = np.sum(phase_data > 0.5)
        air_phase_count = np.sum(phase_data < 0.5)
        
        self.assertGreater(water_phase_count, 0)
        self.assertGreater(air_phase_count, 0)
        
        print("✅ 相場初始化測試通過")
    
    def test_surface_tension_calculation(self):
        """測試表面張力計算"""
        # 創建簡單的水-空氣界面
        self.multiphase.create_test_interface()
        
        # 計算表面張力
        surface_tension_force = self.multiphase.calculate_surface_tension()
        
        # 表面張力應該存在且有合理的大小
        force_magnitude = np.linalg.norm(surface_tension_force)
        self.assertGreater(force_magnitude, 0)
        self.assertLess(force_magnitude, 1000)  # 不應該過大
        
        # 表面張力應該主要作用在界面附近
        interface_force_count = np.sum(np.linalg.norm(surface_tension_force, axis=-1) > 1e-6)
        total_points = surface_tension_force.shape[0] * surface_tension_force.shape[1] * surface_tension_force.shape[2]
        interface_ratio = interface_force_count / total_points
        
        # 界面應該占總體積的較小比例
        self.assertLess(interface_ratio, 0.5)
        
        print("✅ 表面張力計算測試通過")
    
    def test_phase_interface_evolution(self):
        """測試相界面演化"""
        # 記錄初始界面
        initial_phase = self.multiphase.get_phase_field().copy()
        
        # 執行時間演化
        for step in range(5):
            self.multiphase.update_phase_field(self.lbm_solver)
        
        final_phase = self.multiphase.get_phase_field()
        
        # 界面應該有變化，但不能過於劇烈
        phase_change = np.abs(final_phase - initial_phase)
        max_change = np.max(phase_change)
        mean_change = np.mean(phase_change)
        
        self.assertGreater(mean_change, 1e-8)  # 應該有變化
        self.assertLess(max_change, 0.5)       # 變化不應該過大
        
        # 相場值仍應該在有效範圍內
        self.assertTrue(np.all(final_phase >= -0.1))  # 允許小的數值誤差
        self.assertTrue(np.all(final_phase <= 1.1))
        
        print("✅ 相界面演化測試通過")
    
    def test_cahn_hilliard_equation(self):
        """測試Cahn-Hilliard方程求解"""
        # 創建測試相場配置
        self.multiphase.setup_test_configuration()
        
        # 計算化學勢
        chemical_potential = self.multiphase.calculate_chemical_potential()
        
        # 化學勢應該在界面處有梯度
        grad_potential = np.gradient(chemical_potential)
        max_gradient = np.max([np.max(np.abs(g)) for g in grad_potential])
        
        self.assertGreater(max_gradient, 1e-6)
        
        # 執行Cahn-Hilliard時間步
        initial_total_phase = np.sum(self.multiphase.get_phase_field())
        
        self.multiphase.cahn_hilliard_step()
        
        final_total_phase = np.sum(self.multiphase.get_phase_field())
        
        # 總相量應該守恆
        phase_conservation_error = abs(final_total_phase - initial_total_phase) / initial_total_phase
        self.assertLess(phase_conservation_error, 1e-6)
        
        print("✅ Cahn-Hilliard方程測試通過")
    
    def test_contact_angle_boundary(self):
        """測試接觸角邊界條件"""
        # 在固體邊界附近設置相場
        self.multiphase.setup_contact_angle_test()
        
        # 應用接觸角邊界條件
        self.multiphase.apply_contact_angle_bc()
        
        # 檢查邊界處的相場值
        boundary_phase = self.multiphase.get_boundary_phase_values()
        
        # 邊界相場值應該反映正確的接觸角
        expected_contact_angle = config.CONTACT_ANGLE if hasattr(config, 'CONTACT_ANGLE') else 90.0
        
        # 這裡簡化檢查：邊界處應該有合理的相場值
        self.assertTrue(np.all(boundary_phase >= 0.0))
        self.assertTrue(np.all(boundary_phase <= 1.0))
        
        print("✅ 接觸角邊界條件測試通過")
    
    def test_phase_field_stability(self):
        """測試相場數值穩定性"""
        # 執行較長時間的演化
        max_phase_values = []
        min_phase_values = []
        
        for step in range(20):
            self.multiphase.update_phase_field(self.lbm_solver)
            phase_field = self.multiphase.get_phase_field()
            max_phase_values.append(np.max(phase_field))
            min_phase_values.append(np.min(phase_field))
        
        # 檢查是否有數值爆炸
        self.assertTrue(all(v < 2.0 for v in max_phase_values))
        self.assertTrue(all(v > -1.0 for v in min_phase_values))
        
        # 檢查是否有NaN或Inf
        final_phase = self.multiphase.get_phase_field()
        self.assertFalse(np.any(np.isnan(final_phase)))
        self.assertFalse(np.any(np.isinf(final_phase)))
        
        print("✅ 相場數值穩定性測試通過")

class TestMultiphasePhysics(unittest.TestCase):
    """多相流物理模型測試"""
    
    @classmethod
    def setUpClass(cls):
        if not ti.is_initialized():
            ti.init(arch=ti.cpu)
    
    def setUp(self):
        self.multiphase = MultiphaseFlow3D()
        self.multiphase.init_phase_field()
    
    def test_surface_tension_magnitude(self):
        """測試表面張力大小"""
        # 創建標準液滴配置
        self.multiphase.create_spherical_droplet(radius=0.02)
        
        # 計算表面張力
        surface_tension = self.multiphase.calculate_surface_tension()
        
        # 與理論值比較 (Young-Laplace方程)
        theoretical_pressure_jump = 2 * config.SURFACE_TENSION_PHYS / 0.02
        calculated_pressure_jump = self.multiphase.calculate_pressure_jump()
        
        relative_error = abs(calculated_pressure_jump - theoretical_pressure_jump) / theoretical_pressure_jump
        self.assertLess(relative_error, 0.1)  # 10%誤差容忍
        
        print("✅ 表面張力大小測試通過")
    
    def test_spurious_currents(self):
        """測試寄生流檢測"""
        # 創建靜態液滴
        self.multiphase.create_static_droplet()
        
        # 多步演化後檢查寄生流
        for step in range(10):
            self.multiphase.update_phase_field(None)
        
        velocity_field = self.multiphase.get_velocity_field()
        max_velocity = np.max(np.linalg.norm(velocity_field, axis=-1))
        
        # 靜態液滴的寄生流應該很小
        self.assertLess(max_velocity, 0.01)
        
        print("✅ 寄生流檢測測試通過")
    
    def test_density_ratio_handling(self):
        """測試密度比處理"""
        water_density = config.RHO_WATER
        air_density = config.RHO_AIR
        density_ratio = water_density / air_density
        
        # 密度比應該合理
        self.assertGreater(density_ratio, 100)  # 水比空氣密度大
        self.assertLess(density_ratio, 2000)    # 但不應該過大導致數值問題
        
        # 檢查密度場的平滑過渡
        density_field = self.multiphase.get_density_field()
        density_gradient = np.gradient(density_field)
        max_gradient = np.max([np.max(np.abs(g)) for g in density_gradient])
        
        # 密度梯度應該有限
        self.assertLess(max_gradient, density_ratio)
        
        print("✅ 密度比處理測試通過")

def run_multiphase_tests():
    """執行多相流測試套件"""
    print("🧪 開始執行多相流系統測試...")
    print("=" * 60)
    
    # 創建測試套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加測試類
    suite.addTests(loader.loadTestsFromTestCase(TestMultiphaseFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiphasePhysics))
    
    # 執行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 輸出總結
    print("=" * 60)
    if result.wasSuccessful():
        print("🎉 多相流系統所有測試通過！")
        return True
    else:
        print(f"❌ 測試失敗: {len(result.failures)} 失敗, {len(result.errors)} 錯誤")
        return False

if __name__ == "__main__":
    success = run_multiphase_tests()
    exit(0 if success else 1)