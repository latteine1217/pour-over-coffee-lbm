# test_lbm_solver_unit.py
"""
LBM求解器單元測試
測試核心功能的正確性和數值穩定性

開發：opencode + GitHub Copilot
"""

# 設置Python路徑以便導入模組
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

import unittest
import numpy as np
import taichi as ti
import config
from src.core.lbm_solver import LBMSolver
from src.physics.boundary_conditions import BoundaryConditionManager
from src.core.numerical_stability import NumericalStabilityMonitor

@ti.data_oriented
class TestLBMSolver(unittest.TestCase):
    """LBM求解器測試類"""
    
    @classmethod
    def setUpClass(cls):
        """測試類初始化"""
        # Taichi已經在init.py中初始化，跳過檢查
        pass
    
    def setUp(self):
        """每個測試前的初始化"""
        self.manager = BoundaryConditionManager()
        self.solver = LBMSolver()
        self.solver.init_fields()
    
    def test_boundary_manager_initialization(self):
        """測試邊界條件管理器初始化"""
        self.assertIsNotNone(self.manager.bounce_back)
        self.assertIsNotNone(self.manager.outlet)
        self.assertIsNotNone(self.manager.top)
        self.assertIsNotNone(self.manager.bottom)
        
        info = self.manager.get_boundary_info()
        self.assertEqual(len(info), 4)
        
        print("✅ 邊界條件管理器初始化測試通過")
    
    def test_boundary_application(self):
        """測試邊界條件應用"""
        try:
            self.manager.apply_all_boundaries(self.solver)
            success = True
        except Exception as e:
            success = False
            print(f"邊界條件應用失敗: {e}")
        
        self.assertTrue(success, "邊界條件應用失敗")
        print("✅ 邊界條件應用測試通過")

class TestNumericalStability(unittest.TestCase):
    """數值穩定性測試類"""
    
    @classmethod
    def setUpClass(cls):
        """測試類初始化"""
        # Taichi已經在init.py中初始化，跳過檢查
        pass
    
    def setUp(self):
        """每個測試前的初始化"""
        self.monitor = NumericalStabilityMonitor()
        self.solver = LBMSolver()
        self.solver.init_fields()
    
    def test_stability_monitor_initialization(self):
        """測試穩定性監控器初始化"""
        self.assertIsNotNone(self.monitor.max_velocity)
        self.assertIsNotNone(self.monitor.min_density)
        self.assertEqual(self.monitor.consecutive_errors, 0)
        
        print("✅ 穩定性監控器初始化測試通過")
    
    def test_stability_check(self):
        """測試穩定性檢查"""
        # 正常情況下應該穩定
        status = self.monitor.check_field_stability(self.solver)
        self.assertEqual(status, 0, "正常初始化狀態應該穩定")
        
        report = self.monitor.diagnose_stability(self.solver, 0)
        self.assertTrue(report['is_stable'])
        self.assertEqual(report['nan_count'], 0)
        
        print("✅ 穩定性檢查測試通過")

def run_all_tests():
    """執行所有測試"""
    print("🧪 開始執行LBM核心模塊測試套件...")
    print("=" * 50)
    
    # 創建測試套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加測試類
    suite.addTests(loader.loadTestsFromTestCase(TestLBMSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundaryConditions))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalStability))
    
    # 執行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 輸出總結
    print("=" * 50)
    if result.wasSuccessful():
        print("🎉 所有測試通過！")
        return True
    else:
        print(f"❌ 測試失敗: {len(result.failures)} 失敗, {len(result.errors)} 錯誤")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)