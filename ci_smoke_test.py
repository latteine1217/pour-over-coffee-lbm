#!/usr/bin/env python3
"""
CI Smoke Test - 快速煙霧測試
用於GitHub Actions中快速驗證核心功能
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti

def test_taichi_init():
    """測試Taichi初始化"""
    try:
        # 直接使用CPU架構，不依賴環境變數
        ti.init(arch=ti.cpu, debug=False, offline_cache=True)
        print("✓ Taichi CPU初始化成功")
        return True
    except Exception as e:
        print(f"✗ Taichi初始化失敗: {e}")
        return False

def test_config_import():
    """測試配置模組導入"""
    try:
        import config
        assert hasattr(config, 'NX')
        assert hasattr(config, 'NY') 
        assert hasattr(config, 'NZ')
        print("✓ 配置模組導入成功")
        return True
    except Exception as e:
        print(f"✗ 配置模組導入失敗: {e}")
        return False

def test_core_modules():
    """測試核心模組導入"""
    try:
        from src.core.lbm_solver import LBMSolver
        from src.core.thermal_fluid_coupled import ThermalFluidCoupledSolver
        print("✓ 核心模組導入成功")
        return True
    except Exception as e:
        print(f"✗ 核心模組導入失敗: {e}")
        return False

def test_basic_simulation():
    """測試基礎模擬功能"""
    try:
        # 超小網格快速測試
        from src.core.lbm_solver import LBMSolver
        
        # 創建最小求解器
        solver = LBMSolver()
        solver.init_fields()
        
        # 運行1步
        solver.step()
        
        print("✓ 基礎模擬測試成功")
        return True
    except Exception as e:
        print(f"✗ 基礎模擬測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("=== CI煙霧測試 ===")
    
    tests = [
        test_taichi_init,
        test_config_import, 
        test_core_modules,
        test_basic_simulation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"=== 測試結果: {passed}/{len(tests)} 通過 ===")
    
    if passed == len(tests):
        print("✅ 所有煙霧測試通過")
        return 0
    else:
        print("❌ 煙霧測試失敗")
        return 1

if __name__ == "__main__":
    sys.exit(main())