# simple_coupling_test.py - 簡單耦合驗證測試
"""
簡化版Phase 2耦合驗證
不依賴pytest，直接驗證核心功能

測試內容：
1. 基本初始化
2. 單步運行
3. 短期穩定性
4. 基本性能

開發：opencode + GitHub Copilot
"""

# 設置Python路徑以便導入模組
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import taichi as ti
import numpy as np
import time
import sys

# 初始化Taichi
try:
    ti.init(arch=ti.metal, device_memory_GB=2.0)
    print("✅ Taichi Metal GPU初始化成功")
except:
    ti.init(arch=ti.cpu)
    print("⚠️  回退到CPU模式")

# 導入模組
try:
    from src.core.thermal_fluid_coupled import ThermalFluidCoupledSolver, CouplingConfig
    from src.physics.thermal_lbm import ThermalLBM
    from src.core.lbm_solver import LBMSolver
    import config
    print("✅ 所有模組導入成功")
except ImportError as e:
    print(f"❌ 模組導入失敗: {e}")
    sys.exit(1)

def test_basic_initialization():
    """測試基本初始化"""
    print("\n🧪 測試1: 基本初始化")
    
    try:
        # 簡單配置
        coupling_config = CouplingConfig(
            coupling_frequency=1,
            velocity_smoothing=False,
            thermal_subcycles=1,
            enable_diagnostics=True
        )
        
        # 創建耦合系統
        coupled_solver = ThermalFluidCoupledSolver(
            coupling_config=coupling_config,
            thermal_diffusivity=1.6e-7
        )
        
        # 檢查基本屬性
        assert hasattr(coupled_solver, 'fluid_solver')
        assert hasattr(coupled_solver, 'thermal_solver')
        assert coupled_solver.is_initialized == False
        
        print("✅ 基本初始化測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 初始化測試失敗: {e}")
        return False

def test_system_initialization():
    """測試系統初始化"""
    print("\n🧪 測試2: 系統初始化")
    
    try:
        # 創建系統
        coupling_config = CouplingConfig(coupling_frequency=1)
        coupled_solver = ThermalFluidCoupledSolver(coupling_config=coupling_config)
        
        # 準備初始條件
        fluid_conditions = {
            'density_field': np.ones((config.NX, config.NY, config.NZ), dtype=np.float32)
        }
        
        thermal_conditions = {
            'T_initial': 25.0,
            'T_hot_region': 80.0,
            'hot_region_height': 15
        }
        
        # 基礎熱源
        base_heat_source = np.zeros((config.NX, config.NY, config.NZ), dtype=np.float32)
        base_heat_source[:, :, :5] = 50.0  # 小的熱源
        
        # 初始化系統
        coupled_solver.initialize_system(
            fluid_initial_conditions=fluid_conditions,
            thermal_initial_conditions=thermal_conditions,
            base_heat_source=base_heat_source
        )
        
        assert coupled_solver.is_initialized == True
        assert coupled_solver.thermal_solver.enable_convection == True
        
        print("✅ 系統初始化測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 系統初始化測試失敗: {e}")
        return False

def test_single_step():
    """測試單步運行"""
    print("\n🧪 測試3: 單步運行")
    
    try:
        # 創建並初始化系統
        coupling_config = CouplingConfig(coupling_frequency=1)
        coupled_solver = ThermalFluidCoupledSolver(coupling_config=coupling_config)
        
        # 初始條件
        fluid_conditions = {'density_field': np.ones((config.NX, config.NY, config.NZ), dtype=np.float32)}
        thermal_conditions = {'T_initial': 25.0, 'T_hot_region': 70.0, 'hot_region_height': 10}
        base_heat_source = np.zeros((config.NX, config.NY, config.NZ), dtype=np.float32)
        
        coupled_solver.initialize_system(
            fluid_initial_conditions=fluid_conditions,
            thermal_initial_conditions=thermal_conditions,
            base_heat_source=base_heat_source
        )
        
        # 執行單步
        start_time = time.time()
        success = coupled_solver.step()
        step_time = time.time() - start_time
        
        assert success == True, "單步執行失敗"
        assert coupled_solver.coupling_step == 1
        
        # 檢查診斷資訊
        diagnostics = coupled_solver.get_coupling_diagnostics()
        
        print(f"✅ 單步運行測試通過 (耗時: {step_time:.3f}s)")
        print(f"   溫度範圍: {diagnostics['thermal_stats']['T_min']:.1f} - {diagnostics['thermal_stats']['T_max']:.1f}°C")
        return True
        
    except Exception as e:
        print(f"❌ 單步運行測試失敗: {e}")
        return False

def test_multi_step_stability():
    """測試多步穩定性"""
    print("\n🧪 測試4: 多步穩定性 (5步)")
    
    try:
        # 創建系統
        coupling_config = CouplingConfig(coupling_frequency=1, enable_diagnostics=True)
        coupled_solver = ThermalFluidCoupledSolver(coupling_config=coupling_config)
        
        # 初始化
        fluid_conditions = {'density_field': np.ones((config.NX, config.NY, config.NZ), dtype=np.float32)}
        thermal_conditions = {'T_initial': 25.0, 'T_hot_region': 65.0, 'hot_region_height': 8}
        base_heat_source = np.zeros((config.NX, config.NY, config.NZ), dtype=np.float32)
        
        coupled_solver.initialize_system(
            fluid_initial_conditions=fluid_conditions,
            thermal_initial_conditions=thermal_conditions,
            base_heat_source=base_heat_source
        )
        
        # 執行多步
        test_steps = 5  # 保守的步數
        successful_steps = 0
        
        for step in range(test_steps):
            success = coupled_solver.step()
            
            if success:
                successful_steps += 1
                
                # 獲取溫度統計
                diagnostics = coupled_solver.get_coupling_diagnostics()
                temp_stats = diagnostics['thermal_stats']
                
                print(f"   步驟{step+1}: T_avg={temp_stats['T_avg']:.1f}°C")
                
                # 基本合理性檢查
                if temp_stats['T_min'] < -50 or temp_stats['T_max'] > 200:
                    print(f"⚠️  溫度超出合理範圍")
                    break
            else:
                print(f"❌ 步驟{step+1}失敗")
                break
        
        success_rate = successful_steps / test_steps
        
        print(f"✅ 多步穩定性測試完成 ({successful_steps}/{test_steps}步, {success_rate:.0%})")
        return success_rate >= 0.8  # 80%成功率
        
    except Exception as e:
        print(f"❌ 多步穩定性測試失敗: {e}")
        return False

def test_performance_basic():
    """測試基本性能"""
    print("\n🧪 測試5: 基本性能")
    
    try:
        # 創建系統
        coupling_config = CouplingConfig(coupling_frequency=1)
        coupled_solver = ThermalFluidCoupledSolver(coupling_config=coupling_config)
        
        # 初始化
        fluid_conditions = {'density_field': np.ones((config.NX, config.NY, config.NZ), dtype=np.float32)}
        thermal_conditions = {'T_initial': 25.0, 'T_hot_region': 60.0, 'hot_region_height': 5}
        base_heat_source = np.zeros((config.NX, config.NY, config.NZ), dtype=np.float32)
        
        coupled_solver.initialize_system(
            fluid_initial_conditions=fluid_conditions,
            thermal_initial_conditions=thermal_conditions,
            base_heat_source=base_heat_source
        )
        
        # 性能測試
        test_steps = 3  # 保守的步數
        start_time = time.time()
        
        for step in range(test_steps):
            success = coupled_solver.step()
            if not success:
                print(f"性能測試在步驟{step}失敗")
                return False
        
        total_time = time.time() - start_time
        steps_per_second = test_steps / total_time
        
        # 獲取性能統計
        diagnostics = coupled_solver.get_coupling_diagnostics()
        performance = diagnostics['performance']
        
        print(f"✅ 基本性能測試完成")
        print(f"   總時間: {total_time:.3f}s")
        print(f"   步數/秒: {steps_per_second:.2f}")
        print(f"   流體佔比: {performance['fluid_fraction']:.1%}")
        print(f"   熱傳佔比: {performance['thermal_fraction']:.1%}")
        
        # 寬鬆的性能要求
        return steps_per_second > 0.05  # 至少0.05步/秒
        
    except Exception as e:
        print(f"❌ 基本性能測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    
    print("=" * 60)
    print("🧪 Phase 2 熱流弱耦合簡化驗證測試")
    print(f"🔧 網格尺寸: {config.NX}×{config.NY}×{config.NZ}")
    print("=" * 60)
    
    # 執行測試
    tests = [
        ("基本初始化", test_basic_initialization),
        ("系統初始化", test_system_initialization),
        ("單步運行", test_single_step),
        ("多步穩定性", test_multi_step_stability),
        ("基本性能", test_performance_basic)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed_tests += 1
            else:
                print(f"❌ {test_name} 測試失敗")
        except Exception as e:
            print(f"❌ {test_name} 測試異常: {e}")
    
    # 總結
    print("\n" + "=" * 60)
    success_rate = passed_tests / total_tests
    
    if success_rate >= 0.8:
        print(f"🎉 Phase 2 弱耦合驗證成功！({passed_tests}/{total_tests})")
        print("✅ 熱流弱耦合系統基本功能正常")
        print("🚀 可以繼續Phase 3開發")
    else:
        print(f"⚠️  Phase 2 驗證部分失敗 ({passed_tests}/{total_tests})")
        print("🔧 需要進一步調試和優化")
    
    print("=" * 60)
    
    return success_rate >= 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)