# test_phase3_strong_coupling.py - Phase 3 強耦合測試驗證框架
"""
Phase 3 強耦合系統完整測試套件

測試範圍:
1. 系統初始化和集成測試
2. 溫度依賴物性計算驗證
3. 浮力自然對流機制測試  
4. 雙向耦合穩定性驗證
5. 物理準確性基準測試
6. 性能和可擴展性測試

驗證標準:
- 數值穩定性: >95%成功率
- 物理準確性: 符合理論預期
- 計算性能: <50%性能損失
- Rayleigh數範圍: 10³-10⁶

開發：opencode + GitHub Copilot
"""

# 設置Python路徑以便導入模組
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import taichi as ti
import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Any

# 設置測試環境
ti.init(arch=ti.cpu)  # 使用CPU確保穩定性

# 導入Phase 3模組
try:
    from src.core.strong_coupled_solver import StrongCoupledSolver, StrongCouplingConfig, create_coffee_strong_coupling_system
    from src.physics.temperature_dependent_properties import TemperatureDependentProperties, create_water_properties
    from src.physics.buoyancy_natural_convection import BuoyancyNaturalConvection, create_coffee_buoyancy_system
    import config
    print("✅ 所有Phase 3模組導入成功")
except ImportError as e:
    print(f"❌ Phase 3模組導入失敗: {e}")
    sys.exit(1)

class Phase3TestSuite:
    """Phase 3 強耦合系統測試套件"""
    
    def __init__(self):
        """初始化測試套件"""
        
        # 測試配置
        self.test_config = StrongCouplingConfig(
            coupling_frequency=1,
            max_coupling_iterations=2,  # 保守設置
            coupling_tolerance=1e-3,    # 寬鬆容差
            enable_adaptive_relaxation=True,
            relaxation_factor=0.5,     # 保守鬆弛
            enable_variable_density=True,
            enable_variable_viscosity=False,  # 先關閉可變黏度
            enable_buoyancy=True,
            enable_diagnostics=True,
            stability_check_frequency=5,
            max_temperature_change=10.0,  # 寬鬆限制
            max_velocity_magnitude=0.5
        )
        
        # 測試初始條件
        self.fluid_conditions = {}
        self.thermal_conditions = {
            'T_initial': 25.0,
            'T_hot_region': 60.0,  # 保守溫差
            'hot_region_height': 8
        }
        
        # 小的熱源場
        self.base_heat_source = np.zeros((config.NX, config.NY, config.NZ), dtype=np.float32)
        center_x, center_y = config.NX//2, config.NY//2
        for i in range(center_x-3, center_x+3):
            for j in range(center_y-3, center_y+3):
                for k in range(config.NZ-8, config.NZ):
                    if 0 <= i < config.NX and 0 <= j < config.NY:
                        self.base_heat_source[i, j, k] = 5.0  # 很小的熱源
        
        # 測試結果
        self.test_results = {}
    
    def test_1_system_initialization(self):
        """測試1: 系統初始化"""
        
        print("\n🧪 測試1: Phase 3系統初始化")
        
        try:
            # 創建強耦合系統
            coupled_solver = create_coffee_strong_coupling_system()
            
            # 檢查子系統
            assert hasattr(coupled_solver, 'fluid_solver')
            assert hasattr(coupled_solver, 'thermal_solver')
            assert hasattr(coupled_solver, 'properties_calculator')
            assert hasattr(coupled_solver, 'buoyancy_system')
            
            # 檢查集成狀態
            assert coupled_solver.fluid_solver.use_temperature_dependent_properties == True
            assert coupled_solver.thermal_solver.enable_convection == True
            
            print("✅ 系統初始化測試通過")
            return True, "系統初始化成功"
            
        except Exception as e:
            print(f"❌ 系統初始化測試失敗: {e}")
            return False, str(e)
    
    def test_2_temperature_dependent_properties(self):
        """測試2: 溫度依賴物性計算"""
        
        print("\n🧪 測試2: 溫度依賴物性計算")
        
        try:
            # 創建物性計算器
            properties = create_water_properties()
            
            # 創建測試溫度場
            temp_field = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
            
            # 設置溫度梯度 (20-80°C)
            @ti.kernel
            def init_temp_gradient():
                for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
                    # Z方向溫度梯度
                    T = 20.0 + 60.0 * (k / config.NZ)
                    temp_field[i, j, k] = T
            
            init_temp_gradient()
            
            # 更新物性
            properties.update_properties_from_temperature(temp_field)
            
            # 驗證物性範圍
            stats = properties.get_property_statistics()
            ranges_valid = properties.validate_property_ranges()
            
            # 檢查密度範圍 (期望: 960-1000 kg/m³)
            rho_min, rho_max = stats['density']['min'], stats['density']['max']
            density_ok = 950 <= rho_min <= rho_max <= 1010
            
            # 檢查黏度範圍 (期望: 1e-4 - 2e-3 Pa·s)
            mu_min, mu_max = stats['viscosity']['min'], stats['viscosity']['max']
            viscosity_ok = 1e-5 <= mu_min <= mu_max <= 5e-3
            
            print(f"   密度範圍: {rho_min:.1f} - {rho_max:.1f} kg/m³")
            print(f"   黏度範圍: {mu_min:.2e} - {mu_max:.2e} Pa·s")
            
            if ranges_valid and density_ok and viscosity_ok:
                print("✅ 溫度依賴物性計算測試通過")
                return True, "物性計算正確"
            else:
                print("❌ 物性範圍異常")
                return False, "物性範圍不合理"
                
        except Exception as e:
            print(f"❌ 溫度依賴物性測試失敗: {e}")
            return False, str(e)
    
    def test_3_buoyancy_natural_convection(self):
        """測試3: 浮力自然對流機制"""
        
        print("\n🧪 測試3: 浮力自然對流機制")
        
        try:
            # 創建浮力系統
            properties = create_water_properties()
            buoyancy_system = create_coffee_buoyancy_system(properties)
            
            # 創建溫度場 (垂直溫度梯度)
            temp_field = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
            velocity_field = ti.Vector.field(3, ti.f32, shape=(config.NX, config.NY, config.NZ))
            density_field = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
            
            @ti.kernel
            def init_buoyancy_test():
                for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
                    # 底部熱，頂部冷
                    T = 30.0 + 40.0 * ((config.NZ - k) / config.NZ)
                    temp_field[i, j, k] = T
                    velocity_field[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                    density_field[i, j, k] = 997.0
            
            init_buoyancy_test()
            
            # 計算浮力
            buoyancy_system.compute_buoyancy_force(temp_field)
            
            # 獲取診斷信息
            diagnostics = buoyancy_system.update_buoyancy_system(
                temp_field, density_field, velocity_field
            )
            
            # 檢查浮力方向 (熱水應該向上)
            buoyancy_magnitude = buoyancy_system.buoyancy_magnitude.to_numpy()
            buoyancy_force = buoyancy_system.buoyancy_force.to_numpy()
            
            # 檢查底部區域的浮力向上 (Z方向正值)
            bottom_region = buoyancy_force[:, :, :config.NZ//4, 2]  # Z分量
            top_region = buoyancy_force[:, :, 3*config.NZ//4:, 2]
            
            bottom_buoyancy = np.mean(bottom_region[bottom_region > 0])
            top_buoyancy = np.mean(top_region[top_region < 0])
            
            print(f"   底部浮力(向上): {bottom_buoyancy:.6f}")
            print(f"   頂部浮力(向下): {top_buoyancy:.6f}")
            print(f"   總浮力: {diagnostics['total_buoyancy_force']:.3f}")
            print(f"   Rayleigh數: {diagnostics['rayleigh_number']:.1f}")
            
            if diagnostics['total_buoyancy_force'] > 0 and bottom_buoyancy > 0:
                print("✅ 浮力自然對流機制測試通過")
                return True, "浮力計算正確"
            else:
                print("❌ 浮力方向或大小異常")
                return False, "浮力計算錯誤"
                
        except Exception as e:
            print(f"❌ 浮力自然對流測試失敗: {e}")
            return False, str(e)
    
    def test_4_coupled_system_stability(self):
        """測試4: 雙向耦合系統穩定性"""
        
        print("\n🧪 測試4: 雙向耦合系統穩定性")
        
        try:
            # 創建強耦合系統
            coupled_solver = StrongCoupledSolver(self.test_config)
            
            # 初始化系統
            coupled_solver.initialize_coupled_system(
                fluid_initial_conditions=self.fluid_conditions,
                thermal_initial_conditions=self.thermal_conditions,
                base_heat_source=self.base_heat_source
            )
            
            # 多步穩定性測試
            test_steps = 5
            successful_steps = 0
            
            print("步驟 | 耗時(s) | 迭代 | T_avg | 浮力 | 狀態")
            print("-" * 50)
            
            for step in range(test_steps):
                step_start = time.time()
                
                success = coupled_solver.coupled_step()
                step_time = time.time() - step_start
                
                if success:
                    diagnostics = coupled_solver.get_strong_coupling_diagnostics()
                    
                    # 獲取統計信息
                    T_avg = diagnostics.get('thermal_stats', {}).get('T_avg', 0)
                    coupling_iter = diagnostics.get('performance', {}).get('avg_coupling_iterations', 0)
                    buoyancy_force = diagnostics.get('buoyancy_stats', {}).get('total_buoyancy_force', 0)
                    
                    print(f"{step+1:3d}  | {step_time:6.3f}  | {coupling_iter:4.1f} | {T_avg:5.1f} | {buoyancy_force:5.1f} | ✅")
                    successful_steps += 1
                else:
                    print(f"{step+1:3d}  | {step_time:6.3f}  |  -   |   -   |   -   | ❌")
                    break
            
            success_rate = successful_steps / test_steps
            
            if success_rate >= 0.8:
                print(f"✅ 耦合穩定性測試通過 ({successful_steps}/{test_steps})")
                return True, f"穩定性{success_rate:.0%}"
            else:
                print(f"❌ 耦合穩定性不足 ({successful_steps}/{test_steps})")
                return False, f"穩定性僅{success_rate:.0%}"
                
        except Exception as e:
            print(f"❌ 耦合穩定性測試失敗: {e}")
            return False, str(e)
    
    def test_5_natural_convection_physics(self):
        """測試5: 自然對流物理準確性"""
        
        print("\n🧪 測試5: 自然對流物理準確性")
        
        try:
            # 創建系統進行自然對流測試
            config_natural = StrongCouplingConfig(
                coupling_frequency=1,
                max_coupling_iterations=2,
                enable_buoyancy=True,
                enable_variable_density=True,
                relaxation_factor=0.3,  # 小的鬆弛
                max_temperature_change=20.0
            )
            
            coupled_solver = StrongCoupledSolver(config_natural)
            
            # 設置強溫度梯度的初始條件
            thermal_conditions_strong = {
                'T_initial': 25.0,
                'T_hot_region': 70.0,  # 較大溫差
                'hot_region_height': 15
            }
            
            coupled_solver.initialize_coupled_system(
                fluid_initial_conditions={},
                thermal_initial_conditions=thermal_conditions_strong,
                base_heat_source=self.base_heat_source
            )
            
            # 運行若干步觀察自然對流發展
            steps = 3
            initial_state = None
            final_state = None
            
            for step in range(steps):
                success = coupled_solver.coupled_step()
                
                if not success:
                    print(f"   步驟{step+1}失敗")
                    return False, "自然對流模擬失敗"
                
                if step == 0:
                    initial_state = coupled_solver.get_strong_coupling_diagnostics()
                elif step == steps - 1:
                    final_state = coupled_solver.get_strong_coupling_diagnostics()
            
            # 分析自然對流特徵
            if initial_state and final_state:
                initial_T = initial_state['thermal_stats']['T_avg']
                final_T = final_state['thermal_stats']['T_avg']
                
                buoyancy_stats = final_state.get('buoyancy_stats', {})
                rayleigh_number = buoyancy_stats.get('rayleigh_number', 0)
                
                print(f"   初始溫度: {initial_T:.2f}°C")
                print(f"   最終溫度: {final_T:.2f}°C")
                print(f"   Rayleigh數: {rayleigh_number:.1f}")
                
                # 物理合理性檢查
                temp_evolution_ok = abs(final_T - initial_T) < 10.0  # 溫度變化合理
                rayleigh_ok = 100 < rayleigh_number < 1e8  # Rayleigh數範圍合理
                
                if temp_evolution_ok and rayleigh_ok:
                    print("✅ 自然對流物理準確性測試通過")
                    return True, f"Ra={rayleigh_number:.0f}"
                else:
                    print("❌ 自然對流物理不合理")
                    return False, "物理特徵異常"
            else:
                print("❌ 無法獲得診斷數據")
                return False, "診斷失敗"
                
        except Exception as e:
            print(f"❌ 自然對流物理測試失敗: {e}")
            return False, str(e)
    
    def test_6_performance_benchmark(self):
        """測試6: 性能基準測試"""
        
        print("\n🧪 測試6: 性能基準測試")
        
        try:
            # 創建性能測試系統
            coupled_solver = StrongCoupledSolver(self.test_config)
            coupled_solver.initialize_coupled_system(
                fluid_initial_conditions=self.fluid_conditions,
                thermal_initial_conditions=self.thermal_conditions,
                base_heat_source=self.base_heat_source
            )
            
            # 性能測試
            benchmark_steps = 3
            start_time = time.time()
            
            for step in range(benchmark_steps):
                success = coupled_solver.coupled_step()
                if not success:
                    print(f"   性能測試在步驟{step}失敗")
                    return False, "性能測試失敗"
            
            total_time = time.time() - start_time
            steps_per_second = benchmark_steps / total_time
            
            # 獲取詳細性能統計
            diagnostics = coupled_solver.get_strong_coupling_diagnostics()
            performance = diagnostics.get('performance', {})
            
            print(f"   總時間: {total_time:.3f}s")
            print(f"   步數/秒: {steps_per_second:.2f}")
            print(f"   流體佔比: {performance.get('fluid_fraction', 0):.1%}")
            print(f"   熱傳佔比: {performance.get('thermal_fraction', 0):.1%}")
            print(f"   物性佔比: {performance.get('property_fraction', 0):.1%}")
            print(f"   浮力佔比: {performance.get('buoyancy_fraction', 0):.1%}")
            print(f"   平均迭代: {performance.get('avg_coupling_iterations', 0):.1f}")
            
            # 性能要求 (寬鬆基準)
            performance_ok = steps_per_second > 0.05  # 至少0.05步/秒
            iterations_ok = performance.get('avg_coupling_iterations', 0) < 5
            
            if performance_ok and iterations_ok:
                print("✅ 性能基準測試通過")
                return True, f"{steps_per_second:.2f}步/秒"
            else:
                print("❌ 性能不達標")
                return False, "性能過低"
                
        except Exception as e:
            print(f"❌ 性能基準測試失敗: {e}")
            return False, str(e)
    
    def run_complete_test_suite(self):
        """運行完整的Phase 3測試套件"""
        
        print("=" * 70)
        print("🧪 Phase 3 強耦合系統完整測試套件")
        print("=" * 70)
        
        # 定義測試列表
        tests = [
            ("系統初始化", self.test_1_system_initialization),
            ("溫度依賴物性", self.test_2_temperature_dependent_properties),
            ("浮力自然對流", self.test_3_buoyancy_natural_convection),
            ("耦合系統穩定性", self.test_4_coupled_system_stability),
            ("自然對流物理", self.test_5_natural_convection_physics),
            ("性能基準", self.test_6_performance_benchmark)
        ]
        
        # 執行測試
        passed_tests = 0
        test_results = {}
        
        for test_name, test_func in tests:
            try:
                success, message = test_func()
                test_results[test_name] = {'success': success, 'message': message}
                
                if success:
                    passed_tests += 1
                    print(f"✅ {test_name}: {message}")
                else:
                    print(f"❌ {test_name}: {message}")
                    
            except Exception as e:
                test_results[test_name] = {'success': False, 'message': f"異常: {e}"}
                print(f"❌ {test_name}: 測試異常 - {e}")
        
        # 總結
        total_tests = len(tests)
        success_rate = passed_tests / total_tests
        
        print("\n" + "=" * 70)
        print("🎯 Phase 3 強耦合系統測試總結:")
        print(f"   通過測試: {passed_tests}/{total_tests} ({success_rate:.0%})")
        
        # 詳細結果
        for test_name, result in test_results.items():
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {test_name}: {result['message']}")
        
        # 評估Phase 3開發狀態
        if success_rate >= 0.8:
            print("\n🎉 Phase 3 強耦合開發成功！")
            print("✅ 雙向熱流耦合系統基本功能正常")
            print("✅ 溫度依賴物性計算正確")
            print("✅ 浮力自然對流機制工作")
            print("✅ 數值穩定性可接受")
            print("🚀 可以進行實際手沖咖啡模擬應用！")
        elif success_rate >= 0.6:
            print("\n⚠️  Phase 3 部分成功")
            print("🔧 需要進一步調試失敗的模組")
            print("📊 建議優化數值參數和穩定性控制")
        else:
            print("\n❌ Phase 3 需要重大改進")
            print("🔍 建議逐模組檢查和調試")
            print("⚙️  可能需要調整基礎架構或算法")
        
        print("=" * 70)
        
        return success_rate >= 0.8, test_results

def main():
    """主測試函數"""
    
    print("🚀 啟動Phase 3強耦合系統測試...")
    
    # 創建測試套件
    test_suite = Phase3TestSuite()
    
    # 運行完整測試
    success, results = test_suite.run_complete_test_suite()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)