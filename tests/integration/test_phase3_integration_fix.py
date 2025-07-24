# test_phase3_integration_fix.py - Phase 3 修正後集成測試
"""
修正後的Phase 3強耦合系統端到端集成測試

測試內容:
1. LBM求解器基礎功能修正驗證
2. 物理模型修正驗證 (Boussinesq一致性、浮力量綱)
3. 數據流完整性驗證 (溫度→物性→LBM時序)
4. 強耦合系統完整功能測試
5. 數值穩定性長期運行測試

開發：opencode + GitHub Copilot
"""

# 設置Python路徑以便導入模組
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import taichi as ti
import numpy as np
import time
import traceback
from typing import Dict, Any, Tuple, List

# 系統模組
import config.config as config
from src.core.strong_coupled_solver import create_coffee_strong_coupling_system
from src.physics.temperature_dependent_properties import create_water_properties
from src.physics.buoyancy_natural_convection import create_coffee_buoyancy_system

# ============================================================================
# 測試工具函數
# ============================================================================

def setup_test_environment():
    """設置測試環境"""
    print("🧪 設置Phase 3修正測試環境...")
    
    # Taichi初始化 (調試模式)
    ti.init(arch=ti.metal, debug=True, device_memory_fraction=0.7)
    
    print(f"   網格大小: {config.NX}×{config.NY}×{config.NZ}")
    print(f"   格子解析度: {config.DX:.4f} m")
    print(f"   時間步長: {config.DT:.6f} s")

def validate_field_ranges(field: ti.field, 
                         field_name: str, 
                         expected_min: float, 
                         expected_max: float,
                         tolerance: float = 0.1) -> bool:
    """驗證場的數值範圍 - 修正版邏輯"""
    
    field_np = field.to_numpy()
    actual_min = float(np.min(field_np))
    actual_max = float(np.max(field_np))
    
    # 修正邏輯：檢查實際範圍是否在期望範圍內 (允許容差)
    min_ok = actual_min >= (expected_min - tolerance)
    max_ok = actual_max <= (expected_max + tolerance)
    
    status = "✅" if (min_ok and max_ok) else "❌"
    print(f"   {status} {field_name}: {actual_min:.3f} - {actual_max:.3f} (期望: {expected_min:.3f} - {expected_max:.3f})")
    
    return min_ok and max_ok

def check_for_numerical_issues(field: ti.field, field_name: str) -> bool:
    """檢查數值問題 (NaN, Inf)"""
    
    field_np = field.to_numpy()
    has_nan = np.any(np.isnan(field_np))
    has_inf = np.any(np.isinf(field_np))
    
    if has_nan or has_inf:
        print(f"❌ {field_name} 包含NaN/Inf值!")
        return False
    
    print(f"✅ {field_name} 數值正常")
    return True

# ============================================================================
# 測試1: LBM求解器基礎功能修正驗證
# ============================================================================

def test_lbm_solver_basic_fixes() -> bool:
    """測試LBM求解器基礎修正"""
    
    print("\n📋 測試1: LBM求解器基礎功能修正驗證")
    print("=" * 50)
    
    try:
        from src.core.lbm_solver import LBMSolver
        
        # 1. 創建求解器並檢查屬性初始化
        print("🔧 創建LBM求解器...")
        solver = LBMSolver()
        
        # 2. 檢查修正的屬性是否存在
        required_attrs = ['cx', 'cy', 'cz', 'f_old', 'e']
        missing_attrs = []
        
        for attr in required_attrs:
            if not hasattr(solver, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"❌ 缺失屬性: {missing_attrs}")
            return False
        
        print("✅ 所有必需屬性已正確初始化")
        
        # 3. 測試基本步驟執行
        print("🏃 測試基本LBM步驟...")
        solver.init_fields()
        
        for step in range(3):
            solver.step()
            
            # 檢查數值穩定性
            if not check_for_numerical_issues(solver.rho, f"密度(步驟{step})"):
                return False
            if not check_for_numerical_issues(solver.u, f"速度(步驟{step})"):
                return False
        
        print("✅ LBM求解器基礎功能修正驗證通過")
        return True
        
    except Exception as e:
        print(f"❌ LBM求解器測試失敗: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# 測試2: 物理模型修正驗證
# ============================================================================

def test_physics_model_fixes() -> bool:
    """測試物理模型修正"""
    
    print("\n📋 測試2: 物理模型修正驗證")
    print("=" * 50)
    
    try:
        # 1. 測試溫度依賴物性計算
        print("🧮 測試溫度依賴物性計算...")
        properties = create_water_properties()
        
        # 創建測試溫度場
        test_temp = ti.field(ti.f32, shape=(config.NX, config.NY, config.NZ))
        test_temp.fill(25.0)  # 參考溫度
        
        # 設置不同溫度區域
        @ti.kernel
        def set_test_temperatures():
            for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
                if k < config.NZ // 3:
                    test_temp[i, j, k] = 15.0  # 冷區
                elif k > 2 * config.NZ // 3:
                    test_temp[i, j, k] = 85.0  # 熱區
                else:
                    test_temp[i, j, k] = 50.0  # 中溫區
        
        set_test_temperatures()
        
        # 更新物性
        properties.update_properties_from_temperature(test_temp)
        
        # 驗證物性範圍 (使用更寬鬆的範圍)
        if not validate_field_ranges(properties.density_field, "密度", 975.0, 1005.0, tolerance=5.0):
            return False
        if not validate_field_ranges(properties.viscosity_field, "黏度", 1e-4, 5e-3, tolerance=1e-3):
            return False
        if not validate_field_ranges(properties.relaxation_time_field, "鬆弛時間", 0.51, 2.0, tolerance=0.5):
            return False
        
        # 2. 測試浮力模型修正
        print("🌊 測試浮力模型修正...")
        buoyancy_system = create_coffee_buoyancy_system(properties)
        
        # 檢查格子單位轉換
        gravity_lattice = buoyancy_system.gravity_lattice_magnitude
        buoyancy_coeff = buoyancy_system.buoyancy_coefficient
        
        print(f"   格子單位重力: {gravity_lattice:.6f}")
        print(f"   浮力係數: {buoyancy_coeff:.6f}")
        
        # 驗證量綱合理性
        if not (1e-8 < gravity_lattice < 1e-3):
            print(f"❌ 格子單位重力量級異常: {gravity_lattice}")
            return False
        
        if not (1e-10 < abs(buoyancy_coeff) < 1e-1):
            print(f"❌ 浮力係數量級異常: {buoyancy_coeff}")
            return False
        
        print("✅ 物理模型修正驗證通過")
        return True
        
    except Exception as e:
        print(f"❌ 物理模型測試失敗: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# 測試3: 數據流完整性驗證
# ============================================================================

def test_data_flow_integrity() -> bool:
    """測試數據流完整性"""
    
    print("\n📋 測試3: 數據流完整性驗證")
    print("=" * 50)
    
    try:
        # 創建強耦合系統
        print("🔗 創建強耦合系統...")
        coupled_system = create_coffee_strong_coupling_system()
        
        # 初始化條件
        fluid_conditions = {}
        thermal_conditions = {
            'T_initial': 25.0,
            'T_hot_region': 80.0,
            'hot_region_height': 20
        }
        
        coupled_system.initialize_coupled_system(fluid_conditions, thermal_conditions)
        
        # 測試數據流時序
        print("🔄 測試數據流時序...")
        
        initial_temp_stats = coupled_system.thermal_solver.get_temperature_stats()
        print(f"   初始溫度: {initial_temp_stats[0]:.1f} - {initial_temp_stats[1]:.1f}°C")
        
        # 執行幾個耦合步驟
        for step in range(5):
            print(f"   執行耦合步驟 {step+1}...")
            success = coupled_system.coupled_step()
            
            if not success:
                print(f"❌ 耦合步驟{step+1}失敗")
                return False
            
            # 檢查物性更新是否正常
            prop_stats = coupled_system.properties_calculator.get_property_statistics()
            density_range = prop_stats['density']
            
            if not (980.0 <= density_range['min'] <= density_range['max'] <= 1010.0):
                print(f"❌ 步驟{step+1}密度範圍異常: {density_range}")
                return False
            
            # 檢查浮力統計
            if coupled_system.buoyancy_system:
                buoyancy_diag = coupled_system.buoyancy_system.get_natural_convection_diagnostics()
                total_buoyancy = buoyancy_diag['total_buoyancy_force']
                
                if abs(total_buoyancy) > 1e6:  # 避免過大的浮力
                    print(f"❌ 步驟{step+1}浮力過大: {total_buoyancy}")
                    return False
        
        print("✅ 數據流完整性驗證通過")
        return True
        
    except Exception as e:
        print(f"❌ 數據流測試失敗: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# 測試4: 強耦合系統完整功能測試
# ============================================================================

def test_full_coupling_system() -> bool:
    """測試完整強耦合系統"""
    
    print("\n📋 測試4: 強耦合系統完整功能測試")
    print("=" * 50)
    
    try:
        # 創建系統
        coupled_system = create_coffee_strong_coupling_system()
        
        # 完整初始化
        fluid_conditions = {}
        thermal_conditions = {
            'T_initial': 20.0,
            'T_hot_region': 90.0,
            'hot_region_height': 30
        }
        
        coupled_system.initialize_coupled_system(fluid_conditions, thermal_conditions)
        
        print("🔄 執行完整耦合模擬...")
        
        # 運行較長的模擬
        max_steps = 15
        diagnostics_history = []
        
        for step in range(max_steps):
            step_start = time.time()
            
            success = coupled_system.coupled_step()
            if not success:
                print(f"❌ 耦合失敗於步驟{step+1}")
                return False
            
            # 獲取診斷信息
            diag = coupled_system.get_strong_coupling_diagnostics()
            diagnostics_history.append(diag)
            
            step_time = time.time() - step_start
            
            # 每5步報告一次
            if (step + 1) % 5 == 0:
                thermal_stats = diag.get('thermal_stats', {})
                T_avg = thermal_stats.get('T_avg', 0)
                
                buoyancy_stats = diag.get('buoyancy_stats', {})
                rayleigh = buoyancy_stats.get('rayleigh_number', 0)
                
                print(f"   步驟{step+1}: T_avg={T_avg:.1f}°C, Ra={rayleigh:.1e}, 時間={step_time:.3f}s")
        
        # 驗證最終狀態
        final_diag = diagnostics_history[-1]
        
        # 檢查溫度範圍
        thermal_stats = final_diag.get('thermal_stats', {})
        T_min = thermal_stats.get('T_min', 0)
        T_max = thermal_stats.get('T_max', 0)
        
        if not (10.0 <= T_min <= T_max <= 100.0):
            print(f"❌ 最終溫度範圍異常: {T_min:.1f} - {T_max:.1f}°C")
            return False
        
        # 檢查性能統計
        performance = final_diag.get('performance', {})
        if performance:
            steps_per_sec = performance.get('steps_per_second', 0)
            print(f"   性能: {steps_per_sec:.2f} 步/秒")
        
        print("✅ 強耦合系統完整功能測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 完整系統測試失敗: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# 測試5: 數值穩定性長期測試
# ============================================================================

def test_numerical_stability() -> bool:
    """測試數值穩定性"""
    
    print("\n📋 測試5: 數值穩定性長期測試")
    print("=" * 50)
    
    try:
        coupled_system = create_coffee_strong_coupling_system()
        
        # 初始化
        fluid_conditions = {}
        thermal_conditions = {
            'T_initial': 25.0,
            'T_hot_region': 75.0,
            'hot_region_height': 25
        }
        
        coupled_system.initialize_coupled_system(fluid_conditions, thermal_conditions)
        
        print("⏱️ 執行長期穩定性測試...")
        
        # 長期運行
        max_steps = 25
        stability_issues = []
        
        for step in range(max_steps):
            success = coupled_system.coupled_step()
            
            if not success:
                stability_issues.append(f"耦合失敗於步驟{step+1}")
                break
            
            # 檢查數值穩定性
            diag = coupled_system.get_strong_coupling_diagnostics()
            
            # 溫度穩定性
            thermal_stats = diag.get('thermal_stats', {})
            T_min = thermal_stats.get('T_min', 0)
            T_max = thermal_stats.get('T_max', 0)
            
            if T_max > 150.0 or T_min < -10.0:
                stability_issues.append(f"步驟{step+1}: 溫度失控 {T_min:.1f}-{T_max:.1f}°C")
            
            # 浮力穩定性
            buoyancy_stats = diag.get('buoyancy_stats', {})
            if buoyancy_stats:
                total_buoyancy = buoyancy_stats.get('total_buoyancy_force', 0)
                if abs(total_buoyancy) > 1e8:
                    stability_issues.append(f"步驟{step+1}: 浮力發散 {total_buoyancy:.2e}")
            
            # 每10步報告
            if (step + 1) % 10 == 0:
                print(f"   ✓ 步驟{step+1}: 穩定運行")
        
        # 評估穩定性結果
        if stability_issues:
            print(f"❌ 發現{len(stability_issues)}個穩定性問題:")
            for issue in stability_issues[:3]:  # 顯示前3個
                print(f"     - {issue}")
            return False
        
        print("✅ 數值穩定性長期測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 穩定性測試失敗: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# 主測試執行器
# ============================================================================

def run_phase3_integration_tests() -> Dict[str, bool]:
    """運行所有Phase 3修正集成測試"""
    
    print("🧪 Phase 3 修正後完整集成測試")
    print("=" * 60)
    
    # 設置環境
    setup_test_environment()
    
    # 測試列表
    tests = [
        ("LBM求解器基礎修正", test_lbm_solver_basic_fixes),
        ("物理模型修正", test_physics_model_fixes),
        ("數據流完整性", test_data_flow_integrity),
        ("完整耦合系統", test_full_coupling_system),
        ("數值穩定性", test_numerical_stability)
    ]
    
    results = {}
    start_time = time.time()
    
    # 執行所有測試
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            test_start = time.time()
            result = test_func()
            test_time = time.time() - test_start
            
            results[test_name] = result
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"{status} - 用時: {test_time:.2f}s")
            
        except Exception as e:
            results[test_name] = False
            print(f"❌ 測試異常: {e}")
    
    total_time = time.time() - start_time
    
    # 測試總結
    print(f"\n{'='*60}")
    print("📊 Phase 3 修正測試總結")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n📈 通過率: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"⏱️ 總用時: {total_time:.2f}秒")
    
    if passed == total:
        print("🎉 所有測試通過！Phase 3修正成功！")
    else:
        print("⚠️ 部分測試失敗，需要進一步調試")
    
    return results

if __name__ == "__main__":
    results = run_phase3_integration_tests()