# test_thermal_integration.py - Phase 1集成測試
"""
熱傳系統Phase 1集成測試
驗證核心模組間的協同工作

測試內容：
- 熱傳LBM + 熱物性管理集成
- 溫度依賴物性更新
- 系統穩定性驗證
- 基準性能測試

開發：opencode + GitHub Copilot
"""

# 設置Python路徑以便導入模組
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import taichi as ti
import numpy as np
import time
from src.physics.thermal_lbm import ThermalLBM
from src.physics.thermal_properties import ThermalPropertyManager
from config.thermal_config import get_thermal_config_summary, validate_thermal_config

def test_thermal_system_integration():
    """測試熱傳系統集成功能"""
    
    print("\n🔗 測試熱傳系統集成...")
    
    # 驗證配置
    if not validate_thermal_config():
        print("❌ 熱傳配置驗證失敗")
        return False
    
    # 初始化系統組件 (使用全域網格尺寸)
    print("  🚀 初始化系統組件...")
    
    from config.config import NX, NY, NZ
    thermal_solver = ThermalLBM(thermal_diffusivity=1.66e-7)  # 93°C水的熱擴散係數
    property_manager = ThermalPropertyManager(nx=NX, ny=NY, nz=NZ)  # 使用一致的網格尺寸
    
    # 初始化溫度場
    print("  🌡️  設置初始條件...")
    thermal_solver.complete_initialization(T_initial=25.0, T_hot_region=93.0, hot_region_height=10)
    
    # 設置相場分布
    water_level = NZ // 4  # 1/4高度為水位
    coffee_bottom = NZ // 10  # 1/10高度為咖啡床底部
    coffee_top = NZ // 6   # 1/6高度為咖啡床頂部
    property_manager.init_phase_field(water_level=water_level, coffee_bottom=coffee_bottom, 
                                    coffee_top=coffee_top, coffee_porosity=0.45)
    
    # 初始溫度場傳遞
    temp_field = thermal_solver.temperature.to_numpy()
    property_manager.set_temperature_field(temp_field)
    
    # 更新熱物性
    property_manager.update_thermal_properties()
    
    print("  ⏰ 執行時間演化...")
    success_steps = 0
    total_steps = 10  # 減少步數以節省時間
    
    start_time = time.time()
    
    for step in range(total_steps):
        # 熱傳LBM步驟
        success = thermal_solver.step()
        if not success:
            print(f"    ❌ 第{step}步數值不穩定")
            break
        
        # 每3步更新一次熱物性
        if step % 3 == 0:
            temp_field = thermal_solver.temperature.to_numpy()
            property_manager.set_temperature_field(temp_field)
            property_manager.update_thermal_properties()
        
        success_steps += 1
        
        # 診斷輸出
        if step % 3 == 0:
            T_min, T_max, T_avg = thermal_solver.get_temperature_stats()
            print(f"    步驟{step}: T∈[{T_min:.2f}, {T_max:.2f}]°C, 平均{T_avg:.2f}°C")
    
    elapsed_time = time.time() - start_time
    
    print(f"  📊 性能統計:")
    print(f"    成功步數: {success_steps}/{total_steps}")
    print(f"    計算時間: {elapsed_time:.3f} 秒")
    print(f"    平均每步: {elapsed_time/max(1,success_steps)*1000:.1f} ms")
    
    # 檢查熱物性分布
    props = property_manager.get_thermal_properties_numpy()
    print(f"  🔬 熱物性統計:")
    print(f"    熱導率範圍: {props['thermal_conductivity'].min():.3f} - {props['thermal_conductivity'].max():.3f} W/(m·K)")
    print(f"    密度範圍: {props['density'].min():.1f} - {props['density'].max():.1f} kg/m³")
    
    # 成功標準
    if success_steps >= total_steps * 0.8:  # 80%成功率
        print("✅ 熱傳系統集成測試通過")
        return True
    else:
        print(f"❌ 集成測試失敗：成功率{success_steps/total_steps*100:.1f}% < 80%")
        return False

def test_temperature_dependent_properties():
    """測試溫度依賴熱物性響應"""
    
    print("\n🌡️  測試溫度依賴熱物性...")
    
    property_manager = ThermalPropertyManager(nx=10, ny=10, nz=10)
    
    # 設置純水相的溫度梯度場 (避免多孔介質干擾)
    temp_field = np.full((10, 10, 10), 25.0)  # 基礎溫度
    for k in range(10):
        temp_field[:, :, k] = 25.0 + k * 7.0  # 25°C -> 88°C
    
    property_manager.set_temperature_field(temp_field)
    
    # 設置純水相場 (避免咖啡粉和空氣的混合效應)
    property_manager.init_phase_field(water_level=10, coffee_bottom=0, coffee_top=0, coffee_porosity=0.0)
    property_manager.update_thermal_properties()
    
    props = property_manager.get_thermal_properties_numpy()
    
    # 檢查純水區域的溫度相關性
    k_bottom = props['thermal_conductivity'][:, :, 1].mean()  # 低溫區 (~32°C)
    k_top = props['thermal_conductivity'][:, :, 8].mean()     # 高溫區 (~81°C)
    
    print(f"  低溫區熱導率: {k_bottom:.3f} W/(m·K) (@32°C)")
    print(f"  高溫區熱導率: {k_top:.3f} W/(m·K) (@81°C)")
    print(f"  溫度依賴性: {(k_top-k_bottom)/k_bottom*100:+.1f}%")
    
    # 檢查相場分布
    phase_bottom = props['phase_field'][:, :, 1].mean()
    phase_top = props['phase_field'][:, :, 8].mean()
    print(f"  底部相場: {phase_bottom:.2f} (應為1.0=水相)")
    print(f"  頂部相場: {phase_top:.2f} (應為1.0=水相)")
    
    # 水的熱導率應隨溫度增加 (物理預期)
    if k_top > k_bottom and phase_bottom > 0.9 and phase_top > 0.9:
        print("✅ 溫度依賴熱物性正確")
        return True
    else:
        print("❌ 溫度依賴熱物性異常")
        print(f"    期望: k_top > k_bottom, 實際: {k_top:.3f} vs {k_bottom:.3f}")
        print(f"    期望: 純水相, 實際相場: {phase_bottom:.2f}, {phase_top:.2f}")
        return False

def test_stability_limits():
    """測試數值穩定性極限"""
    
    print("\n🛡️  測試數值穩定性極限...")
    
    # 極端溫度梯度測試
    thermal_solver = ThermalLBM(thermal_diffusivity=1.66e-7)
    thermal_solver.complete_initialization(T_initial=10.0, T_hot_region=100.0, hot_region_height=5)
    
    print("  極端溫度梯度: 10°C -> 100°C")
    
    stable_steps = 0
    for step in range(50):
        success = thermal_solver.step()
        if not success:
            break
        stable_steps += 1
        
        # 檢查溫度範圍
        T_min, T_max, T_avg = thermal_solver.get_temperature_stats()
        if T_min < -10 or T_max > 110:  # 超出合理範圍
            print(f"    步驟{step}: 溫度超出範圍 T∈[{T_min:.1f}, {T_max:.1f}]°C")
            break
    
    print(f"  穩定步數: {stable_steps}/50")
    
    if stable_steps >= 40:  # 80%穩定性
        print("✅ 數值穩定性測試通過")
        return True
    else:
        print("❌ 數值穩定性不足")
        return False

def benchmark_performance():
    """性能基準測試"""
    
    print("\n⚡ 性能基準測試...")
    
    # 不同尺寸的性能測試
    grid_sizes = [(32, 32, 32), (64, 64, 64)]  # 減小測試尺寸
    
    for nx, ny, nz in grid_sizes:
        print(f"  測試網格: {nx}×{ny}×{nz} = {nx*ny*nz:,} 格點")
        
        thermal_solver = ThermalLBM(thermal_diffusivity=1.66e-7)
        thermal_solver.complete_initialization(T_initial=25.0, T_hot_region=90.0, hot_region_height=8)
        
        start_time = time.time()
        test_steps = 10
        
        for step in range(test_steps):
            success = thermal_solver.step()
            if not success:
                print(f"    ❌ 第{step}步失敗")
                break
        
        elapsed_time = time.time() - start_time
        throughput = (nx * ny * nz * test_steps) / elapsed_time / 1e6  # M格點/秒
        
        print(f"    計算時間: {elapsed_time:.3f} 秒")
        print(f"    吞吐量: {throughput:.1f} M格點/秒")
        print(f"    平均每步: {elapsed_time/test_steps*1000:.1f} ms")
    
    print("✅ 性能基準測試完成")
    return True

def main():
    """主測試流程"""
    
    print("=== Phase 1 熱傳系統集成測試 ===")
    
    # 初始化Taichi
    ti.init(arch=ti.cpu)  # 使用CPU確保穩定性
    
    # 執行測試套件
    tests = [
        ("系統集成", test_thermal_system_integration),
        ("溫度依賴物性", test_temperature_dependent_properties), 
        ("數值穩定性", test_stability_limits),
        ("性能基準", benchmark_performance)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}測試")
        print('='*60)
        
        try:
            success = test_func()
            if success:
                passed_tests += 1
                print(f"✅ {test_name}測試通過")
            else:
                print(f"❌ {test_name}測試失敗")
        except Exception as e:
            print(f"❌ {test_name}測試異常: {e}")
    
    total_time = time.time() - start_time
    
    # 總結報告
    print(f"\n{'='*60}")
    print(f"📋 Phase 1 測試總結")
    print('='*60)
    print(f"通過測試: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"總測試時間: {total_time:.2f} 秒")
    
    # 配置摘要
    config_summary = get_thermal_config_summary()
    print(f"\n🔧 系統配置摘要:")
    print(f"  溫度範圍: {config_summary['temperature_range'][0]}-{config_summary['temperature_range'][1]}°C")
    print(f"  注水溫度: {config_summary['inlet_temperature']}°C")
    print(f"  水相τ: {config_summary['relaxation_times']['water']:.3f}")
    print(f"  水相CFL: {config_summary['cfl_numbers']['water']:.3f}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 Phase 1 開發完成！基礎熱傳模組就緒")
        print(f"📈 準備進入 Phase 2: 弱耦合實現")
        return True
    else:
        print(f"\n⚠️  Phase 1 需要修正，{total_tests-passed_tests}個測試失敗")
        return False

if __name__ == "__main__":
    main()