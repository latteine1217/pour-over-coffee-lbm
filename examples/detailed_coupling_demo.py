# detailed_coupling_demo.py - 詳細耦合演示
"""
Phase 2 弱耦合系統詳細演示
展示系統運行狀態、溫度演化、性能統計等
"""

import taichi as ti
import numpy as np
import time

# 初始化Taichi
ti.init(arch=ti.cpu)
print("🔧 Taichi CPU模式初始化")

from src.core.thermal_fluid_coupled import ThermalFluidCoupledSolver, CouplingConfig
import config.config

def detailed_coupling_demonstration():
    """詳細耦合演示"""
    
    print("=" * 70)
    print("🧪 Phase 2 熱流弱耦合系統詳細演示")
    print("=" * 70)
    
    # 1. 系統配置展示
    print("\n📋 1. 系統配置")
    print(f"   網格尺寸: {config.NX}×{config.NY}×{config.NZ} = {config.NX*config.NY*config.NZ:,}格點")
    print(f"   格子解析度: {config.DX*1000:.3f} mm/格點")
    print(f"   時間步長: {config.DT*1000:.1f} ms/步")
    
    # 創建耦合配置
    coupling_config = CouplingConfig(
        coupling_frequency=1,      # 每步耦合
        velocity_smoothing=False,  # 不平滑
        thermal_subcycles=1,       # 單一熱傳子循環
        enable_diagnostics=True,   # 啟用診斷
        max_coupling_error=500.0   # 溫度誤差限制
    )
    
    print(f"   耦合頻率: 每{coupling_config.coupling_frequency}步")
    print(f"   熱傳子循環: {coupling_config.thermal_subcycles}次/步")
    print(f"   診斷監控: {'啟用' if coupling_config.enable_diagnostics else '禁用'}")
    
    # 2. 系統初始化
    print("\n🚀 2. 系統初始化")
    start_time = time.time()
    
    coupled_solver = ThermalFluidCoupledSolver(
        coupling_config=coupling_config,
        thermal_diffusivity=1.6e-7  # 水的熱擴散係數
    )
    
    # 初始條件設置
    fluid_initial_conditions = {}  # 使用默認流體初始化
    
    thermal_initial_conditions = {
        'T_initial': 25.0,      # 環境溫度 25°C
        'T_hot_region': 85.0,   # 熱水溫度 85°C
        'hot_region_height': 10  # 熱區域高度 10格點
    }
    
    # 基礎熱源場 (模擬持續熱水注入)
    base_heat_source = np.zeros((config.NX, config.NY, config.NZ), dtype=np.float32)
    # 在頂部中心區域設置熱源
    center_x, center_y = config.NX//2, config.NY//2
    for i in range(center_x-5, center_x+5):
        for j in range(center_y-5, center_y+5):
            for k in range(config.NZ-15, config.NZ):  # 頂部15層
                if 0 <= i < config.NX and 0 <= j < config.NY:
                    base_heat_source[i, j, k] = 200.0  # W/m³
    
    coupled_solver.initialize_system(
        fluid_initial_conditions=fluid_initial_conditions,
        thermal_initial_conditions=thermal_initial_conditions,
        base_heat_source=base_heat_source
    )
    
    init_time = time.time() - start_time
    print(f"   初始化耗時: {init_time:.3f}秒")
    
    # 3. 初始狀態展示
    print("\n📊 3. 初始狀態")
    
    # 獲取初始溫度統計 (直接從熱傳求解器)
    T_min_init, T_max_init, T_avg_init = coupled_solver.thermal_solver.get_temperature_stats()
    thermal_diffusivity = coupled_solver.thermal_solver.get_effective_thermal_diffusivity()
    
    print(f"   初始溫度範圍: {T_min_init:.1f} - {T_max_init:.1f}°C")
    print(f"   初始平均溫度: {T_avg_init:.1f}°C")
    print(f"   熱擴散係數: {thermal_diffusivity:.2e} m²/s")
    print(f"   對流耦合狀態: {'啟用' if coupled_solver.thermal_solver.enable_convection else '禁用'}")
    print(f"   熱源功率: {np.sum(base_heat_source):.1f} W/m³ (總量)")
    
    # 4. 多步演化模擬
    print("\n🔄 4. 多步演化模擬")
    simulation_steps = 8
    results_history = []
    
    print("步驟  |  時間(s) | T_min(°C) | T_avg(°C) | T_max(°C) | 狀態")
    print("-" * 60)
    
    total_start = time.time()
    
    for step in range(simulation_steps):
        step_start = time.time()
        
        # 執行一步
        success = coupled_solver.step()
        
        step_time = time.time() - step_start
        
        if success:
            # 獲取當前診斷
            diagnostics = coupled_solver.get_coupling_diagnostics()
            thermal_stats = diagnostics['thermal_stats']
            
            results_history.append({
                'step': step + 1,
                'time': step_time,
                'T_min': thermal_stats['T_min'],
                'T_avg': thermal_stats['T_avg'],
                'T_max': thermal_stats['T_max'],
                'success': True
            })
            
            print(f"{step+1:3d}   | {step_time:7.3f}  | {thermal_stats['T_min']:8.1f}  | {thermal_stats['T_avg']:8.1f}  | {thermal_stats['T_max']:8.1f}  | ✅")
            
        else:
            results_history.append({
                'step': step + 1,
                'time': step_time,
                'success': False
            })
            print(f"{step+1:3d}   | {step_time:7.3f}  |    -     |    -     |    -     | ❌")
            break
    
    total_simulation_time = time.time() - total_start
    
    # 5. 性能統計
    print(f"\n⚡ 5. 性能統計")
    final_diagnostics = coupled_solver.get_coupling_diagnostics()
    performance = final_diagnostics['performance']
    
    successful_steps = sum(1 for r in results_history if r['success'])
    avg_step_time = np.mean([r['time'] for r in results_history if r['success']])
    
    print(f"   成功步數: {successful_steps}/{simulation_steps}")
    print(f"   平均步時: {avg_step_time:.3f}秒/步")
    print(f"   總模擬時間: {total_simulation_time:.3f}秒")
    print(f"   模擬效率: {successful_steps/total_simulation_time:.2f}步/秒")
    print(f"   流體計算佔比: {performance['fluid_fraction']:.1%}")
    print(f"   熱傳計算佔比: {performance['thermal_fraction']:.1%}")
    print(f"   耦合計算佔比: {performance['coupling_fraction']:.1%}")
    
    # 6. 溫度演化分析
    if successful_steps > 1:
        print(f"\n🌡️  6. 溫度演化分析")
        
        initial_result = results_history[0]
        final_result = results_history[successful_steps-1]
        
        temp_change_min = final_result['T_min'] - initial_result['T_min']
        temp_change_avg = final_result['T_avg'] - initial_result['T_avg']
        temp_change_max = final_result['T_max'] - initial_result['T_max']
        
        print(f"   最低溫度變化: {initial_result['T_min']:.1f} → {final_result['T_min']:.1f}°C ({temp_change_min:+.1f}°C)")
        print(f"   平均溫度變化: {initial_result['T_avg']:.1f} → {final_result['T_avg']:.1f}°C ({temp_change_avg:+.1f}°C)")
        print(f"   最高溫度變化: {initial_result['T_max']:.1f} → {final_result['T_max']:.1f}°C ({temp_change_max:+.1f}°C)")
        
        # 溫度梯度
        temp_gradient = final_result['T_max'] - final_result['T_min']
        print(f"   最終溫度梯度: {temp_gradient:.1f}°C")
        
        # 熱傳效率估算
        if temp_change_avg > 0:
            heating_rate = temp_change_avg / (successful_steps * config.DT)
            print(f"   平均升溫速率: {heating_rate:.2f}°C/s")
    
    # 7. 系統狀態總結
    print(f"\n📋 7. 系統狀態總結")
    
    if successful_steps >= simulation_steps * 0.8:
        print("   ✅ 系統穩定性: 優秀")
        stability_status = "優秀"
    elif successful_steps >= simulation_steps * 0.6:
        print("   ⚠️  系統穩定性: 良好")
        stability_status = "良好"
    else:
        print("   ❌ 系統穩定性: 需要改進")
        stability_status = "需要改進"
    
    if avg_step_time < 1.0:
        print("   ✅ 計算性能: 高效")
        performance_status = "高效"
    elif avg_step_time < 3.0:
        print("   ⚠️  計算性能: 中等")
        performance_status = "中等"
    else:
        print("   ❌ 計算性能: 需要優化")
        performance_status = "需要優化"
    
    # 物理合理性檢查
    if successful_steps > 0:
        final_diagnostics = coupled_solver.get_coupling_diagnostics()
        final_thermal = final_diagnostics['thermal_stats']
        
        if 0 <= final_thermal['T_min'] <= 120 and 0 <= final_thermal['T_max'] <= 120:
            print("   ✅ 物理合理性: 溫度範圍正常")
            physics_status = "正常"
        else:
            print("   ❌ 物理合理性: 溫度範圍異常")
            physics_status = "異常"
    else:
        physics_status = "無法評估"
    
    print("\n" + "=" * 70)
    print("🎯 Phase 2 弱耦合系統評估結果:")
    print(f"   📊 數值穩定性: {stability_status}")
    print(f"   ⚡ 計算性能: {performance_status}")
    print(f"   🔬 物理合理性: {physics_status}")
    
    if stability_status == "優秀" and physics_status == "正常":
        print("   🎉 Phase 2 開發成功！可以進行Phase 3開發")
        overall_success = True
    else:
        print("   🔧 需要進一步調試和優化")
        overall_success = False
    
    print("=" * 70)
    
    return overall_success, results_history

if __name__ == "__main__":
    success, history = detailed_coupling_demonstration()
    
    if success:
        print("\n🚀 系統準備就緒，可以開始更複雜的熱流耦合應用！")
    else:
        print("\n🔍 建議檢查系統配置和參數設置")