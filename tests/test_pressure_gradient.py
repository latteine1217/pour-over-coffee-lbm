# test_pressure_gradient.py
"""
壓力梯度驅動系統測試腳本
快速驗證各種驅動模式的數值穩定性
"""

# 設置Python路徑以便導入模組
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

import sys
import numpy as np
import time

# 引入模組
from config.init import initialize_taichi_once
initialize_taichi_once()

import config.config as config

def test_pressure_mode(mode, steps=50):
    """測試特定壓力驅動模式"""
    print(f"\n{'='*60}")
    print(f"🧪 測試模式: {mode.upper()}")
    print(f"{'='*60}")
    
    # 引入完整系統而不是單獨的LBM
    from main import CoffeeSimulation
    from main import setup_pressure_drive
    
    # 初始化完整模擬系統
    print("🔧 初始化完整模擬系統...")
    sim = CoffeeSimulation()
    
    print("🔧 設置壓力梯度驅動模式...")
    setup_pressure_drive(sim, mode)
    
    # 記錄初始狀態
    initial_stats = get_field_statistics_from_sim(sim)
    print(f"📊 初始狀態:")
    print(f"   ├─ 平均密度: {initial_stats['avg_rho']:.6f}")
    print(f"   ├─ 密度範圍: [{initial_stats['min_rho']:.6f}, {initial_stats['max_rho']:.6f}]")
    print(f"   └─ 最大速度: {initial_stats['max_velocity']:.6f}")
    
    # 運行測試
    print(f"\n🚀 開始 {steps} 步測試...")
    stable = True
    
    for step in range(1, steps + 1):
        # 使用模擬系統的step方法
        success = sim.step()
        if not success:
            print(f"❌ 步驟 {step}: 模擬系統報告失敗")
            stable = False
            break
        
        # 檢查數值狀態
        stats = get_field_statistics_from_sim(sim)
        
        # 顯示進度
        if step % 10 == 0 or step <= 5:
            pressure_stats = {}
            if hasattr(sim, 'pressure_drive') and sim.pressure_drive:
                try:
                    pressure_stats = sim.pressure_drive.get_statistics()
                except Exception as e:
                    print(f"      壓力統計失敗: {e}")
                    pressure_stats = {}
            
            print(f"   步驟 {step:2d}: 速度={stats['max_velocity']:.6f}, "
                  f"密度=[{stats['min_rho']:.3f}, {stats['max_rho']:.3f}]", end="")
            
            if pressure_stats and 'pressure_drop' in pressure_stats:
                print(f", 壓差={pressure_stats['pressure_drop']:.6f}")
            else:
                print(f", 壓差=N/A (無壓力驅動)")
        
        # 穩定性檢查
        if stats['max_velocity'] > 0.15:
            print(f"❌ 步驟 {step}: 速度過高 {stats['max_velocity']:.6f} > 0.15")
            stable = False
            break
        
        if np.isnan(stats['max_velocity']) or np.isinf(stats['max_velocity']):
            print(f"❌ 步驟 {step}: 數值發散 (NaN/Inf)")
            stable = False
            break
        
        if stats['max_rho'] > 5.0 or stats['min_rho'] < 0.001:
            print(f"❌ 步驟 {step}: 密度異常 [{stats['min_rho']:.3f}, {stats['max_rho']:.3f}]")
            stable = False
            break
    
    # 最終結果
    final_stats = get_field_statistics_from_sim(sim)
    final_pressure = {}
    if hasattr(sim, 'pressure_drive') and sim.pressure_drive:
        try:
            final_pressure = sim.pressure_drive.get_statistics()
            print(f"🔍 壓力系統狀態: {sim.pressure_drive.get_status()}")
        except Exception as e:
            print(f"⚠️  最終壓力統計失敗: {e}")
            final_pressure = {}
    
    print(f"\n📊 最終結果:")
    print(f"   ├─ 穩定性: {'✅ 穩定' if stable else '❌ 不穩定'}")
    print(f"   ├─ 最大速度: {final_stats['max_velocity']:.6f}")
    print(f"   ├─ 密度變化: {initial_stats['avg_rho']:.6f} → {final_stats['avg_rho']:.6f}")
    
    if final_pressure and 'pressure_drop' in final_pressure:
        print(f"   ├─ 壓力差: {final_pressure['pressure_drop']:.6f}")
        print(f"   └─ 壓力比: {final_pressure.get('pressure_ratio', 0):.3f}")
    else:
        print(f"   └─ 壓力系統: 無效或未啟動")
    
    # 評級
    if stable:
        if final_stats['max_velocity'] < 0.05:
            grade = "A (優秀)"
        elif final_stats['max_velocity'] < 0.1:
            grade = "B (良好)"
        else:
            grade = "C (可接受)"
    else:
        grade = "F (失敗)"
    
    print(f"\n🎯 模式評級: {grade}")
    
    return {
        'mode': mode,
        'stable': stable,
        'final_velocity': final_stats['max_velocity'],
        'pressure_drop': final_pressure.get('pressure_drop', 0),
        'grade': grade[0]
    }

def get_field_statistics_from_sim(sim):
    """從模擬系統獲取場的統計數據 - 修復版本"""
    try:
        # 強制同步GPU數據到CPU
        rho_data = sim.lbm.rho.to_numpy()
        u_data = sim.lbm.u.to_numpy()
        
        # 計算速度幅度
        u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
        
        # 安全的平均速度計算 - 避免空切片
        nonzero_velocities = u_magnitude[u_magnitude > 1e-10]
        avg_velocity = float(np.mean(nonzero_velocities)) if len(nonzero_velocities) > 0 else 0.0
        
        # 密度統計
        avg_rho = float(np.mean(rho_data))
        min_rho = float(np.min(rho_data))
        max_rho = float(np.max(rho_data))
        max_velocity = float(np.max(u_magnitude))
        
        # 調試信息
        print(f"      統計: 密度範圍=[{min_rho:.3f}, {max_rho:.3f}], 速度範圍=[0, {max_velocity:.6f}]")
        
        return {
            'avg_rho': avg_rho,
            'min_rho': min_rho,
            'max_rho': max_rho,
            'max_velocity': max_velocity,
            'avg_velocity': avg_velocity
        }
    except Exception as e:
        print(f"⚠️  統計計算失敗: {e}")
        return {
            'avg_rho': 1.0,
            'min_rho': 1.0,
            'max_rho': 1.0,
            'max_velocity': 0.0,
            'avg_velocity': 0.0
        }

def main():
    """主測試函數"""
    print("🧪 壓力梯度驅動系統 - 快速穩定性測試")
    print(f"   ├─ 網格大小: {config.NX}×{config.NY}×{config.NZ}")
    print(f"   ├─ 測試步數: 50")
    print(f"   └─ 穩定性閾值: 最大速度 < 0.15 lu/ts")
    
    # 測試所有模式
    modes = ["none", "density", "force", "mixed"]
    results = []
    
    start_time = time.time()
    
    for mode in modes:
        try:
            result = test_pressure_mode(mode, steps=50)
            results.append(result)
        except Exception as e:
            print(f"❌ 模式 {mode} 測試失敗: {e}")
            results.append({
                'mode': mode,
                'stable': False,
                'final_velocity': float('inf'),
                'pressure_drop': 0,
                'grade': 'F'
            })
    
    total_time = time.time() - start_time
    
    # 總結報告
    print(f"\n{'='*80}")
    print("📋 總結報告")
    print(f"{'='*80}")
    
    print(f"📊 測試結果對比:")
    print(f"{'模式':<8} | {'穩定性':<6} | {'最大速度':<12} | {'壓力差':<12} | {'評級'}")
    print("-" * 70)
    
    for result in results:
        stability = "✅" if result['stable'] else "❌"
        velocity = f"{result['final_velocity']:.6f}"
        pressure = f"{result['pressure_drop']:.6f}"
        grade = result['grade']
        
        print(f"{result['mode']:<8} | {stability:<6} | {velocity:<12} | {pressure:<12} | {grade}")
    
    # 推薦
    stable_modes = [r for r in results if r['stable']]
    if stable_modes:
        best_mode = min(stable_modes, key=lambda x: x['final_velocity'])
        print(f"\n🏆 推薦模式: {best_mode['mode'].upper()}")
        print(f"   └─ 最大速度: {best_mode['final_velocity']:.6f} lu/ts")
        print(f"   └─ 評級: {best_mode['grade']}")
    else:
        print(f"\n⚠️  所有模式均不穩定，建議調整參數")
    
    print(f"\n⏱️  總測試時間: {total_time:.2f} 秒")
    print(f"🎯 建議: 選擇穩定模式進行完整模擬")

if __name__ == "__main__":
    main()