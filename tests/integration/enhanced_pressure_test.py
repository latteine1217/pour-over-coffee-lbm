# enhanced_pressure_test.py
"""
中等增強版壓力梯度測試
驗證50-70%參數提升後的效果與穩定性
"""

# 設置Python路徑以便導入模組
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
from config.init import initialize_taichi_once
initialize_taichi_once()

import config
from main import CoffeeSimulation, setup_pressure_drive

def test_enhanced_mode(mode, steps=50):
    """測試增強版壓力驅動模式"""
    print(f"\n{'='*70}")
    print(f"🚀 測試增強版模式: {mode.upper()}")
    print(f"{'='*70}")
    
    # 創建模擬系統
    print("🔧 初始化增強版模擬系統...")
    sim = CoffeeSimulation()
    
    # 設置壓力驅動
    setup_pressure_drive(sim, mode)
    
    # 顯示增強參數
    if hasattr(sim, 'pressure_drive') and sim.pressure_drive:
        enhanced_diag = sim.pressure_drive.get_enhanced_diagnostics()
        print(f"📊 增強參數:")
        print(f"   ├─ 增強級別: {enhanced_diag['enhancement_level']}")
        print(f"   ├─ 壓力比範圍: {enhanced_diag['pressure_ratio_range']}")
        print(f"   ├─ 最大體力: {enhanced_diag['max_force']:.3f} lu/ts²")
        print(f"   └─ 預期提升: 2-3倍效果")
    
    # 記錄初始狀態
    print(f"\n🔍 運行 {steps} 步增強測試...")
    
    results = []
    for step in range(1, steps + 1):
        # 執行模擬步驟
        success = sim.step()
        if not success:
            print(f"❌ 步驟 {step}: 模擬失敗")
            break
        
        # 增強版診斷
        if hasattr(sim, 'pressure_drive') and sim.pressure_drive:
            enhanced_diag = sim.pressure_drive.get_enhanced_diagnostics()
            
            # 記錄關鍵數據
            results.append({
                'step': step,
                'max_velocity': enhanced_diag['max_velocity'],
                'stability_code': enhanced_diag['stability_code'],
                'density_range': enhanced_diag['density_range']
            })
            
            # 即時監控顯示
            if step % 10 == 0 or step <= 5 or enhanced_diag['stability_code'] > 0:
                status = enhanced_diag['stability_status']
                vel = enhanced_diag['max_velocity']
                rho_min, rho_max = enhanced_diag['density_range']
                
                print(f"   步驟 {step:2d}: {status} | 速度={vel:.6f} | 密度=[{rho_min:.3f}, {rho_max:.3f}]")
                
                # 警告處理
                if enhanced_diag['stability_code'] >= 2:
                    print(f"      ⚠️  {enhanced_diag['stability_status']}")
                    if enhanced_diag['stability_code'] == 3:
                        print(f"      💀 嚴重不穩定，建議停止測試")
                        break
    
    # 最終分析
    if results:
        final_result = results[-1]
        max_velocities = [r['max_velocity'] for r in results]
        stability_issues = sum(1 for r in results if r['stability_code'] > 0)
        
        print(f"\n📊 增強版測試結果:")
        print(f"   ├─ 最終速度: {final_result['max_velocity']:.6f} lu/ts")
        print(f"   ├─ 速度峰值: {max(max_velocities):.6f} lu/ts")
        print(f"   ├─ 穩定性問題: {stability_issues}/{len(results)} 步")
        print(f"   └─ 最終狀態: {['✅ 穩定', '⚠️ 速度警告', '⚠️ 密度警告', '❌ 不穩定'][final_result['stability_code']]}")
        
        # 效果評估
        if final_result['stability_code'] <= 1:
            if final_result['max_velocity'] > 0.05:
                grade = "A+ (優異增強)"
            elif final_result['max_velocity'] > 0.03:
                grade = "A (良好增強)"
            elif final_result['max_velocity'] > 0.015:
                grade = "B (輕微增強)"
            else:
                grade = "C (效果不明顯)"
        else:
            grade = "F (不穩定)"
        
        print(f"   🎯 增強評級: {grade}")
        
        return {
            'mode': mode,
            'final_velocity': final_result['max_velocity'],
            'peak_velocity': max(max_velocities),
            'stability_issues': stability_issues,
            'total_steps': len(results),
            'grade': grade,
            'stable': final_result['stability_code'] <= 1
        }
    
    return None

def compare_enhancement_effects():
    """對比增強前後的效果"""
    print(f"\n{'='*80}")
    print("📈 增強效果對比分析")
    print(f"{'='*80}")
    
    # 理論對比 (基於之前5%重力的測試)
    print("📊 理論增強效果預測:")
    print("   ├─ 原版 Force: ~0.020000 lu/ts")
    print("   ├─ 增強 Force: ~0.040-0.060 lu/ts (預期2-3倍)")
    print("   ├─ 原版 Mixed: ~0.010005 lu/ts") 
    print("   └─ 增強 Mixed: ~0.020-0.030 lu/ts (預期2-3倍)")
    
    # 測試各模式
    modes = ["none", "force", "mixed"]
    enhanced_results = []
    
    for mode in modes:
        try:
            result = test_enhanced_mode(mode, steps=30)  # 較短測試避免過長
            if result:
                enhanced_results.append(result)
        except Exception as e:
            print(f"❌ {mode} 模式測試失敗: {e}")
    
    # 結果對比表
    if enhanced_results:
        print(f"\n📋 增強版測試結果總表:")
        print(f"{'模式':<8} | {'最終速度':<12} | {'峰值速度':<12} | {'穩定性':<8} | {'評級'}")
        print("-" * 75)
        
        for result in enhanced_results:
            mode = result['mode']
            final_vel = f"{result['final_velocity']:.6f}"
            peak_vel = f"{result['peak_velocity']:.6f}"
            stable = "✅" if result['stable'] else "❌"
            grade = result['grade'].split()[0]  # 取評級字母
            
            print(f"{mode:<8} | {final_vel:<12} | {peak_vel:<12} | {stable:<8} | {grade}")
        
        # 推薦最佳模式
        stable_results = [r for r in enhanced_results if r['stable']]
        if stable_results:
            best = max(stable_results, key=lambda x: x['final_velocity'])
            print(f"\n🏆 推薦增強模式: {best['mode'].upper()}")
            print(f"   └─ 最佳效果: {best['final_velocity']:.6f} lu/ts")
            print(f"   └─ 評級: {best['grade']}")

def main():
    """主測試函數"""
    print("🚀 中等增強版壓力梯度驅動測試")
    print(f"   ├─ 壓力比例增強: 30%")
    print(f"   ├─ 體力場增強: 70%") 
    print(f"   ├─ 調整速率增強: 50%")
    print(f"   └─ 預期效果: 2-3倍提升")
    
    compare_enhancement_effects()
    
    print(f"\n✅ 增強版測試完成")
    print(f"🎯 如效果良好且穩定，可考慮進一步優化")

if __name__ == "__main__":
    main()