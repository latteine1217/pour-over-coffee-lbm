# test_lbm_diagnostics.py
"""
LBM診斷系統測試腳本
測試診斷功能的正確性和效率影響
"""

import time
import numpy as np
import sys
import os

# 導入必要的模組
sys.path.append('.')
from main import CoffeeSimulation
import config

def test_diagnostics_functionality():
    """測試診斷系統功能完整性"""
    print("🧪 測試1: 診斷系統功能完整性")
    print("="*60)
    
    # 創建模擬實例
    sim = CoffeeSimulation()
    
    # 檢查診斷系統是否正確初始化
    assert hasattr(sim, 'diagnostics'), "診斷系統未正確初始化"
    print("✅ 診斷系統初始化成功")
    
    # 運行幾步模擬以生成診斷數據
    print("🔄 運行10步模擬以生成測試數據...")
    for i in range(10):
        success = sim.step()
        if not success:
            print(f"❌ 模擬在第{i+1}步失敗")
            return False
        
        # 檢查診斷數據是否正確生成
        current_diagnostics = sim.diagnostics.get_current_diagnostics()
        if i >= 5 and current_diagnostics:  # 前幾步可能沒有診斷數據
            print(f"   步驟{i+1}: 診斷數據包含 {len(current_diagnostics)} 個項目")
    
    # 檢查診斷摘要
    summary = sim.diagnostics.get_summary_report()
    print("\n📊 診斷摘要:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_val in value.items():
                print(f"     └─ {sub_key}: {sub_val}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ 功能測試完成")
    return True

def test_diagnostics_performance():
    """測試診斷系統的效率影響"""
    print("\n🚀 測試2: 診斷系統效率影響")
    print("="*60)
    
    # 測試參數
    test_steps = 50
    
    # 第一次測試：含診斷系統
    print("🔬 測試含診斷系統的性能...")
    sim_with_diagnostics = CoffeeSimulation()
    
    start_time = time.time()
    diagnostic_success_count = 0
    
    for i in range(test_steps):
        step_start = time.time()
        success = sim_with_diagnostics.step()
        step_time = time.time() - step_start
        
        if not success:
            print(f"   ❌ 步驟{i+1}失敗")
            break
            
        # 檢查診斷是否成功執行
        diagnostics = sim_with_diagnostics.diagnostics.get_current_diagnostics()
        if diagnostics:
            diagnostic_success_count += 1
        
        if i % 10 == 0:
            print(f"   步驟{i+1}: {step_time*1000:.2f}ms")
    
    with_diagnostics_time = time.time() - start_time
    
    # 獲取診斷性能統計
    perf_stats = sim_with_diagnostics.diagnostics.get_performance_stats()
    
    print(f"\n📊 含診斷系統結果:")
    print(f"   總時間: {with_diagnostics_time:.3f}秒")
    print(f"   平均步長時間: {with_diagnostics_time/test_steps*1000:.2f}ms")
    print(f"   診斷成功率: {diagnostic_success_count/test_steps*100:.1f}%")
    print(f"   診斷計算統計:")
    for calc_type in ['light', 'medium', 'heavy']:
        count = perf_stats.get(f'{calc_type}_calc_count', 0)
        total_time = perf_stats.get(f'{calc_type}_total_time', 0)
        avg_time = perf_stats.get(f'{calc_type}_avg_time', 0)
        print(f"     {calc_type}: {count}次, 總時間{total_time*1000:.2f}ms, 平均{avg_time*1000:.2f}ms")
    
    # 計算診斷開銷
    total_diagnostic_time = sum([perf_stats.get(f'{t}_total_time', 0) for t in ['light', 'medium', 'heavy']])
    diagnostic_overhead = (total_diagnostic_time / with_diagnostics_time) * 100
    
    print(f"\n📈 效率影響分析:")
    print(f"   診斷總開銷: {total_diagnostic_time*1000:.2f}ms")
    print(f"   診斷開銷比例: {diagnostic_overhead:.2f}%")
    
    if diagnostic_overhead < 5.0:
        print(f"   ✅ 效率影響在可接受範圍內 (<5%)")
    elif diagnostic_overhead < 10.0:
        print(f"   🟡 效率影響中等 (5-10%)")
    else:
        print(f"   ⚠️  效率影響較高 (>10%)")
    
    return diagnostic_overhead

def test_diagnostics_visualization():
    """測試診斷視覺化功能"""
    print("\n📊 測試3: 診斷視覺化功能")
    print("="*60)
    
    # 創建模擬實例並運行足夠步數以生成有意義的數據
    sim = CoffeeSimulation()
    
    print("🔄 運行50步模擬以生成視覺化數據...")
    for i in range(50):
        success = sim.step()
        if not success:
            print(f"❌ 模擬在第{i+1}步失敗")
            return False
        
        if (i+1) % 10 == 0:
            print(f"   進度: {i+1}/50 步")
    
    # 測試LBM監控圖表生成
    print("\n🎨 測試LBM監控圖表生成...")
    try:
        simulation_time = 50 * config.DT
        lbm_chart = sim.enhanced_viz.save_lbm_monitoring_chart(simulation_time, 50)
        
        if lbm_chart and os.path.exists(lbm_chart):
            print(f"   ✅ LBM監控圖表生成成功: {lbm_chart}")
            file_size = os.path.getsize(lbm_chart) / 1024  # KB
            print(f"   📁 文件大小: {file_size:.1f} KB")
        else:
            print("   ❌ LBM監控圖表生成失敗")
            return False
    except Exception as e:
        print(f"   ❌ 視覺化測試異常: {e}")
        return False
    
    # 測試完整科研報告生成
    print("\n🔬 測試完整科研報告生成...")
    try:
        report_files = sim.enhanced_viz.generate_research_report(simulation_time, 50)
        
        if report_files:
            print(f"   ✅ 科研報告生成成功: {len(report_files)} 個文件")
            for i, file in enumerate(report_files[:3], 1):  # 只顯示前3個
                if os.path.exists(file):
                    size = os.path.getsize(file) / 1024
                    print(f"   📄 {i}. {file} ({size:.1f} KB)")
            
            if len(report_files) > 3:
                print(f"   📄 ... 及其他 {len(report_files)-3} 個文件")
        else:
            print("   ⚠️  科研報告生成為空")
    except Exception as e:
        print(f"   ❌ 科研報告測試異常: {e}")
        return False
    
    print("\n✅ 視覺化測試完成")
    return True

def main():
    """主測試函數"""
    print("🧪 LBM診斷系統完整測試")
    print("="*60)
    print("此測試將驗證:")
    print("1. 診斷系統功能完整性")
    print("2. 效率影響評估")  
    print("3. 視覺化生成能力")
    print("="*60)
    
    all_tests_passed = True
    
    # 測試1: 功能完整性
    try:
        if not test_diagnostics_functionality():
            all_tests_passed = False
    except Exception as e:
        print(f"❌ 功能測試異常: {e}")
        all_tests_passed = False
    
    # 測試2: 效率影響
    try:
        overhead = test_diagnostics_performance()
        if overhead > 15.0:  # 如果開銷超過15%則標記為問題
            print(f"⚠️  診斷開銷過高: {overhead:.2f}%")
            all_tests_passed = False
    except Exception as e:
        print(f"❌ 效率測試異常: {e}")
        all_tests_passed = False
    
    # 測試3: 視覺化功能
    try:
        if not test_diagnostics_visualization():
            all_tests_passed = False
    except Exception as e:
        print(f"❌ 視覺化測試異常: {e}")
        all_tests_passed = False
    
    # 總結
    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 所有測試通過！LBM診斷系統運行正常")
        print("✅ 系統已準備好進行科研級CFD分析")
    else:
        print("⚠️  部分測試失敗，請檢查系統配置")
        print("🔧 建議檢查依賴項和配置參數")
    
    print("\n📝 測試完成")
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)