#!/usr/bin/env python3
"""
測試增強視覺化功能 - 動態範圍調整和時序分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from src.visualization.enhanced_visualizer import EnhancedVisualizer
import config.config as config

class MockLBMSolver:
    """模擬LBM求解器用於測試"""
    def __init__(self):
        # 創建模擬數據
        nx, ny, nz = 50, 50, 50  # 小尺寸用於快速測試
        
        # 模擬密度場
        self.rho = MockField(np.ones((nx, ny, nz)) + 0.1 * np.random.random((nx, ny, nz)))
        
        # 模擬速度場 - 創建合理的流動模式
        u = np.zeros((nx, ny, nz, 3))
        
        # 在中心區域創建向下的流動
        center = nx // 2
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 距離中心的距離
                    r = np.sqrt((i - center)**2 + (j - center)**2)
                    if r < center * 0.5:  # 在中心區域
                        # 向下流動，強度隨距離衰減
                        u[i, j, k, 2] = -0.1 * (1 - r / (center * 0.5))  # uz
                        # 添加一些徑向速度
                        if r > 1e-6:
                            u[i, j, k, 0] = 0.02 * (i - center) / r  # ux
                            u[i, j, k, 1] = 0.02 * (j - center) / r  # uy
        
        self.u = MockField(u)

class MockField:
    """模擬Taichi場"""
    def __init__(self, data):
        self.data = data
    
    def to_numpy(self):
        return self.data

def test_dynamic_colorbar():
    """測試動態範圍調整colorbar功能"""
    print("🧪 測試動態範圍調整colorbar功能...")
    
    # 創建模擬求解器
    mock_lbm = MockLBMSolver()
    
    # 創建增強視覺化器
    viz = EnhancedVisualizer(mock_lbm)
    
    # 測試動態範圍計算
    test_data = np.array([1, 2, 3, 100, 4, 5, 6, 7, 8, 9, 200])  # 包含極值
    vmin, vmax = viz._calculate_dynamic_range(test_data, 10, 90)
    
    print(f"   原始數據範圍: {np.min(test_data)} ~ {np.max(test_data)}")
    print(f"   動態範圍調整: {vmin:.2f} ~ {vmax:.2f}")
    
    # 測試智能colorbar
    fig, ax = plt.subplots(figsize=(8, 6))
    data_2d = np.random.random((20, 20)) * 100 + np.random.random((20, 20)) * 1000  # 包含不同尺度的數據
    im = ax.imshow(data_2d, cmap='viridis')
    
    cbar = viz._create_smart_colorbar(ax, im, data_2d, "Test Data", "Units")
    
    plt.title("動態範圍調整colorbar測試")
    plt.tight_layout()
    plt.savefig("test_dynamic_colorbar.png", dpi=150)
    plt.close()
    
    print("   ✅ 動態colorbar測試完成，圖像已保存: test_dynamic_colorbar.png")

def test_time_series_analysis():
    """測試時序分析功能"""
    print("🧪 測試時序分析功能...")
    
    # 創建模擬求解器
    mock_lbm = MockLBMSolver()
    
    # 創建增強視覺化器
    viz = EnhancedVisualizer(mock_lbm)
    
    # 模擬多步時序數據收集
    print("   收集模擬時序數據...")
    for step in range(1, 21):  # 模擬20步
        # 隨著時間步增加，輕微改變數據來模擬演化
        noise_factor = 0.05 * step
        mock_lbm.rho.data += np.random.normal(0, noise_factor * 0.01, mock_lbm.rho.data.shape)
        mock_lbm.u.data[:,:,:,2] *= (1 + noise_factor * 0.02)  # 速度逐漸增加
        
        # 收集時序數據
        viz._collect_time_series_data(step)
    
    # 生成時序分析圖
    print("   生成時序分析圖...")
    result_file = viz.save_time_series_analysis(20)
    
    if result_file:
        print(f"   ✅ 時序分析測試完成，圖像已保存: {result_file}")
    else:
        print("   ❌ 時序分析生成失敗")
    
    # 檢查數據存儲
    print(f"   時序數據點數: {len(viz.time_series_data['step_numbers'])}")
    print(f"   Reynolds數範圍: {min(viz.time_series_data['reynolds_numbers']):.3f} ~ {max(viz.time_series_data['reynolds_numbers']):.3f}")

def test_enhanced_pressure_analysis():
    """測試增強壓力分析"""
    print("🧪 測試增強壓力分析功能...")
    
    # 創建模擬求解器
    mock_lbm = MockLBMSolver()
    
    # 創建增強視覺化器
    viz = EnhancedVisualizer(mock_lbm)
    
    # 創建模擬壓力場數據（包含極值）
    nx, ny, nz = mock_lbm.rho.data.shape
    pressure_data = np.random.random((nx, ny, nz)) * 1000
    # 添加一些極值
    pressure_data[0, 0, 0] = 10000  # 極大值
    pressure_data[1, 1, 1] = -5000  # 極小值
    
    # 測試動態範圍對壓力數據的處理
    vmin, vmax = viz._calculate_dynamic_range(pressure_data, 5, 95)
    
    print(f"   壓力數據原始範圍: {np.min(pressure_data):.1f} ~ {np.max(pressure_data):.1f}")
    print(f"   動態範圍調整後: {vmin:.1f} ~ {vmax:.1f}")
    
    # 創建對比圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左圖：使用原始範圍
    im1 = ax1.imshow(pressure_data[:, ny//2, :].T, origin='lower', 
                     vmin=np.min(pressure_data), vmax=np.max(pressure_data),
                     cmap='RdBu_r')
    ax1.set_title("原始範圍（含極值）")
    plt.colorbar(im1, ax=ax1)
    
    # 右圖：使用動態範圍
    im2 = ax2.imshow(pressure_data[:, ny//2, :].T, origin='lower', 
                     vmin=vmin, vmax=vmax, cmap='RdBu_r')
    ax2.set_title("動態範圍調整")
    viz._create_smart_colorbar(ax2, im2, pressure_data[:, ny//2, :], "Pressure", "Pa")
    
    plt.tight_layout()
    plt.savefig("test_pressure_dynamic_range.png", dpi=150)
    plt.close()
    
    print("   ✅ 增強壓力分析測試完成，圖像已保存: test_pressure_dynamic_range.png")

def main():
    """主測試函數"""
    print("🚀 開始測試增強視覺化功能")
    print("=" * 50)
    
    try:
        # 測試動態colorbar
        test_dynamic_colorbar()
        print()
        
        # 測試時序分析
        test_time_series_analysis()
        print()
        
        # 測試增強壓力分析
        test_enhanced_pressure_analysis()
        print()
        
        print("🎉 所有測試完成！")
        print("📁 生成的文件:")
        print("   - test_dynamic_colorbar.png")
        print("   - test_pressure_dynamic_range.png") 
        print("   - report/目錄下的時序分析圖")
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()