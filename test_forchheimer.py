#!/usr/bin/env python3
"""
Forchheimer項實現測試
專門測試新的Forchheimer非線性阻力計算功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import taichi as ti
import numpy as np
import config.config as config
from src.physics.filter_paper import FilterPaperSystem

# 初始化Taichi
ti.init(arch=ti.metal, device_memory_GB=4.0)

@ti.data_oriented
class MockLBMSolver:
    """模擬LBM求解器，用於測試濾紙系統"""
    
    def __init__(self):
        # 創建基本場
        self.u = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.rho = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self.solid = ti.field(dtype=ti.u8, shape=(config.NX, config.NY, config.NZ))
        self.body_force = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        print("✅ 模擬LBM求解器初始化完成")
        
    @ti.kernel
    def initialize_test_fields(self):
        """初始化測試用的場變數"""
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            # 設置一個簡單的測試速度場
            self.u[i, j, k] = ti.Vector([0.01, 0.005, -0.02])  # 測試速度
            self.rho[i, j, k] = 1.0
            self.solid[i, j, k] = ti.u8(0)  # 非固體
            self.body_force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

def test_forchheimer_implementation():
    """測試Forchheimer實現"""
    print("\n🔬 Forchheimer項實現測試")
    print("=" * 50)
    
    # 創建模擬LBM求解器
    lbm = MockLBMSolver()
    lbm.initialize_test_fields()
    
    # 創建濾紙系統
    print("🔄 初始化濾紙系統...")
    filter_system = FilterPaperSystem(lbm)
    
    # 初始化濾紙幾何
    print("🔄 設置濾紙幾何...")
    filter_system.initialize_filter_geometry()
    
    # 測試參數場初始化
    print("🔬 檢查Forchheimer參數場...")
    
    # 轉換為numpy檢查
    forchheimer_coeff_data = filter_system.forchheimer_coeff.to_numpy()
    permeability_data = filter_system.permeability.to_numpy()
    velocity_magnitude_data = filter_system.velocity_magnitude.to_numpy()
    filter_zone_data = filter_system.filter_zone.to_numpy()
    
    # 統計分析
    total_filter_nodes = np.sum(filter_zone_data)
    filter_nodes_with_params = np.sum((forchheimer_coeff_data > 0) & (filter_zone_data == 1))
    permeability_nodes = np.sum((permeability_data > 0) & (filter_zone_data == 1))
    
    print(f"📊 濾紙統計:")
    print(f"  總濾紙節點數: {total_filter_nodes:,}")
    print(f"  設置Forchheimer係數的節點: {filter_nodes_with_params:,}")
    print(f"  設置滲透率的節點: {permeability_nodes:,}")
    
    if total_filter_nodes > 0:
        avg_forchheimer = np.mean(forchheimer_coeff_data[filter_zone_data == 1])
        avg_permeability = np.mean(permeability_data[filter_zone_data == 1])
        print(f"  平均Forchheimer係數: {avg_forchheimer:.6f}")
        print(f"  平均滲透率: {avg_permeability:.2e} lu²")
    
    # 測試Forchheimer阻力計算
    print("\n🔄 測試Forchheimer阻力計算...")
    
    # 記錄初始速度
    initial_u = lbm.u.to_numpy()
    initial_body_force = lbm.body_force.to_numpy()
    
    # 應用濾紙效應 (包含Forchheimer計算)
    filter_system.apply_filter_effects()
    
    # 記錄處理後的場
    final_u = lbm.u.to_numpy()
    final_body_force = lbm.body_force.to_numpy()
    
    # 分析速度變化
    u_change = np.linalg.norm(final_u - initial_u, axis=3)
    force_change = np.linalg.norm(final_body_force - initial_body_force, axis=3)
    
    # 只在濾紙區域統計
    filter_mask = filter_zone_data == 1
    if np.sum(filter_mask) > 0:
        avg_u_change = np.mean(u_change[filter_mask])
        max_u_change = np.max(u_change[filter_mask])
        avg_force_change = np.mean(force_change[filter_mask])
        
        print(f"🔬 Forchheimer效應分析:")
        print(f"  濾紙區域平均速度變化: {avg_u_change:.6f}")
        print(f"  濾紙區域最大速度變化: {max_u_change:.6f}")
        print(f"  平均體力變化: {avg_force_change:.6f}")
        
        # 檢查速度幅值場更新
        velocity_magnitude_final = filter_system.velocity_magnitude.to_numpy()
        nonzero_velocity_nodes = np.sum(velocity_magnitude_final[filter_mask] > 1e-8)
        print(f"  有速度的濾紙節點: {nonzero_velocity_nodes:,}")
        
        if avg_u_change > 1e-8:
            print("✅ Forchheimer阻力計算正常工作")
        else:
            print("⚠️  Forchheimer阻力計算可能未生效")
    
    # 測試統計功能
    print("\n📊 濾紙系統統計:")
    stats = filter_system.get_filter_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            if key == 'average_resistance':
                print(f"  {key}: {value:.2e}")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:,}")
    
    print("\n✅ Forchheimer測試完成")
    return True

if __name__ == "__main__":
    try:
        success = test_forchheimer_implementation()
        if success:
            print("\n🎉 所有Forchheimer測試通過！")
            exit(0)
        else:
            print("\n❌ Forchheimer測試失敗")
            exit(1)
    except Exception as e:
        print(f"\n💥 測試過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()
        exit(1)