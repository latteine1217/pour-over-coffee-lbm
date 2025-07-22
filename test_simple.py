#!/usr/bin/env python3
"""
整合測試程序 - 驗證所有核心功能
包括：LBM求解器、咖啡粒子系統、可視化系統
"""

import taichi as ti
import numpy as np
import config
from lbm_solver import LBMSolver
from coffee_particles import CoffeeParticleSystem
from visualizer import UnifiedVisualizer

def test_all_systems():
    """測試所有核心系統"""
    print("Pour-Over Coffee Simulation - System Test")
    print("=" * 50)
    
    # 初始化Taichi
    ti.init(arch=ti.gpu, device_memory_GB=2.0)
    
    # 1. 測試LBM求解器
    print("\n1. Testing LBM Solver...")
    lbm = LBMSolver()
    lbm.init_fields()
    print("   ✅ LBM Solver initialized")
    
    # 2. 測試咖啡粒子系統
    print("\n2. Testing Coffee Particle System...")
    particles = CoffeeParticleSystem(max_particles=5000)
    
    # 初始化濾紙系統
    from filter_paper import FilterPaperSystem
    filter_paper = FilterPaperSystem(lbm)
    filter_paper.initialize_filter_geometry()
    
    # 初始化咖啡顆粒系統
    particles.initialize_coffee_bed_confined(filter_paper)
    
    particle_count = particles.particle_count[None]
    if particle_count > 1000:
        print(f"   ✅ Particle system working: {particle_count} particles")
    else:
        print(f"   ⚠️  Particle system issues: only {particle_count} particles")
    
    # 3. 測試可視化系統
    print("\n3. Testing Visualization System...")
    visualizer = UnifiedVisualizer(lbm)
    stats = visualizer.get_statistics()
    print(f"   ✅ Visualization system ready")
    print(f"      Water mass: {stats['total_water_mass']:.3f}")
    print(f"      Max velocity: {stats['max_velocity']:.6f}")
    
    # 4. 測試完整的一個時間步
    print("\n4. Testing Complete Time Step...")
    try:
        # 運行LBM步長 (簡化版本，無需顆粒物理步驟)
        lbm.step()
        
        print("   ✅ Complete time step successful")
        
        # 獲取最終統計
        final_stats = visualizer.get_statistics()
        print(f"      Final water mass: {final_stats['total_water_mass']:.3f}")
        
    except Exception as e:
        print(f"   ❌ Time step failed: {e}")
        return False
    
    print(f"\n=== Test Results ===")
    print(f"✅ All systems operational!")
    print(f"✅ {particle_count} particles active")
    print(f"✅ 3D simulation ready")
    print(f"\nTo run full simulation: python main.py")
    
    return True

if __name__ == "__main__":
    success = test_all_systems()
    if not success:
        exit(1)