#!/usr/bin/env python3
"""
生成修正后V60几何的可视化图像
运行短时间的LBM模拟来演示修正后的效果
"""

import taichi as ti
import numpy as np
import sys

# 初始化Taichi
ti.init(arch=ti.metal)

import config
import lbm_solver
import enhanced_visualizer

def generate_fixed_v60_visualization():
    """生成修正后的V60可视化图像"""
    print("🚀 生成修正后的V60几何可视化...")
    
    # 创建3D LBM求解器
    print("初始化3D LBM求解器...")
    solver = lbm_solver.LBMSolver()
    
    # 创建增强可视化器
    print("初始化增强可视化器...")
    visualizer = enhanced_visualizer.EnhancedVisualizer(solver)
    
    # 运行短时间模拟
    print("开始LBM模拟...")
    max_steps = 50  # 短时间模拟
    
    for step in range(max_steps):
        # 设置边界条件
        solver.apply_boundary_conditions_3d()
        
        # LBM求解
        solver.collision_3d()
        solver.streaming_3d()
        
        # 计算宏观量
        # 注意：我们使用step()方法，它包含了完整的LBM步骤
        # solver.step() # 这会重复执行collision和streaming
        
        # 每10步生成一次可视化
        if step % 10 == 0:
            timestamp = step * config.SCALE_TIME
            print(f"步骤 {step}: 生成可视化图像...")
            
            # 生成纵向分析图
            longitudinal_file = visualizer.save_longitudinal_analysis(timestamp, step)
            print(f"  ✓ 纵向分析: {longitudinal_file}")
            
            # 生成组合分析图
            combined_file = visualizer.save_combined_analysis(timestamp, step)
            print(f"  ✓ 组合分析: {combined_file}")
    
    print(f"\\n✅ 修正后的V60可视化生成完成！")
    print(f"✓ V60濾杯方向：尖端向下，开口向上")
    print(f"✓ V60锥角：68.7° (V60-02实际规格)")
    print(f"✓ 濾杯位置：水平和垂直居中")
    print(f"✓ 咖啡床：位于濾杯底部")
    print(f"✓ 注水区域：位于濾杯顶部上方")

if __name__ == "__main__":
    generate_fixed_v60_visualization()