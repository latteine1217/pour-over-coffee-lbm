# demo_enhanced_output.py
"""
演示增强版可视化输出功能
展示纵向剖面图和流速分析
"""

import taichi as ti
import numpy as np
import sys
from lbm_solver import LBMSolver
from enhanced_visualizer import EnhancedVisualizer
import config

# 初始化Taichi
ti.init(arch=ti.gpu, device_memory_GB=2.0)

def demo_enhanced_visualization():
    """演示增强版可视化功能"""
    print("=== Pour-Over Coffee 增强版可视化演示 ===")
    print("使用opencode + GitHub Copilot开发")
    print("=" * 60)
    
    # 创建求解器和可视化系统
    print("初始化模拟系统...")
    lbm = LBMSolver()
    enhanced_viz = EnhancedVisualizer(lbm)
    
    # 初始化场
    print("初始化场变量...")
    lbm.init_fields()
    
    # 运行模拟步骤
    print("运行模拟...")
    max_steps = 50
    
    for step in range(max_steps):
        lbm.step()
        
        # 每10步显示进度
        if step % 10 == 0:
            print(f"  Step {step:3d}/{max_steps}")
        
        # 在特定步骤生成可视化
        if step in [20, 40]:
            print(f"\n--- 生成Step {step}的可视化分析 ---")
            simulation_time = step * config.DT * config.SCALE_TIME
            
            # 生成三种类型的分析图
            longitudinal_file = enhanced_viz.save_longitudinal_analysis(simulation_time, step)
            velocity_file = enhanced_viz.save_velocity_analysis(simulation_time, step)
            combined_file = enhanced_viz.save_combined_analysis(simulation_time, step)
            
            print(f"✅ Step {step} 分析图已生成:")
            print(f"   📊 纵向剖面: {longitudinal_file}")
            print(f"   📈 流速分析: {velocity_file}")
            print(f"   📋 综合分析: {combined_file}")
    
    print(f"\n=== 演示完成 ===")
    print("✅ 成功生成以下类型的可视化输出:")
    print("   1. 纵向剖面图 - 显示水流从上到下的过程")
    print("   2. 流速分析图 - 分析水在濾杯中的流速")
    print("   3. 综合分析图 - 结合密度和速度的完整视图")
    print("\n这些图像采用了test_*.png相同的高质量呈现方式:")
    print("   - 英文标签，清晰易读")
    print("   - 300 DPI高分辨率")
    print("   - 专业的科学绘图风格")
    print("   - 详细的图例和坐标轴")

if __name__ == "__main__":
    try:
        demo_enhanced_visualization()
        print("\n🎉 演示成功完成！")
        print("\n现在main.py程序已集成这些功能，")
        print("运行完整模拟时会自动生成这些分析图。")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        sys.exit(1)