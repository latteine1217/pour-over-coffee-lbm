# test_enhanced_visualization.py
"""
测试增强版可视化功能
生成纵向剖面图和流速分析示例
"""

import taichi as ti
import numpy as np
from lbm_solver import LBMSolver
from enhanced_visualizer import EnhancedVisualizer
import config

# 初始化Taichi (使用较小的记忆体)
ti.init(arch=ti.gpu, device_memory_GB=2.0)

def test_enhanced_visualization():
    """测试增强版可视化功能"""
    print("=== 测试增强版可视化功能 ===")
    
    # 创建求解器
    lbm = LBMSolver()
    enhanced_viz = EnhancedVisualizer(lbm)
    
    # 初始化
    lbm.init_fields()
    
    # 运行几步以产生一些流动
    for step in range(20):
        lbm.step()
        if step % 5 == 0:
            print(f"Step {step}")
    
    # 测试各种可视化功能
    timestamp = 0.5  # 假设时间
    step_num = 20
    
    print("\n=== 生成可视化图像 ===")
    
    try:
        # 测试纵向剖面分析
        print("1. 生成纵向剖面分析...")
        longitudinal_file = enhanced_viz.save_longitudinal_analysis(timestamp, step_num)
        
        # 测试流速分析  
        print("2. 生成流速分析...")
        velocity_file = enhanced_viz.save_velocity_analysis(timestamp, step_num)
        
        # 测试综合分析
        print("3. 生成综合分析...")
        combined_file = enhanced_viz.save_combined_analysis(timestamp, step_num)
        
        print("\n=== 测试成功 ===")
        print(f"生成的文件:")
        print(f"  - {longitudinal_file}")
        print(f"  - {velocity_file}")
        print(f"  - {combined_file}")
        
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_enhanced_visualization()
    if success:
        print("\n✅ 增强版可视化测试通过！")
        print("现在可以在main.py中使用这些功能。")
    else:
        print("\n❌ 测试失败，请检查代码。")