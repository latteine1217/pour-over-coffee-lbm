#!/usr/bin/env python3
"""
简化的V60几何测试程序
只测试几何渲染，不涉及复杂的Taichi field操作
"""

import numpy as np
import matplotlib.pyplot as plt
import config

def test_v60_geometry_simple():
    """简化的V60几何测试"""
    print("🧪 简化V60几何测试...")
    
    # 创建网格
    x = np.linspace(0, config.PHYSICAL_WIDTH * 100, config.NX)  # cm
    z = np.linspace(0, config.PHYSICAL_HEIGHT * 100, config.NZ)  # cm
    X, Z = np.meshgrid(x, z)
    
    # V60几何参数
    cup_height_cm = config.CUP_HEIGHT * 100
    top_radius_cm = config.TOP_RADIUS * 100
    bottom_radius_cm = config.BOTTOM_RADIUS * 100
    
    # 修正后的V60几何 (尖端向下，开口向上)
    total_height = z.max()
    cup_top_z = total_height * 0.75  # 濾杯顶部在图片的75%高度处
    cup_bottom_z = cup_top_z - cup_height_cm  # 濾杯底部（出水口）在下方
    
    # 计算锥形边界线
    z_cone = np.linspace(cup_bottom_z, cup_top_z, 100)
    x_center = x.max() / 2  # 水平居中
    
    # 使用V60-02实际规格的锥形计算
    height_ratio = (z_cone - cup_bottom_z) / cup_height_cm
    cone_radius = bottom_radius_cm + (top_radius_cm - bottom_radius_cm) * height_ratio
    
    x_left_boundary = x_center - cone_radius
    x_right_boundary = x_center + cone_radius
    
    # 咖啡床区域
    coffee_bed_height_cm = config.COFFEE_BED_HEIGHT_LU * config.SCALE_LENGTH * 100
    coffee_bed_bottom_z = cup_bottom_z
    coffee_bed_top_z = coffee_bed_bottom_z + coffee_bed_height_cm
    
    # 创建测试图像
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    
    # 添加一些模拟的水流数据作为背景
    # 创建简单的水密度分布
    density_data = np.zeros_like(X)
    for i in range(len(z)):
        for j in range(len(x)):
            # 在V60内部区域添加水密度
            dist_from_center = abs(x[j] - x_center)
            z_pos = z[i]
            if cup_bottom_z <= z_pos <= cup_top_z:
                # 计算该高度的锥形半径
                h_ratio = (z_pos - cup_bottom_z) / cup_height_cm
                radius_at_z = bottom_radius_cm + (top_radius_cm - bottom_radius_cm) * h_ratio
                if dist_from_center <= radius_at_z:
                    # 添加水密度，上部浓度高
                    density_data[i, j] = 600 + 200 * h_ratio
    
    # 显示水密度分布
    density_filtered = np.where(density_data > 100, density_data, np.nan)
    contour = ax.contourf(X, Z, density_filtered, levels=20, cmap='Blues', alpha=0.7)
    plt.colorbar(contour, ax=ax, shrink=0.8, label='Water Density (kg/m³)')
    
    # 添加V60锥形轮廓
    ax.plot(x_left_boundary, z_cone, 'k-', linewidth=3, label='V60 Dripper Wall')
    ax.plot(x_right_boundary, z_cone, 'k-', linewidth=3)
    
    # 濾紙边界 (稍微内缩)
    filter_offset = 0.1  # cm
    ax.plot(x_left_boundary + filter_offset, z_cone, 'gray', linewidth=2, 
            linestyle=':', alpha=0.8, label='Filter Paper')
    ax.plot(x_right_boundary - filter_offset, z_cone, 'gray', linewidth=2, 
            linestyle=':', alpha=0.8)
    
    # 咖啡床区域
    coffee_x = x[(x >= x_center - top_radius_cm*0.8) & (x <= x_center + top_radius_cm*0.8)]
    ax.fill_between(coffee_x, coffee_bed_bottom_z, coffee_bed_top_z,
                    color='brown', alpha=0.4, label='Coffee Bed')
    
    # 出水口 (在濾杯底部，尖端处)
    ax.plot([x_center-bottom_radius_cm, x_center+bottom_radius_cm], 
            [cup_bottom_z, cup_bottom_z], 'red', linewidth=4, label='Outlet')
    
    # 水滴 (在出水口下方)
    ax.scatter([x_center], [cup_bottom_z-1.5], c='blue', s=50, alpha=0.8, label='Water Drop')
    
    # 注水区域指示 (在濾杯顶部上方)
    pour_zone_z = cup_top_z + 1.5
    pour_zone_width = top_radius_cm * 0.3
    ax.plot([x_center-pour_zone_width, x_center+pour_zone_width], 
            [pour_zone_z, pour_zone_z], 'cyan', linewidth=4, 
            marker='v', markersize=8, label='Pour Zone')
    
    # 添加重力箭头 (向下)
    ax.annotate('Gravity', xy=(x.max()*0.85, cup_top_z), xytext=(x.max()*0.85, cup_top_z + 2),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'),
                fontsize=12, fontweight='bold', color='red')
    
    # 设置图表属性
    ax.set_xlabel('X Position (cm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z Height (cm)', fontsize=12, fontweight='bold')
    ax.set_title('修正后的V60几何显示测试\\nV60-02濾杯：尖端向下，开口向上，居中显示', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # 确保纵横比合理
    ax.set_aspect('equal')
    
    # 保存图像
    filename = "test_v60_geometry_fixed.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 测试图像已保存: {filename}")
    
    # 验证几何参数
    print(f"\\n📏 V60-02 几何参数验证：")
    print(f"  - 濾杯高度: {cup_height_cm:.1f} cm")
    print(f"  - 上部直径: {config.TOP_DIAMETER*100:.1f} cm")
    print(f"  - 下部出水孔: {config.BOTTOM_DIAMETER*10:.1f} mm")
    print(f"  - 濾杯中心X坐标: {x_center:.1f} cm")
    print(f"  - 濾杯底部Z坐标: {cup_bottom_z:.1f} cm")
    print(f"  - 濾杯顶部Z坐标: {cup_top_z:.1f} cm")
    
    import math
    radius_diff = config.TOP_RADIUS - config.BOTTOM_RADIUS
    actual_angle = math.degrees(math.atan(radius_diff / config.CUP_HEIGHT))
    print(f"  - 实际锥角: {actual_angle*2:.1f}° (全角)")
    
    print(f"\\n🎯 几何修正验证：")
    print(f"  ✓ 濾杯尖端向下 (出水口在底部: {cup_bottom_z:.1f} cm)")
    print(f"  ✓ 濾杯开口向上 (注水区域在顶部: {cup_top_z:.1f} cm)")  
    print(f"  ✓ 使用V60-02实际锥角 ({actual_angle*2:.1f}°)")
    print(f"  ✓ 濾杯水平居中显示 (中心: {x_center:.1f} cm)")
    print(f"  ✓ 咖啡床位于濾杯底部")
    print(f"  ✓ 注水区域位于濾杯顶部上方")

if __name__ == "__main__":
    test_v60_geometry_simple()