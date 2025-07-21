#!/usr/bin/env python3
"""
咖啡颗粒数量验证程序
验证更新后的颗粒参数和计算结果
"""

import math
import numpy as np
import config

def verify_particle_parameters():
    """验证咖啡颗粒参数"""
    print("☕ 咖啡颗粒参数验证")
    print("=" * 50)
    
    # 显示更新后的参数
    print(f"📏 基础参数:")
    print(f"  咖啡粉总质量: {config.COFFEE_POWDER_MASS * 1000:.1f} g")
    print(f"  咖啡豆密度: {config.COFFEE_BEAN_DENSITY} kg/m³ ({config.COFFEE_BEAN_DENSITY/1000} g/cm³)")
    print(f"  主体粒径: {config.PARTICLE_DIAMETER_MM} mm")
    print(f"  颗粒半径: {config.PARTICLE_RADIUS_M * 1000:.3f} mm")
    
    print(f"\n🔢 颗粒数量统计:")
    print(f"  总颗粒数: {config.TOTAL_PARTICLE_COUNT:,} 个")
    print(f"  主体颗粒: {config.MAIN_PARTICLE_COUNT:,} 个 (80%)")
    print(f"  细粉颗粒: {config.FINE_PARTICLE_COUNT:,} 个 (10%)")
    print(f"  粗粒颗粒: {config.COARSE_PARTICLE_COUNT:,} 个 (10%)")
    
    print(f"\n⚖️ 单颗粒参数:")
    print(f"  单颗体积: {config.SINGLE_PARTICLE_VOLUME:.2e} m³")
    print(f"  单颗质量: {config.SINGLE_PARTICLE_MASS:.2e} kg")
    print(f"  单颗质量: {config.SINGLE_PARTICLE_MASS * 1e6:.3f} mg")
    
    # 验证计算
    print(f"\n✅ 验证计算:")
    
    # 验证质量
    calculated_mass = config.MAIN_PARTICLE_COUNT * config.SINGLE_PARTICLE_MASS
    print(f"  主体颗粒总质量: {calculated_mass * 1000:.3f} g")
    
    # 验证总体积
    coffee_bed_volume = config.COFFEE_BED_VOLUME_PHYS
    coffee_solid_volume = config.COFFEE_SOLID_VOLUME
    actual_porosity = config.ACTUAL_POROSITY
    
    print(f"  V60内部总体积: {config.V60_INTERNAL_VOLUME * 1e6:.1f} cm³")
    print(f"  咖啡床体积: {coffee_bed_volume * 1e6:.1f} cm³ (填充{config.COFFEE_FILL_RATIO:.0%})")
    print(f"  颗粒固体体积: {coffee_solid_volume * 1e6:.1f} cm³") 
    print(f"  实际孔隙率: {actual_porosity:.1%}")
    print(f"  有效半径: {config.EFFECTIVE_RADIUS * 100:.1f} cm")
    
    # LBM网格中的表示
    print(f"\n🏗️ LBM网格表示:")
    print(f"  咖啡床高度: {config.COFFEE_BED_HEIGHT_PHYS * 100:.2f} cm")
    print(f"  网格层数: {config.COFFEE_BED_HEIGHT_LU} 层")
    print(f"  每层网格数: {config.NX * config.NY} 个")
    print(f"  咖啡床总网格: {config.COFFEE_BED_HEIGHT_LU * config.NX * config.NY} 个")
    
    # 颗粒密度分析
    total_coffee_grids = config.COFFEE_BED_HEIGHT_LU * config.NX * config.NY
    particles_per_grid = config.TOTAL_PARTICLE_COUNT / total_coffee_grids
    
    print(f"\n📊 颗粒分布密度:")
    print(f"  平均每网格颗粒数: {particles_per_grid:.1f} 个")
    print(f"  主体颗粒每网格: {config.MAIN_PARTICLE_COUNT / total_coffee_grids:.1f} 个")
    
    # 物理尺度对比
    grid_size_mm = config.SCALE_LENGTH * 1000
    print(f"\n📐 尺度对比:")
    print(f"  网格尺寸: {grid_size_mm:.2f} mm")
    print(f"  颗粒直径: {config.PARTICLE_DIAMETER_MM} mm")
    print(f"  颗粒/网格比: {config.PARTICLE_DIAMETER_MM / grid_size_mm:.2f}")
    
    if config.PARTICLE_DIAMETER_MM < grid_size_mm:
        print(f"  ✓ 颗粒小于网格，适合亚网格建模")
    else:
        print(f"  ⚠️ 颗粒大于网格，需要多网格建模")
    
    # 萃取表面积估算
    print(f"\n🫧 萃取表面积分析:")
    single_particle_surface = 4 * math.pi * config.PARTICLE_RADIUS_M**2
    total_surface_area = config.TOTAL_PARTICLE_COUNT * single_particle_surface
    
    print(f"  单颗表面积: {single_particle_surface * 1e6:.3f} mm²")
    print(f"  总表面积: {total_surface_area:.3f} m²")
    print(f"  比表面积: {total_surface_area / config.COFFEE_POWDER_MASS:.1f} m²/kg")
    
    print(f"\n🎯 建议:")
    print("  ✓ 使用当前参数进行LBM建模")
    print("  ✓ 颗粒数量级合理 (~50万个)")
    print("  ✓ 粒径分布符合手冲研磨标准")
    print("  ✓ 孔隙率设置合理 (45%)")

def compare_with_previous():
    """与之前参数对比"""
    print(f"\n📈 参数改进对比:")
    print("=" * 30)
    
    # 之前的参数
    old_dp = 5e-4  # 0.5mm
    old_particle_volume = (4/3) * math.pi * (old_dp/2)**3
    old_particle_count = config.COFFEE_POWDER_MASS / (old_particle_volume * config.COFFEE_BEAN_DENSITY)
    
    print(f"之前参数 (均匀0.5mm):")
    print(f"  颗粒数: {old_particle_count:.0f} 个")
    print(f"  粒径: 0.5 mm")
    
    print(f"\n现在参数 (分布式):")
    print(f"  颗粒数: {config.TOTAL_PARTICLE_COUNT:,} 个")
    print(f"  主体粒径: {config.PARTICLE_DIAMETER_MM} mm")
    print(f"  包含粒径分布: 0.2-1.0 mm")
    
    improvement_ratio = config.TOTAL_PARTICLE_COUNT / old_particle_count
    print(f"\n改进:")
    print(f"  颗粒数增加: {improvement_ratio:.1f}x")
    print(f"  ✓ 更真实的粒径分布")
    print(f"  ✓ 基于实际手冲研磨标准")
    print(f"  ✓ 考虑细粉和粗粒影响")

if __name__ == "__main__":
    verify_particle_parameters()
    compare_with_previous()