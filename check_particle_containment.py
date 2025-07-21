# check_particle_containment.py
"""
檢查咖啡粉顆粒分布是否完全在V60濾杯內部
"""

import math
import numpy as np
import config

def check_v60_geometry():
    """檢查V60幾何參數"""
    print("=== V60濾杯幾何參數檢查 ===")
    print(f"V60內部高度: {config.CUP_HEIGHT*100:.1f} cm")
    print(f"V60內部上徑: {config.TOP_DIAMETER*100:.1f} cm")
    print(f"V60內部底徑: {config.BOTTOM_DIAMETER*100:.1f} cm")
    print(f"V60錐角: {config.V60_CONE_ANGLE:.1f}°")
    print(f"V60內部體積: {config.V60_INTERNAL_VOLUME*1e6:.1f} cm³")
    print()

def check_v60_constraints(x_lu, y_lu, z_lu):
    """檢查給定位置是否在V60濾杯內部（格子單位）"""
    # 轉換為物理單位
    x_phys = x_lu * config.SCALE_LENGTH
    y_phys = y_lu * config.SCALE_LENGTH  
    z_phys = z_lu * config.SCALE_LENGTH
    
    # 計算距離中心的半徑
    center_x = config.NX / 2 * config.SCALE_LENGTH
    center_y = config.NY / 2 * config.SCALE_LENGTH
    radius_from_center = math.sqrt((x_phys - center_x)**2 + (y_phys - center_y)**2)
    
    # 計算在當前高度z處的V60半徑
    # V60是倒錐形：z=0為底部（小半徑），z=CUP_HEIGHT為頂部（大半徑）
    height_ratio = z_phys / config.CUP_HEIGHT
    if height_ratio < 0 or height_ratio > 1:
        return False
    
    # 線性插值計算當前高度的半徑
    current_radius = config.BOTTOM_RADIUS + (config.TOP_RADIUS - config.BOTTOM_RADIUS) * height_ratio
    
    # 檢查是否在半徑內
    return radius_from_center <= current_radius

def analyze_coffee_bed_distribution():
    """分析咖啡床分布"""
    print("=== 咖啡床分布分析 ===")
    print(f"咖啡床高度: {config.COFFEE_BED_HEIGHT_PHYS*100:.1f} cm")
    print(f"咖啡床體積: {config.COFFEE_BED_VOLUME_PHYS*1e6:.1f} cm³")
    print(f"咖啡床填充比例: {config.COFFEE_FILL_RATIO*100:.1f}%")
    print(f"實際孔隙率: {config.ACTUAL_POROSITY*100:.1f}%")
    print()
    
    # 檢查咖啡床是否超出V60範圍
    if config.COFFEE_BED_HEIGHT_PHYS > config.CUP_HEIGHT:
        print("❌ 問題：咖啡床高度超出V60濾杯高度！")
        print(f"   咖啡床高度: {config.COFFEE_BED_HEIGHT_PHYS*100:.1f} cm")
        print(f"   V60高度: {config.CUP_HEIGHT*100:.1f} cm")
        return False
    
    # 檢查咖啡床分布是否在V60內部
    bed_height = config.COFFEE_BED_HEIGHT_PHYS
    bed_top_radius = config.COFFEE_BED_TOP_RADIUS
    
    print(f"咖啡床頂部半徑: {bed_top_radius*100:.1f} cm")
    print(f"V60底部半徑: {config.BOTTOM_RADIUS*100:.2f} cm")
    print(f"V60頂部半徑: {config.TOP_RADIUS*100:.1f} cm")
    
    # 檢查床的頂部是否在V60內
    top_height_ratio = bed_height / config.CUP_HEIGHT
    top_allowed_radius = config.BOTTOM_RADIUS + (config.TOP_RADIUS - config.BOTTOM_RADIUS) * top_height_ratio
    
    print(f"咖啡床頂部高度比例: {top_height_ratio:.2f}")
    print(f"該高度處V60允許半徑: {top_allowed_radius*100:.1f} cm")
    
    if bed_top_radius > top_allowed_radius * 1.01:  # 1%容差
        print("❌ 問題：咖啡床半徑超出V60在該高度的允許範圍！")
        return False
    else:
        print("✅ 咖啡床半徑在V60允許範圍內")
    
    return True

def simulate_particle_positions():
    """模擬粒子位置分布並檢查是否在V60內"""
    print("=== 模擬粒子位置檢查 ===")
    
    # 模擬咖啡粒子初始化邏輯 (基於修正後的coffee_particles.py)
    center_x_lu = config.NX / 2
    center_y_lu = config.NY / 2
    bottom_z_lu = 5  # 底部留一些空間
    bed_height_lu = config.COFFEE_BED_HEIGHT_LU
    
    # 使用正確的錐形半徑
    bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
    bed_top_radius_lu = config.COFFEE_BED_TOP_RADIUS / config.SCALE_LENGTH
    
    print(f"模擬參數 (格子單位):")
    print(f"  中心位置: ({center_x_lu:.1f}, {center_y_lu:.1f})")
    print(f"  底部Z: {bottom_z_lu}")
    print(f"  床高度: {bed_height_lu}")
    print(f"  底部半徑: {bottom_radius_lu:.1f}")
    print(f"  頂部半徑: {bed_top_radius_lu:.1f}")
    print()
    
    # 模擬一些粒子位置
    particles_per_layer = 50
    layers = max(1, int(bed_height_lu / 2))  # 每2個網格一層
    
    total_particles = 0
    particles_outside = 0
    
    np.random.seed(42)  # 固定隨機種子便於復現
    
    for layer in range(layers):
        z_pos = bottom_z_lu + (layer + 0.5) * bed_height_lu / layers
        height_ratio = (layer + 0.5) / layers  # 當前層在咖啡床中的相對高度
        
        # 根據V60錐形計算當前層的允許半徑
        current_radius_lu = bottom_radius_lu + (bed_top_radius_lu - bottom_radius_lu) * height_ratio
        
        for i in range(particles_per_layer):
            # 在當前層的圓形區域內隨機分佈，留安全邊距
            r = np.sqrt(np.random.random()) * current_radius_lu * 0.95  # 留5%安全邊距
            theta = np.random.random() * 2.0 * math.pi
            
            x_pos = center_x_lu + r * math.cos(theta)
            y_pos = center_y_lu + r * math.sin(theta)
            
            # 檢查是否在網格範圍內
            if 0 < x_pos < config.NX and 0 < y_pos < config.NY and 0 < z_pos < config.NZ:
                total_particles += 1
                
                # 檢查是否在V60內部
                if not check_v60_constraints(x_pos, y_pos, z_pos):
                    particles_outside += 1
                    if particles_outside <= 5:  # 只打印前5個超出範圍的粒子
                        print(f"❌ 粒子超出V60: ({x_pos:.1f}, {y_pos:.1f}, {z_pos:.1f}) lu")
    
    print(f"模擬結果:")
    print(f"  總粒子數: {total_particles}")
    print(f"  超出V60的粒子數: {particles_outside}")
    print(f"  超出比例: {particles_outside/max(1,total_particles)*100:.1f}%")
    
    if particles_outside > 0:
        print("❌ 發現有粒子超出V60濾杯範圍！")
        return False
    else:
        print("✅ 所有模擬粒子都在V60濾杯內部")
        return True

def main():
    """主函數"""
    print("咖啡粉顆粒包含性檢查")
    print("=" * 50)
    
    # 檢查V60幾何
    check_v60_geometry()
    
    # 分析咖啡床分布
    bed_ok = analyze_coffee_bed_distribution()
    
    # 模擬粒子位置
    particles_ok = simulate_particle_positions()
    
    print("\n=== 最終結果 ===")
    if bed_ok and particles_ok:
        print("✅ 咖啡粉顆粒分布完全在V60濾杯內部")
    else:
        print("❌ 咖啡粉顆粒分布存在問題，需要調整參數")
        print("\n建議修正:")
        if not bed_ok:
            print("- 減少咖啡床填充比例")
            print("- 調整咖啡床半徑計算")
        if not particles_ok:
            print("- 調整粒子初始化邏輯")
            print("- 確保粒子分布遵循V60錐形約束")

if __name__ == "__main__":
    main()