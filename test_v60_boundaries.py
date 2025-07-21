# test_v60_boundaries.py
"""
測試V60濾杯邊界條件設定
驗證水流是否正確約束在濾杯內部
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import config
from lbm_solver import LBMSolver

ti.init(arch=ti.gpu)

def test_v60_boundary_conditions():
    """測試V60邊界條件"""
    print("=== 測試V60濾杯邊界條件 ===")
    
    # 初始化LBM求解器
    solver = LBMSolver()
    solver.init_fields()
    
    print(f"模擬網格: {config.NX}×{config.NY}×{config.NZ}")
    
    # 獲取固體場
    solid_field = solver.solid.to_numpy()
    
    # 統計V60幾何
    total_points = config.NX * config.NY * config.NZ
    solid_points = np.sum(solid_field)
    fluid_points = total_points - solid_points
    
    print(f"總網格點: {total_points:,}")
    print(f"固體點 (V60壁): {solid_points:,}")
    print(f"流體點 (V60內): {fluid_points:,}")
    print(f"V60內部體積比: {fluid_points/total_points*100:.1f}%")
    
    # 檢查V60幾何合理性
    center_x, center_y = config.NX//2, config.NY//2
    bottom_z = 5
    cup_height_lu = int(config.CUP_HEIGHT / config.SCALE_LENGTH)
    
    print(f"\nV60幾何檢查:")
    print(f"中心位置: ({center_x}, {center_y})")
    print(f"底部Z: {bottom_z}")
    print(f"濾杯高度: {cup_height_lu} 格子單位")
    
    # 檢查不同高度的V60橫截面
    heights_to_check = [bottom_z, bottom_z + cup_height_lu//4, 
                        bottom_z + cup_height_lu//2, bottom_z + 3*cup_height_lu//4, 
                        bottom_z + cup_height_lu]
    
    for z in heights_to_check:
        if z < config.NZ:
            # 計算該高度的流體網格點數
            slice_data = solid_field[:, :, z]
            fluid_points_slice = np.sum(slice_data == 0)
            
            # 計算理論V60半徑
            height_ratio = (z - bottom_z) / cup_height_lu
            theoretical_radius = config.BOTTOM_RADIUS + (config.TOP_RADIUS - config.BOTTOM_RADIUS) * height_ratio
            theoretical_area = np.pi * (theoretical_radius / config.SCALE_LENGTH)**2
            
            print(f"Z={z:2d}: 流體點={fluid_points_slice:4d}, "
                  f"理論面積={theoretical_area:.1f}, "
                  f"實際/理論={fluid_points_slice/theoretical_area:.2f}")
    
    # 創建可視化
    create_v60_boundary_visualization(solid_field)
    
    return True

def create_v60_boundary_visualization(solid_field):
    """創建V60邊界條件可視化"""
    print("\n生成V60邊界條件可視化...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. XZ平面剖面 (Y中心)
    y_center = config.NY // 2
    xz_slice = solid_field[:, y_center, :]
    ax1.imshow(xz_slice.T, origin='lower', cmap='RdBu', aspect='auto')
    ax1.set_title('V60 Boundary: XZ Cross-Section (Y=center)')
    ax1.set_xlabel('X (grid points)')
    ax1.set_ylabel('Z (grid points)')
    
    # 2. YZ平面剖面 (X中心) 
    x_center = config.NX // 2
    yz_slice = solid_field[x_center, :, :]
    ax2.imshow(yz_slice.T, origin='lower', cmap='RdBu', aspect='auto')
    ax2.set_title('V60 Boundary: YZ Cross-Section (X=center)')
    ax2.set_xlabel('Y (grid points)')
    ax2.set_ylabel('Z (grid points)')
    
    # 3. XY平面剖面 (Z中間)
    z_middle = 5 + int(config.CUP_HEIGHT / config.SCALE_LENGTH / 2)
    if z_middle < config.NZ:
        xy_slice = solid_field[:, :, z_middle]
        ax3.imshow(xy_slice.T, origin='lower', cmap='RdBu', aspect='equal')
        ax3.set_title(f'V60 Boundary: XY Cross-Section (Z={z_middle})')
        ax3.set_xlabel('X (grid points)')
        ax3.set_ylabel('Y (grid points)')
        
        # 添加理論V60圓形輪廓
        height_ratio = (z_middle - 5) / (config.CUP_HEIGHT / config.SCALE_LENGTH)
        theoretical_radius_lu = (config.BOTTOM_RADIUS + 
                               (config.TOP_RADIUS - config.BOTTOM_RADIUS) * height_ratio) / config.SCALE_LENGTH
        circle = plt.Circle((config.NX//2, config.NY//2), theoretical_radius_lu, 
                           fill=False, color='yellow', linewidth=2, label='Theoretical V60')
        ax3.add_patch(circle)
        ax3.legend()
    
    # 4. V60高度剖面分析
    bottom_z = 5
    cup_height_lu = int(config.CUP_HEIGHT / config.SCALE_LENGTH)
    z_range = range(bottom_z, min(bottom_z + cup_height_lu, config.NZ))
    
    fluid_areas = []
    theoretical_areas = []
    
    for z in z_range:
        # 實際流體面積
        slice_data = solid_field[:, :, z]
        fluid_points = np.sum(slice_data == 0)
        fluid_areas.append(fluid_points)
        
        # 理論V60面積
        height_ratio = (z - bottom_z) / cup_height_lu
        theoretical_radius = config.BOTTOM_RADIUS + (config.TOP_RADIUS - config.BOTTOM_RADIUS) * height_ratio
        theoretical_area = np.pi * (theoretical_radius / config.SCALE_LENGTH)**2
        theoretical_areas.append(theoretical_area)
    
    ax4.plot(z_range, fluid_areas, 'b-', label='Actual Fluid Area', linewidth=2)
    ax4.plot(z_range, theoretical_areas, 'r--', label='Theoretical V60 Area', linewidth=2)
    ax4.set_xlabel('Z (grid points)')
    ax4.set_ylabel('Cross-sectional Area (grid points)')
    ax4.set_title('V60 Cross-sectional Area vs Height')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v60_boundary_conditions_test.png', dpi=300, bbox_inches='tight')
    print("V60邊界條件測試圖已保存: v60_boundary_conditions_test.png")
    
    # 計算匹配度
    if len(fluid_areas) > 0:
        match_ratio = np.mean(np.array(fluid_areas) / np.array(theoretical_areas))
        print(f"實際/理論面積匹配度: {match_ratio:.3f}")
        if 0.8 <= match_ratio <= 1.2:
            print("✅ V60邊界條件設定合理")
        else:
            print("⚠️  V60邊界條件可能需要調整")

def main():
    """主函數"""
    print("V60濾杯邊界條件測試")
    print("=" * 50)
    
    success = test_v60_boundary_conditions()
    
    if success:
        print("\n✅ V60邊界條件測試完成")
    else:
        print("\n❌ V60邊界條件測試失敗")

if __name__ == "__main__":
    main()