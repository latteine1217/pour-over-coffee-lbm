# check_v60_size.py
"""
檢查V60濾杯尺寸是否適合當前網格範圍
"""

import math
import config

def check_v60_vs_grid_size():
    """檢查V60尺寸與網格範圍的匹配度"""
    print("=== V60尺寸與網格範圍檢查 ===")
    
    # 網格參數
    print(f"網格大小: {config.NX}×{config.NY}×{config.NZ}")
    print(f"每格實際尺寸: {config.SCALE_LENGTH*1000:.2f} mm")
    print(f"物理域大小: {config.NX*config.SCALE_LENGTH*100:.1f}×{config.NY*config.SCALE_LENGTH*100:.1f}×{config.NZ*config.SCALE_LENGTH*100:.1f} cm")
    print()
    
    # V60參數
    print(f"V60物理尺寸:")
    print(f"  高度: {config.CUP_HEIGHT*100:.1f} cm")
    print(f"  頂部直徑: {config.TOP_DIAMETER*100:.1f} cm")
    print(f"  頂部半徑: {config.TOP_RADIUS*100:.1f} cm")
    print(f"  底部直徑: {config.BOTTOM_DIAMETER*100:.1f} cm")
    print()
    
    # 轉換為格子單位
    cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
    top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
    bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
    
    print(f"V60格子單位尺寸:")
    print(f"  高度: {cup_height_lu:.1f} 格子")
    print(f"  頂部半徑: {top_radius_lu:.1f} 格子")
    print(f"  底部半徑: {bottom_radius_lu:.1f} 格子")
    print()
    
    # 檢查是否超出網格範圍
    center_x = config.NX / 2
    center_y = config.NY / 2
    
    print(f"網格中心: ({center_x:.1f}, {center_y:.1f})")
    print(f"V60頂部需要的範圍:")
    print(f"  X: {center_x - top_radius_lu:.1f} ~ {center_x + top_radius_lu:.1f}")
    print(f"  Y: {center_y - top_radius_lu:.1f} ~ {center_y + top_radius_lu:.1f}")
    print(f"  Z: 5 ~ {5 + cup_height_lu:.1f}")
    print()
    
    # 檢查邊界
    issues = []
    
    if center_x - top_radius_lu < 0:
        issues.append(f"X負方向超出: 需要 {center_x - top_radius_lu:.1f} (< 0)")
    if center_x + top_radius_lu >= config.NX:
        issues.append(f"X正方向超出: 需要 {center_x + top_radius_lu:.1f} (>= {config.NX})")
    if center_y - top_radius_lu < 0:
        issues.append(f"Y負方向超出: 需要 {center_y - top_radius_lu:.1f} (< 0)")
    if center_y + top_radius_lu >= config.NY:
        issues.append(f"Y正方向超出: 需要 {center_y + top_radius_lu:.1f} (>= {config.NY})")
    if 5 + cup_height_lu >= config.NZ:
        issues.append(f"Z方向超出: 需要 {5 + cup_height_lu:.1f} (>= {config.NZ})")
    
    if issues:
        print("❌ V60尺寸問題:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        
        # 計算建議的縮放比例
        max_radius_allowed = min(center_x, center_y, config.NX - center_x, config.NY - center_y) * 0.9
        scale_factor = max_radius_allowed / top_radius_lu
        
        print(f"建議解決方案:")
        print(f"  最大允許半徑: {max_radius_allowed:.1f} 格子")
        print(f"  建議縮放比例: {scale_factor:.3f}")
        print(f"  縮放後V60頂部半徑: {top_radius_lu * scale_factor:.1f} 格子")
        print(f"  縮放後V60物理頂部半徑: {config.TOP_RADIUS * scale_factor * 100:.1f} cm")
        
        return False, scale_factor
    else:
        print("✅ V60尺寸在網格範圍內")
        return True, 1.0

def suggest_grid_resize():
    """建議網格大小調整"""
    print("\n=== 網格調整建議 ===")
    
    # 計算理想網格大小
    desired_margin = 10  # 格子單位的邊距
    top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
    cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
    
    ideal_nx = int((top_radius_lu + desired_margin) * 2)
    ideal_ny = int((top_radius_lu + desired_margin) * 2)
    ideal_nz = int(cup_height_lu + desired_margin * 2)
    
    # 調整為2的倍數便於GPU優化
    ideal_nx = ((ideal_nx + 7) // 8) * 8
    ideal_ny = ((ideal_ny + 7) // 8) * 8
    ideal_nz = ((ideal_nz + 7) // 8) * 8
    
    print(f"當前網格: {config.NX}×{config.NY}×{config.NZ}")
    print(f"建議網格: {ideal_nx}×{ideal_ny}×{ideal_nz}")
    print(f"記憶體影響: {ideal_nx*ideal_ny*ideal_nz / (config.NX*config.NY*config.NZ):.2f}倍")

def main():
    """主函數"""
    print("V60濾杯尺寸檢查")
    print("=" * 50)
    
    is_valid, scale_factor = check_v60_vs_grid_size()
    
    if not is_valid:
        suggest_grid_resize()
        print(f"\n⚠️  需要調整V60尺寸或網格大小")
        print(f"   快速修正: 將V60縮放至 {scale_factor:.3f} 倍")
    else:
        print(f"\n✅ V60尺寸設定合理")

if __name__ == "__main__":
    main()