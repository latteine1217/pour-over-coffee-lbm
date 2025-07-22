# geometry_visualizer.py
"""
V60幾何模型視覺化工具
輸出當前的濾杯和濾紙幾何配置給用戶檢查
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import config
from coffee_particles import CoffeeParticleSystem
from filter_paper import FilterPaperSystem
from lbm_solver import LBMSolver

def visualize_v60_geometry_with_particles():
    """生成包含咖啡顆粒分佈的V60幾何模型視覺化"""

    # 基本幾何參數（格子單位）
    center_x = config.NX * 0.5
    center_y = config.NY * 0.5
    top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
    bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
    cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH

    # V60位置
    v60_bottom_z = 5.0
    v60_top_z = v60_bottom_z + cup_height_lu
    wall_thickness = 2.0

    # 2mm空隙和濾紙參數
    air_gap_lu = 0.002 / config.SCALE_LENGTH
    paper_thickness_lu = max(1.0, 0.0001 / config.SCALE_LENGTH)

    print("=== V60幾何模型詳細規格 (含咖啡顆粒) ===")
    print(f"\n📐 基本尺寸:")
    print(f"   計算域大小: {config.NX} × {config.NY} × {config.NZ} 格子單位")
    print(f"   物理域大小: {config.NX * config.SCALE_LENGTH * 100:.1f} × {config.NY * config.SCALE_LENGTH * 100:.1f} × {config.NZ * config.SCALE_LENGTH * 100:.1f} cm")
    print(f"   格子解析度: {config.SCALE_LENGTH * 1000:.3f} mm/格子")

    print(f"\n🍵 V60濾杯:")
    print(f"   高度: {config.CUP_HEIGHT * 100:.1f} cm ({cup_height_lu:.1f} 格子)")
    print(f"   頂部直徑: {config.TOP_RADIUS * 2 * 100:.1f} cm ({top_radius_lu * 2:.1f} 格子)")
    print(f"   底部直徑: {config.BOTTOM_RADIUS * 2 * 100:.1f} cm ({bottom_radius_lu * 2:.1f} 格子)")
    print(f"   壁厚: {wall_thickness * config.SCALE_LENGTH * 1000:.1f} mm ({wall_thickness:.1f} 格子)")
    print(f"   位置: Z = {v60_bottom_z:.1f} ~ {v60_top_z:.1f} 格子")

    print(f"\n📄 濾紙系統:")
    print(f"   濾紙厚度: 0.1 mm ({paper_thickness_lu:.1f} 格子)")
    print(f"   V60-濾紙空隙: 2.0 mm ({air_gap_lu:.2f} 格子)")
    print(f"   形狀: 完整圓錐形（非平底）")
    print(f"   覆蓋範圍: 完整V60內表面")

    print(f"\n🌊 流體路徑:")
    print(f"   入口: V60頂部注水區域")
    print(f"   經過: 咖啡 → 濾紙 → 2mm空隙")
    print(f"   出口: 計算域邊界（非V60底部直接出口）")
    print(f"   V60底部: 完全封閉")

    # 初始化系統以生成咖啡顆粒
    print(f"\n☕ 生成咖啡顆粒分佈...")
    try:
        # 初始化必要的系統
        import taichi as ti
        ti.init(arch=ti.cpu)

        lbm = LBMSolver()
        filter_system = FilterPaperSystem(lbm)
        filter_system.initialize_filter_geometry()

        particle_system = CoffeeParticleSystem(max_particles=2000)
        particles_created = particle_system.initialize_coffee_bed_confined(filter_system)

        print(f"   └─ 成功生成 {particles_created} 個咖啡顆粒")

        # 獲取顆粒數據
        particle_stats = particle_system.get_particle_statistics()
        particle_positions = particle_stats['positions']
        particle_radii = particle_stats['radii']

        print(f"   └─ 有效顆粒: {len(particle_positions)}")
        print(f"   └─ 平均半徑: {particle_stats['mean_radius']*1000:.2f} mm")
        print(f"   └─ 顆粒分佈: Z = {np.min(particle_positions[:, 2]):.1f} ~ {np.max(particle_positions[:, 2]):.1f} 格子")

    except Exception as e:
        print(f"   ❌ 顆粒生成失敗: {e}")
        particle_positions = np.array([])
        particle_radii = np.array([])

    # 生成包含顆粒的視覺化
    create_cross_section_plots_with_particles(particle_positions, particle_radii)
    create_3d_model_with_particles(particle_positions, particle_radii)

    print(f"\n✅ 幾何模型視覺化已生成 (含咖啡顆粒)")
    print(f"   └─ cross_section_view_with_particles.png - 橫截面圖 (含顆粒)")
    print(f"   └─ 3d_geometry_model_with_particles.png - 3D模型圖 (含顆粒)")

def create_cross_section_plots_with_particles(particle_positions, particle_radii):
    """創建包含咖啡顆粒的橫截面視圖"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

    # 基本參數
    center_x = config.NX * 0.5
    center_y = config.NY * 0.5
    top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
    bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
    cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
    v60_bottom_z = 5.0
    v60_top_z = v60_bottom_z + cup_height_lu
    wall_thickness = 2.0
    air_gap_lu = 0.002 / config.SCALE_LENGTH
    paper_thickness_lu = max(1.0, 0.0001 / config.SCALE_LENGTH)

    # 1. XZ橫截面 (Y = center) - 含咖啡顆粒
    ax1.set_title('XZ橫截面 (側面視圖) - 含咖啡顆粒分佈', fontsize=12)
    z_range = np.linspace(0, config.NZ, 200)

    # 繪製V60結構
    for z in z_range:
        if v60_bottom_z <= z <= v60_top_z:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            outer_radius = inner_radius + wall_thickness
            ax1.plot([center_x - outer_radius, center_x + outer_radius], [z, z], 'k-', linewidth=1, alpha=0.6)
            # V60內表面
            ax1.plot([center_x - inner_radius, center_x + inner_radius], [z, z], 'b-', linewidth=0.5, alpha=0.4)

    # 濾紙位置
    for z in z_range:
        if v60_bottom_z <= z <= v60_top_z:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            filter_outer = inner_radius - air_gap_lu
            filter_inner = filter_outer - paper_thickness_lu
            ax1.plot([center_x - filter_outer, center_x + filter_outer], [z, z], 'brown', linewidth=1, alpha=0.6)

    # 繪製咖啡顆粒 - XZ截面 (Y接近中心的顆粒)
    if len(particle_positions) > 0:
        # 選擇Y坐標接近中心的顆粒 (±5格子範圍)
        y_center_mask = np.abs(particle_positions[:, 1] - center_y) <= 5
        xz_particles = particle_positions[y_center_mask]
        xz_radii = particle_radii[y_center_mask]

        if len(xz_particles) > 0:
            # 顆粒用圓點表示，大小按真實半徑縮放
            sizes = (xz_radii / config.SCALE_LENGTH) * 30  # 視覺化縮放因子
            ax1.scatter(xz_particles[:, 0], xz_particles[:, 2],
                       s=sizes, c='saddlebrown', alpha=0.7,
                       label=f'咖啡顆粒 ({len(xz_particles)}個)')

    # V60底部封閉
    bottom_inner = bottom_radius_lu
    ax1.fill_between([center_x - bottom_inner, center_x + bottom_inner],
                     [v60_bottom_z, v60_bottom_z], [v60_bottom_z - wall_thickness, v60_bottom_z - wall_thickness],
                     color='black', alpha=0.8, label='V60底部(封閉)')

    ax1.set_xlabel('X (格子單位)')
    ax1.set_ylabel('Z (格子單位)')
    ax1.set_xlim(center_x - top_radius_lu - 15, center_x + top_radius_lu + 15)
    ax1.set_ylim(0, v60_top_z + 15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['V60外壁', 'V60內表面', '濾紙外表面', '咖啡顆粒', 'V60底部(封閉)'])

    # 2. 咖啡顆粒分佈統計
    ax2.set_title('咖啡顆粒分佈統計')
    if len(particle_positions) > 0:
        # 高度分佈直方圖
        z_coords = particle_positions[:, 2]
        ax2.hist(z_coords, bins=20, alpha=0.7, color='saddlebrown', edgecolor='black')
        ax2.axvline(np.mean(z_coords), color='red', linestyle='--',
                   label=f'平均高度: {np.mean(z_coords):.1f}')
        ax2.axvline(v60_bottom_z + 2, color='brown', linestyle='-',
                   label=f'濾紙表面: {v60_bottom_z + 2:.1f}')
        ax2.set_xlabel('Z 高度 (格子單位)')
        ax2.set_ylabel('顆粒數量')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, '無咖啡顆粒數據', ha='center', va='center', transform=ax2.transAxes)
    ax2.grid(True, alpha=0.3)

    # 3. 顆粒半徑分佈
    ax3.set_title('咖啡顆粒半徑分佈')
    if len(particle_radii) > 0:
        radii_mm = particle_radii * 1000  # 轉換為毫米
        ax3.hist(radii_mm, bins=15, alpha=0.7, color='chocolate', edgecolor='black')
        ax3.axvline(np.mean(radii_mm), color='red', linestyle='--',
                   label=f'平均: {np.mean(radii_mm):.2f}mm')
        ax3.set_xlabel('顆粒半徑 (mm)')
        ax3.set_ylabel('顆粒數量')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, '無顆粒半徑數據', ha='center', va='center', transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)

    # 4. 徑向分佈分析
    ax4.set_title('咖啡顆粒徑向分佈')
    if len(particle_positions) > 0:
        # 計算每個顆粒到中心軸的距離
        dx = particle_positions[:, 0] - center_x
        dy = particle_positions[:, 1] - center_y
        radial_distances = np.sqrt(dx**2 + dy**2)

        # 不同高度層的徑向分佈
        z_coords = particle_positions[:, 2]
        z_layers = np.linspace(np.min(z_coords), np.max(z_coords), 4)

        colors = ['red', 'green', 'blue', 'orange']
        for i, (z_low, z_high) in enumerate(zip(z_layers[:-1], z_layers[1:])):
            layer_mask = (z_coords >= z_low) & (z_coords < z_high)
            if np.any(layer_mask):
                layer_radial = radial_distances[layer_mask]
                ax4.hist(layer_radial, bins=10, alpha=0.6, color=colors[i],
                        label=f'Z: {z_low:.1f}-{z_high:.1f}')

        ax4.set_xlabel('徑向距離 (格子單位)')
        ax4.set_ylabel('顆粒數量')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, '無徑向分佈數據', ha='center', va='center', transform=ax4.transAxes)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cross_section_view_with_particles.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_model_with_particles(particle_positions, particle_radii):
    """創建包含咖啡顆粒的3D幾何模型"""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 基本參數
    center_x = config.NX * 0.5
    center_y = config.NY * 0.5
    top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
    bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
    cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
    v60_bottom_z = 5.0
    v60_top_z = v60_bottom_z + cup_height_lu
    air_gap_lu = 0.002 / config.SCALE_LENGTH

    # 創建錐形表面
    theta = np.linspace(0, 2*np.pi, 50)
    z_levels = np.linspace(v60_bottom_z, v60_top_z, 25)

    # V60外表面
    for z in z_levels[::3]:
        height_ratio = (z - v60_bottom_z) / cup_height_lu
        radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
        x_circle = center_x + radius * np.cos(theta)
        y_circle = center_y + radius * np.sin(theta)
        z_circle = np.full_like(x_circle, z)
        ax.plot(x_circle, y_circle, z_circle, 'b-', alpha=0.5, linewidth=1)

    # 濾紙表面
    for z in z_levels[::4]:
        height_ratio = (z - v60_bottom_z) / cup_height_lu
        v60_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
        filter_radius = v60_radius - air_gap_lu
        x_filter = center_x + filter_radius * np.cos(theta)
        y_filter = center_y + filter_radius * np.sin(theta)
        z_filter = np.full_like(x_filter, z)
        ax.plot(x_filter, y_filter, z_filter, 'brown', alpha=0.7, linewidth=1.5)

    # 繪製咖啡顆粒 (3D散點圖)
    if len(particle_positions) > 0:
        # 為了性能，只顯示部分顆粒
        max_display_particles = 500
        if len(particle_positions) > max_display_particles:
            # 隨機選擇顆粒
            indices = np.random.choice(len(particle_positions), max_display_particles, replace=False)
            display_positions = particle_positions[indices]
            display_radii = particle_radii[indices]
        else:
            display_positions = particle_positions
            display_radii = particle_radii

        # 顆粒大小按真實半徑縮放
        sizes = (display_radii / config.SCALE_LENGTH) * 20  # 視覺化縮放因子

        # 根據高度著色
        colors = display_positions[:, 2]  # Z坐標作為顏色

        scatter = ax.scatter(display_positions[:, 0], display_positions[:, 1], display_positions[:, 2],
                           s=sizes, c=colors, cmap='YlOrBr', alpha=0.8,
                           label=f'咖啡顆粒 ({len(display_positions)}個)')

        # 添加顏色條
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.8, label='高度 (格子單位)')

    # 垂直結構線
    for angle in np.linspace(0, 2*np.pi, 8):
        x_line = []
        y_line = []
        z_line = []
        for z in z_levels[::2]:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            x_line.append(center_x + radius * np.cos(angle))
            y_line.append(center_y + radius * np.sin(angle))
            z_line.append(z)
        ax.plot(x_line, y_line, z_line, 'b-', alpha=0.3, linewidth=0.5)

    # 設置視角和標籤
    ax.set_xlabel('X (格子單位)')
    ax.set_ylabel('Y (格子單位)')
    ax.set_zlabel('Z (格子單位)')
    ax.set_title('V60 3D幾何模型 + 咖啡顆粒分佈\n藍色: V60濾杯, 棕色: 濾紙, 散點: 咖啡顆粒')

    # 設置相等的軸比例
    max_range = max(top_radius_lu, cup_height_lu) * 1.2
    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_zlim(v60_bottom_z - 5, v60_top_z + 15)

    # 添加統計信息
    if len(particle_positions) > 0:
        coffee_bed_height = np.max(particle_positions[:, 2]) - np.min(particle_positions[:, 2])
        coffee_bed_height_cm = coffee_bed_height * config.SCALE_LENGTH * 100
        total_particles = len(particle_positions)
        avg_radius_mm = np.mean(particle_radii) * 1000

        info_text = f"""幾何參數:
• V60高度: {config.CUP_HEIGHT*100:.1f}cm
• 頂部直徑: {config.TOP_RADIUS*2*100:.1f}cm
• 底部直徑: {config.BOTTOM_RADIUS*2*100:.1f}cm
• 濾紙-V60空隙: 2.0mm

咖啡顆粒:
• 總數: {total_particles:,}個
• 平均粒徑: {avg_radius_mm:.2f}mm
• 咖啡床高度: {coffee_bed_height_cm:.1f}cm
• 底部: 完全封閉
• 出口: 域邊界"""
    else:
        info_text = f"""關鍵參數:
• V60高度: {config.CUP_HEIGHT*100:.1f}cm
• 頂部直徑: {config.TOP_RADIUS*2*100:.1f}cm
• 底部直徑: {config.BOTTOM_RADIUS*2*100:.1f}cm
• 濾紙-V60空隙: 2.0mm
• 底部: 完全封閉
• 出口: 域邊界

⚠️ 咖啡顆粒數據不可用"""

    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.savefig('3d_geometry_model_with_particles.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cross_section_plots():
    """創建橫截面視圖"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 基本參數
    center_x = config.NX * 0.5
    center_y = config.NY * 0.5
    top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
    bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
    cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
    v60_bottom_z = 5.0
    v60_top_z = v60_bottom_z + cup_height_lu
    wall_thickness = 2.0
    air_gap_lu = 0.002 / config.SCALE_LENGTH
    paper_thickness_lu = max(1.0, 0.0001 / config.SCALE_LENGTH)

    # 1. XZ橫截面 (Y = center)
    ax1.set_title('XZ plane (side view))')
    z_range = np.linspace(0, config.NZ, 200)

    # V60外壁
    for z in z_range:
        if v60_bottom_z <= z <= v60_top_z:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            outer_radius = inner_radius + wall_thickness
            ax1.plot([center_x - outer_radius, center_x + outer_radius], [z, z], 'k-', linewidth=1, alpha=0.8)
            # V60內表面
            ax1.plot([center_x - inner_radius, center_x + inner_radius], [z, z], 'b-', linewidth=0.5, alpha=0.6)

    # 濾紙位置
    for z in z_range:
        if v60_bottom_z <= z <= v60_top_z:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            filter_outer = inner_radius - air_gap_lu
            filter_inner = filter_outer - paper_thickness_lu
            ax1.plot([center_x - filter_outer, center_x + filter_outer], [z, z], 'brown', linewidth=1.5)
            ax1.plot([center_x - filter_inner, center_x + filter_inner], [z, z], 'brown', linewidth=0.8, alpha=0.7)

    # V60底部封閉
    bottom_inner = bottom_radius_lu
    ax1.fill_between([center_x - bottom_inner, center_x + bottom_inner],
                     [v60_bottom_z, v60_bottom_z], [v60_bottom_z - wall_thickness, v60_bottom_z - wall_thickness],
                     color='black', alpha=0.8, label='V60 bottom(closed)')

    ax1.set_xlabel('X (lattice)')
    ax1.set_ylabel('Z (lattice)')
    ax1.set_xlim(center_x - top_radius_lu - 10, center_x + top_radius_lu + 10)
    ax1.set_ylim(0, v60_top_z + 10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['V60 outer surface', 'V60 interior surface', 'outer surface (filter paper)', 'interior surface (filter paper)'])

    # 2. 空隙細節圖
    ax2.set_title('2mm gap detail')
    z_mid = v60_bottom_z + cup_height_lu / 2
    height_ratio = 0.5
    inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio

    x_detail = np.linspace(center_x + inner_radius - 10, center_x + inner_radius + 10, 100)

    # V60內表面
    ax2.axvline(center_x + inner_radius, color='blue', linewidth=3, label='V60 interior surface')
    # 濾紙外表面
    ax2.axvline(center_x + inner_radius - air_gap_lu, color='brown', linewidth=2, label='filter paper outer surface')
    # 濾紙內表面
    ax2.axvline(center_x + inner_radius - air_gap_lu - paper_thickness_lu, color='orange', linewidth=2, label='innerior ')

    # 標註距離
    ax2.annotate(f'2mm gap\n({air_gap_lu:.2f}lattice)',
                xy=(center_x + inner_radius - air_gap_lu/2, 0.5),
                xytext=(center_x + inner_radius - air_gap_lu/2, 0.8),
                arrowprops=dict(arrowstyle='<->', color='red'),
                ha='center', fontsize=10, color='red')

    ax2.set_xlim(center_x + inner_radius - 8, center_x + inner_radius + 5)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('X (lattice)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 尺寸標註圖
    ax3.set_title('size note')
    ax3.plot([center_x - top_radius_lu, center_x + top_radius_lu], [v60_top_z, v60_top_z], 'k-', linewidth=2)
    ax3.plot([center_x - bottom_radius_lu, center_x + bottom_radius_lu], [v60_bottom_z, v60_bottom_z], 'k-', linewidth=2)
    ax3.plot([center_x, center_x], [v60_bottom_z, v60_top_z], 'k--', linewidth=1)

    # 標註尺寸
    ax3.annotate(f'top diameter\n{config.TOP_RADIUS*2*100:.1f}cm',
                xy=(center_x, v60_top_z + 5), ha='center')
    ax3.annotate(f'height\n{config.CUP_HEIGHT*100:.1f}cm',
                xy=(center_x + top_radius_lu + 5, (v60_bottom_z + v60_top_z)/2),
                rotation=90, ha='center')
    ax3.annotate(f'bottom diameter\n{config.BOTTOM_RADIUS*2*100:.1f}cm',
                xy=(center_x, v60_bottom_z - 5), ha='center')

    ax3.set_xlim(center_x - top_radius_lu - 15, center_x + top_radius_lu + 15)
    ax3.set_ylim(v60_bottom_z - 15, v60_top_z + 15)
    ax3.grid(True, alpha=0.3)

    # 4. 流體路徑示意圖
    ax4.set_title('fluid contour')
    # 繪製V60輪廓
    z_points = np.linspace(v60_bottom_z, v60_top_z, 50)
    x_inner = []
    x_filter_inner = []

    for z in z_points:
        height_ratio = (z - v60_bottom_z) / cup_height_lu
        inner_r = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
        filter_inner_r = inner_r - air_gap_lu - paper_thickness_lu
        x_inner.append(center_x + inner_r)
        x_filter_inner.append(center_x + filter_inner_r)

    ax4.plot(x_inner, z_points, 'b-', linewidth=2, label='V60 interior wall')
    ax4.plot(x_filter_inner, z_points, 'brown', linewidth=2, label='interior surface (filter paper)')

    # 流體路徑箭頭
    for i in range(5, len(z_points), 10):
        z = z_points[i]
        x_start = x_filter_inner[i]
        x_end = x_inner[i] - 1
        ax4.arrow(x_start, z, x_end - x_start, 0,
                 head_width=1, head_length=1, fc='cyan', ec='cyan', alpha=0.7)

    # 出口標示
    ax4.annotate('outlet', xy=(config.NX-5, v60_bottom_z + cup_height_lu/3),
                xytext=(config.NX-15, v60_bottom_z + cup_height_lu/2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax4.set_xlim(center_x - 10, config.NX)
    ax4.set_ylim(v60_bottom_z - 5, v60_top_z + 5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cross_section_view.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_model():
    """創建3D幾何模型"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 基本參數
    center_x = config.NX * 0.5
    center_y = config.NY * 0.5
    top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
    bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
    cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
    v60_bottom_z = 5.0
    v60_top_z = v60_bottom_z + cup_height_lu
    air_gap_lu = 0.002 / config.SCALE_LENGTH

    # 創建錐形表面
    theta = np.linspace(0, 2*np.pi, 50)
    z_levels = np.linspace(v60_bottom_z, v60_top_z, 30)

    # V60外表面
    for z in z_levels[::3]:
        height_ratio = (z - v60_bottom_z) / cup_height_lu
        radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
        x_circle = center_x + radius * np.cos(theta)
        y_circle = center_y + radius * np.sin(theta)
        z_circle = np.full_like(x_circle, z)
        ax.plot(x_circle, y_circle, z_circle, 'b-', alpha=0.6, linewidth=1)

    # 濾紙表面
    for z in z_levels[::4]:
        height_ratio = (z - v60_bottom_z) / cup_height_lu
        v60_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
        filter_radius = v60_radius - air_gap_lu
        x_filter = center_x + filter_radius * np.cos(theta)
        y_filter = center_y + filter_radius * np.sin(theta)
        z_filter = np.full_like(x_filter, z)
        ax.plot(x_filter, y_filter, z_filter, 'brown', alpha=0.8, linewidth=1.5)

    # 垂直結構線
    for angle in np.linspace(0, 2*np.pi, 8):
        x_line = []
        y_line = []
        z_line = []
        for z in z_levels:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            x_line.append(center_x + radius * np.cos(angle))
            y_line.append(center_y + radius * np.sin(angle))
            z_line.append(z)
        ax.plot(x_line, y_line, z_line, 'b-', alpha=0.4, linewidth=0.5)

        # 濾紙垂直線
        x_filter_line = []
        y_filter_line = []
        for z in z_levels:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            v60_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            filter_radius = v60_radius - air_gap_lu
            x_filter_line.append(center_x + filter_radius * np.cos(angle))
            y_filter_line.append(center_y + filter_radius * np.sin(angle))
        ax.plot(x_filter_line, y_filter_line, z_line, 'brown', alpha=0.6, linewidth=0.8)

    # 設置視角和標籤
    ax.set_xlabel('X (lattice)')
    ax.set_ylabel('Y (lattice)')
    ax.set_zlabel('Z (lattice)')
    ax.set_title('V60 3D model\nblue: V60, brown: filter paper')

    # 設置相等的軸比例
    max_range = max(top_radius_lu, cup_height_lu) * 1.2
    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_zlim(v60_bottom_z - 5, v60_top_z + 5)

    # 添加說明文字
    info_text = f"""coefficient:
• height: {config.CUP_HEIGHT*100:.1f}cm
• top diameter: {config.TOP_RADIUS*2*100:.1f}cm
• bottom diameter: {config.BOTTOM_RADIUS*2*100:.1f}cm
• filter-paper-V60 gap: 2.0mm
• bottom: completely closed
• outlet: domain boundary"""

    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig('3d_geometry_model.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_v60_geometry_with_particles()
