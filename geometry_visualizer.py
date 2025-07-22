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

# 設置matplotlib中文字體支援
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

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

    print("=== V60 Geometry Model Detailed Specs (with Coffee Particles) ===")
    print(f"\n📐 Basic Dimensions:")
    print(f"   Computational Domain: {config.NX} × {config.NY} × {config.NZ} lattice units")
    print(f"   Physical Domain: {config.NX * config.SCALE_LENGTH * 100:.1f} × {config.NY * config.SCALE_LENGTH * 100:.1f} × {config.NZ * config.SCALE_LENGTH * 100:.1f} cm")
    print(f"   Grid Resolution: {config.SCALE_LENGTH * 1000:.3f} mm/lattice")

    print(f"\n🍵 V60 Dripper:")
    print(f"   Height: {config.CUP_HEIGHT * 100:.1f} cm ({cup_height_lu:.1f} lattice)")
    print(f"   Top Diameter: {config.TOP_RADIUS * 2 * 100:.1f} cm ({top_radius_lu * 2:.1f} lattice)")
    print(f"   Bottom Diameter: {config.BOTTOM_RADIUS * 2 * 100:.1f} cm ({bottom_radius_lu * 2:.1f} lattice)")
    print(f"   Wall Thickness: {wall_thickness * config.SCALE_LENGTH * 1000:.1f} mm ({wall_thickness:.1f} lattice)")
    print(f"   Position: Z = {v60_bottom_z:.1f} ~ {v60_top_z:.1f} lattice")

    print(f"\n📄 Filter Paper System:")
    print(f"   Paper Thickness: 0.1 mm ({paper_thickness_lu:.1f} lattice)")
    print(f"   V60-Paper Gap: 2.0 mm ({air_gap_lu:.2f} lattice)")
    print(f"   Shape: Complete cone (no flat bottom)")
    print(f"   Coverage: Full V60 interior surface")

    print(f"\n🌊 Fluid Path:")
    print(f"   Inlet: V60 top water inlet area")
    print(f"   Through: Coffee → Filter Paper → 2mm Gap")
    print(f"   Outlet: V60 bottom opening (correct design)")
    print(f"   V60 Bottom: Open hole (diameter: {config.BOTTOM_RADIUS*2*100:.1f}cm)")

    # Initialize system to generate coffee particles
    print(f"\n☕ Generating Coffee Particle Distribution...")
    try:
        # Initialize necessary systems - use unified initialization
        from init import initialize_taichi_once
        initialize_taichi_once()

        lbm = LBMSolver()
        filter_system = FilterPaperSystem(lbm)
        filter_system.initialize_filter_geometry()

        particle_system = CoffeeParticleSystem(max_particles=2000)
        particles_created = particle_system.initialize_coffee_bed_confined(filter_system)

        print(f"   └─ Successfully generated {particles_created} coffee particles")

        # Get particle data
        particle_stats = particle_system.get_particle_statistics()
        particle_positions = particle_stats['positions']
        particle_radii = particle_stats['radii']

        print(f"   └─ Valid particles: {len(particle_positions)}")
        print(f"   └─ Average radius: {particle_stats['mean_radius']*1000:.2f} mm")
        print(f"   └─ Particle distribution: Z = {np.min(particle_positions[:, 2]):.1f} ~ {np.max(particle_positions[:, 2]):.1f} lattice")

    except Exception as e:
        print(f"   ❌ Particle generation failed: {e}")
        particle_positions = np.array([])
        particle_radii = np.array([])

    # Generate visualization with particles
    create_cross_section_plots_with_particles(particle_positions, particle_radii)
    create_3d_model_with_particles(particle_positions, particle_radii)

    print(f"\n✅ Geometry Model Visualization Generated (with Coffee Particles)")
    print(f"   └─ cross_section_view_with_particles.png - Cross Section View (with particles)")
    print(f"   └─ 3d_geometry_model_with_particles.png - 3D Model View (with particles)")

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
    ax1.set_title('XZ cross section (side view)', fontsize=12)
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

    # Filter paper position
    for z in z_range:
        if v60_bottom_z <= z <= v60_top_z:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            filter_outer = inner_radius - air_gap_lu
            filter_inner = filter_outer - paper_thickness_lu
            ax1.plot([center_x - filter_outer, center_x + filter_outer], [z, z], 'brown', linewidth=1, alpha=0.6)

    # Draw coffee particles - XZ cross section (particles near center Y)
    if len(particle_positions) > 0:
        # Select particles with Y coordinate near center (±5 lattice range)
        y_center_mask = np.abs(particle_positions[:, 1] - center_y) <= 5
        xz_particles = particle_positions[y_center_mask]
        xz_radii = particle_radii[y_center_mask]

        if len(xz_particles) > 0:
            # Particles represented as dots, size scaled by real radius
            sizes = (xz_radii / config.SCALE_LENGTH) * 30  # Visual scaling factor
            ax1.scatter(xz_particles[:, 0], xz_particles[:, 2],
                       s=sizes, c='saddlebrown', alpha=0.7,
                       label=f'coffee particles ({len(xz_particles)}#)')

    # V60 bottom opening (not sealed)
    # Draw the rim of the bottom opening instead of solid fill
    bottom_inner = bottom_radius_lu
    theta_bottom = np.linspace(0, 2*np.pi, 50)
    x_bottom_rim = center_x + bottom_inner * np.cos(theta_bottom)
    z_bottom_rim = np.full_like(x_bottom_rim, v60_bottom_z)
    ax1.plot(x_bottom_rim, z_bottom_rim, 'k-', linewidth=3, 
             label='V60 bottom opening rim')
    
    # Show the opening with arrows
    ax1.annotate('Bottom Opening\n(Coffee Exit)', 
                xy=(center_x, v60_bottom_z-2), 
                xytext=(center_x+15, v60_bottom_z-10),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', ha='center')

    ax1.set_xlabel('X (lattice)')
    ax1.set_ylabel('Z (lattice)')
    ax1.set_xlim(center_x - top_radius_lu - 15, center_x + top_radius_lu + 15)
    ax1.set_ylim(0, v60_top_z + 15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['V60 outer wall', 'V60 interior surface', 'filter paper outer surface', 'coffee particles', 'V60 bottom opening rim'])

    # 2. Coffee particle distribution statistics
    ax2.set_title('coffee particle distribution statistics')
    if len(particle_positions) > 0:
        # Height distribution histogram
        z_coords = particle_positions[:, 2]
        ax2.hist(z_coords, bins=20, alpha=0.7, color='saddlebrown', edgecolor='black')
        ax2.axvline(np.mean(z_coords), color='red', linestyle='--',
                   label=f'average height: {np.mean(z_coords):.1f}')
        ax2.axvline(v60_bottom_z + 2, color='brown', linestyle='-',
                   label=f'filter paper surface: {v60_bottom_z + 2:.1f}')
        ax2.set_xlabel('Z height (grid units)')
        ax2.set_ylabel('particle count')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'no coffee particle data', ha='center', va='center', transform=ax2.transAxes)
    ax2.grid(True, alpha=0.3)

    # 3. Particle radius distribution
    ax3.set_title('coffee particle radius distribution')
    if len(particle_radii) > 0:
        radii_mm = particle_radii * 1000  # Convert to millimeters
        ax3.hist(radii_mm, bins=15, alpha=0.7, color='chocolate', edgecolor='black')
        ax3.axvline(np.mean(radii_mm), color='red', linestyle='--',
                   label=f'Average: {np.mean(radii_mm):.2f}mm')
        ax3.set_xlabel('particle radius (mm)')
        ax3.set_ylabel('particle count')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No particle radius data', ha='center', va='center', transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)

    # 4. Radial distribution analysis
    ax4.set_title('coffee particle radial distribution')
    if len(particle_positions) > 0:
        # Calculate distance from each particle to central axis
        dx = particle_positions[:, 0] - center_x
        dy = particle_positions[:, 1] - center_y
        radial_distances = np.sqrt(dx**2 + dy**2)

        # Radial distribution at different height layers
        z_coords = particle_positions[:, 2]
        z_layers = np.linspace(np.min(z_coords), np.max(z_coords), 4)

        colors = ['red', 'green', 'blue', 'orange']
        for i, (z_low, z_high) in enumerate(zip(z_layers[:-1], z_layers[1:])):
            layer_mask = (z_coords >= z_low) & (z_coords < z_high)
            if np.any(layer_mask):
                layer_radial = radial_distances[layer_mask]
                ax4.hist(layer_radial, bins=10, alpha=0.6, color=colors[i],
                        label=f'Z: {z_low:.1f}-{z_high:.1f}')

        ax4.set_xlabel('radial distance (lattice)')
        ax4.set_ylabel('# of particles')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No radial distribution data', ha='center', va='center', transform=ax4.transAxes)
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

    # Draw coffee particles (3D scatter plot)
    if len(particle_positions) > 0:
        # For performance, only display some particles
        max_display_particles = 500
        if len(particle_positions) > max_display_particles:
            # Randomly select particles
            indices = np.random.choice(len(particle_positions), max_display_particles, replace=False)
            display_positions = particle_positions[indices]
            display_radii = particle_radii[indices]
        else:
            display_positions = particle_positions
            display_radii = particle_radii

        # Particle size scaled by real radius
        sizes = (display_radii / config.SCALE_LENGTH) * 20  # Visual scaling factor

        # Color by height
        colors = display_positions[:, 2]  # Z coordinate as color

        scatter = ax.scatter(display_positions[:, 0], display_positions[:, 1], display_positions[:, 2],
                           s=sizes, c=colors, cmap='YlOrBr', alpha=0.8,
                           label=f'coffee particles ({len(display_positions)}#)')

        # Add colorbar
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.8, label='height (lattice)')

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
    ax.set_xlabel('X (lattice units)')
    ax.set_ylabel('Y (lattice units)')
    ax.set_zlabel('Z (lattice units)')
    ax.set_title('V60 3D Geometry Model + Coffee Particle Distribution\nBlue: V60 Dripper, Brown: Filter Paper, Dots: Coffee Particles')

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

        info_text = f"""Geometry Parameters:
• V60 Height: {config.CUP_HEIGHT*100:.1f}cm
• Top Diameter: {config.TOP_RADIUS*2*100:.1f}cm
• Bottom Diameter: {config.BOTTOM_RADIUS*2*100:.1f}cm
• Paper-V60 Gap: 2.0mm

Coffee Particles:
• Total: {total_particles:,} particles
• Avg Diameter: {avg_radius_mm:.2f}mm
• Coffee Bed Height: {coffee_bed_height_cm:.1f}cm
• Bottom: Open hole ({config.BOTTOM_RADIUS*2*100:.1f}cm diameter)
• Outlet: Direct flow through bottom opening"""
    else:
        info_text = f"""Key Parameters:
• V60 Height: {config.CUP_HEIGHT*100:.1f}cm
• Top Diameter: {config.TOP_RADIUS*2*100:.1f}cm
• Bottom Diameter: {config.BOTTOM_RADIUS*2*100:.1f}cm
• Paper-V60 Gap: 2.0mm
• Bottom: Completely sealed
• Outlet: Domain boundary

⚠️ Coffee particle data unavailable"""

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

    # 1. XZ cross-section (Y = center)
    ax1.set_title('XZ plane (side view)')
    z_range = np.linspace(0, config.NZ, 200)

    # Draw V60 structure
    for z in z_range:
        if v60_bottom_z <= z <= v60_top_z:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            outer_radius = inner_radius + wall_thickness
            ax1.plot([center_x - outer_radius, center_x + outer_radius], [z, z], 'k-', linewidth=1, alpha=0.8)
            # V60 interior surface
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

    # V60底部開放設計（正確）
    # 繪製底部開口邊緣而非實心
    bottom_inner = bottom_radius_lu
    # 左邊緣線
    ax1.plot([center_x - bottom_inner, center_x - bottom_inner], 
             [v60_bottom_z - wall_thickness, v60_bottom_z], 'k-', linewidth=3)
    # 右邊緣線  
    ax1.plot([center_x + bottom_inner, center_x + bottom_inner], 
             [v60_bottom_z - wall_thickness, v60_bottom_z], 'k-', linewidth=3)
    # 底部開口標記
    ax1.annotate('底部開口\n(咖啡出口)', 
                xy=(center_x, v60_bottom_z-1), 
                xytext=(center_x+20, v60_bottom_z-8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', ha='center')

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

    # Annotate distances
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

    # Annotate dimensions
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
    # Draw V60 contour
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

    # Fluid path arrows
    for i in range(5, len(z_points), 10):
        z = z_points[i]
        x_start = x_filter_inner[i]
        x_end = x_inner[i] - 1
        ax4.arrow(x_start, z, x_end - x_start, 0,
                 head_width=1, head_length=1, fc='cyan', ec='cyan', alpha=0.7)

    # Outlet marking - now points to V60 bottom opening
    ax4.annotate('bottom opening outlet', xy=(center_x, v60_bottom_z),
                xytext=(center_x+15, v60_bottom_z-10),
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
• bottom: open hole ({config.BOTTOM_RADIUS*2*100:.1f}cm)
• outlet: V60 bottom opening"""

    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig('3d_geometry_model.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_v60_geometry_with_particles()
