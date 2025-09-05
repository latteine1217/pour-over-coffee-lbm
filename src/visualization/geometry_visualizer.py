# geometry_visualizer.py
"""
V60幾何模型專業視覺化工具 - 工業級CFD分析版本
輸出高質量的濾杯和濾紙幾何配置圖表，支援智能報告目錄管理
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from scipy import stats
import config as config
from src.physics.coffee_particles import CoffeeParticleSystem
from src.physics.filter_paper import FilterPaperSystem
# Note: LBMSolver import removed - geometry visualizer doesn't need core solver

# 設置專業級matplotlib配置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class GeometryVisualizer:
    """V60幾何專業視覺化系統"""
    
    def __init__(self):
        """初始化視覺化系統"""
        self.setup_output_directory()
        self.setup_professional_style()
        
    def setup_output_directory(self):
        """設置智能輸出目錄"""
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = f"report/{self.session_timestamp}"
        os.makedirs(f"{self.report_dir}/images", exist_ok=True)
        os.makedirs(f"{self.report_dir}/geometry", exist_ok=True)
        
    def setup_professional_style(self):
        """設置專業圖表樣式"""
        plt.style.use('default')
        self.color_scheme = {
            'v60_outer': '#1f77b4',
            'v60_inner': '#ff7f0e', 
            'filter_paper': '#8B4513',
            'coffee_particles': '#A0522D',
            'fluid_flow': '#17becf',
            'annotations': '#2E8B57',
            'grid': '#D3D3D3'
        }
        
    def get_output_path(self, filename, subdir='geometry'):
        """獲取輸出路徑"""
        return f"{self.report_dir}/{subdir}/{filename}"

    def visualize_v60_geometry_with_particles(self):
        """生成包含咖啡顆粒分佈的V60幾何模型專業視覺化"""

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

        print("=" * 80)
        print("🔬 V60 幾何模型專業分析報告")
        print("=" * 80)
        print(f"\n📏 計算域規格:")
        print(f"   └─ 網格: {config.NX} × {config.NY} × {config.NZ} 格子單位")
        print(f"   └─ 物理域: {config.NX * config.SCALE_LENGTH * 100:.1f} × {config.NY * config.SCALE_LENGTH * 100:.1f} × {config.NZ * config.SCALE_LENGTH * 100:.1f} cm")
        print(f"   └─ 解析度: {config.SCALE_LENGTH * 1000:.3f} mm/格子")

        print(f"\n🍵 V60濾杯規格:")
        print(f"   └─ 高度: {config.CUP_HEIGHT * 100:.1f} cm ({cup_height_lu:.1f} 格子)")
        print(f"   └─ 上口直徑: {config.TOP_RADIUS * 2 * 100:.1f} cm ({top_radius_lu * 2:.1f} 格子)")
        print(f"   └─ 下口直徑: {config.BOTTOM_RADIUS * 2 * 100:.1f} cm ({bottom_radius_lu * 2:.1f} 格子)")
        print(f"   └─ 壁厚: {wall_thickness * config.SCALE_LENGTH * 1000:.1f} mm ({wall_thickness:.1f} 格子)")
        print(f"   └─ 位置: Z = {v60_bottom_z:.1f} ~ {v60_top_z:.1f} 格子")

        print(f"\n📄 濾紙系統規格:")
        print(f"   └─ 濾紙厚度: 0.1 mm ({paper_thickness_lu:.1f} 格子)")
        print(f"   └─ V60-濾紙間隙: 2.0 mm ({air_gap_lu:.2f} 格子)")
        print(f"   └─ 形狀: 完整圓錐形（無平底）")
        print(f"   └─ 覆蓋: V60內表面完整覆蓋")

        print(f"\n🌊 流體路徑:")
        print(f"   └─ 入口: V60頂部注水區域")
        print(f"   └─ 路徑: 咖啡床 → 濾紙 → 2mm間隙")
        print(f"   └─ 出口: V60底部開口（正確設計）")
        print(f"   └─ 底口: 開放孔洞（直徑: {config.BOTTOM_RADIUS*2*100:.1f}cm）")

        # 初始化系統以生成咖啡顆粒
        print(f"\n☕ 生成咖啡顆粒分佈...")
        try:
            # 使用統一初始化
            from config.init import initialize_taichi_once
            initialize_taichi_once()

            # FilterPaperSystem初始化 (不需要完整LBM求解器)
            filter_system = FilterPaperSystem(lbm_solver=None)  # 傳入None作為佔位符
            filter_system.initialize_filter_geometry()

            particle_system = CoffeeParticleSystem(max_particles=2000)
            particles_created = particle_system.initialize_coffee_bed_confined(filter_system)

            print(f"   └─ 成功生成 {particles_created} 顆咖啡顆粒")

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

        # 生成專業視覺化
        print(f"\n🎨 生成專業視覺化圖表...")
        self.create_professional_cross_section_analysis(particle_positions, particle_radii)
        self.create_professional_3d_model(particle_positions, particle_radii)
        self.create_engineering_drawings()
        self.create_coffee_particle_distribution_analysis(particle_positions, particle_radii)
        self.create_particle_size_distribution_analysis(particle_positions, particle_radii)

        print(f"\n✅ 專業幾何分析報告生成完成")
        print(f"   └─ 報告目錄: {self.report_dir}")
        print(f"   └─ 專業橫截面分析: geometry/professional_cross_section_analysis.png")
        print(f"   └─ 3D工程模型: geometry/professional_3d_geometry_model.png")
        print(f"   └─ 工程製圖: geometry/engineering_drawings.png")
        print(f"   └─ 咖啡粒子分布分析: geometry/coffee_particle_distribution.png")
        print(f"   └─ 顆粒大小分佈分析: geometry/particle_size_distribution.png")
        
        return self.report_dir

    def create_professional_cross_section_analysis(self, particle_positions, particle_radii):
        """創建專業級橫截面分析圖表"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
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

        # 1. 主要XZ橫截面（左上，跨兩格）
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_title('V60 Cross-Section Analysis (XZ Plane)', fontsize=16, fontweight='bold')
        
        z_range = np.linspace(0, config.NZ, 200)

        # 繪製V60結構
        for z in z_range:
            if v60_bottom_z <= z <= v60_top_z:
                height_ratio = (z - v60_bottom_z) / cup_height_lu
                inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                outer_radius = inner_radius + wall_thickness
                ax1.plot([center_x - outer_radius, center_x + outer_radius], [z, z], 
                        color=self.color_scheme['v60_outer'], linewidth=1.5, alpha=0.8)
                # V60內表面
                ax1.plot([center_x - inner_radius, center_x + inner_radius], [z, z], 
                        color=self.color_scheme['v60_inner'], linewidth=1, alpha=0.6)

        # 濾紙位置
        for z in z_range:
            if v60_bottom_z <= z <= v60_top_z:
                height_ratio = (z - v60_bottom_z) / cup_height_lu
                inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                filter_outer = inner_radius - air_gap_lu
                ax1.plot([center_x - filter_outer, center_x + filter_outer], [z, z], 
                        color=self.color_scheme['filter_paper'], linewidth=2, alpha=0.8)

        # 繪製咖啡顆粒
        if len(particle_positions) > 0:
            y_center_mask = np.abs(particle_positions[:, 1] - center_y) <= 5
            xz_particles = particle_positions[y_center_mask]
            xz_radii = particle_radii[y_center_mask]

            if len(xz_particles) > 0:
                sizes = (xz_radii / config.SCALE_LENGTH) * 40
                ax1.scatter(xz_particles[:, 0], xz_particles[:, 2],
                           s=sizes, c=self.color_scheme['coffee_particles'], alpha=0.7,
                           label=f'Coffee Particles ({len(xz_particles)})')

        # V60底部開口標記
        ax1.annotate('Bottom Opening\n(Coffee Exit)', 
                    xy=(center_x, v60_bottom_z-2), 
                    xytext=(center_x+20, v60_bottom_z-12),
                    arrowprops=dict(arrowstyle='->', color=self.color_scheme['annotations'], lw=2),
                    fontsize=12, color=self.color_scheme['annotations'], ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax1.set_xlabel('X Position (Lattice Units)', fontsize=12)
        ax1.set_ylabel('Z Position (Lattice Units)', fontsize=12)
        ax1.set_xlim(center_x - top_radius_lu - 20, center_x + top_radius_lu + 20)
        ax1.set_ylim(0, v60_top_z + 20)
        ax1.legend(fontsize=10, loc='upper right')

        # 2. 咖啡顆粒高度分佈（右上）
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title('Particle Height Distribution', fontsize=14, fontweight='bold')
        if len(particle_positions) > 0:
            z_coords = particle_positions[:, 2]
            n, bins, patches = ax2.hist(z_coords, bins=20, alpha=0.7, 
                                       color=self.color_scheme['coffee_particles'], 
                                       edgecolor='black', linewidth=0.5)
            ax2.axvline(np.mean(z_coords), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(z_coords):.1f}')
            ax2.axvline(v60_bottom_z + 2, color=self.color_scheme['filter_paper'], 
                       linestyle='-', linewidth=2,
                       label=f'Filter Surface: {v60_bottom_z + 2:.1f}')
            ax2.set_xlabel('Z Height (Lattice Units)')
            ax2.set_ylabel('Particle Count')
            ax2.legend(fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No Particle Data\nAvailable', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)

        # 3. 幾何尺寸標註（左中）
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title('Engineering Dimensions', fontsize=14, fontweight='bold')
        
        # 繪製主要尺寸線
        ax3.plot([center_x - top_radius_lu, center_x + top_radius_lu], [v60_top_z, v60_top_z], 
                'k-', linewidth=3)
        ax3.plot([center_x - bottom_radius_lu, center_x + bottom_radius_lu], [v60_bottom_z, v60_bottom_z], 
                'k-', linewidth=3)
        ax3.plot([center_x, center_x], [v60_bottom_z, v60_top_z], 'k--', linewidth=2)

        # 尺寸標註
        ax3.annotate(f'Top Ø: {config.TOP_RADIUS*2*100:.1f}cm',
                    xy=(center_x, v60_top_z + 8), ha='center', fontsize=11, fontweight='bold')
        ax3.annotate(f'Height:\n{config.CUP_HEIGHT*100:.1f}cm',
                    xy=(center_x + top_radius_lu + 8, (v60_bottom_z + v60_top_z)/2),
                    rotation=90, ha='center', fontsize=11, fontweight='bold')
        ax3.annotate(f'Bottom Ø: {config.BOTTOM_RADIUS*2*100:.1f}cm',
                    xy=(center_x, v60_bottom_z - 8), ha='center', fontsize=11, fontweight='bold')

        ax3.set_xlim(center_x - top_radius_lu - 25, center_x + top_radius_lu + 25)
        ax3.set_ylim(v60_bottom_z - 20, v60_top_z + 20)
        ax3.set_aspect('equal')

        # 4. 間隙細節分析（中中）
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('Gap Analysis (2mm Air Gap)', fontsize=14, fontweight='bold')
        
        z_mid = v60_bottom_z + cup_height_lu / 2
        height_ratio = 0.5
        inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio

        # V60內表面
        ax4.axvline(center_x + inner_radius, color=self.color_scheme['v60_inner'], 
                   linewidth=4, label='V60 Inner Wall')
        # 濾紙外表面
        ax4.axvline(center_x + inner_radius - air_gap_lu, color=self.color_scheme['filter_paper'], 
                   linewidth=3, label='Filter Paper Outer')
        # 濾紙內表面
        ax4.axvline(center_x + inner_radius - air_gap_lu - paper_thickness_lu, 
                   color='orange', linewidth=2, label='Filter Paper Inner')

        # 標註間隙
        gap_center = center_x + inner_radius - air_gap_lu/2
        ax4.annotate(f'2mm Gap\n({air_gap_lu:.2f} lattice)',
                    xy=(gap_center, 0.5), xytext=(gap_center, 0.8),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2),
                    ha='center', fontsize=10, color='red', fontweight='bold')

        ax4.set_xlim(center_x + inner_radius - 10, center_x + inner_radius + 8)
        ax4.set_ylim(0, 1)
        ax4.set_xlabel('X Position (Lattice Units)')
        ax4.legend(fontsize=9)

        # 5. 顆粒半徑分佈（右中）
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_title('Particle Size Distribution', fontsize=14, fontweight='bold')
        if len(particle_radii) > 0:
            radii_mm = particle_radii * 1000
            n, bins, patches = ax5.hist(radii_mm, bins=15, alpha=0.7, 
                                       color='chocolate', edgecolor='black', linewidth=0.5)
            ax5.axvline(np.mean(radii_mm), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(radii_mm):.2f}mm')
            ax5.set_xlabel('Particle Radius (mm)')
            ax5.set_ylabel('Count')
            ax5.legend(fontsize=9)
        else:
            ax5.text(0.5, 0.5, 'No Particle\nRadius Data', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)

        # 6. 流體路徑分析（左下）
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.set_title('Fluid Flow Path Analysis', fontsize=14, fontweight='bold')
        
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

        ax6.plot(x_inner, z_points, color=self.color_scheme['v60_inner'], 
                linewidth=3, label='V60 Inner Wall')
        ax6.plot(x_filter_inner, z_points, color=self.color_scheme['filter_paper'], 
                linewidth=3, label='Filter Inner Surface')

        # 流體路徑箭頭
        for i in range(5, len(z_points), 10):
            z = z_points[i]
            x_start = x_filter_inner[i]
            x_end = x_inner[i] - 2
            if x_end > x_start:
                ax6.arrow(x_start, z, x_end - x_start, 0,
                         head_width=2, head_length=2, fc=self.color_scheme['fluid_flow'], 
                         ec=self.color_scheme['fluid_flow'], alpha=0.8)

        ax6.set_xlim(center_x - 15, center_x + top_radius_lu + 10)
        ax6.set_ylim(v60_bottom_z - 5, v60_top_z + 5)
        ax6.legend(fontsize=9)
        ax6.set_xlabel('X Position (Lattice Units)')
        ax6.set_ylabel('Z Position (Lattice Units)')

        # 7. 徑向分佈分析（中下）
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.set_title('Radial Particle Distribution', fontsize=14, fontweight='bold')
        if len(particle_positions) > 0:
            dx = particle_positions[:, 0] - center_x
            dy = particle_positions[:, 1] - center_y
            radial_distances = np.sqrt(dx**2 + dy**2)

            z_coords = particle_positions[:, 2]
            z_layers = np.linspace(np.min(z_coords), np.max(z_coords), 4)

            colors = ['red', 'green', 'blue', 'orange']
            for i, (z_low, z_high) in enumerate(zip(z_layers[:-1], z_layers[1:])):
                layer_mask = (z_coords >= z_low) & (z_coords < z_high)
                if np.any(layer_mask):
                    layer_radial = radial_distances[layer_mask]
                    ax7.hist(layer_radial, bins=10, alpha=0.6, color=colors[i],
                            label=f'Z: {z_low:.1f}-{z_high:.1f}')

            ax7.set_xlabel('Radial Distance (Lattice Units)')
            ax7.set_ylabel('Particle Count')
            ax7.legend(fontsize=9)
        else:
            ax7.text(0.5, 0.5, 'No Radial\nDistribution Data', ha='center', va='center', 
                    transform=ax7.transAxes, fontsize=12)

        # 8. 統計摘要（右下）
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.set_title('Geometry Statistics', fontsize=14, fontweight='bold')
        ax8.axis('off')
        if len(particle_positions) > 0:
            coffee_bed_height = np.max(particle_positions[:, 2]) - np.min(particle_positions[:, 2])
            coffee_bed_height_cm = coffee_bed_height * config.SCALE_LENGTH * 100
            total_particles = len(particle_positions)
            avg_radius_mm = np.mean(particle_radii) * 1000

            stats_text = f"""Geometry Statistics

V60 Specifications:
• Height: {config.CUP_HEIGHT*100:.1f} cm
• Top Diameter: {config.TOP_RADIUS*2*100:.1f} cm  
• Bottom Diameter: {config.BOTTOM_RADIUS*2*100:.1f} cm
• Wall Thickness: {wall_thickness * config.SCALE_LENGTH * 1000:.1f} mm

Filter Paper:
• Paper-V60 Gap: 2.0 mm
• Paper Thickness: 0.1 mm
• Shape: Complete cone

Coffee Bed:
• Total Particles: {total_particles:,}
• Avg Diameter: {avg_radius_mm:.2f} mm
• Bed Height: {coffee_bed_height_cm:.1f} cm
• Packing Density: {total_particles/coffee_bed_height:.0f} particles/lattice

Flow Path:
• Inlet: Top opening
• Outlet: Bottom opening
• Gap Flow: 2mm drainage space"""
        else:
            stats_text = f"""Geometry Statistics

V60 Specifications:
• Height: {config.CUP_HEIGHT*100:.1f} cm
• Top Diameter: {config.TOP_RADIUS*2*100:.1f} cm
• Bottom Diameter: {config.BOTTOM_RADIUS*2*100:.1f} cm

Filter Paper:
• Paper-V60 Gap: 2.0 mm
• Shape: Complete cone

Coffee Particles:
• Data unavailable
• Particle generation failed"""

        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9))

        plt.suptitle('V60 Professional Cross-Section Analysis', fontsize=20, fontweight='bold', y=0.98)
        plt.savefig(self.get_output_path('professional_cross_section_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_professional_3d_model(self, particle_positions, particle_radii):
        """創建專業級3D幾何模型"""
        fig = plt.figure(figsize=(16, 12))
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
            ax.plot(x_circle, y_circle, z_circle, color=self.color_scheme['v60_outer'], 
                   alpha=0.6, linewidth=1.5)

        # 濾紙表面
        for z in z_levels[::4]:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            v60_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            filter_radius = v60_radius - air_gap_lu
            x_filter = center_x + filter_radius * np.cos(theta)
            y_filter = center_y + filter_radius * np.sin(theta)
            z_filter = np.full_like(x_filter, z)
            ax.plot(x_filter, y_filter, z_filter, color=self.color_scheme['filter_paper'], 
                   alpha=0.8, linewidth=2)

        # 繪製咖啡顆粒
        if len(particle_positions) > 0:
            max_display_particles = 800
            if len(particle_positions) > max_display_particles:
                indices = np.random.choice(len(particle_positions), max_display_particles, replace=False)
                display_positions = particle_positions[indices]
                display_radii = particle_radii[indices]
            else:
                display_positions = particle_positions
                display_radii = particle_radii

            sizes = (display_radii / config.SCALE_LENGTH) * 25
            colors = display_positions[:, 2]

            scatter = ax.scatter(display_positions[:, 0], display_positions[:, 1], display_positions[:, 2],
                               s=sizes, c=colors, cmap='YlOrBr', alpha=0.8,
                               label=f'Coffee Particles ({len(display_positions)})')
            
            plt.colorbar(scatter, ax=ax, shrink=0.7, label='Height (Lattice Units)')

        # 垂直結構線
        for angle in np.linspace(0, 2*np.pi, 8):
            x_line, y_line, z_line = [], [], []
            for z in z_levels[::2]:
                height_ratio = (z - v60_bottom_z) / cup_height_lu
                radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                x_line.append(center_x + radius * np.cos(angle))
                y_line.append(center_y + radius * np.sin(angle))
                z_line.append(z)
            ax.plot(x_line, y_line, z_line, color=self.color_scheme['v60_outer'], 
                   alpha=0.4, linewidth=0.8)

        # 設置視角和標籤
        ax.set_xlabel('X Position (Lattice Units)', fontsize=12)
        ax.set_ylabel('Y Position (Lattice Units)', fontsize=12)
        ax.set_zlabel('Z Position (Lattice Units)', fontsize=12)
        ax.set_title('V60 Professional 3D Geometry Model\nwith Coffee Particle Distribution', 
                    fontsize=16, fontweight='bold', pad=20)

        # 設置相等的軸比例
        max_range = max(top_radius_lu, cup_height_lu) * 1.3
        ax.set_xlim(center_x - max_range, center_x + max_range)
        ax.set_ylim(center_y - max_range, center_y + max_range)
        ax.set_zlim(v60_bottom_z - 10, v60_top_z + 20)

        # 添加專業統計信息
        if len(particle_positions) > 0:
            coffee_bed_height = np.max(particle_positions[:, 2]) - np.min(particle_positions[:, 2])
            coffee_bed_height_cm = coffee_bed_height * config.SCALE_LENGTH * 100
            total_particles = len(particle_positions)
            avg_radius_mm = np.mean(particle_radii) * 1000

            info_text = f"""Professional V60 Analysis:
• V60 Height: {config.CUP_HEIGHT*100:.1f} cm
• Top Diameter: {config.TOP_RADIUS*2*100:.1f} cm
• Bottom Diameter: {config.BOTTOM_RADIUS*2*100:.1f} cm
• Paper-V60 Gap: 2.0 mm

Coffee Bed Statistics:
• Total Particles: {total_particles:,}
• Average Diameter: {avg_radius_mm:.2f} mm
• Bed Height: {coffee_bed_height_cm:.1f} cm
• Bottom Opening: {config.BOTTOM_RADIUS*2*100:.1f} cm diameter
• Flow Path: Direct drainage through bottom"""
        else:
            info_text = f"""V60 Geometry Specifications:
• Height: {config.CUP_HEIGHT*100:.1f} cm
• Top Diameter: {config.TOP_RADIUS*2*100:.1f} cm
• Bottom Diameter: {config.BOTTOM_RADIUS*2*100:.1f} cm
• Paper-V60 Gap: 2.0 mm
• Bottom: Open drainage hole

⚠️ Coffee particle data unavailable"""

        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))

        plt.savefig(self.get_output_path('professional_3d_geometry_model.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_engineering_drawings(self):
        """創建工程製圖"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('V60 Engineering Drawings - Technical Specifications', 
                    fontsize=18, fontweight='bold', y=0.95)

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

        # 1. 主要橫截面圖 (XZ平面)
        ax1.set_title('Side View (XZ Cross-Section)', fontsize=14, fontweight='bold')
        z_range = np.linspace(0, config.NZ, 200)

        # 繪製V60結構
        for z in z_range:
            if v60_bottom_z <= z <= v60_top_z:
                height_ratio = (z - v60_bottom_z) / cup_height_lu
                inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                outer_radius = inner_radius + wall_thickness
                ax1.plot([center_x - outer_radius, center_x + outer_radius], [z, z], 
                        color=self.color_scheme['v60_outer'], linewidth=1.5, alpha=0.8)
                ax1.plot([center_x - inner_radius, center_x + inner_radius], [z, z], 
                        color=self.color_scheme['v60_inner'], linewidth=1, alpha=0.6)

        # 濾紙位置
        for z in z_range:
            if v60_bottom_z <= z <= v60_top_z:
                height_ratio = (z - v60_bottom_z) / cup_height_lu
                inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
                filter_outer = inner_radius - air_gap_lu
                filter_inner = filter_outer - paper_thickness_lu
                ax1.plot([center_x - filter_outer, center_x + filter_outer], [z, z], 
                        color=self.color_scheme['filter_paper'], linewidth=2)
                ax1.plot([center_x - filter_inner, center_x + filter_inner], [z, z], 
                        'orange', linewidth=1.2, alpha=0.8)

        # 底部開口設計
        bottom_inner = bottom_radius_lu
        ax1.plot([center_x - bottom_inner, center_x - bottom_inner], 
                [v60_bottom_z - wall_thickness, v60_bottom_z], 'k-', linewidth=3)
        ax1.plot([center_x + bottom_inner, center_x + bottom_inner], 
                [v60_bottom_z - wall_thickness, v60_bottom_z], 'k-', linewidth=3)
        
        ax1.annotate('Bottom Opening\n(Coffee Exit)', 
                    xy=(center_x, v60_bottom_z-2), 
                    xytext=(center_x+25, v60_bottom_z-15),
                    arrowprops=dict(arrowstyle='->', color=self.color_scheme['annotations'], lw=2),
                    fontsize=11, color=self.color_scheme['annotations'], ha='center')

        ax1.set_xlabel('X Position (Lattice Units)', fontsize=11)
        ax1.set_ylabel('Z Position (Lattice Units)', fontsize=11)
        ax1.set_xlim(center_x - top_radius_lu - 30, center_x + top_radius_lu + 30)
        ax1.set_ylim(0, v60_top_z + 25)
        ax1.grid(True, alpha=0.3)
        ax1.legend(['V60 Outer Wall', 'V60 Inner Wall', 'Filter Paper Outer', 'Filter Paper Inner'], 
                  fontsize=9, loc='upper right')

        # 2. 間隙詳細分析
        ax2.set_title('Gap Analysis Detail (2mm Air Gap)', fontsize=14, fontweight='bold')
        z_mid = v60_bottom_z + cup_height_lu / 2
        height_ratio = 0.5
        inner_radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio

        ax2.axvline(center_x + inner_radius, color=self.color_scheme['v60_inner'], 
                   linewidth=4, label='V60 Inner Wall')
        ax2.axvline(center_x + inner_radius - air_gap_lu, color=self.color_scheme['filter_paper'], 
                   linewidth=3, label='Filter Paper Outer')
        ax2.axvline(center_x + inner_radius - air_gap_lu - paper_thickness_lu, 
                   color='orange', linewidth=2, label='Filter Paper Inner')

        # 標註間隙尺寸
        gap_center = center_x + inner_radius - air_gap_lu/2
        ax2.annotate(f'2.0mm Gap\n({air_gap_lu:.2f} lattice units)',
                    xy=(gap_center, 0.5), xytext=(gap_center, 0.8),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2),
                    ha='center', fontsize=10, color='red', fontweight='bold')

        paper_center = center_x + inner_radius - air_gap_lu - paper_thickness_lu/2
        ax2.annotate(f'0.1mm Paper\n({paper_thickness_lu:.1f} lattice)',
                    xy=(paper_center, 0.3), xytext=(paper_center, 0.1),
                    arrowprops=dict(arrowstyle='<->', color='blue', lw=2),
                    ha='center', fontsize=10, color='blue', fontweight='bold')

        ax2.set_xlim(center_x + inner_radius - 12, center_x + inner_radius + 8)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('X Position (Lattice Units)', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. 尺寸標註圖
        ax3.set_title('Dimensional Specifications', fontsize=14, fontweight='bold')
        
        # 繪製主要尺寸線
        ax3.plot([center_x - top_radius_lu, center_x + top_radius_lu], [v60_top_z, v60_top_z], 
                'k-', linewidth=3)
        ax3.plot([center_x - bottom_radius_lu, center_x + bottom_radius_lu], [v60_bottom_z, v60_bottom_z], 
                'k-', linewidth=3)
        ax3.plot([center_x, center_x], [v60_bottom_z, v60_top_z], 'k--', linewidth=2)

        # 尺寸標註
        ax3.annotate(f'Top Diameter: {config.TOP_RADIUS*2*100:.1f} cm',
                    xy=(center_x, v60_top_z + 12), ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax3.annotate(f'Height:\n{config.CUP_HEIGHT*100:.1f} cm',
                    xy=(center_x + top_radius_lu + 15, (v60_bottom_z + v60_top_z)/2),
                    rotation=90, ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        ax3.annotate(f'Bottom Diameter: {config.BOTTOM_RADIUS*2*100:.1f} cm',
                    xy=(center_x, v60_bottom_z - 12), ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

        ax3.set_xlim(center_x - top_radius_lu - 30, center_x + top_radius_lu + 30)
        ax3.set_ylim(v60_bottom_z - 25, v60_top_z + 25)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)

        # 4. 流體路徑示意圖
        ax4.set_title('Fluid Flow Path Design', fontsize=14, fontweight='bold')
        
        # 繪製V60輪廓
        z_points = np.linspace(v60_bottom_z, v60_top_z, 50)
        x_inner, x_filter_inner = [], []

        for z in z_points:
            height_ratio = (z - v60_bottom_z) / cup_height_lu
            inner_r = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            filter_inner_r = inner_r - air_gap_lu - paper_thickness_lu
            x_inner.append(center_x + inner_r)
            x_filter_inner.append(center_x + filter_inner_r)

        ax4.plot(x_inner, z_points, color=self.color_scheme['v60_inner'], 
                linewidth=3, label='V60 Inner Wall')
        ax4.plot(x_filter_inner, z_points, color=self.color_scheme['filter_paper'], 
                linewidth=3, label='Filter Inner Surface')

        # 流體路徑箭頭
        for i in range(5, len(z_points), 12):
            z = z_points[i]
            x_start = x_filter_inner[i]
            x_end = x_inner[i] - 3
            if x_end > x_start:
                ax4.arrow(x_start, z, x_end - x_start, 0,
                         head_width=3, head_length=3, fc=self.color_scheme['fluid_flow'], 
                         ec=self.color_scheme['fluid_flow'], alpha=0.8, linewidth=2)

        # 出口標記
        ax4.annotate('Bottom Opening Outlet\n(Direct Drainage)', 
                    xy=(center_x, v60_bottom_z),
                    xytext=(center_x+20, v60_bottom_z-15),
                    arrowprops=dict(arrowstyle='->', color='green', lw=3),
                    fontsize=11, color='green', fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

        ax4.set_xlim(center_x - 20, center_x + top_radius_lu + 15)
        ax4.set_ylim(v60_bottom_z - 20, v60_top_z + 10)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('X Position (Lattice Units)', fontsize=11)
        ax4.set_ylabel('Z Position (Lattice Units)', fontsize=11)

        plt.tight_layout()
        plt.savefig(self.get_output_path('engineering_drawings.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_coffee_particle_distribution_analysis(self, particle_positions, particle_radii):
        """創建咖啡粒子分布專業分析圖表"""
        if len(particle_positions) == 0:
            print("   ⚠️  無咖啡粒子數據，跳過粒子分布分析")
            return
            
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        
        # 基本參數
        center_x = config.NX * 0.5
        center_y = config.NY * 0.5
        
        # 計算粒子統計
        x_coords = particle_positions[:, 0]
        y_coords = particle_positions[:, 1] 
        z_coords = particle_positions[:, 2]
        
        # 徑向距離計算
        dx = x_coords - center_x
        dy = y_coords - center_y
        radial_distances = np.sqrt(dx**2 + dy**2)
        
        # 1. 3D散點圖 (左上，跨兩格)
        ax1 = fig.add_subplot(gs[0, :2], projection='3d')
        ax1.set_title('3D Coffee Particle Distribution', fontsize=16, fontweight='bold')
        
        # 根據高度著色
        scatter = ax1.scatter(x_coords, y_coords, z_coords,
                             c=z_coords, cmap='YlOrBr', s=30, alpha=0.7)
        plt.colorbar(scatter, ax=ax1, shrink=0.6, label='Height (Lattice Units)')
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position') 
        ax1.set_zlabel('Z Height')
        
        # 2. XY平面投影 (右上)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title('XY Plane Projection\n(Top View)', fontsize=14, fontweight='bold')
        
        scatter2 = ax2.scatter(x_coords, y_coords, c=z_coords, cmap='viridis', s=25, alpha=0.8)
        plt.colorbar(scatter2, ax=ax2, label='Height')
        
        # 添加V60邊界圓圈
        v60_bottom_z = 5.0
        cup_height_lu = config.CUP_HEIGHT / config.SCALE_LENGTH
        v60_top_z = v60_bottom_z + cup_height_lu
        
        # 在不同高度畫V60邊界
        for z_level in [v60_bottom_z + cup_height_lu*0.1, v60_bottom_z + cup_height_lu*0.5, v60_bottom_z + cup_height_lu*0.9]:
            height_ratio = (z_level - v60_bottom_z) / cup_height_lu
            top_radius_lu = config.TOP_RADIUS / config.SCALE_LENGTH
            bottom_radius_lu = config.BOTTOM_RADIUS / config.SCALE_LENGTH
            radius = bottom_radius_lu + (top_radius_lu - bottom_radius_lu) * height_ratio
            
            circle = plt.Circle((center_x, center_y), radius, fill=False, 
                              color='red', alpha=0.6, linewidth=1.5)
            ax2.add_patch(circle)
        
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_aspect('equal')
        
        # 3. XZ平面投影 (右上右)
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.set_title('XZ Plane Projection\n(Side View)', fontsize=14, fontweight='bold')
        
        # 只顯示接近Y中心的粒子
        y_center_mask = np.abs(y_coords - center_y) <= 10
        if np.any(y_center_mask):
            ax3.scatter(x_coords[y_center_mask], z_coords[y_center_mask], 
                       c=radial_distances[y_center_mask], cmap='plasma', s=25, alpha=0.8)
        
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Z Height')
        
        # 4. 高度分佈直方圖 (左中)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_title('Height Distribution', fontsize=14, fontweight='bold')
        
        n, bins, patches = ax4.hist(z_coords, bins=25, alpha=0.7, color='brown', edgecolor='black')
        ax4.axvline(np.mean(z_coords), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(z_coords):.1f}')
        ax4.axvline(np.median(z_coords), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(z_coords):.1f}')
        
        ax4.set_xlabel('Z Height (Lattice Units)')
        ax4.set_ylabel('Particle Count')
        ax4.legend()
        
        # 5. 徑向分佈直方圖 (中中)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_title('Radial Distribution', fontsize=14, fontweight='bold')
        
        ax5.hist(radial_distances, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax5.axvline(np.mean(radial_distances), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(radial_distances):.1f}')
        
        ax5.set_xlabel('Radial Distance (Lattice Units)')
        ax5.set_ylabel('Particle Count')
        ax5.legend()
        
        # 6. 層次分析 (右中)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_title('Layer-wise Analysis', fontsize=14, fontweight='bold')
        
        # 將高度分為5層
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        n_layers = 5
        layer_boundaries = np.linspace(z_min, z_max, n_layers + 1)
        
        layer_counts = []
        layer_labels = []
        for i in range(n_layers):
            mask = (z_coords >= layer_boundaries[i]) & (z_coords < layer_boundaries[i+1])
            layer_counts.append(np.sum(mask))
            layer_labels.append(f'L{i+1}\n{layer_boundaries[i]:.1f}-{layer_boundaries[i+1]:.1f}')
        
        bars = ax6.bar(range(n_layers), layer_counts, alpha=0.7, color='orange', edgecolor='black')
        ax6.set_xlabel('Layer')
        ax6.set_ylabel('Particle Count')
        ax6.set_xticks(range(n_layers))
        ax6.set_xticklabels(layer_labels, fontsize=9)
        
        # 在條形圖上標註數值
        for bar, count in zip(bars, layer_counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 7. 密度熱圖 (右中右)
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.set_title('Particle Density Heatmap\n(XY Plane)', fontsize=14, fontweight='bold')
        
        # 創建2D直方圖
        H, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=15)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        im = ax7.imshow(H.T, extent=extent, origin='lower', cmap='hot', alpha=0.8)
        plt.colorbar(im, ax=ax7, label='Particle Density')
        
        ax7.set_xlabel('X Position')
        ax7.set_ylabel('Y Position')
        
        # 8. 統計摘要 (左下，跨兩格)
        ax8 = fig.add_subplot(gs[2, :2])
        ax8.set_title('Particle Distribution Statistics', fontsize=14, fontweight='bold')
        ax8.axis('off')
        
        # 計算統計數據
        total_particles = len(particle_positions)
        bed_height = z_max - z_min
        bed_height_cm = bed_height * config.SCALE_LENGTH * 100
        avg_radius_mm = np.mean(particle_radii) * 1000
        
        # 計算密度
        coffee_bed_volume_approx = np.pi * (np.mean(radial_distances) * config.SCALE_LENGTH)**2 * (bed_height * config.SCALE_LENGTH)
        particle_density = total_particles / coffee_bed_volume_approx if coffee_bed_volume_approx > 0 else 0
        
        stats_text = f"""Coffee Particle Distribution Statistics

Count & Density:
• Total Particles: {total_particles:,}
• Particle Density: {particle_density:.0f} particles/m³
• Packing Efficiency: {(total_particles * np.mean(particle_radii)**3) / (bed_height * np.mean(radial_distances)**2) * 100:.1f}%

Spatial Distribution:
• Bed Height: {bed_height_cm:.1f} cm ({bed_height:.1f} lattice units)
• Average Radius: {np.mean(radial_distances):.1f} lattice units
• Height Range: {z_min:.1f} - {z_max:.1f} lattice units

Statistical Measures:
• Height Mean: {np.mean(z_coords):.2f} ± {np.std(z_coords):.2f}
• Radial Mean: {np.mean(radial_distances):.2f} ± {np.std(radial_distances):.2f}
• Height Skewness: {np.mean((z_coords - np.mean(z_coords))**3) / np.std(z_coords)**3:.2f}

Physical Properties:
• Avg Particle Diameter: {avg_radius_mm:.2f} mm
• Total Coffee Mass: {config.COFFEE_POWDER_MASS*1000:.0f}g
• Bed Volume: {coffee_bed_volume_approx*1e6:.1f} cm³"""
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        # 9. 高度-徑向關係 (右下)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.set_title('Height vs Radial Position', fontsize=14, fontweight='bold')
        
        scatter9 = ax9.scatter(radial_distances, z_coords, c=particle_radii*1000, 
                              cmap='coolwarm', s=20, alpha=0.7)
        plt.colorbar(scatter9, ax=ax9, label='Particle Radius (mm)')
        
        # 添加趨勢線
        z_fit = np.polyfit(radial_distances, z_coords, 1)
        z_trend = np.poly1d(z_fit)
        ax9.plot(radial_distances, z_trend(radial_distances), 'r--', linewidth=2, 
                label=f'Trend: slope={z_fit[0]:.2f}')
        
        ax9.set_xlabel('Radial Distance (Lattice Units)')
        ax9.set_ylabel('Z Height (Lattice Units)')
        ax9.legend()
        
        # 10. 角度分佈 (右下右)
        ax10 = fig.add_subplot(gs[2, 3])
        ax10.set_title('Angular Distribution', fontsize=14, fontweight='bold')
        
        # 計算角度
        angles = np.arctan2(dy, dx) * 180 / np.pi
        angles[angles < 0] += 360  # 轉換到0-360度
        
        n_bins = 16
        counts, bin_edges = np.histogram(angles, bins=n_bins, range=(0, 360))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 極坐標直方圖
        ax10_polar = plt.subplot(gs[2, 3], projection='polar')
        ax10_polar.set_title('Angular Distribution\n(Polar View)', fontsize=12, fontweight='bold', pad=20)
        
        theta = np.radians(bin_centers)
        bars = ax10_polar.bar(theta, counts, width=2*np.pi/n_bins, alpha=0.7, 
                             color='purple', edgecolor='black')
        
        ax10_polar.set_theta_zero_location('E')
        ax10_polar.set_theta_direction(1)
        
        plt.suptitle('Coffee Particle Distribution Professional Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(self.get_output_path('coffee_particle_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_particle_size_distribution_analysis(self, particle_positions, particle_radii):
        """創建顆粒大小分佈專業分析圖表"""
        if len(particle_radii) == 0:
            print("   ⚠️  無顆粒大小數據，跳過大小分佈分析")
            return
            
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 轉換單位
        radii_mm = particle_radii * 1000  # 轉換為毫米
        diameters_mm = radii_mm * 2
        
        # 計算基本統計
        mean_radius = np.mean(radii_mm)
        std_radius = np.std(radii_mm)
        median_radius = np.median(radii_mm)
        
        # 1. 主要大小分佈直方圖 (左上，跨兩格)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_title('Particle Size Distribution - Detailed Analysis', fontsize=16, fontweight='bold')
        
        # 分別畫半徑和直徑
        n_bins = 25
        n1, bins1, patches1 = ax1.hist(radii_mm, bins=n_bins, alpha=0.7, color='steelblue', 
                                      edgecolor='black', linewidth=0.5, label='Radius (mm)')
        
        # 添加統計線
        ax1.axvline(mean_radius, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_radius:.3f} mm')
        ax1.axvline(median_radius, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_radius:.3f} mm')
        ax1.axvline(mean_radius + std_radius, color='orange', linestyle=':', linewidth=2,
                   label=f'+1σ: {mean_radius + std_radius:.3f} mm')
        ax1.axvline(mean_radius - std_radius, color='orange', linestyle=':', linewidth=2,
                   label=f'-1σ: {mean_radius - std_radius:.3f} mm')
        
        ax1.set_xlabel('Particle Radius (mm)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 累積分佈函數 (右上)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title('Cumulative Distribution\nFunction (CDF)', fontsize=14, fontweight='bold')
        
        sorted_radii = np.sort(radii_mm)
        cumulative = np.arange(1, len(sorted_radii) + 1) / len(sorted_radii)
        
        ax2.plot(sorted_radii, cumulative, 'b-', linewidth=2, label='CDF')
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50th percentile')
        ax2.axhline(0.25, color='green', linestyle='--', alpha=0.7, label='25th percentile')
        ax2.axhline(0.75, color='green', linestyle='--', alpha=0.7, label='75th percentile')
        
        ax2.set_xlabel('Particle Radius (mm)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q 正態分佈檢驗圖 (左中)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title('Q-Q Plot vs Normal\nDistribution', fontsize=14, fontweight='bold')
        
        from scipy import stats
        stats.probplot(radii_mm, dist="norm", plot=ax3)
        ax3.set_xlabel('Theoretical Quantiles')
        ax3.set_ylabel('Sample Quantiles')
        ax3.grid(True, alpha=0.3)
        
        # 4. 箱型圖 (中中)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('Box Plot Analysis', fontsize=14, fontweight='bold')
        
        box_data = [radii_mm, diameters_mm]
        labels = ['Radius (mm)', 'Diameter (mm)']
        
        bp = ax4.boxplot(box_data, tick_labels=labels, patch_artist=True, notch=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax4.set_ylabel('Size (mm)')
        ax4.grid(True, alpha=0.3)
        
        # 5. 大小 vs 位置關係 (右中)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_title('Size vs Z-Position\nRelationship', fontsize=14, fontweight='bold')
        
        if len(particle_positions) > 0:
            z_coords = particle_positions[:, 2]
            scatter5 = ax5.scatter(radii_mm, z_coords, c=z_coords, cmap='viridis', 
                                  s=30, alpha=0.7)
            plt.colorbar(scatter5, ax=ax5, label='Height')
            
            # 添加趨勢線
            correlation = np.corrcoef(radii_mm, z_coords)[0, 1]
            ax5.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax5.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax5.set_xlabel('Particle Radius (mm)')
        ax5.set_ylabel('Z Position (Lattice Units)')
        ax5.grid(True, alpha=0.3)
        
        # 6. 概率密度函數擬合 (左下)
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.set_title('PDF Fitting Analysis', fontsize=14, fontweight='bold')
        
        # 嘗試擬合多種分佈
        x_range = np.linspace(np.min(radii_mm), np.max(radii_mm), 100)
        
        # 正態分佈擬合
        mu, sigma = stats.norm.fit(radii_mm)
        normal_pdf = stats.norm.pdf(x_range, mu, sigma)
        
        # 對數正態分佈擬合
        s, loc, scale = stats.lognorm.fit(radii_mm)
        lognorm_pdf = stats.lognorm.pdf(x_range, s, loc, scale)
        
        # 繪製直方圖和擬合曲線
        ax6.hist(radii_mm, bins=20, density=True, alpha=0.7, color='lightgray', 
                edgecolor='black', label='Observed')
        ax6.plot(x_range, normal_pdf, 'r-', linewidth=2, 
                label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
        ax6.plot(x_range, lognorm_pdf, 'b-', linewidth=2, 
                label=f'Lognormal (s={s:.3f})')
        
        ax6.set_xlabel('Particle Radius (mm)')
        ax6.set_ylabel('Probability Density')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. 分層大小分析 (中下)
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.set_title('Size Distribution by\nHeight Layers', fontsize=14, fontweight='bold')
        
        if len(particle_positions) > 0:
            z_coords = particle_positions[:, 2]
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            
            # 分為3層
            n_layers = 3
            layer_boundaries = np.linspace(z_min, z_max, n_layers + 1)
            colors = ['red', 'green', 'blue']
            
            for i in range(n_layers):
                mask = (z_coords >= layer_boundaries[i]) & (z_coords < layer_boundaries[i+1])
                if np.any(mask):
                    layer_radii = radii_mm[mask]
                    ax7.hist(layer_radii, bins=12, alpha=0.6, color=colors[i],
                           label=f'Layer {i+1} (Z: {layer_boundaries[i]:.1f}-{layer_boundaries[i+1]:.1f})')
            
            ax7.set_xlabel('Particle Radius (mm)')
            ax7.set_ylabel('Frequency')
            ax7.legend(fontsize=9)
        else:
            ax7.text(0.5, 0.5, 'No Position Data\nAvailable', ha='center', va='center',
                    transform=ax7.transAxes, fontsize=12)
        
        ax7.grid(True, alpha=0.3)
        
        # 8. 統計摘要表 (右下)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.set_title('Size Statistics Summary', fontsize=14, fontweight='bold')
        ax8.axis('off')
        
        # 計算百分位數
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        perc_values = np.percentile(radii_mm, percentiles)
        
        # 計算變異係數
        cv = (std_radius / mean_radius) * 100
        
        # 計算偏度和峰度
        skewness = stats.skew(radii_mm)
        kurtosis = stats.kurtosis(radii_mm)
        
        # Goodness of fit測試
        _, p_norm = stats.shapiro(radii_mm[:min(5000, len(radii_mm))])  # Shapiro-Wilk test
        
        stats_text = f"""Particle Size Statistics

Basic Statistics:
• Count: {len(radii_mm):,} particles
• Mean Radius: {mean_radius:.3f} ± {std_radius:.3f} mm
• Mean Diameter: {mean_radius*2:.3f} ± {std_radius*2:.3f} mm
• Median: {median_radius:.3f} mm
• Coefficient of Variation: {cv:.1f}%

Distribution Shape:
• Skewness: {skewness:.3f} {'(right-skewed)' if skewness > 0.5 else '(symmetric)' if abs(skewness) < 0.5 else '(left-skewed)'}
• Kurtosis: {kurtosis:.3f} {'(heavy-tailed)' if kurtosis > 1 else '(normal)' if abs(kurtosis) < 1 else '(light-tailed)'}
• Normality p-value: {p_norm:.4f}

Percentiles (mm):
• 5th: {perc_values[0]:.3f}    • 95th: {perc_values[6]:.3f}
• 10th: {perc_values[1]:.3f}   • 90th: {perc_values[5]:.3f}
• 25th: {perc_values[2]:.3f}   • 75th: {perc_values[4]:.3f}
• 50th (Median): {perc_values[3]:.3f}

Target vs Actual:
• Target Diameter: {config.PARTICLE_DIAMETER_MM:.2f} mm
• Actual Mean: {mean_radius*2:.3f} mm
• Deviation: {((mean_radius*2 - config.PARTICLE_DIAMETER_MM)/config.PARTICLE_DIAMETER_MM)*100:+.1f}%"""
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9))
        
        plt.suptitle('Coffee Particle Size Distribution Professional Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(self.get_output_path('particle_size_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    visualizer = GeometryVisualizer()
    report_dir = visualizer.visualize_v60_geometry_with_particles()
    print(f"✅ 專業幾何視覺化報告生成完成: {report_dir}")
