# enhanced_visualizer.py
"""
增强版可视化系统 - 专门用于咖啡冲泡过程分析
提供纵向剖面图、流速分析等高级可视化功能
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import config

# 设置matplotlib字体支持
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

@ti.data_oriented
class EnhancedVisualizer:
    def __init__(self, lbm_solver, multiphase=None, geometry=None):
        self.lbm = lbm_solver
        self.multiphase = multiphase 
        self.geometry = geometry
        
        # 3D切片场
        self.xz_density = ti.field(dtype=ti.f32, shape=(config.NX, config.NZ))
        self.xz_velocity = ti.field(dtype=ti.f32, shape=(config.NX, config.NZ))
        self.xz_velocity_z = ti.field(dtype=ti.f32, shape=(config.NX, config.NZ))
        self.yz_density = ti.field(dtype=ti.f32, shape=(config.NY, config.NZ))
        self.yz_velocity = ti.field(dtype=ti.f32, shape=(config.NY, config.NZ))
        
        # 统计场
        self.velocity_stats = ti.field(dtype=ti.f32, shape=10)
        
        print("增强版可视化系统初始化完成")
    
    @ti.kernel
    def extract_xz_slice(self, y_slice: ti.i32):
        """提取XZ平面切片（纵向剖面）"""
        for i, k in ti.ndrange(config.NX, config.NZ):
            if y_slice < config.NY and self.lbm.solid[i, y_slice, k] == 0:
                # 密度场
                self.xz_density[i, k] = self.lbm.rho[i, y_slice, k]
                
                # 速度大小
                u = self.lbm.u[i, y_slice, k]
                self.xz_velocity[i, k] = u.norm()
                
                # Z方向速度分量（垂直流动）
                self.xz_velocity_z[i, k] = u[2]  # Z分量
            else:
                self.xz_density[i, k] = 0.0
                self.xz_velocity[i, k] = 0.0
                self.xz_velocity_z[i, k] = 0.0
    
    @ti.kernel
    def extract_yz_slice(self, x_slice: ti.i32):
        """提取YZ平面切片"""
        for j, k in ti.ndrange(config.NY, config.NZ):
            if x_slice < config.NX and self.lbm.solid[x_slice, j, k] == 0:
                self.yz_density[j, k] = self.lbm.rho[x_slice, j, k]
                u = self.lbm.u[x_slice, j, k]
                self.yz_velocity[j, k] = u.norm()
            else:
                self.yz_density[j, k] = 0.0
                self.yz_velocity[j, k] = 0.0
    
    @ti.kernel
    def analyze_flow_velocities(self):
        """分析流速统计"""
        max_velocity = 0.0
        avg_velocity = 0.0
        max_z_velocity = 0.0
        avg_z_velocity = 0.0
        fluid_nodes = 0
        water_nodes = 0
        total_flow_rate = 0.0
        
        for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
            if self.lbm.solid[i, j, k] == 0:  # 流体区域
                u = self.lbm.u[i, j, k]
                u_mag = u.norm()
                u_z = u[2]  # 垂直分量
                
                max_velocity = ti.max(max_velocity, u_mag)
                avg_velocity += u_mag
                
                max_z_velocity = ti.max(max_z_velocity, ti.abs(u_z))
                avg_z_velocity += ti.abs(u_z)
                
                fluid_nodes += 1
                
                # 检查是否为水相（简化判断）
                if self.lbm.rho[i, j, k] > 0.5 * config.RHO_WATER:
                    water_nodes += 1
                    total_flow_rate += u_mag
        
        if fluid_nodes > 0:
            avg_velocity /= fluid_nodes
            avg_z_velocity /= fluid_nodes
        
        if water_nodes > 0:
            total_flow_rate /= water_nodes
        
        # 存储统计结果
        self.velocity_stats[0] = max_velocity
        self.velocity_stats[1] = avg_velocity
        self.velocity_stats[2] = max_z_velocity
        self.velocity_stats[3] = avg_z_velocity
        self.velocity_stats[4] = fluid_nodes
        self.velocity_stats[5] = water_nodes
        self.velocity_stats[6] = total_flow_rate
    
    def save_longitudinal_analysis(self, timestamp, step_num):
        """保存纵向剖面分析图 - 增强V60锥形显示"""
        # 提取中间切片
        y_center = config.NY // 2
        self.extract_xz_slice(y_center)
        
        # 转换为numpy数组
        density_data = self.xz_density.to_numpy()
        velocity_data = self.xz_velocity.to_numpy()
        velocity_z_data = self.xz_velocity_z.to_numpy()
        
        # 创建坐标网格
        x = np.linspace(0, config.PHYSICAL_WIDTH * 100, config.NX)  # 转换为cm
        z = np.linspace(0, config.PHYSICAL_HEIGHT * 100, config.NZ)  # 转换为cm
        X, Z = np.meshgrid(x, z)
        
        # 计算V60锥形轮廓
        cup_height_cm = config.CUP_HEIGHT * 100  # 转换为cm
        top_radius_cm = config.TOP_RADIUS * 100
        bottom_radius_cm = config.BOTTOM_RADIUS * 100
        
        # V60锥形边界 (修正方向：尖端向下，开口向上)
        # 将濾杯居中显示在图片中间
        total_height = z.max()
        cup_top_z = total_height * 0.75  # 濾杯顶部在图片的75%高度处
        cup_bottom_z = cup_top_z - cup_height_cm  # 濾杯底部（出水口）在下方
        
        # 计算锥形边界线 (从底部到顶部)
        z_cone = np.linspace(cup_bottom_z, cup_top_z, 100)
        x_center = x.max() / 2  # 水平居中
        
        # 修正锥形计算：使用V60-02实际规格，底部小（出水口），顶部大（开口）
        # V60-02实际锥角约为68.7度 (全角)，我们使用真实的几何比例
        height_ratio = (z_cone - cup_bottom_z) / cup_height_cm
        cone_radius = bottom_radius_cm + (top_radius_cm - bottom_radius_cm) * height_ratio
        
        x_left_boundary = x_center - cone_radius
        x_right_boundary = x_center + cone_radius
        
        # 咖啡床区域 (在濾杯底部)
        coffee_bed_height_cm = config.COFFEE_BED_HEIGHT_LU * config.SCALE_LENGTH * 100
        coffee_bed_bottom_z = cup_bottom_z  # 咖啡床底部与濾杯底部对齐
        coffee_bed_top_z = coffee_bed_bottom_z + coffee_bed_height_cm
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=100)
        
        # 1. 水密度分布（纵向剖面）+ V60轮廓
        density_filtered = np.where(density_data.T > 0.1, density_data.T, np.nan)
        contour1 = ax1.contourf(X, Z, density_filtered, levels=20, cmap='Blues', alpha=0.8)
        cbar1 = plt.colorbar(contour1, ax=ax1, shrink=0.8)
        cbar1.set_label('Water Density (kg/m³)', fontsize=11, fontweight='bold')
        
        # 添加V60锥形轮廓
        ax1.plot(x_left_boundary, z_cone, 'k-', linewidth=3, label='V60 Dripper Wall')
        ax1.plot(x_right_boundary, z_cone, 'k-', linewidth=3)
        
        # 添加咖啡床区域标识 (修正位置)
        coffee_z_line = np.full_like(x, coffee_bed_top_z)
        ax1.plot(x, coffee_z_line, 'brown', linewidth=2, linestyle='--', alpha=0.8, label='Coffee Bed Surface')
        
        # 添加出水口 (在濾杯底部，尖端处)
        ax1.plot([x_center-bottom_radius_cm, x_center+bottom_radius_cm], 
                [cup_bottom_z, cup_bottom_z], 'red', linewidth=4, label='Outlet')
        
        # 添加注水区域指示 (在濾杯顶部上方)
        pour_zone_z = cup_top_z + 1.5  # 注水区域在濾杯上方
        pour_zone_width = top_radius_cm * 0.3  # 注水区域宽度
        ax1.plot([x_center-pour_zone_width, x_center+pour_zone_width], 
                [pour_zone_z, pour_zone_z], 'cyan', linewidth=4, 
                marker='v', markersize=8, label='Pour Zone')
        
        ax1.set_xlabel('X Position (cm)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Z Height (cm)', fontsize=11, fontweight='bold')
        ax1.set_title('V60 Longitudinal Cross-Section\nWater Density Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)
        
        # 添加重力方向箭头 (向下)
        ax1.annotate('Gravity', xy=(x.max()*0.85, cup_top_z), xytext=(x.max()*0.85, cup_top_z + 2),
                    arrowprops=dict(arrowstyle='->', lw=3, color='red'),
                    fontsize=12, fontweight='bold', color='red')
        
        # 2. 总流速大小 + V60轮廓
        velocity_filtered = np.where(velocity_data.T > 1e-6, velocity_data.T, np.nan)
        contour2 = ax2.contourf(X, Z, velocity_filtered, levels=20, cmap='Reds', alpha=0.8)
        cbar2 = plt.colorbar(contour2, ax=ax2, shrink=0.8)
        cbar2.set_label('Velocity Magnitude (m/s)', fontsize=11, fontweight='bold')
        
        # 添加V60轮廓
        ax2.plot(x_left_boundary, z_cone, 'k-', linewidth=2, alpha=0.7)
        ax2.plot(x_right_boundary, z_cone, 'k-', linewidth=2, alpha=0.7)
        ax2.plot(x, coffee_z_line, 'brown', linewidth=2, linestyle='--', alpha=0.6)
        
        ax2.set_xlabel('X Position (cm)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Z Height (cm)', fontsize=11, fontweight='bold')
        ax2.set_title('V60 Flow Velocity Magnitude\nWater Movement Speed', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 垂直流速分量（关键分析）+ V60轮廓
        velocity_z_filtered = np.where(np.abs(velocity_z_data.T) > 1e-6, velocity_z_data.T, np.nan)
        contour3 = ax3.contourf(X, Z, velocity_z_filtered, levels=20, cmap='RdBu_r', 
                               vmin=-np.nanmax(np.abs(velocity_z_filtered)), 
                               vmax=np.nanmax(np.abs(velocity_z_filtered)), alpha=0.8)
        cbar3 = plt.colorbar(contour3, ax=ax3, shrink=0.8)
        cbar3.set_label('Vertical Velocity (m/s)', fontsize=11, fontweight='bold')
        
        # 添加V60轮廓
        ax3.plot(x_left_boundary, z_cone, 'k-', linewidth=2, alpha=0.7)
        ax3.plot(x_right_boundary, z_cone, 'k-', linewidth=2, alpha=0.7)
        ax3.plot(x, coffee_z_line, 'brown', linewidth=2, linestyle='--', alpha=0.6)
        
        ax3.set_xlabel('X Position (cm)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Z Height (cm)', fontsize=11, fontweight='bold')
        ax3.set_title('V60 Vertical Flow Component\nDownward (blue) / Upward (red)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 流线图（速度向量场）+ V60轮廓和水位
        skip = max(1, config.NX // 20)  # 采样间隔
        X_sample = X[::skip, ::skip]
        Z_sample = Z[::skip, ::skip]
        velocity_x = np.zeros_like(velocity_z_data.T[::skip, ::skip])  # X方向速度为0（纵向切片）
        velocity_z_sample = velocity_z_data.T[::skip, ::skip]
        
        # 背景显示速度大小
        ax4.contourf(X, Z, velocity_filtered, levels=15, cmap='Reds', alpha=0.5)
        
        # 添加速度向量
        valid_mask = ~np.isnan(velocity_z_sample)
        if np.any(valid_mask):
            ax4.quiver(X_sample[valid_mask], Z_sample[valid_mask], 
                      velocity_x[valid_mask], velocity_z_sample[valid_mask],
                      scale=np.nanmax(np.abs(velocity_z_sample)) * 20, 
                      alpha=0.8, color='darkred', width=0.003)
        
        # 添加完整的V60设计
        ax4.plot(x_left_boundary, z_cone, 'k-', linewidth=3, label='V60 Wall')
        ax4.plot(x_right_boundary, z_cone, 'k-', linewidth=3)
        
        # 濾紙边界 (稍微内缩)
        filter_offset = 0.1  # cm
        ax4.plot(x_left_boundary + filter_offset, z_cone, 'gray', linewidth=2, 
                linestyle=':', alpha=0.8, label='Filter Paper')
        ax4.plot(x_right_boundary - filter_offset, z_cone, 'gray', linewidth=2, 
                linestyle=':', alpha=0.8)
        
        # 咖啡床
        ax4.fill_between(x, cup_bottom_z, coffee_bed_top_z, 
                        where=((x >= x_center - top_radius_cm*0.8) & (x <= x_center + top_radius_cm*0.8)),
                        color='brown', alpha=0.4, label='Coffee Bed')
        
        # 出水口和水滴 (出水口在濾杯底部，水滴在下方)
        ax4.plot([x_center-bottom_radius_cm, x_center+bottom_radius_cm], 
                [cup_bottom_z, cup_bottom_z], 'red', linewidth=4, label='Outlet')
        ax4.scatter([x_center], [cup_bottom_z-1.5], c='blue', s=50, alpha=0.8, label='Water Drop')
        
        ax4.set_xlabel('X Position (cm)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Z Height (cm)', fontsize=11, fontweight='bold')
        ax4.set_title('V60 Flow Streamlines & Geometry\nComplete Brewing Visualization', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right', fontsize=8)
        
        # 主标题
        fig.suptitle(f'V60 Pour-Over Coffee Flow Analysis - Longitudinal Section\nTime: {timestamp:.1f}s (Step: {step_num})', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存图像
        filename = f"v60_longitudinal_analysis_step_{step_num:04d}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ V60纵向剖面分析图已保存: {filename}")
        return filename
    
    def save_velocity_analysis(self, timestamp, step_num):
        """保存流速分析图"""
        # 计算统计数据
        self.analyze_flow_velocities()
        stats = self.velocity_stats.to_numpy()
        
        max_vel = stats[0]
        avg_vel = stats[1]
        max_z_vel = stats[2]
        avg_z_vel = stats[3]
        fluid_nodes = int(stats[4])
        water_nodes = int(stats[5])
        flow_rate = stats[6]
        
        # 提取不同高度的流速分布
        heights = [config.NZ - 10, config.NZ // 2, 10]  # 顶部、中部、底部
        height_labels = ['Top (Pour Zone)', 'Middle (Coffee Bed)', 'Bottom (Exit)']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=100)
        
        # 1. 不同高度的流速分布
        colors = ['red', 'orange', 'blue']
        for i, (height, label, color) in enumerate(zip(heights, height_labels, colors)):
            if height < config.NZ:
                y_center = config.NY // 2
                velocities = []
                positions = []
                
                # 提取该高度的速度数据
                for x in range(config.NX):
                    if self.lbm.solid[x, y_center, height] == 0:
                        u = self.lbm.u[x, y_center, height].to_numpy()
                        vel_mag = np.linalg.norm(u)
                        if vel_mag > 1e-6:
                            velocities.append(vel_mag)
                            positions.append(x * config.PHYSICAL_WIDTH / config.NX * 100)  # 转换为cm
                
                if velocities:
                    ax1.plot(positions, velocities, 'o-', color=color, label=label, alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('X Position (cm)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Velocity Magnitude (m/s)', fontsize=11, fontweight='bold')
        ax1.set_title('Velocity Distribution at Different Heights\nHorizontal Flow Profile', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 流速统计柱状图
        categories = ['Max\nVelocity', 'Avg\nVelocity', 'Max Vertical\nVelocity', 'Avg Vertical\nVelocity']
        values = [max_vel, avg_vel, max_z_vel, avg_z_vel]
        bars = ax2.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
        
        ax2.set_ylabel('Velocity (m/s)', fontsize=11, fontweight='bold')
        ax2.set_title('Flow Velocity Statistics\nCurrent Simulation State', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 3. 水相节点分布
        node_categories = ['Total Fluid\nNodes', 'Water Phase\nNodes', 'Flow Rate\nIndex']
        node_values = [fluid_nodes, water_nodes, flow_rate * 1000]  # 流速指数放大1000倍显示
        node_colors = ['lightblue', 'darkblue', 'green']
        
        bars2 = ax3.bar(node_categories, node_values, color=node_colors, alpha=0.8)
        ax3.set_ylabel('Count / Index', fontsize=11, fontweight='bold')
        ax3.set_title('Flow Domain Statistics\nNode Distribution & Flow Index', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars2, node_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(node_values)*0.01,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. 时间序列（如果有历史数据）
        # 这里暂时显示当前状态的总结
        ax4.text(0.1, 0.8, f'Current Analysis Summary', fontsize=14, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Simulation Time: {timestamp:.2f} seconds', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f'Step Number: {step_num}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f'Max Flow Speed: {max_vel:.5f} m/s', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.4, f'Avg Flow Speed: {avg_vel:.5f} m/s', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.3, f'Water Coverage: {water_nodes/max(1,fluid_nodes)*100:.1f}%', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.2, f'Vertical Flow Strength: {avg_z_vel:.5f} m/s', fontsize=12, transform=ax4.transAxes)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Flow Characteristics Summary\nKey Performance Indicators', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 主标题
        fig.suptitle(f'Pour-Over Coffee Velocity Analysis\nTime: {timestamp:.1f}s (Step: {step_num})', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存图像
        filename = f"velocity_analysis_step_{step_num:04d}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 流速分析图已保存: {filename}")
        return filename
    
    def save_combined_analysis(self, timestamp, step_num):
        """保存组合分析图（纵向剖面 + 流速分析）"""
        # 提取中间切片
        y_center = config.NY // 2
        self.extract_xz_slice(y_center)
        self.analyze_flow_velocities()
        
        # 转换数据
        density_data = self.xz_density.to_numpy()
        velocity_data = self.xz_velocity.to_numpy()
        velocity_z_data = self.xz_velocity_z.to_numpy()
        stats = self.velocity_stats.to_numpy()
        
        # 创建坐标
        x = np.linspace(0, config.PHYSICAL_WIDTH * 100, config.NX)
        z = np.linspace(0, config.PHYSICAL_HEIGHT * 100, config.NZ)
        X, Z = np.meshgrid(x, z)
        
        # 创建图表
        fig = plt.figure(figsize=(20, 12), dpi=100)
        
        # 1. 主要纵向剖面图 (左侧，占2/3宽度) - 增强V60显示
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        density_filtered = np.where(density_data.T > 0.1, density_data.T, np.nan)
        contour1 = ax1.contourf(X, Z, density_filtered, levels=25, cmap='Blues', alpha=0.8)
        cbar1 = plt.colorbar(contour1, ax=ax1, shrink=0.8)
        cbar1.set_label('Water Density (kg/m³)', fontsize=12, fontweight='bold')
        
        # 计算V60锥形轮廓 (修正方向：尖端向下，开口向上)
        cup_height_cm = config.CUP_HEIGHT * 100
        top_radius_cm = config.TOP_RADIUS * 100
        bottom_radius_cm = config.BOTTOM_RADIUS * 100
        
        # 将濾杯居中显示在图片中间
        total_height = z.max()
        cup_top_z = total_height * 0.75  # 濾杯顶部在图片的75%高度处
        cup_bottom_z = cup_top_z - cup_height_cm  # 濾杯底部（出水口）在下方
        
        z_cone = np.linspace(cup_bottom_z, cup_top_z, 100)
        x_center = x.max() / 2
        
        # 修正锥形计算：使用V60-02实际规格，底部小（出水口），顶部大（开口）
        # V60-02实际锥角约为68.7度 (全角)，我们使用真实的几何比例
        height_ratio = (z_cone - cup_bottom_z) / cup_height_cm
        cone_radius = bottom_radius_cm + (top_radius_cm - bottom_radius_cm) * height_ratio
        
        x_left_boundary = x_center - cone_radius
        x_right_boundary = x_center + cone_radius
        
        # 添加V60几何元素
        ax1.plot(x_left_boundary, z_cone, 'k-', linewidth=3, alpha=0.9, label='V60 Dripper')
        ax1.plot(x_right_boundary, z_cone, 'k-', linewidth=3, alpha=0.9)
        
        # 咖啡床区域 (在濾杯底部)
        coffee_bed_height_cm = config.COFFEE_BED_HEIGHT_LU * config.SCALE_LENGTH * 100
        coffee_bed_bottom_z = cup_bottom_z
        coffee_bed_top_z = coffee_bed_bottom_z + coffee_bed_height_cm
        coffee_z_line = np.full_like(x, coffee_bed_top_z)
        ax1.plot(x, coffee_z_line, 'brown', linewidth=2, linestyle='--', alpha=0.8, label='Coffee Bed')
        
        # 濾紙边界
        filter_offset = 0.1
        ax1.plot(x_left_boundary + filter_offset, z_cone, 'gray', linewidth=1.5, 
                linestyle=':', alpha=0.7, label='Filter Paper')
        ax1.plot(x_right_boundary - filter_offset, z_cone, 'gray', linewidth=1.5, 
                linestyle=':', alpha=0.7)
        
        # 出水口
        ax1.plot([x_center-bottom_radius_cm, x_center+bottom_radius_cm], 
                [cup_bottom_z, cup_bottom_z], 'red', linewidth=4, label='Outlet')
        
        # 注水区域 (在濾杯顶部上方)
        pour_zone_z = cup_top_z + 1.5
        pour_zone_width = top_radius_cm * 0.3
        ax1.plot([x_center-pour_zone_width, x_center+pour_zone_width], 
                [pour_zone_z, pour_zone_z], 'cyan', linewidth=3, 
                marker='v', markersize=6, label='Pour Zone')
        
        # 叠加速度向量
        skip = max(1, config.NX // 15)
        X_vec = X[::skip, ::skip]
        Z_vec = Z[::skip, ::skip]
        u_x_vec = np.zeros_like(velocity_z_data.T[::skip, ::skip])
        u_z_vec = velocity_z_data.T[::skip, ::skip]
        
        valid_mask = ~np.isnan(u_z_vec) & (np.abs(u_z_vec) > 1e-6)
        if np.any(valid_mask):
            ax1.quiver(X_vec[valid_mask], Z_vec[valid_mask], 
                      u_x_vec[valid_mask], u_z_vec[valid_mask],
                      scale=np.nanmax(np.abs(u_z_vec)) * 15, 
                      alpha=0.7, color='darkred', width=0.002)
        
        ax1.set_xlabel('X Position (cm)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Z Height (cm)', fontsize=12, fontweight='bold')
        ax1.set_title('V60 Longitudinal Cross-Section: Water Flow from Top to Bottom\nDensity Distribution + Velocity Vectors + Dripper Geometry', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=8, ncol=2)
        
        # 添加重力箭头 (向下)
        ax1.annotate('Gravity', xy=(x.max()*0.85, cup_top_z), xytext=(x.max()*0.85, cup_top_z + 2),
                    arrowprops=dict(arrowstyle='->', lw=3, color='red'),
                    fontsize=12, fontweight='bold', color='red')
        
        # 2. 速度大小分布 (右上)
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        velocity_filtered = np.where(velocity_data.T > 1e-6, velocity_data.T, np.nan)
        contour2 = ax2.contourf(X, Z, velocity_filtered, levels=15, cmap='Reds')
        cbar2 = plt.colorbar(contour2, ax=ax2, shrink=0.8)
        cbar2.set_label('Speed (m/s)', fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('X (cm)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Z (cm)', fontsize=10, fontweight='bold')
        ax2.set_title('Flow Speed\nMagnitude', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # 3. 流速统计 (左下)
        ax3 = plt.subplot2grid((2, 3), (1, 0))
        categories = ['Max\nVel', 'Avg\nVel', 'Max\nVertical', 'Avg\nVertical']
        values = [stats[0], stats[1], stats[2], stats[3]]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax3.bar(categories, values, color=colors, alpha=0.8)
        
        ax3.set_ylabel('Velocity (m/s)', fontsize=10, fontweight='bold')
        ax3.set_title('Velocity Statistics', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 4. 流域统计 (中下)
        ax4 = plt.subplot2grid((2, 3), (1, 1))
        node_cats = ['Fluid\nNodes', 'Water\nNodes']
        node_vals = [int(stats[4]), int(stats[5])]
        node_colors = ['lightblue', 'darkblue']
        bars2 = ax4.bar(node_cats, node_vals, color=node_colors, alpha=0.8)
        
        ax4.set_ylabel('Node Count', fontsize=10, fontweight='bold')
        ax4.set_title('Domain Statistics', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars2, node_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(node_vals)*0.02,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 5. 总结信息 (右下)
        ax5 = plt.subplot2grid((2, 3), (1, 2))
        summary_text = f'''Flow Analysis Summary
        
Time: {timestamp:.2f}s
Step: {step_num}

Max Speed: {stats[0]:.5f} m/s
Avg Speed: {stats[1]:.5f} m/s

Vertical Flow: {stats[3]:.5f} m/s
Water Coverage: {int(stats[5])/max(1,int(stats[4]))*100:.1f}%

Flow Rate Index: {stats[6]:.5f}
Active Nodes: {int(stats[4])}'''
        
        ax5.text(0.05, 0.95, summary_text, fontsize=9, transform=ax5.transAxes, 
                verticalalignment='top', fontfamily='monospace')
        ax5.set_title('Analysis Summary', fontsize=11, fontweight='bold')
        ax5.axis('off')
        
        # 主标题
        fig.suptitle(f'Pour-Over Coffee Flow Analysis - Complete Overview\nTime: {timestamp:.1f}s (Step: {step_num})', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存图像
        filename = f"combined_analysis_step_{step_num:04d}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 综合分析图已保存: {filename}")
        return filename
    
    def display_longitudinal_animation(self, update_interval=100):
        """显示动态纵向截面动画"""
        import matplotlib.animation as animation
        
        print("=== 启动纵向截面动画 ===")
        print("按 Ctrl+C 退出动画")
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # 预计算V60几何
        x = np.linspace(0, config.PHYSICAL_WIDTH * 100, config.NX)
        z = np.linspace(0, config.PHYSICAL_HEIGHT * 100, config.NZ)
        X, Z = np.meshgrid(x, z)
        
        # V60几何参数
        cup_height_cm = config.CUP_HEIGHT * 100
        top_radius_cm = config.TOP_RADIUS * 100
        bottom_radius_cm = config.BOTTOM_RADIUS * 100
        total_height = z.max()
        cup_top_z = total_height * 0.75
        cup_bottom_z = cup_top_z - cup_height_cm
        
        z_cone = np.linspace(cup_bottom_z, cup_top_z, 100)
        x_center = x.max() / 2
        height_ratio = (z_cone - cup_bottom_z) / cup_height_cm
        cone_radius = bottom_radius_cm + (top_radius_cm - bottom_radius_cm) * height_ratio
        x_left_boundary = x_center - cone_radius
        x_right_boundary = x_center + cone_radius
        
        coffee_bed_height_cm = config.COFFEE_BED_HEIGHT_LU * config.SCALE_LENGTH * 100
        coffee_bed_top_z = cup_bottom_z + coffee_bed_height_cm
        coffee_z_line = np.full_like(x, coffee_bed_top_z)
        
        def animate(frame):
            """动画更新函数"""
            # 清除所有子图
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # 提取当前中间切片
            y_center = config.NY // 2
            self.extract_xz_slice(y_center)
            self.analyze_flow_velocities()
            
            # 获取数据
            density_data = self.xz_density.to_numpy()
            velocity_data = self.xz_velocity.to_numpy()
            velocity_z_data = self.xz_velocity_z.to_numpy()
            stats = self.velocity_stats.to_numpy()
            
            # 1. 水密度分布 + V60轮廓
            density_filtered = np.where(density_data.T > 0.1, density_data.T, np.nan)
            if not np.all(np.isnan(density_filtered)):
                contour1 = ax1.contourf(X, Z, density_filtered, levels=15, cmap='Blues', alpha=0.8)
                fig.colorbar(contour1, ax=ax1, shrink=0.6)
            
            # V60几何
            ax1.plot(x_left_boundary, z_cone, 'k-', linewidth=2, label='V60 Wall')
            ax1.plot(x_right_boundary, z_cone, 'k-', linewidth=2)
            ax1.plot(x, coffee_z_line, 'brown', linewidth=2, linestyle='--', label='Coffee Bed')
            ax1.plot([x_center-bottom_radius_cm, x_center+bottom_radius_cm], 
                    [cup_bottom_z, cup_bottom_z], 'red', linewidth=3, label='Outlet')
            
            ax1.set_xlabel('X Position (cm)')
            ax1.set_ylabel('Z Height (cm)')
            ax1.set_title('Water Density Distribution')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 2. 流速大小
            velocity_filtered = np.where(velocity_data.T > 1e-6, velocity_data.T, np.nan)
            if not np.all(np.isnan(velocity_filtered)):
                contour2 = ax2.contourf(X, Z, velocity_filtered, levels=15, cmap='Reds', alpha=0.8)
                fig.colorbar(contour2, ax=ax2, shrink=0.6)
            
            ax2.plot(x_left_boundary, z_cone, 'k-', linewidth=1, alpha=0.7)
            ax2.plot(x_right_boundary, z_cone, 'k-', linewidth=1, alpha=0.7)
            ax2.set_xlabel('X Position (cm)')
            ax2.set_ylabel('Z Height (cm)')
            ax2.set_title('Flow Velocity Magnitude')
            ax2.grid(True, alpha=0.3)
            
            # 3. 垂直流速分量
            velocity_z_filtered = np.where(np.abs(velocity_z_data.T) > 1e-6, velocity_z_data.T, np.nan)
            if not np.all(np.isnan(velocity_z_filtered)):
                max_abs = np.nanmax(np.abs(velocity_z_filtered))
                if max_abs > 0:
                    contour3 = ax3.contourf(X, Z, velocity_z_filtered, levels=15, cmap='RdBu_r', 
                                           vmin=-max_abs, vmax=max_abs, alpha=0.8)
                    fig.colorbar(contour3, ax=ax3, shrink=0.6)
            
            ax3.plot(x_left_boundary, z_cone, 'k-', linewidth=1, alpha=0.7)
            ax3.plot(x_right_boundary, z_cone, 'k-', linewidth=1, alpha=0.7)
            ax3.set_xlabel('X Position (cm)')
            ax3.set_ylabel('Z Height (cm)')
            ax3.set_title('Vertical Flow Component')
            ax3.grid(True, alpha=0.3)
            
            # 4. 统计信息
            ax4.text(0.1, 0.8, f'Real-time Flow Analysis', fontsize=12, fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.1, 0.7, f'Max Velocity: {stats[0]:.5f} m/s', fontsize=10, transform=ax4.transAxes)
            ax4.text(0.1, 0.6, f'Avg Velocity: {stats[1]:.5f} m/s', fontsize=10, transform=ax4.transAxes)
            ax4.text(0.1, 0.5, f'Max Z-Velocity: {stats[2]:.5f} m/s', fontsize=10, transform=ax4.transAxes)
            ax4.text(0.1, 0.4, f'Avg Z-Velocity: {stats[3]:.5f} m/s', fontsize=10, transform=ax4.transAxes)
            ax4.text(0.1, 0.3, f'Fluid Nodes: {int(stats[4])}', fontsize=10, transform=ax4.transAxes)
            ax4.text(0.1, 0.2, f'Water Nodes: {int(stats[5])}', fontsize=10, transform=ax4.transAxes)
            ax4.text(0.1, 0.1, f'Flow Rate Index: {stats[6]:.5f}', fontsize=10, transform=ax4.transAxes)
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_title('Live Statistics')
            ax4.axis('off')
            
            # 设置整体标题
            fig.suptitle('V60 Pour-Over Coffee - Live Longitudinal Cross-Section Analysis', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
        
        # 创建动画
        try:
            ani = animation.FuncAnimation(fig, animate, interval=update_interval, cache_frame_data=False)
            plt.show()
        except KeyboardInterrupt:
            print("\n动画已停止")
        except Exception as e:
            print(f"动画显示错误: {e}")
        finally:
            plt.close(fig)