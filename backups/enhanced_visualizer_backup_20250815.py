# enhanced_visualizer.py
"""
科研級增強視覺化系統 - 專為咖啡萃取CFD研究設計
提供多物理場分析、量化統計、時間序列追蹤等功能
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# 設置matplotlib後端和字體
plt.rcParams['font.family'] = 'DejaVu Sans'

class EnhancedVisualizer:
            print(f"\n=== 生成科研級分析報告 (步驟 {step_num}) ===")
            
            # 1. 流場分析
            velocity_file = self.save_velocity_analysis(simulation_time, step_num)
            if velocity_file:
                generated_files.append(velocity_file)
                print(f"   ✓ 速度場分析: {velocity_file}")
            
            # 2. 綜合分析
            combined_file = self.save_combined_analysis(simulation_time, step_num)
            if combined_file:
                generated_files.append(combined_file)
                print(f"   ✓ 綜合分析: {combined_file}")
            
            # 3. 流動統計分析
            flow_stats_file = self.save_flow_statistics_chart(simulation_time, step_num)
            if flow_stats_file:
                generated_files.append(flow_stats_file)
                print(f"   ✓ 流動統計: {flow_stats_file}")
            
            # 4. 壓力分析
            pressure_file = self.save_pressure_analysis(simulation_time, step_num)
            if pressure_file:
                generated_files.append(pressure_file)
                print(f"   ✓ 壓力分析: {pressure_file}")
            
            # 5. 時間演化分析
            temporal_file = self.save_temporal_analysis(simulation_time, step_num)
            if temporal_file:
                generated_files.append(temporal_file)
                print(f"   ✓ 時間分析: {temporal_file}")
            
            # 6. LBM診斷監控
            if hasattr(self.simulation, 'lbm_diagnostics') and self.simulation.lbm_diagnostics:
                lbm_file = self.save_lbm_monitoring_chart(simulation_time, step_num)
                if lbm_file:
                    generated_files.append(lbm_file)
                    print(f"   ✓ LBM診斷: {lbm_file}")
            
            print(f"   >>> 共生成 {len(generated_files)} 個分析檔案")
            
            return generated_files
            
        except Exception as e:
            print(f"   ❌ 生成報告時發生錯誤: {str(e)}")
            return []
    
    def save_velocity_analysis(self, simulation_time, step_num):
        """保存速度分析圖（修復版）"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            if hasattr(self.lbm, 'u'):
                u_data = self.lbm.u.to_numpy()
                u_data = np.nan_to_num(u_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                u_magnitude = np.sqrt(u_data[:, :, :, 0]**2 + u_data[:, :, :, 1]**2 + u_data[:, :, :, 2]**2)
                u_magnitude = np.clip(u_magnitude, 0.0, 0.5)  # 限制速度範圍
                
                # 取XY平面切片
                z_level = config.NZ // 2
                u_slice = u_magnitude[:, :, z_level]
                
                im = ax.imshow(u_slice.T, origin='lower', aspect='equal', cmap=self.velocity_cmap, vmin=0.0, vmax=0.1)
                ax.set_title(f'Velocity Field (t={simulation_time:.2f}s, Z={z_level})', fontsize=12)
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                plt.colorbar(im, ax=ax)
                
                # 添加V60頂視圖輪廓
                self._add_v60_outline_fixed(ax, 'xy')
                
                # 添加顆粒可視化
                self._add_particles_to_plot(ax, 'xy', z_level)
            
            filename = f'velocity_analysis_step_{step_num:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            # 靜默處理錯誤
            return None
    
    def save_combined_analysis(self, simulation_time, step_num):
        """保存組合分析圖（修復版）"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            if hasattr(self.lbm, 'rho') and hasattr(self.lbm, 'u'):
                rho_data = self.lbm.rho.to_numpy()
                u_data = self.lbm.u.to_numpy()
                
                # 安全數據處理
                rho_data = np.nan_to_num(rho_data, nan=1.0, posinf=1.0, neginf=0.0)
                rho_data = np.clip(rho_data, 0.0, 2.0)
                u_data = np.nan_to_num(u_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                u_magnitude = np.sqrt(u_data[:, :, :, 0]**2 + u_data[:, :, :, 1]**2 + u_data[:, :, :, 2]**2)
                u_magnitude = np.clip(u_magnitude, 0.0, 0.5)
                
                # 密度 XZ切面
                z_slice_rho = rho_data[:, config.NY//2, :]
                im1 = ax1.imshow(z_slice_rho.T, origin='lower', aspect='auto', cmap=self.density_cmap, vmin=0.0, vmax=1.5)
                ax1.set_title('Density (XZ plane)', fontsize=10)
                plt.colorbar(im1, ax=ax1)
                self._add_v60_outline_fixed(ax1, 'xz')
                self._add_particles_to_plot(ax1, 'xz', config.NY//2)
                
                # 速度 XZ切面
                z_slice_u = u_magnitude[:, config.NY//2, :]
                im2 = ax2.imshow(z_slice_u.T, origin='lower', aspect='auto', cmap=self.velocity_cmap, vmin=0.0, vmax=0.1)
                ax2.set_title('Velocity (XZ plane)', fontsize=10)
                plt.colorbar(im2, ax=ax2)
                self._add_v60_outline_fixed(ax2, 'xz')
                self._add_particles_to_plot(ax2, 'xz', config.NY//2)
                
                # 密度 XY切面
                xy_slice_rho = rho_data[:, :, config.NZ//2]
                im3 = ax3.imshow(xy_slice_rho.T, origin='lower', aspect='equal', cmap=self.density_cmap, vmin=0.0, vmax=1.5)
                ax3.set_title('Density (XY plane)', fontsize=10)
                plt.colorbar(im3, ax=ax3)
                self._add_v60_outline_fixed(ax3, 'xy')
                self._add_particles_to_plot(ax3, 'xy', config.NZ//2)
                
                # 速度 XY切面
                xy_slice_u = u_magnitude[:, :, config.NZ//2]
                im4 = ax4.imshow(xy_slice_u.T, origin='lower', aspect='equal', cmap=self.velocity_cmap, vmin=0.0, vmax=0.1)
                ax4.set_title('Velocity (XY plane)', fontsize=10)
                plt.colorbar(im4, ax=ax4)
                self._add_v60_outline_fixed(ax4, 'xy')
                self._add_particles_to_plot(ax4, 'xy', config.NZ//2)
            
            plt.suptitle(f'Combined Analysis (t={simulation_time:.2f}s)', fontsize=14)
            filename = f'combined_analysis_step_{step_num:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            # 靜默處理錯誤
            return None
    
    def display_longitudinal_animation(self):
        """顯示縱向截面動畫 - CFD科研指引版本"""
        print("\n🎬 CFD動畫系統指引:")
        print("   ├─ 完整分析: generate_research_report()")
        print("   ├─ 時間序列: save_temporal_analysis()")
        print("   └─ 建議：使用批次處理模式進行長時間動畫生成")
    
    def _add_v60_outline_fixed(self, ax, plane='xz'):
        """添加修復版V60輪廓到圖表"""
        try:
            if plane == 'xz':
                # V60幾何參數
                center_x = config.NX // 2
                bottom_z = 5
                top_z = bottom_z + config.CUP_HEIGHT / config.SCALE_LENGTH
                top_radius = config.TOP_RADIUS / config.SCALE_LENGTH
                bottom_radius = config.BOTTOM_RADIUS / config.SCALE_LENGTH
                
                # 繪製V60錐形輪廓
                x_left_top = center_x - top_radius
                x_right_top = center_x + top_radius
                x_left_bottom = center_x - bottom_radius
                x_right_bottom = center_x + bottom_radius
                
                # V60內壁輪廓
                ax.plot([x_left_top, x_left_bottom], [top_z, bottom_z], 
                       'k-', linewidth=2, alpha=0.8, label='V60 Inner Wall')
                ax.plot([x_right_top, x_right_bottom], [top_z, bottom_z], 
                       'k-', linewidth=2, alpha=0.8)
                
                # V60底部
                ax.plot([x_left_bottom, x_right_bottom], [bottom_z, bottom_z], 
                       'k-', linewidth=2, alpha=0.8)
                
                # 出水孔
                hole_radius = config.BOTTOM_RADIUS / config.SCALE_LENGTH / 2
                ax.plot([center_x - hole_radius, center_x + hole_radius], [bottom_z, bottom_z], 
                       'r-', linewidth=3, alpha=0.8, label='Outlet Hole')
                
                # 添加圖例
                ax.legend(loc='upper right', fontsize=8)
                
            elif plane == 'xy':
                # XY平面的V60圓形輪廓
                center_x = config.NX // 2
                center_y = config.NY // 2
                top_radius = config.TOP_RADIUS / config.SCALE_LENGTH
                bottom_radius = config.BOTTOM_RADIUS / config.SCALE_LENGTH
                
                # 繪製V60頂部圓形輪廓
                circle_top = plt.Circle((center_x, center_y), top_radius, 
                                      fill=False, color='black', linewidth=2, alpha=0.8, label='V60 Top')
                ax.add_patch(circle_top)
                
                # 繪製出水孔
                hole_radius = bottom_radius / 2
                circle_hole = plt.Circle((center_x, center_y), hole_radius, 
                                       fill=False, color='red', linewidth=2, alpha=0.8, label='Outlet Hole')
                ax.add_patch(circle_hole)
                
                # 添加圖例
                ax.legend(loc='upper right', fontsize=8)
                
        except Exception as e:
            # 如果輪廓繪製失敗，靜默忽略
            pass
    
    def save_flow_statistics_chart(self, simulation_time, step_num):
        """生成流動統計圖表 - CFD參數分析"""
        try:
            flow_chars = self.calculate_flow_characteristics()
            if not flow_chars:
                return None
                
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Reynolds數歷史
            ax1.text(0.5, 0.5, f"Re = {flow_chars.get('reynolds_number', 0):.1f}", 
                    ha='center', va='center', fontsize=16, transform=ax1.transAxes)
            ax1.set_title('Reynolds Number')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            
            # 速度統計
            max_vel = flow_chars.get('max_velocity_physical', 0)
            mean_vel = flow_chars.get('mean_velocity_physical', 0)
            ax2.bar(['Max', 'Mean'], [max_vel, mean_vel])
            ax2.set_title('Velocity Statistics (m/s)')
            ax2.set_ylabel('Velocity (m/s)')
            
            # 壓力降
            pressure_drop = flow_chars.get('pressure_drop_pa', 0)
            ax3.text(0.5, 0.5, f"ΔP = {pressure_drop:.2f} Pa", 
                    ha='center', va='center', fontsize=14, transform=ax3.transAxes)
            ax3.set_title('Pressure Drop')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            
            # 質量守恆品質
            mass_cons = flow_chars.get('mass_conservation', {})
            quality = mass_cons.get('conservation_quality', 'Unknown')
            colors = {'Good': 'green', 'Moderate': 'orange', 'Poor': 'red', 'Unknown': 'gray'}
            ax4.text(0.5, 0.5, f"Mass Conservation:\n{quality}", 
                    ha='center', va='center', fontsize=12, transform=ax4.transAxes,
                    color=colors.get(quality, 'black'))
            ax4.set_title('Conservation Quality')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            
            plt.suptitle(f'CFD Flow Statistics (t={simulation_time:.2f}s)', fontsize=14)
            filename = f'cfd_flow_stats_step_{step_num:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            return None
    
    def save_pressure_analysis(self, simulation_time, step_num):
        """壓力場分析圖"""
        try:
            if not hasattr(self.lbm, 'rho'):
                return None
                
            rho_data = self.lbm.rho.to_numpy()
            pressure_data = rho_data * config.CS2 * config.SCALE_DENSITY * config.SCALE_VELOCITY**2
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # XZ切面壓力
            p_xz = pressure_data[:, config.NY//2, :]
            im1 = ax1.imshow(p_xz.T, origin='lower', aspect='auto', cmap='RdBu_r')
            ax1.set_title('Pressure (XZ plane)')
            plt.colorbar(im1, ax=ax1, label='Pressure (Pa)')
            
            # XY切面壓力
            p_xy = pressure_data[:, :, config.NZ//2]
            im2 = ax2.imshow(p_xy.T, origin='lower', aspect='equal', cmap='RdBu_r')
            ax2.set_title('Pressure (XY plane)')  
            plt.colorbar(im2, ax=ax2, label='Pressure (Pa)')
            
            filename = f'pressure_analysis_step_{step_num:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            return None
    
    def save_temperature_analysis(self, simulation_time, step_num):
        """溫度場分析圖 - 簡化版本（未實現完整溫度求解器）"""
        try:
            # 作為示例，基於密度場創建溫度相關的可視化
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            ax.text(0.5, 0.5, 'Temperature Analysis\n(Not implemented)', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title(f'Temperature Field (t={simulation_time:.2f}s)')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            filename = f'temperature_analysis_step_{step_num:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            return None
    
    def save_temporal_analysis(self, simulation_time, step_num):
        """時間序列分析圖"""
        try:
            if len(self.analysis_data['timestamps']) < 2:
                return None
                
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            times = self.analysis_data['timestamps']
            
            # 流量變化
            if self.analysis_data['flow_rates']:
                ax1.plot(times, self.analysis_data['flow_rates'], 'b-o')
                ax1.set_title('Flow Rate Evolution')
                ax1.set_ylabel('Flow Rate')
                ax1.grid(True)
            
            # 萃取效率
            if self.analysis_data['extraction_rates']:
                ax2.plot(times, self.analysis_data['extraction_rates'], 'g-s')
                ax2.set_title('Extraction Rate')
                ax2.set_ylabel('Extraction Rate')
                ax2.grid(True)
            
            # 壓力降變化
            if self.analysis_data['pressure_drops']:
                ax3.plot(times, self.analysis_data['pressure_drops'], 'r-^')
                ax3.set_title('Pressure Drop')
                ax3.set_ylabel('ΔP (Pa)')
                ax3.set_xlabel('Time (s)')
                ax3.grid(True)
            
            # V60效率
            if self.analysis_data['v60_efficiency']:
                ax4.plot(times, self.analysis_data['v60_efficiency'], 'm-d')
                ax4.set_title('V60 Efficiency')
                ax4.set_ylabel('Efficiency (%)')
                ax4.set_xlabel('Time (s)')
                ax4.grid(True)
            
            plt.suptitle(f'Temporal Analysis (up to t={simulation_time:.2f}s)', fontsize=14)
            filename = f'temporal_analysis_step_{step_num:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            return None
    
    def _add_particles_to_plot(self, ax, plane='xz', slice_idx=None):
        """添加咖啡顆粒到圖表"""
        if not self.particles:
            return
            
        try:
            # 獲取顆粒數據
            positions = self.particles.position.to_numpy()
            active = self.particles.active.to_numpy()
            
            active_particles = positions[active == 1]
            
            if len(active_particles) == 0:
                return
            
            if plane == 'xz' and slice_idx is not None:
                # 在XZ平面顯示，選擇Y坐標接近slice_idx的顆粒
                tolerance = 5.0  # 容忍範圍
                selected_particles = active_particles[
                    np.abs(active_particles[:, 1] - slice_idx) <= tolerance
                ]
                
                if len(selected_particles) > 0:
                    # 繪製顆粒
                    ax.scatter(selected_particles[:, 0], selected_particles[:, 2], 
                             c='brown', s=2, alpha=0.6, label=f'Coffee Particles ({len(selected_particles)})')
                    
            elif plane == 'xy' and slice_idx is not None:
                # 在XY平面顯示，選擇Z坐標接近slice_idx的顆粒
                tolerance = 5.0  # 容忍範圍
                selected_particles = active_particles[
                    np.abs(active_particles[:, 2] - slice_idx) <= tolerance
                ]
                
                if len(selected_particles) > 0:
                    # 繪製顆粒
                    ax.scatter(selected_particles[:, 0], selected_particles[:, 1], 
                             c='brown', s=2, alpha=0.6, label=f'Coffee Particles ({len(selected_particles)})')
                    
            # 更新圖例
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(handles, labels, loc='upper right', fontsize=8)
                    
        except Exception as e:
            # 如果顆粒繪製失敗，靜默忽略
            pass
    
    def save_lbm_monitoring_chart(self, simulation_time, step_num):
        """保存LBM專用監控圖表 - CFD診斷視覺化"""
        try:
            # 獲取診斷數據
            if hasattr(self, 'simulation') and hasattr(self.simulation, 'diagnostics'):
                diagnostics = self.simulation.diagnostics.get_current_diagnostics()
            else:
                # 如果沒有simulation引用，嘗試通過其他方式獲取
                diagnostics = {}
                
            # 如果沒有診斷數據，創建基本監控圖表
            if not diagnostics:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                
                # 基本速度統計
                if hasattr(self.lbm, 'u'):
                    u_data = self.lbm.u.to_numpy()
                    u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
                    
                    ax1.hist(u_magnitude.flatten(), bins=50, alpha=0.7)
                    ax1.set_title('Velocity Distribution')
                    ax1.set_xlabel('Velocity (LU)')
                    ax1.set_ylabel('Frequency')
                    ax1.set_yscale('log')
                
                # 基本密度統計
                if hasattr(self.lbm, 'rho'):
                    rho_data = self.lbm.rho.to_numpy()
                    
                    ax2.hist(rho_data.flatten(), bins=50, alpha=0.7)
                    ax2.set_title('Density Distribution')
                    ax2.set_xlabel('Density (LU)')
                    ax2.set_ylabel('Frequency')
                    ax2.set_yscale('log')
                
                # 基本Mach數檢查
                if hasattr(self.lbm, 'u'):
                    cs_lattice = 1.0 / np.sqrt(3.0)
                    mach_numbers = u_magnitude / cs_lattice
                    
                    ax3.hist(mach_numbers.flatten(), bins=50, alpha=0.7)
                    ax3.set_title('Mach Number Distribution')
                    ax3.set_xlabel('Mach Number')
                    ax3.set_ylabel('Frequency')
                    ax3.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Ma=0.1 limit')
                    ax3.set_yscale('log')
                    ax3.legend()
                
                # 狀態信息
                ax4.text(0.5, 0.5, 'LBM Basic Monitoring\n\nFor detailed diagnostics,\nuse LBMDiagnostics system', 
                        ha='center', va='center', fontsize=12, transform=ax4.transAxes)
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.set_title('System Status')
                
                plt.suptitle(f'LBM Basic Monitoring (t={simulation_time:.2f}s)', fontsize=14)
                
            else:
                # 使用完整診斷數據創建高級監控圖表
                fig = plt.figure(figsize=(15, 10))
                gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
                
                # 1. 時間穩定性
                ax1 = fig.add_subplot(gs[0, 0])
                stability = diagnostics.get('temporal_stability', {})
                if stability and 'stability_grade' in stability:
                    colors = {'Excellent': 'green', 'Good': 'blue', 'Monitoring': 'orange'}
                    grade = stability['stability_grade']
                    ax1.text(0.5, 0.5, f"穩定性評級\n{grade}", ha='center', va='center',
                            fontsize=12, color=colors.get(grade, 'black'), weight='bold')
                    
                    if 'relative_density_change' in stability:
                        ax1.text(0.5, 0.2, f"密度變化率: {stability['relative_density_change']:.2e}",
                                ha='center', va='center', fontsize=8)
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax1.set_title('時間穩定性', fontsize=10)
                
                # 2. LBM數值品質
                ax2 = fig.add_subplot(gs[0, 1])
                lbm_quality = diagnostics.get('lbm_quality', {})
                if lbm_quality and 'lbm_grade' in lbm_quality:
                    colors = {'Excellent': 'green', 'Good': 'blue', 'Caution': 'red'}
                    grade = lbm_quality['lbm_grade']
                    ax2.text(0.5, 0.7, f"LBM品質\n{grade}", ha='center', va='center',
                            fontsize=12, color=colors.get(grade, 'black'), weight='bold')
                    
                    if 'max_mach' in lbm_quality:
                        ax2.text(0.5, 0.4, f"最大Mach數: {lbm_quality['max_mach']:.4f}",
                                ha='center', va='center', fontsize=8)
                    if 'max_density_deviation' in lbm_quality:
                        ax2.text(0.5, 0.2, f"密度偏差: {lbm_quality['max_density_deviation']:.4f}",
                                ha='center', va='center', fontsize=8)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.set_title('LBM數值品質', fontsize=10)
                
                # 3. 守恆定律
                ax3 = fig.add_subplot(gs[0, 2])
                conservation = diagnostics.get('conservation', {})
                if conservation and 'conservation_grade' in conservation:
                    colors = {'Excellent': 'green', 'Good': 'blue', 'Moderate': 'orange'}
                    grade = conservation['conservation_grade']
                    ax3.text(0.5, 0.7, f"守恆品質\n{grade}", ha='center', va='center',
                            fontsize=12, color=colors.get(grade, 'black'), weight='bold')
                    
                    if 'relative_mass_error' in conservation:
                        ax3.text(0.5, 0.4, f"質量誤差: {conservation['relative_mass_error']:.2e}",
                                ha='center', va='center', fontsize=8)
                    if 'total_mass' in conservation:
                        ax3.text(0.5, 0.2, f"總質量: {conservation['total_mass']:.2f}",
                                ha='center', va='center', fontsize=8)
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
                ax3.set_title('守恆定律', fontsize=10)
                
                # 4. V60物理過程
                ax4 = fig.add_subplot(gs[1, :])
                v60_physics = diagnostics.get('v60_physics', {})
                if v60_physics:
                    # 流量平衡圖表
                    categories = ['Inlet Flow', 'Outlet Flow', 'System Volume']
                    values = [
                        v60_physics.get('inlet_flow_rate', 0),
                        v60_physics.get('outlet_flow_rate', 0),
                        v60_physics.get('system_water_volume', 0) * 0.1  # 縮放以便顯示
                    ]
                    
                    bars = ax4.bar(categories, values, color=['blue', 'red', 'green'], alpha=0.7)
                    ax4.set_ylabel('Flow Rate / Volume (scaled)')
                    ax4.set_title('V60 Flow Balance Analysis')
                    
                    # 添加數值標籤
                    for bar, val in zip(bars, values):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
                
                # 5. 診斷歷史趨勢（如果有歷史數據）
                ax5 = fig.add_subplot(gs[2, :])
                if hasattr(self, 'simulation') and hasattr(self.simulation, 'diagnostics'):
                    times, history = self.simulation.diagnostics.get_diagnostics_history()
                    
                    if len(times) > 1:
                        # 提取關鍵指標的歷史
                        mach_history = []
                        mass_error_history = []
                        
                        for data in history:
                            lbm_q = data.get('lbm_quality', {})
                            cons_q = data.get('conservation', {})
                            
                            mach_history.append(lbm_q.get('max_mach', 0))
                            mass_error_history.append(cons_q.get('relative_mass_error', 0))
                        
                        ax5_twin = ax5.twinx()
                        
                        line1 = ax5.plot(times, mach_history, 'b-o', label='Max Mach Number', markersize=3)
                        line2 = ax5_twin.plot(times, mass_error_history, 'r-s', label='Mass Error', markersize=3)
                        
                        ax5.set_xlabel('Simulation Time (s)')
                        ax5.set_ylabel('Mach Number', color='blue')
                        ax5_twin.set_ylabel('Relative Mass Error', color='red')
                        ax5.set_title('Diagnostic History Trends')
                        
                        # 合併圖例
                        lines = line1 + line2
                        labels = [l.get_label() for l in lines]
                        ax5.legend(lines, labels, loc='upper left', fontsize=8)
                    else:
                        ax5.text(0.5, 0.5, 'Insufficient history data\nfor trend analysis', 
                                ha='center', va='center', fontsize=12)
                        ax5.set_xlim(0, 1)
                        ax5.set_ylim(0, 1)
                
                plt.suptitle(f'LBM Comprehensive Diagnostics (t={simulation_time:.2f}s)', fontsize=14)
            
            filename = f'lbm_monitoring_step_{step_num:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            return None