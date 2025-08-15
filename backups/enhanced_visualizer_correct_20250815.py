# enhanced_visualizer.py
"""
科研級增強視覺化系統 - 專為咖啡萃取CFD研究設計
提供多物理場分析、量化統計、時間序列追蹤等功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import ndimage
from scipy.stats import pearsonr
import json
import time
from datetime import datetime
import config

class EnhancedVisualizer:
    def __init__(self, lbm_solver, multiphase=None, geometry=None, particle_system=None, filter_system=None):
        """
        科研級視覺化系統初始化
        
        Args:
            lbm_solver: LBM求解器
            multiphase: 多相流系統  
            geometry: 幾何系統
            particle_system: 咖啡顆粒系統
            filter_system: 濾紙系統
        """
        self.lbm = lbm_solver
        self.multiphase = multiphase
        self.geometry = geometry
        self.particles = particle_system
        self.filter = filter_system
        
        # 科研分析參數
        self.analysis_data = {
            'timestamps': [],
            'flow_rates': [],
            'extraction_rates': [],
            'pressure_drops': [],
            'v60_efficiency': [],
            'particle_dynamics': [],
            'filter_performance': []
        }
        
        # 專業配色方案
        self.setup_colormaps()
        
        # 分析區域定義
        self.define_analysis_regions()
        
        print("🔬 科研級增強視覺化系統已初始化")
        print(f"   └─ 多物理場分析: {'✅' if multiphase else '❌'}")
        print(f"   └─ 顆粒追蹤: {'✅' if particle_system else '❌'}")
        print(f"   └─ 濾紙分析: {'✅' if filter_system else '❌'}")
    
    def setup_colormaps(self):
        """設置專業科研配色"""
        # 流體密度配色（藍色系）
        self.density_cmap = LinearSegmentedColormap.from_list(
            'density', ['#f7fbff', '#08519c'], N=256)
        
        # 速度場配色（紅色系）  
        self.velocity_cmap = LinearSegmentedColormap.from_list(
            'velocity', ['#fff5f0', '#67000d'], N=256)
        
        # 相場配色（綠色系）
        self.phase_cmap = LinearSegmentedColormap.from_list(
            'phase', ['#f7fcf5', '#00441b'], N=256)
        
        # 咖啡濃度配色（棕色系）
        self.coffee_cmap = LinearSegmentedColormap.from_list(
            'coffee', ['#fff8dc', '#3e2723'], N=256)
        
        # 溫度場配色（彩虹系）
        self.temp_cmap = plt.cm.plasma
    
    def define_analysis_regions(self):
        """定義V60關鍵分析區域"""
        center_x = config.NX * 0.5
        center_y = config.NY * 0.5
        
        # V60幾何參數（格子單位）
        top_radius = config.TOP_RADIUS / config.SCALE_LENGTH
        bottom_radius = config.BOTTOM_RADIUS / config.SCALE_LENGTH
        cup_height = config.CUP_HEIGHT / config.SCALE_LENGTH
        
        self.regions = {
            'pouring_zone': {
                'center': (center_x, center_y),
                'radius': top_radius * 0.3,
                'z_range': (config.NZ * 0.8, config.NZ * 0.95),
                'description': '注水區域'
            },
            'extraction_zone': {
                'center': (center_x, center_y),  
                'radius': top_radius * 0.8,
                'z_range': (20, 60),
                'description': '主要萃取區域'
            },
            'filter_zone': {
                'center': (center_x, center_y),
                'radius': bottom_radius,
                'z_range': (5, 15),
                'description': '濾紙過濾區域'  
            },
            'outlet_zone': {
                'center': (center_x, center_y),
                'radius': bottom_radius * 0.5,
                'z_range': (0, 5),
                'description': '出口區域'
            }
        }
    
    def calculate_flow_characteristics(self):
        """計算流體力學特徵參數 (科研級修正版)"""
        if not hasattr(self.lbm, 'u') or not hasattr(self.lbm, 'rho'):
            return {}
        
        u_data = self.lbm.u.to_numpy()
        rho_data = self.lbm.rho.to_numpy()
        
        # 速度場分析 (轉換為物理單位)
        u_mag = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
        u_mag_physical = u_mag * config.SCALE_VELOCITY  # m/s
        
        # Reynolds數計算 (正確的物理方法) - 修復空數組問題
        active_velocities = u_mag_physical[u_mag_physical > 1e-6]
        characteristic_velocity = np.mean(active_velocities) if len(active_velocities) > 0 else config.U_CHAR
        characteristic_length = config.L_CHAR  # 特徵長度 (V60高度)
        kinematic_viscosity = config.NU_CHAR  # 特徵運動黏滯度
        
        if characteristic_velocity > 0 and kinematic_viscosity > 0:
            reynolds = (characteristic_velocity * characteristic_length) / kinematic_viscosity
        else:
            reynolds = 0.0
        
        # 壓力場分析 (轉換為物理單位)
        pressure_lu = rho_data * config.CS2  # 格子單位壓力
        pressure_physical = pressure_lu * config.SCALE_DENSITY * config.SCALE_VELOCITY**2  # Pa
        
        # 壓力梯度計算
        grad_p = np.gradient(pressure_physical)
        pressure_drop = np.max(pressure_physical) - np.min(pressure_physical)
        
        # 流量計算（各區域）
        flow_rates = {}
        for region_name, region in self.regions.items():
            flow_rates[region_name] = self._calculate_regional_flow_rate(u_data, region)
        
        # Weber數和Froude數
        if hasattr(config, 'SURFACE_TENSION_LU') and config.SURFACE_TENSION_LU > 0:
            weber_number = (config.RHO_WATER * config.SCALE_DENSITY * characteristic_velocity**2 * 
                          characteristic_length) / (config.SURFACE_TENSION_LU * config.SCALE_DENSITY * 
                          config.SCALE_VELOCITY**2 * config.SCALE_LENGTH)
        else:
            weber_number = 0.0
        
        froude_number = characteristic_velocity / np.sqrt(config.GRAVITY_PHYS * characteristic_length)
        
        return {
            'reynolds_number': reynolds,
            'weber_number': weber_number,
            'froude_number': froude_number,
            'max_velocity_physical': np.max(u_mag_physical),
            'mean_velocity_physical': np.mean(active_velocities) if len(active_velocities) > 0 else 0.0,
            'max_velocity_lu': np.max(u_mag),
            'pressure_drop_pa': pressure_drop,
            'flow_rates': flow_rates,
            'vorticity': self._calculate_vorticity(u_data),
            'mass_conservation': self._check_mass_conservation(rho_data),
            'characteristic_scales': {
                'length': characteristic_length,
                'velocity': characteristic_velocity,
                'time': characteristic_length / characteristic_velocity if characteristic_velocity > 0 else 0,
                'viscosity': kinematic_viscosity
            }
        }
    
    def _calculate_regional_flow_rate(self, u_data, region):
        """計算指定區域的流量"""
        center_x, center_y = region['center']
        radius = region['radius']
        z_min, z_max = region['z_range']
        
        flow_rate = 0.0
        count = 0
        
        for i in range(max(0, int(center_x - radius)), min(config.NX, int(center_x + radius))):
            for j in range(max(0, int(center_y - radius)), min(config.NY, int(center_y + radius))):
                for k in range(max(0, int(z_min)), min(config.NZ, int(z_max))):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist <= radius:
                        flow_rate += np.sqrt(u_data[i,j,k,0]**2 + u_data[i,j,k,1]**2 + u_data[i,j,k,2]**2)
                        count += 1
        
        return flow_rate / count if count > 0 else 0.0
    
    def _calculate_vorticity(self, u_data):
        """計算渦度"""
        # 計算速度場的旋度
        omega_x = np.gradient(u_data[:,:,:,2], axis=1) - np.gradient(u_data[:,:,:,1], axis=2)
        omega_y = np.gradient(u_data[:,:,:,0], axis=2) - np.gradient(u_data[:,:,:,2], axis=0)
        omega_z = np.gradient(u_data[:,:,:,1], axis=0) - np.gradient(u_data[:,:,:,0], axis=1)
        
        vorticity_magnitude = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        return {
            'max_vorticity': np.max(vorticity_magnitude),
            'mean_vorticity': np.mean(vorticity_magnitude),
            'vorticity_field': vorticity_magnitude
        }
    
    def _check_mass_conservation(self, rho_data):
        """檢查質量守恆"""
        total_mass = np.sum(rho_data)
        mass_variation = np.std(rho_data) / np.mean(rho_data)
        return {
            'total_mass': total_mass,
            'mass_variation_coefficient': mass_variation,
            'conservation_quality': 'Good' if mass_variation < 0.1 else 'Moderate' if mass_variation < 0.2 else 'Poor'
        }
        
    def analyze_particle_dynamics(self):
        """分析咖啡顆粒動力學"""
        if not self.particles:
            return {}
        
        try:
            # 獲取顆粒數據
            positions = self.particles.position.to_numpy()
            velocities = self.particles.velocity.to_numpy()
            active = self.particles.active.to_numpy()
            
            active_particles = positions[active == 1]
            active_velocities = velocities[active == 1]
            
            if len(active_particles) == 0:
                return {'status': 'no_active_particles'}
            
            # 顆粒分佈分析
            z_distribution = active_particles[:, 2]
            radial_distribution = np.sqrt((active_particles[:, 0] - config.NX/2)**2 + 
                                        (active_particles[:, 1] - config.NY/2)**2)
            
            # 速度統計
            particle_speeds = np.sqrt(np.sum(active_velocities**2, axis=1))
            
            # 沉降分析
            settling_velocity = np.mean(active_velocities[:, 2])  # Z方向平均速度
            
            return {
                'active_particle_count': len(active_particles),
                'z_distribution': {
                    'mean': np.mean(z_distribution),
                    'std': np.std(z_distribution),
                    'range': [np.min(z_distribution), np.max(z_distribution)]
                },
                'radial_distribution': {
                    'mean': np.mean(radial_distribution), 
                    'std': np.std(radial_distribution)
                },
                'velocity_stats': {
                    'mean_speed': np.mean(particle_speeds),
                    'max_speed': np.max(particle_speeds),
                    'settling_velocity': settling_velocity
                },
                'bed_compaction': 1.0 - (np.std(z_distribution) / np.mean(z_distribution))
            }
            
        except Exception as e:
            print(f"Warning: Could not analyze particle dynamics: {e}")
            return {}
    
    def generate_research_report(self, simulation_time, step_num):
        """生成完整的科研報告"""
        print(f"🔬 生成科研級分析報告 (t={simulation_time:.2f}s)...")
        
        generated_files = []
        
        # 1. 綜合分析
        multi_file = self.save_combined_analysis(simulation_time, step_num)
        if multi_file:
            generated_files.append(multi_file)
        
        # 2. 速度場分析
        velocity_file = self.save_velocity_analysis(simulation_time, step_num)
        if velocity_file:
            generated_files.append(velocity_file)
        
        # 3. 保持原有功能兼容性
        longitudinal_file = self.save_longitudinal_analysis(simulation_time, step_num)
        if longitudinal_file:
            generated_files.append(longitudinal_file)
        
        velocity_file = self.save_velocity_analysis(simulation_time, step_num)
        if velocity_file:
            generated_files.append(velocity_file)
        
        combined_file = self.save_combined_analysis(simulation_time, step_num)
        if combined_file:
            generated_files.append(combined_file)
        
        print(f"✅ 科研報告生成完成，共 {len(generated_files)} 個文件:")
        for file in generated_files:
            print(f"   📄 {file}")
        
        return generated_files
    
    def export_data_for_analysis(self, simulation_time, step_num):
        """導出數據供外部分析工具使用"""
        try:
            export_data = {}
            
            # 流體場數據
            if hasattr(self.lbm, 'u') and hasattr(self.lbm, 'rho'):
                u_data = self.lbm.u.to_numpy()
                rho_data = self.lbm.rho.to_numpy()
                
                export_data['velocity_field'] = {
                    'u_x': u_data[:,:,:,0].tolist(),
                    'u_y': u_data[:,:,:,1].tolist(), 
                    'u_z': u_data[:,:,:,2].tolist(),
                    'magnitude': np.sqrt(np.sum(u_data**2, axis=3)).tolist()
                }
                
                export_data['density_field'] = rho_data.tolist()
                export_data['pressure_field'] = (rho_data * config.CS2_LU).tolist()
            
            # 顆粒數據
            if self.particles:
                positions = self.particles.position.to_numpy()
                velocities = self.particles.velocity.to_numpy()
                active = self.particles.active.to_numpy()
                
                export_data['particles'] = {
                    'positions': positions[active == 1].tolist(),
                    'velocities': velocities[active == 1].tolist(),
                    'count': int(np.sum(active))
                }
            
            # 濾紙數據
            if self.filter:
                filter_stats = self.filter.get_filter_statistics()
                export_data['filter'] = filter_stats
            
            # 元數據
            export_data['metadata'] = {
                'simulation_time': simulation_time,
                'step_number': step_num,
                'grid_size': [config.NX, config.NY, config.NZ],
                'physical_parameters': {
                    'scale_length': config.SCALE_LENGTH,
                    'scale_time': config.SCALE_TIME,
                    'reynolds_number': self.calculate_flow_characteristics().get('reynolds_number', 0)
                }
            }
            
            # 保存為多種格式
            base_filename = f'cfd_data_export_step_{step_num:04d}'
            
            # JSON格式（通用）
            json_file = f'{base_filename}.json'
            with open(json_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            # 嘗試保存為NumPy格式（如果可用）
            numpy_file = f'{base_filename}.npz'
            np.savez_compressed(numpy_file, **{
                k: np.array(v) for k, v in export_data.items() 
                if isinstance(v, (list, np.ndarray))
            })
            
            print(f"📊 數據導出完成:")
            print(f"   📄 {json_file} (JSON格式)")
            print(f"   📄 {numpy_file} (NumPy格式)")
            
            return [json_file, numpy_file]
            
        except Exception as e:
            print(f"❌ 數據導出失敗: {e}")
            return []
    
    # === 保持向後兼容性的舊版函數 ===
    
    def save_longitudinal_analysis(self, simulation_time, step_num):
        """保存縱向分析圖（修復版 - 添加顆粒和邊界可視化）"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 密度分析
            if hasattr(self.lbm, 'rho'):
                rho_data = self.lbm.rho.to_numpy()
                
                # 使用安全的數據處理
                rho_data = np.nan_to_num(rho_data, nan=1.0, posinf=1.0, neginf=0.0)
                rho_data = np.clip(rho_data, 0.0, 2.0)  # 限制密度範圍
                
                z_slice = rho_data[:, config.NY//2, :]
                
                im1 = ax1.imshow(z_slice.T, origin='lower', aspect='auto', cmap=self.density_cmap, vmin=0.0, vmax=1.5)
                ax1.set_title(f'Density Profile (t={simulation_time:.2f}s)', fontsize=12)
                ax1.set_xlabel('X Position')
                ax1.set_ylabel('Z Position')
                plt.colorbar(im1, ax=ax1)
                
                # 添加V60輪廓和邊界
                self._add_v60_outline_fixed(ax1, 'xz')
                
                # 添加顆粒可視化
                self._add_particles_to_plot(ax1, 'xz', config.NY//2)
                
                # 速度分析
                if hasattr(self.lbm, 'u'):
                    u_data = self.lbm.u.to_numpy()
                    u_data = np.nan_to_num(u_data, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    u_magnitude = np.sqrt(u_data[:, :, :, 0]**2 + u_data[:, :, :, 1]**2 + u_data[:, :, :, 2]**2)
                    u_magnitude = np.clip(u_magnitude, 0.0, 0.5)  # 限制速度範圍
                    u_slice = u_magnitude[:, config.NY//2, :]
                    
                    im2 = ax2.imshow(u_slice.T, origin='lower', aspect='auto', cmap=self.velocity_cmap, vmin=0.0, vmax=0.1)
                    ax2.set_title(f'Velocity Magnitude (t={simulation_time:.2f}s)', fontsize=12)
                    ax2.set_xlabel('X Position')
                    ax2.set_ylabel('Z Position')
                    plt.colorbar(im2, ax=ax2)
                    
                    # 添加V60輪廓
                    self._add_v60_outline_fixed(ax2, 'xz')
                    
                    # 添加顆粒可視化
                    self._add_particles_to_plot(ax2, 'xz', config.NY//2)
            
            filename = f'v60_longitudinal_analysis_step_{step_num:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Could not save longitudinal analysis: {e}")
            return None
    
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
            print(f"Warning: Could not save velocity analysis: {e}")
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
            print(f"Warning: Could not save combined analysis: {e}")
            return None
    
    def display_longitudinal_animation(self):
        """顯示縱向動畫（兼容性函數）"""
        print("🎬 動畫功能已整合到科研級分析中")
        print("💡 使用 generate_research_report() 獲得完整分析")
        print("📊 使用 save_temporal_analysis() 獲得時間序列分析")
    
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