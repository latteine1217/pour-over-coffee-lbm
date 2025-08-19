# enhanced_visualizer.py
"""
科研級增強視覺化系統 - 專為咖啡萃取CFD研究設計
提供多物理場分析、量化統計、時間序列追蹤等功能
"""

# 標準庫導入
import json
import os
import time
from datetime import datetime

# 第三方庫導入
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from scipy import ndimage
from scipy.stats import pearsonr

# 本地模組導入
import config.config as config

class EnhancedVisualizer:
    def __init__(self, lbm_solver, multiphase=None, geometry=None, particle_system=None, filter_system=None, simulation=None):
        """
        科研級視覺化系統初始化
        
        Args:
            lbm_solver: LBM求解器
            multiphase: 多相流系統  
            geometry: 幾何系統
            particle_system: 咖啡顆粒系統
            filter_system: 濾紙系統
            simulation: 主模擬系統引用（用於診斷數據訪問）
        """
        self.lbm = lbm_solver
        self.multiphase = multiphase
        self.geometry = geometry
        self.particles = particle_system
        self.filter = filter_system
        self.simulation = simulation  # 新增：用於訪問診斷數據
        
        # 輸出目錄管理
        self.report_dir = None
        self.session_timestamp = None
        self._setup_output_directory()
        
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
        
        # 時序數據存儲系統
        self.time_series_data = {
            'step_numbers': [],
            'physical_times': [],
            'reynolds_numbers': [],
            'pressure_drops': [],
            'max_velocities': [],
            'mean_velocities': [],
            'turbulent_kinetic_energy': [],
            'interface_area': [],
            'extraction_efficiency': [],
            'pressure_gradients': [],
            'vorticity_magnitude': [],
            'mass_flow_rates': []
        }
        
        # 視覺化增強參數
        self.viz_config = {
            'dynamic_range': True,
            'percentile_range': (5, 95),
            'time_series_buffer': 1000,
            'difference_analysis': True,
            'adaptive_colorbar': True
        }
        
        # 專業配色方案
        self.setup_colormaps()
        
        # 分析區域定義
        self.define_analysis_regions()
        
        print("🔬 科研級增強視覺化系統已初始化")
        print(f"   └─ 報告目錄: {self.report_dir}")
        print(f"   └─ 多物理場分析: {'✅' if multiphase else '❌'}")
        print(f"   └─ 顆粒追蹤: {'✅' if particle_system else '❌'}")
        print(f"   └─ 濾紙分析: {'✅' if filter_system else '❌'}")

    def _calculate_dynamic_range(self, data, percentile_low=5, percentile_high=95):
        """
        計算動態範圍，排除極值影響，提升視覺化效果
        
        Args:
            data: 數據數組
            percentile_low: 下限百分位數
            percentile_high: 上限百分位數
            
        Returns:
            tuple: (vmin, vmax) 動態範圍
        """
        # 過濾有效數據（排除NaN和Inf）
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return 0, 1
        
        # 使用百分位數確定範圍，排除極值干擾
        vmin = np.percentile(valid_data, percentile_low)
        vmax = np.percentile(valid_data, percentile_high)
        
        # 確保範圍有效
        if vmax <= vmin:
            vmax = vmin + 1e-10
        
        return vmin, vmax

    def _create_smart_colorbar(self, ax, im, data, title="", units="", include_stats=True):
        """
        創建智能colorbar，包含統計信息和動態範圍
        
        Args:
            ax: matplotlib axes對象
            im: imshow對象
            data: 原始數據
            title: colorbar標題
            units: 物理單位
            include_stats: 是否包含統計信息
            
        Returns:
            colorbar對象
        """
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 設置標籤
        if units:
            cbar.set_label(f'{title} [{units}]', fontsize=10)
        else:
            cbar.set_label(title, fontsize=10)
        
        # 添加統計信息
        if include_stats:
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                
                # 格式化統計信息
                if abs(mean_val) > 1000 or abs(mean_val) < 0.01:
                    stats_text = f'μ={mean_val:.2e}\nσ={std_val:.2e}\nmin={min_val:.2e}\nmax={max_val:.2e}'
                else:
                    stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={min_val:.3f}\nmax={max_val:.3f}'
                
                cbar.ax.text(1.05, 0.5, stats_text, 
                           transform=cbar.ax.transAxes, 
                           fontsize=8, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        return cbar

    def _collect_time_series_data(self, step_num):
        """
        收集關鍵參數的時序數據
        
        Args:
            step_num: 當前時間步數
        """
        try:
            # 計算物理時間
            physical_time = step_num * config.SCALE_TIME
            
            # 收集流體力學特徵
            flow_chars = self.calculate_flow_characteristics()
            
            # 添加到時序數據
            self.time_series_data['step_numbers'].append(step_num)
            self.time_series_data['physical_times'].append(physical_time)
            
            # Reynolds數
            reynolds = flow_chars.get('reynolds_number', 0)
            self.time_series_data['reynolds_numbers'].append(reynolds)
            
            # 壓力分析
            pressure_analysis = flow_chars.get('pressure_analysis', {})
            pressure_drop = pressure_analysis.get('pressure_drop_total', 0)
            self.time_series_data['pressure_drops'].append(pressure_drop)
            
            # 速度統計
            if hasattr(self.lbm, 'u'):
                u_data = self.lbm.u.to_numpy()
                u_mag = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
                max_vel = np.max(u_mag) * config.SCALE_VELOCITY
                mean_vel = np.mean(u_mag[u_mag > 1e-6]) * config.SCALE_VELOCITY if np.any(u_mag > 1e-6) else 0
                
                self.time_series_data['max_velocities'].append(max_vel)
                self.time_series_data['mean_velocities'].append(mean_vel)
            else:
                self.time_series_data['max_velocities'].append(0)
                self.time_series_data['mean_velocities'].append(0)
            
            # 湍流特徵
            turbulence_analysis = flow_chars.get('turbulence_analysis', {})
            tke = turbulence_analysis.get('turbulent_kinetic_energy', 0)
            self.time_series_data['turbulent_kinetic_energy'].append(tke)
            
            # 多相流界面面積
            if self.multiphase and hasattr(self.multiphase, 'phi'):
                phi_data = self.multiphase.phi.to_numpy()
                # 計算界面面積（基於phi梯度）
                grad_phi = np.gradient(phi_data)
                interface_area = np.sum(np.sqrt(sum(g**2 for g in grad_phi)))
                self.time_series_data['interface_area'].append(interface_area)
            else:
                self.time_series_data['interface_area'].append(0)
            
            # 限制緩衝區大小
            buffer_size = self.viz_config['time_series_buffer']
            for key in self.time_series_data:
                if len(self.time_series_data[key]) > buffer_size:
                    self.time_series_data[key] = self.time_series_data[key][-buffer_size:]
                    
        except Exception as e:
            print(f"Warning: Time series data collection failed: {e}")

    def save_time_series_analysis(self, step_num):
        """
        保存關鍵參數時序分析圖
        
        Args:
            step_num: 當前時間步數
            
        Returns:
            str: 保存的文件路徑
        """
        try:
            # 收集當前步的數據
            self._collect_time_series_data(step_num)
            
            # 檢查是否有足夠的數據
            if len(self.time_series_data['step_numbers']) < 2:
                return None
            
            # 創建時序分析圖
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle(f'關鍵參數時序分析 - Step {step_num}', fontsize=16)
            
            steps = self.time_series_data['step_numbers']
            times = self.time_series_data['physical_times']
            
            # 1. Reynolds數演化
            ax1 = axes[0, 0]
            reynolds = self.time_series_data['reynolds_numbers']
            ax1.plot(steps, reynolds, 'b-', linewidth=2, label='Reynolds Number')
            ax1.set_title('Reynolds數時序演化')
            ax1.set_ylabel('Re')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. 壓力損失時序
            ax2 = axes[0, 1]
            pressure_drops = self.time_series_data['pressure_drops']
            ax2.plot(steps, pressure_drops, 'r-', linewidth=2, label='Pressure Drop')
            ax2.set_title('壓力損失時序變化')
            ax2.set_ylabel('ΔP [Pa]')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. 速度場統計
            ax3 = axes[1, 0]
            max_vels = self.time_series_data['max_velocities']
            mean_vels = self.time_series_data['mean_velocities']
            ax3.plot(steps, max_vels, 'g-', linewidth=2, label='Max Velocity')
            ax3.plot(steps, mean_vels, 'g--', linewidth=2, label='Mean Velocity')
            ax3.set_title('速度場統計')
            ax3.set_ylabel('Velocity [m/s]')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. 湍流動能
            ax4 = axes[1, 1]
            tke = self.time_series_data['turbulent_kinetic_energy']
            ax4.plot(steps, tke, 'm-', linewidth=2, label='Turbulent Kinetic Energy')
            ax4.set_title('湍流動能演化')
            ax4.set_ylabel('TKE [J/kg]')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # 5. 多相流界面演化
            ax5 = axes[2, 0]
            interface_areas = self.time_series_data['interface_area']
            ax5.plot(steps, interface_areas, 'c-', linewidth=2, label='Interface Area')
            ax5.set_title('多相流界面面積')
            ax5.set_ylabel('Interface Area')
            ax5.set_xlabel('Time Step')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            # 6. 系統收斂性分析
            ax6 = axes[2, 1]
            if len(reynolds) > 10:
                # 計算Reynolds數的變化率（數值穩定性指標）
                re_changes = np.abs(np.diff(reynolds[-10:]))  # 最近10步的變化
                ax6.plot(steps[-len(re_changes):], re_changes, 'orange', linewidth=2, label='Re Change Rate')
                ax6.set_title('數值收斂性分析')
                ax6.set_ylabel('|ΔRe|')
                ax6.set_xlabel('Time Step')
                ax6.grid(True, alpha=0.3)
                ax6.legend()
                
                # 添加收斂判斷
                recent_change = np.mean(re_changes[-5:]) if len(re_changes) >= 5 else float('inf')
                convergence_threshold = 0.01
                if recent_change < convergence_threshold:
                    ax6.text(0.05, 0.95, '✅ 已收斂', transform=ax6.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
                else:
                    ax6.text(0.05, 0.95, '⏳ 收斂中', transform=ax6.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
            
            plt.tight_layout()
            
            # 保存圖像
            filename = self.get_output_path(f'time_series_analysis_step_{step_num:04d}.png')
            self._safe_savefig(fig, filename, dpi=200)
            plt.close()
            
            # 同時保存數據到JSON
            data_filename = self.get_output_path(f'time_series_data_step_{step_num:04d}.json', 'data')
            with open(data_filename, 'w') as f:
                # 轉換numpy數組為列表以便JSON序列化
                json_data = {k: [float(x) for x in v] for k, v in self.time_series_data.items()}
                json.dump(json_data, f, indent=2)
            
            return filename
            
        except Exception as e:
            print(f"Warning: Time series analysis failed: {e}")
            return None

    def _safe_savefig(self, fig, filename, dpi=200, max_pixels=65000):
        """安全儲存圖像，動態限制DPI避免超大像素尺寸錯誤。

        - 控制 fig 寬高像素: width_px = width_in * dpi, height_px = height_in * dpi
        - 限制任一方向像素小於 max_pixels（matplotlib 上限 < 2^16）
        """
        try:
            w_in, h_in = fig.get_size_inches()
            # 計算可用的最大DPI
            max_dpi_w = int(max_pixels / max(w_in, 1e-3))
            max_dpi_h = int(max_pixels / max(h_in, 1e-3))
            safe_dpi = max(50, min(int(dpi), max_dpi_w, max_dpi_h, 300))  # 下限50, 上限300
            try:
                fig.tight_layout()
            except Exception:
                pass
            fig.savefig(filename, dpi=safe_dpi)
        except Exception as e:
            print(f"Warning: safe savefig failed ({e}), fallback to low DPI")
            try:
                fig.savefig(filename, dpi=100)
            except Exception as e2:
                print(f"Warning: fallback savefig failed: {e2}")

    def _setup_output_directory(self):
        """設置輸出目錄結構"""
        # 創建時間戳
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建報告目錄
        self.report_dir = f"report/{self.session_timestamp}"
        os.makedirs(self.report_dir, exist_ok=True)
        
        # 創建子目錄
        subdirs = ['images', 'data', 'analysis']
        for subdir in subdirs:
            os.makedirs(f"{self.report_dir}/{subdir}", exist_ok=True)
    
    def get_output_path(self, filename, subdir='images'):
        """獲取輸出路徑"""
        return f"{self.report_dir}/{subdir}/{filename}"
    
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
        self.temp_cmap = 'viridis'
    
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
        """計算流體力學特徵參數 (CFD工程師專業版)"""
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
        
        # ===== CFD工程師專業分析 =====
        
        # 1. 擴展無量綱數分析
        dimensionless_numbers = self._calculate_extended_dimensionless_numbers(
            u_mag_physical, characteristic_velocity, characteristic_length, kinematic_viscosity
        )
        
        # 2. 壓力場專業分析
        pressure_analysis = self._calculate_pressure_field_analysis(rho_data, u_data)
        
        # 3. 湍流特徵分析
        turbulence_analysis = self._calculate_turbulence_characteristics(u_data)
        
        # 4. 邊界層分析
        boundary_layer_analysis = self._calculate_boundary_layer_properties(u_data)
        
        # 5. 流動拓撲分析
        flow_topology = self._calculate_flow_topology(u_data)
        
        # 壓力場分析 (轉換為物理單位)
        pressure_lu = rho_data * config.CS2  # 格子單位壓力
        pressure_physical = pressure_lu * config.SCALE_DENSITY * config.SCALE_VELOCITY**2  # Pa
        
        # 壓力梯度計算（優先使用內部梯度場）
        try:
            if hasattr(self.lbm, 'grad_rho'):
                grad_rho = self.lbm.grad_rho.to_numpy()
                grad_p = grad_rho * config.CS2 * config.SCALE_DENSITY * (config.SCALE_VELOCITY**2)
            else:
                grad_p = np.gradient(pressure_physical)
        except Exception:
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
            # 基本參數
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
            },
            # ===== CFD工程師專業參數 =====
            'dimensionless_numbers': dimensionless_numbers,
            'pressure_analysis': pressure_analysis,
            'turbulence_analysis': turbulence_analysis,
            'boundary_layer_analysis': boundary_layer_analysis,
            'flow_topology': flow_topology
        }
    
    def _calculate_extended_dimensionless_numbers(self, u_mag_physical, u_char, l_char, nu):
        """計算擴展無量綱數"""
        try:
            # Capillary數 (表面張力效應)
            if hasattr(config, 'SURFACE_TENSION_PHYS') and config.SURFACE_TENSION_PHYS > 0:
                mu_phys = config.RHO_WATER * nu  # 動力黏滯度
                capillary_number = (mu_phys * u_char) / config.SURFACE_TENSION_PHYS
            else:
                capillary_number = 0.0
            
            # Bond數 (重力vs表面張力)
            if hasattr(config, 'SURFACE_TENSION_PHYS') and config.SURFACE_TENSION_PHYS > 0:
                bond_number = (config.RHO_WATER * config.GRAVITY_PHYS * l_char**2) / config.SURFACE_TENSION_PHYS
            else:
                bond_number = 0.0
            
            # Péclet數 (對流vs擴散)
            diffusivity = getattr(config, 'DIFFUSIVITY', 1e-9)  # 默認擴散係數
            if diffusivity > 0:
                peclet_number = (u_char * l_char) / diffusivity
            else:
                peclet_number = 0.0
            
            # 局部Reynolds數分佈
            local_reynolds = (u_mag_physical * l_char) / nu
            
            local_reynolds_positive = local_reynolds[local_reynolds > 0]
            
            return {
                'capillary_number': capillary_number,
                'bond_number': bond_number,
                'peclet_number': peclet_number,
                'local_reynolds_max': np.max(local_reynolds_positive) if local_reynolds_positive.size > 0 else 0.0,
                'local_reynolds_mean': np.mean(local_reynolds_positive) if local_reynolds_positive.size > 0 else 0.0,
                'local_reynolds_std': np.std(local_reynolds_positive) if local_reynolds_positive.size > 0 else 0.0,
                'local_reynolds_field': local_reynolds
            }
        except Exception as e:
            print(f"Warning: Extended dimensionless numbers calculation failed: {e}")
            return {}
    
    def _calculate_pressure_field_analysis(self, rho_data, u_data):
        """專業壓力場分析"""
        try:
            # 壓力場轉換為物理單位
            pressure_lu = rho_data * config.CS2
            pressure_physical = pressure_lu * config.SCALE_DENSITY * config.SCALE_VELOCITY**2  # Pa
            
            # 壓力梯度計算
            grad_p_x, grad_p_y, grad_p_z = np.gradient(pressure_physical)
            grad_p_magnitude = np.sqrt(grad_p_x**2 + grad_p_y**2 + grad_p_z**2)
            
            # 壓力係數 (Cp)
            u_mag = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
            u_max = np.max(u_mag) if u_mag.size > 0 else 0.0
            if u_max > 0:
                dynamic_pressure = 0.5 * config.RHO_WATER * (u_max * config.SCALE_VELOCITY)**2
                pressure_coefficient = (pressure_physical - np.mean(pressure_physical)) / dynamic_pressure
            else:
                pressure_coefficient = np.zeros_like(pressure_physical)
            
            # 沿程壓力損失
            pressure_profile = self._calculate_streamwise_pressure_profile(pressure_physical)
            
            return {
                'pressure_gradient_magnitude': grad_p_magnitude,
                'pressure_gradient_components': [grad_p_x, grad_p_y, grad_p_z],
                'pressure_coefficient': pressure_coefficient,
                'max_pressure_gradient': np.max(grad_p_magnitude) if grad_p_magnitude.size > 0 else 0.0,
                'pressure_drop_total': (np.max(pressure_physical) - np.min(pressure_physical)) if pressure_physical.size > 0 else 0.0,
                'pressure_profile': pressure_profile
            }
        except Exception as e:
            print(f"Warning: Pressure field analysis failed: {e}")
            return {}
    
    def _calculate_turbulence_characteristics(self, u_data):
        """湍流特徵分析"""
        try:
            # Q-criterion (渦流識別)
            q_criterion = self._calculate_q_criterion(u_data)
            
            # λ2-criterion (另一種渦流識別方法)
            lambda2_criterion = self._calculate_lambda2_criterion(u_data)
            
            # 湍流強度
            turbulence_intensity = self._calculate_turbulence_intensity(u_data)
            
            # 湍流耗散率估算
            dissipation_rate = self._estimate_dissipation_rate(u_data)
            
            return {
                'q_criterion': q_criterion,
                'lambda2_criterion': lambda2_criterion,
                'turbulence_intensity': turbulence_intensity,
                'dissipation_rate': dissipation_rate,
                'turbulent_kinetic_energy': np.mean(turbulence_intensity**2) * 1.5
            }
        except Exception as e:
            print(f"Warning: Turbulence analysis failed: {e}")
            return {}
    
    def _calculate_boundary_layer_properties(self, u_data):
        """邊界層特性分析"""
        try:
            # 近壁速度梯度
            wall_shear_stress = self._calculate_wall_shear_stress(u_data)
            
            # 邊界層厚度估算
            boundary_layer_thickness = self._estimate_boundary_layer_thickness(u_data)
            
            # 位移厚度和動量厚度
            displacement_thickness, momentum_thickness = self._calculate_boundary_layer_thicknesses(u_data)
            
            return {
                'wall_shear_stress': wall_shear_stress,
                'boundary_layer_thickness': boundary_layer_thickness,
                'displacement_thickness': displacement_thickness,
                'momentum_thickness': momentum_thickness,
                'shape_factor': displacement_thickness / momentum_thickness if momentum_thickness > 0 else 0
            }
        except Exception as e:
            print(f"Warning: Boundary layer analysis failed: {e}")
            return {}
    
    def _calculate_flow_topology(self, u_data):
        """流動拓撲分析"""
        try:
            # 流線曲率
            streamline_curvature = self._calculate_streamline_curvature(u_data)
            
            # 流動分離點識別
            separation_points = self._identify_separation_points(u_data)
            
            # 駐點和鞍點識別
            critical_points = self._identify_critical_points(u_data)
            
            return {
                'streamline_curvature': streamline_curvature,
                'separation_points': separation_points,
                'critical_points': critical_points
            }
        except Exception as e:
            print(f"Warning: Flow topology analysis failed: {e}")
            return {}
    
    # ===== 輔助計算方法 =====
    
    def _calculate_q_criterion(self, u_data):
        """計算Q-criterion (渦流識別)"""
        try:
            # 速度梯度張量
            dudx = np.gradient(u_data[:,:,:,0], axis=0)
            dudy = np.gradient(u_data[:,:,:,0], axis=1)
            dudz = np.gradient(u_data[:,:,:,0], axis=2)
            dvdx = np.gradient(u_data[:,:,:,1], axis=0)
            dvdy = np.gradient(u_data[:,:,:,1], axis=1)
            dvdz = np.gradient(u_data[:,:,:,1], axis=2)
            dwdx = np.gradient(u_data[:,:,:,2], axis=0)
            dwdy = np.gradient(u_data[:,:,:,2], axis=1)
            dwdz = np.gradient(u_data[:,:,:,2], axis=2)
            
            # 應變率張量 S 和渦度張量 Ω
            S11, S22, S33 = dudx, dvdy, dwdz
            S12 = 0.5 * (dudy + dvdx)
            S13 = 0.5 * (dudz + dwdx)
            S23 = 0.5 * (dvdz + dwdy)
            
            O12 = 0.5 * (dudy - dvdx)
            O13 = 0.5 * (dudz - dwdx)
            O23 = 0.5 * (dvdz - dwdy)
            
            # Q = 0.5 * (|Ω|² - |S|²)
            S_magnitude_sq = S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2)
            O_magnitude_sq = 2*(O12**2 + O13**2 + O23**2)
            
            Q = 0.5 * (O_magnitude_sq - S_magnitude_sq)
            
            return Q
        except Exception as e:
            print(f"Warning: Q-criterion calculation failed: {e}")
            return np.zeros_like(u_data[:,:,:,0])
    
    def _calculate_lambda2_criterion(self, u_data):
        """計算λ2-criterion"""
        try:
            # 簡化版：使用渦度大小作為近似
            omega_x = np.gradient(u_data[:,:,:,2], axis=1) - np.gradient(u_data[:,:,:,1], axis=2)
            omega_y = np.gradient(u_data[:,:,:,0], axis=2) - np.gradient(u_data[:,:,:,2], axis=0)
            omega_z = np.gradient(u_data[:,:,:,1], axis=0) - np.gradient(u_data[:,:,:,0], axis=1)
            
            lambda2 = -(omega_x**2 + omega_y**2 + omega_z**2)
            
            return lambda2
        except Exception as e:
            print(f"Warning: λ2-criterion calculation failed: {e}")
            return np.zeros_like(u_data[:,:,:,0])
    
    def _calculate_turbulence_intensity(self, u_data):
        """計算湍流強度"""
        try:
            u_mag = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
            u_mean = np.mean(u_mag)
            
            # 簡化：使用速度波動近似湍流強度
            turbulence_intensity = np.abs(u_mag - u_mean) / (u_mean + 1e-10)
            
            return turbulence_intensity
        except Exception as e:
            print(f"Warning: Turbulence intensity calculation failed: {e}")
            return np.zeros_like(u_data[:,:,:,0])
    
    def _estimate_dissipation_rate(self, u_data):
        """估算湍流耗散率"""
        try:
            # 簡化：使用速度梯度估算
            dudx = np.gradient(u_data[:,:,:,0], axis=0)
            dudy = np.gradient(u_data[:,:,:,0], axis=1)
            dudz = np.gradient(u_data[:,:,:,0], axis=2)
            
            dissipation = config.NU_CHAR * (dudx**2 + dudy**2 + dudz**2)
            
            return dissipation
        except Exception as e:
            print(f"Warning: Dissipation rate calculation failed: {e}")
            return np.zeros_like(u_data[:,:,:,0])
    
    def _calculate_wall_shear_stress(self, u_data):
        """計算壁面剪應力"""
        try:
            # 在V60壁面附近計算剪應力
            center_x, center_y = config.NX//2, config.NY//2
            radius = config.TOP_RADIUS / config.SCALE_LENGTH
            
            # 簡化：在半徑處計算速度梯度
            wall_shear = np.zeros_like(u_data[:,:,:,0])
            
            for i in range(config.NX):
                for j in range(config.NY):
                    for k in range(config.NZ):
                        dist_from_center = np.sqrt((i-center_x)**2 + (j-center_y)**2)
                        if abs(dist_from_center - radius) < 2:  # 近壁區域
                            # 計算法向速度梯度
                            if i > 0 and i < config.NX-1:
                                wall_shear[i,j,k] = config.NU_CHAR * (u_data[i+1,j,k,0] - u_data[i-1,j,k,0]) / 2
            
            return wall_shear
        except Exception as e:
            print(f"Warning: Wall shear stress calculation failed: {e}")
            return np.zeros_like(u_data[:,:,:,0])
    
    def _estimate_boundary_layer_thickness(self, u_data):
        """估算邊界層厚度"""
        try:
            # 簡化：使用99%自由流速度定義
            u_mag = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
            u_max = np.max(u_mag)
            
            # 邊界層厚度定義為速度達到99%自由流的距離
            boundary_layer_thickness = np.zeros((config.NX, config.NY))
            
            for i in range(config.NX):
                for j in range(config.NY):
                    velocity_profile = u_mag[i, j, :]
                    threshold = 0.99 * u_max
                    
                    # 找到第一個超過閾值的點
                    indices = np.where(velocity_profile > threshold)[0]
                    if len(indices) > 0:
                        boundary_layer_thickness[i, j] = indices[0]
            
            return boundary_layer_thickness
        except Exception as e:
            print(f"Warning: Boundary layer thickness calculation failed: {e}")
            return np.zeros((config.NX, config.NY))
    
    def _calculate_boundary_layer_thicknesses(self, u_data):
        """計算位移厚度和動量厚度"""
        try:
            # 簡化實現
            displacement_thickness = np.mean(self._estimate_boundary_layer_thickness(u_data)) * 0.3
            momentum_thickness = displacement_thickness * 0.37  # 層流邊界層近似
            
            return displacement_thickness, momentum_thickness
        except Exception as e:
            print(f"Warning: Boundary layer thicknesses calculation failed: {e}")
            return 0.0, 0.0
    
    def _calculate_streamline_curvature(self, u_data):
        """計算流線曲率"""
        try:
            # 使用速度方向變化率估算曲率
            u_mag = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
            
            # 單位速度向量
            u_unit = u_data / (u_mag[:,:,:,np.newaxis] + 1e-10)
            
            # 曲率近似：單位切向量的變化率
            curvature = np.sqrt(
                np.gradient(u_unit[:,:,:,0], axis=0)**2 + 
                np.gradient(u_unit[:,:,:,1], axis=1)**2 + 
                np.gradient(u_unit[:,:,:,2], axis=2)**2
            )
            
            return curvature
        except Exception as e:
            print(f"Warning: Streamline curvature calculation failed: {e}")
            return np.zeros_like(u_data[:,:,:,0])
    
    def _identify_separation_points(self, u_data):
        """識別流動分離點"""
        try:
            # 簡化：尋找壁面剪應力為零的點
            wall_shear = self._calculate_wall_shear_stress(u_data)
            
            # 分離點：剪應力接近零且有負值
            separation_mask = (np.abs(wall_shear) < 1e-6) & (wall_shear <= 0)
            separation_points = np.where(separation_mask)
            
            return {
                'count': len(separation_points[0]),
                'locations': list(zip(separation_points[0], separation_points[1], separation_points[2]))
            }
        except Exception as e:
            print(f"Warning: Separation points identification failed: {e}")
            return {'count': 0, 'locations': []}
    
    def _identify_critical_points(self, u_data):
        """識別臨界點"""
        try:
            # 尋找速度為零的點
            u_mag = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
            
            critical_mask = u_mag < 1e-6
            critical_points = np.where(critical_mask)
            
            return {
                'count': len(critical_points[0]),
                'locations': list(zip(critical_points[0], critical_points[1], critical_points[2]))
            }
        except Exception as e:
            print(f"Warning: Critical points identification failed: {e}")
            return {'count': 0, 'locations': []}
    
    def _calculate_streamwise_pressure_profile(self, pressure_field):
        """計算沿程壓力分佈"""
        try:
            # 沿Z方向（主流方向）的壓力分佈
            center_x, center_y = config.NX//2, config.NY//2
            
            pressure_profile = []
            for k in range(config.NZ):
                avg_pressure = np.mean(pressure_field[
                    center_x-5:center_x+5, 
                    center_y-5:center_y+5, 
                    k
                ])
                pressure_profile.append(avg_pressure)
            
            return pressure_profile
        except Exception as e:
            print(f"Warning: Streamwise pressure profile calculation failed: {e}")
            return []
    
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
        """生成完整的科研報告 - CFD工程師專業版"""
        # 計算真實物理時間
        physical_time = step_num * config.SCALE_TIME
        print(f"🔬 生成CFD工程師級分析報告 (t={physical_time:.2f}s)...")
        
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
            
        # 4. LBM診斷監控
        if hasattr(self, 'simulation') and hasattr(self.simulation, 'lbm_diagnostics'):
            lbm_file = self.save_lbm_monitoring_chart(simulation_time, step_num)
            if lbm_file:
                generated_files.append(lbm_file)
        
        # ===== CFD工程師專業分析 =====
        
        # 5. 壓力場專業分析
        pressure_file = self.save_pressure_field_analysis(simulation_time, step_num)
        if pressure_file:
            generated_files.append(pressure_file)
        
        # 6. 湍流特徵分析
        turbulence_file = self.save_turbulence_analysis(simulation_time, step_num)
        if turbulence_file:
            generated_files.append(turbulence_file)
        
        # 7. 無量綱數時序分析
        dimensionless_file = self.save_dimensionless_analysis(simulation_time, step_num)
        if dimensionless_file:
            generated_files.append(dimensionless_file)
        
        # 8. 邊界層分析 (每100步生成一次)
        if step_num % 100 == 0:
            boundary_file = self.save_boundary_layer_analysis(simulation_time, step_num)
            if boundary_file:
                generated_files.append(boundary_file)
        
        print(f"✅ CFD工程師級報告生成完成，共 {len(generated_files)} 個文件:")
        for file in generated_files:
            print(f"   📄 {file}")
        
        return generated_files
    
    def save_pressure_field_analysis(self, simulation_time, step_num):
        """保存壓力場專業分析圖"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 計算流體特徵
            flow_chars = self.calculate_flow_characteristics()
            pressure_analysis = flow_chars.get('pressure_analysis', {})
            physical_time = step_num * config.SCALE_TIME
            
            if hasattr(self.lbm, 'rho') and hasattr(self.lbm, 'u'):
                rho_data = self.lbm.rho.to_numpy()
                u_data = self.lbm.u.to_numpy()
                
                # 壓力場
                pressure_lu = rho_data * config.CS2
                pressure_physical = pressure_lu * config.SCALE_DENSITY * config.SCALE_VELOCITY**2
                
                # 1. 壓力場分佈 (XZ切面) - 使用動態範圍調整
                pressure_slice = pressure_physical[:, config.NY//2, :]
                if self.viz_config['dynamic_range']:
                    vmin, vmax = self._calculate_dynamic_range(pressure_slice, *self.viz_config['percentile_range'])
                else:
                    vmin, vmax = np.min(pressure_slice), np.max(pressure_slice)
                
                im1 = ax1.imshow(pressure_slice.T, origin='lower', aspect='auto', 
                               cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax1.set_title('Pressure Field (Pa)', fontsize=12)
                ax1.set_xlabel('X Position')
                ax1.set_ylabel('Z Position')
                self._create_smart_colorbar(ax1, im1, pressure_slice, 'Pressure', 'Pa')
                self._add_v60_outline_fixed(ax1, 'xz')
                
                # 2. 壓力梯度 - 使用動態範圍調整
                if 'pressure_gradient_magnitude' in pressure_analysis:
                    grad_p = pressure_analysis['pressure_gradient_magnitude']
                    grad_slice = grad_p[:, config.NY//2, :]
                    
                    if self.viz_config['dynamic_range']:
                        vmin_grad, vmax_grad = self._calculate_dynamic_range(grad_slice, 0, 95)  # 壓力梯度通常從0開始
                    else:
                        vmin_grad, vmax_grad = 0, np.max(grad_slice)
                    
                    im2 = ax2.imshow(grad_slice.T, origin='lower', aspect='auto', 
                                   cmap='plasma', vmin=vmin_grad, vmax=vmax_grad)
                    ax2.set_title('Pressure Gradient Magnitude (Pa/m)', fontsize=12)
                    ax2.set_xlabel('X Position')
                    ax2.set_ylabel('Z Position')
                    self._create_smart_colorbar(ax2, im2, grad_slice, '|∇P|', 'Pa/m')
                    self._add_v60_outline_fixed(ax2, 'xz')
                
                # 3. 壓力係數 - 使用智能範圍
                if 'pressure_coefficient' in pressure_analysis:
                    cp = pressure_analysis['pressure_coefficient']
                    cp_slice = cp[:, config.NY//2, :]
                    
                    if self.viz_config['dynamic_range']:
                        vmin_cp, vmax_cp = self._calculate_dynamic_range(cp_slice, *self.viz_config['percentile_range'])
                        # 確保Cp範圍對稱且合理
                        cp_max = max(abs(vmin_cp), abs(vmax_cp))
                        vmin_cp, vmax_cp = -cp_max, cp_max
                    else:
                        vmin_cp, vmax_cp = -2, 2
                    
                    im3 = ax3.imshow(cp_slice.T, origin='lower', aspect='auto', 
                                   cmap='RdBu_r', vmin=vmin_cp, vmax=vmax_cp)
                    ax3.set_title('Pressure Coefficient Cp', fontsize=12)
                    ax3.set_xlabel('X Position')
                    ax3.set_ylabel('Z Position')
                    self._create_smart_colorbar(ax3, im3, cp_slice, 'Cp', '-')
                    self._add_v60_outline_fixed(ax3, 'xz')
                
                # 4. 沿程壓力分佈
                if 'pressure_profile' in pressure_analysis:
                    pressure_profile = pressure_analysis['pressure_profile']
                    z_coords = np.arange(len(pressure_profile))
                    ax4.plot(pressure_profile, z_coords, 'b-', linewidth=2, label='Pressure Profile')
                    ax4.set_xlabel('Pressure (Pa)')
                    ax4.set_ylabel('Z Position')
                    ax4.set_title('Streamwise Pressure Distribution', fontsize=12)
                    ax4.grid(True)
                    ax4.legend()
                    
                    # 添加壓力損失標註
                    pressure_drop = pressure_analysis.get('pressure_drop_total', 0)
                    ax4.text(0.05, 0.95, f'ΔP = {pressure_drop:.2f} Pa', 
                           transform=ax4.transAxes, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
            
            plt.suptitle(f'CFD Pressure Field Analysis (t={physical_time:.2f}s)', fontsize=14)
            filename = self.get_output_path(f'cfd_pressure_analysis_step_{step_num:04d}.png')
            fig.suptitle(f'CFD Pressure Field Analysis - Step {step_num}', fontsize=14)
            self._safe_savefig(fig, filename, dpi=getattr(config, 'VIZ_DPI', 200))
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Pressure field analysis failed: {e}")
            return None
    
    def save_turbulence_analysis(self, simulation_time, step_num):
        """保存湍流特徵分析圖"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 計算流體特徵
            flow_chars = self.calculate_flow_characteristics()
            turbulence_analysis = flow_chars.get('turbulence_analysis', {})
            physical_time = step_num * config.SCALE_TIME
            
            if hasattr(self.lbm, 'u'):
                u_data = self.lbm.u.to_numpy()
                
                # 1. Q-criterion
                if 'q_criterion' in turbulence_analysis:
                    q_field = turbulence_analysis['q_criterion']
                    q_slice = q_field[:, config.NY//2, :]
                    # 只顯示正值區域 (渦流區域)
                    q_positive = np.where(q_slice > 0, q_slice, 0)
                    im1 = ax1.imshow(q_positive.T, origin='lower', aspect='auto', 
                                   cmap='viridis', vmin=0, vmax=np.percentile(q_positive[q_positive>0], 90) if np.any(q_positive>0) else 1)
                    ax1.set_title('Q-Criterion (Vortex Identification)', fontsize=12)
                    ax1.set_xlabel('X Position')
                    ax1.set_ylabel('Z Position')
                    plt.colorbar(im1, ax=ax1)
                    self._add_v60_outline_fixed(ax1, 'xz')
                
                # 2. λ2-criterion
                if 'lambda2_criterion' in turbulence_analysis:
                    lambda2_field = turbulence_analysis['lambda2_criterion']
                    lambda2_slice = lambda2_field[:, config.NY//2, :]
                    # 只顯示負值區域 (渦流區域)
                    lambda2_negative = np.where(lambda2_slice < 0, -lambda2_slice, 0)
                    im2 = ax2.imshow(lambda2_negative.T, origin='lower', aspect='auto', 
                                   cmap='plasma', vmin=0, vmax=np.percentile(lambda2_negative[lambda2_negative>0], 90) if np.any(lambda2_negative>0) else 1)
                    ax2.set_title('λ2-Criterion (Vortex Identification)', fontsize=12)
                    ax2.set_xlabel('X Position')
                    ax2.set_ylabel('Z Position')
                    plt.colorbar(im2, ax=ax2)
                    self._add_v60_outline_fixed(ax2, 'xz')
                
                # 3. 湍流強度
                if 'turbulence_intensity' in turbulence_analysis:
                    ti_field = turbulence_analysis['turbulence_intensity']
                    ti_slice = ti_field[:, config.NY//2, :]
                    im3 = ax3.imshow(ti_slice.T, origin='lower', aspect='auto', 
                                   cmap='hot', vmin=0, vmax=np.percentile(ti_slice, 95))
                    ax3.set_title('Turbulence Intensity', fontsize=12)
                    ax3.set_xlabel('X Position')
                    ax3.set_ylabel('Z Position')
                    plt.colorbar(im3, ax=ax3)
                    self._add_v60_outline_fixed(ax3, 'xz')
                
                # 4. 耗散率
                if 'dissipation_rate' in turbulence_analysis:
                    dissipation_field = turbulence_analysis['dissipation_rate']
                    dissipation_slice = dissipation_field[:, config.NY//2, :]
                    im4 = ax4.imshow(dissipation_slice.T, origin='lower', aspect='auto', 
                                   cmap='inferno', vmin=0, vmax=np.percentile(dissipation_slice, 95))
                    ax4.set_title('Turbulent Dissipation Rate', fontsize=12)
                    ax4.set_xlabel('X Position')
                    ax4.set_ylabel('Z Position')
                    plt.colorbar(im4, ax=ax4)
                    self._add_v60_outline_fixed(ax4, 'xz')
                    
                    # 添加湍流統計
                    tke = turbulence_analysis.get('turbulent_kinetic_energy', 0)
                    ax4.text(0.05, 0.95, f'TKE = {tke:.2e}', 
                           transform=ax4.transAxes, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
            
            plt.suptitle(f'CFD Turbulence Analysis (t={physical_time:.2f}s)', fontsize=14)
            filename = self.get_output_path(f'cfd_turbulence_analysis_step_{step_num:04d}.png')
            fig.suptitle(f'CFD Turbulence Analysis - Step {step_num}', fontsize=14)
            self._safe_savefig(fig, filename, dpi=getattr(config, 'VIZ_DPI', 200))
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Turbulence analysis failed: {e}")
            return None
    
    def save_dimensionless_analysis(self, simulation_time, step_num):
        """保存無量綱數分析圖"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 計算流體特徵
            flow_chars = self.calculate_flow_characteristics()
            dimensionless = flow_chars.get('dimensionless_numbers', {})
            physical_time = step_num * config.SCALE_TIME
            
            # 1. 局部Reynolds數分佈
            if 'local_reynolds_field' in dimensionless:
                re_field = dimensionless['local_reynolds_field']
                re_slice = re_field[:, config.NY//2, :]
                im1 = ax1.imshow(re_slice.T, origin='lower', aspect='auto', 
                               cmap='viridis', vmin=0, vmax=np.percentile(re_slice, 95))
                ax1.set_title('Local Reynolds Number', fontsize=12)
                ax1.set_xlabel('X Position')
                ax1.set_ylabel('Z Position')
                plt.colorbar(im1, ax=ax1)
                self._add_v60_outline_fixed(ax1, 'xz')
                
                # 添加統計信息
                re_max = dimensionless.get('local_reynolds_max', 0)
                re_mean = dimensionless.get('local_reynolds_mean', 0)
                ax1.text(0.05, 0.95, f'Re_max = {re_max:.1f}\nRe_mean = {re_mean:.1f}', 
                       transform=ax1.transAxes, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # 2. 無量綱數柱狀圖
            dimensionless_values = [
                ('Re', flow_chars.get('reynolds_number', 0)),
                ('We', flow_chars.get('weber_number', 0)),
                ('Fr', flow_chars.get('froude_number', 0)),
                ('Ca', dimensionless.get('capillary_number', 0)),
                ('Bo', dimensionless.get('bond_number', 0)),
                ('Pe', dimensionless.get('peclet_number', 0))
            ]
            
            names = [item[0] for item in dimensionless_values]
            values = [item[1] for item in dimensionless_values]
            
            bars = ax2.bar(names, values, color=['blue', 'red', 'green', 'orange', 'purple', 'brown'])
            ax2.set_title('Dimensionless Numbers', fontsize=12)
            ax2.set_ylabel('Value')
            ax2.set_yscale('log')
            
            # 添加數值標籤
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2e}', ha='center', va='bottom', fontsize=8)
            
            # 3. 流動特徵圖
            if hasattr(self.lbm, 'u'):
                u_data = self.lbm.u.to_numpy()
                flow_topology = flow_chars.get('flow_topology', {})
                
                # 流線曲率
                if 'streamline_curvature' in flow_topology:
                    curvature = flow_topology['streamline_curvature']
                    curvature_slice = curvature[:, config.NY//2, :]
                    im3 = ax3.imshow(curvature_slice.T, origin='lower', aspect='auto', 
                                   cmap='coolwarm', vmin=0, vmax=np.percentile(curvature_slice, 95))
                    ax3.set_title('Streamline Curvature', fontsize=12)
                    ax3.set_xlabel('X Position')
                    ax3.set_ylabel('Z Position')
                    plt.colorbar(im3, ax=ax3)
                    self._add_v60_outline_fixed(ax3, 'xz')
                    
                    # 添加分離點標記
                    separation_points = flow_topology.get('separation_points', {})
                    if separation_points.get('count', 0) > 0:
                        for loc in separation_points['locations'][:10]:  # 最多顯示10個點
                            if loc[1] == config.NY//2:  # 只顯示當前切面的點
                                ax3.plot(loc[0], loc[2], 'ro', markersize=6, label='Separation')
            
            # 4. CFD質量指標
            conservation = flow_chars.get('mass_conservation', {})
            stability_metrics = [
                ('Mass Variation', conservation.get('mass_variation_coefficient', 0)),
                ('Max Velocity', flow_chars.get('max_velocity_physical', 0)),
                ('Pressure Drop', flow_chars.get('pressure_drop_pa', 0) / 1000),  # kPa
            ]
            
            metric_names = [item[0] for item in stability_metrics]
            metric_values = [item[1] for item in stability_metrics]
            
            bars4 = ax4.bar(metric_names, metric_values, color=['cyan', 'magenta', 'yellow'])
            ax4.set_title('CFD Quality Metrics', fontsize=12)
            ax4.set_ylabel('Value')
            
            # 添加數值標籤
            for bar, value in zip(bars4, metric_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle(f'CFD Dimensionless Analysis (t={physical_time:.2f}s)', fontsize=14)
            filename = self.get_output_path(f'cfd_dimensionless_analysis_step_{step_num:04d}.png')
            fig.suptitle(f'CFD Dimensionless Numbers Analysis - Step {step_num}', fontsize=14)
            # 可關閉重型圖（效能模式）
            if not getattr(config, 'VIZ_HEAVY', False):
                plt.close()
                return None
            self._safe_savefig(fig, filename, dpi=getattr(config, 'VIZ_DPI', 200))
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Dimensionless analysis failed: {e}")
            return None
    
    def save_boundary_layer_analysis(self, simulation_time, step_num):
        """保存邊界層分析圖"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 計算流體特徵
            flow_chars = self.calculate_flow_characteristics()
            boundary_analysis = flow_chars.get('boundary_layer_analysis', {})
            physical_time = step_num * config.SCALE_TIME
            
            if hasattr(self.lbm, 'u'):
                u_data = self.lbm.u.to_numpy()
                
                # 1. 邊界層厚度分佈
                if 'boundary_layer_thickness' in boundary_analysis:
                    bl_thickness = boundary_analysis['boundary_layer_thickness']
                    im1 = ax1.imshow(bl_thickness.T, origin='lower', aspect='equal', 
                                   cmap='viridis', vmin=0, vmax=np.percentile(bl_thickness, 95))
                    ax1.set_title('Boundary Layer Thickness', fontsize=12)
                    ax1.set_xlabel('X Position')
                    ax1.set_ylabel('Y Position')
                    plt.colorbar(im1, ax=ax1)
                    self._add_v60_outline_fixed(ax1, 'xy')
                
                # 2. 壁面剪應力
                if 'wall_shear_stress' in boundary_analysis:
                    wall_shear = boundary_analysis['wall_shear_stress']
                    shear_slice = wall_shear[:, config.NY//2, :]
                    im2 = ax2.imshow(shear_slice.T, origin='lower', aspect='auto', 
                                   cmap='plasma', vmin=0, vmax=np.percentile(shear_slice, 95))
                    ax2.set_title('Wall Shear Stress', fontsize=12)
                    ax2.set_xlabel('X Position')
                    ax2.set_ylabel('Z Position')
                    plt.colorbar(im2, ax=ax2)
                    self._add_v60_outline_fixed(ax2, 'xz')
                
                # 3. 速度剖面示例
                center_x, center_y = config.NX//2, config.NY//2
                radius_pos = int(center_x + config.TOP_RADIUS / config.SCALE_LENGTH * 0.7)
                
                if radius_pos < config.NX:
                    velocity_profile = np.sqrt(
                        u_data[radius_pos, center_y, :]**2 + 
                        u_data[radius_pos, center_y, :]**2 + 
                        u_data[radius_pos, center_y, :]**2
                    )
                    z_coords = np.arange(len(velocity_profile))
                    
                    ax3.plot(velocity_profile, z_coords, 'b-', linewidth=2, label='Velocity Profile')
                    ax3.set_xlabel('Velocity (lu/ts)')
                    ax3.set_ylabel('Z Position')
                    ax3.set_title('Near-Wall Velocity Profile', fontsize=12)
                    ax3.grid(True)
                    ax3.legend()
                
                # 4. 邊界層參數統計
                displacement_thickness = boundary_analysis.get('displacement_thickness', 0)
                momentum_thickness = boundary_analysis.get('momentum_thickness', 0)
                shape_factor = boundary_analysis.get('shape_factor', 0)
                
                bl_params = ['δ* (Displacement)', 'θ (Momentum)', 'H (Shape Factor)']
                bl_values = [displacement_thickness, momentum_thickness, shape_factor]
                
                bars = ax4.bar(bl_params, bl_values, color=['blue', 'red', 'green'])
                ax4.set_title('Boundary Layer Parameters', fontsize=12)
                ax4.set_ylabel('Value')
                
                # 添加數值標籤
                for bar, value in zip(bars, bl_values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle(f'CFD Boundary Layer Analysis (t={physical_time:.2f}s)', fontsize=14)
            filename = self.get_output_path(f'cfd_boundary_layer_analysis_step_{step_num:04d}.png')
            fig.suptitle(f'CFD Boundary Layer Analysis - Step {step_num}', fontsize=14)
            self._safe_savefig(fig, filename, dpi=getattr(config, 'VIZ_DPI', 200))
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Boundary layer analysis failed: {e}")
            return None
    
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
                export_data['pressure_field'] = (rho_data * config.CS2).tolist()
            
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
            
            # 計算真實物理時間
            physical_time = step_num * config.SCALE_TIME
            
            # 密度分析
            if hasattr(self.lbm, 'rho'):
                rho_data = self.lbm.rho.to_numpy()
                
                # 使用安全的數據處理
                rho_data = np.nan_to_num(rho_data, nan=1.0, posinf=1.0, neginf=0.0)
                rho_data = np.clip(rho_data, 0.0, 2.0)  # 限制密度範圍
                
                z_slice = rho_data[:, config.NY//2, :]
                
                im1 = ax1.imshow(z_slice.T, origin='lower', aspect='auto', cmap=self.density_cmap, vmin=0.0, vmax=1.5)
                ax1.set_title(f'Density Profile (t={physical_time:.2f}s)', fontsize=12)
                ax1.set_xlabel('X Position')
                ax1.set_ylabel('Z Position')
                plt.colorbar(im1, ax=ax1)
                
                # 添加V60輪廓和邊界
                self._add_v60_outline_fixed(ax1, 'xz')
                
                # 添加濾紙虛線
                self._add_filter_paper_outline(ax1, 'xz')
                
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
                    ax2.set_title(f'Velocity Magnitude (t={physical_time:.2f}s)', fontsize=12)
                    ax2.set_xlabel('X Position')
                    ax2.set_ylabel('Z Position')
                    plt.colorbar(im2, ax=ax2)
                    
                    # 添加V60輪廓
                    self._add_v60_outline_fixed(ax2, 'xz')
                    
                    # 添加濾紙虛線
                    self._add_filter_paper_outline(ax2, 'xz')
                    
                    # 添加顆粒可視化
                    self._add_particles_to_plot(ax2, 'xz', config.NY//2)
            
            filename = self.get_output_path(f'v60_longitudinal_analysis_step_{step_num:04d}.png')
            fig.suptitle(f'V60 Longitudinal Analysis - Step {step_num}', fontsize=14)
            # 可關閉重型圖（效能模式）
            if not getattr(config, 'VIZ_HEAVY', False):
                plt.close()
                return None
            self._safe_savefig(fig, filename, dpi=getattr(config, 'VIZ_DPI', 200))
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Could not save longitudinal analysis: {e}")
            return None
    
    def save_velocity_analysis(self, simulation_time, step_num):
        """保存速度分析圖（修復版）"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # 計算真實物理時間
            physical_time = step_num * config.SCALE_TIME
            
            if hasattr(self.lbm, 'u'):
                u_data = self.lbm.u.to_numpy()
                u_data = np.nan_to_num(u_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                u_magnitude = np.sqrt(u_data[:, :, :, 0]**2 + u_data[:, :, :, 1]**2 + u_data[:, :, :, 2]**2)
                u_magnitude = np.clip(u_magnitude, 0.0, 0.5)  # 限制速度範圍
                
                # 取XY平面切片
                z_level = config.NZ // 2
                u_slice = u_magnitude[:, :, z_level]
                
                im = ax.imshow(u_slice.T, origin='lower', aspect='equal', cmap=self.velocity_cmap, vmin=0.0, vmax=0.1)
                ax.set_title(f'Velocity Field (t={physical_time:.2f}s, Z={z_level})', fontsize=12)
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                plt.colorbar(im, ax=ax)
                
                # 添加V60頂視圖輪廓
                self._add_v60_outline_fixed(ax, 'xy')
                
                # 添加濾紙虛線
                self._add_filter_paper_outline(ax, 'xy')
                
                # 添加顆粒可視化
                self._add_particles_to_plot(ax, 'xy', z_level)
            
            filename = self.get_output_path(f'velocity_analysis_step_{step_num:04d}.png')
            fig.suptitle(f'Velocity Field Analysis - Step {step_num}', fontsize=14)
            self._safe_savefig(fig, filename, dpi=200)
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Could not save velocity analysis: {e}")
            return None
    
    def save_combined_analysis(self, simulation_time, step_num):
        """保存組合分析圖（修復版）"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # 計算真實物理時間
            physical_time = step_num * config.SCALE_TIME
            
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
                self._add_filter_paper_outline(ax1, 'xz')
                self._add_particles_to_plot(ax1, 'xz', config.NY//2)
                
                # 速度 XZ切面
                z_slice_u = u_magnitude[:, config.NY//2, :]
                im2 = ax2.imshow(z_slice_u.T, origin='lower', aspect='auto', cmap=self.velocity_cmap, vmin=0.0, vmax=0.1)
                ax2.set_title('Velocity (XZ plane)', fontsize=10)
                plt.colorbar(im2, ax=ax2)
                self._add_v60_outline_fixed(ax2, 'xz')
                self._add_filter_paper_outline(ax2, 'xz')
                self._add_particles_to_plot(ax2, 'xz', config.NY//2)
                
                # 密度 XY切面
                xy_slice_rho = rho_data[:, :, config.NZ//2]
                im3 = ax3.imshow(xy_slice_rho.T, origin='lower', aspect='equal', cmap=self.density_cmap, vmin=0.0, vmax=1.5)
                ax3.set_title('Density (XY plane)', fontsize=10)
                plt.colorbar(im3, ax=ax3)
                self._add_v60_outline_fixed(ax3, 'xy')
                self._add_filter_paper_outline(ax3, 'xy')
                self._add_particles_to_plot(ax3, 'xy', config.NZ//2)
                
                # 速度 XY切面
                xy_slice_u = u_magnitude[:, :, config.NZ//2]
                im4 = ax4.imshow(xy_slice_u.T, origin='lower', aspect='equal', cmap=self.velocity_cmap, vmin=0.0, vmax=0.1)
                ax4.set_title('Velocity (XY plane)', fontsize=10)
                plt.colorbar(im4, ax=ax4)
                self._add_v60_outline_fixed(ax4, 'xy')
                self._add_filter_paper_outline(ax4, 'xy')
                self._add_particles_to_plot(ax4, 'xy', config.NZ//2)
            
            plt.suptitle(f'Combined Analysis (t={physical_time:.2f}s)', fontsize=14)
            filename = self.get_output_path(f'combined_analysis_step_{step_num:04d}.png')
            fig.suptitle(f'Combined Multi-Physics Analysis - Step {step_num}', fontsize=14)
            self._safe_savefig(fig, filename, dpi=200)
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
    
    def _add_filter_paper_outline(self, ax, plane='xz'):
        """添加濾紙虛線輪廓到圖表"""
        try:
            if plane == 'xz':
                # V60幾何參數
                center_x = config.NX // 2
                bottom_z = 5  # 濾紙底部位置
                top_z = bottom_z + config.CUP_HEIGHT / config.SCALE_LENGTH
                top_radius = config.TOP_RADIUS / config.SCALE_LENGTH
                bottom_radius = config.BOTTOM_RADIUS / config.SCALE_LENGTH
                
                # 濾紙與V60內壁有2mm空隙
                filter_gap = 0.002 / config.SCALE_LENGTH  # 2mm空隙轉換為格子單位
                
                # 計算濾紙輪廓（比V60內壁小一點）
                filter_top_radius = top_radius - filter_gap
                filter_bottom_radius = bottom_radius + filter_gap  # 底部濾紙略大於出水孔
                
                # 繪製濾紙錐形輪廓（虛線）
                x_left_top = center_x - filter_top_radius
                x_right_top = center_x + filter_top_radius
                x_left_bottom = center_x - filter_bottom_radius
                x_right_bottom = center_x + filter_bottom_radius
                
                # 濾紙側壁（虛線）
                ax.plot([x_left_top, x_left_bottom], [top_z, bottom_z], 
                       'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Filter Paper')
                ax.plot([x_right_top, x_right_bottom], [top_z, bottom_z], 
                       'gray', linestyle='--', linewidth=1.5, alpha=0.7)
                
                # 濾紙底部（虛線圓弧）
                filter_bottom_y = bottom_z + 1  # 濾紙底部稍微高一點
                ax.plot([x_left_bottom, x_right_bottom], [filter_bottom_y, filter_bottom_y], 
                       'gray', linestyle='--', linewidth=1.5, alpha=0.7)
                
            elif plane == 'xy':
                # XY平面的濾紙圓形輪廓
                center_x = config.NX // 2
                center_y = config.NY // 2
                
                # 濾紙與V60內壁有2mm空隙
                filter_gap = 0.002 / config.SCALE_LENGTH
                top_radius = config.TOP_RADIUS / config.SCALE_LENGTH
                filter_radius = top_radius - filter_gap
                
                # 繪製濾紙圓形輪廓（虛線）
                circle_filter = Circle((center_x, center_y), filter_radius, 
                                     fill=False, color='gray', linestyle='--', 
                                     linewidth=1.5, alpha=0.7, label='Filter Paper')
                ax.add_patch(circle_filter)
                
        except Exception as e:
            # 如果濾紙輪廓繪製失敗，靜默忽略
            pass
            
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
                circle_top = Circle((center_x, center_y), top_radius, 
                                      fill=False, color='black', linewidth=2, alpha=0.8, label='V60 Top')
                ax.add_patch(circle_top)
                
                # 繪製出水孔
                hole_radius = bottom_radius / 2
                circle_hole = Circle((center_x, center_y), hole_radius, 
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

    def save_lbm_monitoring_chart(self, simulation_time, step_num):
        """
        生成LBM診斷監控圖表
        """
        try:
            if not hasattr(self, 'simulation') or self.simulation is None or not hasattr(self.simulation, 'lbm_diagnostics'):
                return None
                
            diagnostics = getattr(self.simulation, 'lbm_diagnostics', None)
            if not diagnostics:
                return None
                
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # 1. 穩定性監控
            ax1 = fig.add_subplot(gs[0, 0])
            if len(diagnostics.history['time_stability']) > 0:
                ax1.plot(diagnostics.history['time_stability'], label='Time Stability', color='blue')
                ax1.set_title('LBM Stability Metrics')
                ax1.set_ylabel('Stability')
                ax1.set_xlabel('Step')
                ax1.grid(True)
                ax1.legend()
            else:
                ax1.text(0.5, 0.5, 'No stability data', ha='center', va='center')
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
            
            # 2. Mach數監控
            ax2 = fig.add_subplot(gs[0, 1])
            if len(diagnostics.history['max_mach']) > 0:
                ax2.plot(diagnostics.history['max_mach'], label='Max Mach', color='red')
                ax2.axhline(y=0.1, color='orange', linestyle='--', label='Warning (Ma=0.1)')
                ax2.axhline(y=0.3, color='red', linestyle='--', label='Critical (Ma=0.3)')
                ax2.set_title('Mach Number Monitoring')
                ax2.set_ylabel('Mach Number')
                ax2.set_xlabel('Step')
                ax2.grid(True)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No Mach data', ha='center', va='center')
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
            
            # 3. 守恆定律
            ax3 = fig.add_subplot(gs[1, 0])
            if len(diagnostics.history['mass_conservation']) > 0:
                ax3.plot(diagnostics.history['mass_conservation'], label='Mass Conservation', color='green')
                if len(diagnostics.history['momentum_conservation']) > 0:
                    ax3.plot(diagnostics.history['momentum_conservation'], label='Momentum Conservation', color='purple')
                ax3.set_title('Conservation Laws')
                ax3.set_ylabel('Conservation Error')
                ax3.set_xlabel('Step')
                ax3.grid(True)
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No conservation data', ha='center', va='center')
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
            
            # 4. V60物理參數
            ax4 = fig.add_subplot(gs[1, 1])
            if len(diagnostics.history['v60_flow_rate']) > 0:
                ax4.plot(diagnostics.history['v60_flow_rate'], label='V60 Flow Rate', color='brown')
                if len(diagnostics.history['extraction_efficiency']) > 0:
                    ax4_twin = ax4.twinx()
                    ax4_twin.plot(diagnostics.history['extraction_efficiency'], label='Extraction Efficiency', color='orange')
                    ax4_twin.set_ylabel('Efficiency (%)', color='orange')
                ax4.set_title('V60 Performance Metrics')
                ax4.set_ylabel('Flow Rate', color='brown')
                ax4.set_xlabel('Step')
                ax4.grid(True)
                ax4.legend(loc='upper left')
            else:
                ax4.text(0.5, 0.5, 'No V60 data', ha='center', va='center')
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
            
            # 5. 趨勢分析（合併兩個子圖）
            ax5 = fig.add_subplot(gs[2, :])
            if len(diagnostics.history['time_stability']) > 5:
                steps = range(len(diagnostics.history['time_stability']))
                
                # 左Y軸：穩定性相關
                ax5.plot(steps, diagnostics.history['time_stability'], 'b-', label='Stability', alpha=0.7)
                if len(diagnostics.history['max_mach']) > 0:
                    ax5.plot(steps, diagnostics.history['max_mach'], 'r-', label='Max Mach', alpha=0.7)
                ax5.set_xlabel('Simulation Step')
                ax5.set_ylabel('Stability & Mach', color='blue')
                ax5.tick_params(axis='y', labelcolor='blue')
                ax5.grid(True, alpha=0.3)
                
                # 右Y軸：物理參數
                if len(diagnostics.history['v60_flow_rate']) > 0:
                    ax5_twin = ax5.twinx()
                    ax5_twin.plot(steps, diagnostics.history['v60_flow_rate'], 'g-', label='Flow Rate', alpha=0.7)
                    ax5_twin.set_ylabel('Flow Rate & Physics', color='green')
                    ax5_twin.tick_params(axis='y', labelcolor='green')
                    
                    # 組合圖例
                    lines1, labels1 = ax5.get_legend_handles_labels()
                    lines2, labels2 = ax5_twin.get_legend_handles_labels()
                    lines = lines1 + lines2
                    labels = labels1 + labels2
                    ax5.legend(lines, labels, loc='upper left', fontsize=8)
                else:
                    ax5.legend(loc='upper left', fontsize=8)
                    
                ax5.set_title('LBM Diagnostics Trend Analysis')
            else:
                ax5.text(0.5, 0.5, 'Insufficient data for trend analysis', ha='center', va='center')
                ax5.set_xlim(0, 1)
                ax5.set_ylim(0, 1)
            
            plt.suptitle(f'LBM Comprehensive Diagnostics (t={step_num * config.SCALE_TIME:.2f}s)', fontsize=14)
            
            filename = self.get_output_path(f'lbm_monitoring_step_{step_num:04d}.png')
            fig.suptitle(f'LBM System Monitoring - Step {step_num}', fontsize=14)
            self._safe_savefig(fig, filename, dpi=200)
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"❌ LBM監控圖表生成失敗: {str(e)}")
            return None
