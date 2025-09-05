# lbm_diagnostics.py
"""
LBM診斷監控系統 - 專為CFD科研設計
提供時間穩定性、守恆定律、數值品質等專業診斷功能
"""

import numpy as np
import time
from collections import deque, defaultdict
from datetime import datetime
import config.config

class CircularBuffer:
    """循環緩衝區 - 高效歷史數據管理"""
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
    
    def add(self, timestamp, data_dict):
        self.timestamps.append(timestamp)
        self.data.append(data_dict.copy())
    
    def get_recent(self, n=10):
        """獲取最近n個數據點"""
        recent_data = list(self.data)[-n:] if len(self.data) >= n else list(self.data)
        recent_times = list(self.timestamps)[-n:] if len(self.timestamps) >= n else list(self.timestamps)
        return recent_times, recent_data
    
    def get_all(self):
        return list(self.timestamps), list(self.data)

class LBMDiagnostics:
    """LBM專用診斷監控系統"""
    
    def __init__(self, lbm_solver, multiphase=None, particles=None, pouring=None, filter_system=None):
        self.lbm = lbm_solver
        self.multiphase = multiphase
        self.particles = particles
        self.pouring = pouring
        self.filter = filter_system
        
        # 歷史數據存儲
        self.history = CircularBuffer(max_size=1000)
        self.detailed_history = CircularBuffer(max_size=100)  # 詳細分析的歷史
        
        # 監控頻率控制
        self.light_monitoring_freq = 5      # 輕量監控：每5步
        self.medium_monitoring_freq = 10    # 中等監控：每10步  
        self.heavy_monitoring_freq = 100    # 重計算：每100步
        
        # 前一步數據緩存（用於計算變化率）
        self.prev_rho = None
        self.prev_u = None
        self.prev_phi = None
        self.prev_momentum = np.zeros(3)
        
        # 初始化參考值
        self.initial_mass = 0.0
        self.accumulated_inflow = 0.0
        self.reference_density = 1.0  # LBM標準參考密度
        
        # 統計計數器
        self.calculation_times = {
            'light': [],
            'medium': [], 
            'heavy': []
        }
        
        print("🔬 LBM診斷監控系統已初始化")
        print(f"   └─ 監控頻率: 輕量({self.light_monitoring_freq}步) 中等({self.medium_monitoring_freq}步) 重計算({self.heavy_monitoring_freq}步)")
    
    def adaptive_monitoring_frequency(self, step_num):
        """適應性監控頻率調整"""
        if step_num < 100:
            self.light_monitoring_freq = 2
            self.medium_monitoring_freq = 5
        elif step_num < 500:
            self.light_monitoring_freq = 5
            self.medium_monitoring_freq = 10
        else:
            self.light_monitoring_freq = 10
            self.medium_monitoring_freq = 20
    
    def update_diagnostics(self, step_num, simulation_time, force_update=False):
        """主要診斷更新函數 - 根據頻率控制計算"""
        
        # 適應性調整頻率
        self.adaptive_monitoring_frequency(step_num)
        
        diagnostics = {
            'step': step_num,
            'time': simulation_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # 輕量監控（每步或按頻率）
        if step_num % self.light_monitoring_freq == 0 or force_update:
            light_start = time.time()
            
            stability = self.calculate_temporal_stability()
            lbm_quality = self.analyze_lbm_numerical_quality()
            
            diagnostics.update({
                'temporal_stability': stability,
                'lbm_quality': lbm_quality
            })
            
            self.calculation_times['light'].append(time.time() - light_start)
        
        # 中等監控
        if step_num % self.medium_monitoring_freq == 0 or force_update:
            medium_start = time.time()
            
            conservation = self.check_conservation_laws()
            phase_analysis = self.analyze_multiphase_quality()
            
            diagnostics.update({
                'conservation': conservation,
                'multiphase': phase_analysis
            })
            
            self.calculation_times['medium'].append(time.time() - medium_start)
        
        # 重計算監控
        if step_num % self.heavy_monitoring_freq == 0 or force_update:
            heavy_start = time.time()
            
            v60_physics = self.track_v60_physics()
            flow_analysis = self.detailed_flow_analysis()
            
            diagnostics.update({
                'v60_physics': v60_physics,
                'flow_analysis': flow_analysis,
                'diagnostics_performance': self.get_performance_stats()
            })
            
            # 存入詳細歷史
            self.detailed_history.add(simulation_time, diagnostics)
            self.calculation_times['heavy'].append(time.time() - heavy_start)
        
        # 存入常規歷史
        if diagnostics:  # 確保有數據才存儲
            self.history.add(simulation_time, diagnostics)
        
        # 更新前一步數據（為下次計算做準備）
        self._update_previous_step_data()
        
        return diagnostics
    
    def calculate_temporal_stability(self):
        """計算時間穩定性 - 輕量計算 O(N)"""
        try:
            if not hasattr(self.lbm, 'rho') or not hasattr(self.lbm, 'u'):
                return {'status': 'no_data'}
            
            current_rho = self.lbm.rho.to_numpy()
            current_u = self.lbm.u.to_numpy()
            
            stability = {}
            
            if self.prev_rho is not None and self.prev_u is not None:
                # 密度變化率
                rho_change = np.linalg.norm(current_rho - self.prev_rho)
                stability['density_change_rate'] = rho_change / config.DT
                stability['relative_density_change'] = rho_change / np.linalg.norm(current_rho)
                
                # 速度變化率
                u_change = np.linalg.norm(current_u - self.prev_u)
                stability['velocity_change_rate'] = u_change / config.DT
                stability['relative_velocity_change'] = u_change / (np.linalg.norm(current_u) + 1e-10)
                
                # 穩定性評級
                if stability['relative_density_change'] < 1e-6 and stability['relative_velocity_change'] < 1e-4:
                    stability['stability_grade'] = 'Excellent'
                elif stability['relative_density_change'] < 1e-4 and stability['relative_velocity_change'] < 1e-2:
                    stability['stability_grade'] = 'Good'
                else:
                    stability['stability_grade'] = 'Monitoring'
            else:
                stability['status'] = 'first_step'
            
            return stability
            
        except Exception as e:
            return {'error': f'temporal_stability_error: {str(e)[:50]}'}
    
    def analyze_lbm_numerical_quality(self):
        """分析LBM數值品質 - 輕量計算 O(N)"""
        try:
            if not hasattr(self.lbm, 'u') or not hasattr(self.lbm, 'rho'):
                return {'status': 'no_data'}
            
            u_data = self.lbm.u.to_numpy()
            rho_data = self.lbm.rho.to_numpy()
            
            quality = {}
            
            # Mach數檢查（LBM關鍵限制）
            u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
            cs_lattice = 1.0 / np.sqrt(3.0)  # LBM聲速
            mach_numbers = u_magnitude / cs_lattice
            
            quality['max_mach'] = float(np.max(mach_numbers))
            active_mach = mach_numbers[mach_numbers > 1e-10]
            quality['mean_mach'] = float(np.mean(active_mach)) if len(active_mach) > 0 else 0.0
            quality['mach_violation_ratio'] = float(np.sum(mach_numbers > 0.1) / mach_numbers.size)
            
            # 密度變化檢查（LBM假設小密度變化）
            density_deviation = np.abs(rho_data - self.reference_density)
            quality['max_density_deviation'] = float(np.max(density_deviation))
            quality['mean_density'] = float(np.mean(rho_data))
            quality['density_variation_coeff'] = float(np.std(rho_data) / np.mean(rho_data))
            
            # LBM品質評級
            if quality['max_mach'] < 0.1 and quality['max_density_deviation'] < 0.1:
                quality['lbm_grade'] = 'Excellent'
            elif quality['max_mach'] < 0.2 and quality['max_density_deviation'] < 0.2:
                quality['lbm_grade'] = 'Good'
            else:
                quality['lbm_grade'] = 'Caution'
            
            return quality
            
        except Exception as e:
            return {'error': f'lbm_quality_error: {str(e)[:50]}'}
    
    def check_conservation_laws(self):
        """檢查守恆定律 - 中等開銷 O(N)"""
        try:
            if not hasattr(self.lbm, 'rho') or not hasattr(self.lbm, 'u'):
                return {'status': 'no_data'}
            
            rho_data = self.lbm.rho.to_numpy()
            u_data = self.lbm.u.to_numpy()
            
            conservation = {}
            
            # 質量守恆檢查（開放系統版本）
            current_mass = np.sum(rho_data)
            
            if self.initial_mass == 0.0:  # 第一次計算時設定初始質量
                self.initial_mass = current_mass
            
            # 計算淨流入（注水 - 流出）
            net_inflow = 0.0
            if self.pouring and hasattr(self.pouring, 'get_current_flow_rate'):
                try:
                    inlet_flow = self.pouring.get_current_flow_rate()
                    net_inflow = inlet_flow * config.DT
                    self.accumulated_inflow += net_inflow
                except:
                    pass
            
            # 質量守恆誤差
            expected_mass = self.initial_mass + self.accumulated_inflow
            mass_error = abs(current_mass - expected_mass)
            
            conservation['total_mass'] = float(current_mass)
            conservation['initial_mass'] = float(self.initial_mass)
            conservation['accumulated_inflow'] = float(self.accumulated_inflow)
            conservation['mass_conservation_error'] = float(mass_error)
            conservation['relative_mass_error'] = float(mass_error / self.initial_mass)
            
            # 動量守恆檢查
            total_momentum = np.sum(rho_data[:,:,:,np.newaxis] * u_data, axis=(0,1,2))
            momentum_change = total_momentum - self.prev_momentum
            
            # 重力應該產生的動量變化
            gravity_momentum_change = config.GRAVITY_LU * np.sum(rho_data) * config.DT
            momentum_error = abs(momentum_change[2] - gravity_momentum_change) if len(momentum_change) > 2 else 0
            
            conservation['total_momentum'] = total_momentum.tolist()
            conservation['momentum_z_change'] = float(momentum_change[2]) if len(momentum_change) > 2 else 0
            conservation['expected_gravity_change'] = float(gravity_momentum_change)
            conservation['momentum_conservation_error'] = float(momentum_error)
            
            # 守恆品質評級
            if conservation['relative_mass_error'] < 1e-6:
                conservation['conservation_grade'] = 'Excellent'
            elif conservation['relative_mass_error'] < 1e-4:
                conservation['conservation_grade'] = 'Good'  
            else:
                conservation['conservation_grade'] = 'Moderate'
            
            self.prev_momentum = total_momentum.copy()
            
            return conservation
            
        except Exception as e:
            return {'error': f'conservation_error: {str(e)[:50]}'}
    
    def analyze_multiphase_quality(self):
        """分析多相流品質 - 中等開銷"""
        try:
            if not self.multiphase or not hasattr(self.multiphase, 'phi'):
                return {'status': 'no_multiphase'}
            
            phi_data = self.multiphase.phi.to_numpy()
            
            analysis = {}
            
            # 相場界面厚度分析
            interface_cells = np.logical_and(phi_data > 0.1, phi_data < 0.9)
            analysis['interface_thickness'] = int(np.sum(interface_cells))
            analysis['water_fraction'] = float(np.mean(phi_data))
            
            # 相場變化率
            if self.prev_phi is not None:
                phi_change = np.linalg.norm(phi_data - self.prev_phi)
                analysis['phase_change_rate'] = float(phi_change / config.DT)
            else:
                analysis['phase_change_rate'] = 0.0
            
            # 相場穩定性評級
            if analysis['phase_change_rate'] < 1e-6:
                analysis['phase_stability_grade'] = 'Excellent'
            elif analysis['phase_change_rate'] < 1e-4:
                analysis['phase_stability_grade'] = 'Good'
            else:
                analysis['phase_stability_grade'] = 'Dynamic'
            
            return analysis
            
        except Exception as e:
            return {'error': f'multiphase_error: {str(e)[:50]}'}
    
    def track_v60_physics(self):
        """追蹤V60物理過程 - 重計算"""
        try:
            physics = {}
            
            if hasattr(self.lbm, 'rho'):
                rho_data = self.lbm.rho.to_numpy()
                
                # 萃取前鋒面追蹤
                z_profile = np.mean(rho_data, axis=(0,1))  # XY平面平均
                wet_threshold = self.reference_density + 0.1
                
                wet_regions = np.where(z_profile > wet_threshold)[0]
                if len(wet_regions) > 0:
                    wet_front_z = np.max(wet_regions)
                    physics['wetting_front_position'] = float(wet_front_z * config.SCALE_LENGTH)
                else:
                    physics['wetting_front_position'] = 0.0
                
                # 系統內水量估算
                total_water_volume = np.sum(rho_data - self.reference_density) * (config.SCALE_LENGTH ** 3)
                physics['system_water_volume'] = float(max(0, total_water_volume))
            
            # 流量平衡分析
            inlet_flow = 0.0
            if self.pouring and hasattr(self.pouring, 'get_current_flow_rate'):
                try:
                    inlet_flow = self.pouring.get_current_flow_rate()
                except:
                    pass
            
            physics['inlet_flow_rate'] = float(inlet_flow)
            physics['outlet_flow_rate'] = self._estimate_outlet_flow()
            
            # 咖啡床動力學
            if self.particles:
                particle_dynamics = self._analyze_coffee_bed_dynamics()
                physics['particle_dynamics'] = particle_dynamics
            
            return physics
            
        except Exception as e:
            return {'error': f'v60_physics_error: {str(e)[:50]}'}
    
    def detailed_flow_analysis(self):
        """詳細流場分析 - 重計算"""
        try:
            if not hasattr(self.lbm, 'u'):
                return {'status': 'no_velocity_data'}
            
            u_data = self.lbm.u.to_numpy()
            analysis = {}
            
            # 速度場統計
            u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
            analysis['max_velocity'] = float(np.max(u_magnitude))
            active_velocities = u_magnitude[u_magnitude > 1e-10]
            analysis['mean_velocity'] = float(np.mean(active_velocities)) if len(active_velocities) > 0 else 0.0
            analysis['velocity_std'] = float(np.std(u_magnitude))
            
            # 流場複雜度（渦度簡化估算）
            # 簡化的渦度估算（避免完整梯度計算的高開銷）
            u_variation = np.std(u_data, axis=(0,1,2))
            analysis['flow_complexity'] = float(np.linalg.norm(u_variation))
            
            # 區域流量分析
            if hasattr(config, 'NX') and hasattr(config, 'NY') and hasattr(config, 'NZ'):
                center_x, center_y = config.NX // 2, config.NY // 2
                top_region = u_magnitude[:, :, int(config.NZ * 0.8):]
                middle_region = u_magnitude[:, :, int(config.NZ * 0.3):int(config.NZ * 0.8)]
                bottom_region = u_magnitude[:, :, :int(config.NZ * 0.3)]
                
                analysis['top_region_flow'] = float(np.mean(top_region))
                analysis['middle_region_flow'] = float(np.mean(middle_region))
                analysis['bottom_region_flow'] = float(np.mean(bottom_region))
            
            return analysis
            
        except Exception as e:
            return {'error': f'flow_analysis_error: {str(e)[:50]}'}
    
    def _analyze_coffee_bed_dynamics(self):
        """分析咖啡床動力學"""
        try:
            if not hasattr(self.particles, 'position') or not hasattr(self.particles, 'active'):
                return {'status': 'no_particle_data'}
            
            positions = self.particles.position.to_numpy()
            active = self.particles.active.to_numpy()
            
            active_particles = positions[active == 1]
            
            if len(active_particles) == 0:
                return {'status': 'no_active_particles'}
            
            dynamics = {}
            dynamics['active_particle_count'] = len(active_particles)
            
            # 床層高度分析
            z_coords = active_particles[:, 2]
            dynamics['bed_height_mean'] = float(np.mean(z_coords) * config.SCALE_LENGTH)
            dynamics['bed_height_std'] = float(np.std(z_coords) * config.SCALE_LENGTH)
            dynamics['bed_height_range'] = [float(np.min(z_coords) * config.SCALE_LENGTH),
                                          float(np.max(z_coords) * config.SCALE_LENGTH)]
            
            # 徑向分佈
            center_x, center_y = config.NX * 0.5, config.NY * 0.5
            radial_distances = np.sqrt((active_particles[:, 0] - center_x)**2 + 
                                     (active_particles[:, 1] - center_y)**2)
            dynamics['radial_distribution_mean'] = float(np.mean(radial_distances) * config.SCALE_LENGTH)
            dynamics['radial_distribution_std'] = float(np.std(radial_distances) * config.SCALE_LENGTH)
            
            return dynamics
            
        except Exception as e:
            return {'error': f'particle_dynamics_error: {str(e)[:50]}'}
    
    def _estimate_outlet_flow(self):
        """估算出口流量"""
        try:
            if not hasattr(self.lbm, 'u') or not hasattr(config, 'NZ'):
                return 0.0
            
            u_data = self.lbm.u.to_numpy()
            
            # 在底部區域估算向下的流量
            bottom_slice = u_data[:, :, 0:5, 2]  # Z方向速度的底部切片
            downward_flow = np.sum(np.maximum(-bottom_slice, 0))  # 只考慮向下的流動
            
            return float(downward_flow * config.SCALE_VELOCITY * config.SCALE_LENGTH**2)
            
        except Exception as e:
            return 0.0
    
    def _update_previous_step_data(self):
        """更新前一步數據"""
        try:
            if hasattr(self.lbm, 'rho'):
                self.prev_rho = self.lbm.rho.to_numpy().copy()
            if hasattr(self.lbm, 'u'):
                self.prev_u = self.lbm.u.to_numpy().copy()
            if self.multiphase and hasattr(self.multiphase, 'phi'):
                self.prev_phi = self.multiphase.phi.to_numpy().copy()
        except Exception as e:
            pass  # 靜默處理錯誤
    
    def get_performance_stats(self):
        """獲取診斷系統性能統計"""
        stats = {}
        
        for calc_type in ['light', 'medium', 'heavy']:
            times = self.calculation_times[calc_type]
            if times:
                stats[f'{calc_type}_calc_count'] = len(times)
                stats[f'{calc_type}_avg_time'] = np.mean(times)
                stats[f'{calc_type}_total_time'] = np.sum(times)
            else:
                stats[f'{calc_type}_calc_count'] = 0
                stats[f'{calc_type}_avg_time'] = 0.0
                stats[f'{calc_type}_total_time'] = 0.0
        
        return stats
    
    def get_current_diagnostics(self):
        """獲取當前診斷數據"""
        if self.history.data:
            return self.history.data[-1]
        else:
            return {}
    
    def get_diagnostics_history(self, detailed=False):
        """獲取診斷歷史數據"""
        if detailed:
            return self.detailed_history.get_all()
        else:
            return self.history.get_all()
    
    def get_summary_report(self):
        """獲取診斷摘要報告"""
        current = self.get_current_diagnostics()
        performance = self.get_performance_stats()
        
        report = {
            'monitoring_status': 'active',
            'total_calculations': sum([performance.get(f'{t}_calc_count', 0) for t in ['light', 'medium', 'heavy']]),
            'total_diagnostic_time': sum([performance.get(f'{t}_total_time', 0) for t in ['light', 'medium', 'heavy']]),
            'history_length': len(self.history.data),
            'current_quality_grades': {
                'stability': current.get('temporal_stability', {}).get('stability_grade', 'Unknown'),
                'lbm_quality': current.get('lbm_quality', {}).get('lbm_grade', 'Unknown'),
                'conservation': current.get('conservation', {}).get('conservation_grade', 'Unknown')
            }
        }
        
        return report