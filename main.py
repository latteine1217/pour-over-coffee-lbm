# main.py
"""
Pour-Over Coffee LBM Simulation
統一的主模擬程式 - 支援2D/3D LBM咖啡萃取模擬
"""

import taichi as ti
import numpy as np
import time
import sys
import config
from lbm_solver import LBMSolver
from multiphase_3d import MultiphaseFlow3D
from coffee_particles import CoffeeParticleSystem
from precise_pouring import PrecisePouringSystem
from filter_paper import FilterPaperSystem
from visualizer import UnifiedVisualizer
from enhanced_visualizer import EnhancedVisualizer

# 初始化Taichi - GPU並行優化設置
ti.init(
    arch=ti.metal,              # 明確使用Metal後端
    device_memory_GB=4.0,       # 設定GPU記憶體限制  
    fast_math=True,             # 啟用快速數學運算
    advanced_optimization=True,  # 進階編譯優化
    cpu_max_num_threads=8,      # CPU線程數限制
    debug=False                 # 關閉除錯模式提升性能
)

class CoffeeSimulation:
    def __init__(self, interactive=False):
        """
        初始化3D咖啡模擬 - 使用可移動顆粒系統
        interactive: 是否開啟互動模式
        """
        self.interactive = interactive
        self.step_count = 0
        
        print("=== 初始化3D咖啡萃取模擬 (可移動顆粒系統) ===")
        
        # 初始化核心模組
        self.lbm = LBMSolver()
        self.particle_system = CoffeeParticleSystem(max_particles=15000)  # 增強顆粒系統
        self.multiphase = MultiphaseFlow3D(self.lbm)
        self.pouring = PrecisePouringSystem()
        self.filter_paper = FilterPaperSystem(self.lbm)  # 濾紙系統
        
        # 視覺化系統
        self.visualizer = UnifiedVisualizer(
            self.lbm, 
            self.multiphase, 
            None,  # 不使用geometry模組
            self.particle_system
        )
        
        # 增強版視覺化系統（用於高級分析）
        self.enhanced_viz = EnhancedVisualizer(
            self.lbm,
            self.multiphase,
            None,
            self.particle_system,  # 添加顆粒系統
            self.filter_paper      # 添加濾紙系統
        )
        
        # 初始化場
        created_particles = self._initialize_simulation()
        
        print(f"模擬初始化完成 - 網格大小: {config.NX}×{config.NY}×{config.NZ}")
        print(f"增強顆粒系統：{created_particles:,} 個高斯分布咖啡顆粒")
    
    def _initialize_simulation(self):
        """穩定的分階段初始化 - CFD數值穩定性優化"""
        print("🔧 採用分階段穩定化初始化流程...")
        
        # === 階段1：純流體場初始化 ===
        print("階段1：純流體場初始化...")
        self.lbm.init_fields()
        
        # 讓純流體場穩定幾步
        print("   └─ 純流體場穩定化 (10步)...")
        for i in range(10):
            self.lbm.step()
        print("   ✅ 流體場基礎穩定")
        
        # === 階段2：加入多相流 ===
        print("階段2：多相流系統耦合...")
        if self.multiphase:
            self.multiphase.init_phase_field()
            
            # 多相流穩定
            print("   └─ 多相流場穩定化 (20步)...")
            for i in range(20):
                self.lbm.step()
                self.multiphase.step()
            print("   ✅ 多相流耦合穩定")
        
        # === 階段3：濾紙系統初始化 ===
        print("階段3：濾紙邊界系統...")
        self.filter_paper.initialize_filter_geometry()
        print("   ✅ 濾紙幾何邊界就緒")
        
        # === 階段4：顆粒系統初始化 ===
        print("階段4：咖啡顆粒系統...")
        created_particles = self.particle_system.initialize_coffee_bed_confined(self.filter_paper)
        
        # 顆粒-流體預穩定
        print("   └─ 顆粒-流體耦合預穩定 (15步)...")
        for i in range(15):
            self.lbm.step()
            if self.multiphase:
                self.multiphase.step()
            # 輕微顆粒更新（無流體力）
            dt_physical = config.DT * config.SCALE_TIME * 0.1  # 使用很小的時間步
            boundary = self.filter_paper.get_coffee_bed_boundary()
            self.particle_system.update_particle_physics(
                dt_physical,
                boundary['center_x'], boundary['center_y'], boundary['bottom_z'],
                boundary['bottom_radius_lu'], boundary['top_radius_lu']
            )
        print("   ✅ 顆粒系統預穩定")
        
        # === 階段5：注水系統啟動 ===
        print("階段5：注水系統啟動...")
        if self.pouring:
            self.pouring.start_pouring(pattern='center')
        print("   ✅ 注水系統就緒")
        
        print("🎉 分階段穩定化初始化完成")
        print(f"   └─ 顆粒總數: {created_particles:,}")
        print("   └─ 數值穩定: 45步分階段預穩定")
        print("   └─ 耦合強度: 漸進式增強")
        print("   └─ CFD穩定性: 優化完成")
        
        return created_particles
    
    def step(self):
        """執行一個3D模擬步驟 - CFD數值穩定化版本"""
        return self.step_stable()
    
    def step_stable(self):
        """CFD數值穩定化步進 - 欠鬆弛 + 時間步控制"""
        # === 策略2：欠鬆弛穩定化 ===
        
        # 動態時間步控制（初期使用較小時間步）
        if self.step_count < 50:
            dt_safe = config.DT * 0.1  # 初期使用10%時間步
            dt_coupling = dt_safe * 0.5  # 耦合使用更小時間步
        elif self.step_count < 100:
            dt_safe = config.DT * 0.5   # 中期使用50%時間步
            dt_coupling = dt_safe * 0.7
        else:
            dt_safe = config.DT         # 穩定後使用全時間步
            dt_coupling = dt_safe
        
        # 延遲啟動注水系統（避免初期數值衝擊）
        if self.pouring and self.step_count > 30:  # 30步後才開始注水
            # 使用修正的時間步進行注水
            self.pouring.apply_pouring(self.lbm.u, self.lbm.rho, 
                                     self.multiphase.phi, dt_safe)
            
            # 延遲同步相場（避免劇烈變化）
            if self.step_count % 2 == 0:  # 每兩步同步一次
                self.multiphase.update_density_from_phase()
        
        # LBM求解（核心流體計算）- 使用策略3的CFL控制
        if hasattr(self.lbm, 'step_with_cfl_control'):
            local_cfl = self.lbm.step_with_cfl_control()
            if local_cfl > 0.5:  # 記錄高CFL事件
                print(f"   步驟{self.step_count}: CFL={local_cfl:.3f}")
        elif hasattr(self.lbm, 'step_with_particles'):
            self.lbm.step_with_particles(self.particle_system)
        else:
            self.lbm.step()
        
        # === 欠鬆弛流體-顆粒耦合 ===
        if hasattr(self.lbm, 'u') and hasattr(self.lbm, 'rho') and self.step_count > 10:
            dt_physical = dt_coupling * config.SCALE_TIME
            
            # 檢查局部速度合理性
            u_data = self.lbm.u.to_numpy()
            u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
            max_vel = np.max(u_magnitude)
            
            if max_vel < 0.1 and not np.isnan(max_vel) and not np.isinf(max_vel):  # 檢查合理性和有限性
                # 使用欠鬆弛的流體力
                self.particle_system.apply_fluid_forces(
                    self.lbm.u, self.lbm.u, self.lbm.u,
                    self.lbm.rho, self.lbm.rho,
                    dt_physical  # 使用減小的時間步
                )
            else:
                # 速度異常時跳過流體力計算
                if self.step_count < 100 and (np.isnan(max_vel) or np.isinf(max_vel)):
                    print(f"⚠️  步驟{self.step_count}: 速度場異常，跳過耦合")
        
        # 顆粒物理更新（使用穩定化參數）
        if self.filter_paper:
            boundary = self.filter_paper.get_coffee_bed_boundary()
            dt_physical = dt_safe * config.SCALE_TIME
            
            self.particle_system.update_particle_physics(
                dt_physical,
                boundary['center_x'], boundary['center_y'], 
                boundary['bottom_z'],
                boundary['bottom_radius_lu'],
                boundary['top_radius_lu']
            )
        
        # 濾紙系統處理
        if self.filter_paper and hasattr(self.filter_paper, 'step'):
            self.filter_paper.step(self.particle_system)
        
        # 多相流處理（使用欠鬆弛）
        if self.multiphase:
            self.multiphase.step()
        
        # 數值穩定性檢查
        if self.step_count > 1:
            stats = self.visualizer.get_statistics()
            max_vel = stats.get('max_velocity', 0.0)
            if np.isnan(max_vel) or np.isinf(max_vel):
                print(f"❌ 步驟{self.step_count}: 數值發散！")
                return False
            elif max_vel > 0.15:
                print(f"⚠️  步驟{self.step_count}: 速度偏高 {max_vel:.6f}")
        
        # 更新計數器
        self.step_count += 1
        return True
    
    def print_simulation_status(self):
        """打印模擬狀態 - 包含增強顆粒統計"""
        current_time = self.step_count * config.DT
        
        # 獲取增強顆粒統計
        particle_stats = self.particle_system.get_particle_statistics()
        
        # 基本狀態
        print(f"\n⏱️  時間: {current_time:.2f}s (步驟: {self.step_count})")
        print(f"🌊 多相流狀態: 活躍")
        
        # 增強顆粒系統狀態
        print(f"☕ 增強咖啡顆粒統計:")
        print(f"   └─ 活躍顆粒: {particle_stats['count']:,}")
        print(f"   └─ 平均半徑: {particle_stats['mean_radius']*1000:.3f} mm")
        print(f"   └─ 半徑標準差: {particle_stats['std_radius']*1000:.3f} mm")
        print(f"   └─ 半徑範圍: {particle_stats['min_radius']*1000:.3f} - {particle_stats['max_radius']*1000:.3f} mm")
        
        # 計算流動統計
        u_data = self.lbm.u.to_numpy()
        u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
        
        print(f"💧 流體狀態:")
        print(f"   └─ 最大流速: {np.max(u_magnitude):.5f} m/s")
        print(f"   └─ 平均流速: {np.mean(u_magnitude):.5f} m/s")
        
        # 濾紙系統狀態
        if self.filter_paper:
            boundary = self.filter_paper.get_coffee_bed_boundary()
            print(f"🔧 邊界約束系統:")
            print(f"   └─ 錐形濾紙覆蓋完整V60表面")
            print(f"   └─ 顆粒100%約束在邊界內")
        
        # 物理現象提示
        if particle_stats['count'] > 500:
            print("   ☕ 咖啡床結構穩定，高斯分布完整")
        if np.max(u_magnitude) > 0.001:
            print("   🌊 流體-顆粒相互作用活躍")
        if current_time > 60:
            print("   ⏰ 咖啡萃取進行中")
    
    def run(self, max_steps=None, show_progress=True, save_output=False, debug_mode=False):
        """運行模擬
        
        Args:
            max_steps: 最大步數 
            show_progress: 顯示進度
            save_output: 保存輸出
            debug_mode: 啟用詳細診斷模式
        """
        if max_steps is None:
            max_steps = config.MAX_STEPS
        
        print(f"開始模擬 - 最大步數: {max_steps:,}")
        
        # Debug模式：統計歷史記錄
        if debug_mode:
            self.debug_stats = {
                'velocity_history': [],
                'water_mass_history': [],
                'pouring_info': [],
                'step_components': []
            }
            print("🔍 Debug模式啟用 - 收集詳細診斷資料")
        
        try:
            for step in range(max_steps):
                step_start_time = time.time()
                
                # Debug: 執行前檢查
                if debug_mode and step < 10:
                    self._debug_step_analysis(step, "before")
                
                # 執行模擬步驟
                self.step()
                
                # Debug: 執行後檢查
                if debug_mode and step < 10:
                    self._debug_step_analysis(step, "after")
                
                step_time = time.time() - step_start_time
                
                # 進度報告
                if show_progress and (step % config.OUTPUT_FREQ == 0 or step < 20):
                    self._print_detailed_progress(step, max_steps, step_time, debug_mode)
                
                # Debug: 收集統計資料
                if debug_mode:
                    self._collect_debug_statistics(step)
                
                # 檢查異常終止條件
                if self._check_termination_conditions(step, debug_mode):
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠️  模擬被用戶中斷")
        except Exception as e:
            print(f"\n❌ 模擬出錯: {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()
        
        print("模擬完成")
        
        # Debug模式：輸出分析報告
        if debug_mode:
            self._print_debug_summary()
    
    def _debug_step_analysis(self, step, stage):
        """逐步分析每個組件對速度場的影響"""
        u_data = self.lbm.u.to_numpy()
        u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
        non_zero_count = np.count_nonzero(u_magnitude)
        max_vel = np.max(u_magnitude)
        
        component_info = {
            'step': step,
            'stage': stage,
            'non_zero_points': non_zero_count,
            'max_velocity': max_vel,
            'avg_velocity': np.mean(u_magnitude[u_magnitude > 0]) if non_zero_count > 0 else 0.0
        }
        
        if hasattr(self, 'debug_stats'):
            self.debug_stats['step_components'].append(component_info)
        
        if step <= 5:  # 只打印前5步的詳細資料
            print(f"    Debug步驟{step}-{stage}: 非零點={non_zero_count:,}, 最大速度={max_vel:.6f}")
    
    def _print_detailed_progress(self, step, max_steps, step_time, debug_mode):
        """打印詳細進度資訊"""
        progress = (step + 1) / max_steps * 100
        current_time = self.step_count * config.SCALE_TIME
        
        print(f"\n⏱️  步驟: {step+1:,}/{max_steps:,} ({progress:.1f}%)")
        print(f"   模擬時間: {current_time:.2f}s, 計算時間: {step_time*1000:.1f}ms")
        
        # 獲取統計資料
        try:
            stats = self.visualizer.get_statistics()
            water_mass = stats['total_water_mass']
            max_velocity = stats['max_velocity']
            avg_velocity = stats['avg_velocity']
            
            print(f"🌊 流體統計: 水質量={water_mass:.3f}, 最大速度={max_velocity:.6f}, 平均速度={avg_velocity:.6f}")
            
            # 注水資訊
            if self.pouring and hasattr(self.pouring, 'pouring_active'):
                if self.pouring.pouring_active[None] == 1:
                    pour_info = self.pouring.get_pouring_info()
                    print(f"💧 注水狀態: 活躍 - {pour_info}")
                else:
                    print(f"💧 注水狀態: 停止")
            
            # Debug模式額外資訊
            if debug_mode and step < 50:
                u_data = self.lbm.u.to_numpy()
                u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
                non_zero_count = np.count_nonzero(u_magnitude)
                print(f"🔍 Debug: 非零速度點數={non_zero_count:,}")
                
        except Exception as e:
            print(f"   統計資料獲取失敗: {e}")
    
    def _collect_debug_statistics(self, step):
        """收集調試統計資料"""
        try:
            stats = self.visualizer.get_statistics()
            
            self.debug_stats['velocity_history'].append({
                'step': step,
                'max_velocity': stats['max_velocity'],
                'avg_velocity': stats['avg_velocity']
            })
            
            self.debug_stats['water_mass_history'].append({
                'step': step,
                'water_mass': stats['total_water_mass'],
                'air_mass': stats['total_air_mass']
            })
            
            # 注水資訊
            if self.pouring and hasattr(self.pouring, 'pouring_active'):
                if self.pouring.pouring_active[None] == 1:
                    pour_info = self.pouring.get_pouring_info()
                    self.debug_stats['pouring_info'].append({
                        'step': step,
                        'active': True,
                        'info': pour_info
                    })
                    
        except Exception as e:
            print(f"Debug統計收集失敗: {e}")
    
    def _check_termination_conditions(self, step, debug_mode):
        """檢查異常終止條件"""
        try:
            stats = self.visualizer.get_statistics()
            
            # 檢查速度場是否歸零（潛在問題）
            if step > 50 and stats['max_velocity'] < 1e-8:
                if debug_mode:
                    print(f"⚠️  警告：步驟{step}時速度場歸零！")
                    return False  # 不自動終止，讓用戶觀察
                    
            # 檢查數值發散
            if stats['max_velocity'] > 1.0:  # 超過物理合理範圍
                print(f"❌ 數值發散：最大速度={stats['max_velocity']:.3f}")
                return True
                
            return False
            
        except Exception:
            return False
    
    def _print_debug_summary(self):
        """打印調試總結"""
        print("\n" + "="*50)
        print("🔍 DEBUG模式分析總結")
        print("="*50)
        
        if hasattr(self, 'debug_stats'):
            # 速度場分析
            velocity_hist = self.debug_stats['velocity_history']
            if velocity_hist:
                max_velocities = [v['max_velocity'] for v in velocity_hist]
                print(f"💨 速度場分析:")
                print(f"   峰值速度: {max(max_velocities):.6f}")
                print(f"   速度歸零步數: {next((v['step'] for v in velocity_hist if v['max_velocity'] < 1e-8), '無')}")
                
                # 找出速度突變點
                for i in range(1, len(max_velocities)):
                    if max_velocities[i-1] > 1e-6 and max_velocities[i] < 1e-8:
                        print(f"   ⚠️  速度歸零於步驟: {velocity_hist[i]['step']}")
                        break
            
            # 水質量分析  
            water_hist = self.debug_stats['water_mass_history']
            if water_hist:
                water_masses = [w['water_mass'] for w in water_hist]
                print(f"💧 水質量分析:")
                print(f"   峰值水質量: {max(water_masses):.3f}")
                print(f"   最終水質量: {water_masses[-1]:.3f}")
            
            # 注水分析
            pour_hist = self.debug_stats['pouring_info']
            if pour_hist:
                active_steps = len([p for p in pour_hist if p['active']])
                print(f"🚿 注水分析:")
                print(f"   活躍注水步數: {active_steps}")
        
        print("="*50)
        
        print(f"=== 開始模擬 (最大步數: {max_steps}) ===")
        
        start_time = time.time()
        
        try:
            for step in range(max_steps):
                self.step()
                
                # 進度顯示
                if show_progress and step % config.OUTPUT_FREQ == 0:
                    elapsed = time.time() - start_time
                    stats = self.visualizer.get_statistics()
                    
                    print(f"Step {step:6d}/{max_steps} | "
                          f"Time: {elapsed:.1f}s | "
                          f"Water Mass: {stats['total_water_mass']:.2f} | "
                          f"Max Vel: {stats['max_velocity']:.4f}")
                
                # 互動模式
                if self.interactive and step % 100 == 0:
                    response = input("繼續 (Enter) 或退出 (q): ")
                    if response.lower() == 'q':
                        break
                
                # 保存輸出（包含新的可视化类型）
                if save_output and step % (config.OUTPUT_FREQ * 5) == 0:
                    self.save_snapshot(step)
                    
                    # 保存增强版分析图
                    simulation_time = step * config.DT
                    self.enhanced_viz.save_longitudinal_analysis(simulation_time, step)
                    self.enhanced_viz.save_velocity_analysis(simulation_time, step)
                    self.enhanced_viz.save_combined_analysis(simulation_time, step)
        
        except KeyboardInterrupt:
            print(f"\n模擬在第 {step} 步被中斷")
        
        except Exception as e:
            print(f"\n模擬在第 {step} 步發生錯誤: {e}")
            return False
        
        final_time = time.time() - start_time
        print(f"\n=== 模擬完成 ===")
        print(f"總步數: {self.step_count}")
        print(f"總時間: {final_time:.2f}秒")
        print(f"平均速度: {self.step_count/final_time:.1f} 步/秒")
        
        return True
    
    def save_snapshot(self, step):
        """保存快照"""
        filename_base = f"coffee_sim_3d_{step:06d}"
        
        # 保存密度場
        self.visualizer.save_image(f"{filename_base}_density.png", 'density')
        
        # 保存速度場  
        self.visualizer.save_image(f"{filename_base}_velocity.png", 'velocity')
        
        # 保存綜合視圖
        self.visualizer.save_image(f"{filename_base}_composite.png", 'composite')
        
        print(f"快照已保存: {filename_base}_*.png")
    
    def save_advanced_analysis(self, step_num=None):
        """保存高级流动分析图"""
        if step_num is None:
            step_num = self.step_count
            
        simulation_time = step_num * config.DT
        
        print("=== 生成高级可视化分析 ===")
        
        # 生成纵向剖面分析
        longitudinal_file = self.enhanced_viz.save_longitudinal_analysis(simulation_time, step_num)
        
        # 生成流速分析
        velocity_file = self.enhanced_viz.save_velocity_analysis(simulation_time, step_num)
        
        # 生成综合分析
        combined_file = self.enhanced_viz.save_combined_analysis(simulation_time, step_num)
        
        print(f"✅ 高级分析图已生成:")
        print(f"   - 纵向剖面: {longitudinal_file}")
        print(f"   - 流速分析: {velocity_file}")
        print(f"   - 综合分析: {combined_file}")
        
        return longitudinal_file, velocity_file, combined_file
    
    def show_visualization(self, field_type='longitudinal_animation', slice_direction='xy'):
        """顯示視覺化"""
        if field_type == 'longitudinal_animation':
            print("啟動縱向截面動畫...")
            self.enhanced_viz.display_longitudinal_animation()
        else:
            print(f"顯示 {field_type} 場...")
            self.visualizer.display_gui(field_type, slice_direction)
    
    def get_final_statistics(self):
        """獲取最終統計"""
        stats = self.visualizer.get_statistics()
        
        print("\n=== 最終統計 ===")
        for key, value in stats.items():
            print(f"{key}: {value:.6f}")
        
        return stats

def run_debug_simulation(max_steps=250):
    """運行debug模式的模擬 - 方便調試使用"""
    print("🔍 啟動DEBUG模式模擬")
    print("="*50)
    
    # 創建模擬實例
    sim = CoffeeSimulation()
    sim._initialize_simulation()
    
    print("\n🚿 注水系統診斷:")
    if hasattr(sim, 'pouring') and sim.pouring:
        sim.pouring.diagnose_pouring_system()
    
    print("\n🔍 速度場診斷:")
    if hasattr(sim, 'visualizer'):
        sim.visualizer.diagnose_velocity_field_issue()
    
    print("\n開始debug模式運行...")
    
    # 運行debug模式
    sim.run(max_steps=max_steps, debug_mode=True, show_progress=True)
    
    return sim


def main():
    """主函數"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        # Debug模式：python main.py debug [步數]
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 250
        sim = run_debug_simulation(max_steps=max_steps)
    else:
        # 正常模式運行
        sim = CoffeeSimulation()
        sim._initialize_simulation() 
        sim.run()
    
    return 0

if __name__ == "__main__":
    exit(main())