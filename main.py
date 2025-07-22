# main.py
"""
Pour-Over Coffee LBM Simulation
統一的主模擬程式 - 支援2D/3D LBM咖啡萃取模擬
"""

import taichi as ti
import numpy as np
import time
import sys
import os
import sys
import time
import signal
import datetime
import numpy as np

import taichi as ti

# 引入各模組 - 自動處理Taichi初始化
from init import initialize_taichi_once  # 統一初始化
import config
from lbm_solver import LBMSolver
from multiphase_3d import MultiphaseFlow3D
from coffee_particles import CoffeeParticleSystem
from precise_pouring import PrecisePouringSystem
from filter_paper import FilterPaperSystem
from visualizer import UnifiedVisualizer
from enhanced_visualizer import EnhancedVisualizer
from lbm_diagnostics import LBMDiagnostics

# 確保Taichi已正確初始化
initialize_taichi_once()

# Taichi已在init.py中初始化，不需要重複初始化

class SimulationDisplay:
    """統一的模擬輸出管理系統"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_update_time = 0
        self.terminal_width = self._get_terminal_width()
        
    def _get_terminal_width(self):
        """獲取終端寬度"""
        try:
            return os.get_terminal_size().columns
        except:
            return 80  # 預設寬度
    
    def show_header(self):
        """顯示程式標題"""
        width = self.terminal_width
        print("\n" + "="*width)
        print("☕ 手沖咖啡 3D 流體力學模擬系統 v2.0")
        print("🔬 工業級數值穩定性 | 🎯 V60 精確建模")
        print("="*width)
        
    def show_initialization_progress(self, stage, progress, description):
        """顯示初始化進度"""
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        print(f"\r📋 {stage}: [{bar}] {progress*100:.0f}% - {description}", end="")
        if progress >= 1.0:
            print()  # 完成後換行
    
    def show_simulation_progress(self, step, max_steps, stats, simulation_time):
        """顯示模擬進度 - 單行更新"""
        progress = step / max_steps
        bar_width = 20
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # 計算時間信息
        elapsed = time.time() - self.start_time
        if step > 0:
            eta = (elapsed / step) * (max_steps - step)
            eta_str = f"{eta/60:.1f}分" if eta > 60 else f"{eta:.0f}秒"
        else:
            eta_str = "計算中"
        
        # 格式化關鍵信息
        water_mass = stats.get('total_water_mass', 0)
        max_velocity = stats.get('max_velocity', 0)
        cfl = 0.010  # 從config獲取
        
        progress_line = (
            f"\r🌊 進度: [{bar}] {progress*100:.0f}% "
            f"({step:,}/{max_steps:,}) | "
            f"⏱️ {elapsed/60:.1f}分 剩餘{eta_str} | "
            f"💧 水量:{water_mass:.2f} | "
            f"🚀 速度:{max_velocity:.5f} | "
            f"🛡️ CFL:{cfl:.3f}"
        )
        
        print(progress_line[:self.terminal_width-1], end="")
        
        # 每30秒或重要步驟時顯示詳細狀態
        current_time = time.time()
        if current_time - self.last_update_time > 30 or step % 500 == 0:
            self._show_detailed_status(step, stats, simulation_time)
            self.last_update_time = current_time
    
    def _show_detailed_status(self, step, stats, simulation_time):
        """顯示詳細狀態面板"""
        print("\n" + "="*self.terminal_width)
        
        # 第一行：基本信息
        particle_count = stats.get('particle_count', 0)
        water_temp = "90°C"
        status = "進行中" if step < config.MAX_STEPS * 0.95 else "接近完成"
        
        print(f"☕ 咖啡顆粒: {particle_count:,}個 | 🌡️ 溫度: {water_temp} | ⚖️ 萃取: {status}")
        
        # 第二行：技術參數
        max_vel = stats.get('max_velocity', 0)
        avg_vel = stats.get('avg_velocity', 0)
        stability = "100%" if max_vel < 0.1 else "監控中"
        
        print(f"🔄 數值穩定性: {stability} | 💨 最大流速: {max_vel:.6f} | 平均: {avg_vel:.6f}")
        
        print("="*self.terminal_width)
        print()  # 空行分隔
    
    def show_completion(self, total_steps, total_time):
        """顯示完成信息"""
        print("\n")
        print("="*self.terminal_width)
        print("🎉 模擬完成！")
        print(f"📊 總步數: {total_steps:,}")
        print(f"⏱️  總時間: {total_time/60:.1f}分鐘")
        print(f"⚡ 平均速度: {total_steps/total_time:.1f} 步/秒")
        print("="*self.terminal_width)
    
    def show_interruption_message(self):
        """顯示中斷信息"""
        print("\n")
        print("⚠️  檢測到用戶中斷 (Ctrl+C)")
        print("🔄 正在安全停止模擬並生成結果圖...")
        
    def show_error_message(self, error, step):
        """顯示錯誤信息"""
        print(f"\n❌ 模擬在第 {step:,} 步發生錯誤")
        print(f"📝 錯誤詳情: {error}")
        print("🔄 正在生成診斷報告...")

class ResultsGenerator:
    """結果生成管理器"""
    
    def __init__(self, simulation):
        self.simulation = simulation
        self.output_dir = self._create_output_directory()
    
    def _create_output_directory(self):
        """創建輸出目錄"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/simulation_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def generate_all_results(self, step_num, reason="completion"):
        """生成所有結果文件 - 使用enhanced_visualizer系統"""
        print(f"\n📊 生成分析結果 ({reason})...")
        
        results = {}
        
        try:
            simulation_time = step_num * config.DT
            
            # 生成完整科研報告（包含多種視覺化）
            research_files = self.simulation.enhanced_viz.generate_research_report(simulation_time, step_num)
            if research_files:
                results['research_report'] = research_files
                print(f"   ✅ 科研報告: {len(research_files)} 文件")
            
            # 生成最終快照
            snapshot_files = self.simulation.save_snapshot(step_num)
            if snapshot_files:
                results['snapshots'] = snapshot_files
            
            # 導出完整數據
            data_files = self.simulation.enhanced_viz.export_data_for_analysis(simulation_time, step_num)
            if data_files:
                results['data_export'] = data_files
            
            # 保存統計數據
            self._save_statistics(step_num)
            
            # 顯示結果摘要
            self._show_results_summary(results, step_num, reason)
            
        except Exception as e:
            print(f"   ❌ 結果生成失敗: {e}")
            print("   └─ 嘗試基本結果生成...")
            # 備用方案
            try:
                basic_files = self.simulation.save_snapshot(step_num)
                if basic_files:
                    results['basic_snapshots'] = basic_files
            except Exception as backup_e:
                print(f"   ❌ 備用方案也失敗: {backup_e}")
                
        return results
    
    def _save_statistics(self, step_num):
        """保存統計數據為JSON"""
        import json
        
        try:
            stats = self.simulation.visualizer.get_statistics()
            stats['step_number'] = step_num
            stats['simulation_time'] = step_num * config.DT
            stats['timestamp'] = datetime.datetime.now().isoformat()
            
            stats_file = os.path.join(self.output_dir, f"statistics_step_{step_num:06d}.json")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            print(f"   警告: 統計數據保存失敗 - {e}")
    
    def _show_results_summary(self, results, step_num, reason="completion"):
        """顯示結果摘要 - 增強版"""
        print(f"\n✅ 科研級分析結果生成完成！")
        print(f"📁 輸出目錄: {self.output_dir}")
        print(f"🎯 生成原因: {reason}")
        print(f"📋 結果文件類型:")
        
        total_files = 0
        
        # 顯示各類型結果
        if 'research_report' in results and results['research_report']:
            print(f"   📊 科研報告: {len(results['research_report'])} 個文件")
            for file in results['research_report'][:3]:  # 只顯示前3個
                print(f"       └─ {file}")
            if len(results['research_report']) > 3:
                print(f"       └─ ... 及其他 {len(results['research_report'])-3} 個文件")
            total_files += len(results['research_report'])
        
        if 'snapshots' in results and results['snapshots']:
            print(f"   📸 快照圖片: {len(results['snapshots'])} 個文件")
            for file in results['snapshots']:
                print(f"       └─ {file}")
            total_files += len(results['snapshots'])
        
        if 'data_export' in results and results['data_export']:
            print(f"   💾 數據導出: {len(results['data_export'])} 個文件")
            for file in results['data_export']:
                print(f"       └─ {file}")
            total_files += len(results['data_export'])
        
        # 顯示統計數據文件
        print(f"   📈 統計數據: statistics_step_{step_num:06d}.json")
        total_files += 1
        
        print(f"\n📊 總計生成: {total_files} 個分析文件")
        print(f"🎉 所有文件均為高質量科研級輸出！")

class CoffeeSimulation:
    def __init__(self, interactive=False):
        """
        初始化3D咖啡模擬 - 使用可移動顆粒系統
        interactive: 是否開啟互動模式
        """
        print("🔄 CoffeeSimulation初始化開始...")
        
        self.interactive = interactive
        self.step_count = 0
        
        print("🔄 創建SimulationDisplay...")
        self.display = SimulationDisplay()
        self.results_generator = None  # 延遲初始化
        
        # 顯示標題
        print("🔄 顯示標題...")
        self.display.show_header()
        
        # 初始化核心模組
        print("🔧 系統初始化中...")
        
        print("🔄 初始化LBMSolver...")
        self.lbm = LBMSolver()
        
        print("🔄 初始化CoffeeParticleSystem...")
        self.particle_system = CoffeeParticleSystem(max_particles=15000)
        
        print("🔄 初始化MultiphaseFlow3D...")
        self.multiphase = MultiphaseFlow3D(self.lbm)
        
        print("🔄 初始化PrecisePouringSystem...")
        self.pouring = PrecisePouringSystem()
        
        print("🔄 初始化FilterPaperSystem...")
        self.filter_paper = FilterPaperSystem(self.lbm)
        
        # 視覺化系統
        self.visualizer = UnifiedVisualizer(
            self.lbm, 
            self.multiphase, 
            None,  # 不使用geometry模組
            self.particle_system
        )
        
        # LBM診斷監控系統
        print("🔧 建立LBM診斷系統...")
        self.diagnostics = LBMDiagnostics(
            self.lbm,
            self.multiphase,
            self.particle_system,
            self.pouring,
            self.filter_paper
        )
        
        # 增強版視覺化系統（用於高級分析）
        self.enhanced_viz = EnhancedVisualizer(
            self.lbm,
            self.multiphase,
            None,
            self.particle_system,  # 添加顆粒系統
            self.filter_paper,     # 添加濾紙系統
            self                   # 添加simulation引用以訪問診斷數據
        )
        
        # 初始化場
        created_particles = self._initialize_simulation()
        
        # 初始化結果生成器
        self.results_generator = ResultsGenerator(self)
        
        print(f"\n✅ 模擬系統就緒")
        print(f"   └─ {config.NX}×{config.NY}×{config.NZ} 網格，{created_particles:,} 咖啡顆粒")
    
    
    def _initialize_simulation(self):
        """穩定的分階段初始化 - CFD數值穩定性優化"""
        
        # === 階段1：純流體場初始化 ===
        self.lbm.init_fields()
        
        # 讓純流體場穩定幾步
        for i in range(10):
            self.lbm.step()
        
        # === 階段2：加入多相流 ===
        if self.multiphase:
            self.multiphase.init_phase_field()
            
            # 多相流穩定
            for i in range(20):
                self.lbm.step()
                self.multiphase.step()
        
        # === 階段3：濾紙系統初始化 ===
        self.filter_paper.initialize_filter_geometry()
        
        # === 階段4：顆粒系統初始化 ===
        created_particles = self.particle_system.initialize_coffee_bed_confined(self.filter_paper)
        
        # 顆粒-流體預穩定
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
        
        # === 階段5：注水系統啟動 ===
        if self.pouring:
            self.pouring.start_pouring(pattern='center')
        
        return created_particles
    
    def step(self):
        """執行一個3D模擬步驟 - CFD數值穩定化版本"""
        return self.step_stable()
    
    def step_stable(self):
        """CFD數值穩定化步進 - 欠鬆弛 + 時間步控制 + 診斷監控"""
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
        if self.pouring and self.step_count > 15:  # 縮短到15步後開始注水
            # 使用修正的時間步進行注水
            self.pouring.apply_pouring(self.lbm.u, self.lbm.rho, 
                                     self.multiphase.phi, dt_safe)
            
            # 延遲同步相場（避免劇烈變化）
            if self.step_count % 2 == 0:  # 每兩步同步一次
                self.multiphase.update_density_from_phase()
        
        # 添加調試信息
        if self.step_count == 16:  # 注水剛開始時
            print(f"\n🚿 注水系統啟動 (步驟 {self.step_count})")
            if hasattr(self.pouring, 'get_pouring_info'):
                info = self.pouring.get_pouring_info()
                print(f"   └─ 注水狀態: {info}")
        
        
        
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
        
        # === LBM診斷監控系統 ===
        simulation_time = self.step_count * config.DT
        try:
            # 關鍵步驟或定期診斷
            force_diagnostics = (
                self.step_count in [16, 50, 100, 250, 500] or  # 關鍵步驟
                self.step_count % 500 == 0 or                   # 定期強制診斷
                self.step_count < 100                           # 初期密集監控
            )
            
            diagnostic_result = self.diagnostics.update_diagnostics(
                self.step_count, simulation_time, force_update=force_diagnostics
            )
            
            # 重要診斷結果即時反饋
            if force_diagnostics and diagnostic_result:
                lbm_quality = diagnostic_result.get('lbm_quality', {})
                conservation = diagnostic_result.get('conservation', {})
                
                if lbm_quality.get('lbm_grade') in ['Caution'] or conservation.get('conservation_grade') in ['Moderate']:
                    print(f"   📊 步驟{self.step_count} 診斷: LBM品質={lbm_quality.get('lbm_grade', 'N/A')}, "
                          f"守恆品質={conservation.get('conservation_grade', 'N/A')}")
                
        except Exception as e:
            if self.step_count % 100 == 0:  # 避免錯誤訊息刷屏
                print(f"   ⚠️  步驟{self.step_count} 診斷計算異常: {str(e)[:50]}")
        
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
    

    
    def run(self, max_steps=None, show_progress=True, save_output=False, debug_mode=False):
        """運行模擬 - 新的用戶友善界面"""
        if max_steps is None:
            max_steps = config.MAX_STEPS
        
        print(f"\n🚀 開始模擬運行")
        print(f"📊 預計步數: {max_steps:,} 步")
        print(f"⏱️  預估時間: {max_steps/300:.1f} 分鐘")
        print(f"🛡️  數值穩定性: 工業級保證")
        print()
        
        start_time = time.time()
        last_save_step = -1
        
        try:
            for step in range(max_steps):
                # 執行模擬步驟
                success = self.step()
                if not success:
                    print(f"\n❌ 模擬在第 {step:,} 步失敗")
                    if hasattr(self, 'results_generator') and self.results_generator:
                        self.results_generator.generate_all_results(step, "數值不穩定")
                    return False
                
                # 更新進度顯示 - 強制輸出
                if show_progress and step % 5 == 0:  # 每5步輸出一次
                    stats = self._get_current_stats()
                    simulation_time = step * config.DT
                    print(f"📊 步驟 {step:,}/{max_steps:,} | 速度: {stats.get('max_velocity', 0):.6f} | 密度: {stats.get('avg_density', 1):.6f}")
                
                # 定期保存結果
                if save_output and step > 0 and step % (config.OUTPUT_FREQ * 5) == 0:
                    if step != last_save_step:  # 避免重複保存
                        self._save_intermediate_results(step)
                        last_save_step = step
                
                # 互動模式檢查 - 簡化
                if self.interactive and step % 100 == 0 and step > 0:
                    print(f"\n第 {step:,} 步完成。按Enter繼續或q退出:")
                    try:
                        response = input().strip()
                        if response.lower() == 'q':
                            print("用戶選擇退出")
                            break
                    except KeyboardInterrupt:
                        print("\n用戶中斷")
                        break
        
        except KeyboardInterrupt:
            # 優雅處理中斷
            self.display.show_interruption_message()
            self.results_generator.generate_all_results(self.step_count, "用戶中斷")
            return True
            
        except Exception as e:
            # 錯誤處理
            self.display.show_error_message(str(e), self.step_count)
            self.results_generator.generate_all_results(self.step_count, "系統錯誤")
            if debug_mode:
                import traceback
                traceback.print_exc()
            return False
        
        # 正常完成
        total_time = time.time() - start_time
        self.display.show_completion(self.step_count, total_time)
        
        # 生成最終結果
        print("\n📊 正在生成最終分析結果...")
        self.results_generator.generate_all_results(self.step_count, "正常完成")
        
        return True
    
    def _get_current_stats(self):
        """獲取當前統計數據"""
        try:
            stats = self.visualizer.get_statistics()
            # 添加顆粒統計
            particle_stats = self.particle_system.get_particle_statistics()
            stats['particle_count'] = particle_stats['count']
            
            # 添加注水狀態調試信息
            if hasattr(self, 'pouring') and self.pouring:
                pouring_info = self.pouring.get_pouring_info()
                stats['pouring_active'] = pouring_info.get('active', False)
                
                # 調試：如果注水活躍但水量為0，打印警告
                if (pouring_info.get('active', False) and 
                    stats.get('total_water_mass', 0) < 0.01 and 
                    self.step_count > 50):
                    if self.step_count % 100 == 0:  # 每100步打印一次避免刷屏
                        print(f"\n⚠️  調試：步驟{self.step_count} - 注水活躍但水量={stats['total_water_mass']:.4f}")
                        print(f"   └─ 注水信息: {pouring_info}")
            
            return stats
        except:
            return {
                'total_water_mass': 0.0,
                'max_velocity': 0.0,
                'avg_velocity': 0.0,
                'particle_count': 0,
                'pouring_active': False
            }
    
    def _save_intermediate_results(self, step):
        """保存中間結果 - 使用enhanced_visualizer"""
        try:
            # 基本科研級快照
            snapshot_files = self.save_snapshot(step)
            
            # 高級分析（每500步或重要節點）
            if step % 500 == 0 or step in [100, 250, 750]:
                simulation_time = step * config.DT
                
                # 生成完整科研報告
                research_files = self.enhanced_viz.generate_research_report(simulation_time, step)
                
                # 導出數據供外部分析
                if step % 1000 == 0:  # 每1000步導出一次數據
                    data_files = self.enhanced_viz.export_data_for_analysis(simulation_time, step)
                    
        except Exception as e:
            print(f"\n⚠️  中間結果保存失敗: {e}")
            print("   └─ 繼續模擬運行...")
    
    

    
    def save_snapshot(self, step):
        """保存快照 - 使用enhanced_visualizer生成高質量圖片"""
        simulation_time = step * config.DT
        
        # 使用科研級視覺化系統生成多種分析圖
        files = []
        
        try:
            # 1. 縱向分析（密度+速度場XZ切面）
            longitudinal_file = self.enhanced_viz.save_longitudinal_analysis(simulation_time, step)
            if longitudinal_file:
                files.append(longitudinal_file)
            
            # 2. 速度場分析（XY切面）
            velocity_file = self.enhanced_viz.save_velocity_analysis(simulation_time, step)
            if velocity_file:
                files.append(velocity_file)
            
            # 3. 組合分析（四合一視圖）
            combined_file = self.enhanced_viz.save_combined_analysis(simulation_time, step)
            if combined_file:
                files.append(combined_file)
            
        except Exception as e:
            print(f"❌ 快照保存過程中發生錯誤: {e}")
            # 備用方案：嘗試使用原始visualizer
            try:
                filename_base = f"coffee_sim_3d_{step:06d}"
                self._save_basic_snapshot(filename_base)
            except Exception as backup_e:
                print(f"   ❌ 備用方案也失敗: {backup_e}")
        
        return files
    
    def _save_basic_snapshot(self, filename_base):
        """備用快照保存方法"""
        try:
            # 使用numpy直接保存數據
            import numpy as np
            
            if hasattr(self.lbm, 'rho'):
                rho_data = self.lbm.rho.to_numpy()
                np.save(f"{filename_base}_density_data.npy", rho_data)
                print(f"   └─ 密度數據已保存: {filename_base}_density_data.npy")
            
            if hasattr(self.lbm, 'u'):
                u_data = self.lbm.u.to_numpy()
                u_mag = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
                np.save(f"{filename_base}_velocity_data.npy", u_mag)
                print(f"   └─ 速度數據已保存: {filename_base}_velocity_data.npy")
                
        except Exception as e:
            print(f"   ❌ 備用數據保存失敗: {e}")
    
    
    def save_advanced_analysis(self, step_num=None):
        """保存高级流动分析图 - 手動調用版本"""
        if step_num is None:
            step_num = self.step_count
            
        print("📊 正在生成高級分析圖...")
        return self.results_generator.generate_all_results(step_num, "手動生成")
    
    def show_visualization(self, field_type='longitudinal_animation', slice_direction='xy'):
        """顯示視覺化"""
        if field_type == 'longitudinal_animation':
            print("啟動縱向截面動畫...")
            self.enhanced_viz.display_longitudinal_animation()
        else:
            print(f"顯示 {field_type} 場...")
            self.visualizer.display_gui(field_type, slice_direction)
    
    def get_final_statistics(self):
        """獲取最終統計 - 包含LBM診斷摘要"""
        stats = self.visualizer.get_statistics()
        
        print("\n📊 最終統計數據")
        print("="*50)
        for key, value in stats.items():
            print(f"{key}: {value:.6f}")
        
        # 添加LBM診斷摘要
        if hasattr(self, 'diagnostics'):
            print("\n🔬 LBM診斷摘要")
            print("="*50)
            diagnostic_summary = self.diagnostics.get_summary_report()
            for key, value in diagnostic_summary.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  └─ {sub_key}: {sub_value}")
                else:
                    print(f"{key}: {value}")
        print("="*50)
        
        return stats

def run_debug_simulation(max_steps=250):
    """運行debug模式的模擬"""
    print("🔍 啟動DEBUG模式模擬")
    print("🎨 使用科研級enhanced_visualizer生成高質量圖片")
    
    print("🔄 正在創建模擬實例...")
    # 創建模擬實例
    sim = CoffeeSimulation()
    print("✅ 模擬實例創建成功")
    
    print("\n🔍 系統診斷:")
    if hasattr(sim, 'pouring') and sim.pouring:
        print("   └─ 注水系統: 正常")
        # 測試注水系統參數
        print(f"   └─ 注水直徑: {sim.pouring.POUR_DIAMETER_GRID:.2f} 格")
        print(f"   └─ 注水速度: {sim.pouring.POUR_VELOCITY:.6f} LU")
        print(f"   └─ 注水高度: {sim.pouring.POUR_HEIGHT:.1f}")
    if hasattr(sim, 'visualizer'):
        print("   └─ 基本視覺化系統: 正常")
    if hasattr(sim, 'enhanced_viz'):
        print("   ✅ 科研級視覺化系統: 正常 (用於圖片生成)")
        print("   └─ 支援: 密度場、速度場、組合分析、數據導出")
    
    print(f"\n🔍 初始統計:")
    initial_stats = sim._get_current_stats()
    for key, value in initial_stats.items():
        print(f"   └─ {key}: {value}")
    
    # 運行debug模式
    success = sim.run(max_steps=max_steps, debug_mode=True, show_progress=True)
    
    if success:
        print("\n🎉 Debug模擬成功完成")
        print("📊 所有圖片均為高質量科研級PNG格式")
        # 顯示最終統計
        final_stats = sim._get_current_stats()
        print(f"\n📊 最終統計對比:")
        for key in initial_stats:
            initial = initial_stats[key]
            final = final_stats.get(key, 0)
            change = final - initial if isinstance(initial, (int, float)) else "N/A"
            print(f"   └─ {key}: {initial:.4f} → {final:.4f} (變化: {change})")
    else:
        print("\n⚠️  Debug模擬異常結束")
        print("📊 已生成診斷用的科研級分析圖")
    
    return sim


def main():
    """主函數 - 新的用戶界面"""
    import sys
    
    print("🚀 進入main函數")
    
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        # Debug模式：python main.py debug [步數]
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 250
        print(f"🔍 Debug模式 - 最大步數: {max_steps:,}")
        print("🔄 準備運行debug模擬...")
        sim = run_debug_simulation(max_steps=max_steps)
        print("✅ Debug模擬完成")
    else:
        # 正常模式運行
        print("☕ 手沖咖啡3D模擬系統")
        print("💡 使用 'python main.py debug [步數]' 進入調試模式")
        print()
        
        # 詢問用戶偏好
        try:
            interactive = input("是否啟用互動模式? (y/N): ").lower() == 'y'
            save_output = input("是否保存中間結果? (Y/n): ").lower() != 'n'
        except KeyboardInterrupt:
            print("\n取消運行")
            return 0
        
        # 創建並運行模擬
        sim = CoffeeSimulation(interactive=interactive)
        success = sim.run(save_output=save_output, show_progress=True)
        
        if success:
            print("\n🎉 模擬成功完成！")
            print("📊 查看 results/ 目錄獲取結果文件")
        else:
            print("\n⚠️  模擬異常結束，請查看診斷報告")
    
    return 0

if __name__ == "__main__":
    exit(main())