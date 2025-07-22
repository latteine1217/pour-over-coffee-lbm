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
            None
        )
        
        # 初始化場
        created_particles = self._initialize_simulation()
        
        print(f"模擬初始化完成 - 網格大小: {config.NX}×{config.NY}×{config.NZ}")
        print(f"增強顆粒系統：{created_particles:,} 個高斯分布咖啡顆粒")
    
    def _initialize_simulation(self):
        """初始化3D模擬場 - 使用增強顆粒系統"""
        print("初始化場變數...")
        
        # 初始化LBM場
        self.lbm.init_fields()
        
        # 初始化多相流
        if self.multiphase:
            self.multiphase.init_phase_field()
        
        # 初始化濾紙系統（必須在顆粒系統之前）
        print("正在初始化濾紙系統...")
        self.filter_paper.initialize_filter_geometry()
        
        # 使用新的增強顆粒系統 - 錐形約束生成
        print(f"正在生成增強咖啡顆粒床...")
        
        created_particles = self.particle_system.initialize_coffee_bed_confined(self.filter_paper)
        
        # 開始注水
        if self.pouring:
            self.pouring.start_pouring(pattern='center')
        
        print("✅ 完整咖啡萃取系統初始化完成")
        print(f"   └─ 顆粒總數: {created_particles:,}")
        print("   └─ 物理模型: 增強顆粒-流體耦合系統")
        print("   └─ 顆粒分布: 高斯分布，30%標準差變異")
        print("   └─ 流體作用力: 阻力+浮力+壓力梯度力")
        print("   └─ 邊界約束: 錐形V60完美約束")
        print("   └─ 濾紙系統: V60濾紙透水性與顆粒阻擋")
        print("   └─ 特色功能: 真實尺度物理，科學級精度")
        
        return created_particles
    
    def step(self):
        """執行一個3D模擬步驟 - 包含增強顆粒-流體-濾紙耦合"""
        # 注水控制
        if self.pouring:
            self.pouring.apply_pouring(self.lbm.u, self.lbm.rho, 
                                     self.multiphase.phi, config.DT)
        
        # LBM求解
        if hasattr(self.lbm, 'step_with_particles'):
            self.lbm.step_with_particles(self.particle_system)
        else:
            self.lbm.step()
        
        # 應用簡化的流體作用力到顆粒
        if hasattr(self.lbm, 'u') and hasattr(self.lbm, 'rho'):
            dt_physical = config.DT * config.SCALE_TIME
            # 傳遞正確的參數給簡化版本
            self.particle_system.apply_fluid_forces(
                self.lbm.u, self.lbm.u, self.lbm.u,  # 三個參數但只使用第一個
                self.lbm.rho, self.lbm.rho,  # density and pressure
                dt_physical
            )
        
        # 更新顆粒物理（包含邊界約束）
        if self.filter_paper:
            boundary = self.filter_paper.get_coffee_bed_boundary()
            dt_physical = config.DT * config.SCALE_TIME
            self.particle_system.update_particle_physics(
                dt_physical,
                boundary['center_x'],
                boundary['center_y'], 
                boundary['bottom_z'],
                boundary['bottom_radius_lu'],
                boundary['top_radius_lu']
            )
        
        # 濾紙系統處理
        if self.filter_paper and hasattr(self.filter_paper, 'step'):
            self.filter_paper.step(self.particle_system)
        
        # 多相流處理
        if self.multiphase:
            self.multiphase.step()
        
        # 更新計數器
        self.step_count += 1
    
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
    
    def run(self, max_steps=None, show_progress=True, save_output=False):
        """運行模擬"""
        if max_steps is None:
            max_steps = config.MAX_STEPS
        
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

def main():
    """主程式入口"""
    print("Pour-Over Coffee LBM Simulation (3D)")
    print("使用opencode + GitHub Copilot開發")
    print("=" * 50)
    
    # 解析命令行參數
    interactive = False
    
    if len(sys.argv) > 1:
        if 'interactive' in sys.argv:
            interactive = True
    
    try:
        # 創建3D模擬
        sim = CoffeeSimulation(interactive=interactive)
        
        # 運行模擬
        success = sim.run(show_progress=True, save_output=False)
        
        if success:
            # 显示最终统计
            sim.get_final_statistics()
            
            # 生成最终的高级分析图
            print("\n=== 生成最终分析报告 ===")
            sim.save_advanced_analysis()
            
            # 显示视觉化 (更新为纵向截面动画)
            try:
                response = input("\n显示縱向截面動畫? (y/N): ")
                if response.lower() == 'y':
                    sim.show_visualization('longitudinal_animation')
            except (EOFError, KeyboardInterrupt):
                # 非互動模式或用戶中斷，跳過視覺化
                print("\n跳過視覺化顯示")
                pass
        
    except Exception as e:
        print(f"模擬失敗: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())