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
from porous_media_3d import PorousMedia3D
from coffee_particles import CoffeeParticleSystem
from precise_pouring import PrecisePouringSystem
from visualizer import UnifiedVisualizer

# 初始化Taichi
ti.init(arch=ti.gpu, device_memory_GB=4.0)

class CoffeeSimulation:
    def __init__(self, interactive=False):
        """
        初始化3D咖啡模擬
        interactive: 是否開啟互動模式
        """
        self.interactive = interactive
        self.step_count = 0
        
        print("=== 初始化3D咖啡萃取模擬 ===")
        
        # 初始化核心模組 (僅3D模式)
        self.lbm = LBMSolver()
        self.particles = CoffeeParticleSystem()
        self.multiphase = MultiphaseFlow3D(self.lbm)
        self.porous = PorousMedia3D(self.lbm, self.particles)
        self.pouring = PrecisePouringSystem()
        
        # 視覺化系統
        self.visualizer = UnifiedVisualizer(
            self.lbm, 
            self.multiphase, 
            None,  # 不使用geometry模組
            self.particles
        )
        
        # 初始化場
        self._initialize_simulation()
        
        print(f"模擬初始化完成 - 網格大小: {config.NX}×{config.NY}×{config.NZ}")
    
    def _initialize_simulation(self):
        """初始化3D模擬場"""
        print("初始化場變數...")
        
        # 初始化LBM場
        self.lbm.init_fields()
        
        # 初始化3D模組
        if self.multiphase:
            self.multiphase.init_phase_field()
        
        if self.porous:
            self.porous.init_porous_properties()
        
        if self.particles:
            # 初始化咖啡粒子床
            bed_height = config.COFFEE_BED_HEIGHT_LU * config.SCALE_LENGTH
            bed_radius = config.BOTTOM_RADIUS
            center_x = config.NX // 2
            center_y = config.NY // 2
            bottom_z = 5
            self.particles.initialize_coffee_bed(bed_height, bed_radius, center_x, center_y, bottom_z)
        
        print("場變數初始化完成")
    
    def step(self):
        """執行一個3D模擬步驟"""
        # 3D完整模擬步驟
        if self.pouring:
            self.pouring.apply_pouring(self.step_count)
        
        # LBM求解
        self.lbm.step()
        
        # 多相流處理
        if self.multiphase:
            self.multiphase.update_phase_field(self.lbm.rho, self.lbm.u)
        
        # 多孔介質處理
        if self.porous:
            self.porous.apply_porous_effects()  # 修正方法名稱
        
        # 粒子追蹤
        if self.particles:
            self.particles.update_particles(self.lbm.u, self.lbm.rho)
        
        self.step_count += 1
    
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
                
                # 保存輸出
                if save_output and step % (config.OUTPUT_FREQ * 5) == 0:
                    self.save_snapshot(step)
        
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
    
    def show_visualization(self, field_type='density', slice_direction='xy'):
        """顯示視覺化"""
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
            # 顯示最終統計
            sim.get_final_statistics()
            
            # 顯示視覺化 (可選)
            response = input("\n顯示視覺化? (y/N): ")
            if response.lower() == 'y':
                sim.show_visualization('composite')
        
    except Exception as e:
        print(f"模擬失敗: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())