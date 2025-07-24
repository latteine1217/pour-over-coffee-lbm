# working_main.py - 能工作的簡化版本
"""
簡化但完整的CFD模擬主程式
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

import sys
import time

# 導入核心模組
from config.init import initialize_taichi_once
import config.config as config
from src.core.lbm_solver import LBMSolver
from src.core.multiphase_3d import MultiphaseFlow3D
from src.physics.coffee_particles import CoffeeParticleSystem
from src.physics.precise_pouring import PrecisePouringSystem
from src.physics.filter_paper import FilterPaperSystem

class SimpleCoffeeSimulation:
    def __init__(self):
        print("🔧 咖啡模擬初始化...")
        self.step_count = 0
        
        # 創建核心組件
        print("  └─ 創建LBM求解器...")
        self.lbm = LBMSolver()
        
        print("  └─ 創建多相流...")
        self.multiphase = MultiphaseFlow3D(self.lbm)
        
        print("  └─ 創建顆粒系統...")
        self.particle_system = CoffeeParticleSystem(max_particles=2000)  # 減少顆粒數
        
        print("  └─ 創建注水系統...")
        self.pouring = PrecisePouringSystem()
        
        print("  └─ 創建濾紙系統...")
        self.filter_paper = FilterPaperSystem(self.lbm)
        
        # 簡化初始化
        print("  └─ 初始化場變數...")
        self.lbm.init_fields()
        self.multiphase.init_phase_field()
        
        print("  └─ 初始化濾紙幾何...")
        self.filter_paper.initialize_filter_geometry()
        
        print("  └─ 生成咖啡床...")
        particles = self.particle_system.initialize_coffee_bed_confined(self.filter_paper)
        
        print("  └─ 開始注水...")
        self.pouring.start_pouring(pattern='center')
        
        print(f"✅ 模擬就緒 - {particles} 顆粒")
    
    def step(self):
        """執行一步模擬"""
        try:
            # 注水
            if self.step_count > 10:  # 延遲開始注水
                self.pouring.apply_pouring(self.lbm.u, self.lbm.rho, self.multiphase.phi, config.DT)
                
                if self.step_count % 3 == 0:  # 每3步同步一次相場
                    self.multiphase.update_density_from_phase()
            
            # LBM步驟
            self.lbm.step()
            
            # 多相流步驟
            self.multiphase.step()
            
            # 顆粒更新（簡化版）
            if self.step_count > 5:
                import numpy as np
                u_data = self.lbm.u.to_numpy()
                u_magnitude = np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2)
                max_vel = np.max(u_magnitude)
                
                if max_vel < 0.1 and not np.isnan(max_vel):
                    dt_physical = config.DT * config.SCALE_TIME * 0.5
                    boundary = self.filter_paper.get_coffee_bed_boundary()
                    self.particle_system.update_particle_physics(
                        dt_physical,
                        boundary['center_x'], boundary['center_y'], boundary['bottom_z'],
                        boundary['bottom_radius_lu'], boundary['top_radius_lu']
                    )
            
            self.step_count += 1
            return True
            
        except Exception as e:
            print(f"❌ 步驟 {self.step_count} 失敗: {e}")
            return False
    
    def get_stats(self):
        """獲取統計數據"""
        import numpy as np
        try:
            u_data = self.lbm.u.to_numpy()
            rho_data = self.lbm.rho.to_numpy()
            phi_data = self.multiphase.phi.to_numpy()
            
            max_u = np.max(np.sqrt(u_data[:,:,:,0]**2 + u_data[:,:,:,1]**2 + u_data[:,:,:,2]**2))
            avg_rho = np.mean(rho_data)
            avg_phi = np.mean(phi_data)
            
            return {
                'max_velocity': max_u,
                'avg_density': avg_rho,
                'avg_phase': avg_phi,
                'step_count': self.step_count
            }
        except:
            return {'step_count': self.step_count}

def run_simulation(max_steps=250):
    """運行模擬"""
    print("☕ 3D手沖咖啡CFD模擬")
    print("="*50)
    
    # 創建模擬
    sim = SimpleCoffeeSimulation()
    
    print(f"\n🚀 開始運行 {max_steps} 步模擬...")
    
    start_time = time.time()
    success_count = 0
    
    for step in range(max_steps):
        success = sim.step()
        
        if success:
            success_count += 1
            
            # 每10步輸出進度
            if step % 10 == 0:
                stats = sim.get_stats()
                elapsed = time.time() - start_time
                
                print(f"📊 步驟 {step:3d}/{max_steps} | "
                      f"速度: {stats.get('max_velocity', 0):.6f} | "
                      f"密度: {stats.get('avg_density', 1):.3f} | "
                      f"相場: {stats.get('avg_phase', 0):.3f} | "
                      f"時間: {elapsed:.1f}s")
        else:
            print(f"❌ 模擬在第 {step} 步失敗")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n📊 模擬完成:")
    print(f"  └─ 成功步數: {success_count}/{max_steps}")
    print(f"  └─ 成功率: {success_count/max_steps*100:.1f}%")
    print(f"  └─ 總時間: {total_time:.1f}秒")
    print(f"  └─ 平均步長時間: {total_time/max_steps*1000:.1f}ms")
    
    if success_count == max_steps:
        final_stats = sim.get_stats()
        print(f"\n🎉 最終統計:")
        for key, value in final_stats.items():
            print(f"  └─ {key}: {value}")

def main():
    """主函數"""
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        run_simulation(max_steps)
    else:
        print("用法: python working_main.py debug [步數]")

if __name__ == "__main__":
    main()