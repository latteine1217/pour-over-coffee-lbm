# main_safe.py
"""
安全版本的主程式 - 用於診斷segmentation fault
"""

import config
from init import initialize_d3q19_simulation, print_d3q19_simulation_info
import time

def safe_main():
    """安全的主模擬循環"""
    print("=== 安全版本D3Q19手沖咖啡模擬 ===")
    
    # 初始化
    try:
        lbm_solver, multiphase, porous_solver, visualizer, particles, pouring = initialize_d3q19_simulation()
        print_d3q19_simulation_info()
        print("✅ 初始化成功")
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return
    
    # 限制運行步數避免長時間運行
    MAX_SAFE_STEPS = 10
    print(f"=== 開始安全模擬 (最多{MAX_SAFE_STEPS}步) ===")
    
    start_time = time.time()
    
    for step in range(MAX_SAFE_STEPS):
        try:
            print(f"步驟 {step}...")
            
            # 簡化的模擬步驟
            dt = 0.01
            
            # 1. 注水
            if step == 0:  # 只在第一步測試注水
                pouring.apply_pouring(lbm_solver.u, lbm_solver.rho, multiphase.phi, dt)
                print("  ✅ 注水成功")
            
            # 2. LBM步驟 (最關鍵的部分)
            if hasattr(lbm_solver, 'step'):
                try:
                    lbm_solver.step()
                    print("  ✅ LBM步驟成功")
                except Exception as e:
                    print(f"  ❌ LBM步驟失敗: {e}")
                    break
            
            # 3. 多相流
            if hasattr(multiphase, 'step'):
                try:
                    multiphase.step()
                    print("  ✅ 多相流步驟成功")
                except Exception as e:
                    print(f"  ❌ 多相流步驟失敗: {e}")
                    break
            
            # 4. 粒子更新 (簡化)
            if hasattr(particles, 'update_particles'):
                try:
                    particles.update_particles(dt)
                    print("  ✅ 粒子更新成功")
                except Exception as e:
                    print(f"  ❌ 粒子更新失敗: {e}")
                    break
            
            print(f"✅ 步驟 {step} 完成")
            
            # 每步之間暫停一下
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ 步驟 {step} 失敗: {e}")
            break
    
    end_time = time.time()
    print(f"=== 安全模擬完成，耗時 {end_time - start_time:.2f}s ===")

if __name__ == "__main__":
    safe_main()