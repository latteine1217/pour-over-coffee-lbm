# init.py
"""
D3Q19 LBM模擬初始化模組
使用Taichi GPU加速和高性能並行計算
"""

import taichi as ti
import config
import time

# 強制使用CPU (基於性能測試結果)
ti.init(arch=ti.cpu, 
       kernel_profiler=True,
       offline_cache=True)
print("✓ 使用CPU計算 (性能測試證實CPU更優)")

def initialize_d3q19_simulation():
    """初始化完整的D3Q19手沖咖啡模擬系統 - 包含可移動咖啡粉"""
    print("=== 初始化D3Q19手沖咖啡模擬 ===")
    
    # 1. 創建LBM求解器
    print("--- 創建D3Q19 LBM求解器 ---")
    start_time = time.time()
    from lbm_d3q19 import D3Q19_LBM
    lbm_solver = D3Q19_LBM()
    print(f"    LBM求解器創建完成 ({time.time()-start_time:.2f}s)")
    
    # 2. 創建多相流處理器
    print("--- 初始化3D多相流處理 ---")
    start_time = time.time()
    from multiphase_3d import MultiphaseFlow3D
    multiphase = MultiphaseFlow3D(lbm_solver)
    print(f"    多相流模組創建完成 ({time.time()-start_time:.2f}s)")
    
    # 3. 創建可移動咖啡粉粒子系統
    print("--- 創建咖啡粉粒子系統 ---")
    start_time = time.time()
    from coffee_particles import CoffeeParticleSystem
    particle_system = CoffeeParticleSystem(max_particles=20000)
    print(f"    咖啡粒子系統創建完成 ({time.time()-start_time:.2f}s)")
    
    # 4. 創建精確注水系統
    print("--- 創建精確注水系統 ---")
    start_time = time.time()
    from precise_pouring import PrecisePouringSystem
    pouring_system = PrecisePouringSystem()
    print(f"    注水系統創建完成 ({time.time()-start_time:.2f}s)")
    
    # 5. 創建多孔介質求解器 (動態更新)
    print("--- 設置動態多孔介質求解器 ---")
    start_time = time.time()
    from porous_media_3d import PorousMedia3D
    porous_solver = PorousMedia3D(lbm_solver, particle_system)
    print(f"    多孔介質模組創建完成 ({time.time()-start_time:.2f}s)")
    
    # 6. 創建3D視覺化器 (支援粒子顯示)
    print("--- 創建3D視覺化器 ---")
    start_time = time.time()
    from visualization_3d import Visualizer3D
    visualizer = Visualizer3D(lbm_solver, multiphase, particle_system, pouring_system)
    print(f"    視覺化器創建完成 ({time.time()-start_time:.2f}s)")
    
    # 7. 初始化場變數
    print("--- 初始化場變數 ---")
    start_time = time.time()
    lbm_solver.init_fields()
    print(f"    LBM場初始化完成 ({time.time()-start_time:.2f}s)")
    
    start_time = time.time()
    multiphase.init_phase_field()
    print(f"    相場初始化完成 ({time.time()-start_time:.2f}s)")
    
    # 8. 設置V60幾何
    print("--- 設置V60幾何 ---")
    start_time = time.time()
    # TODO: 需要實現3D幾何設置或使用簡化版本
    # 暫時跳過幾何設置，直接設置為基本配置
    print(f"    幾何設置跳過 ({time.time()-start_time:.2f}s)")
    print("    注意：需要實現3D V60幾何設置")
    
    # 9. 初始化咖啡床
    print("--- 初始化咖啡床 ---")
    start_time = time.time()
    bed_height = config.NZ * 0.15      # 咖啡床高度佔總高度15%
    bed_radius = config.NX * 0.3       # 咖啡床半徑
    center_x = config.NX // 2
    center_y = config.NY // 2
    bottom_z = config.NZ * 0.1         # 底部10%位置
    
    particle_system.initialize_coffee_bed(bed_height, bed_radius, center_x, center_y, bottom_z)
    print(f"    咖啡床初始化完成 ({time.time()-start_time:.2f}s)")
    
    # 10. 開始注水 (預設為中心注水)
    print("--- 設置初始注水 ---")
    pouring_system.start_pouring(center_x=center_x, center_y=center_y, 
                                 flow_rate=1.0, pattern='center')
    print(f"    注水系統啟動 - 0.5cm直徑水流")
    
    total_time = time.time() - start_time
    print(f"=== D3Q19初始化完成 ({total_time:.2f}s) ===")
    
    return lbm_solver, multiphase, porous_solver, visualizer, particle_system, pouring_system

def print_d3q19_simulation_info():
    """打印D3Q19模擬參數信息"""
    print(f"""
=== D3Q19 LBM模擬參數 ===
網格尺寸: {config.NX} x {config.NY} x {config.NZ} = {config.NX*config.NY*config.NZ:,} 格點
LBM模型: D3Q19 (19個速度方向)
物理尺寸: {config.CUP_HEIGHT:.3f}m (高) x {config.TOP_DIAMETER:.3f}m (頂直徑)
格子單位轉換: {config.SCALE_LENGTH:.6f} m/lu, {config.SCALE_TIME:.6f} s/lu

流體參數:
- τ_water={config.TAU_WATER}, τ_air={config.TAU_AIR}
- 重力: {config.GRAVITY_LU:.2e} lu/lt²
- 表面張力: {config.SURFACE_TENSION}
- 入水速度: {config.INLET_VELOCITY:.6f} lu/lt

多孔介質:
- 咖啡粉層高度: {config.COFFEE_BED_HEIGHT_LU} lu
- 孔隙率: {config.PORE_PERC:.1%}
- Darcy數: {config.DARCY_NUMBER:.2e}

計算參數:
- 最大步數: {config.MAX_STEPS:,}
- 輸出頻率: {config.OUTPUT_FREQ}
- 並行塊大小: {config.BLOCK_SIZE}
- 稀疏矩陣: {'啟用' if config.USE_SPARSE_MATRIX else '禁用'}
- 記憶體池: {config.MEMORY_POOL_SIZE}MB

估計記憶體使用:
- 主要場變數: ~{(config.NX*config.NY*config.NZ*config.Q*4 + config.NX*config.NY*config.NZ*12)/1024/1024:.1f} MB
- 總記憶體: ~{(config.NX*config.NY*config.NZ*config.Q*4 + config.NX*config.NY*config.NZ*12)*2/1024/1024:.1f} MB
================
""")

def run_performance_test():
    """運行性能測試"""
    print("=== D3Q19性能測試 ===")
    
    # 初始化
    lbm, multiphase, porous, vis = initialize_d3q19_simulation()
    
    # 測試單步性能
    print("--- 測試單步性能 ---")
    test_steps = 10
    
    start_time = time.time()
    for i in range(test_steps):
        lbm.step()
        multiphase.step()
        porous.step()
    ti.sync()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_step = total_time / test_steps
    grid_points = config.NX * config.NY * config.NZ
    throughput = grid_points / avg_time_per_step / 1e6  # MLUPs (Million Lattice Updates Per Second)
    
    print(f"測試結果:")
    print(f"- 總時間: {total_time:.3f}s ({test_steps}步)")
    print(f"- 平均每步: {avg_time_per_step*1000:.2f}ms")
    print(f"- 吞吐量: {throughput:.2f} MLUPs")
    print(f"- 並行效率: {throughput/grid_points*1e6*100:.1f}%")
    
    # Taichi性能分析
    ti.profiler.print_kernel_profiler_info()
    
    return lbm, multiphase, porous, vis

# 向後兼容函數
def initialize_simulation():
    """向後兼容的初始化函數"""
    print("=== 自動升級到D3Q19模擬 ===")
    return initialize_d3q19_simulation()

def print_simulation_info():
    """向後兼容的信息顯示函數"""
    print_d3q19_simulation_info()

def initialize_fields():
    """舊版向後兼容函數"""
    print("=== 警告：使用舊版初始化接口 ===")
    print("=== 建議使用 initialize_d3q19_simulation() ===")
    
    lbm, multiphase, porous, vis = initialize_d3q19_simulation()
    return lbm  # 只返回LBM求解器以保持兼容性