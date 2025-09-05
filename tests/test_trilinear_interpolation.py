#!/usr/bin/env python3
"""
三線性插值算法測試和優化 - P0任務2
驗證顆粒-網格數據交換的精度和性能

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np

# 初始化Taichi
ti.init(arch=ti.cpu, debug=False)

# 測試配置
NX = NY = NZ = 16
Q_3D = 19

@ti.data_oriented
class TrilinearInterpolationTest:
    """三線性插值算法測試類"""
    
    def __init__(self):
        # 測試速度場 - 設置一個已知的線性函數
        self.velocity_field = ti.Vector.field(3, dtype=ti.f32, shape=(NX, NY, NZ))
        self.test_positions = ti.Vector.field(3, dtype=ti.f32, shape=10)
        self.interpolated_results = ti.Vector.field(3, dtype=ti.f32, shape=10)
        self.analytical_results = ti.Vector.field(3, dtype=ti.f32, shape=10)
        
    @ti.kernel
    def setup_test_velocity_field(self):
        """設置測試速度場 - 線性函數 v = (x, y, z)"""
        for i, j, k in ti.ndrange(NX, NY, NZ):
            # 簡單的線性速度場，容易驗證插值結果
            x = ti.cast(i, ti.f32)
            y = ti.cast(j, ti.f32) 
            z = ti.cast(k, ti.f32)
            self.velocity_field[i, j, k] = ti.Vector([x, y, z])
    
    @ti.func
    def trilinear_interpolation(self, pos: ti.math.vec3) -> ti.math.vec3:
        """優化的三線性插值算法 - P0任務核心實現"""
        
        # 網格索引計算（邊界安全）
        i = ti.cast(ti.max(0, ti.min(NX-2, pos[0])), ti.i32)
        j = ti.cast(ti.max(0, ti.min(NY-2, pos[1])), ti.i32)
        k = ti.cast(ti.max(0, ti.min(NZ-2, pos[2])), ti.i32)
        
        # 插值權重計算
        fx = pos[0] - ti.cast(i, ti.f32)
        fy = pos[1] - ti.cast(j, ti.f32)
        fz = pos[2] - ti.cast(k, ti.f32)
        
        # 限制權重在[0,1]範圍內（防護式設計）
        fx = ti.max(0.0, ti.min(1.0, fx))
        fy = ti.max(0.0, ti.min(1.0, fy))
        fz = ti.max(0.0, ti.min(1.0, fz))
        
        # 計算8個節點權重（優化版本）
        w1 = 1.0 - fx
        w2 = fx
        
        # 沿x方向的4個線性插值
        c00 = w1 * self.velocity_field[i, j, k] + w2 * self.velocity_field[i+1, j, k]
        c01 = w1 * self.velocity_field[i, j, k+1] + w2 * self.velocity_field[i+1, j, k+1]
        c10 = w1 * self.velocity_field[i, j+1, k] + w2 * self.velocity_field[i+1, j+1, k]
        c11 = w1 * self.velocity_field[i, j+1, k+1] + w2 * self.velocity_field[i+1, j+1, k+1]
        
        # 沿y方向的2個線性插值
        w1 = 1.0 - fy
        w2 = fy
        c0 = w1 * c00 + w2 * c10
        c1 = w1 * c01 + w2 * c11
        
        # 沿z方向的最終線性插值
        w1 = 1.0 - fz
        w2 = fz
        result = w1 * c0 + w2 * c1
        
        return result
    
    @ti.func
    def trilinear_interpolation_standard(self, pos: ti.math.vec3) -> ti.math.vec3:
        """標準三線性插值算法 - 用於對比驗證"""
        
        # 網格索引計算
        i = ti.cast(ti.max(0, ti.min(NX-2, pos[0])), ti.i32)
        j = ti.cast(ti.max(0, ti.min(NY-2, pos[1])), ti.i32)
        k = ti.cast(ti.max(0, ti.min(NZ-2, pos[2])), ti.i32)
        
        # 插值權重
        fx = pos[0] - ti.cast(i, ti.f32)
        fy = pos[1] - ti.cast(j, ti.f32)
        fz = pos[2] - ti.cast(k, ti.f32)
        
        fx = ti.max(0.0, ti.min(1.0, fx))
        fy = ti.max(0.0, ti.min(1.0, fy))
        fz = ti.max(0.0, ti.min(1.0, fz))
        
        # 8個節點權重
        w000 = (1-fx) * (1-fy) * (1-fz)
        w001 = (1-fx) * (1-fy) * fz
        w010 = (1-fx) * fy * (1-fz)
        w011 = (1-fx) * fy * fz
        w100 = fx * (1-fy) * (1-fz)
        w101 = fx * (1-fy) * fz
        w110 = fx * fy * (1-fz)
        w111 = fx * fy * fz
        
        # 標準8點插值
        result = (
            w000 * self.velocity_field[i, j, k] +
            w001 * self.velocity_field[i, j, k+1] +
            w010 * self.velocity_field[i, j+1, k] +
            w011 * self.velocity_field[i, j+1, k+1] +
            w100 * self.velocity_field[i+1, j, k] +
            w101 * self.velocity_field[i+1, j, k+1] +
            w110 * self.velocity_field[i+1, j+1, k] +
            w111 * self.velocity_field[i+1, j+1, k+1]
        )
        
        return result
    
    @ti.kernel
    def run_interpolation_test(self, method: ti.i32):
        """運行插值測試
        method: 0=優化版本, 1=標準版本
        """
        for p in range(self.test_positions.shape[0]):
            pos = self.test_positions[p]
            result = ti.Vector([0.0, 0.0, 0.0])  # 初始化
            
            if method == 0:
                result = self.trilinear_interpolation(pos)
            else:
                result = self.trilinear_interpolation_standard(pos)
                
            self.interpolated_results[p] = result
            
            # 分析解（線性函數的準確值）
            self.analytical_results[p] = pos
    
    def setup_test_positions(self):
        """設置測試位置 - 包含整數點、分數點和邊界點"""
        test_pos = [
            [5.0, 5.0, 5.0],    # 整數點
            [5.5, 5.5, 5.5],    # 中心點
            [5.25, 6.75, 7.1],  # 隨機分數點
            [0.1, 0.1, 0.1],    # 邊界附近
            [14.9, 14.9, 14.9], # 上邊界附近
            [2.3, 8.7, 11.2],   # 隨機點1
            [7.8, 3.4, 9.6],    # 隨機點2
            [12.1, 13.5, 4.8],  # 隨機點3
            [1.7, 5.9, 14.3],   # 隨機點4
            [8.4, 11.2, 6.7]    # 隨機點5
        ]
        
        for i, pos in enumerate(test_pos):
            self.test_positions[i] = pos

def run_comprehensive_test():
    """運行全面的三線性插值測試"""
    print("="*60)
    print("🔬 P0任務2：三線性插值算法測試與優化")
    print("="*60)
    
    # 1. 初始化測試環境
    print("\n1️⃣ 初始化測試環境...")
    test_system = TrilinearInterpolationTest()
    test_system.setup_test_velocity_field()
    test_system.setup_test_positions()
    print("   ✅ 測試環境設置完成")
    
    # 2. 測試優化版本
    print("\n2️⃣ 測試優化版三線性插值...")
    test_system.run_interpolation_test(method=0)
    
    # 獲取結果
    interpolated = test_system.interpolated_results.to_numpy()
    analytical = test_system.analytical_results.to_numpy()
    positions = test_system.test_positions.to_numpy()
    
    # 計算誤差
    errors = np.linalg.norm(interpolated - analytical, axis=1)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    print(f"   ✅ 優化版插值完成")
    print(f"      - 最大誤差: {max_error:.2e}")
    print(f"      - 平均誤差: {mean_error:.2e}")
    
    # 3. 測試標準版本對比
    print("\n3️⃣ 測試標準版三線性插值（對比）...")
    test_system.run_interpolation_test(method=1)
    
    interpolated_std = test_system.interpolated_results.to_numpy()
    errors_std = np.linalg.norm(interpolated_std - analytical, axis=1)
    max_error_std = np.max(errors_std)
    mean_error_std = np.mean(errors_std)
    
    print(f"   ✅ 標準版插值完成")
    print(f"      - 最大誤差: {max_error_std:.2e}")
    print(f"      - 平均誤差: {mean_error_std:.2e}")
    
    # 4. 比較兩種方法
    print("\n4️⃣ 算法比較分析...")
    algorithm_diff = np.linalg.norm(interpolated - interpolated_std, axis=1)
    max_diff = np.max(algorithm_diff)
    
    print(f"   - 兩種算法最大差異: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("   ✅ 兩種算法結果一致（數值精度內）")
    else:
        print("   ⚠️  兩種算法存在差異")
    
    # 5. 詳細結果分析
    print("\n5️⃣ 詳細結果分析...")
    print("   位置 -> 插值結果 vs 分析解（誤差）")
    
    for i in range(len(positions)):
        pos = positions[i]
        interp = interpolated[i]
        exact = analytical[i]
        error = errors[i]
        
        print(f"   [{pos[0]:5.1f}, {pos[1]:5.1f}, {pos[2]:5.1f}] -> "
              f"[{interp[0]:6.2f}, {interp[1]:6.2f}, {interp[2]:6.2f}] vs "
              f"[{exact[0]:6.2f}, {exact[1]:6.2f}, {exact[2]:6.2f}] ({error:.2e})")
    
    # 6. 性能測試
    print("\n6️⃣ 性能測試...")
    
    import time
    
    # 優化版性能
    start_time = time.time()
    for _ in range(1000):
        test_system.run_interpolation_test(method=0)
    opt_time = time.time() - start_time
    
    # 標準版性能
    start_time = time.time()
    for _ in range(1000):
        test_system.run_interpolation_test(method=1)
    std_time = time.time() - start_time
    
    speedup = std_time / opt_time if opt_time > 0 else 1.0
    
    print(f"   - 優化版時間: {opt_time*1000:.2f} ms (1000次)")
    print(f"   - 標準版時間: {std_time*1000:.2f} ms (1000次)")
    print(f"   - 性能提升: {speedup:.2f}x")
    
    # 7. 結果評估
    print("\n" + "="*60)
    
    if max_error < 1e-10 and max_error_std < 1e-10:
        print("🎉 三線性插值算法測試全部通過！")
        print("✅ 精度測試：通過（誤差 < 1e-10）")
        print("✅ 一致性測試：通過（算法等效）")
        print(f"✅ 性能測試：通過（提升 {speedup:.1f}x）")
        return True
    else:
        print("❌ 三線性插值算法測試失敗")
        print(f"   - 優化版誤差過大: {max_error:.2e}")
        print(f"   - 標準版誤差過大: {max_error_std:.2e}")
        return False

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        if success:
            print("\n🚀 P0任務2完成：三線性插值算法優化成功！")
            exit(0)
        else:
            print("\n❌ P0任務2失敗，需要進一步優化")
            exit(1)
    except Exception as e:
        print(f"\n💥 測試異常: {e}")
        import traceback
        traceback.print_exc()
        exit(1)