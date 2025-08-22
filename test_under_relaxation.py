#!/usr/bin/env python3
"""
亞鬆弛穩定控制測試 - P1任務2
實現雙向耦合數值穩定性控制

開發：opencode + GitHub Copilot
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

# 初始化Taichi
ti.init(arch=ti.cpu, debug=False)

# 測試配置
NX = NY = NZ = 16
MAX_PARTICLES = 10

@ti.data_oriented
class UnderRelaxationTest:
    """亞鬆弛穩定控制測試類"""
    
    def __init__(self):
        # 顆粒數據
        self.max_particles = MAX_PARTICLES
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)
        self.active = ti.field(dtype=ti.i32, shape=MAX_PARTICLES)
        
        # 拖曳力歷史追蹤
        self.drag_force_old = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)
        self.drag_force_new = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)
        self.drag_force_final = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)
        
        # 流場
        self.fluid_velocity = ti.Vector.field(3, dtype=ti.f32, shape=(NX, NY, NZ))
        
        # 穩定性監控
        self.force_oscillation_history = ti.field(dtype=ti.f32, shape=100)
        self.convergence_history = ti.field(dtype=ti.f32, shape=100)
        self.step_count = ti.field(dtype=ti.i32, shape=())
        
    @ti.kernel
    def initialize_system(self):
        """初始化測試系統"""
        # 設置流場 - 簡單的剪切流
        for i, j, k in ti.ndrange(NX, NY, NZ):
            shear_rate = 0.01
            u_x = shear_rate * ti.cast(k, ti.f32)  # 沿z方向的剪切
            self.fluid_velocity[i, j, k] = ti.Vector([u_x, 0.0, 0.0])
        
        # 設置顆粒
        for p in range(self.max_particles):
            if p < 5:  # 只激活5個顆粒
                # 放置在網格中心附近
                x = 8.0 + ti.cast(p, ti.f32) * 0.5
                y = 8.0
                z = 8.0
                self.position[p] = ti.Vector([x, y, z])
                self.velocity[p] = ti.Vector([0.0, 0.0, 0.0])
                self.active[p] = 1
                
                # 初始化力
                self.drag_force_old[p] = ti.Vector([0.0, 0.0, 0.0])
                self.drag_force_new[p] = ti.Vector([0.0, 0.0, 0.0])
                self.drag_force_final[p] = ti.Vector([0.0, 0.0, 0.0])
            else:
                self.active[p] = 0
        
        self.step_count[None] = 0
    
    @ti.func
    def interpolate_fluid_velocity(self, particle_idx: ti.i32) -> ti.math.vec3:
        """插值流體速度到顆粒位置"""
        pos = self.position[particle_idx]
        
        # 簡單的最近鄰插值
        i = ti.cast(ti.max(0, ti.min(NX-1, pos[0])), ti.i32)
        j = ti.cast(ti.max(0, ti.min(NY-1, pos[1])), ti.i32)
        k = ti.cast(ti.max(0, ti.min(NZ-1, pos[2])), ti.i32)
        
        return self.fluid_velocity[i, j, k]
    
    @ti.kernel
    def compute_drag_forces_without_relaxation(self):
        """計算拖曳力（不使用亞鬆弛）- 用於對比"""
        drag_coeff = 0.1
        
        for p in range(self.max_particles):
            if self.active[p] == 1:
                # 獲取相對速度
                fluid_vel = self.interpolate_fluid_velocity(p)
                relative_vel = fluid_vel - self.velocity[p]
                
                # 計算拖曳力
                drag_magnitude = drag_coeff * relative_vel.norm()
                if relative_vel.norm() > 1e-8:
                    drag_direction = relative_vel / relative_vel.norm()
                    self.drag_force_new[p] = drag_magnitude * drag_direction
                else:
                    self.drag_force_new[p] = ti.Vector([0.0, 0.0, 0.0])
                
                # 不使用亞鬆弛，直接更新
                self.drag_force_final[p] = self.drag_force_new[p]
    
    @ti.kernel
    def compute_drag_forces_with_relaxation(self, relaxation_factor: ti.f32):
        """計算拖曳力（使用亞鬆弛）- P1任務2核心實現"""
        drag_coeff = 0.1
        
        for p in range(self.max_particles):
            if self.active[p] == 1:
                # 獲取相對速度
                fluid_vel = self.interpolate_fluid_velocity(p)
                relative_vel = fluid_vel - self.velocity[p]
                
                # 計算新的拖曳力
                drag_magnitude = drag_coeff * relative_vel.norm()
                if relative_vel.norm() > 1e-8:
                    drag_direction = relative_vel / relative_vel.norm()
                    self.drag_force_new[p] = drag_magnitude * drag_direction
                else:
                    self.drag_force_new[p] = ti.Vector([0.0, 0.0, 0.0])
                
                # 亞鬆弛公式：F_final = α·F_new + (1-α)·F_old
                self.drag_force_final[p] = (
                    relaxation_factor * self.drag_force_new[p] + 
                    (1.0 - relaxation_factor) * self.drag_force_old[p]
                )
                
                # 更新歷史值
                self.drag_force_old[p] = self.drag_force_final[p]
    
    @ti.kernel
    def update_particles(self, dt: ti.f32):
        """更新顆粒運動"""
        for p in range(self.max_particles):
            if self.active[p] == 1:
                # 簡化的運動方程：只考慮拖曳力
                mass = 1e-6  # 微小質量
                acceleration = self.drag_force_final[p] / mass
                
                # 更新速度和位置
                self.velocity[p] += acceleration * dt
                self.position[p] += self.velocity[p] * dt
                
                # 邊界約束
                for dim in ti.static(range(3)):
                    if self.position[p][dim] < 1.0:
                        self.position[p][dim] = 1.0
                        self.velocity[p][dim] = 0.0
                    elif self.position[p][dim] > 14.0:
                        self.position[p][dim] = 14.0
                        self.velocity[p][dim] = 0.0
    
    @ti.kernel
    def compute_stability_metrics(self) -> ti.f32:
        """計算穩定性指標"""
        total_force_change = 0.0
        
        for p in range(self.max_particles):
            if self.active[p] == 1:
                force_change = (self.drag_force_new[p] - self.drag_force_old[p]).norm()
                total_force_change += force_change
        
        return total_force_change
    
    @ti.kernel
    def record_history(self, force_change: ti.f32, convergence_metric: ti.f32):
        """記錄歷史數據"""
        step = self.step_count[None]
        if step < 100:
            self.force_oscillation_history[step] = force_change
            self.convergence_history[step] = convergence_metric
        self.step_count[None] += 1

def run_stability_comparison():
    """運行穩定性比較測試"""
    print("="*60)
    print("🔬 P1任務2：亞鬆弛穩定控制測試")
    print("="*60)
    
    # 1. 初始化測試系統
    print("\n1️⃣ 初始化亞鬆弛測試系統...")
    test_system = UnderRelaxationTest()
    test_system.initialize_system()
    print("   ✅ 測試系統初始化完成")
    
    # 2. 測試不同的亞鬆弛因子
    relaxation_factors = [0.1, 0.3, 0.5, 0.8, 1.0]  # 1.0表示無亞鬆弛
    results = {}
    
    for alpha in relaxation_factors:
        print(f"\n2️⃣ 測試亞鬆弛因子 α = {alpha}")
        
        # 重新初始化系統
        test_system.initialize_system()
        
        # 運行模擬
        dt = 0.001
        num_steps = 50
        force_oscillations = []
        convergence_metrics = []
        
        for step in range(num_steps):
            # 計算拖曳力
            if alpha == 1.0:
                test_system.compute_drag_forces_without_relaxation()
            else:
                test_system.compute_drag_forces_with_relaxation(alpha)
            
            # 更新顆粒
            test_system.update_particles(dt)
            
            # 計算穩定性指標
            force_change = test_system.compute_stability_metrics()
            
            # 計算收斂性指標（速度變化率）
            particle_velocities = test_system.velocity.to_numpy()
            active_particles = test_system.active.to_numpy()
            
            # 只考慮活躍顆粒
            active_vels = particle_velocities[active_particles == 1]
            if len(active_vels) > 0:
                velocity_magnitude = np.mean(np.linalg.norm(active_vels, axis=1))
            else:
                velocity_magnitude = 0.0
            
            force_oscillations.append(force_change)
            convergence_metrics.append(velocity_magnitude)
        
        # 分析結果
        avg_oscillation = np.mean(force_oscillations[10:])  # 跳過初始階段
        final_convergence = convergence_metrics[-1]
        oscillation_std = np.std(force_oscillations[10:])
        
        results[alpha] = {
            'avg_oscillation': avg_oscillation,
            'final_convergence': final_convergence,
            'oscillation_std': oscillation_std,
            'force_history': force_oscillations,
            'convergence_history': convergence_metrics
        }
        
        print(f"   - 平均力振盪: {avg_oscillation:.6f}")
        print(f"   - 最終收斂值: {final_convergence:.6f}")
        print(f"   - 振盪標準差: {oscillation_std:.6f}")
    
    # 3. 分析穩定性
    print("\n3️⃣ 亞鬆弛穩定性分析...")
    
    best_alpha = None
    best_stability = float('inf')
    
    print("   亞鬆弛因子 | 平均振盪  | 振盪標準差 | 收斂值    | 穩定性評分")
    print("   ----------|----------|----------|----------|----------")
    
    for alpha in relaxation_factors:
        result = results[alpha]
        # 穩定性評分：低振盪 + 低標準差 = 更穩定
        stability_score = result['avg_oscillation'] + result['oscillation_std']
        
        print(f"   {alpha:8.1f}  | {result['avg_oscillation']:8.6f} | "
              f"{result['oscillation_std']:8.6f} | {result['final_convergence']:8.6f} | "
              f"{stability_score:8.6f}")
        
        if stability_score < best_stability:
            best_stability = stability_score
            best_alpha = alpha
    
    print(f"\n   🏆 最佳亞鬆弛因子: α = {best_alpha} (穩定性評分: {best_stability:.6f})")
    
    # 4. 數值穩定性驗證
    print("\n4️⃣ 數值穩定性驗證...")
    
    # 檢查是否有發散情況
    stable_cases = 0
    for alpha in relaxation_factors:
        result = results[alpha]
        # 檢查最後10步的力是否穩定
        last_forces = result['force_history'][-10:]
        if all(f < 1.0 for f in last_forces):  # 力保持在合理範圍
            stable_cases += 1
            status = "✅ 穩定"
        else:
            status = "❌ 不穩定"
        
        print(f"   α = {alpha}: {status}")
    
    # 5. 性能與收斂速度分析
    print("\n5️⃣ 收斂速度分析...")
    
    for alpha in relaxation_factors:
        result = results[alpha]
        convergence_hist = result['convergence_history']
        
        # 找到接近穩態的時間步
        final_value = convergence_hist[-1]
        tolerance = 0.05 * abs(final_value) if final_value > 0 else 0.001
        
        convergence_step = -1
        for i in range(10, len(convergence_hist)):
            if abs(convergence_hist[i] - final_value) < tolerance:
                convergence_step = i
                break
        
        if convergence_step > 0:
            print(f"   α = {alpha}: 收斂時間 {convergence_step} 步")
        else:
            print(f"   α = {alpha}: 未完全收斂")
    
    # 6. 綜合評估
    print("\n" + "="*60)
    
    # 評估標準
    stability_good = (stable_cases >= len(relaxation_factors) * 0.8)
    best_alpha_reasonable = (0.1 <= best_alpha <= 0.8)  # 合理的亞鬆弛範圍
    
    print("🎯 亞鬆弛穩定控制測試結果：")
    print(f"   - 穩定案例比例: {stable_cases}/{len(relaxation_factors)} ({'✅' if stability_good else '❌'})")
    print(f"   - 最佳亞鬆弛因子: {best_alpha} ({'✅' if best_alpha_reasonable else '❌'})")
    print(f"   - 數值穩定性: {'✅' if stability_good else '❌'}")
    
    if stability_good and best_alpha_reasonable:
        print("\n🎉 P1任務2測試全部通過！")
        print("✅ 亞鬆弛穩定控制正確實現")
        print("✅ 數值穩定性得到保證")
        print("✅ 最佳參數識別成功")
        print(f"✅ 推薦亞鬆弛因子: α = {best_alpha}")
        return True
    else:
        print("\n❌ P1任務2測試失敗")
        return False

if __name__ == "__main__":
    try:
        success = run_stability_comparison()
        if success:
            print("\n🚀 P1任務2完成：亞鬆弛穩定控制實現成功！")
        else:
            print("\n❌ P1任務2失敗，需要進一步調試")
    except Exception as e:
        print(f"\n💥 測試異常: {e}")
        import traceback
        traceback.print_exc()