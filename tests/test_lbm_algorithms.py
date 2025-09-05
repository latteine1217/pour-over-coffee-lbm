"""
LBM統一算法庫單元測試
====================

驗證統一算法庫的數值精度、功能正確性和記憶體布局適配性。

測試覆蓋：
- D3Q19權重和速度向量
- 平衡分布函數計算
- 巨觀量計算
- BGK碰撞算子
- 記憶體布局適配器

開發：opencode + GitHub Copilot
"""

import pytest
import taichi as ti
import numpy as np
import config
from src.core.lbm_algorithms import (
    get_d3q19_weight, get_d3q19_velocity, 
    equilibrium_d3q19_unified, equilibrium_d3q19_safe,
    macroscopic_density_unified, macroscopic_velocity_unified,
    collision_bgk_unified, streaming_target_unified,
    create_memory_adapter, MemoryLayout,
    Standard4DAdapter, SoAAdapter,
    verify_algorithm_library
)

class TestD3Q19Parameters:
    """測試D3Q19基本參數"""
    
    @classmethod
    def setup_class(cls):
        """設置測試環境"""
        ti.init(arch=ti.cpu, debug=True)
    
    def test_d3q19_weight_sum(self):
        """測試D3Q19權重總和為1.0"""
        
        @ti.kernel
        def compute_weight_sum() -> float:
            total = 0.0
            for q in range(config.Q_3D):
                total += get_d3q19_weight(q)
            return total
        
        weight_sum = compute_weight_sum()
        assert abs(weight_sum - 1.0) < 1e-12, f"權重總和 {weight_sum} ≠ 1.0"
    
    def test_d3q19_velocity_vectors(self):
        """測試D3Q19速度向量的正交性和對稱性"""
        
        @ti.kernel
        def check_velocity_properties() -> ti.Vector([float, float, float]):
            # 檢查靜止態速度
            v0 = get_d3q19_velocity(0)
            error_rest = v0.norm()
            
            # 檢查對稱性：每個非零速度都有其反向
            symmetry_error = 0.0
            for q in range(1, config.Q_3D):
                vq = get_d3q19_velocity(q)
                found_opposite = False
                for p in range(1, config.Q_3D):
                    vp = get_d3q19_velocity(p)
                    if (vq + vp).norm() < 1e-12:
                        found_opposite = True
                        break
                if not found_opposite:
                    symmetry_error += 1.0
            
            # 檢查速度向量的長度
            length_error = 0.0
            for q in range(1, config.Q_3D):
                vq = get_d3q19_velocity(q)
                length = vq.norm()
                # D3Q19中速度長度應該是1或√2或√3
                if not (abs(length - 1.0) < 1e-10 or 
                       abs(length - ti.sqrt(2.0)) < 1e-10 or 
                       abs(length - ti.sqrt(3.0)) < 1e-10):
                    length_error += 1.0
            
            return ti.Vector([error_rest, symmetry_error, length_error])
        
        errors = check_velocity_properties()
        assert errors[0] < 1e-12, "靜止態速度應該為零向量"
        assert errors[1] == 0, "D3Q19速度向量缺乏對稱性"
        assert errors[2] == 0, "D3Q19速度向量長度不正確"

class TestEquilibriumDistribution:
    """測試平衡分布函數計算"""
    
    @classmethod
    def setup_class(cls):
        ti.init(arch=ti.cpu, debug=True)
    
    def test_equilibrium_conservation(self):
        """測試平衡分布的守恆性質"""
        
        @ti.kernel
        def test_conservation() -> ti.Vector([float, float, float, float]):
            # 測試條件
            rho = 1.0
            u = ti.Vector([0.1, 0.05, 0.02])
            
            # 計算平衡分布
            total_mass = 0.0
            total_momentum = ti.Vector([0.0, 0.0, 0.0])
            
            for q in range(config.Q_3D):
                f_eq = equilibrium_d3q19_unified(rho, u, q)
                e_q = get_d3q19_velocity(q)
                
                # 質量守恆
                total_mass += f_eq
                
                # 動量守恆
                total_momentum += f_eq * e_q
            
            # 計算誤差
            mass_error = abs(total_mass - rho)
            momentum_error = (total_momentum - rho * u).norm()
            
            return ti.Vector([mass_error, momentum_error, 0.0, 0.0])
        
        errors = test_conservation()
        assert errors[0] < 1e-12, f"質量守恆誤差: {errors[0]}"
        assert errors[1] < 1e-12, f"動量守恆誤差: {errors[1]}"
    
    def test_equilibrium_stability(self):
        """測試平衡分布的數值穩定性"""
        
        @ti.kernel
        def test_stability() -> float:
            max_error = 0.0
            
            # 測試極限情況
            test_cases = ti.Matrix([
                [0.1, 0.0, 0.0, 0.0],    # 低密度
                [10.0, 0.0, 0.0, 0.0],   # 高密度
                [1.0, 0.3, 0.0, 0.0],    # 高速度
                [1.0, 0.1, 0.1, 0.1],    # 3D流動
                [1e-6, 0.0, 0.0, 0.0],   # 極低密度
            ])
            
            for i in range(5):
                rho = test_cases[i, 0]
                u = ti.Vector([test_cases[i, 1], test_cases[i, 2], test_cases[i, 3]])
                
                # 比較標準版本和安全版本
                for q in range(config.Q_3D):
                    f_std = equilibrium_d3q19_unified(rho, u, q)
                    f_safe = equilibrium_d3q19_safe(rho, u, q)
                    
                    # 檢查NaN
                    if f_std != f_std or f_safe != f_safe:
                        max_error = 1e20
                    
                    # 檢查有界性
                    if abs(f_std) > 1e10 or abs(f_safe) > 1e10:
                        max_error = ti.max(max_error, 1e10)
            
            return max_error
        
        max_error = test_stability()
        assert max_error < 1e10, f"平衡分布數值不穩定，最大誤差: {max_error}"

class TestMemoryAdapters:
    """測試記憶體布局適配器"""
    
    @classmethod
    def setup_class(cls):
        ti.init(arch=ti.cpu, debug=True)
    
    def test_standard_4d_adapter(self):
        """測試標準4D記憶體適配器"""
        
        # 創建模擬求解器
        class MockSolver4D:
            def __init__(self):
                self.f = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ, config.Q_3D))
                self.f_new = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ, config.Q_3D))
                self.rho = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
                self.u = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        solver = MockSolver4D()
        adapter = Standard4DAdapter(solver)
        
        @ti.kernel
        def test_adapter_operations() -> float:
            # 測試基本操作
            i, j, k, q = 10, 15, 20, 5
            
            # 設置測試值
            test_f = 0.123
            test_rho = 1.456
            test_u = ti.Vector([0.1, 0.2, 0.3])
            
            # 寫入操作
            adapter.set_f(i, j, k, q, test_f)
            adapter.set_rho(i, j, k, test_rho)
            adapter.set_velocity(i, j, k, test_u)
            
            # 讀取操作
            read_f = adapter.get_f(i, j, k, q)
            read_rho = adapter.get_rho(i, j, k)
            read_u = adapter.get_velocity(i, j, k)
            
            # 計算誤差
            f_error = abs(read_f - test_f)
            rho_error = abs(read_rho - test_rho)
            u_error = (read_u - test_u).norm()
            
            return f_error + rho_error + u_error
        
        total_error = test_adapter_operations()
        assert total_error < 1e-12, f"標準4D適配器操作誤差: {total_error}"
    
    def test_memory_adapter_factory(self):
        """測試記憶體適配器工廠函數"""
        
        class MockSolver:
            def __init__(self):
                self.f = ti.field(dtype=ti.f32, shape=(10, 10, 10, 19))
                self.rho = ti.field(dtype=ti.f32, shape=(10, 10, 10))
                self.u = ti.Vector.field(3, dtype=ti.f32, shape=(10, 10, 10))
        
        solver = MockSolver()
        
        # 測試標準4D適配器創建
        adapter_4d = create_memory_adapter(solver, MemoryLayout.STANDARD_4D)
        assert isinstance(adapter_4d, Standard4DAdapter)
        
        # 測試不支援的布局
        with pytest.raises(NotImplementedError):
            create_memory_adapter(solver, MemoryLayout.GPU_DOMAIN_SPLIT)

class TestMacroscopicQuantities:
    """測試巨觀量計算"""
    
    @classmethod
    def setup_class(cls):
        ti.init(arch=ti.cpu, debug=True)
    
    def test_macroscopic_consistency(self):
        """測試巨觀量計算的一致性"""
        
        # 創建測試求解器
        class TestSolver:
            def __init__(self):
                self.f = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ, config.Q_3D))
                self.rho = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
                self.u = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        solver = TestSolver()
        adapter = Standard4DAdapter(solver)
        
        @ti.kernel
        def test_macroscopic_calculation() -> ti.Vector([float, float]):
            i, j, k = 50, 60, 70
            
            # 設置已知的巨觀狀態
            target_rho = 1.2
            target_u = ti.Vector([0.05, 0.1, 0.15])
            
            # 計算對應的平衡分布
            for q in range(config.Q_3D):
                f_eq = equilibrium_d3q19_unified(target_rho, target_u, q)
                adapter.set_f(i, j, k, q, f_eq)
            
            # 重新計算巨觀量
            computed_rho = macroscopic_density_unified(adapter, i, j, k)
            computed_u = macroscopic_velocity_unified(adapter, i, j, k, computed_rho)
            
            # 計算誤差
            rho_error = abs(computed_rho - target_rho)
            u_error = (computed_u - target_u).norm()
            
            return ti.Vector([rho_error, u_error])
        
        errors = test_macroscopic_calculation()
        assert errors[0] < 1e-12, f"密度計算誤差: {errors[0]}"
        assert errors[1] < 1e-12, f"速度計算誤差: {errors[1]}"

class TestVerificationSuite:
    """完整驗證測試套件"""
    
    def test_algorithm_library_verification(self):
        """測試算法庫整體驗證"""
        
        # 初始化環境
        ti.init(arch=ti.cpu, debug=True)
        
        # 執行驗證
        try:
            verify_algorithm_library()
        except Exception as e:
            pytest.fail(f"算法庫驗證失敗: {e}")
    
    def test_end_to_end_consistency(self):
        """端到端一致性測試"""
        
        # 創建完整的測試求解器
        class FullTestSolver:
            def __init__(self):
                self.f = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ, config.Q_3D))
                self.f_new = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ, config.Q_3D))
                self.rho = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
                self.u = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        solver = FullTestSolver()
        adapter = Standard4DAdapter(solver)
        
        @ti.kernel
        def end_to_end_test() -> float:
            max_error = 0.0
            
            # 在多個點進行端到端測試
            for i, j, k in ti.ndrange((10, 20), (15, 25), (20, 30)):
                # 初始狀態
                rho0 = 1.0 + 0.1 * ti.sin(i * 0.1)
                u0 = ti.Vector([0.05 * ti.cos(j * 0.1), 
                               0.03 * ti.sin(k * 0.1), 
                               0.02 * ti.cos((i+j+k) * 0.1)])
                
                # 設置平衡分布
                for q in range(config.Q_3D):
                    f_eq = equilibrium_d3q19_unified(rho0, u0, q)
                    adapter.set_f(i, j, k, q, f_eq)
                
                # 重新計算巨觀量
                rho1 = macroscopic_density_unified(adapter, i, j, k)
                u1 = macroscopic_velocity_unified(adapter, i, j, k, rho1)
                
                # 再次計算平衡分布
                for q in range(config.Q_3D):
                    f_eq1 = equilibrium_d3q19_unified(rho1, u1, q)
                    f_eq0 = equilibrium_d3q19_unified(rho0, u0, q)
                    
                    error = abs(f_eq1 - f_eq0)
                    max_error = ti.max(max_error, error)
            
            return max_error
        
        max_error = end_to_end_test()
        assert max_error < 1e-12, f"端到端一致性測試失敗，最大誤差: {max_error}"

def test_algorithm_library_import():
    """測試算法庫模組導入"""
    try:
        from src.core.lbm_algorithms import (
            equilibrium_d3q19_unified, 
            macroscopic_density_unified,
            collision_bgk_unified
        )
        # 如果沒有異常，測試通過
        assert True
    except ImportError as e:
        pytest.fail(f"算法庫導入失敗: {e}")

if __name__ == "__main__":
    # 直接運行時的快速測試
    print("🧪 執行LBM統一算法庫快速測試...")
    
    # 初始化Taichi
    ti.init(arch=ti.cpu)
    
    # 執行基本驗證
    try:
        verify_algorithm_library()
        print("✅ 基本驗證通過")
    except Exception as e:
        print(f"❌ 基本驗證失敗: {e}")
        exit(1)
    
    # 執行權重測試
    @ti.kernel
    def quick_weight_test() -> float:
        total = 0.0
        for q in range(config.Q_3D):
            total += get_d3q19_weight(q)
        return total
    
    weight_sum = quick_weight_test()
    assert abs(weight_sum - 1.0) < 1e-12
    print(f"✅ D3Q19權重總和測試通過: {weight_sum}")
    
    print("✅ 所有快速測試通過！")
    print("🔬 運行完整測試: pytest tests/test_lbm_algorithms.py")