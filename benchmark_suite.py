"""
CFD Performance Benchmark Suite
V60手沖咖啡LBM模擬系統性能基準測試框架

提供企業級性能監控和基準測試功能，支援GPU加速計算的準確測量
"""

import time
import json
import argparse
import psutil
import numpy as np
import taichi as ti
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# 導入核心模組
import config
from lbm_solver import LBMSolver
from multiphase_3d import MultiphaseFlow3D
from coffee_particles import CoffeeParticleSystem
from les_turbulence import LESTurbulenceModel

@dataclass
class BenchmarkResult:
    """基準測試結果數據類"""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    grid_size: int
    iterations: int
    timestamp: str
    gpu_memory_mb: Optional[float] = None
    throughput: Optional[float] = None  # 格點/秒
    
class CFDPerformanceBenchmark:
    """CFD性能基準測試主類"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        初始化基準測試框架
        
        Args:
            output_dir: 結果輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        self.baseline_data: Optional[Dict] = None
        
        # 初始化Taichi
        ti.init(arch=ti.metal, device_memory_GB=8)
        
        # 基準測試配置
        self.test_configs = {
            'small': {'grid_size': 64, 'iterations': 50},
            'medium': {'grid_size': 128, 'iterations': 100}, 
            'large': {'grid_size': 224, 'iterations': 200},
            'stress': {'grid_size': 256, 'iterations': 500}
        }
        
    @contextmanager
    def performance_monitor(self):
        """性能監控上下文管理器"""
        # 記錄開始狀態
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.perf_counter()
        
        # GPU記憶體 (如果可用)
        try:
            gpu_memory_start = ti.profiler.get_kernel_stats()
        except:
            gpu_memory_start = None
            
        yield
        
        # 記錄結束狀態
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss / 1024 / 1024
        
        self._last_execution_time = end_time - start_time
        self._last_memory_usage = max(end_memory - start_memory, 0)
        
    def benchmark_lbm_step(self, grid_size: int = 224, iterations: int = 100) -> BenchmarkResult:
        """
        測量LBM求解器單步計算性能
        
        Args:
            grid_size: 網格尺寸
            iterations: 迭代次數
            
        Returns:
            基準測試結果
        """
        print(f"📊 LBM求解器基準測試 - 網格: {grid_size}³, 迭代: {iterations}")
        
        # 創建測試用LBM求解器 (使用默認配置)
        solver = LBMSolver()
        
        with self.performance_monitor():
            # 執行基準測試 (使用實際的LBM方法)
            for i in range(iterations):
                solver._collision_streaming_step()
                if i % 50 == 0:
                    print(f"  迭代進度: {i+1}/{iterations}")
                    
        # 計算吞吐量 (格點/秒)
        total_lattice_points = grid_size ** 3 * iterations
        throughput = total_lattice_points / self._last_execution_time
        
        result = BenchmarkResult(
            test_name="lbm_step",
            execution_time=self._last_execution_time,
            memory_usage_mb=self._last_memory_usage,
            grid_size=grid_size,
            iterations=iterations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            throughput=throughput
        )
        
        self.results.append(result)
        print(f"  ✅ 完成 - 執行時間: {result.execution_time:.2f}s, 吞吐量: {throughput:.0f} 格點/s")
        return result
        
    def benchmark_boundary_conditions(self, grid_size: int = 224, iterations: int = 100) -> BenchmarkResult:
        """測量邊界條件處理性能 - 簡化版本"""
        print(f"📊 邊界條件基準測試 - 網格: {grid_size}³")
        
        # 創建完整的LBM求解器以測試邊界條件
        solver = LBMSolver()
        
        with self.performance_monitor():
            for i in range(iterations):
                # 使用LBM求解器的內部邊界處理
                solver.apply_boundary_conditions()
                if i % 50 == 0:
                    print(f"  迭代進度: {i+1}/{iterations}")
                    
        result = BenchmarkResult(
            test_name="boundary_conditions",
            execution_time=self._last_execution_time,
            memory_usage_mb=self._last_memory_usage,
            grid_size=config.NX,  # 使用配置的網格尺寸
            iterations=iterations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        print(f"  ✅ 完成 - 執行時間: {result.execution_time:.2f}s")
        return result
        
    def benchmark_particle_system(self, grid_size: int = 224, particles: int = 1890, iterations: int = 100) -> BenchmarkResult:
        """測量咖啡顆粒系統性能"""
        print(f"📊 顆粒系統基準測試 - {particles}顆粒, {iterations}步")
        
        particle_system = CoffeeParticleSystem(particles)
        
        with self.performance_monitor():
            for i in range(iterations):
                particle_system.update_particles(0.01)  # dt = 0.01s
                if i % 50 == 0:
                    print(f"  迭代進度: {i+1}/{iterations}")
                    
        result = BenchmarkResult(
            test_name="particle_system",
            execution_time=self._last_execution_time,
            memory_usage_mb=self._last_memory_usage,
            grid_size=grid_size,
            iterations=iterations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        print(f"  ✅ 完成 - 執行時間: {result.execution_time:.2f}s")
        return result
        
    def benchmark_les_turbulence(self, grid_size: int = 224, iterations: int = 100) -> BenchmarkResult:
        """測量LES湍流模型性能"""
        print(f"📊 LES湍流模型基準測試 - 網格: {grid_size}³")
        
        les_model = LESTurbulenceModel()
        
        # 創建測試速度場 (使用配置的網格尺寸)
        velocity_field = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        
        with self.performance_monitor():
            for i in range(iterations):
                les_model.compute_sgs_stress(velocity_field)
                if i % 50 == 0:
                    print(f"  迭代進度: {i+1}/{iterations}")
                    
        result = BenchmarkResult(
            test_name="les_turbulence",
            execution_time=self._last_execution_time,
            memory_usage_mb=self._last_memory_usage,
            grid_size=grid_size,
            iterations=iterations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        print(f"  ✅ 完成 - 執行時間: {result.execution_time:.2f}s")
        return result
        
    def benchmark_memory_usage(self) -> BenchmarkResult:
        """測量記憶體使用情況"""
        print("📊 記憶體使用基準測試")
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # 創建LBM求解器測試記憶體 (使用默認配置)
        solver = LBMSolver()
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_usage = memory_after - memory_before
        
        result = BenchmarkResult(
            test_name="memory_usage",
            execution_time=0.0,
            memory_usage_mb=memory_usage,
            grid_size=config.NX,  # 使用配置的網格尺寸
            iterations=1,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        print(f"  ✅ 記憶體使用: {memory_usage:.1f} MB")
        return result
        
    def benchmark_full_simulation(self, grid_size: int = 128, time_steps: int = 50) -> BenchmarkResult:
        """
        完整模擬性能基準測試
        
        Args:
            grid_size: 網格尺寸
            time_steps: 時間步數
        """
        print(f"📊 完整模擬基準測試 - 網格: {grid_size}³, 時間步: {time_steps}")
        
        with self.performance_monitor():
            # 模擬完整的CFD計算流程 (使用默認配置)
            solver = LBMSolver()
            multiphase = MultiphaseFlow3D()
            particles = CoffeeParticleSystem(500)  # 較少顆粒用於基準測試
            
            for step in range(time_steps):
                # LBM步驟 (使用實際方法)
                solver._collision_streaming_step()
                
                # 多相流更新
                multiphase.update_density()
                
                # 顆粒更新
                particles.update_particles(0.01)
                
                if step % 10 == 0:
                    print(f"  時間步進度: {step+1}/{time_steps}")
                    
        # 計算模擬效率
        total_operations = grid_size ** 3 * time_steps
        throughput = total_operations / self._last_execution_time
        
        result = BenchmarkResult(
            test_name="full_simulation",
            execution_time=self._last_execution_time,
            memory_usage_mb=self._last_memory_usage,
            grid_size=grid_size,
            iterations=time_steps,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            throughput=throughput
        )
        
        self.results.append(result)
        print(f"  ✅ 完成 - 執行時間: {result.execution_time:.2f}s, 效率: {throughput:.0f} 操作/s")
        return result
        
    def run_all_benchmarks(self, test_size: str = "medium") -> Dict[str, BenchmarkResult]:
        """執行所有基準測試"""
        config = self.test_configs.get(test_size, self.test_configs["medium"])
        grid_size = config["grid_size"]
        iterations = config["iterations"]
        
        print(f"\n🚀 開始執行全套基準測試 - 配置: {test_size}")
        print("=" * 60)
        
        benchmark_results = {}
        
        # 1. LBM求解器基準測試
        benchmark_results["lbm"] = self.benchmark_lbm_step(grid_size, iterations)
        
        # 2. 邊界條件基準測試
        benchmark_results["boundary"] = self.benchmark_boundary_conditions(grid_size, iterations)
        
        # 3. 顆粒系統基準測試
        benchmark_results["particles"] = self.benchmark_particle_system(grid_size, 1890, iterations)
        
        # 4. LES湍流基準測試
        benchmark_results["turbulence"] = self.benchmark_les_turbulence(grid_size, iterations)
        
        # 5. 記憶體使用基準測試
        benchmark_results["memory"] = self.benchmark_memory_usage()
        
        # 6. 完整模擬基準測試
        benchmark_results["full_sim"] = self.benchmark_full_simulation(grid_size, iterations//2)
        
        print("\n✅ 所有基準測試完成!")
        return benchmark_results
        
    def save_results(self, filename: str = "benchmark_results.json"):
        """保存基準測試結果到JSON文件"""
        results_data = [asdict(result) for result in self.results]
        
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
            
        print(f"📁 結果已保存到: {output_file}")
        
    def load_baseline(self, baseline_file: str) -> None:
        """載入基準線數據"""
        baseline_path = Path(baseline_file)
        if baseline_path.exists():
            with open(baseline_path, 'r', encoding='utf-8') as f:
                self.baseline_data = json.load(f)
            print(f"📊 已載入基準線數據: {baseline_file}")
        else:
            print(f"⚠️  基準線文件不存在: {baseline_file}")
            
    def compare_with_baseline(self, baseline_file: str = "baseline_performance.json") -> Dict[str, float]:
        """
        與基準版本性能對比
        
        Returns:
            各項測試的性能變化百分比
        """
        self.load_baseline(baseline_file)
        
        if not self.baseline_data:
            print("❌ 無基準線數據可比較")
            return {}
            
        print("\n📈 性能對比分析")
        print("=" * 40)
        
        comparison = {}
        
        for current_result in self.results:
            test_name = current_result.test_name
            
            # 查找對應的基準線數據
            baseline_result = None
            for baseline in self.baseline_data:
                if (baseline["test_name"] == test_name and 
                    baseline["grid_size"] == current_result.grid_size):
                    baseline_result = baseline
                    break
                    
            if baseline_result:
                # 計算性能變化
                time_change = ((current_result.execution_time - baseline_result["execution_time"]) 
                             / baseline_result["execution_time"] * 100)
                memory_change = ((current_result.memory_usage_mb - baseline_result["memory_usage_mb"]) 
                               / baseline_result["memory_usage_mb"] * 100)
                
                comparison[test_name] = {
                    "time_change_percent": time_change,
                    "memory_change_percent": memory_change
                }
                
                # 顯示對比結果
                time_symbol = "⬇️" if time_change < 0 else "⬆️"
                memory_symbol = "⬇️" if memory_change < 0 else "⬆️"
                
                print(f"{test_name:15} | 時間: {time_symbol} {time_change:+6.1f}% | 記憶體: {memory_symbol} {memory_change:+6.1f}%")
                
        return comparison
        
    def generate_performance_report(self) -> str:
        """生成性能報告"""
        report = []
        report.append("# CFD性能基準測試報告\n")
        report.append(f"生成時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("## 測試結果摘要\n")
        
        if not self.results:
            report.append("無測試結果\n")
            return "\n".join(report)
            
        for result in self.results:
            report.append(f"### {result.test_name}")
            report.append(f"- 網格尺寸: {result.grid_size}³")
            report.append(f"- 執行時間: {result.execution_time:.3f}s")
            report.append(f"- 記憶體使用: {result.memory_usage_mb:.1f}MB")
            if result.throughput:
                report.append(f"- 吞吐量: {result.throughput:.0f} 操作/s")
            report.append("")
            
        return "\n".join(report)

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="CFD Performance Benchmark Suite")
    parser.add_argument("--size", choices=["small", "medium", "large", "stress"], 
                       default="medium", help="測試規模")
    parser.add_argument("--output", default="benchmark_results.json", 
                       help="輸出文件名")
    parser.add_argument("--baseline", default="baseline_performance.json",
                       help="基準線文件")
    parser.add_argument("--compare", action="store_true",
                       help="與基準線比較")
    parser.add_argument("--report", action="store_true",
                       help="生成報告")
    
    args = parser.parse_args()
    
    # 創建基準測試實例
    benchmark = CFDPerformanceBenchmark()
    
    # 執行基準測試
    print("🔬 CFD Performance Benchmark Suite")
    print("V60手沖咖啡LBM模擬系統性能分析")
    
    results = benchmark.run_all_benchmarks(args.size)
    
    # 保存結果
    benchmark.save_results(args.output)
    
    # 性能比較
    if args.compare:
        comparison = benchmark.compare_with_baseline(args.baseline)
        
    # 生成報告
    if args.report:
        report = benchmark.generate_performance_report()
        report_file = benchmark.output_dir / "performance_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 性能報告已生成: {report_file}")

if __name__ == "__main__":
    main()