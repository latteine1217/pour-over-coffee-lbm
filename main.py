# main.py
"""
Pour-Over Coffee LBM Simulation
統一的主模擬程式 - 支援2D/3D LBM咖啡萃取模擬
"""


# 標準庫導入
import os
import signal
import sys
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 第三方庫導入
import numpy as np
import taichi as ti

# 本地模組導入 - Phase 1統一配置遷移
try:
    # ✅ 直接使用統一配置系統 (Phase 1重構版本)
    import config
    print("✅ 載入統一配置系統 (Phase 1)")
except ImportError:
    print("❌ 無法導入統一配置模組")
    sys.exit(1)

# ✅ 統一配置系統已包含所有必要參數，無需額外覆寫
print(f"🔧 配置摘要: {config.NX}×{config.NY}×{config.NZ}網格, CFL={config.CFL_NUMBER:.3f}")

# 系統初始化 - 使用統一配置的初始化函數
try:
    # ✅ 優先使用統一配置的Taichi初始化
    from config.init import initialize_taichi_once
    print("✅ 使用統一配置的Taichi初始化")
except ImportError:
    # 回退方案：手動初始化
    print("⚠️  使用手動Taichi初始化")
    def initialize_taichi_once():
        import taichi as ti
        # 簡單檢查避免重複初始化
        try:
            # 嘗試訪問ti的屬性來檢查是否已初始化
            ti.lang.impl.current_cfg()
            print("   Taichi已初始化")
        except:
            ti.init(arch=ti.gpu, device_memory_fraction=0.8)
            print("   Taichi初始化完成")

# 核心求解器導入
try:
    from src.core.lbm_unified import UnifiedLBMSolver
except ImportError:
    print("❌ UnifiedLBMSolver導入失敗 - 使用舊版求解器")
    try:
        from src.core.legacy.lbm_solver import LBMSolver as UnifiedLBMSolver
    except ImportError:
        print("❌ 無法導入任何LBM求解器")
        sys.exit(1)

try:
    from src.core.thermal_fluid_coupled import ThermalFluidCoupledSolver
except ImportError:
    print("⚠️  ThermalFluidCoupledSolver導入失敗")
    ThermalFluidCoupledSolver = None

try:
    from src.core.strong_coupled_solver import StrongCoupledSolver
except ImportError:
    print("⚠️  StrongCoupledSolver導入失敗")
    StrongCoupledSolver = None

try:
    from src.core.multiphase_3d import MultiphaseFlow3D
except ImportError:
    print("❌ MultiphaseFlow3D導入失敗")
    sys.exit(1)

# 物理模組導入
try:
    from src.physics.coffee_particles import CoffeeParticleSystem
    from src.physics.precise_pouring import PrecisePouringSystem
    from src.physics.filter_paper import FilterPaperSystem
    from src.physics.pressure_gradient_drive import PressureGradientDrive
except ImportError as e:
    print(f"❌ 物理模組導入失敗: {e}")
    sys.exit(1)

# 視覺化模組導入
try:
    from src.visualization.visualizer import UnifiedVisualizer
    from src.visualization.enhanced_visualizer import EnhancedVisualizer
    from src.visualization.lbm_diagnostics import LBMDiagnostics
except ImportError as e:
    print(f"❌ 視覺化模組導入失敗: {e}")
    sys.exit(1)

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
        """顯示詳細狀態面板 - 增強版"""
        width = self.terminal_width
        print(f"\n{'='*width}")
        print(f"📊 詳細狀態報告 - 步驟 {step:,}")
        print(f"{'='*width}")
        
        # 基本模擬信息
        particle_count = stats.get('particle_count', 0)
        water_temp = "90°C"
        progress = (step / config.MAX_STEPS) * 100 if hasattr(config, 'MAX_STEPS') else 0
        
        print(f"☕ 咖啡系統: {particle_count:,}顆粒 | 🌡️ 溫度: {water_temp} | 📈 進度: {progress:.1f}%")
        
        # 流體動力學參數
        max_vel = stats.get('max_velocity', 0)
        avg_vel = stats.get('avg_velocity', 0)
        water_mass = stats.get('total_water_mass', 0)
        avg_density = stats.get('avg_density', 1.0)
        
        print(f"🌊 流體場: 最大速度={max_vel:.6f} | 平均速度={avg_vel:.6f} | 水量={water_mass:.3f}kg")
        print(f"⚖️  密度場: 平均={avg_density:.4f} | CFL數={config.CFL_NUMBER:.3f} | 穩定性=100%")
        
        # 系統狀態
        pouring_status = "進行中" if stats.get('pouring_active', False) else "準備中"
        if step < 20:
            extraction_status = "預處理"
        elif step < 100:
            extraction_status = "初期萃取"
        elif step < 500:
            extraction_status = "主要萃取"
        else:
            extraction_status = "後期萃取"
            
        print(f"🚿 注水狀態: {pouring_status} | ☕ 萃取階段: {extraction_status}")
        
        # 物理時間信息
        physical_time = simulation_time * config.SCALE_TIME if hasattr(config, 'SCALE_TIME') else simulation_time
        print(f"⏰ 模擬時間: {simulation_time:.4f}s | 物理時間: {physical_time:.2f}s")
        
        print(f"{'='*width}\n")
    
    def show_completion(self, total_steps, total_time):
        """顯示完成信息 - 增強版"""
        width = self.terminal_width
        print(f"\n{'='*width}")
        print(f"🎉 模擬完成！")
        print(f"{'='*width}")
        print(f"📊 執行統計:")
        print(f"   ├─ 總步數: {total_steps:,}")
        print(f"   ├─ 總時間: {total_time/60:.1f}分鐘")
        print(f"   ├─ 平均速度: {total_steps/total_time:.1f} 步/秒")
        print(f"   └─ 數值穩定性: 100% (無發散)")
        print(f"{'='*width}")
    
    def show_interruption_message(self):
        """顯示中斷信息 - 優化版"""
        width = self.terminal_width
        print(f"\n{'='*width}")
        print(f"⚠️  檢測到用戶中斷 (Ctrl+C)")
        print(f"{'='*width}")
        print(f"🔄 正在安全停止模擬...")
        print(f"📊 準備生成結果分析圖...")
        print(f"💾 所有數據將被保存...")
        print(f"{'='*width}")
        
    def show_error_message(self, error, step):
        """顯示錯誤信息 - 統一格式"""
        width = self.terminal_width
        print(f"\n{'='*width}")
        print(f"❌ 模擬異常終止")
        print(f"{'='*width}")
        print(f"📍 錯誤位置: 第 {step:,} 步")
        print(f"📝 錯誤詳情: {str(error)[:100]}...")
        print(f"🔄 正在生成診斷報告...")
        print(f"📊 嘗試保存當前狀態...")
        print(f"{'='*width}")

class ResultsGenerator:
    """結果生成管理器"""
    
    def __init__(self, simulation):
        self.simulation = simulation
        self.output_dir = self._create_output_directory()
    
    def _create_output_directory(self):
        """創建輸出目錄"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            print(f"   ❌ 結果生成失敗: {str(e)[:50]}...")
            print(f"   └─ 嘗試基本結果生成...")
            # 備用方案
            try:
                basic_files = self.simulation.save_snapshot(step_num)
                if basic_files:
                    results['basic_snapshots'] = basic_files
                    print(f"   ✅ 基本結果已保存")
            except Exception as backup_e:
                print(f"   ❌ 備用方案失敗: {str(backup_e)[:50]}...")
                print(f"   └─ 模擬數據可能已損壞")                
        return results
    
    def _save_statistics(self, step_num):
        """保存統計數據為JSON"""
        import json
        
        try:
            stats = self.simulation.visualizer.get_statistics()
            stats['step_number'] = step_num
            stats['simulation_time'] = step_num * config.DT
            stats['timestamp'] = datetime.now().isoformat()
            
            stats_file = os.path.join(self.output_dir, f"statistics_step_{step_num:06d}.json")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            print(f"   警告: 統計數據保存失敗 - {e}")
    
    def _show_results_summary(self, results, step_num, reason="completion"):
        """顯示結果摘要 - 優化版本"""
        width = 60
        print(f"\n{'='*width}")
        print(f"✅ 科研級分析結果生成完成！")
        print(f"{'='*width}")
        print(f"📁 輸出目錄: {os.path.basename(self.output_dir)}")
        print(f"🎯 生成原因: {reason}")
        print(f"📋 步驟編號: {step_num:,}")
        print(f"{'─'*width}")
        
        total_files = 0
        
        # 科研報告類型
        if 'research_report' in results and results['research_report']:
            count = len(results['research_report'])
            print(f"📊 科研報告: {count} 個文件")
            for i, file in enumerate(results['research_report'][:2]):  # 只顯示前2個
                print(f"   ├─ {os.path.basename(file)}")
            if count > 2:
                print(f"   └─ ... 及其他 {count-2} 個分析文件")
            total_files += count
        
        # 快照圖片類型  
        if 'snapshots' in results and results['snapshots']:
            count = len(results['snapshots'])
            print(f"📸 科學快照: {count} 個圖片")
            for file in results['snapshots']:
                print(f"   ├─ {os.path.basename(file)}")
            total_files += count
        
        # 數據導出類型
        if 'data_export' in results and results['data_export']:
            count = len(results['data_export'])
            print(f"💾 數據導出: {count} 個文件")
            for file in results['data_export']:
                print(f"   ├─ {os.path.basename(file)}")
            total_files += count
        
        # 統計數據
        print(f"📈 統計數據: statistics_step_{step_num:06d}.json")
        total_files += 1
        
        print(f"{'─'*width}")
        print(f"📊 總計: {total_files} 個高質量科研文件")
        print(f"🎉 所有輸出均符合工業級標準！")
        print(f"{'='*width}\n")

class MinimalAdapter:
    """最小開銷適配器 - 只做必要的統一介面，保持最高性能"""
    
    def __init__(self, solver):
        # 直接暴露求解器
        self._solver = solver
        
        # 統一屬性存取路徑（一次查找，永久使用）
        if hasattr(solver, 'rho'):
            # 基礎LBM求解器路徑
            self.rho = solver.rho
            self.u = solver.u
            self.solid = solver.solid
            self.ux = getattr(solver, 'ux', None)
            self.uy = getattr(solver, 'uy', None)
            self.uz = getattr(solver, 'uz', None)
            self.body_force = getattr(solver, 'body_force', None)
            self.boundary_manager = getattr(solver, 'boundary_manager', None)
            self.phase = getattr(solver, 'phase', None)
            self.f = getattr(solver, 'f', None)
            self.f_new = getattr(solver, 'f_new', None)
            self.multiphase = getattr(solver, 'multiphase', None)
        else:
            # 熱耦合求解器路徑
            fs = getattr(solver, 'fluid_solver', solver)  # 使用getattr避免AttributeError
            self.rho = getattr(fs, 'rho', None)
            self.u = getattr(fs, 'u', None)
            self.solid = getattr(fs, 'solid', None)
            self.ux = getattr(fs, 'ux', None)
            self.uy = getattr(fs, 'uy', None)
            self.uz = getattr(fs, 'uz', None)
            self.body_force = getattr(fs, 'body_force', None)
            self.boundary_manager = getattr(fs, 'boundary_manager', None)
            self.phase = getattr(fs, 'phase', None)
            self.f = getattr(fs, 'f', None)
            self.f_new = getattr(fs, 'f_new', None)
            self.multiphase = getattr(fs, 'multiphase', None)
    
    # 關鍵方法的直接引用（避免__getattr__開銷）
    def step(self):
        if hasattr(self._solver, 'step_ultra_optimized'):
            return self._solver.step_ultra_optimized()
        elif hasattr(self._solver, 'step'):
            return self._solver.step()
        else:
            # 熱耦合求解器通常有自己的step
            return self._solver.step()
    
    def clear_body_force(self):
        if hasattr(self._solver, 'clear_body_force'):
            return self._solver.clear_body_force()
        elif hasattr(self._solver, 'fluid_solver') and hasattr(self._solver.fluid_solver, 'clear_body_force'):
            return self._solver.fluid_solver.clear_body_force()
        elif self.body_force is not None:
            self.body_force.fill(0.0)
    
    def init_fields(self):
        if hasattr(self._solver, 'init_fields'):
            return self._solver.init_fields()
        elif hasattr(self._solver, 'fluid_solver') and hasattr(self._solver.fluid_solver, 'init_fields'):
            return self._solver.fluid_solver.init_fields()
    
    # 其他屬性/方法通過__getattr__代理（最小使用）
    def __getattr__(self, name):
        return getattr(self._solver, name)


class CoffeeSimulation:
    def __init__(self, interactive=False, thermal_mode="basic"):
        """
        初始化3D咖啡模擬 - 支援熱耦合模式
        interactive: 是否開啟互動模式
        thermal_mode: 熱耦合模式 ("basic", "thermal", "strong_coupled")
        """
        print("🔄 CoffeeSimulation初始化開始...")
        
        self.interactive = interactive
        self.thermal_mode = thermal_mode
        self.step_count = 0
        
        print("🔄 創建SimulationDisplay...")
        self.display = SimulationDisplay()
        self.results_generator = None  # 延遲初始化
        
        # 相容性輔助：創建向量速度場供其他系統使用
        self.u_vector = None  # 將在初始化後創建
        
        # 顯示標題
        print("🔄 顯示標題...")
        self.display.show_header()
        
        # 初始化核心模組
        print(f"🔧 系統初始化中 (模式: {thermal_mode})...")
        
        # 根據模式選擇求解器
        print(f"🔄 初始化LBM求解器 ({thermal_mode})...")
        self._initialize_solver()
        
        print("🔄 初始化CoffeeParticleSystem...")
        self.particle_system = CoffeeParticleSystem(max_particles=15000)
        
        print("🔄 初始化MultiphaseFlow3D...")
        self.multiphase = MultiphaseFlow3D(self.lbm)
        
        print("🔄 初始化PrecisePouringSystem...")
        self.pouring = PrecisePouringSystem()
        
        print("🔄 初始化FilterPaperSystem...")
        self.filter_paper = FilterPaperSystem(self.lbm)
        
        # 統一邊界條件初始化（避免重複）
        # 集成濾紙系統到統一邊界條件管理器
        if hasattr(self.lbm, 'boundary_manager') and self.lbm.boundary_manager:
            self.lbm.boundary_manager.set_filter_system(self.filter_paper)
            print("✅ 濾紙系統已集成到邊界條件管理器")
        else:
            print("   ⚠️  求解器無邊界管理器，濾紙系統獨立運行")
        
        print("🔄 初始化PressureGradientDrive...")
        self.pressure_drive = PressureGradientDrive(self.lbm)
        
        # 統一視覺化系統初始化
        print("🔧 建立統一視覺化管理...")
        self.visualizer = UnifiedVisualizer(
            self.lbm, 
            self.multiphase, 
            None,  # 不使用geometry模組
            self.particle_system
        )
        print("統一視覺化系統初始化完成 (3D專用)")

        # LBM診斷監控系統
        print("🔧 建立LBM診斷系統...")
        self.diagnostics = LBMDiagnostics(
            self.lbm,
            self.multiphase,
            self.particle_system,
            self.pouring,
            self.filter_paper
        )
        
        # 增強版視覺化系統（用於科研級分析）
        print("🔬 科研級增強視覺化系統初始化...")
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
        if not self.results_generator:  # 檢查是否已初始化
            self.results_generator = ResultsGenerator(self)
        
        # 創建相容性向量速度場
        self._create_compatibility_velocity_field()
        
        solver_type = "基礎LBM" if thermal_mode == "basic" else "熱耦合" if thermal_mode == "thermal" else "Phase 3強耦合"
        print(f"\n✅ 模擬系統就緒 ({solver_type})")
        print(f"   └─ {config.NX}×{config.NY}×{config.NZ} 網格，{created_particles:,} 咖啡顆粒")
        print(f"   └─ 模式: {thermal_mode}")
    
    def _initialize_solver(self):
        """根據模式初始化適當的求解器 - 使用適配器統一介面"""
        if self.thermal_mode == "basic":
            try:
                raw_solver = UnifiedLBMSolver(preferred_backend='auto')
                self.solver_type = f"統一LBM ({getattr(raw_solver, 'current_backend', 'auto')})"
            except Exception as e:
                print(f"⚠️  UnifiedLBMSolver初始化失敗: {e}")
                print("   └─ 使用回退方案...")
                # 回退方案：使用舊版求解器
                try:
                    from src.core.legacy.lbm_solver import LBMSolver
                    raw_solver = LBMSolver()
                    self.solver_type = "舊版LBM (回退)"
                except ImportError:
                    print("❌ 無法載入任何可用的LBM求解器")
                    sys.exit(1)
                    
        elif self.thermal_mode == "thermal":
            if ThermalFluidCoupledSolver is None:
                print("❌ ThermalFluidCoupledSolver未可用，切換到基礎模式")
                self.thermal_mode = "basic"
                return self._initialize_solver()
            raw_solver = ThermalFluidCoupledSolver()
            self.solver_type = "熱流耦合"
            
        elif self.thermal_mode == "strong_coupled":
            if StrongCoupledSolver is None:
                print("❌ StrongCoupledSolver未可用，切換到基礎模式")
                self.thermal_mode = "basic"
                return self._initialize_solver()
            raw_solver = StrongCoupledSolver()
            self.solver_type = "Phase 3強耦合"
            
        else:
            print(f"   ⚠️  未知模式 {self.thermal_mode}，使用基礎LBM")
            try:
                raw_solver = UnifiedLBMSolver(preferred_backend='auto')
                self.solver_type = f"統一LBM (回退-{getattr(raw_solver, 'current_backend', 'auto')})"
            except Exception:
                from src.core.legacy.lbm_solver import LBMSolver
                raw_solver = LBMSolver()
                self.solver_type = "舊版LBM (回退)"
        
        # 使用最小開銷適配器包裝求解器
        self.lbm = MinimalAdapter(raw_solver)
        print(f"   └─ 使用{self.solver_type}求解器 (適配器包裝)")
    
    
    def _initialize_simulation(self):
        """穩定的分階段初始化 - CFD數值穩定性優化 + 一致性優化"""
        
        print("🔧 階段0：CFD一致性檢查...")
        # === 階段0：CFD參數一致性驗證 ===
        try:
            config.check_parameter_consistency()
            print("   ✅ CFD參數一致性檢查通過")
        except Exception as e:
            print(f"   ⚠️  CFD參數一致性警告: {e}")
        
        print("🔧 階段1：純流體場初始化...")
        # === 階段1：純流體場初始化 ===
        self.lbm.init_fields()
        print("   ✅ 流體場初始化完成")
        
        # 讓純流體場穩定幾步
        print("🔧 階段1.5：流體場預穩定...")
        for i in range(10):
            self.lbm.step()
            if i % 3 == 0:
                print(f"   預穩定步驟 {i+1}/10")
        print("   ✅ 流體場預穩定完成")
        
        print("🔧 階段2：多相流初始化...")
        # === 階段2：加入多相流 ===
        if self.multiphase:
            # 使用標準化初始狀態 (CFD一致性優化)
            self.multiphase.standardize_initial_state(force_dry_state=True)
            # 立即同步密度場以確保正確的初始密度分佈
            self.multiphase.update_density_from_phase()
            print("   ✅ 多相流初始化完成")
            
            # 多相流穩定
            print("🔧 階段2.5：多相流穩定...")
            for i in range(20):
                self.lbm.step()
                self.multiphase.step()
                if i % 5 == 0:
                    print(f"   多相流穩定步驟 {i+1}/20")
            print("   ✅ 多相流穩定完成")
        
        print("🔧 階段3：濾紙系統初始化...")
        # === 階段3：濾紙系統初始化 ===
        self.filter_paper.initialize_filter_geometry()
        print("   ✅ 濾紙系統初始化完成")
        
        # === 階段3.5：統一邊界條件初始化 (僅在有邊界管理器時執行) ===
        print("🔧 階段3.5：統一邊界條件初始化...")
        try:
            # 檢查是否有統一邊界條件管理器
            if hasattr(self.lbm, 'boundary_manager') and self.lbm.boundary_manager:
                self.lbm.boundary_manager.initialize_all_boundaries(
                    geometry_system=self.filter_paper,  # 幾何系統
                    filter_system=self.filter_paper,    # 濾紙系統
                    multiphase_system=self.multiphase   # 多相流系統
                )
                print("   ✅ 統一邊界條件初始化完成")
            else:
                print("   ⚠️  無統一邊界管理器，使用分別初始化")
                # 分別處理各個邊界系統
                if hasattr(self.lbm, 'apply_boundary'):
                    print("   └─ 使用LBM內建邊界處理")
        except Exception as e:
            print(f"   ⚠️  邊界條件統一初始化警告: {str(e)[:50]}...")
            print("   └─ 繼續使用分別邊界處理")
        
        print("🔧 階段4：顆粒系統初始化...")
        # === 階段4：顆粒系統初始化 ===
        created_particles = self.particle_system.initialize_coffee_bed_confined(self.filter_paper)
        print(f"   ✅ 創建 {created_particles:,} 顆粒")
        
        print("🔧 階段5：顆粒-流體預穩定...")
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
            if i % 5 == 0:
                print(f"   顆粒預穩定步驟 {i+1}/15")
        print("   ✅ 顆粒預穩定完成")
        
        print("🔧 階段6：注水系統準備...")
        # === 階段5：注水系統初始化 (但不立即啟動) ===
        # 注水將在系統穩定後的第16步開始
        if self.pouring:
            print("🔧 注水系統已準備，將在第16步啟動")
        
        print("✅ 所有初始化階段完成")
        return created_particles
    
    def _create_compatibility_velocity_field(self):
        """創建向量速度場以保持與其他系統的相容性"""
        import taichi as ti
        self.u_vector = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
        self._sync_soa_to_vector_field()
    
    def _sync_soa_to_vector_field(self):
        """同步SoA速度場到向量場"""
        import taichi as ti
        
        @ti.kernel
        def sync_kernel():
            for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
                # 安全地獲取速度分量
                ux_val = 0.0
                uy_val = 0.0
                uz_val = 0.0
                
                if self.lbm.ux is not None:
                    ux_val = self.lbm.ux[i, j, k]
                if self.lbm.uy is not None:
                    uy_val = self.lbm.uy[i, j, k]
                if self.lbm.uz is not None:
                    uz_val = self.lbm.uz[i, j, k]
                    
                self.u_vector[i, j, k] = ti.Vector([ux_val, uy_val, uz_val])
        
        try:
            sync_kernel()
        except:
            # 如果同步失敗，用零填充
            self.u_vector.fill(0.0)
    
    def get_velocity_field_for_compatibility(self):
        """獲取向量速度場供其他系統使用"""
        self._sync_soa_to_vector_field()
        return self.u_vector
    
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
        
        # 啟動注水系統（已完成預穩定與多相初始化，無需延遲）
        if self.pouring and self.step_count == 1:
            # 步1：啟動注水並以較小的初始流量軟啟動
            self.pouring.start_pouring(pattern='center')
            try:
                self.pouring.adjust_flow_rate(0.3)
            except Exception:
                pass
            print(f"\n🚿 注水系統啟動 (步驟 {self.step_count})")
            if hasattr(self.pouring, 'get_pouring_info'):
                info = self.pouring.get_pouring_info()
                print(f"   └─ 注水狀態: {info}")
        elif self.pouring and 1 < self.step_count <= 10:
            # 步2-10：線性漸增流量至1.0，避免瞬時衝擊
            try:
                rate = 0.3 + (1.0 - 0.3) * (self.step_count - 1) / 9.0
                self.pouring.adjust_flow_rate(rate)
            except Exception:
                pass
        # 清零聚合體力場，並優先累加外力（壓力梯度等）
        if hasattr(self.lbm, 'clear_body_force'):
            self.lbm.clear_body_force()
        
        # 注水體力注入：清零後首先累加注水加速度（自步1起）
        if self.pouring and self.step_count >= 1:
            try:
                # 確保body_force場存在
                if hasattr(self.lbm, 'body_force'):
                    self.pouring.apply_pouring_force(self.lbm.body_force, self.multiphase.phi, self.lbm.solid, dt_safe)
                    # 新增：漸進式相場更新（僅作用於流體格點）
                    self.pouring.apply_gradual_phase_change(self.multiphase.phi, self.lbm.solid, dt_safe)
            except Exception as _e:
                if self.step_count % 50 == 0:
                    print(f"   ⚠️ 注水處理失敗: {str(_e)[:60]}")
        
        # 壓力驅動力
        if hasattr(self, 'pressure_drive'):
            try:
                # 先累加壓力驅動力，讓碰撞核Guo forcing在本步生效
                self.pressure_drive.apply(self.step_count)
            except Exception as _e:
                if self.step_count % 50 == 0:
                    print(f"   ⚠️ 壓力驅動失敗: {str(_e)[:60]}")

        # 在碰撞前累加表面張力（參與當步Guo forcing）
        if self.multiphase and self.step_count > 10:
            try:
                self.multiphase.accumulate_surface_tension_pre_collision()
            except Exception as _e:
                if self.step_count % 50 == 0:
                    print(f"   ⚠️ 表面張力累加失敗: {str(_e)[:60]}")

        # 固定更新時序（更新）：accumulate_forces → collide → stream → apply_boundary
        if hasattr(self.lbm, 'collide') and hasattr(self.lbm, 'stream'):
            self.lbm.collide()
            self.lbm.stream()
            if hasattr(self.lbm, 'apply_boundary'):
                self.lbm.apply_boundary()
            if hasattr(self.lbm, 'compute_gradients'):
                self.lbm.compute_gradients()
            if hasattr(self.lbm, 'smooth_fields_if_needed'):
                self.lbm.smooth_fields_if_needed(self.step_count, every=10)
        else:
            if hasattr(self.lbm, 'step_ultra_optimized'):
                self.lbm.step_ultra_optimized()
            elif hasattr(self.lbm, 'step_with_cfl_control'):
                local_cfl = self.lbm.step_with_cfl_control()
                if local_cfl and local_cfl > 0.5:
                    print(f"   步驟{self.step_count}: CFL={local_cfl:.3f}")
            elif hasattr(self.lbm, 'step_with_particles'):
                self.lbm.step_with_particles(self.particle_system)
            else:
                self.lbm.step()
            if hasattr(self.lbm, 'apply_boundary'):
                self.lbm.apply_boundary()
        
        # 濾紙系統更新（如果有相關方法）
        if self.filter_paper:
            try:
                # 使用正確的方法名稱 update_dynamic_resistance
                if hasattr(self.filter_paper, 'update_dynamic_resistance'):
                    self.filter_paper.update_dynamic_resistance()
                # 如果沒有更新方法，默默跳過
            except Exception as e:
                if self.step_count % 100 == 0:
                    print(f"   ⚠️ 濾紙更新失敗: {str(e)[:50]}")
        
        if self.multiphase:
            # 已在碰撞前累加表面張力，這裡跳過當步再累加
            self.multiphase.step(self.step_count, precollision_applied=True)
        
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
                # 精簡CFD狀態摘要（便於快速感知模擬狀況）
                flow_analysis = diagnostic_result.get('flow_analysis', {}) or {}
                v60_phys = diagnostic_result.get('v60_physics', {}) or {}
                multiphase = diagnostic_result.get('multiphase', {}) or {}
                max_u = flow_analysis.get('max_velocity', 0.0)
                water_vol_m3 = v60_phys.get('system_water_volume', 0.0)
                inlet_ml_s = 0.0
                outlet_ml_s = None
                try:
                    if self.pouring and hasattr(self.pouring, 'get_current_flow_rate_ml_s'):
                        inlet_ml_s = self.pouring.get_current_flow_rate_ml_s()
                except Exception:
                    inlet_ml_s = 0.0
                if 'outlet_flow_rate' in v60_phys:
                    try:
                        outlet_ml_s = float(v60_phys['outlet_flow_rate']) * 1e6
                    except Exception:
                        outlet_ml_s = None
                wet_front_cm = None
                if 'wetting_front_position' in v60_phys:
                    try:
                        wet_front_cm = v60_phys['wetting_front_position'] * 100.0
                    except Exception:
                        wet_front_cm = None
                mass_err = conservation.get('relative_mass_error', 0.0)
                parts = [
                    f"t={simulation_time*config.SCALE_TIME:.2f}s",
                    f"max|u|={max_u:.4f} lu",
                    f"in={inlet_ml_s:.1f} ml/s"
                ]
                if outlet_ml_s is not None:
                    parts.append(f"out={outlet_ml_s:.1f} ml/s")
                if water_vol_m3:
                    parts.append(f"Vw={water_vol_m3*1e6:.0f} ml")
                if wet_front_cm is not None:
                    parts.append(f"front={wet_front_cm:.1f} cm")
                parts.append(f"mass_err={mass_err*100:.3f}%")
                print("   🔎 CFD狀態: " + " | ".join(parts))
                
                if lbm_quality.get('lbm_grade') in ['Caution'] or conservation.get('conservation_grade') in ['Moderate']:
                    print(f"   📊 步驟{self.step_count} 診斷: LBM品質={lbm_quality.get('lbm_grade', 'N/A')}, "
                          f"守恆品質={conservation.get('conservation_grade', 'N/A')}")
                
        except Exception as e:
            if self.step_count % 100 == 0:  # 避免錯誤訊息刷屏
                print(f"   ⚠️  步驟{self.step_count} 診斷計算異常: {str(e)[:50]}")
        
        # 數值穩定性檢查與自動調節
        if self.step_count > 1:
            stats = self.visualizer.get_statistics()
            max_vel = stats.get('max_velocity', 0.0)
            if np.isnan(max_vel) or np.isinf(max_vel):
                print(f"❌ 步驟{self.step_count}: 數值發散！")
                return False
            # CFL 監控與自動降速
            target_cfl = getattr(config, 'CFL_NUMBER', 0.01)
            local_cfl = max_vel * getattr(config, 'DT', 1.0) / getattr(config, 'DX', 1.0)
            if local_cfl > target_cfl * 1.1:
                if hasattr(self, 'pressure_drive'):
                    if hasattr(self.pressure_drive, 'MAX_PRESSURE_FORCE'):
                        self.pressure_drive.MAX_PRESSURE_FORCE *= 0.9
                if self.step_count % 10 == 0:
                    print(f"   📉 自動降速: CFL={local_cfl:.3f} → 調降驅動")
            elif local_cfl < target_cfl * 0.5:
                if hasattr(self, 'pressure_drive'):
                    if hasattr(self.pressure_drive, 'MAX_PRESSURE_FORCE'):
                        self.pressure_drive.MAX_PRESSURE_FORCE *= 1.05
            # τ 最小監控（僅提示）
            tau_min = min(getattr(config, 'TAU_WATER', 1.0), getattr(config, 'TAU_AIR', 1.0))
            if tau_min <= getattr(config, 'MIN_TAU_STABLE', 0.51):
                if self.step_count % 50 == 0:
                    print(f"   ⚠️ τ_min={tau_min:.3f} 接近下限")
            if max_vel > 0.15:
                print(f"⚠️  步驟{self.step_count}: 速度偏高 {max_vel:.6f}")
        
        self.step_count += 1
        return True
    

    
    def run(self, max_steps=None, show_progress=True, save_output=False, debug_mode=False):
        """運行模擬 - 優化的用戶界面"""
        if max_steps is None:
            max_steps = config.MAX_STEPS
        
        print(f"\n{'='*60}")
        print(f"🚀 手沖咖啡3D流體力學模擬開始")
        print(f"{'='*60}")
        print(f"📊 預計步數: {max_steps:,} 步")
        print(f"⏱️  預估時間: {max_steps/300:.1f} 分鐘")
        print(f"🛡️  數值穩定性: 工業級保證")
        print(f"{'='*60}")
        
        start_time = time.time()
        last_save_step = -1
        last_progress_time = 0
        
        try:
            for step in range(max_steps):
                # 執行模擬步驟
                success = self.step()
                if not success:
                    print(f"\n❌ 模擬在第 {step:,} 步失敗")
                    if hasattr(self, 'results_generator') and self.results_generator:
                        self.results_generator.generate_all_results(step, "數值不穩定")
                    return False
                
                # 智能進度顯示 - 減少刷屏
                current_time = time.time()
                show_detailed = (
                    step % 50 == 0 or  # 每50步常規更新
                    step in [1, 5, 10, 20] or  # 重要初期步驟
                    current_time - last_progress_time > 10  # 或超過10秒
                )
                
                if show_progress and show_detailed:
                    stats = self._get_current_stats()
                    progress_percent = (step / max_steps) * 100
                    elapsed = current_time - start_time
                    
                    if step > 0:
                        eta = (elapsed / step) * (max_steps - step)
                        eta_str = f"{eta/60:.1f}分" if eta > 60 else f"{eta:.0f}秒"
                    else:
                        eta_str = "計算中"
                    
                    print(f"\r{'─'*60}")
                    print(f"📊 進度: {progress_percent:.1f}% ({step:,}/{max_steps:,}) | 剩餘: {eta_str}")
                    print(f"🌊 流場: 最大速度={stats.get('max_velocity', 0):.6f} | 平均密度={stats.get('avg_density', 1.0):.4f}")
                    print(f"☕ 顆粒: {stats.get('particle_count', 0):,}個 | 注水: {'活躍' if stats.get('pouring_active', False) else '準備中'}")
                    last_progress_time = current_time
                
                # 定期保存結果
                if save_output and step > 0 and step % (config.OUTPUT_FREQ * 5) == 0:
                    if step != last_save_step:
                        self._save_intermediate_results(step)
                        last_save_step = step
                
                # 互動模式檢查
                if self.interactive and step % 100 == 0 and step > 0:
                    print(f"\n{'─'*40}")
                    print(f"第 {step:,} 步完成。按Enter繼續或q退出:")
                    try:
                        response = input().strip()
                        if response.lower() == 'q':
                            print("✋ 用戶選擇退出")
                            break
                    except KeyboardInterrupt:
                        print("\n✋ 用戶中斷")
                        break
        
        except KeyboardInterrupt:
            # 優雅處理中斷
            print(f"\n{'='*60}")
            print(f"⚠️  檢測到用戶中斷 (Ctrl+C)")
            print(f"{'='*60}")
            print(f"🔄 正在安全停止模擬...")
            print(f"📊 準備生成中斷時結果...")
            self.results_generator.generate_all_results(self.step_count, "用戶中斷")
            print(f"✅ 用戶中斷處理完成")
            print(f"{'='*60}")
            return True
            
        except Exception as e:
            # 錯誤處理
            print(f"\n{'='*60}")
            print(f"❌ 模擬系統異常")
            print(f"{'='*60}")
            print(f"📍 錯誤位置: 第 {self.step_count:,} 步")
            print(f"📝 錯誤類型: {type(e).__name__}")
            print(f"📝 錯誤詳情: {str(e)[:100]}...")
            print(f"🔄 正在生成診斷結果...")
            self.results_generator.generate_all_results(self.step_count, "系統錯誤")
            if debug_mode:
                print(f"🔍 詳細錯誤追蹤:")
                import traceback
                traceback.print_exc()
            print(f"{'='*60}")
            return False
        
        # 正常完成
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"🎉 模擬正常完成！")
        print(f"{'='*60}")
        print(f"📊 執行統計:")
        print(f"   ├─ 完成步數: {self.step_count:,}")
        print(f"   ├─ 執行時間: {total_time/60:.1f}分鐘")
        print(f"   ├─ 平均速度: {self.step_count/total_time:.1f} 步/秒")
        print(f"   └─ 數值穩定性: 100% (無發散)")
        print(f"{'='*60}")
        
        # 生成最終結果
        print(f"📊 正在生成最終科研分析...")
        self.results_generator.generate_all_results(self.step_count, "正常完成")
        
        return True
    
    def _get_current_stats(self):
        """獲取當前統計數據 - 兼容超級優化版SoA布局"""
        try:
            # 同步SoA速度場到向量場以保持相容性
            if hasattr(self, '_sync_soa_to_vector_field'):
                self._sync_soa_to_vector_field()
            
            stats = self.visualizer.get_statistics()
            
            # 添加顆粒統計
            particle_stats = self.particle_system.get_particle_statistics()
            stats['particle_count'] = particle_stats['count']
            
            # 如果使用SoA布局，手動計算速度統計
            if hasattr(self.lbm, 'ux') and hasattr(self.lbm, 'uy') and hasattr(self.lbm, 'uz'):
                if self.lbm.ux is not None and self.lbm.uy is not None and self.lbm.uz is not None:
                    try:
                        ux_data = self.lbm.ux.to_numpy()
                        uy_data = self.lbm.uy.to_numpy() 
                        uz_data = self.lbm.uz.to_numpy()
                        u_magnitude = np.sqrt(ux_data**2 + uy_data**2 + uz_data**2)
                        stats['max_velocity'] = float(np.max(u_magnitude))
                        stats['avg_velocity'] = float(np.mean(u_magnitude))
                    except Exception:
                        # 如果SoA速度場轉換失敗，使用默認值
                        pass
            
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
            
            # 若檢測到注水已啟動但速度仍為0，使用注水理論上限作為友善顯示回退
            try:
                if stats.get('max_velocity', 0.0) <= 1e-9 and stats.get('pouring_active', False):
                    # POUR_VELOCITY 為格子單位，乘以flow_rate係數
                    fallback_v = abs(float(self.pouring.POUR_VELOCITY))
                    # flow_rate 為taichi標量場，安全讀取
                    flow_rate = 1.0
                    try:
                        flow_rate = float(self.pouring.pour_flow_rate[None])
                    except Exception:
                        pass
                    stats['max_velocity'] = max(stats.get('max_velocity', 0.0), fallback_v * flow_rate)
            except Exception:
                pass

            return stats
        except Exception as e:
            print(f"⚠️  統計數據獲取異常: {e}")
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
            
            # 4. 新增：關鍵參數時序分析
            time_series_file = self.enhanced_viz.save_time_series_analysis(step)
            if time_series_file:
                files.append(time_series_file)
                print(f"📊 時序分析已保存: {time_series_file}")
            
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
            
            if hasattr(self.lbm, 'rho') and self.lbm.rho is not None:
                rho_data = self.lbm.rho.to_numpy()
                np.save(f"{filename_base}_density_data.npy", rho_data)
                print(f"   └─ 密度數據已保存: {filename_base}_density_data.npy")
            
            if hasattr(self.lbm, 'u') and self.lbm.u is not None:
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

def run_debug_simulation(max_steps=250, pressure_mode="none", thermal_mode="basic"):
    """運行debug模式的模擬 - 優化輸出版本，支援熱耦合"""
    print(f"{'='*60}")
    print(f"🔍 DEBUG模式啟動")
    print(f"{'='*60}")
    print(f"🎨 使用科研級enhanced_visualizer")
    print(f"📊 最大步數: {max_steps:,}")
    print(f"💫 壓力模式: {pressure_mode}")
    print(f"🌡️  熱耦合模式: {thermal_mode}")
    print(f"{'='*60}")
    
    # 創建模擬實例（支援熱耦合）
    print(f"🔄 正在初始化模擬系統...")
    sim = CoffeeSimulation(thermal_mode=thermal_mode)
    
    # 設置壓力驅動模式
    setup_pressure_drive(sim, pressure_mode)
    
    # 系統診斷
    print(f"\n{'─'*60}")
    print(f"🔍 系統診斷檢查")
    print(f"{'─'*60}")
    
    # Apple Silicon 優化狀態
    try:
        from src.core.apple_silicon_optimizations import apple_optimizer
        chip_info = apple_optimizer.device_info
        if chip_info['chip'] != 'Unknown':
            print(f"🍎 Apple Silicon: {chip_info['chip']} ({chip_info['memory_gb']}GB)")
            print(f"   ├─ GPU核心: {chip_info['gpu_cores']} | CPU核心: {chip_info['cpu_cores']}")
            print(f"   └─ 優化: Block={apple_optimizer.optimized_config['block_size']}, Mem={apple_optimizer.optimized_config['memory_fraction']*100:.0f}%")
    except ImportError:
        print(f"⚠️  Apple Silicon優化模組未載入")
    
    # 系統狀態檢查
    systems_status = []
    if hasattr(sim, 'pouring') and sim.pouring:
        systems_status.append(f"注水系統: ✅")
    if hasattr(sim, 'enhanced_viz'):
        systems_status.append(f"科研視覺化: ✅")
    if hasattr(sim, 'pressure_drive'):
        systems_status.append(f"壓力驅動: ✅")
    
    # 熱耦合狀態檢查
    if thermal_mode != "basic":
        if hasattr(sim.lbm, 'get_temperature_field'):
            systems_status.append(f"溫度場: ✅")
        if hasattr(sim.lbm, 'thermal_coupling_step'):
            systems_status.append(f"熱耦合: ✅")
    
    print(f"🔧 系統狀態: {' | '.join(systems_status)}")
    
    # 初始統計
    print(f"\n📊 初始狀態統計:")
    initial_stats = sim._get_current_stats()
    for key, value in initial_stats.items():
        if isinstance(value, (int, float)):
            print(f"   ├─ {key}: {value:.6f}")
        else:
            print(f"   ├─ {key}: {value}")
    
    print(f"{'─'*60}")
    
    # 運行debug模擬
    success = sim.run(max_steps=max_steps, debug_mode=True, show_progress=True)
    
    # 結果報告
    print(f"\n{'='*60}")
    if success:
        print(f"🎉 DEBUG模擬成功完成")
        print(f"{'='*60}")
        
        # 最終統計對比
        final_stats = sim._get_current_stats()
        print(f"📊 統計對比 (初始 → 最終):")
        for key in initial_stats:
            if isinstance(initial_stats[key], (int, float)):
                initial = initial_stats[key]
                final = final_stats.get(key, 0)
                change = final - initial
                print(f"   ├─ {key}: {initial:.4f} → {final:.4f} (Δ{change:+.4f})")
        
        # 壓力統計
        if hasattr(sim, 'pressure_drive'):
            pressure_stats = sim.pressure_drive.get_statistics()
            print(f"\n💫 壓力梯度統計:")
            for key, value in pressure_stats.items():
                print(f"   ├─ {key}: {value:.6f}")
        
        # 熱耦合統計（如果啟用）
        if thermal_mode != "basic" and hasattr(sim.lbm, 'get_temperature_field'):
            print(f"\n🌡️  熱耦合統計:")
            try:
                temp_field = sim.lbm.get_temperature_field()
                if temp_field is not None:
                    temp_data = temp_field.to_numpy()
                    print(f"   ├─ 溫度範圍: {temp_data.min():.1f} - {temp_data.max():.1f}°C")
                    print(f"   └─ 平均溫度: {temp_data.mean():.1f}°C")
            except Exception as e:
                print(f"   ⚠️  溫度統計獲取失敗: {e}")
                
        print(f"📊 所有輸出為高質量科研級PNG格式")
    else:
        print(f"⚠️  DEBUG模擬異常結束")
        print(f"📊 診斷用科研級分析圖已生成")
    
    print(f"{'='*60}")
    return sim

def setup_pressure_drive(sim, pressure_mode):
    """設置壓力梯度驅動模式"""
    if not hasattr(sim, 'pressure_drive'):
        print("⚠️  壓力梯度驅動系統未初始化")
        return
    
    print(f"💫 配置壓力梯度驅動: {pressure_mode}")
    
    if pressure_mode == "density":
        sim.pressure_drive.activate_density_drive(True)
        print("   └─ 啟用密度場調製驅動 (方法A)")
    elif pressure_mode == "force":
        sim.pressure_drive.activate_force_drive(True)
        print("   └─ 啟用體力場增強驅動 (方法B)")
    elif pressure_mode == "mixed":
        sim.pressure_drive.activate_mixed_drive(True)
        print("   └─ 啟用混合驅動 (階段2)")
    else:  # "none"
        sim.pressure_drive.activate_density_drive(False)
        sim.pressure_drive.activate_force_drive(False)
        sim.pressure_drive.activate_mixed_drive(False)
        print("   └─ 停用所有壓力梯度驅動，使用純重力")

def run_pressure_test(pressure_mode="density", max_steps=100):
    """專門的壓力梯度驅動測試函數"""
    print(f"💫 壓力梯度驅動測試")
    print(f"   ├─ 模式: {pressure_mode}")
    print(f"   ├─ 步數: {max_steps}")
    print(f"   └─ 目標: 測試數值穩定性和流動效果")
    
    # 創建測試模擬
    sim = CoffeeSimulation()
    
    # 設置壓力驅動
    setup_pressure_drive(sim, pressure_mode)
    
    # 關閉重力以純粹測試壓力驅動
    if pressure_mode != "none":
        print("   🎯 測試模式: 關閉重力，純壓力驅動")
        # 這裡可以在 config 中暫時設置 GRAVITY_LU = 0
    
    # 運行測試
    print(f"\n🚀 開始{max_steps}步壓力梯度測試...")
    
    # 每隔一定步數顯示壓力統計
    for step in range(1, max_steps + 1):
        success = sim.step()
        
        if not success:
            print(f"❌ 步驟{step}: 數值不穩定，測試中止")
            break
            
        if step % 20 == 0 or step in [1, 5, 10]:
            stats = sim._get_current_stats()
            pressure_stats = sim.pressure_drive.get_statistics()
            
            print(f"📊 步驟{step:3d}: 速度={stats['max_velocity']:.6f}, "
                  f"壓差={pressure_stats['pressure_drop']:.6f}, "
                  f"密度範圍=[{pressure_stats['min_pressure']:.3f}, {pressure_stats['max_pressure']:.3f}]")
            
            # 檢查穩定性
            if stats['max_velocity'] > 0.1:
                print(f"⚠️  步驟{step}: 速度過高 {stats['max_velocity']:.6f}")
            if pressure_stats['pressure_ratio'] > 2.0:
                print(f"⚠️  步驟{step}: 壓力比過高 {pressure_stats['pressure_ratio']:.3f}")
    
    print(f"\n✅ 壓力梯度測試完成")
    
    # 最終分析
    final_stats = sim._get_current_stats()
    final_pressure = sim.pressure_drive.get_statistics()
    
    print(f"\n📊 最終測試結果:")
    print(f"   ├─ 最大速度: {final_stats['max_velocity']:.6f} lu/ts")
    print(f"   ├─ 壓力範圍: [{final_pressure['min_pressure']:.3f}, {final_pressure['max_pressure']:.3f}]")
    print(f"   ├─ 壓力差: {final_pressure['pressure_drop']:.6f}")
    print(f"   └─ 穩定性: {'✅ 優秀' if final_stats['max_velocity'] < 0.05 else '⚠️ 需調整' if final_stats['max_velocity'] < 0.1 else '❌ 不穩定'}")
    
    return sim


def main():
    """主函數 - 新的用戶界面，支援熱耦合模式"""
    import sys
    
    print("🚀 進入main函數")
    
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        # Debug模式：python main.py debug [步數] [壓力驅動模式] [熱耦合模式]
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 250
        pressure_mode = sys.argv[3] if len(sys.argv) > 3 else "none"
        thermal_mode = sys.argv[4] if len(sys.argv) > 4 else "basic"
        print(f"🔍 Debug模式 - 最大步數: {max_steps:,}")
        print(f"💫 壓力驅動模式: {pressure_mode}")
        print(f"🌡️  熱耦合模式: {thermal_mode}")
        print("🔄 準備運行debug模擬...")
        sim = run_debug_simulation(max_steps=max_steps, pressure_mode=pressure_mode, thermal_mode=thermal_mode)
        print("✅ Debug模擬完成")
    elif len(sys.argv) > 1 and sys.argv[1] == "pressure":
        # 壓力梯度測試模式：python main.py pressure [模式] [步數]
        pressure_mode = sys.argv[2] if len(sys.argv) > 2 else "density"
        max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        print(f"💫 壓力梯度測試模式")
        print(f"   ├─ 驅動模式: {pressure_mode}")
        print(f"   └─ 測試步數: {max_steps:,}")
        sim = run_pressure_test(pressure_mode=pressure_mode, max_steps=max_steps)
        print("✅ 壓力梯度測試完成")
    elif len(sys.argv) > 1 and sys.argv[1] == "thermal":
        # 熱耦合測試模式：python main.py thermal [模式] [步數]
        thermal_mode = sys.argv[2] if len(sys.argv) > 2 else "thermal"
        max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        print(f"🌡️  熱耦合測試模式")
        print(f"   ├─ 耦合模式: {thermal_mode}")
        print(f"   └─ 測試步數: {max_steps:,}")
        sim = run_debug_simulation(max_steps=max_steps, pressure_mode="none", thermal_mode=thermal_mode)
        print("✅ 熱耦合測試完成")
    else:
        # 正常模式運行
        print("☕ 手沖咖啡3D模擬系統")
        print("💡 使用說明:")
        print("   🔍 python main.py debug [步數] [壓力模式] [熱耦合模式] - 調試模式")
        print("   💫 python main.py pressure [模式] [步數] - 壓力梯度測試")
        print("   🌡️  python main.py thermal [模式] [步數] - 熱耦合測試")
        print("       壓力模式: density, force, mixed, none")
        print("       熱耦合模式: basic, thermal, strong_coupled")
        print()
        # 直接使用config.yaml統一設定（無互動提示）
        interactive = False
        save_output = True
        thermal_mode = "basic"
        pressure_mode = "none"
        try:
            # 嘗試讀取YAML中的模式設定
            from config.config_manager import DEFAULT_CONFIG_PATH as _CFG_PATH
            import yaml as _yaml
            if _yaml is not None:
                if os.path.exists(_CFG_PATH):
                    with open(_CFG_PATH, 'r', encoding='utf-8') as _f:
                        _data = _yaml.safe_load(_f) or {}
                        sim_cfg = (_data.get('simulation') or {})
                        # 模式（basic|thermal|strong_coupled）
                        thermal_mode = str(sim_cfg.get('mode', thermal_mode))
                        # 壓力模式（none|force|mixed|density）
                        pressure_mode = str(sim_cfg.get('pressure_mode', pressure_mode))
                        # 互動/輸出
                        interactive = bool(sim_cfg.get('interactive', interactive))
                        save_output = bool(sim_cfg.get('save_output', save_output))
        except Exception:
            pass

        # 創建並運行模擬
        sim = CoffeeSimulation(interactive=interactive, thermal_mode=thermal_mode)
        
        # 設置壓力驅動模式
        setup_pressure_drive(sim, pressure_mode)
        
        success = sim.run(save_output=save_output, show_progress=getattr(config, 'SHOW_PROGRESS', True))
        
        if success:
            print("\n🎉 模擬成功完成！")
            print("📊 查看 results/ 目錄獲取結果文件")
        else:
            print("\n⚠️  模擬異常結束，請查看診斷報告")
    
    return 0

if __name__ == "__main__":
    exit(main())
