#!/usr/bin/env python3
"""
更新項目重構後的import路徑
作為CFD工程師，確保所有模組導入路徑正確更新
"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path, import_mapping):
    """更新單個文件中的import路徑"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 更新import語句
        for old_import, new_import in import_mapping.items():
            # 處理 "from module import ..."
            pattern1 = rf"from {re.escape(old_import)} import"
            replacement1 = f"from {new_import} import"
            content = re.sub(pattern1, replacement1, content)
            
            # 處理 "import module"
            pattern2 = rf"^import {re.escape(old_import)}$"
            replacement2 = f"import {new_import}"
            content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE)
            
            # 處理 "import module as alias"
            pattern3 = rf"import {re.escape(old_import)} as"
            replacement3 = f"import {new_import} as"
            content = re.sub(pattern3, replacement3, content)
        
        # 如果內容有變化，寫回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"錯誤處理文件 {file_path}: {e}")
        return False

def main():
    """主要更新邏輯"""
    
    # 定義import映射關係
    import_mapping = {
        # 核心模組（短名 → 統一路徑）
        'lbm_solver': 'src.core.lbm_solver',
        'lbm_protocol': 'src.core.lbm_protocol', 
        'multiphase_3d': 'src.core.multiphase_3d',
        'numerical_stability': 'src.core.numerical_stability',
        'memory_optimizer': 'src.core.memory_optimizer',
        'apple_silicon_optimizations': 'src.core.apple_silicon_optimizations',
        'strong_coupled_solver': 'src.core.strong_coupled_solver',
        'ultimate_cfd_system': 'src.core.ultimate_cfd_system',
        'ultra_optimized_lbm': 'src.core.ultra_optimized_lbm',
        'thermal_fluid_coupled': 'src.core.thermal_fluid_coupled',
        
        # 核心模組（完整 legacy 路徑 → 統一路徑）
        'src.core.legacy.lbm_solver': 'src.core.lbm_solver',
        'src.core.legacy.ultra_optimized_lbm': 'src.core.ultra_optimized_lbm',
        'src.core.legacy.cuda_dual_gpu_lbm': 'src.core.cuda_dual_gpu_lbm',
        
        # 物理模組
        'temperature_dependent_properties': 'src.physics.temperature_dependent_properties',
        'buoyancy_natural_convection': 'src.physics.buoyancy_natural_convection',
        'les_turbulence': 'src.physics.les_turbulence',
        'thermal_lbm': 'src.physics.thermal_lbm',
        'boundary_conditions': 'src.physics.boundary_conditions',
        'filter_paper': 'src.physics.filter_paper',
        'coffee_particles': 'src.physics.coffee_particles',
        'pressure_gradient_drive': 'src.physics.pressure_gradient_drive',
        'precise_pouring': 'src.physics.precise_pouring',
        'thermal_properties': 'src.physics.thermal_properties',
        
        # 視覺化模組
        'enhanced_visualizer': 'src.visualization.enhanced_visualizer',
        'visualizer': 'src.visualization.visualizer',
        'geometry_visualizer': 'src.visualization.geometry_visualizer',
        'lbm_diagnostics': 'src.visualization.lbm_diagnostics',
        
        # 工具模組
        'error_handling': 'src.utils.error_handling',
        'config_validator': 'src.utils.config_validator',
        'physics_plugin_system': 'src.utils.physics_plugin_system',
        'data_structure_analysis': 'src.utils.data_structure_analysis',
        
        # 配置模組
        'config': 'config.config',
        'thermal_config': 'config.thermal_config',
        'init': 'config.init',
        # 統一配置入口：將相容層導入改為統一入口
        'config.config': 'config',
    }
    
    # 需要更新的文件列表
    files_to_update = []
    
    # 主目錄的Python文件
    main_dir_files = ['main.py', 'working_main.py', 'jax_hybrid_core.py']
    for file in main_dir_files:
        if os.path.exists(file):
            files_to_update.append(file)
    
    # src目錄下的所有Python文件
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                files_to_update.append(os.path.join(root, file))
    
    # tests目錄下的所有Python文件
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                files_to_update.append(os.path.join(root, file))
    
    # examples和benchmarks目錄
    for directory in ['examples', 'benchmarks']:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        files_to_update.append(os.path.join(root, file))
    
    # 執行更新
    updated_count = 0
    for file_path in files_to_update:
        if update_imports_in_file(file_path, import_mapping):
            print(f"✅ 更新: {file_path}")
            updated_count += 1
        else:
            print(f"⏸️  無變化: {file_path}")
    
    print(f"\n🎯 更新完成: {updated_count}/{len(files_to_update)} 個文件已更新")

if __name__ == "__main__":
    main()