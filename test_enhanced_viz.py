#!/usr/bin/env python3
"""
測試Enhanced Visualizer的報告輸出功能
"""

import os
import numpy as np
from datetime import datetime

def test_report_directory():
    """測試報告目錄創建"""
    print("🧪 測試報告目錄創建...")
    
    # 創建時間戳
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建報告目錄
    report_dir = f"report/{session_timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # 創建子目錄
    subdirs = ['images', 'data', 'analysis']
    for subdir in subdirs:
        os.makedirs(f"{report_dir}/{subdir}", exist_ok=True)
    
    print(f"✅ 報告目錄已創建: {report_dir}")
    
    # 測試圖片保存
    import matplotlib.pyplot as plt
    
    # 創建測試圖
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    ax.set_title('Test CFD Analysis Report')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存測試圖片
    test_image_path = f"{report_dir}/images/test_cfd_analysis.png"
    plt.savefig(test_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 測試圖片已保存: {test_image_path}")
    
    # 測試數據保存
    test_data = {
        'timestamp': session_timestamp,
        'test_parameters': {
            'reynolds': 5396.8,
            'froude': 0.022,
            'cfl': 0.010
        },
        'analysis_results': {
            'pressure_drop': 1250.5,
            'max_velocity': 0.05,
            'flow_rate': 4.0
        }
    }
    
    import json
    data_file = f"{report_dir}/data/test_simulation_data.json"
    with open(data_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"📄 測試數據已保存: {data_file}")
    
    # 列出創建的文件
    print("\n📁 創建的目錄結構:")
    for root, dirs, files in os.walk(report_dir):
        level = root.replace(report_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    return report_dir

if __name__ == "__main__":
    print("="*50)
    print("🔬 Enhanced Visualizer 報告輸出測試")
    print("="*50)
    
    try:
        report_dir = test_report_directory()
        print(f"\n✅ 測試成功完成!")
        print(f"📂 報告目錄: {report_dir}")
        print("\n💡 您可以檢查以上目錄中的輸出文件")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()