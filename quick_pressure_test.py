#!/usr/bin/env python3
"""
快速壓力驅動測試 - 驗證修正是否生效
"""

import sys
import os
sys.path.append('.')

# 確保可以導入所需模組
try:
    from main import CoffeeSimulation, setup_pressure_drive
    print("✅ 成功導入模組")
except ImportError as e:
    print(f"❌ 模組導入失敗: {e}")
    sys.exit(1)

def test_pressure_setup():
    """測試壓力驅動設置"""
    print("🧪 開始壓力驅動快速測試...")
    
    # 創建模擬實例
    print("🔄 創建模擬實例...")
    try:
        sim = CoffeeSimulation(interactive=False)
        print("✅ 模擬實例創建成功")
    except Exception as e:
        print(f"❌ 模擬實例創建失敗: {e}")
        return False
    
    # 檢查壓力驅動是否存在
    if hasattr(sim, 'pressure_drive'):
        print("✅ 壓力驅動系統已初始化")
    else:
        print("❌ 壓力驅動系統未初始化")
        return False
    
    # 測試不同模式
    test_modes = ['none', 'density', 'force', 'mixed']
    
    for mode in test_modes:
        print(f"\n📊 測試模式: {mode}")
        setup_pressure_drive(sim, mode)
        
        # 檢查狀態
        status = sim.pressure_drive.get_status()
        print(f"   狀態: {status}")
    
    print("\n🎯 測試完成")
    return True

if __name__ == "__main__":
    test_pressure_setup()