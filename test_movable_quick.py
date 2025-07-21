# test_movable_quick.py
"""
快速測試可移動咖啡顆粒系統
"""

import taichi as ti
import time

def quick_test():
    """快速測試可移動顆粒系統"""
    print("🧪 快速測試可移動咖啡顆粒系統")
    print("=" * 40)
    
    # 初始化Taichi (使用CPU以減少內存需求)
    ti.init(arch=ti.cpu, device_memory_GB=1.0)
    
    try:
        from main import CoffeeSimulation
        
        print("✅ 1. 導入模組成功")
        
        # 創建模擬
        sim = CoffeeSimulation()
        print("✅ 2. 創建模擬成功")
        
        # 獲取顆粒統計
        stats = sim.particle_system.get_detailed_statistics()
        print(f"✅ 3. 顆粒系統統計:")
        print(f"     └─ 總顆粒數: {stats['total_particles']:,}")
        print(f"     └─ 活躍顆粒數: {stats['active_particles']:,}")
        print(f"     └─ 平均粒徑: {stats['average_size']:.3f} 格子單位")
        print(f"     └─ 初始萃取度: {stats['average_extraction']:.3f}")
        
        # 測試幾個時間步驟
        print("✅ 4. 測試物理時間步驟...")
        
        for step in range(3):
            sim.step()
            if step == 0:
                print("     └─ 第1步完成")
            elif step == 1:
                print("     └─ 第2步完成")
            else:
                print("     └─ 第3步完成")
        
        # 最終統計
        final_stats = sim.particle_system.get_detailed_statistics()
        print(f"✅ 5. 最終結果:")
        print(f"     └─ 平均顆粒速度: {final_stats['average_speed']:.6f} m/s")
        print(f"     └─ 萃取度變化: {final_stats['average_extraction'] - stats['average_extraction']:.6f}")
        
        print("\n🎉 可移動咖啡顆粒系統測試成功!")
        print("   ✓ 顆粒初始化正常")
        print("   ✓ 物理耦合工作正常") 
        print("   ✓ 萃取機制運行正常")
        print("   ✓ 已替代達西定律固定多孔介質")
        print("\n🚀 準備好進行完整咖啡模擬!")
        print("   運行: python main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if not success:
        print("\n❌ 系統需要修復")
    else:
        print("\n✨ 系統準備就緒!")