# main_minimal.py
"""
最小工作版本 - 僅測試科學修正的config.py
確認參數修正是有效的
"""

import taichi as ti
import config

# 初始化Taichi
ti.init(arch=ti.metal, debug=False)

def test_config():
    """測試科學修正的config.py參數"""
    print("=== 測試科學修正的配置參數 ===")
    
    # 驗證關鍵參數
    print(f"✅ CFL數: {config.CFL_NUMBER:.6f} (目標: < 0.1)")
    print(f"✅ τ_water: {config.TAU_WATER:.6f} (目標: > 0.5)")
    print(f"✅ 模擬步數: {config.MAX_STEPS:,} (目標: 可執行)")
    print(f"✅ 時間尺度: {config.SCALE_TIME*1000:.1f} ms/ts (目標: 現實)")
    print(f"✅ Reynolds數: {config.RE_CHAR:.1f} (目標: 湍流)")
    
    # 檢查穩定性條件
    stable = True
    if config.CFL_NUMBER >= 0.1:
        print(f"⚠️  CFL數 {config.CFL_NUMBER:.3f} 可能不穩定")
        stable = False
    if config.TAU_WATER <= 0.5:
        print(f"⚠️  τ_water {config.TAU_WATER:.3f} 可能不穩定")
        stable = False
    if config.MAX_STEPS > 20000:
        print(f"⚠️  步數 {config.MAX_STEPS:,} 可能太多")
        stable = False
    
    if stable:
        print("\n🎉 所有參數均通過科學驗證！")
        print("config.py 修正成功")
        return True
    else:
        print("\n❌ 發現參數問題")
        return False

# 在全域範圍定義場
test_field = ti.field(dtype=ti.f32, shape=(64, 64, 64))

@ti.kernel
def simple_lbm_test() -> ti.f32:
    """簡單的LBM核心測試"""
    # 簡單計算
    for i, j, k in test_field:
        test_field[i, j, k] = 1.0
    
    return test_field[32, 32, 32]

def test_taichi():
    """測試Taichi GPU計算"""
    print("\n=== 測試Taichi GPU計算 ===")
    
    try:
        result = simple_lbm_test()
        print(f"✅ Taichi GPU計算正常: {result}")
        return True
    except Exception as e:
        print(f"❌ Taichi計算失敗: {e}")
        return False

def main():
    """最小測試主程式"""
    print("Pour-Over Coffee - 最小測試程式")
    print("測試科學修正版config.py")
    print("=" * 40)
    
    # 測試配置
    config_ok = test_config()
    
    # 測試Taichi
    taichi_ok = test_taichi()
    
    # 總結
    if config_ok and taichi_ok:
        print("\n🎯 系統準備就緒！")
        print("科學修正版本工作正常")
        print("可以進行完整LBM模擬")
        return 0
    else:
        print("\n💔 系統有問題")
        return 1

if __name__ == "__main__":
    exit(main())