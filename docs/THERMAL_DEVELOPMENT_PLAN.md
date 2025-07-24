# 🌡️ 熱流-流體耦合系統開發計劃
*Pour-Over CFD 專案熱傳模擬擴展*

## 🎯 專案概述

### 目標
將現有的D3Q19 LBM流體動力學系統擴展為**流體-熱傳耦合**的多物理場CFD模擬，專門針對V60手沖咖啡的真實熱傳現象。

### 核心價值
- 🔬 **科學準確性**: 基於真實物理的熱流耦合
- ☕ **咖啡專業性**: 針對沖煮過程的專業化分析
- 🚀 **工業級穩定性**: 保持現有系統的100%穩定性
- 📊 **研究級分析**: 20+種專業CFD分析功能

---

## 🏗️ 技術架構設計

### 雙分布函數LBM方法
```python
class ThermalLBMSolver:
    # 流體分布函數 (現有)
    f_field: ti.field(ti.f32, shape=(NX, NY, NZ, 19))  # D3Q19
    
    # 溫度分布函數 (新增)
    g_field: ti.field(ti.f32, shape=(NX, NY, NZ, 7))   # D3Q7
    
    # 溫度場
    temperature: ti.field(ti.f32, shape=(NX, NY, NZ))
    
    # 熱物性參數
    thermal_diffusivity: ti.field(ti.f32, shape=(NX, NY, NZ))
    heat_capacity: ti.field(ti.f32, shape=(NX, NY, NZ))
```

### 耦合機制
1. **流體 → 熱傳**: 對流傳熱 (`u·∇T`)
2. **熱傳 → 流體**: 浮力效應 (Boussinesq近似)
3. **熱物性耦合**: 溫度依賴的黏度與密度
4. **咖啡專業耦合**: 萃取反應熱與多孔介質熱傳

---

## 🚀 分階段實施計劃

## 🎯 Phase 1: 基礎熱傳模組 (Week 1)
**目標**: 建立獨立的溫度場求解器，不影響現有流體系統

### 📅 Day 1-2: 核心熱傳LBM
**新建檔案**:
- `thermal_lbm.py` - D3Q7溫度場求解器
- `thermal_properties.py` - 熱物性參數管理  
- `thermal_config.py` - 熱傳專用配置

**具體任務**:
- [ ] 實現D3Q7格子結構 (速度集合: {(0,0,0), (±1,0,0), (0,±1,0), (0,0,±1)})
- [ ] BGK碰撞運算子實現
- [ ] 基礎流場步驟
- [ ] 溫度場重建算法
- [ ] 單元測試 (`test_thermal_lbm.py`)

### 📅 Day 3-4: 熱邊界條件
**擴展檔案**:
- `boundary_conditions.py` - 新增熱邊界處理

**邊界類型**:
- **Dirichlet**: 固定溫度 (注水口 93°C)
- **Neumann**: 固定熱流 (絕熱邊界)  
- **Robin**: 對流散熱 (濾杯壁面)
- **多孔介質**: 有效熱傳導 (咖啡粉床)

### 📅 Day 5-7: 獨立驗證測試
**驗證案例**:
1. **1D熱傳導**: 解析解對比 (誤差<1%)
2. **2D對流擴散**: 已知解對比
3. **熱邊界驗證**: 各類邊界條件測試
4. **數值穩定性**: 長時間運行測試

**✅ 成功標準**: 獨立熱傳模組100%穩定運行

---

## 🔗 Phase 2: 弱耦合實現 (Week 2)
**目標**: 實現單向耦合 (流體→熱傳)，保持流體場不變

### 📅 Day 8-10: 對流傳熱耦合
**修改檔案**:
- `thermal_lbm.py` - 新增對流項處理
- `lbm_solver.py` - 提供速度場接口

**核心實現**:
```python
@ti.kernel
def thermal_advection_step():
    # 獲取流體速度場 (只讀)
    u_field = lbm_solver.get_velocity_field()
    
    # 計算對流項：-u·∇T
    for i, j, k in temperature:
        advection_term = compute_advection(u_field[i,j,k], i, j, k)
        thermal_source[i, j, k] = advection_term
```

### 📅 Day 11-12: 系統集成測試
**新建檔案**:
- `thermal_fluid_coupled.py` - 弱耦合系統主控
- `test_weak_coupling.py` - 弱耦合測試套件

**測試場景**:
1. **熱水注入**: 93°C → 25°C溫度場演化
2. **流動攜帶**: 溫度隨流體輸運
3. **穩定性檢查**: 50步以上無發散

### 📅 Day 13-14: 性能優化與驗證
- GPU記憶體優化
- 計算效能基準測試
- 與流體場的數據交換優化

**✅ 成功標準**: 弱耦合系統穩定運行，溫度場合理分布

---

## ⚖️ Phase 3: 雙向耦合實現 (Week 3)
**目標**: 實現完整的熱流雙向耦合，包含浮力效應

### 📅 Day 15-17: 浮力耦合機制
**修改檔案**:
- `lbm_solver.py` - 新增浮力外力項
- `thermal_coupling.py` - 雙向耦合控制器

**核心實現**:
```python
@ti.kernel
def buoyancy_coupling():
    for i, j, k in density:
        T_local = temperature[i, j, k]
        # Boussinesq近似
        rho_thermal = rho_0 * (1 - beta * (T_local - T_ref))
        density[i, j, k] = rho_thermal
        
        # 浮力項加入LBM
        F_buoyancy = rho_0 * g * beta * (T_local - T_ref)
        external_force[i, j, k] += [0, 0, F_buoyancy]
```

### 📅 Day 18-19: 熱物性耦合
**新增功能**:
- `temperature_dependent_viscosity()` - 黏度-溫度關係
- `adaptive_thermal_properties()` - 動態熱物性更新

**物性模型**:
- **水的黏度**: Vogel方程
- **熱傳導係數**: 相態依賴
- **密度**: Boussinesq近似

### 📅 Day 20-21: 耦合穩定性控制
**新建檔案**:
- `coupling_stability.py` - 耦合穩定性監控
- `adaptive_timestep.py` - 自適應時間步長

**穩定性策略**:
- 溫度變化率限制
- 密度梯度限制  
- 浮力項強度限制
- 動態時間步長調整

**✅ 成功標準**: 雙向耦合系統100步以上穩定運行

---

## 🎨 Phase 4: 視覺化與分析 (Week 4)
**目標**: 完整的熱流CFD分析能力，專業級視覺化

### 📅 Day 22-24: 熱傳CFD分析
**擴展檔案**:
- `enhanced_visualizer.py` - 新增6種熱傳分析圖

**新增分析圖表**:
1. **溫度場分布** (`thermal_field_step_XXXX.png`)
2. **熱流密度向量** (`heat_flux_vectors_step_XXXX.png`)
3. **無量綱熱傳數** (`thermal_dimensionless_step_XXXX.png`)
4. **浮力效應分析** (`buoyancy_effects_step_XXXX.png`)
5. **熱邊界層** (`thermal_boundary_layer_step_XXXX.png`)
6. **溫度-速度耦合** (`thermal_velocity_coupling_step_XXXX.png`)

### 📅 Day 25-26: 專業無量綱分析
**新建檔案**:
- `thermal_dimensionless.py` - 無量綱數計算
- `thermal_analytics.py` - 熱傳工程分析

**無量綱數追蹤**:
- **Nusselt數**: 對流vs傳導強度
- **Prandtl數**: 動量vs熱擴散
- **Rayleigh數**: 自然對流臨界
- **Grashof數**: 浮力vs黏性
- **Péclet數**: 對流vs擴散

### 📅 Day 27-28: 報告系統整合
**擴展報告系統**:
```
report/YYYYMMDD_HHMMSS/
├── thermal_analysis/     # 熱傳專業分析
├── coupling_metrics/     # 耦合效應指標  
├── dimensionless/        # 無量綱數時序
└── validation/           # 驗證對比數據
```

**✅ 成功標準**: 完整熱流CFD報告生成

---

## 🔬 Phase 5: 咖啡專業化應用 (Week 5)
**目標**: 針對咖啡沖煮的專業化功能

### 📅 Day 29-31: 咖啡專業物理
**新建檔案**:
- `coffee_thermal_physics.py` - 咖啡專業熱物理
- `extraction_kinetics.py` - 萃取動力學耦合
- `brewing_analytics.py` - 沖煮分析工具

**咖啡專業功能**:
- **萃取溫度窗口**: 85-96°C最佳萃取分析
- **咖啡粉床加熱**: 非均勻加熱模式分析
- **出水溫度**: 時序溫度變化追蹤
- **熱萃取效率**: 溫度-萃取率關係建模

### 📅 Day 32-33: V60專業化模擬
**擴展檔案**:
- `precise_pouring.py` - 熱注水模式
- `v60_thermal_analysis.py` - V60專業熱分析

**V60熱特性**:
- **螺旋肋條效應**: 熱對流增強機制
- **錐形散熱**: 幾何散熱分析
- **濾紙熱阻**: 熱傳導阻抗效應

### 📅 Day 34-35: 專業報告與驗證
**最終成果**:
- **世界級咖啡CFD系統**: 熱流完全耦合
- **20+專業分析圖表**: 流體+熱傳雙重分析
- **咖啡科學報告**: 專業沖煮建議
- **系統穩定性**: 生產級可靠性

**✅ 成功標準**: 完整系統通過所有驗證測試

---

## 🎯 里程碑檢查點

### ✅ Milestone 1 (Day 7): 獨立熱傳
- 熱傳LBM 100%正確實現
- 所有邊界條件驗證通過
- 基準測試性能達標

### ✅ Milestone 2 (Day 14): 弱耦合
- 流體→熱傳耦合穩定
- 溫度場分布物理合理
- 系統記憶體使用最佳化

### ✅ Milestone 3 (Day 21): 雙向耦合  
- 熱流完全耦合運行
- 浮力效應正確實現
- 數值穩定性100%保證

### ✅ Milestone 4 (Day 28): 專業分析
- 完整CFD分析報告
- 所有無量綱數正確
- 視覺化達到研究級水準

### ✅ Milestone 5 (Day 35): 咖啡應用
- 咖啡專業功能完整
- 系統達到工業級穩定性
- 成為世界級咖啡CFD平台

---

## 🛡️ 風險控制策略

### 🚨 Critical Path 保護
- **每日備份**: 確保可隨時回滾
- **分支開發**: 主分支始終保持穩定
- **增量測試**: 每個功能立即驗證

### 📊 性能監控
- **記憶體使用**: 不超過8GB限制
- **計算時間**: 新功能不超過20%開銷
- **數值精度**: 維持現有精度標準

### 🔧 技術債務控制
- **代碼重構**: 避免複雜度過高
- **文檔同步**: 實時更新技術文檔
- **測試覆蓋**: 保持90%以上覆蓋率

---

## 📋 物理參數規格

### 水的熱物性 (93°C注水)
```python
thermal_conductivity_water = 0.68    # W/(m·K)
thermal_diffusivity_water = 1.6e-7   # m²/s
heat_capacity_water = 4180           # J/(kg·K)
thermal_expansion_water = 3.2e-4     # 1/K

T_inlet = 93.0      # °C (注水溫度)
T_ambient = 25.0    # °C (環境溫度)
T_initial = 25.0    # °C (初始咖啡粉溫度)
```

### 咖啡粉的熱物性
```python
thermal_conductivity_coffee = 0.3    # W/(m·K)
thermal_diffusivity_coffee = 1.2e-7  # m²/s
heat_capacity_coffee = 1800          # J/(kg·K)

# 多孔介質有效熱傳導
k_eff = porosity * k_water + (1-porosity) * k_coffee
```

### 邊界條件
```python
# 注水口：固定溫度
apply_dirichlet_thermal(inlet_region, T_inlet)

# 濾杯壁面：對流散熱
h_natural_convection = 5.0   # W/(m²·K)
apply_convective_thermal(wall_region, h_conv, T_ambient)

# 濾紙：多孔介質熱傳
apply_porous_thermal(filter_region, k_eff, porosity)

# 出口：零梯度
apply_neumann_thermal(outlet_region, gradient=0.0)
```

---

## 🔧 核心實現策略

### 雙向耦合時序
```python
def thermal_fluid_coupling_step():
    # Step 1: 更新溫度場 (使用前一步的速度場)
    thermal_collision()
    thermal_streaming()
    compute_temperature()
    
    # Step 2: 更新熱物性
    update_thermal_properties()
    temperature_dependent_viscosity()
    
    # Step 3: 計算浮力項
    compute_buoyancy_force()
    
    # Step 4: 更新流體場 (包含浮力)
    lbm_collision_with_force()
    lbm_streaming()
    compute_velocity_density()
    
    # Step 5: 檢查耦合穩定性
    check_coupling_stability()
```

### 數值穩定性保證
```python
# CFL條件擴展
CFL_thermal = alpha * dt / (dx^2) < 0.5
CFL_convection = u * dt / dx < 0.1
CFL_combined = max(CFL_thermal, CFL_convection) < 0.1

# 穩定性限制
max_dT_dt = 10.0        # K/s
max_drho_dt = 0.1       # kg/m³/s
max_buoyancy = 0.1 * gravity
```

---

## 📊 測試與驗證

### 單元測試套件
- `test_thermal_lbm.py` - D3Q7求解器測試
- `test_thermal_boundaries.py` - 邊界條件測試
- `test_weak_coupling.py` - 弱耦合測試
- `test_strong_coupling.py` - 雙向耦合測試
- `test_thermal_stability.py` - 數值穩定性測試

### 驗證基準
1. **解析解對比**: 1D/2D熱傳導問題
2. **文獻對比**: 經典自然對流問題
3. **實驗驗證**: 咖啡沖煮溫度實測數據
4. **穩定性測試**: 1000步長時間運行

### 性能基準
- **記憶體增長**: <30% 
- **計算時間增長**: <20%
- **數值精度**: 維持現有標準
- **GPU效率**: >80% 利用率

---

## 🎯 執行指令

### 開發環境測試
```bash
# 熱傳獨立測試
python test_thermal_lbm.py

# 弱耦合測試  
python test_weak_coupling.py

# 完整耦合測試
python main.py debug 10

# 專業分析報告
python enhanced_visualizer.py
```

### 階段性驗證
```bash
# Phase 1: 基礎熱傳
python thermal_lbm.py --test-independent

# Phase 2: 弱耦合
python thermal_fluid_coupled.py --test-weak

# Phase 3: 雙向耦合
python main.py thermal debug 50

# Phase 4: 專業分析
python main.py thermal analysis

# Phase 5: 咖啡專業
python brewing_analytics.py --full-analysis
```

---

## 📚 相關文檔

- `README.md` - 專案主要說明
- `AGENTS.md` - 開發指導原則
- `DEPENDENCY_ANALYSIS.md` - 技術依賴分析
- `docs/technical/technical_paper.md` - 技術論文
- `docs/physics/physics_modeling.md` - 物理建模文檔

---

## 🚀 後續擴展計劃

### 短期目標 (1-2個月)
- **質量傳輸**: 咖啡萃取物濃度場模擬
- **多尺度建模**: 分子動力學-連續介質耦合
- **機器學習**: 智能沖煮參數優化

### 長期願景 (6-12個月)  
- **商業化應用**: 咖啡機智能控制系統
- **科研平台**: 開源咖啡科學研究工具
- **教育應用**: CFD教學示範系統

---

*📝 文檔創建日期: 2025-07-25*  
*🔄 最後更新: 待開發進度更新*  
*👨‍💻 開發工具: opencode + GitHub Copilot*