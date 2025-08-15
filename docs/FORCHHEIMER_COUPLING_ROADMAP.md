# 🔧 Forchheimer項與Phase 2顆粒-流體強耦合實現路線圖

> **CFD專家審查後的技術完善計畫**  
> 基於理論分析，優先實現Forchheimer非線性阻力與雙向動量耦合  
> 開發：opencode + GitHub Copilot

## 📋 **實現概述**

基於CFD理論審查，確定兩個關鍵技術提升方向：
1. **Forchheimer項完善**: 高速多孔介質流動的非線性阻力建模
2. **Phase 2強耦合**: 顆粒-流體雙向動量傳遞的完整實現

---

## 🎯 **Part I: Forchheimer項完善**

### **理論基礎**
```
Forchheimer方程: ∇p = (μ/K)u + (ρβ/√K)|u|u
- 第一項: Darcy線性阻力 (低速)
- 第二項: 慣性非線性阻力 (高速)
- K: 滲透率, β: Forchheimer係數
```

### **實現要點**

#### **1. 擴展FilterPaperSystem**
```python
# src/physics/filter_paper.py
class FilterPaperSystem:
    def __init__(self):
        # 新增Forchheimer參數場
        self.forchheimer_coeff = ti.field(dtype=ti.f32, shape=(NX, NY, NZ))
        self.permeability = ti.field(dtype=ti.f32, shape=(NX, NY, NZ))
        self.velocity_magnitude = ti.field(dtype=ti.f32, shape=(NX, NY, NZ))
```

#### **2. 核心計算內核**
```python
@ti.kernel
def compute_forchheimer_resistance(self):
    for i, j, k in ti.ndrange(1, NX-1, 1, NY-1, 1, NZ-1):
        if self.is_in_coffee_bed(i, j, k):
            u_vec = self.lbm.u[i, j, k]
            u_mag = u_vec.norm()
            
            # Darcy線性項
            darcy_resistance = self.viscosity / self.permeability[i, j, k]
            
            # Forchheimer非線性項
            forchheimer_resistance = (
                self.density * self.forchheimer_coeff[i, j, k] * u_mag / 
                ti.sqrt(self.permeability[i, j, k])
            )
            
            # 總阻力應用
            total_resistance = darcy_resistance + forchheimer_resistance
            resistance_force = -total_resistance * u_vec
            self.body_force[i, j, k] += resistance_force
```

#### **3. 參數估算**
```python
def estimate_forchheimer_parameters(self):
    """基於Ergun方程估算參數"""
    dp = config.PARTICLE_DIAMETER_MM * 1e-3
    porosity = config.PORE_PERC
    
    # Kozeny-Carman滲透率
    K = (dp**2 * porosity**3) / (180 * (1 - porosity)**2)
    
    # Ergun Forchheimer係數
    beta = 1.75 / (porosity**3)
    
    return K, beta
```

---

## 🔧 **Part II: Phase 2顆粒-流體強耦合**

### **核心挑戰**
- **雙向動量傳遞**: 流體↔顆粒的完整動量交換
- **拖曳力模型**: Reynolds數依賴的動態拖曳係數
- **數值穩定性**: 強耦合系統的穩定時間積分

### **架構設計**

#### **1. 擴展顆粒系統**
```python
# src/physics/coffee_particles.py
class CoffeeParticleSystem:
    def __init__(self):
        # 強耦合新屬性
        self.drag_coefficient = ti.field(dtype=ti.f32, shape=max_particles)
        self.particle_reynolds = ti.field(dtype=ti.f32, shape=max_particles)
        self.fluid_velocity_at_particle = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.drag_force = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        
        # 反作用力場（顆粒→流體）
        self.reaction_force_field = ti.Vector.field(3, dtype=ti.f32, shape=(NX, NY, NZ))
```

#### **2. 拖曳力模型**
```python
@ti.func
def compute_drag_coefficient(self, re_p: ti.f32) -> ti.f32:
    """Reynolds數依賴拖曳係數"""
    if re_p < 0.1:
        return 24.0 / re_p  # Stokes區域
    elif re_p < 1000.0:
        return 24.0 / re_p * (1.0 + 0.15 * ti.pow(re_p, 0.687))  # Schiller-Naumann
    else:
        return 0.44  # 牛頓阻力區域

@ti.kernel
def compute_particle_drag_forces(self):
    for p in range(self.max_particles):
        if self.active[p]:
            # 插值流體速度
            u_fluid = self.interpolate_fluid_velocity(p)
            u_rel = u_fluid - self.velocity[p]
            u_rel_mag = u_rel.norm()
            
            if u_rel_mag > 1e-8:
                # 顆粒Reynolds數
                re_p = self.water_density * u_rel_mag * 2 * self.radius[p] / self.water_viscosity
                
                # 拖曳力計算
                cd = self.compute_drag_coefficient(re_p)
                area = 3.14159 * self.radius[p] * self.radius[p]
                drag_magnitude = 0.5 * self.water_density * cd * area * u_rel_mag
                self.drag_force[p] = drag_magnitude * u_rel / u_rel_mag
```

#### **3. 雙向耦合核心**
```python
@ti.kernel
def apply_two_way_coupling(self, dt: ti.f32):
    # 清零反作用力場
    for i, j, k in ti.ndrange(NX, NY, NZ):
        self.reaction_force_field[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    
    # 顆粒動量更新 + 反作用力分布
    for p in range(self.max_particles):
        if self.active[p]:
            # 顆粒速度更新（流體→顆粒）
            acceleration = (self.drag_force[p] + self.gravity_force) / self.mass[p]
            self.velocity[p] += acceleration * dt
            self.position[p] += self.velocity[p] * dt
            
            # 反作用力分布到網格（顆粒→流體）
            reaction_force = -self.drag_force[p]
            self.distribute_force_to_grid(p, reaction_force)

@ti.func
def distribute_force_to_grid(self, particle_idx: ti.i32, force: ti.template()):
    """三線性插值分布反作用力"""
    pos = self.position[particle_idx]
    i, j, k = ti.cast(pos[0], ti.i32), ti.cast(pos[1], ti.i32), ti.cast(pos[2], ti.i32)
    
    if 0 <= i < NX-1 and 0 <= j < NY-1 and 0 <= k < NZ-1:
        fx, fy, fz = pos[0] - i, pos[1] - j, pos[2] - k
        
        # 8個權重計算
        weights = [
            (1-fx)*(1-fy)*(1-fz), (1-fx)*(1-fy)*fz, (1-fx)*fy*(1-fz), (1-fx)*fy*fz,
            fx*(1-fy)*(1-fz), fx*(1-fy)*fz, fx*fy*(1-fz), fx*fy*fz
        ]
        
        # 原子操作分布力
        positions = [(i,j,k), (i,j,k+1), (i,j+1,k), (i,j+1,k+1), 
                    (i+1,j,k), (i+1,j,k+1), (i+1,j+1,k), (i+1,j+1,k+1)]
        
        for idx, (ii, jj, kk) in enumerate(positions):
            ti.atomic_add(self.reaction_force_field[ii, jj, kk], weights[idx] * force)
```

#### **4. LBM集成**
```python
# src/core/lbm_solver.py
def step(self, dt):
    """包含顆粒耦合的LBM步進"""
    self.collision_step()
    
    # 顆粒耦合步驟
    if self.particle_system:
        self.particle_system.compute_particle_drag_forces()
        self.particle_system.apply_two_way_coupling(dt)
        self.add_particle_reaction_forces()
    
    self.streaming_step()
    self.apply_boundary_conditions()

@ti.kernel
def add_particle_reaction_forces(self):
    """將顆粒反作用力加入LBM體力項"""
    for i, j, k in ti.ndrange(1, NX-1, 1, NY-1, 1, NZ-1):
        self.body_force[i, j, k] += self.particle_system.reaction_force_field[i, j, k]
```

---

## 📅 **實現時間表**

### **Week 1-2: Forchheimer項**
- [ ] 擴展FilterPaperSystem類
- [ ] 實現非線性阻力計算
- [ ] 參數校準與單元測試
- [ ] 壓降驗證測試

### **Week 3-4: 基礎雙向耦合**
- [ ] 擴展CoffeeParticleSystem
- [ ] 實現Reynolds數依賴拖曳模型
- [ ] 基礎反作用力分布
- [ ] 單顆粒沉降驗證

### **Week 5-6: 完整強耦合**
- [ ] 三線性插值優化
- [ ] 亞鬆弛穩定性控制
- [ ] 多顆粒集體行為測試
- [ ] 性能優化

### **Week 7: 驗證與調優**
- [ ] 物理合理性全面驗證
- [ ] 數值收斂性測試
- [ ] 性能基準測試
- [ ] 文檔更新

---

## 🔍 **數值穩定性保證**

### **亞鬆弛技術**
```python
@ti.kernel
def apply_under_relaxation(self, relaxation_factor: ti.f32):
    """防止數值震蕩"""
    for p in range(self.max_particles):
        if self.active[p]:
            self.drag_force[p] = (
                relaxation_factor * self.drag_force_new[p] + 
                (1.0 - relaxation_factor) * self.drag_force_old[p]
            )
```

**建議參數**: 鬆弛因子 = 0.5-0.8

### **自適應插值**
```python
@ti.func
def adaptive_interpolation(self, pos: ti.template()):
    """根據速度梯度選擇插值精度"""
    gradient = self.compute_velocity_gradient_magnitude(pos)
    if gradient > threshold:
        return self.hermite_interpolation(pos)  # 高精度
    else:
        return self.trilinear_interpolation(pos)  # 高效率
```

---

## 📊 **預期成果**

### **性能影響**
- **計算開銷**: 基礎性能的1.8倍
  - Forchheimer項: +15%
  - 弱耦合: +25% 
  - 強耦合: +40%
- **記憶體增加**: ~185MB (50MB顆粒 + 135MB反作用力場)

### **精度提升**
- **壓力分布準確性**: +30-50%
- **顆粒軌跡真實性**: +60-80%
- **整體物理一致性**: 7.2/10 → 9.0/10

### **驗證基準**
1. **Forchheimer驗證**: 與實驗壓降曲線對比
2. **拖曳驗證**: 單顆粒沉降速度理論對比
3. **耦合驗證**: 多顆粒流化床臨界速度
4. **守恆驗證**: 系統動量守恆檢查

---

## ⚠️ **風險評估與緩解**

### **技術風險**
| 風險 | 影響 | 緩解策略 |
|------|------|----------|
| 數值不穩定 | 高 | 亞鬆弛 + 自適應時間步 |
| 性能下降 | 中 | GPU優化 + 緊湊數據結構 |
| 記憶體不足 | 中 | 動態記憶體管理 |

### **實現風險**
| 風險 | 影響 | 緩解策略 |
|------|------|----------|
| 集成複雜性 | 中 | 模組化設計 + 逐步集成 |
| 調試困難 | 中 | 完整診斷工具 |
| 參數敏感性 | 低 | 自動參數調優 |

---

## 🚀 **後續發展**

完成此路線圖後，系統將具備：
- ✅ **企業級物理準確性**: 真實咖啡沖泡建模
- ✅ **研究價值**: 咖啡工藝最佳化應用
- ✅ **技術擴展性**: 為萃取動力學奠定基礎
- ✅ **工業應用**: 適用於其他多孔介質問題

**下一階段目標**: 萃取動力學與動態孔隙度演化

---

*📝 文檔版本: v1.0 | 創建日期: 2025-01-15 | 最後更新: 2025-01-15*