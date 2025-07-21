# Agent Instructions for Pour-Over Coffee LBM Simulation

## Build/Test/Run Commands
```bash
python main.py                    # Run complete 3D LBM simulation
python main.py interactive        # Interactive mode with step control
python test_simple.py             # Run basic functionality tests
```

## Simplified File Structure
**Core Files:**
- `main.py` - Main simulation program (3D only)
- `lbm_solver.py` - 3D LBM solver (D3Q19)
- `visualizer.py` - 3D visualization system
- `config.py` - All simulation parameters and constants

**Domain-Specific Modules:**
- `multiphase_3d.py` - 3D multiphase flow (air-water interface)
- `porous_media_3d.py` - Coffee bed porous media simulation
- `coffee_particles.py` - Coffee particle tracking system
- `precise_pouring.py` - V60 pouring pattern control
- `init.py` - Initialization utilities

**Testing & Examples:**
- `test_simple.py` - Basic 3D functionality tests
- `test_visualization_labels.py` - Visualization testing (legacy)
- `main_safe_backup.py` - Backup of previous stable version

## Dependencies
- **Taichi 1.7.3**: GPU-accelerated LBM computation (primary dependency)
- **NumPy**: Array operations and numerical computations
- Standard Python libraries (math, time, sys)

## Code Style & Conventions
- **Language**: Python with Chinese comments for domain-specific physics explanations
- **Imports**: Standard library first, then third-party (taichi, numpy), then local modules
- **Naming**: snake_case for variables/functions, UPPER_CASE for constants in config.py
- **Comments**: Use Chinese for physics/domain explanations, English for code logic
- **Module Structure**: Unified core modules, specialized domain modules
- **Taichi Kernels**: Use @ti.kernel decorator and @ti.data_oriented for classes
- **Constants**: Define all simulation parameters in config.py, use np.float32 for Metal compatibility
- **Error Handling**: Keep simple, focus on numerical stability over extensive error handling
- **Formatting**: Use clear separation with comment blocks (# ===), maintain consistent indentation
- **Visualization**: Use English labels for plots, proper DPI (300) for output images, descriptive titles

## LBM-Specific Guidelines
- **3D Only**: Single LBMSolver class supports D3Q19 model only
- **Multiphase**: Phase field method for air-water interface tracking
- **Porous Media**: Modified collision for coffee bed with Darcy resistance
- **Boundary Conditions**: Bounce-back for walls, inlet/outlet velocity conditions
- **GPU Compatibility**: Always use ti.f32, not f64; declare numpy arrays with dtype

## Simplified Workflow
1. **Quick Test**: `python test_simple.py` - Verify core functionality
2. **3D Simulation**: `python main.py` - Full coffee brewing simulation
3. **Interactive Mode**: `python main.py interactive` - Step-by-step debugging

## Performance Notes
- **Grid Size**: 64×64×128 (can be reduced in config.py for testing)
- **Memory Usage**: ~4GB GPU memory for full 3D simulation
- **Stability**: 3D mode is stable with current parameters
- **Optimization**: Use sparse computation and reduced grid size for development

## Physical Parameters (90°C Hot Water)
- **Water Temperature**: 90°C
- **Water Density**: 965.3 kg/m³
- **Kinematic Viscosity**: 3.15×10⁻⁷ m²/s
- **Pour Rate**: 4 ml/s (realistic hand pour speed)
- **Pour Height**: 12.5 cm
- **Total Brew Time**: 2:20 (140 seconds)
- **Coffee Bean Density**: 1200 kg/m³ (medium roast)

## V60 Geometry (Updated to Real Specifications)
- **External Dimensions**: 138×115×95 mm (L×W×H)
- **Internal Height**: 8.5 cm (excluding wall thickness)
- **Internal Top Diameter**: 11.1 cm (excluding wall thickness)
- **Outlet Diameter**: 0.4 cm (standard V60 hole)
- **Cone Angle**: 64.4° (actual geometric angle)
- **Internal Volume**: 284.4 cm³

## Coffee Bed Parameters
- **Coffee Powder Mass**: 20g (hand pour standard)
- **Particle Count**: 493,796 particles (based on grind size distribution)
- **Main Particle Diameter**: 0.65mm (granulated sugar size)
- **Coffee Bed Height**: 3.3cm (58% of V60 height limit)
- **Coffee Bed Volume**: 85.3 cm³ (30% V60 filling)
- **Porosity**: 80.5% (realistic for medium-coarse grind)