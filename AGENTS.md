## Visualization Systems (Dual-Track Design)

### Real-time Visualization (visualizer.py)
**Purpose**: Live monitoring during simulation
- **Technology**: Taichi GPU-accelerated rendering
- **Display**: 3D slice views (XY, XZ, YZ planes)
- **Field Types**: Density, velocity, phase, composite fields
- **Features**: Real-time GUI, low-latency updates
- **Usage**: Simulation monitoring, quick checks

### Research-Grade Analysis (enhanced_visualizer.py)
**Purpose**: Deep scientific analysis and report generation
- **Technology**: matplotlib professional plotting
- **Analysis**: Fluid mechanics parameters (Reynolds, vorticity, pressure)
- **Output**: High-quality PNG charts, data export (JSON/NPZ)
- **Features**: Multi-physics analysis, temporal tracking
- **Usage**: Post-simulation detailed research analysis

**Key Distinction**: visualizer.py for real-time monitoring, enhanced_visualizer.py for scientific analysis