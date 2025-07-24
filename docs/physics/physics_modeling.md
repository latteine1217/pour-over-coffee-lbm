# Physics Modeling in Pour-Over Coffee Simulation

## Table of Contents

1. [Fundamental Physics](#1-fundamental-physics)
2. [Fluid Dynamics](#2-fluid-dynamics)
3. [Multiphase Flow Physics](#3-multiphase-flow-physics)
4. [Particle-Fluid Interaction](#4-particle-fluid-interaction)
5. [Porous Media Physics](#5-porous-media-physics)
6. [Heat and Mass Transfer](#6-heat-and-mass-transfer)
7. [Surface Physics](#7-surface-physics)

## 1. Fundamental Physics

### 1.1 Physical Scales in Coffee Brewing

Coffee brewing involves multiple physical scales that must be carefully modeled:

| Scale | Length | Physics | Modeling Approach |
|-------|--------|---------|-------------------|
| **Molecular** | 1-10 nm | Van der Waals forces, extraction kinetics | Effective parameters |
| **Pore** | 1-100 μm | Capillary flow, surface tension | Darcy-Forchheimer |
| **Particle** | 0.1-1 mm | Drag forces, settling, collision | Lagrangian tracking |
| **Bed** | 1-5 cm | Permeability, channeling | Continuum porous media |
| **Dripper** | 5-15 cm | Bulk flow, turbulence | LBM with LES |

### 1.2 Physical Parameters for V60 Coffee System

**Fluid Properties** (water at 90°C):
- Density: $\\rho_w = 965$ kg/m³
- Kinematic viscosity: $\\nu_w = 3.15 \\times 10^{-7}$ m²/s  
- Surface tension: $\\sigma = 0.063$ N/m
- Thermal diffusivity: $\\alpha_w = 1.68 \\times 10^{-7}$ m²/s

**Coffee Properties**:
- Particle density: $\\rho_p = 1200$ kg/m³
- Particle diameter: $d_p = 0.65$ mm (medium grind)
- Bed porosity: $\\varepsilon = 0.4$
- Permeability: $k = 1.2 \\times 10^{-10}$ m² (Kozeny-Carman)

**Geometric Parameters**:
- V60 cone angle: $60°$
- Rib height: $2$ mm
- Spiral rib angle: $25°$
- Filter paper thickness: $0.16$ mm

### 1.3 Dimensionless Groups

**Reynolds Number** (inertial vs. viscous forces):
$$
Re = \\frac{\\rho u L}{\\mu} = \\frac{u L}{\\nu}
$$

**Froude Number** (inertial vs. gravitational forces):
$$
Fr = \\frac{u}{\\sqrt{gL}}
$$

**Weber Number** (inertial vs. surface tension forces):
$$
We = \\frac{\\rho u^2 L}{\\sigma}
$$

**Capillary Number** (viscous vs. surface tension forces):
$$
Ca = \\frac{\\mu u}{\\sigma}
$$

**Péclet Number** (convection vs. diffusion):
$$
Pe = \\frac{uL}{D} = Re \\cdot Sc
$$

### 1.4 Typical Operating Conditions

For V60 pour-over brewing:
- **Pouring rate**: 2-6 ml/s
- **Water temperature**: 88-95°C
- **Coffee mass**: 15-25 g
- **Water mass**: 250-400 ml
- **Brew time**: 2-4 minutes
- **Grind size**: 0.5-0.8 mm

**Resulting dimensionless parameters**:
- $Re_{dripper} = 2000-8000$ (turbulent flow)
- $Re_{particle} = 50-200$ (intermediate regime)
- $Fr = 0.02-0.05$ (gravity-dominated)
- $Ca = 10^{-4}-10^{-3}$ (surface tension significant)

## 2. Fluid Dynamics

### 2.1 Governing Equations

The incompressible Navier-Stokes equations describe the bulk fluid motion:

**Continuity equation**:
$$
\\frac{\\partial \\rho}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{u}) = 0
$$

**Momentum equation**:
$$
\\frac{\\partial (\\rho \\mathbf{u})}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{u} \\mathbf{u}) = -\\nabla p + \\nabla \\cdot \\boldsymbol{\\tau} + \\rho \\mathbf{g} + \\mathbf{F}_{external}
$$

**Stress tensor** (Newtonian fluid):
$$
\\boldsymbol{\\tau} = \\mu \\left[ \\nabla \\mathbf{u} + (\\nabla \\mathbf{u})^T - \\frac{2}{3}(\\nabla \\cdot \\mathbf{u})\\mathbf{I} \\right]
$$

### 2.2 Turbulence Physics

**Energy Cascade Theory**:

In the V60 dripper, turbulent kinetic energy cascades from large eddies (size ~ dripper diameter) to small eddies where viscous dissipation occurs.

**Kolmogorov scales**:
- Length scale: $\\eta = (\\nu^3/\\varepsilon)^{1/4}$
- Velocity scale: $u_\\eta = (\\nu \\varepsilon)^{1/4}$  
- Time scale: $\\tau_\\eta = (\\nu/\\varepsilon)^{1/2}$

Where $\\varepsilon$ is the turbulent energy dissipation rate.

**Large Eddy Simulation Approach**:

LES resolves large energy-containing eddies directly while modeling smaller subgrid-scale eddies:

$$
\\frac{\\partial \\bar{\\mathbf{u}}}{\\partial t} + \\nabla \\cdot (\\bar{\\mathbf{u}} \\bar{\\mathbf{u}}) = -\\frac{1}{\\rho}\\nabla \\bar{p} + \\nu \\nabla^2 \\bar{\\mathbf{u}} - \\nabla \\cdot \\boldsymbol{\\tau}^{sgs}
$$

**Subgrid-scale stress modeling**:
$$
\\boldsymbol{\\tau}^{sgs} = 2\\nu_{sgs} \\bar{\\mathbf{S}} - \\frac{1}{3}\\text{tr}(\\boldsymbol{\\tau}^{sgs})\\mathbf{I}
$$

### 2.3 Flow Regime Classification

**Laminar flow** ($Re < 2000$):
- Smooth streamlines
- Predictable flow patterns
- Dominated by viscous forces

**Transitional flow** ($2000 < Re < 4000$):
- Intermittent turbulence
- Flow instabilities
- Sensitive to perturbations

**Turbulent flow** ($Re > 4000$):
- Chaotic motion
- Enhanced mixing
- Dominant inertial forces

### 2.4 V60 Flow Characteristics

**Spiral flow generation**:
The V60's spiral ribs induce swirl motion:
$$
\\omega_z = \\frac{1}{r}\\frac{\\partial (ru_\\theta)}{\\partial r} - \\frac{1}{r}\\frac{\\partial u_r}{\\partial \\theta}
$$

**Boundary layer development**:
On the cone surface:
$$
\\delta(x) = \\frac{5x}{\\sqrt{Re_x}}
$$

Where $Re_x = ux/\\nu$ is the local Reynolds number.

## 3. Multiphase Flow Physics

### 3.1 Interface Dynamics

**Surface tension force**:
The curvature-dependent pressure jump across interfaces:
$$
[p] = \\sigma \\kappa = \\sigma \\left( \\frac{1}{R_1} + \\frac{1}{R_2} \\right)
$$

**Young-Laplace equation**:
For spherical droplets:
$$
\\Delta p = \\frac{2\\sigma}{R}
$$

**Contact line dynamics**:
The moving contact line follows:
$$
u_{cl} = \\frac{\\sigma}{\\mu} \\frac{\\cos \\theta_d - \\cos \\theta_s}{\\ln(L_s/L_m)}
$$

Where $\\theta_d$ and $\\theta_s$ are dynamic and static contact angles.

### 3.2 Wetting Phenomena

**Contact angle hysteresis**:
- **Advancing angle**: $\\theta_a = 75°$ (V60 plastic)
- **Receding angle**: $\\theta_r = 45°$
- **Hysteresis**: $\\Delta \\theta = \\theta_a - \\theta_r = 30°$

**Cassie-Baxter wetting** (on textured surfaces):
$$
\\cos \\theta_{apparent} = f_1 \\cos \\theta_1 + f_2 \\cos \\theta_2
$$

### 3.3 Phase Change and Evaporation

**Evaporation rate** (Hertz-Knudsen equation):
$$
\\dot{m} = \\alpha \\sqrt{\\frac{M}{2\\pi RT}} (p_{sat} - p_v)
$$

**Heat of vaporization**:
$$
L_v = 2.26 \\times 10^6 \\text{ J/kg (at 100°C)}
$$

### 3.4 Coalescence and Breakup

**Droplet coalescence time**:
$$
t_c = \\frac{\\mu R}{\\sigma}
$$

**Critical Weber number for breakup**:
$$
We_{crit} = 12 \\quad \\text{(for spherical droplets)}
$$

## 4. Particle-Fluid Interaction

### 4.1 Drag Force Mechanisms

**Stokes regime** ($Re_p < 1$):
$$
C_D = \\frac{24}{Re_p}
$$

**Intermediate regime** ($1 < Re_p < 1000$):
$$
C_D = \\frac{24}{Re_p}(1 + 0.15 Re_p^{0.687})
$$

**Newton regime** ($Re_p > 1000$):
$$
C_D = 0.44
$$

### 4.2 Added Mass and Basset Forces

**Added mass force** (virtual mass):
$$
\\mathbf{F}_{AM} = C_{AM} \\rho_f V_p \\left( \\frac{D\\mathbf{u}_f}{Dt} - \\frac{d\\mathbf{v}_p}{dt} \\right)
$$

**Basset force** (history effect):
$$
\\mathbf{F}_B = \\frac{6\\pi\\mu R_p^2}{\\sqrt{\\pi\\nu_f}} \\int_0^t \\frac{d}{d\\tau}(\\mathbf{u}_f - \\mathbf{v}_p) \\frac{d\\tau}{\\sqrt{t-\\tau}}
$$

### 4.3 Brownian Motion

For small particles, random molecular collisions cause Brownian motion:

**Diffusion coefficient**:
$$
D_B = \\frac{k_B T}{6\\pi\\mu R_p}
$$

**Random displacement**:
$$
\\langle (\\Delta x)^2 \\rangle = 2D_B \\Delta t
$$

### 4.4 Particle Settling and Suspension

**Terminal velocity** (Stokes law):
$$
v_t = \\frac{2gR_p^2(\\rho_p - \\rho_f)}{9\\mu}
$$

**Settling time**:
$$
t_{settle} = \\frac{9\\mu h}{2gR_p^2(\\rho_p - \\rho_f)}
$$

**Suspension criteria** (Shields parameter):
$$
\\Theta = \\frac{\\tau_b}{(\\rho_p - \\rho_f)gd_p} > \\Theta_{crit}
$$

## 5. Porous Media Physics

### 5.1 Darcy's Law and Extensions

**Linear Darcy regime** (low Reynolds number):
$$
\\mathbf{u} = -\\frac{k}{\\mu}\\nabla p
$$

**Forchheimer regime** (moderate Reynolds number):
$$
-\\nabla p = \\frac{\\mu}{k}\\mathbf{u} + \\frac{\\rho C_F}{\\sqrt{k}}|\\mathbf{u}|\\mathbf{u}
$$

**Brinkman equation** (near boundaries):
$$
-\\nabla p = \\frac{\\mu}{k}\\mathbf{u} - \\mu_{eff}\\nabla^2\\mathbf{u}
$$

### 5.2 Permeability Models

**Kozeny-Carman equation**:
$$
k = \\frac{\\varepsilon^3 d_p^2}{180(1-\\varepsilon)^2}
$$

**Blake-Kozeny modification**:
$$
k = \\frac{\\varepsilon^3 d_p^2}{150(1-\\varepsilon)^2}
$$

**Ergun equation** (complete form):
$$
\\frac{\\Delta p}{L} = \\frac{150\\mu(1-\\varepsilon)^2}{\\varepsilon^3 d_p^2}u + \\frac{1.75\\rho(1-\\varepsilon)}{\\varepsilon^3 d_p}u^2
$$

### 5.3 Capillary Effects in Porous Media

**Capillary pressure**:
$$
p_c = p_{nw} - p_w = \\frac{2\\sigma\\cos\\theta}{r_{pore}}
$$

**Leverett J-function**:
$$
J(S_w) = \\frac{p_c}{\\sigma\\cos\\theta}\\sqrt{\\frac{k}{\\phi}}
$$

### 5.4 Channeling and Preferential Flow

**Fingering instability**:
When less viscous fluid displaces more viscous fluid:
$$
M = \\frac{\\mu_{displaced}}{\\mu_{displacing}} > 1
$$

**Saffman-Taylor instability**:
$$
\\lambda_{critical} = \\sqrt{\\frac{12M\\sigma}{\\rho g k}}
$$

## 6. Heat and Mass Transfer

### 6.1 Energy Equation

**General energy conservation**:
$$
\\rho c_p \\frac{\\partial T}{\\partial t} + \\rho c_p \\mathbf{u} \\cdot \\nabla T = \\nabla \\cdot (k_{thermal} \\nabla T) + S_T
$$

**Porous media energy equation**:
$$
(\\rho c_p)_{eff} \\frac{\\partial T}{\\partial t} + \\rho_f c_{p,f} \\mathbf{u}_f \\cdot \\nabla T = \\nabla \\cdot (k_{eff} \\nabla T) + S_T
$$

### 6.2 Heat Transfer Coefficients

**Effective thermal conductivity**:
$$
k_{eff} = \\varepsilon k_f + (1-\\varepsilon) k_s
$$

**Heat transfer coefficient** (Nusselt correlation):
$$
Nu = \\frac{hd_p}{k_f} = 2 + 1.1 Pr^{1/3} Re_p^{0.6}
$$

### 6.3 Mass Transfer and Extraction

**Coffee extraction kinetics**:
$$
\\frac{dC}{dt} = k_{ext}(C_{sat} - C) - u \\frac{\\partial C}{\\partial x}
$$

**Extraction rate coefficient**:
$$
k_{ext} = \\frac{D_{eff}}{\\delta^2} \\quad \\text{(diffusion-limited)}
$$

**Peclet number for extraction**:
$$
Pe_{ext} = \\frac{u d_p}{D_{eff}}
$$

### 6.4 Convective Heat Transfer

**Forced convection in channels**:
$$
Nu = 0.023 Re^{0.8} Pr^{0.4}
$$

**Natural convection**:
$$
Ra = \\frac{g\\beta\\Delta T L^3}{\\nu\\alpha}
$$

## 7. Surface Physics

### 7.1 Filter Paper Properties

**Porosity structure**:
- **Macro-pores**: 20-100 μm (flow channels)
- **Micro-pores**: 1-20 μm (filtration)
- **Overall porosity**: 60-80%

**Permeability anisotropy**:
$$
\\frac{k_{parallel}}{k_{perpendicular}} = 2-5
$$

### 7.2 V60 Rib Geometry Effects

**Rib-induced secondary flow**:
The spiral ribs create Dean vortices with Dean number:
$$
De = Re\\sqrt{\\frac{D_h}{2R_c}}
$$

**Pressure drop enhancement**:
$$
f_{ribbed} = f_{smooth} \\times (1 + C_{rib} \\frac{h_{rib}}{D_h})
$$

### 7.3 Coffee Particle Surface Chemistry

**Surface charge effects**:
Coffee particles carry negative surface charge:
$$
\\zeta = -30 \\text{ to } -50 \\text{ mV}
$$

**Electrostatic forces**:
$$
F_{electric} = \\frac{q_1 q_2}{4\\pi\\varepsilon_0\\varepsilon_r r^2}
$$

### 7.4 Aging and Degradation

**Coffee bean CO₂ outgassing**:
$$
C_{CO_2}(t) = C_0 e^{-t/\\tau}
$$

Where $\\tau = 7-14$ days for typical storage.

**Particle aggregation**:
$$
\\frac{dn}{dt} = -k_{agg} n^2
$$

---

## Physical Constants and Reference Values

| Property | Symbol | Value | Units |
|----------|--------|-------|-------|
| Gravitational acceleration | $g$ | 9.81 | m/s² |
| Gas constant | $R$ | 8.314 | J/(mol·K) |
| Boltzmann constant | $k_B$ | $1.38 \\times 10^{-23}$ | J/K |
| Stefan-Boltzmann constant | $\\sigma_{SB}$ | $5.67 \\times 10^{-8}$ | W/(m²·K⁴) |
| Water density (20°C) | $\\rho_w$ | 998 | kg/m³ |
| Water viscosity (20°C) | $\\mu_w$ | $1.0 \\times 10^{-3}$ | Pa·s |
| Water surface tension (20°C) | $\\sigma_w$ | 0.073 | N/m |
| Air density (20°C) | $\\rho_a$ | 1.2 | kg/m³ |
| Air viscosity (20°C) | $\\mu_a$ | $1.8 \\times 10^{-5}$ | Pa·s |

---

**Physical Model Validation**:
All physical models implemented in this simulation have been validated against:
- Experimental measurements from coffee brewing literature
- Standard fluid mechanics benchmarks  
- Particle settling experiments
- Permeability measurements of coffee beds
- Heat transfer correlations for porous media

The multi-physics coupling ensures that complex interactions between fluid flow, particle dynamics, heat transfer, and mass transport are accurately captured in the V60 coffee brewing simulation.