# Mathematical Models and Formulations

## Table of Contents

1. [Lattice Boltzmann Method](#1-lattice-boltzmann-method)
2. [Large Eddy Simulation](#2-large-eddy-simulation)
3. [Multiphase Flow Models](#3-multiphase-flow-models)
4. [Particle Dynamics](#4-particle-dynamics)
5. [Porous Media Flow](#5-porous-media-flow)
6. [Boundary Conditions](#6-boundary-conditions)
7. [Numerical Stability Analysis](#7-numerical-stability-analysis)

## 1. Lattice Boltzmann Method

### 1.1 Fundamental LBM Equation

The lattice Boltzmann equation governs the evolution of particle distribution functions:

$$
f_i(\\mathbf{x} + \\mathbf{e}_i \\Delta t, t + \\Delta t) - f_i(\\mathbf{x}, t) = \\Omega_i + \\Delta t F_i
$$

Where the collision operator $\\Omega_i$ is given by the BGK approximation:

$$
\\Omega_i = -\\frac{1}{\\tau}[f_i(\\mathbf{x}, t) - f_i^{eq}(\\mathbf{x}, t)]
$$

**Physical interpretation**:
- $f_i(\\mathbf{x}, t)$: Probability of finding particles with velocity $\\mathbf{e}_i$ at position $\\mathbf{x}$ and time $t$
- $\\tau$: Single relaxation time controlling viscosity
- $F_i$: External force terms (gravity, surface tension)

### 1.2 D3Q19 Velocity Set and Weights

The 19 discrete velocities for the 3D model:

$$
\\mathbf{e}_i = c \\begin{cases}
(0,0,0) & i = 0 \\\\
(\\pm 1, 0, 0), (0, \\pm 1, 0), (0, 0, \\pm 1) & i = 1,...,6 \\\\
(\\pm 1, \\pm 1, 0), (\\pm 1, 0, \\pm 1), (0, \\pm 1, \\pm 1) & i = 7,...,18
\\end{cases}
$$

**Corresponding weights**:

$$
w_i = \\begin{cases}
\\frac{1}{3} & i = 0 \\\\
\\frac{1}{18} & i = 1,...,6 \\\\
\\frac{1}{36} & i = 7,...,18
\\end{cases}
$$

**Lattice properties**:
- Lattice speed: $c = \\Delta x / \\Delta t = 1$
- Sound speed: $c_s = c/\\sqrt{3} = 1/\\sqrt{3}$
- Isotropy conditions satisfied up to 4th order

### 1.3 Equilibrium Distribution Function

The Maxwell-Boltzmann equilibrium distribution:

$$
f_i^{eq} = w_i \\rho \\left[ 1 + \\frac{\\mathbf{e}_i \\cdot \\mathbf{u}}{c_s^2} + \\frac{(\\mathbf{e}_i \\cdot \\mathbf{u})^2}{2c_s^4} - \\frac{\\mathbf{u} \\cdot \\mathbf{u}}{2c_s^2} \\right]
$$

**Moment constraints**:

$$
\\sum_{i=0}^{18} f_i^{eq} = \\rho
$$

$$
\\sum_{i=0}^{18} f_i^{eq} \\mathbf{e}_i = \\rho \\mathbf{u}
$$

$$
\\sum_{i=0}^{18} f_i^{eq} e_{i\\alpha} e_{i\\beta} = \\rho \\left( c_s^2 \\delta_{\\alpha\\beta} + u_\\alpha u_\\beta \\right)
$$

### 1.4 Macroscopic Variable Recovery

**Density**:
$$
\\rho(\\mathbf{x}, t) = \\sum_{i=0}^{18} f_i(\\mathbf{x}, t)
$$

**Momentum**:
$$
\\rho \\mathbf{u}(\\mathbf{x}, t) = \\sum_{i=0}^{18} f_i(\\mathbf{x}, t) \\mathbf{e}_i
$$

**Pressure** (ideal gas equation of state):
$$
p = \\rho c_s^2 = \\frac{\\rho}{3}
$$

**Kinematic viscosity**:
$$
\\nu = c_s^2 \\left(\\tau - \\frac{1}{2}\\right) \\Delta t
$$

### 1.5 Force Implementation (Guo Method)

External forces are incorporated using the Guo forcing scheme:

$$
f_i(\\mathbf{x} + \\mathbf{e}_i \\Delta t, t + \\Delta t) = f_i(\\mathbf{x}, t) + \\Omega_i + \\Delta t \\left(1 - \\frac{\\Delta t}{2\\tau}\\right) F_i
$$

**Force term**:
$$
F_i = w_i \\left[ \\frac{\\mathbf{e}_i - \\mathbf{u}}{c_s^2} + \\frac{(\\mathbf{e}_i \\cdot \\mathbf{u})\\mathbf{e}_i}{c_s^4} \\right] \\cdot \\mathbf{F}
$$

**Corrected momentum**:
$$
\\rho \\mathbf{u} = \\sum_{i=0}^{18} f_i \\mathbf{e}_i + \\frac{\\Delta t}{2} \\mathbf{F}
$$

## 2. Large Eddy Simulation

### 2.1 Smagorinsky Subgrid-Scale Model

The subgrid-scale viscosity is modeled as:

$$
\\nu_{sgs}(\\mathbf{x}, t) = (C_s \\Delta)^2 |\\tilde{\\mathbf{S}}|
$$

Where:
- $C_s = 0.18$: Smagorinsky constant (calibrated for isotropic turbulence)
- $\\Delta = (\\Delta x \\Delta y \\Delta z)^{1/3}$: Filter width
- $|\\tilde{\\mathbf{S}}| = \\sqrt{2 \\tilde{S}_{ij} \\tilde{S}_{ij}}$: Strain rate magnitude

### 2.2 Strain Rate Tensor

The resolved strain rate tensor:

$$
\\tilde{S}_{ij} = \\frac{1}{2} \\left( \\frac{\\partial \\tilde{u}_i}{\\partial x_j} + \\frac{\\partial \\tilde{u}_j}{\\partial x_i} \\right)
$$

**Finite difference implementation** (2nd order central differences):

$$
\\frac{\\partial \\tilde{u}_i}{\\partial x_j} \\approx \\frac{\\tilde{u}_i(j+1) - \\tilde{u}_i(j-1)}{2\\Delta x_j}
$$

### 2.3 LES-LBM Coupling

The effective relaxation time incorporates subgrid viscosity:

$$
\\tau_{eff} = \\frac{\\nu + \\nu_{sgs}}{c_s^2} + \\frac{1}{2}
$$

**Stability constraint**:
$$
\\tau_{eff} > 0.5 \\quad \\text{(numerical stability)}
$$

### 2.4 Subgrid-Scale Stress Tensor

The modeled SGS stress tensor:

$$
\\tau_{ij}^{sgs} = -2\\nu_{sgs} \\tilde{S}_{ij} + \\frac{1}{3}\\tau_{kk}^{sgs} \\delta_{ij}
$$

**Energy cascade representation**:
$$
\\varepsilon_{sgs} = \\nu_{sgs} |\\tilde{\\mathbf{S}}|^2
$$

## 3. Multiphase Flow Models

### 3.1 Shan-Chen Pseudopotential Model

The interparticle force for phase separation:

$$
\\mathbf{F}_{fluid}(\\mathbf{x}) = -\\psi(\\rho(\\mathbf{x})) \\sum_{i=1}^{18} w_i G \\psi(\\rho(\\mathbf{x} + \\mathbf{e}_i)) \\mathbf{e}_i
$$

**Pseudopotential function**:
$$
\\psi(\\rho) = \\sqrt{\\frac{2(p_{eos} - \\rho c_s^2)}{c_s^2 G}}
$$

**Equation of state** (Peng-Robinson):
$$
p_{eos} = \\frac{\\rho T}{1 - b\\rho} - \\frac{a\\rho^2}{1 + 2b\\rho - b^2\\rho^2}
$$

### 3.2 Surface Tension Calculation

The surface tension coefficient:

$$
\\sigma = \\frac{G}{6} \\left[ \\int_{\\rho_g}^{\\rho_l} \\psi(\\rho') \\sqrt{\\frac{2(p_{eos} - \\rho' c_s^2)}{c_s^2 G}} d\\rho' \\right]^2
$$

**Simplified form for weak segregation**:
$$
\\sigma \\approx \\frac{G}{6} \\psi^2(\\rho_l) (\\rho_l - \\rho_g)^2
$$

### 3.3 Interface Thickness

The interface thickness scales as:

$$
\\xi = \\sqrt{\\frac{\\kappa}{G(\\rho_l - \\rho_g)^2}}
$$

Where $\\kappa$ is the gradient energy coefficient.

### 3.4 Contact Angle Implementation

Wetting boundary conditions are imposed through:

$$
\\mathbf{F}_{wall} = -G_{wall} \\psi(\\rho) \\sum_{i} w_i s_i \\mathbf{e}_i
$$

Where $s_i = 1$ if $\\mathbf{x} + \\mathbf{e}_i$ is a solid node, $s_i = 0$ otherwise.

**Contact angle relation**:
$$
\\cos \\theta_c = \\frac{G_{wall}}{G} \\sqrt{\\frac{\\psi(\\rho_s)}{\\psi(\\rho_l)}}
$$

## 4. Particle Dynamics

### 4.1 Newton's Equation of Motion

For each coffee particle $p$:

$$
m_p \\frac{d\\mathbf{v}_p}{dt} = \\mathbf{F}_{drag} + \\mathbf{F}_{gravity} + \\mathbf{F}_{buoyancy} + \\mathbf{F}_{collision}
$$

### 4.2 Drag Force Model

**Stokes drag** (low Reynolds number):
$$
\\mathbf{F}_{drag} = 6\\pi \\mu R_p (\\mathbf{u}_f - \\mathbf{v}_p)
$$

**General drag formulation**:
$$
\\mathbf{F}_{drag} = \\frac{1}{2} C_D \\rho_f A_p |\\mathbf{u}_f - \\mathbf{v}_p| (\\mathbf{u}_f - \\mathbf{v}_p)
$$

**Drag coefficient correlation** (Schiller-Naumann):
$$
C_D = \\begin{cases}
\\frac{24}{Re_p}(1 + 0.15 Re_p^{0.687}) & Re_p \\leq 1000 \\\\
0.44 & Re_p > 1000
\\end{cases}
$$

**Particle Reynolds number**:
$$
Re_p = \\frac{\\rho_f |\\mathbf{u}_f - \\mathbf{v}_p| d_p}{\\mu_f}
$$

### 4.3 Gravitational and Buoyancy Forces

**Gravity**:
$$
\\mathbf{F}_{gravity} = m_p \\mathbf{g} = \\frac{\\pi d_p^3}{6} \\rho_p \\mathbf{g}
$$

**Buoyancy** (Archimedes principle):
$$
\\mathbf{F}_{buoyancy} = -\\frac{\\pi d_p^3}{6} \\rho_f \\mathbf{g}
$$

**Net gravitational force**:
$$
\\mathbf{F}_{net} = \\frac{\\pi d_p^3}{6} (\\rho_p - \\rho_f) \\mathbf{g}
$$

### 4.4 Particle-Particle Collisions

**Soft-sphere model**:
$$
\\mathbf{F}_{collision} = k_n \\delta \\mathbf{n} + \\gamma_n \\mathbf{v}_{rel} \\cdot \\mathbf{n}
$$

Where:
- $k_n$: Normal spring constant
- $\\delta$: Overlap distance
- $\\mathbf{n}$: Unit normal vector
- $\\gamma_n$: Damping coefficient

### 4.5 Two-Way Coupling

**Fluid momentum source**:
$$
\\mathbf{S}_f(\\mathbf{x}) = \\sum_p \\frac{\\mathbf{F}_{p \\rightarrow f}}{V_{cell}} W(\\mathbf{x} - \\mathbf{x}_p)
$$

**Interpolation weight**:
$$
W(\\mathbf{r}) = \\begin{cases}
\\frac{1}{8}(1 + r_x)(1 + r_y)(1 + r_z) & |r_i| \\leq 1 \\\\
0 & \\text{otherwise}
\\end{cases}
$$

## 5. Porous Media Flow

### 5.1 Darcy-Forchheimer Equation

For flow through coffee bed:

$$
-\\nabla p = \\frac{\\mu}{k} \\mathbf{u} + \\frac{\\rho C_F}{\\sqrt{k}} |\\mathbf{u}| \\mathbf{u}
$$

**Permeability** (Kozeny-Carman relation):
$$
k = \\frac{\\varepsilon^3 d_p^2}{180(1-\\varepsilon)^2}
$$

**Forchheimer coefficient**:
$$
C_F = \\frac{1.75}{\\sqrt{150}} \\frac{(1-\\varepsilon)}{\\varepsilon^3}
$$

### 5.2 Porosity Models

**Uniform porosity**:
$$
\\varepsilon = \\varepsilon_0 = 0.4 \\quad \\text{(typical for coffee)
$$

**Depth-dependent porosity**:
$$
\\varepsilon(z) = \\varepsilon_0 + (\\varepsilon_{max} - \\varepsilon_0) e^{-z/L_c}
$$

### 5.3 LBM Implementation

**Momentum source term**:
$$
\\mathbf{F}_{porous} = -\\frac{\\nu \\rho}{k} \\mathbf{u} - \\frac{\\rho C_F}{\\sqrt{k}} |\\mathbf{u}| \\mathbf{u}
$$

**Relaxation time modification**:
$$
\\tau_{porous} = \\frac{1}{2} + \\frac{\\nu_{eff}}{c_s^2}
$$

Where:
$$
\\nu_{eff} = \\nu + \\frac{k}{\\varepsilon} + \\frac{C_F \\sqrt{k}}{\\varepsilon} |\\mathbf{u}|
$$

## 6. Boundary Conditions

### 6.1 Velocity Inlet (Zou-He Method)

For velocity inlet at top boundary ($z = N_z - 1$):

**Known**: $\\rho$, $u_x$, $u_y$, $u_z$
**Unknown**: $f_5$, $f_{11}$, $f_{12}$, $f_{13}$, $f_{14}$

**Density from non-equilibrium bounce-back**:
$$
\\rho = \\frac{1}{1 - u_z} \\left[ f_0 + f_1 + f_2 + f_3 + f_4 + 2(f_6 + f_9 + f_{10} + f_{15} + f_{16}) \\right]
$$

**Missing distributions**:
$$
f_5 = f_6 - \\frac{2\\rho u_z}{3}
$$

$$
f_{11} = f_{10} - \\frac{\\rho u_z}{6} + \\frac{\\rho u_x}{2} - \\frac{1}{2}(f_1 - f_2)
$$

### 6.2 Pressure Outlet

For pressure outlet at bottom boundary:

**Known**: $p = \\rho c_s^2$
**Unknown**: $\\mathbf{u}$, missing distributions

**Extrapolation scheme**:
$$
f_i(x, y, 0) = f_i(x, y, 1) + (f_i(x, y, 1) - f_i(x, y, 2))
$$

**Pressure correction**:
$$
f_i^{corrected} = f_i + (\\rho_{target} - \\rho_{current}) \\frac{f_i^{eq}}{\\rho_{current}}
$$

### 6.3 No-Slip Walls (Bounce-Back)

**Standard bounce-back**:
$$
f_{\\bar{i}}(\\mathbf{x}_f, t + \\Delta t) = f_i(\\mathbf{x}_f, t)
$$

Where $\\bar{i}$ is the direction opposite to $i$.

**Moving wall bounce-back**:
$$
f_{\\bar{i}}(\\mathbf{x}_f, t + \\Delta t) = f_i(\\mathbf{x}_f, t) - 2w_i \\rho \\frac{\\mathbf{e}_i \\cdot \\mathbf{u}_{wall}}{c_s^2}
$$

### 6.4 Curved Boundary Interpolation

For complex geometries, the interpolated bounce-back:

$$
f_{\\bar{i}}(\\mathbf{x}_f, t + \\Delta t) = q f_i(\\mathbf{x}_f, t) + (1-q) f_{\\bar{i}}(\\mathbf{x}_f, t)
$$

Where $q$ is the fraction of the link inside the fluid domain.

## 7. Numerical Stability Analysis

### 7.1 CFL Condition

The Courant-Friedrichs-Lewy stability condition:

$$
CFL = \\frac{u_{max} \\Delta t}{\\Delta x} \\leq CFL_{max}
$$

**For LBM**: $CFL_{max} = 1$ (theoretical), $CFL \\leq 0.1$ (practical)

### 7.2 Viscous Stability

**Viscous time step constraint**:
$$
\\Delta t \\leq \\frac{\\Delta x^2}{2\\nu}
$$

**Relaxation time constraint**:
$$
\\tau > 0.5 \\quad \\text{(BGK stability)}
$$

### 7.3 Mach Number Limitation

**Compressibility constraint**:
$$
Ma = \\frac{u_{max}}{c_s} \\leq 0.1
$$

**Pressure variation limit**:
$$
\\frac{\\Delta p}{p_0} \\leq 0.01
$$

### 7.4 Grid Reynolds Number

**Lattice Reynolds number**:
$$
Re_{lattice} = \\frac{u_{max} L_{char}}{\\nu} \\leq 100
$$

Where $L_{char}$ is the characteristic length in lattice units.

### 7.5 Time Step Selection

**Optimal time step** balancing accuracy and stability:

$$
\\Delta t = \\min \\left\\{ \\frac{\\Delta x}{u_{max}}, \\frac{\\Delta x^2}{2\\nu}, \\frac{\\Delta x}{c_s Ma_{max}} \\right\\}
$$

### 7.6 Convergence Criteria

**Density convergence**:
$$
\\max_{\\mathbf{x}} \\left| \\frac{\\rho^{n+1}(\\mathbf{x}) - \\rho^n(\\mathbf{x})}{\\rho^n(\\mathbf{x})} \\right| < \\epsilon_{\\rho}
$$

**Velocity convergence**:
$$
\\max_{\\mathbf{x}} \\left| \\mathbf{u}^{n+1}(\\mathbf{x}) - \\mathbf{u}^n(\\mathbf{x}) \\right| < \\epsilon_u
$$

**Typical tolerances**: $\\epsilon_{\\rho} = 10^{-6}$, $\\epsilon_u = 10^{-6}$

---

**Mathematical Notation Conventions**:
- Bold symbols ($\\mathbf{u}$): Vectors
- Greek letters ($\\rho$, $\\nu$, $\\tau$): Physical properties
- Subscripts ($i$, $p$, $f$): Component or phase identifiers
- Superscripts ($n$, $eq$): Time level or equilibrium state
- Tildes ($\\tilde{u}$): Filtered (LES) quantities