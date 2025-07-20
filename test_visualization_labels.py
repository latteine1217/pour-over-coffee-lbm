# test_visualization_labels.py
"""
Test visualization with proper English labels and good layout
Demonstrates best practices for labels, titles, axes in scientific plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Ensure English-compatible font

def create_sample_lbm_data():
    """Create sample LBM simulation data for testing"""
    nx, ny = 128, 128
    
    # Create sample density field (water droplet)
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y)
    
    # Water density field (Gaussian droplet)
    density = 1.0 + 0.5 * np.exp(-(X**2 + Y**2)/0.5)
    
    # Velocity field (circular flow)
    u_velocity = -Y * np.exp(-(X**2 + Y**2)/2)
    v_velocity = X * np.exp(-(X**2 + Y**2)/2)
    velocity_magnitude = np.sqrt(u_velocity**2 + v_velocity**2)
    
    return X, Y, density, u_velocity, v_velocity, velocity_magnitude

def test_density_field_plot():
    """Test density field visualization with proper English labels"""
    print("=== Testing Density Field Visualization ===")
    
    X, Y, density, _, _, _ = create_sample_lbm_data()
    
    # Create figure with proper size and DPI
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # Create contour plot with good colormap
    contour = ax.contourf(X, Y, density, levels=20, cmap='Blues')
    
    # Add colorbar with proper label
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Density (kg/m³)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Set axes labels with units
    ax.set_xlabel('X Position (cm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (cm)', fontsize=12, fontweight='bold')
    
    # Set title with descriptive information
    ax.set_title('LBM Density Field - Pour-Over Coffee Simulation\nWater Distribution in V60 Dripper', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set aspect ratio
    ax.set_aspect('equal')
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig('test_density_field.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Density field plot saved as 'test_density_field.png'")

def test_velocity_field_plot():
    """Test velocity field visualization with proper English labels"""
    print("=== Testing Velocity Field Visualization ===")
    
    X, Y, _, u_velocity, v_velocity, velocity_magnitude = create_sample_lbm_data()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
    
    # Velocity magnitude contour
    contour1 = ax1.contourf(X, Y, velocity_magnitude, levels=20, cmap='Reds')
    cbar1 = plt.colorbar(contour1, ax=ax1, shrink=0.8)
    cbar1.set_label('Velocity Magnitude (m/s)', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('X Position (cm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y Position (cm)', fontsize=11, fontweight='bold')
    ax1.set_title('Velocity Magnitude\nWater Flow Speed', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Velocity vector field
    skip = 8  # Skip points for cleaner vector display
    ax2.contourf(X, Y, velocity_magnitude, levels=20, cmap='Reds', alpha=0.6)
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              u_velocity[::skip, ::skip], v_velocity[::skip, ::skip],
              velocity_magnitude[::skip, ::skip], cmap='Reds', scale=5)
    
    ax2.set_xlabel('X Position (cm)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y Position (cm)', fontsize=11, fontweight='bold')
    ax2.set_title('Velocity Vector Field\nWater Flow Direction', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Main title for the entire figure
    fig.suptitle('LBM Velocity Field Analysis - Coffee Pour-Over Simulation', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for main title
    
    plt.savefig('test_velocity_field.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Velocity field plot saved as 'test_velocity_field.png'")

def test_performance_comparison_chart():
    """Test performance comparison chart with proper layout"""
    print("=== Testing Performance Comparison Chart ===")
    
    # Sample performance data
    backends = ['CPU\n(Single Core)', 'CPU\n(Multi Core)', 'Metal GPU\n(M1 Pro)', 'Metal GPU\n(M2 Max)']
    throughput = [12.5, 28.3, 15.2, 22.1]  # MLUPs
    memory_usage = [2.1, 3.8, 4.2, 5.1]  # GB
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    
    # Throughput comparison
    bars1 = ax1.bar(backends, throughput, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'], 
                    alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Throughput (MLUPs)', fontsize=12, fontweight='bold')
    ax1.set_title('D3Q19 LBM Performance Comparison\nComputational Throughput', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(throughput) * 1.2)
    
    # Add value labels on bars
    for bar, value in zip(bars1, throughput):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Memory usage comparison
    bars2 = ax2.bar(backends, memory_usage, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'], 
                    alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('Memory Usage (GB)', fontsize=12, fontweight='bold')
    ax2.set_title('Memory Consumption\nSimulation Requirements', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(memory_usage) * 1.2)
    
    # Add value labels on bars
    for bar, value in zip(bars2, memory_usage):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
    
    # Main title and layout
    fig.suptitle('Pour-Over Coffee LBM Simulation - Backend Performance Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    plt.savefig('test_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Performance comparison chart saved as 'test_performance_comparison.png'")

def test_longitudinal_slice_plot():
    """Test longitudinal slice visualization (XZ and YZ planes)"""
    print("=== Testing Longitudinal Slice Visualization ===")
    
    # Create 3D-like data for longitudinal slices
    nx, nz = 128, 64
    x = np.linspace(0, 10, nx)  # X position (cm)
    z = np.linspace(0, 15, nz)  # Z height (cm) 
    X, Z = np.meshgrid(x, z)
    
    # Simulate water flow from top to bottom (gravity effect)
    water_density = np.exp(-(X-5)**2/4) * np.exp(-(Z-12)**2/8) * (1 + 0.3*np.sin(Z))
    velocity_z = -0.5 * np.exp(-(X-5)**2/4) * (1 - Z/15)  # Downward velocity
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    
    # XZ plane (side view) - Water density
    contour1 = ax1.contourf(X, Z, water_density, levels=20, cmap='Blues')
    cbar1 = plt.colorbar(contour1, ax=ax1, shrink=0.8)
    cbar1.set_label('Water Density (kg/m³)', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('X Position (cm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Z Height (cm)', fontsize=12, fontweight='bold')
    ax1.set_title('XZ Plane - Side View\nWater Distribution (Gravity Direction)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add gravity arrow
    ax1.annotate('Gravity', xy=(8, 10), xytext=(8, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=11, fontweight='bold', color='red')
    
    # XZ plane (side view) - Velocity field
    contour2 = ax2.contourf(X, Z, np.abs(velocity_z), levels=20, cmap='Reds')
    cbar2 = plt.colorbar(contour2, ax=ax2, shrink=0.8)
    cbar2.set_label('Velocity Magnitude (m/s)', fontsize=11, fontweight='bold')
    
    # Add velocity vectors
    skip = 6
    ax2.quiver(X[::skip, ::skip], Z[::skip, ::skip], 
              np.zeros_like(velocity_z[::skip, ::skip]), velocity_z[::skip, ::skip],
              scale=3, alpha=0.7, color='darkred')
    
    ax2.set_xlabel('X Position (cm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Z Height (cm)', fontsize=12, fontweight='bold')
    ax2.set_title('XZ Plane - Side View\nVertical Velocity Field', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('Longitudinal Cross-Section Analysis - Pour-Over Coffee Flow\nVertical Water Movement Through Coffee Bed', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    plt.savefig('test_longitudinal_slice.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Longitudinal slice plot saved as 'test_longitudinal_slice.png'")

def main():
    """Run all visualization tests"""
    print("🧪 Testing Visualization Labels and Layout")
    print("=" * 50)
    
    try:
        test_density_field_plot()
        test_velocity_field_plot()
        test_performance_comparison_chart()
        test_longitudinal_slice_plot()
        
        print("\n" + "=" * 50)
        print("✅ All visualization tests completed successfully!")
        print("✅ All plots use proper English labels")
        print("✅ Layout optimized for readability")
        print("✅ High-quality output with proper DPI")
        print("\nGenerated files:")
        print("  - test_density_field.png")
        print("  - test_velocity_field.png") 
        print("  - test_performance_comparison.png")
        print("  - test_longitudinal_slice.png")
        
    except Exception as e:
        print(f"❌ Error in visualization test: {e}")

if __name__ == "__main__":
    main()