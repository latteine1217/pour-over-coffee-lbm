# gui_viewer.py
"""
Taichi GUI實時可視化器
提供交互式3D切片查看和實時模擬
"""

import taichi as ti
import numpy as np
import config
from init import initialize_d3q19_simulation
from main import perform_optimized_step

def run_realtime_simulation():
    """Run real-time simulation with visualization"""
    print("=== Taichi GUI Real-time Coffee Simulation ===")
    
    # Initialize simulation
    lbm_solver, multiphase, porous_solver, visualizer = initialize_d3q19_simulation()
    
    # Create GUI
    gui = ti.GUI("D3Q19 Coffee Simulation - Real-time View", res=(config.NX, config.NY))
    
    # Simulation parameters
    step_count = 0
    z_slice = config.NZ // 2
    field_type = 0  # 0=density, 1=velocity, 2=phase, 3=porosity
    auto_run = False
    steps_per_frame = 1
    
    # Display field names
    field_names = ["Density Field", "Velocity Field", "Phase Field", "Porosity"]
    
    print(f"Initialization complete! Grid: {config.NX}×{config.NY}×{config.NZ}")
    print("Control instructions:")
    print("  SPACE: Pause/Resume auto run")
    print("  UP/DOWN arrows: Switch Z slice")
    print("  C key: Switch display field")
    print("  S key: Single step execution")
    print("  R key: Reset simulation")
    print("  F key: Toggle fast mode (multiple steps per frame)")
    print("  ESC: Exit")
    
    while gui.running:
        # Handle user input
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == ti.GUI.SPACE:
                auto_run = not auto_run
                print(f"Auto run: {'ON' if auto_run else 'OFF'}")
            elif e.key == ti.GUI.UP and z_slice < config.NZ - 1:
                z_slice += 1
                print(f"Slice: Z={z_slice}")
            elif e.key == ti.GUI.DOWN and z_slice > 0:
                z_slice -= 1
                print(f"Slice: Z={z_slice}")
            elif e.key == 'c':
                field_type = (field_type + 1) % 4
                print(f"Display field: {field_names[field_type]}")
            elif e.key == 's':
                # Single step execution
                perform_optimized_step(lbm_solver, multiphase, porous_solver, step_count)
                step_count += 1
                print(f"Single step: Step {step_count}")
            elif e.key == 'r':
                # Reset simulation
                print("Resetting simulation...")
                lbm_solver, multiphase, porous_solver, visualizer = initialize_d3q19_simulation()
                step_count = 0
                print("Simulation reset")
            elif e.key == 'f':
                steps_per_frame = 3 if steps_per_frame == 1 else 1
                print(f"Steps per frame: {steps_per_frame}")
        
        # Auto run simulation
        if auto_run:
            for _ in range(steps_per_frame):
                perform_optimized_step(lbm_solver, multiphase, porous_solver, step_count)
                step_count += 1
        
        # Extract and display slice
        visualizer.extract_z_slice(z_slice, field_type)
        
        # Get image data
        image_data = visualizer.slice_field.to_numpy()
        
        # Set image
        gui.set_image(image_data)
        
        # Display status information
        gui.text(f"Step: {step_count}", pos=(0.05, 0.95), color=0xFFFFFF)
        gui.text(f"Slice: Z={z_slice}/{config.NZ-1}", pos=(0.05, 0.90), color=0xFFFFFF)
        gui.text(f"Field: {field_names[field_type]}", pos=(0.05, 0.85), color=0xFFFFFF)
        gui.text(f"Mode: {'Auto' if auto_run else 'Manual'}", pos=(0.05, 0.80), color=0xFFFFFF)
        
        # Display control hints
        gui.text("SPACE=Auto S=Step C=Field", pos=(0.05, 0.15), color=0x888888)
        gui.text("UP/DOWN=Slice F=Fast R=Reset", pos=(0.05, 0.10), color=0x888888)
        gui.text("ESC=Exit", pos=(0.05, 0.05), color=0x888888)
        
        gui.show()

def run_longitudinal_viewer():
    """Run longitudinal cross-section viewer - showing water flow from top to bottom"""
    print("=== Taichi GUI Longitudinal Flow Visualization ===")
    
    # Initialize simulation
    lbm_solver, multiphase, porous_solver, visualizer = initialize_d3q19_simulation()
    
    # Run some steps to generate flow data
    print("Pre-running simulation to generate longitudinal flow data...")
    for step in range(30):
        perform_optimized_step(lbm_solver, multiphase, porous_solver, step)
        if step % 15 == 0:
            print(f"Completed {step} steps")
    
    # Create GUI - using longitudinal resolution
    gui = ti.GUI("Coffee Simulation - Longitudinal Flow View", res=(config.NX, config.NZ))
    
    # Slice parameters
    slice_pos = config.NY // 2  # Middle slice in Y direction
    slice_type = 0  # 0=XZ plane, 1=YZ plane  
    field_type = 0  # Display field type
    
    field_names = ["Density", "Velocity", "Phase", "Porosity"]
    slice_names = ["XZ Plane (Side View)", "YZ Plane (Front View)"]
    
    print("Longitudinal flow viewer controls:")
    print("  LEFT/RIGHT arrows: Switch slice position")
    print("  V key: Switch view angle (XZ Side ↔ YZ Front)")
    print("  C key: Switch display field")
    print("  S key: Save current longitudinal view")
    print("  ESC: Exit")
    
    while gui.running:
        # Handle events
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == ti.GUI.LEFT:
                if slice_type == 0:  # XZ plane
                    slice_pos = max(0, slice_pos - 1)
                else:  # YZ plane
                    slice_pos = max(0, slice_pos - 1)
            elif e.key == ti.GUI.RIGHT:
                if slice_type == 0:  # XZ plane
                    slice_pos = min(config.NY - 1, slice_pos + 1)
                else:  # YZ plane  
                    slice_pos = min(config.NX - 1, slice_pos + 1)
            elif e.key == 'v':
                slice_type = 1 - slice_type
                # Reset slice position
                slice_pos = (config.NY // 2) if slice_type == 0 else (config.NX // 2)
                print(f"Switched to: {slice_names[slice_type]}")
            elif e.key == 'c':
                field_type = (field_type + 1) % 4
                print(f"Display field: {field_names[field_type]}")
            elif e.key == 's':
                if slice_type == 0:
                    visualizer.create_xz_composite_slice(slice_pos)
                    filename = f"longitudinal_xz_y{slice_pos:03d}.png"
                else:
                    visualizer.create_yz_composite_slice(slice_pos)
                    filename = f"longitudinal_yz_x{slice_pos:03d}.png"
                print(f"Saved: {filename}")
        
        # Extract and display longitudinal slice
        if slice_type == 0:  # XZ plane (side view)
            visualizer.extract_xz_slice(slice_pos, field_type)
            image_data = visualizer.xz_slice_field.to_numpy()
        else:  # YZ plane (front view)
            visualizer.extract_yz_slice(slice_pos, field_type)  
            image_data = visualizer.yz_slice_field.to_numpy()
        
        # Flip Y axis to show top as up (gravity direction)
        image_data_flipped = np.flipud(image_data)
        
        # Display image
        gui.set_image(image_data_flipped)
        
        # Display information
        gui.text(f"View: {slice_names[slice_type]}", pos=(0.05, 0.95), color=0xFFFFFF)
        if slice_type == 0:
            gui.text(f"Y Slice: {slice_pos}/{config.NY-1}", pos=(0.05, 0.90), color=0xFFFFFF)
        else:
            gui.text(f"X Slice: {slice_pos}/{config.NX-1}", pos=(0.05, 0.90), color=0xFFFFFF)
        gui.text(f"Field: {field_names[field_type]}", pos=(0.05, 0.85), color=0xFFFFFF)
        
        # Coordinate axis labels
        if slice_type == 0:
            gui.text("X-axis →", pos=(0.85, 0.05), color=0x888888)
            gui.text("↑ Z-axis (Height)", pos=(0.05, 0.50), color=0x888888)
        else:
            gui.text("Y-axis →", pos=(0.85, 0.05), color=0x888888)  
            gui.text("↑ Z-axis (Height)", pos=(0.05, 0.50), color=0x888888)
        
        # Control hints
        gui.text("LEFT/RIGHT=Slice V=View C=Field", pos=(0.05, 0.15), color=0x888888)
        gui.text("S=Save ESC=Exit", pos=(0.05, 0.10), color=0x888888)
        
        gui.show()
def run_composite_viewer():
    """Run composite view viewer"""
    print("=== Taichi GUI Composite View Viewer ===")
    
    # Initialize simulation
    lbm_solver, multiphase, porous_solver, visualizer = initialize_d3q19_simulation()
    
    # Run a few steps to generate some data
    print("Pre-running simulation to generate visualization data...")
    for step in range(20):
        perform_optimized_step(lbm_solver, multiphase, porous_solver, step)
        if step % 10 == 0:
            print(f"Completed {step} steps")
    
    # Create GUI
    gui = ti.GUI("D3Q19 Coffee Simulation - Composite View", res=(config.NX, config.NY))
    z_slice = config.NZ // 2
    
    print("Composite view controls:")
    print("  UP/DOWN arrows: Switch Z slice") 
    print("  S key: Save current slice")
    print("  ESC: Exit")
    
    while gui.running:
        # Handle events
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == ti.GUI.UP and z_slice < config.NZ - 1:
                z_slice += 1
            elif e.key == ti.GUI.DOWN and z_slice > 0:
                z_slice -= 1
            elif e.key == 's':
                # Save current view
                visualizer.save_3d_snapshot("gui_snapshot", z_slice)
                print(f"Saved slice Z={z_slice}")
        
        # Create composite slice
        visualizer.create_composite_slice(z_slice)
        
        # Compute statistics
        visualizer.compute_statistics()
        stats_np = visualizer.stats.to_numpy()
        
        # Display image
        image_data = visualizer.color_slice.to_numpy()
        gui.set_image(image_data)
        
        # Display information
        gui.text(f"Slice: Z={z_slice}/{config.NZ-1}", pos=(0.05, 0.95), color=0xFFFFFF)
        gui.text(f"Water Phase: {stats_np[0]:.1%}", pos=(0.05, 0.90), color=0xFFFFFF)
        gui.text(f"Avg Kinetic Energy: {stats_np[1]:.2e}", pos=(0.05, 0.85), color=0xFFFFFF)
        gui.text(f"Max Velocity: {stats_np[2]:.2e}", pos=(0.05, 0.80), color=0xFFFFFF)
        
        # Color encoding description
        gui.text("Red=Water Green=Velocity Blue=Structure", pos=(0.05, 0.15), color=0x888888)
        gui.text("UP/DOWN=Slice S=Save ESC=Exit", pos=(0.05, 0.10), color=0x888888)
        
        gui.show()

def run_slice_animation():
    """Run slice animation viewer"""
    print("=== Taichi GUI Slice Animation Viewer ===")
    
    # Initialize simulation
    lbm_solver, multiphase, porous_solver, visualizer = initialize_d3q19_simulation()
    
    # Run some steps
    print("Pre-running 30 steps...")
    for step in range(30):
        perform_optimized_step(lbm_solver, multiphase, porous_solver, step)
    
    # Create GUI
    gui = ti.GUI("D3Q19 Coffee Simulation - Slice Animation", res=(config.NX, config.NY))
    
    z_slice = 0
    direction = 1
    field_type = 0
    auto_animate = True
    
    field_names = ["Density", "Velocity", "Phase", "Porosity"]
    
    print("Animation controls:")
    print("  SPACE: Pause/Resume animation")
    print("  C: Switch field type") 
    print("  ESC: Exit")
    
    while gui.running:
        # Handle events
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == ti.GUI.SPACE:
                auto_animate = not auto_animate
            elif e.key == 'c':
                field_type = (field_type + 1) % 4
        
        # Auto slice animation
        if auto_animate:
            z_slice += direction
            if z_slice >= config.NZ - 1:
                direction = -1
                z_slice = config.NZ - 1
            elif z_slice <= 0:
                direction = 1
                z_slice = 0
        
        # Extract slice
        visualizer.extract_z_slice(z_slice, field_type)
        
        # Display
        gui.set_image(visualizer.slice_field.to_numpy())
        
        # Information display
        gui.text(f"Slice: Z={z_slice}/{config.NZ-1}", pos=(0.05, 0.95), color=0xFFFFFF)
        gui.text(f"Field: {field_names[field_type]}", pos=(0.05, 0.90), color=0xFFFFFF)
        gui.text(f"Animation: {'Running' if auto_animate else 'Paused'}", pos=(0.05, 0.85), color=0xFFFFFF)
        
        gui.show()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "realtime":
            run_realtime_simulation()
        elif mode == "composite":
            run_composite_viewer()
        elif mode == "animation":
            run_slice_animation()
        elif mode == "longitudinal":
            run_longitudinal_viewer()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: realtime, composite, animation, longitudinal")
    else:
        print("Available modes:")
        print("  python gui_viewer.py realtime      - Real-time simulation")
        print("  python gui_viewer.py composite     - Composite view")
        print("  python gui_viewer.py animation     - Slice animation")
        print("  python gui_viewer.py longitudinal  - Longitudinal flow view 🌊")