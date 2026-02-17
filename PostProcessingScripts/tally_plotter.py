"""
Tally Plotter Post-Processing Script
Generates flux and fission rate plots from OpenMC simulation results.
Fixed for 1/6 geometry support with efficient reconstruction.
"""

import openmc
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator


def get_normalization_factor(sp_path, target_power_MW=15.0):
    """
    Calculate manual power normalization factor
    
    Parameters:
    -----------
    sp_path : str
        Path to statepoint file
    target_power_MW : float
        Target reactor power in MW
    
    Returns:
    --------
    float : source_per_sec normalization factor
    """
    sp = openmc.StatePoint(sp_path)
    
    # Get heating tally
    heating_tally = sp.get_tally(name='heating')
    heating_rate_ev = heating_tally.mean[0, 0, 0]
    
    # Convert to J/source
    joule_per_ev = 1.60218e-19
    heating_rate_j = heating_rate_ev * joule_per_ev
    
    # Calculate source rate
    power_watts = target_power_MW * 1e6
    source_per_sec = power_watts / heating_rate_j
    
    return source_per_sec


def reconstruct_full_core_vectorized(data_wedge, mesh, core_radius):
    """
    Efficiently reconstruct full core data from 1/6 wedge by rotating 6 times.
    The wedge spans from 0° to 60° in the first quadrant.
    
    Parameters:
    -----------
    data_wedge : ndarray
        3D array of wedge data (nx, ny, nz)
    mesh : openmc.RegularMesh
        Mesh object for the wedge
    core_radius : float
        Core radius in cm
        
    Returns:
    --------
    tuple : (data_full, x_full, y_full) - reconstructed data and coordinates
    """
    nx, ny, nz = data_wedge.shape
    
    # Create coordinate grids for original wedge
    x_edges = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx + 1)
    y_edges = np.linspace(mesh.lower_left[1], mesh.upper_right[1], ny + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Create full core grid - make it square centered at origin
    n_full = max(nx, ny) * 2
    full_extent = core_radius * 1.05  # Slightly larger to capture edges
    x_full = np.linspace(-full_extent, full_extent, n_full)
    y_full = np.linspace(-full_extent, full_extent, n_full)
    
    # Initialize full core data
    data_full = np.zeros((n_full, n_full, nz))
    
    # Create interpolator for wedge data
    for z_idx in range(nz):
        data_slice = data_wedge[:, :, z_idx]
        data_clean = np.nan_to_num(data_slice, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            interp = RegularGridInterpolator(
                (x_centers, y_centers), 
                data_clean,
                method='nearest',
                bounds_error=False,
                fill_value=0.0
            )
        except ValueError:
            continue
        
        # For each point in the full grid
        for i, x in enumerate(x_full):
            for j, y in enumerate(y_full):
                r = np.sqrt(x**2 + y**2)
                
                if r > core_radius or r < 1e-6:
                    continue
                
                # Get angle in [0, 2π]
                theta = np.arctan2(y, x)
                if theta < 0:
                    theta += 2 * np.pi
                
                # Map angle to wedge sector [0, π/3]
                # Find which 60° sector this point is in
                sector = int(theta / (np.pi / 3)) % 6
                theta_in_wedge = theta - sector * (np.pi / 3)
                
                # Convert back to Cartesian in wedge frame
                x_wedge = r * np.cos(theta_in_wedge)
                y_wedge = r * np.sin(theta_in_wedge)
                
                # Check if within mesh bounds
                if (mesh.lower_left[0] <= x_wedge <= mesh.upper_right[0] and
                    mesh.lower_left[1] <= y_wedge <= mesh.upper_right[1]):
                    
                    value = interp([[x_wedge, y_wedge]])[0]
                    data_full[i, j, z_idx] = value
    
    return data_full, x_full, y_full


def plot_xy_slice(run_dir, batch, z_index, is_wedge=False, reconstruct=False, 
                  target_power_MW=15.0, core_radius=90.0, n_ax_zones=50):
    """
    Plot XY slice of flux and fission at given axial index.
    
    Parameters:
    -----------
    run_dir : str
        Directory containing simulation results
    batch : int
        Batch number for statepoint file
    z_index : int
        Axial index to plot
    is_wedge : bool
        Whether using 1/6 geometry
    reconstruct : bool
        If True and is_wedge, reconstruct full core
    target_power_MW : float
        Target reactor power for normalization
    core_radius : float
        Core radius in cm
    n_ax_zones : int
        Number of axial zones
    """
    
    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    
    if not os.path.exists(sp_path):
        print(f"Warning: Statepoint file not found: {sp_path}")
        return
    
    sp = openmc.StatePoint(sp_path)
    
    # Get normalization factor
    source_per_sec = get_normalization_factor(sp_path, target_power_MW)
    
    # Get mesh tally
    tally = sp.get_tally(name='mesh_rates')
    mesh = tally.find_filter(openmc.MeshFilter).mesh
    
    nx, ny, nz = mesh.dimension
    scores = tally.scores
    
    flux_idx = scores.index('flux')
    fission_idx = scores.index('fission')
    
    # Get mean values
    mean = tally.mean[:, 0, :]
    
    # Extract and normalize
    flux = mean[:, flux_idx] * source_per_sec
    fission = mean[:, fission_idx] * source_per_sec
    
    # Reshape to 3D grid
    flux_3d = flux.reshape((nx, ny, nz), order='F')
    fission_3d = fission.reshape((nx, ny, nz), order='F')
    
    if is_wedge and reconstruct:
        # Reconstruct full core using efficient vectorized method
        print(f"  Reconstructing full core for z_index={z_index}...")
        flux_3d_full, x_full, y_full = reconstruct_full_core_vectorized(flux_3d, mesh, core_radius)
        fission_3d_full, _, _ = reconstruct_full_core_vectorized(fission_3d, mesh, core_radius)
        
        flux_xy = flux_3d_full[:, :, z_index].T
        fission_xy = fission_3d_full[:, :, z_index].T
        
        dx = x_full[1] - x_full[0]
        dy = y_full[1] - y_full[0]
        x_edges = np.concatenate([x_full - dx/2, [x_full[-1] + dx/2]])
        y_edges = np.concatenate([y_full - dy/2, [y_full[-1] + dy/2]])
        geometry_label = "Full Core (Reconstructed)"
    else:
        # Use data directly - mesh bounds reflect actual geometry
        flux_xy = flux_3d[:, :, z_index].T
        fission_xy = fission_3d[:, :, z_index].T
        x_edges = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx + 1)
        y_edges = np.linspace(mesh.lower_left[1], mesh.upper_right[1], ny + 1)
        geometry_label = "1/6 Wedge" if is_wedge else "Full Core"
    
    # Calculate z position for title
    z_edges = np.linspace(mesh.lower_left[2], mesh.upper_right[2], nz + 1)
    z_center = (z_edges[z_index] + z_edges[z_index + 1]) / 2
    
    # Get color scale limits (excluding zeros)
    flux_nonzero = flux_xy[flux_xy > 0]
    fission_nonzero = fission_xy[fission_xy > 0]
    
    flux_max = flux_xy.max() if len(flux_nonzero) > 0 else 1
    fission_max = fission_xy.max() if len(fission_nonzero) > 0 else 1
    
    # Determine filename suffix
    if is_wedge and reconstruct:
        suffix = '_reconstructed'
    elif is_wedge:
        suffix = '_wedge'
    else:
        suffix = ''
    
    # Plot flux
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    pcm = ax.pcolormesh(x_edges, y_edges, flux_xy, shading='auto', cmap='hot', 
                        vmin=0, vmax=flux_max)
    cbar = plt.colorbar(pcm, ax=ax, label='Flux [n/(cm² · s)]')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_title(f'Neutron Flux - {geometry_label}\nZ = {z_center:.1f} cm (index {z_index}), {target_power_MW} MW')
    ax.set_aspect('equal')
    
    save_path = os.path.join(run_dir, f'batch{batch}_flux_xy_z{z_index}{suffix}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Plot fission
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    pcm = ax.pcolormesh(x_edges, y_edges, fission_xy, shading='auto', cmap='hot',
                        vmin=0, vmax=fission_max)
    cbar = plt.colorbar(pcm, ax=ax, label='Fission Rate [fissions/s]')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_title(f'Fission Rate - {geometry_label}\nZ = {z_center:.1f} cm (index {z_index}), {target_power_MW} MW')
    ax.set_aspect('equal')
    
    save_path = os.path.join(run_dir, f'batch{batch}_fission_xy_z{z_index}{suffix}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_axial_profile(run_dir, batch, is_wedge=False, target_power_MW=15.0):
    """
    Plot axial profiles of flux and fission rate.
    
    Parameters:
    -----------
    run_dir : str
        Directory containing simulation results
    batch : int
        Batch number
    is_wedge : bool
        Whether using 1/6 geometry
    target_power_MW : float
        Target reactor power
    """
    
    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    
    if not os.path.exists(sp_path):
        print(f"Warning: Statepoint file not found: {sp_path}")
        return
    
    sp = openmc.StatePoint(sp_path)
    source_per_sec = get_normalization_factor(sp_path, target_power_MW)
    
    tally = sp.get_tally(name='mesh_rates')
    mesh = tally.find_filter(openmc.MeshFilter).mesh
    
    nx, ny, nz = mesh.dimension
    scores = tally.scores
    
    flux_idx = scores.index('flux')
    fission_idx = scores.index('fission')
    
    mean = tally.mean[:, 0, :]
    flux = mean[:, flux_idx] * source_per_sec
    fission = mean[:, fission_idx] * source_per_sec
    
    # Reshape to 3D
    flux_3d = flux.reshape((nx, ny, nz), order='F')
    fission_3d = fission.reshape((nx, ny, nz), order='F')
    
    # Average over XY plane
    flux_axial = flux_3d.mean(axis=(0, 1))
    fission_axial = fission_3d.mean(axis=(0, 1))
    
    # Z coordinates
    z_coords = np.linspace(mesh.lower_left[2], mesh.upper_right[2], nz + 1)
    z_centers = (z_coords[:-1] + z_coords[1:]) / 2
    
    geometry_label = "1/6 Wedge" if is_wedge else "Full Core"
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    ax1.plot(z_centers, flux_axial, 'b-', linewidth=2)
    ax1.set_xlabel('Axial Position [cm]')
    ax1.set_ylabel('Average Flux [n/(cm² · s)]')
    ax1.set_title(f'Axial Flux Profile - {geometry_label}\n{target_power_MW} MW')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    ax2.plot(z_centers, fission_axial, 'r-', linewidth=2)
    ax2.set_xlabel('Axial Position [cm]')
    ax2.set_ylabel('Average Fission Rate [fissions/s]')
    ax2.set_title(f'Axial Fission Profile - {geometry_label}\n{target_power_MW} MW')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    save_path = os.path.join(run_dir, f'batch{batch}_axial_profiles.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_rz_crosssection(run_dir, batch, angle_deg=0, is_wedge=False, 
                         target_power_MW=15.0, include_reflector=True):
    """
    Plot RZ cross-section at specified angle.
    
    Parameters:
    -----------
    run_dir : str
        Directory containing simulation results
    batch : int
        Batch number
    angle_deg : float
        Angle for cross-section (degrees)
    is_wedge : bool
        Whether using 1/6 geometry
    target_power_MW : float
        Target reactor power
    include_reflector : bool
        Whether to include axial reflector regions
    """
    
    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    
    if not os.path.exists(sp_path):
        print(f"Warning: Statepoint file not found: {sp_path}")
        return
    
    sp = openmc.StatePoint(sp_path)
    source_per_sec = get_normalization_factor(sp_path, target_power_MW)
    
    # Use full mesh if including reflector
    tally_name = 'mesh_rates_full' if include_reflector else 'mesh_rates'
    
    try:
        tally = sp.get_tally(name=tally_name)
    except KeyError:
        print(f"Warning: Tally '{tally_name}' not found, using 'mesh_rates'")
        tally = sp.get_tally(name='mesh_rates')
    
    mesh = tally.find_filter(openmc.MeshFilter).mesh
    
    nx, ny, nz = mesh.dimension
    scores = tally.scores
    
    flux_idx = scores.index('flux')
    fission_idx = scores.index('fission')
    
    mean = tally.mean[:, 0, :]
    flux = mean[:, flux_idx] * source_per_sec
    fission = mean[:, fission_idx] * source_per_sec
    
    # Reshape to 3D
    flux_3d = flux.reshape((nx, ny, nz), order='F')
    fission_3d = fission.reshape((nx, ny, nz), order='F')
    
    # Extract RZ slice at specified angle
    x_centers = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx)
    y_centers = np.linspace(mesh.lower_left[1], mesh.upper_right[1], ny)
    z_centers = np.linspace(mesh.lower_left[2], mesh.upper_right[2], nz)
    
    angle_rad = np.radians(angle_deg)
    
    # For each radial position, find the nearest (x, y) along the angle
    r_max = max(abs(mesh.lower_left[0]), abs(mesh.upper_right[0]),
                abs(mesh.lower_left[1]), abs(mesh.upper_right[1]))
    r_vals = np.linspace(0, r_max, nx)
    
    flux_rz = np.zeros((len(r_vals), nz))
    fission_rz = np.zeros((len(r_vals), nz))
    
    for i, r in enumerate(r_vals):
        x_target = r * np.cos(angle_rad)
        y_target = r * np.sin(angle_rad)
        
        # Check if within mesh bounds
        if (mesh.lower_left[0] <= x_target <= mesh.upper_right[0] and
            mesh.lower_left[1] <= y_target <= mesh.upper_right[1]):
            ix = np.argmin(np.abs(x_centers - x_target))
            iy = np.argmin(np.abs(y_centers - y_target))
            
            flux_rz[i, :] = flux_3d[ix, iy, :]
            fission_rz[i, :] = fission_3d[ix, iy, :]
    
    # Create meshgrid for plotting
    R, Z = np.meshgrid(r_vals, z_centers, indexing='ij')
    
    geometry_label = "1/6 Wedge" if is_wedge else "Full Core"
    
    # Calculate proper figure size based on data aspect ratio
    r_range = r_vals[-1] - r_vals[0]
    z_range = z_centers[-1] - z_centers[0]
    
    # Width is fixed, height scales with aspect ratio
    fig_width = 6
    fig_height = max(4, min(12, fig_width * (z_range / r_range) * 0.7))
    
    # Plot flux
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    pcm = ax.pcolormesh(R, Z, flux_rz, shading='auto', cmap='hot')
    cbar = plt.colorbar(pcm, ax=ax, label='Flux [n/(cm² · s)]')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_xlabel('Radial Position [cm]')
    ax.set_ylabel('Axial Position [cm]')
    ax.set_title(f'Flux - RZ Cross-Section at {angle_deg}° - {geometry_label}\n{target_power_MW} MW')
    
    save_path = os.path.join(run_dir, f'batch{batch}_flux_rz_angle{angle_deg}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Plot fission
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    pcm = ax.pcolormesh(R, Z, fission_rz, shading='auto', cmap='hot')
    cbar = plt.colorbar(pcm, ax=ax, label='Fission Rate [fissions/s]')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_xlabel('Radial Position [cm]')
    ax.set_ylabel('Axial Position [cm]')
    ax.set_title(f'Fission Rate - RZ Cross-Section at {angle_deg}° - {geometry_label}\n{target_power_MW} MW')
    
    save_path = os.path.join(run_dir, f'batch{batch}_fission_rz_angle{angle_deg}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_global_rates(run_dir, batch, is_wedge=False, target_power_MW=15.0):
    """
    Print and save global reaction rates.
    
    Parameters:
    -----------
    run_dir : str
        Directory containing simulation results
    batch : int
        Batch number
    is_wedge : bool
        Whether using 1/6 geometry
    target_power_MW : float
        Target reactor power
    """
    
    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    
    if not os.path.exists(sp_path):
        print(f"Warning: Statepoint file not found: {sp_path}")
        return
    
    sp = openmc.StatePoint(sp_path)
    
    # Get heating tally
    heating_tally = sp.get_tally(name='heating')
    heating_rate_ev = heating_tally.mean[0, 0, 0]
    
    joule_per_ev = 1.60218e-19
    heating_rate_j = heating_rate_ev * joule_per_ev
    
    power_watts = target_power_MW * 1e6
    source_per_sec = power_watts / heating_rate_j
    
    # Get global tally
    tally = sp.get_tally(name='global_rates')
    scores = tally.scores
    
    flux_idx = scores.index('flux')
    fission_idx = scores.index('fission')
    nu_fission_idx = scores.index('nu-fission')
    
    mean = tally.mean[0, 0, :]
    
    total_flux = mean[flux_idx] * source_per_sec
    total_fission = mean[fission_idx] * source_per_sec
    total_nu_fission = mean[nu_fission_idx] * source_per_sec
    
    # Calculate power from fission rate
    energy_per_fission = 200e6 * joule_per_ev
    power_mw = (total_fission * energy_per_fission) / 1e6
    
    geometry_label = "1/6 WEDGE GEOMETRY" if is_wedge else "FULL CORE GEOMETRY"
    
    output = []
    output.append('='*80)
    output.append(f"GLOBAL REACTION RATES (Batch {batch}) - {geometry_label}")
    output.append('='*80)
    output.append(f"k-effective: {sp.keff.nominal_value:.5f} ± {sp.keff.std_dev:.5f}")
    output.append(f"\nNormalization:")
    output.append(f"   Heating rate: {heating_rate_ev:.3e} eV/source")
    output.append(f"   Source rate: {source_per_sec:.3e} source/s")
    output.append(f"\nReaction Rates:")
    output.append(f"   Total Flux: {total_flux:.3e} n/(cm^2·s)")
    output.append(f"   Total Fission Rate: {total_fission:.3e} fissions/s")
    output.append(f"   Total Nu-Fission Rate: {total_nu_fission:.3e} neutrons/s")
    output.append(f"\nPower:")
    output.append(f"   Target Power: {target_power_MW:.3f} MW")
    output.append(f"   Calculated Power (from fission): {power_mw:.3f} MW")
    output.append('='*80 + "\n")
    
    # Print to console
    for line in output:
        print(line)
    
    # Save to file
    save_path = os.path.join(run_dir, f'batch{batch}_global_rates.txt')
    with open(save_path, 'w') as f:
        f.write('\n'.join(output))
    print(f"Saved: {save_path}")

def run_tally_plots(run_dir, params, batch=None):
    """
    Run all tally plotting for a simulation.
    
    Parameters:
    -----------
    run_dir : str
        Directory containing simulation results
    params : dict
        Simulation parameters
    batch : int, optional
        Batch number. If None, finds automatically.
    """
    
    print(f"\n{'='*80}")
    print("TALLY PLOTTING")
    print(f"{'='*80}")
    print(f"Run directory: {run_dir}")
    
    # Find batch number if not provided
    if batch is None:
        for f in os.listdir(run_dir):
            if f.startswith('statepoint') and f.endswith('.h5'):
                batch = int(f.split('.')[1])
                break
    
    if batch is None:
        print("ERROR: No statepoint file found!")
        return
    
    print(f"Batch number: {batch}")
    
    is_wedge = params.get("use_1/6_geometry", False)
    target_power = params.get("thermal_power", 15.0)
    core_radius = params.get("core_radius", 90.0)
    n_ax_zones = params.get("n_ax_zones", 50)
    
    print(f"Geometry: {'1/6 Wedge' if is_wedge else 'Full Core'}")
    print(f"Target power: {target_power} MW")
    print(f"{'='*80}")
    
    # Print global rates
    print("\nGenerating global rates...\n")
    print_global_rates(run_dir, batch, is_wedge, target_power)
    
    # Plot axial profiles
    print("\nGenerating axial profiles...")
    plot_axial_profile(run_dir, batch, is_wedge, target_power)
    
    # Plot RZ cross-sections (only at 0° - fuel channels exist along x-axis)
    print("\nGenerating RZ cross-sections...")
    plot_rz_crosssection(run_dir, batch, 0, is_wedge, target_power, True)
    
    # Plot XY slices at different axial levels
    print("\nGenerating XY slices...")
    z_indices = [0, n_ax_zones//4, n_ax_zones//2, 3*n_ax_zones//4, n_ax_zones-1]
    
    for z_idx in z_indices:
        # Always plot wedge view (or full core if not wedge geometry)
        plot_xy_slice(run_dir, batch, z_idx, is_wedge, reconstruct=False, 
                      target_power_MW=target_power, core_radius=core_radius, 
                      n_ax_zones=n_ax_zones)
        
        # # For wedge geometry, also plot reconstructed full core
        # if is_wedge:
        #     plot_xy_slice(run_dir, batch, z_idx, is_wedge, reconstruct=True,
        #                   target_power_MW=target_power, core_radius=core_radius,
        #                   n_ax_zones=n_ax_zones)
    
    print(f"\n{'='*80}")
    print("TALLY PLOTTING COMPLETE")
    print(f"{'='*80}\n")

def load_params_from_run_dir(run_dir):
    """
    Load parameters from run_params.json in the run directory.
    
    Returns:
        dict: Parameters, or None if not found
    """
    import json
    
    params_path = os.path.join(run_dir, 'run_params.json')
    
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        # Remove n_trisos if present (not needed for plotting)
        params.pop('n_trisos', None)
        return params
    
    return None

def detect_geometry_from_statepoint(run_dir, batch=None):
    """
    Auto-detect geometry parameters from statepoint file.
    Falls back to this if run_params.json is not available.
    
    Returns:
        dict: Detected parameters
    """
    # Find batch number if not provided
    if batch is None:
        for f in os.listdir(run_dir):
            if f.startswith('statepoint') and f.endswith('.h5'):
                batch = int(f.split('.')[1])
                break
    
    if batch is None:
        return None
    
    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    sp = openmc.StatePoint(sp_path)
    
    # Get mesh to detect geometry
    tally = sp.get_tally(name='mesh_rates')
    mesh = tally.find_filter(openmc.MeshFilter).mesh
    
    # Detect if wedge geometry based on mesh bounds
    # Wedge: x starts at 0, y starts at 0
    # Full: x and y are symmetric around 0
    is_wedge = mesh.lower_left[0] >= -0.1 and mesh.lower_left[1] >= -0.1
    
    # Core radius is max of upper bounds
    core_radius = max(mesh.upper_right[0], mesh.upper_right[1])
    
    # Number of axial zones
    n_ax_zones = mesh.dimension[2]
    
    return {
        "use_1/6_geometry": is_wedge,
        "thermal_power": 15.0,
        "core_radius": core_radius,
        "n_ax_zones": n_ax_zones
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tally_plotter.py <run_directory> [batch_number]")
        print("\nThe script will load parameters from run_params.json in the run directory.")
        print("If not found, it will auto-detect geometry from the statepoint file.")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"\nProcessing: {run_dir}")
    
    # Try to load from run_params.json first
    params = load_params_from_run_dir(run_dir)
    
    if params is not None:
        print("\nLoaded parameters from run_params.json")
        print(f"  Geometry: {'1/6 Wedge' if params.get('use_1/6_geometry', False) else 'Full Core'}")
        print(f"  Core radius: {params.get('core_radius', 'unknown')} cm")
        print(f"  Axial zones: {params.get('n_ax_zones', 'unknown')}")
    else:
        print("run_params.json not found, detecting from statepoint...")
        params = detect_geometry_from_statepoint(run_dir, batch)
        
        if params is None:
            print("ERROR: Could not load or detect parameters.")
            sys.exit(1)
        
        print(f"  Detected geometry: {'1/6 Wedge' if params['use_1/6_geometry'] else 'Full Core'}")
        print(f"  Core radius: {params['core_radius']:.1f} cm")
        print(f"  Axial zones: {params['n_ax_zones']}")
    
    run_tally_plots(run_dir, params, batch)