import openmc
import numpy as np
import matplotlib.pyplot as plt
import os

# Set path to your run directory
BASE_DIR = '/home/cade/Desktop/OpenMC/SeniorDesign/MicroHTGR_Output/htgr_run_02.03.2026_23.31.58_SingleRun'
batch_number = 100
target_power_MW = 15.0  # Target reactor power
IS_WEDGE_GEOMETRY = False  # Set to True if using 1/6 geometry

def get_normalization_factor(sp_path):
    """
    Calculate manual power normalization factor
    
    Parameters:
    -----------
    sp_path : str
        Path to statepoint file
    
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
    
    # # If wedge geometry, multiply by 6 to get full core normalization
    # if IS_WEDGE_GEOMETRY:
    #     source_per_sec *= 6
    
    return source_per_sec


def reconstruct_full_core(data_wedge, mesh):
    """
    Reconstruct full core data from 1/6 wedge by rotating 6 times
    
    Parameters:
    -----------
    data_wedge : ndarray
        3D array of wedge data (nx, ny, nz)
    mesh : openmc.RegularMesh
        Mesh object
        
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
    
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    # Create full core grid with higher resolution
    core_radius = max(abs(mesh.lower_left[0]), abs(mesh.upper_right[0]))
    n_full = max(nx, ny) * 2
    x_full = np.linspace(-core_radius, core_radius, n_full)
    y_full = np.linspace(-core_radius, core_radius, n_full)
    X_full, Y_full = np.meshgrid(x_full, y_full, indexing='ij')
    
    # Initialize full core data
    data_full = np.zeros((n_full, n_full, nz))
    
    # For each z-level
    for z_idx in range(nz):
        data_slice = data_wedge[:, :, z_idx]
        
        # Rotate and fill each of 6 sectors
        for sector in range(6):
            angle = sector * 60.0  # degrees
            angle_rad = np.radians(angle)
            
            # Rotation matrix
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Rotate coordinates
            X_rot = cos_a * X - sin_a * Y
            Y_rot = sin_a * X + cos_a * Y
            
            # Interpolate wedge data onto rotated positions in full grid
            for i in range(n_full):
                for j in range(n_full):
                    x_point = X_full[i, j]
                    y_point = Y_full[i, j]
                    
                    # Rotate point back to wedge frame
                    x_wedge = cos_a * x_point + sin_a * y_point
                    y_wedge = -sin_a * x_point + cos_a * y_point
                    
                    # Check if point is in wedge region (0° to 60°)
                    r = np.sqrt(x_point**2 + y_point**2)
                    if r < core_radius and r > 0:
                        angle_point = np.arctan2(y_point, x_point)
                        # Normalize to [0, 2π]
                        if angle_point < 0:
                            angle_point += 2 * np.pi
                        
                        # Map to wedge sector
                        sector_angle = (angle_point - angle_rad) % (2 * np.pi)
                        
                        if 0 <= sector_angle <= np.radians(60):
                            # Interpolate from wedge data
                            if (mesh.lower_left[0] <= x_wedge <= mesh.upper_right[0] and
                                mesh.lower_left[1] <= y_wedge <= mesh.upper_right[1]):
                                
                                # Find nearest neighbor (simple interpolation)
                                i_wedge = np.argmin(np.abs(x_centers - x_wedge))
                                j_wedge = np.argmin(np.abs(y_centers - y_wedge))
                                
                                data_full[i, j, z_idx] = data_slice[i_wedge, j_wedge]
    
    return data_full, x_full, y_full


def plot_htgr_xyslice(batch, z_index, reconstruct=False):
    """
    Plot XY slice of flux and fission for HTGR at given axial index
    With manual power normalization
    
    Parameters:
    -----------
    batch : int
        Batch number for statepoint file
    z_index : int
        Axial index to plot (0 to n_ax_zones-1)
    reconstruct : bool
        If True and using wedge geometry, reconstruct full core
    """
    
    sp_path = os.path.join(BASE_DIR, f'statepoint.{batch}.h5')
    sp = openmc.StatePoint(sp_path)
    
    # Get normalization factor
    source_per_sec = get_normalization_factor(sp_path)
    
    # Get mesh tally
    tally = sp.get_tally(name='mesh_rates')
    mesh = tally.find_filter(openmc.MeshFilter).mesh
    
    nx, ny, nz = mesh.dimension
    scores = tally.scores
    
    flux_idx = scores.index('flux')
    fission_idx = scores.index('fission')
    
    # Get mean values (shape: [n_mesh_cells, n_scores])
    mean = tally.mean[:, 0, :]
    
    # Extract flux and fission (per source particle)
    flux_per_source = mean[:, flux_idx]
    fission_per_source = mean[:, fission_idx]
    
    # APPLY MANUAL NORMALIZATION
    flux = flux_per_source * source_per_sec  # n/(cm²·s)
    fission = fission_per_source * source_per_sec  # fissions/s
    
    # Reshape to 3D grid (Fortran order to match OpenMC convention)
    flux_3d = flux.reshape((nx, ny, nz), order='F')
    fission_3d = fission.reshape((nx, ny, nz), order='F')
    
    if IS_WEDGE_GEOMETRY and reconstruct:
        # Reconstruct full core
        flux_3d_full, x_full, y_full = reconstruct_full_core(flux_3d, mesh)
        fission_3d_full, _, _ = reconstruct_full_core(fission_3d, mesh)
        
        flux_xy = flux_3d_full[:, :, z_index].T
        fission_xy = fission_3d_full[:, :, z_index].T
        x_edges = np.concatenate([x_full, [x_full[-1] + (x_full[-1] - x_full[-2])]])
        y_edges = np.concatenate([y_full, [y_full[-1] + (y_full[-1] - y_full[-2])]])
        core_radius = max(abs(mesh.lower_left[0]), abs(mesh.upper_right[0]))
        geometry_label = "Full Core (Reconstructed)"
    else:
        # Use wedge data directly
        flux_xy = flux_3d[:, :, z_index].T
        fission_xy = fission_3d[:, :, z_index].T
        x_edges = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx + 1)
        y_edges = np.linspace(mesh.lower_left[1], mesh.upper_right[1], ny + 1)
        core_radius = max(abs(mesh.lower_left[0]), abs(mesh.upper_right[0]))
        geometry_label = "1/6 Wedge" if IS_WEDGE_GEOMETRY else "Full Core"
    
    # Get min/max for consistent color scaling
    flux_min, flux_max = np.min(flux_xy[flux_xy > 0]), flux_xy.max()
    fission_min, fission_max = np.min(fission_xy[fission_xy > 0]), fission_xy.max()
    
    # Plot flux
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    pcm = ax.pcolormesh(x_edges, y_edges, flux_xy, shading='auto', cmap='hot', 
                        vmin=0, vmax=flux_max)
    cbar = plt.colorbar(pcm, ax=ax, label='Flux [n/(cm² · s)]')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_title(f'Neutron Flux - {geometry_label} - Axial Level {z_index} (Batch {batch})\n{target_power_MW} MW')
    ax.set_aspect('equal')
    
    # Add circle to show core boundary
    circle = plt.Circle((0, 0), core_radius, fill=False, 
                       edgecolor='white', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(circle)
    
    # Add wedge boundaries if showing wedge
    if IS_WEDGE_GEOMETRY and not reconstruct:
        ax.plot([0, core_radius], [0, 0], 'w--', linewidth=1.5, alpha=0.7, label='Wedge boundary')
        ax.plot([0, core_radius * np.cos(np.radians(60))], 
                [0, core_radius * np.sin(np.radians(60))], 'w--', linewidth=1.5, alpha=0.7)
        ax.legend()
    
    suffix = '_reconstructed' if (IS_WEDGE_GEOMETRY and reconstruct) else '_wedge' if IS_WEDGE_GEOMETRY else ''
    save_path = os.path.join(BASE_DIR, f'batch{batch}_flux_xy_z{z_index}_normalized{suffix}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved flux plot: {save_path}")
    
    # Plot fission
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    pcm = ax.pcolormesh(x_edges, y_edges, fission_xy, shading='auto', cmap='hot',
                        vmin=0, vmax=fission_max)
    cbar = plt.colorbar(pcm, ax=ax, label='Fission Rate [fissions/s]')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_title(f'Fission Rate - {geometry_label} - Axial Level {z_index} (Batch {batch})\n{target_power_MW} MW')
    ax.set_aspect('equal')
    
    # Add circle to show core boundary
    circle = plt.Circle((0, 0), core_radius, fill=False, 
                       edgecolor='white', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.add_patch(circle)
    
    # Add wedge boundaries if showing wedge
    if IS_WEDGE_GEOMETRY and not reconstruct:
        ax.plot([0, core_radius], [0, 0], 'w--', linewidth=1.5, alpha=0.7, label='Wedge boundary')
        ax.plot([0, core_radius * np.cos(np.radians(60))], 
                [0, core_radius * np.sin(np.radians(60))], 'w--', linewidth=1.5, alpha=0.7)
        ax.legend()
    
    save_path = os.path.join(BASE_DIR, f'batch{batch}_fission_xy_z{z_index}_normalized{suffix}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved fission plot: {save_path}")

def plot_htgr_axial_crosssection(batch, angle_deg=0, use_full_mesh=True):
    """
    Plot axial cross-section (RZ slice) through the core
    Shows full height including reflectors
    
    Parameters:
    -----------
    batch : int
        Batch number for statepoint file
    angle_deg : float
        Angle (degrees) for cross-section plane (0 = along x-axis)
    use_full_mesh : bool
        If True, use 'mesh_rates_full' tally (includes reflectors)
        If False, use 'mesh_rates' tally (active core only)
    """
    
    sp_path = os.path.join(BASE_DIR, f'statepoint.{batch}.h5')
    sp = openmc.StatePoint(sp_path)
    
    # Get normalization factor
    source_per_sec = get_normalization_factor(sp_path)
    
    # Get mesh tally - choose full or active core mesh
    tally_name = 'mesh_rates_full' if use_full_mesh else 'mesh_rates'
    try:
        tally = sp.get_tally(name=tally_name)
    except KeyError:
        print(f"Warning: Tally '{tally_name}' not found. Using 'mesh_rates' instead.")
        tally = sp.get_tally(name='mesh_rates')
    
    mesh = tally.find_filter(openmc.MeshFilter).mesh
    
    nx, ny, nz = mesh.dimension
    scores = tally.scores
    
    flux_idx = scores.index('flux')
    fission_idx = scores.index('fission')
    
    mean = tally.mean[:, 0, :]
    flux_per_source = mean[:, flux_idx]
    fission_per_source = mean[:, fission_idx]
    
    # APPLY MANUAL NORMALIZATION
    flux = flux_per_source * source_per_sec
    fission = fission_per_source * source_per_sec
    
    # Reshape to 3D
    flux_3d = flux.reshape((nx, ny, nz), order='F')
    fission_3d = fission.reshape((nx, ny, nz), order='F')
    
    # Create coordinate arrays
    x_edges = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx + 1)
    y_edges = np.linspace(mesh.lower_left[1], mesh.upper_right[1], ny + 1)
    z_edges = np.linspace(mesh.lower_left[2], mesh.upper_right[2], nz + 1)
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    
    # Extract data along the specified angle
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Create RZ slice
    n_r = nx
    r_max = max(abs(mesh.lower_left[0]), abs(mesh.upper_right[0]))
    r_points = np.linspace(-r_max, r_max, n_r)
    
    flux_rz = np.zeros((len(r_points), nz))
    fission_rz = np.zeros((len(r_points), nz))
    
    for i, r in enumerate(r_points):
        x_point = r * cos_a
        y_point = r * sin_a
        
        # Find nearest mesh indices
        if mesh.lower_left[0] <= x_point <= mesh.upper_right[0] and \
           mesh.lower_left[1] <= y_point <= mesh.upper_right[1]:
            i_x = np.argmin(np.abs(x_centers - x_point))
            i_y = np.argmin(np.abs(y_centers - y_point))
            
            flux_rz[i, :] = flux_3d[i_x, i_y, :]
            fission_rz[i, :] = fission_3d[i_x, i_y, :]
    
    # Create meshgrid for plotting
    R, Z = np.meshgrid(r_points, z_centers, indexing='ij')
    
    # Calculate active core boundaries for reference lines
    # You'll need to pass these from your params or read from geometry
    active_core_bottom = mesh.lower_left[2] + (mesh.upper_right[2] - mesh.lower_left[2]) * 0.2  # Approximate
    active_core_top = mesh.upper_right[2] - (mesh.upper_right[2] - mesh.lower_left[2]) * 0.2     # Approximate
    
    # Plot flux cross-section
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    pcm = ax.pcolormesh(R, Z, flux_rz, shading='auto', cmap='hot')
    cbar = plt.colorbar(pcm, ax=ax, label='Flux [n/(cm² · s)]')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_xlabel('Radial Position [cm]')
    ax.set_ylabel('Axial Position [cm]')
    ax.set_title(f'Neutron Flux - Full Height Axial Cross-Section at {angle_deg}° (Batch {batch})\n{target_power_MW} MW')
    ax.set_aspect('equal')
    
    # Add reference lines
    ax.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.5, label='Core centerline')
    ax.axhline(y=active_core_bottom, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7, label='Active core boundary')
    ax.axhline(y=active_core_top, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.legend(loc='upper right')
    
    save_path = os.path.join(BASE_DIR, f'batch{batch}_flux_rz_fullheight_angle{angle_deg}_normalized.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved full-height flux RZ plot: {save_path}")
    
    # Plot fission cross-section
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    pcm = ax.pcolormesh(R, Z, fission_rz, shading='auto', cmap='hot')
    cbar = plt.colorbar(pcm, ax=ax, label='Fission Rate [fissions/s]')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_xlabel('Radial Position [cm]')
    ax.set_ylabel('Axial Position [cm]')
    ax.set_title(f'Fission Rate - Full Height Axial Cross-Section at {angle_deg}° (Batch {batch})\n{target_power_MW} MW')
    ax.set_aspect('equal')
    
    # Add reference lines
    ax.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.5, label='Core centerline')
    ax.axhline(y=active_core_bottom, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7, label='Active core boundary')
    ax.axhline(y=active_core_top, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.legend(loc='upper right')
    
    save_path = os.path.join(BASE_DIR, f'batch{batch}_fission_rz_fullheight_angle{angle_deg}_normalized.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved full-height fission RZ plot: {save_path}")

def plot_htgr_axial_profile(batch):
    """
    Plot axial profiles of flux and fission rate
    With manual power normalization
    
    Parameters:
    -----------
    batch : int
        Batch number for statepoint file
    """
    
    sp_path = os.path.join(BASE_DIR, f'statepoint.{batch}.h5')
    sp = openmc.StatePoint(sp_path)
    
    # Get normalization factor
    source_per_sec = get_normalization_factor(sp_path)
    
    # Get mesh tally
    tally = sp.get_tally(name='mesh_rates')
    mesh = tally.find_filter(openmc.MeshFilter).mesh
    
    nx, ny, nz = mesh.dimension
    scores = tally.scores
    
    flux_idx = scores.index('flux')
    fission_idx = scores.index('fission')
    
    mean = tally.mean[:, 0, :]
    flux_per_source = mean[:, flux_idx]
    fission_per_source = mean[:, fission_idx]
    
    # APPLY MANUAL NORMALIZATION
    flux = flux_per_source * source_per_sec
    fission = fission_per_source * source_per_sec
    
    # Reshape to 3D
    flux_3d = flux.reshape((nx, ny, nz), order='F')
    fission_3d = fission.reshape((nx, ny, nz), order='F')
    
    # Average over XY plane for each Z level
    flux_axial = flux_3d.mean(axis=(0, 1))
    fission_axial = fission_3d.mean(axis=(0, 1))
    
    # Z coordinates (centers of mesh cells)
    z_coords = np.linspace(mesh.lower_left[2], mesh.upper_right[2], nz + 1)
    z_centers = (z_coords[:-1] + z_coords[1:]) / 2
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    geometry_label = "1/6 Wedge" if IS_WEDGE_GEOMETRY else "Full Core"
    
    ax1.plot(z_centers, flux_axial, 'b-', linewidth=2)
    ax1.set_xlabel('Axial Position [cm]')
    ax1.set_ylabel('Average Flux [n/(cm² · s)]')
    ax1.set_title(f'Axial Flux Profile - {geometry_label} (Batch {batch})\n{target_power_MW} MW')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    ax2.plot(z_centers, fission_axial, 'r-', linewidth=2)
    ax2.set_xlabel('Axial Position [cm]')
    ax2.set_ylabel('Average Fission Rate [fissions/s]')
    ax2.set_title(f'Axial Fission Profile - {geometry_label} (Batch {batch})\n{target_power_MW} MW')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, f'batch{batch}_axial_profiles_normalized.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved axial profile: {save_path}")


def print_global_rates(batch):
    """
    Print global reaction rates with manual power normalization
    
    Parameters:
    -----------
    batch : int
        Batch number for statepoint file
    """
    
    sp_path = os.path.join(BASE_DIR, f'statepoint.{batch}.h5')
    sp = openmc.StatePoint(sp_path)
    
    # Get normalization factor
    heating_tally = sp.get_tally(name='heating')
    heating_rate_ev = heating_tally.mean[0, 0, 0]
    
    joule_per_ev = 1.60218e-19
    heating_rate_j = heating_rate_ev * joule_per_ev
    
    power_watts = target_power_MW * 1e6
    source_per_sec = power_watts / heating_rate_j
    
    # # Multiply by 6 for wedge geometry
    # if IS_WEDGE_GEOMETRY:
    #     source_per_sec *= 6
    
    # Get global tally
    tally = sp.get_tally(name='global_rates')
    scores = tally.scores
    
    flux_idx = scores.index('flux')
    fission_idx = scores.index('fission')
    nu_fission_idx = scores.index('nu-fission')
    
    mean = tally.mean[0, 0, :]
    
    # Per source particle
    total_flux_per_source = mean[flux_idx]
    total_fission_per_source = mean[fission_idx]
    total_nu_fission_per_source = mean[nu_fission_idx]
    
    # APPLY MANUAL NORMALIZATION
    total_flux = total_flux_per_source * source_per_sec
    total_fission = total_fission_per_source * source_per_sec
    total_nu_fission = total_nu_fission_per_source * source_per_sec
    
    # Calculate power from fission rate (verification)
    energy_per_fission = 200e6 * joule_per_ev
    power_watts_calc = total_fission * energy_per_fission
    power_mw = power_watts_calc / 1e6
    
    # Calculate power from heating (should match target exactly)
    power_from_heating_watts = heating_rate_ev * joule_per_ev * source_per_sec  # W
    power_from_heating_MW = power_from_heating_watts / 1e6  # Convert to MW
    
    geometry_label = "1/6 WEDGE GEOMETRY" if IS_WEDGE_GEOMETRY else "FULL CORE GEOMETRY"
    
    print("\n" + '='*80)
    print(f"GLOBAL REACTION RATES (Batch {batch}) - {geometry_label}")
    print('='*80)
    print(f"k-effective: {sp.keff.nominal_value:.5f} ± {sp.keff.std_dev:.5f}")
    print(f"\nNormalization:")
    print(f"  Heating rate: {heating_rate_ev:.3e} eV/source")
    print(f"  Source rate: {source_per_sec:.3e} source/s")
    if IS_WEDGE_GEOMETRY:
        print(f"  (Multiplied by 6 for full core equivalence)")
    print(f"\nReaction Rates:")
    print(f"  Total Flux: {total_flux:.3e} n/(cm² · s)")
    print(f"  Total Fission Rate: {total_fission:.3e} fissions/s")
    print(f"  Total Nu-Fission Rate: {total_nu_fission:.3e} neutrons/s")
    print(f"\nPower:")
    print(f"  Target Power: {target_power_MW:.3f} MW")
    print(f"  Calculated Power (from fission): {power_mw:.3f} MW")
    print(f"  Calculated Power (from heating): {power_from_heating_MW:.3f} MW")
    print('='*80 + "\n")


# Example usage:
if __name__ == "__main__":
      
    # Print global rates
    print_global_rates(batch_number)
    
    # Plot axial profiles
    plot_htgr_axial_profile(batch_number)
    
    # Plot axial cross-sections (RZ plane)
    # Show cross-section at 0° (along x-axis)
    plot_htgr_axial_crosssection(batch_number, 0, True)
    
    # If wedge geometry, also show at 30° (middle of wedge)
    if IS_WEDGE_GEOMETRY:
        plot_htgr_axial_crosssection(batch_number, 30, True)
    
    # Plot XY slices at different axial levels
    n_ax_zones = 50  # Match your params["n_ax_zones"]
    z_indices = [0, 6, n_ax_zones//4, n_ax_zones//2, 3*n_ax_zones//4, n_ax_zones-1]
    
    for z_idx in z_indices:
        # Plot wedge view
        plot_htgr_xyslice(batch_number, z_idx, reconstruct=False)
        
        # Also plot reconstructed full core if using wedge geometry
        if IS_WEDGE_GEOMETRY:
            plot_htgr_xyslice(batch_number, z_idx, reconstruct=True)