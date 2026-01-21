import openmc
import numpy as np
import matplotlib.pyplot as plt
import os

# Set path to your run directory
BASE_DIR = '/home/cade/Desktop/OpenMC/SeniorDesign/MicroHTGR_Output/htgr_run_10.37.14_01.21.2026_SingleRun'
batch_number = 300
target_power_MW = 15.0  # Target reactor power

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
    
    return source_per_sec


def plot_htgr_xyslice(batch, z_index):
    """
    Plot XY slice of flux and fission for HTGR at given axial index
    With manual power normalization
    
    Parameters:
    -----------
    batch : int
        Batch number for statepoint file
    z_index : int
        Axial index to plot (0 to n_ax_zones-1)
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
    
    # Get min/max for consistent color scaling
    flux_min, flux_max = flux_3d.min(), flux_3d.max()
    fission_min, fission_max = fission_3d.min(), fission_3d.max()
    
    # Extract XY slice at z_index
    flux_xy = flux_3d[:, :, z_index].T
    fission_xy = fission_3d[:, :, z_index].T
    
    # Create mesh edges for plotting
    x_edges = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx + 1)
    y_edges = np.linspace(mesh.lower_left[1], mesh.upper_right[1], ny + 1)
    
    # Plot flux
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    pcm = ax.pcolormesh(x_edges, y_edges, flux_xy, shading='auto', cmap='hot', 
                        vmin=flux_min, vmax=flux_max)
    cbar = plt.colorbar(pcm, ax=ax, label='Flux [n/(cm² · s)]')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_title(f'Neutron Flux - Axial Level {z_index} (Batch {batch})\n{target_power_MW} MW (Manual Normalization)')
    ax.set_aspect('equal')
    
    # Add circle to show core boundary
    circle = plt.Circle((0, 0), mesh.upper_right[0], fill=False, 
                       edgecolor='white', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(circle)
    
    save_path = os.path.join(BASE_DIR, f'batch{batch}_flux_xy_z{z_index}_normalized.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved flux plot: {save_path}")
    
    # Plot fission
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    pcm = ax.pcolormesh(x_edges, y_edges, fission_xy, shading='auto', cmap='hot',
                        vmin=fission_min, vmax=fission_max)
    cbar = plt.colorbar(pcm, ax=ax, label='Fission Rate [fissions/s]')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_title(f'Fission Rate - Axial Level {z_index} (Batch {batch})\n{target_power_MW} MW (Manual Normalization)')
    ax.set_aspect('equal')
    
    # Add circle to show core boundary
    circle = plt.Circle((0, 0), mesh.upper_right[0], fill=False, 
                       edgecolor='white', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(circle)
    
    save_path = os.path.join(BASE_DIR, f'batch{batch}_fission_xy_z{z_index}_normalized.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved fission plot: {save_path}")


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
    
    ax1.plot(z_centers, flux_axial, 'b-', linewidth=2)
    ax1.set_xlabel('Axial Position [cm]')
    ax1.set_ylabel('Average Flux [n/(cm² · s)]')
    ax1.set_title(f'Axial Flux Profile (Batch {batch})\n{target_power_MW} MW')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    ax2.plot(z_centers, fission_axial, 'r-', linewidth=2)
    ax2.set_xlabel('Axial Position [cm]')
    ax2.set_ylabel('Average Fission Rate [fissions/s]')
    ax2.set_title(f'Axial Fission Profile (Batch {batch})\n{target_power_MW} MW')
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
    
    print("\n" + '='*80)
    print(f"GLOBAL REACTION RATES (Batch {batch}) - MANUAL NORMALIZATION")
    print('='*80)
    print(f"k-effective: {sp.keff.nominal_value:.5f} ± {sp.keff.std_dev:.5f}")
    print(f"\nNormalization:")
    print(f"  Heating rate: {heating_rate_ev:.3e} eV/source")
    print(f"  Source rate: {source_per_sec:.3e} source/s")
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
    
    # Plot XY slices at different axial levels
    # Bottom, quarter, middle, three-quarter, top
    n_ax_zones = 50  # Match your params["n_ax_zones"]
    z_indices = [0, n_ax_zones//4, n_ax_zones//2, 3*n_ax_zones//4, n_ax_zones-1]
    
    for z_idx in z_indices:
        plot_htgr_xyslice(batch_number, z_idx)