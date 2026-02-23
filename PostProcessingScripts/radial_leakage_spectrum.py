import openmc
import numpy as np
import matplotlib.pyplot as plt
import os

def run_radial_leakage_analysis(run_dir, params, batch):
    """
    Extract and plot the neutron energy spectrum at the radial core boundary.
    """

    print("\nRunning radial leakage spectrum analysis...")

    # Parse k_eff and output from openmc_output.txt instead of loading statepoint
    # directly to avoid memory issues on full core runs — statepoint is only opened
    # in lightweight summary mode
    
    if batch is None:
        for f in os.listdir(run_dir):
            if f.startswith("statepoint") and f.endswith(".h5"):
                batch = int(f.split(".")[1])
                break

    if batch is None:
        print("ERROR: No statepoint file found!")
        return None

    sp_path = os.path.join(run_dir, f"statepoint.{batch}.h5")
    print(f"Statepoint: {sp_path}")

    sp = openmc.StatePoint(sp_path)

    # ==================================================================
    # CURRENT TALLY
    # ==================================================================

    current_tally = sp.get_tally(name='radial_leakage_current')
    current_data  = current_tally.get_slice(scores=['current'])
    current_mean  = current_data.mean.flatten()
    current_std   = current_data.std_dev.flatten()

    # ==================================================================
    # FLUX TALLY
    # ==================================================================

    flux_tally = sp.get_tally(name='radial_leakage_flux')
    flux_data  = flux_tally.get_slice(scores=['flux'])
    flux_mean  = flux_data.mean.flatten()

    # ==================================================================
    # ENERGY BINS
    # ==================================================================

    energy_filter = current_tally.find_filter(openmc.EnergyFilter)
    energy_bins   = energy_filter.bins        # shape (N, 2)
    energy_mids   = np.sqrt(energy_bins[:, 0] * energy_bins[:, 1])  # log midpoints in eV
    delta_E       = energy_bins[:, 1] - energy_bins[:, 0]

    # Lethargy-normalized flux: phi(u) = E * phi(E)
    lethargy_flux    = energy_mids * flux_mean
    lethargy_current = energy_mids * current_mean

    # ==================================================================
    # GROUP FRACTIONS
    # ==================================================================

    thermal_mask    = energy_mids < 0.625        # < 0.625 eV
    epithermal_mask = (energy_mids >= 0.625) & (energy_mids < 100e3)
    fast_mask       = energy_mids >= 100e3

    total_current = current_mean.sum()
    if total_current > 0:
        f_thermal    = current_mean[thermal_mask].sum()    / total_current * 100
        f_epithermal = current_mean[epithermal_mask].sum() / total_current * 100
        f_fast       = current_mean[fast_mask].sum()       / total_current * 100

        print(f"\n   Radial leakage spectrum fractions:")
        print(f"   Thermal    (< 0.625 eV):      {f_thermal:.1f}%")
        print(f"   Epithermal (0.625 eV - 100 keV): {f_epithermal:.1f}%")
        print(f"   Fast       (> 100 keV):        {f_fast:.1f}%")

    # ==================================================================
    # AXIAL DISTRIBUTION OF LEAKAGE
    # ==================================================================

    mesh_filter   = current_tally.find_filter(openmc.MeshFilter)
    mesh          = mesh_filter.mesh
    n_z           = len(mesh.z_grid) - 1
    n_E           = len(energy_bins)

    # current_mean is flattened over (r, phi, z, E) — reshape accordingly
    n_r   = len(mesh.r_grid) - 1
    n_phi = len(mesh.phi_grid) - 1
    current_4d = current_mean.reshape(n_r, n_phi, n_z, n_E)

    # Sum over r, phi, and E to get axial profile
    axial_current = current_4d[0, :, :, :].sum(axis=(0, 2))  # shape (n_z,)
    z_mids = 0.5 * (mesh.z_grid[:-1] + mesh.z_grid[1:])

    # ==================================================================
    # PLOTTING
    # ==================================================================

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Radial Core Boundary Neutron Leakage", fontsize=14)

    # Plot 1: lethargy-weighted energy spectrum of current
    axes[0].semilogx(energy_mids, lethargy_current / lethargy_current.max(), 'b-', linewidth=1.2)
    axes[0].axvline(0.625,   color='green',  linestyle='--', linewidth=0.8, label='Thermal cutoff (0.625 eV)')
    axes[0].axvline(100e3,   color='orange', linestyle='--', linewidth=0.8, label='Fast cutoff (100 keV)')
    axes[0].set_xlabel('Energy (eV)')
    axes[0].set_ylabel('Lethargy-weighted current (normalized)')
    axes[0].set_title('Leakage Energy Spectrum')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: lethargy-weighted flux in the boundary annulus
    axes[1].semilogx(energy_mids, lethargy_flux / lethargy_flux.max(), 'r-', linewidth=1.2)
    axes[1].axvline(0.625, color='green',  linestyle='--', linewidth=0.8)
    axes[1].axvline(100e3, color='orange', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('Energy (eV)')
    axes[1].set_ylabel('Lethargy-weighted flux (normalized)')
    axes[1].set_title('Boundary Annulus Flux Spectrum')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: axial distribution of radial leakage
    axes[2].plot(axial_current / axial_current.max(), z_mids, 'k-', linewidth=1.2)
    axes[2].set_xlabel('Radial leakage current (normalized)')
    axes[2].set_ylabel('Axial position (cm)')
    axes[2].set_title('Axial Profile of Radial Leakage')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(run_dir, 'radial_leakage_spectrum.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Plot saved to: {output_path}")

    # ==================================================================
    # SAVE NUMERICAL RESULTS
    # ==================================================================

    results_path = os.path.join(run_dir, 'radial_leakage_spectrum.npz')
    np.savez(
        results_path,
        energy_mids      = energy_mids,
        energy_bins      = energy_bins,
        current_mean     = current_mean,
        current_std      = current_std,
        flux_mean        = flux_mean,
        axial_current    = axial_current,
        z_mids           = z_mids,
        f_thermal        = f_thermal    if total_current > 0 else 0.0,
        f_epithermal     = f_epithermal if total_current > 0 else 0.0,
        f_fast           = f_fast       if total_current > 0 else 0.0,
    )
    print(f"   Numerical results saved to: {results_path}\n")

# ===========================================================================
# Standalone entry point
# ===========================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python spectrum_thermalization.py <run_directory> [batch_number]")
        print("\nExtracts neutron energy spectrum from the 'flux_energy_spectrum' tally")
        print("and computes thermalization metrics.  Parameters are loaded from")
        print("run_params.json in the run directory.")
        sys.exit(1)

    run_dir = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"\nProcessing: {run_dir}")

    # Load params
    params_path = os.path.join(run_dir, "run_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        params = {}

    run_radial_leakage_analysis(run_dir, params, batch)