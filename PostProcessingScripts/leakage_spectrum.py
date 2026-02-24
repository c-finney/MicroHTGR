import openmc
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def run_leakage_analysis(run_dir, params, batch):

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

    leakage_surfaces = {
        'radial':     sp.get_tally(name='radial_leakage_current'),
        'axial_top':  sp.get_tally(name='axial_top_leakage_current'),
        'axial_bot':  sp.get_tally(name='axial_bot_leakage_current'),
    }

    results = {}
    for name, tally in leakage_surfaces.items():
        current_mean = tally.get_slice(scores=['current']).mean.flatten()
        current_std  = tally.get_slice(scores=['current']).std_dev.flatten()

        energy_filter = tally.find_filter(openmc.EnergyFilter)
        energy_bins   = energy_filter.bins
        energy_mids   = np.sqrt(energy_bins[:, 0] * energy_bins[:, 1])

        lethargy_current = energy_mids * current_mean

        thermal_mask    = energy_mids < 0.625
        epithermal_mask = (energy_mids >= 0.625) & (energy_mids < 100e3)
        fast_mask       = energy_mids >= 100e3

        total = current_mean.sum()
        if total > 0:
            f_thermal    = current_mean[thermal_mask].sum()    / total * 100
            f_epithermal = current_mean[epithermal_mask].sum() / total * 100
            f_fast       = current_mean[fast_mask].sum()       / total * 100
        else:
            f_thermal = f_epithermal = f_fast = 0.0

        label = {'radial': 'Radial', 'axial_top': 'Axial Top', 'axial_bot': 'Axial Bottom'}[name]
        print(f"\n   {label} leakage spectrum:")
        print(f"      Thermal    (< 0.625 eV):         {f_thermal:.1f}%")
        print(f"      Epithermal (0.625 eV - 100 keV): {f_epithermal:.1f}%")
        print(f"      Fast       (> 100 keV):          {f_fast:.1f}%")

        results[name] = {
            'energy_mids':      energy_mids,
            'energy_bins':      energy_bins,
            'current_mean':     current_mean,
            'current_std':      current_std,
            'lethargy_current': lethargy_current,
            'f_thermal':        f_thermal,
            'f_epithermal':     f_epithermal,
            'f_fast':           f_fast,
        }

    # =========================================================================
    # PLOTTING
    # =========================================================================

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Neutron Leakage Energy Spectra at Reflector Outer Boundaries", fontsize=13)

    colors = {'radial': 'steelblue', 'axial_top': 'firebrick', 'axial_bot': 'seagreen'}
    labels = {'radial': 'Radial', 'axial_top': 'Axial Top', 'axial_bot': 'Axial Bottom'}

    for ax, (name, r) in zip(axes, results.items()):
        norm = r['lethargy_current'].max()
        ax.semilogx(
            r['energy_mids'],
            r['lethargy_current'] / norm if norm > 0 else r['lethargy_current'],
            color=colors[name], linewidth=1.2
        )
        ax.axvline(0.625, color='green',  linestyle='--', linewidth=0.8, label='0.625 eV')
        ax.axvline(100e3, color='orange', linestyle='--', linewidth=0.8, label='100 keV')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Lethargy-weighted current (norm.)')
        ax.set_title(
            f"{labels[name]}\n"
            f"T={r['f_thermal']:.1f}%  Ep={r['f_epithermal']:.1f}%  F={r['f_fast']:.1f}%"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(run_dir, 'leakage_spectrum.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Plot saved to: {output_path}")

    np.savez(
        os.path.join(run_dir, 'leakage_spectrum.npz'),
        **{f'{name}_{k}': v for name, r in results.items() for k, v in r.items()}
    )
    print(f"   Numerical results saved to: leakage_spectrum.npz\n")

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

    run_leakage_analysis(run_dir, params, batch)