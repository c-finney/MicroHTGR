"""
Leakage Spectrum Post-Processing Script

Extracts neutron leakage energy spectra and absolute leakage rates from all core boundaries.

Usage:
    # As a module:
    from leakage_spectrum import run_leakage_analysis
    run_leakage_analysis(run_dir, params, batch=None)

    # Standalone:
    python leakage_spectrum.py <reactivity_study_directory> <batch=None>
"""

import openmc
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import glob

# ====================================================================================================
# NORMALIZATION FACTOR FUNCTION
# ====================================================================================================

def get_normalization_factor(sp, target_power_MW=15.0):
    """
    Calculate source rate normalization factor from heating tally.
    Identical approach to tally_plotter.py for consistency.

    Args:
        sp:               Already-opened openmc.StatePoint
        target_power_MW:  Target reactor power in MW

    Returns:
        float: source_per_sec — source neutrons per second at target power
    """
    heating_tally    = sp.get_tally(name='heating')
    heating_rate_ev  = heating_tally.mean[0, 0, 0]
    joule_per_ev     = 1.60218e-19
    heating_rate_j   = heating_rate_ev * joule_per_ev
    power_watts      = target_power_MW * 1e6
    source_per_sec   = power_watts / heating_rate_j
    return source_per_sec, heating_rate_ev, heating_rate_j

# ====================================================================================================
# PERFORM LEAKAGE ANALYSIS PLOTTING AND SAVE RESULTS
# ====================================================================================================

def run_leakage_analysis(run_dir, params, statepoint_path=None, batch=None):
    """
    Extract and plot neutron leakage energy spectra at all reflector outer boundaries.
    Absolute rates are normalized using the heating tally, matching tally_plotter.py.

    Args:
        run_dir: Directory containing simulation results
        params:  Merged parameter dictionary (thermal_power_MW, use_1/6_geometry)
        batch:   Batch number for statepoint file (auto-detected if None)
    """

    print(f"\n{'='*80}")
    print("LEAKAGE SPECTRUM ANALYSIS")
    print(f"{'='*80}")

    # ================================================================================
    # 1. LOCATE AND OPEN STATEPOINT
    # ================================================================================

    if statepoint_path is not None:
        sp_path = statepoint_path
    elif batch is not None:
        sp_path = os.path.join(run_dir, f"statepoint.{batch}.h5")
    else:
        # Try eigenvalue naming first, then depletion naming
        for f in sorted(os.listdir(run_dir)):
            if f.startswith("statepoint") and f.endswith(".h5"):
                sp_path = os.path.join(run_dir, f)
                break
        else:
            # Fall back to last depletion statepoint
            dep_sps = sorted(glob.glob(os.path.join(run_dir, "openmc_simulation_n*.h5")))
            if dep_sps:
                sp_path = dep_sps[-1]
            else:
                print("ERROR: No statepoint file found!")
                return None

    print(f"\nStatepoint: {sp_path}")

    sp = openmc.StatePoint(sp_path)

    # ================================================================================
    # 2. POWER NORMALIZATION
    # ================================================================================

    thermal_power_MW = params.get("thermal_power_MW", params.get("thermal_power", 15.0))
    geometry_factor  = 6 if params.get("use_1/6_geometry", False) else 1

    try:
        source_per_sec, heating_rate_ev, heating_rate_j = get_normalization_factor(
            sp, thermal_power_MW
        )
        normalization_ok = True
    except Exception as e:
        print(f"WARNING: Could not compute normalization from heating tally: {e}")
        print("         Absolute leakage rates will not be computed.")
        source_per_sec   = 0.0
        heating_rate_ev  = 0.0
        heating_rate_j   = 0.0
        normalization_ok = False

    print(f"\nThermal power:              {thermal_power_MW:.2f} MWth")
    if normalization_ok:
        print(f"Heating rate:               {heating_rate_ev:.4e} eV/source")
        print(f"Heating rate:               {heating_rate_j:.4e} J/source")
        print(f"Source rate:                {source_per_sec:.4e} source/s")
        print(f"Geometry factor:            {geometry_factor}x")

    # ================================================================================
    # 3. LOAD LEAKAGE CURRENT TALLIES
    # ================================================================================

    leakage_tally_names = {
        'radial':    'radial_leakage_current',
        'axial_top': 'axial_top_leakage_current',
        'axial_bot': 'axial_bot_leakage_current',
    }

    labels = {
        'radial':    'Radial (outer reflector cylinder)',
        'axial_top': 'Axial Top (top reflector face)',
        'axial_bot': 'Axial Bottom (bottom reflector face)',
    }

    results = {}

    for name, tally_name in leakage_tally_names.items():
        try:
            tally = sp.get_tally(name=tally_name)
        except Exception as e:
            print(f"WARNING: Could not find tally '{tally_name}': {e}")
            continue

        current_mean = tally.get_slice(scores=['current']).mean.flatten()
        current_std  = tally.get_slice(scores=['current']).std_dev.flatten()

        # Bottom surface normal points in -z so current is negative — flip to
        # get physical outward leakage
        if name == 'axial_bot':
            current_mean = np.abs(current_mean)
            current_std  = np.abs(current_std)

        energy_filter = tally.find_filter(openmc.EnergyFilter)
        energy_bins   = energy_filter.bins
        energy_mids   = np.sqrt(energy_bins[:, 0] * energy_bins[:, 1])

        lethargy_current = energy_mids * current_mean

        # Group fractions
        thermal_mask    = energy_mids < 0.625
        epithermal_mask = (energy_mids >= 0.625) & (energy_mids < 100e3)
        fast_mask       = energy_mids >= 100e3

        total_current = current_mean.sum()
        if total_current > 0:
            f_thermal    = current_mean[thermal_mask].sum()    / total_current * 100
            f_epithermal = current_mean[epithermal_mask].sum() / total_current * 100
            f_fast       = current_mean[fast_mask].sum()       / total_current * 100
        else:
            f_thermal = f_epithermal = f_fast = 0.0

        # Leakage fraction — tally is already per source neutron in eigenvalue
        # mode, scale to full core. No nu-fission division needed.
        leakage_fraction = total_current

        # Absolute leakage rate (n/s) using heating-based source_per_sec,
        # matching tally_plotter.py normalization exactly
        abs_leakage_n_per_sec = total_current * geometry_factor * source_per_sec

        results[name] = {
            'energy_mids':           energy_mids,
            'energy_bins':           energy_bins,
            'current_mean':          current_mean,
            'current_std':           current_std,
            'lethargy_current':      lethargy_current,
            'total_current':         total_current,
            'f_thermal':             f_thermal,
            'f_epithermal':          f_epithermal,
            'f_fast':                f_fast,
            'abs_leakage_n_per_sec': abs_leakage_n_per_sec,
            'leakage_fraction':      leakage_fraction,
        }

        print(f"\n{labels[name]}:")
        print(f"   Thermal    (< 0.625 eV):         {f_thermal:.1f}%")
        print(f"   Epithermal (0.625 eV - 100 keV): {f_epithermal:.1f}%")
        print(f"   Fast       (> 100 keV):          {f_fast:.1f}%")
        print(f"   Leakage fraction:                {leakage_fraction*100:.4f}%")
        if normalization_ok:
            print(f"      Absolute leakage rate:           {abs_leakage_n_per_sec:.4e} n/s")

    if not results:
        print("ERROR: No leakage tallies found in statepoint")
        return None

    # ================================================================================
    # 4. SURFACE BREAKDOWN SUMMARY
    # ================================================================================

    total_abs_leakage      = sum(r['abs_leakage_n_per_sec'] for r in results.values())
    total_leakage_fraction = sum(r['leakage_fraction']       for r in results.values())

    print(f"\nTotal leakage fraction (all surfaces): {total_leakage_fraction*100:.4f}%")
    if normalization_ok:
        print(f"Total absolute leakage rate:           {total_abs_leakage:.4e} n/s")
    print(f"\nSurface breakdown (fraction of total leakage):")
    for name, r in results.items():
        frac = (
            r['abs_leakage_n_per_sec'] / total_abs_leakage * 100
            if total_abs_leakage > 0 else 0.0
        )
        print(f"   {labels[name]:<45} {frac:.1f}%")

    # ================================================================================
    # 5. PLOTTING
    # ================================================================================

    colors = {'radial': 'steelblue', 'axial_top': 'firebrick', 'axial_bot': 'seagreen'}

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    fig.suptitle("Neutron Leakage Energy Spectra at Reflector Outer Boundaries", fontsize=13)

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

        title_lines = [
            labels[name],
            f"T={r['f_thermal']:.1f}%  Ep={r['f_epithermal']:.1f}%  F={r['f_fast']:.1f}%",
            f"Leakage: {r['leakage_fraction']*100:.3f}%",
        ]
        if normalization_ok:
            title_lines.append(f"({r['abs_leakage_n_per_sec']:.3e} n/s)")
        ax.set_title("\n".join(title_lines), fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, 'leakage_spectrum.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {plot_path}")

    # ================================================================================
    # 6. SAVE RESULTS TO TEXT FILE
    # ================================================================================

    results_file = os.path.join(run_dir, 'leakage_spectrum_results.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NEUTRON LEAKAGE SPECTRUM ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Power Normalization (heating-local tally, matches global_rates.txt):\n")
        f.write(f"   Thermal power:                {thermal_power_MW:.2f} MWth\n")
        if normalization_ok:
            f.write(f"   Heating rate:                 {heating_rate_ev:.4e} eV/source\n")
            f.write(f"   Heating rate:                 {heating_rate_j:.4e} J/source\n")
            f.write(f"   Source rate:                  {source_per_sec:.4e} source/s\n")
        f.write(f"   Geometry factor:              {geometry_factor}x "
                f"(1/{geometry_factor} symmetry)\n\n")

        f.write("=" * 80 + "\n")
        f.write("LEAKAGE BY SURFACE\n")
        f.write("=" * 80 + "\n\n")

        for name, r in results.items():
            f.write(f"{labels[name]}:\n")
            f.write(f"   Spectral fractions:\n")
            f.write(f"      Thermal    (< 0.625 eV):         {r['f_thermal']:.2f}%\n")
            f.write(f"      Epithermal (0.625 eV - 100 keV): {r['f_epithermal']:.2f}%\n")
            f.write(f"      Fast       (> 100 keV):          {r['f_fast']:.2f}%\n")
            f.write(f"   Leakage fraction:                   {r['leakage_fraction']*100:.4f}%\n")
            if normalization_ok:
                f.write(f"   Absolute leakage rate:              {r['abs_leakage_n_per_sec']:.4e} n/s\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("SURFACE BREAKDOWN SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Surface':<45} {'% of total leakage':>20}")
        if normalization_ok:
            f.write(f" {'Absolute rate (n/s)':>22}")
        f.write("\n")
        f.write(f"   {'-'*45} {'-'*20}")
        if normalization_ok:
            f.write(f" {'-'*22}")
        f.write("\n")

        for name, r in results.items():
            frac = (
                r['abs_leakage_n_per_sec'] / total_abs_leakage * 100
                if total_abs_leakage > 0 else 0.0
            )
            f.write(f"{labels[name]:<45} {frac:>19.1f}%")
            if normalization_ok:
                f.write(f" {r['abs_leakage_n_per_sec']:>22.4e}")
            f.write("\n")

        f.write(f"\n{'Total':<45} {'100.0%':>20}")
        if normalization_ok:
            f.write(f" {total_abs_leakage:>22.4e}")
        f.write("\n\n")
        f.write(f"Total leakage fraction (all surfaces): {total_leakage_fraction*100:.4f}%\n")
        f.write("\n")
        f.write("=" * 80)

    print(f"Results saved to: {results_file}")

    # ================================================================================
    # 7. SAVE NUMERICAL ARRAYS
    # ================================================================================

    np.savez(
        os.path.join(run_dir, 'leakage_spectrum.npz'),
        **{f'{name}_{k}': v for name, r in results.items() for k, v in r.items()},
        source_per_sec              = source_per_sec,
        total_abs_leakage_n_per_sec = total_abs_leakage,
        total_leakage_fraction      = total_leakage_fraction,
    )
    print(f"Numerical results saved to: leakage_spectrum.npz\n")

    return results

# ====================================================================================================
# STANDALONE ENTRY POINT
# ====================================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python leakage_spectrum.py <run_directory> [batch_number]")
        print("\nExtracts neutron leakage energy spectra from the leakage current tallies")
        print("and computes absolute leakage rates normalized to thermal power.")
        print("Parameters are loaded from run_params.json in the run directory.")
        sys.exit(1)

    run_dir = sys.argv[1]
    sp_path = sys.argv[2] if len(sys.argv) > 2 else None
    batch   = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print(f"\nProcessing: {run_dir}")

    params_path = os.path.join(run_dir, "run_params.json")
    if os.path.exists(params_path):
        print(f"Loading parameters from run_params.json...")
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        print("WARNING: run_params.json not found, using empty params dict")
        print("Absolute leakage rates will not be available without thermal_power_MW")
        params = {}

    run_leakage_analysis(run_dir, params, sp_path, batch)