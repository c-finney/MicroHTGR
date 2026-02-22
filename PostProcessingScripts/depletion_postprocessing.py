"""
Depletion Post-Processing Script

Extracts and plots results from OpenMC depletion simulations:
  - k_eff vs. burnup and time
  - Key nuclide inventories vs. burnup
  - Discharge burnup / cycle length estimates
  - Fissile inventory ratios

Usage:
    # As a module:
    from depletion_postprocessing import run_depletion_postprocessing
    run_depletion_postprocessing(run_dir, params)

    # Standalone:
    python depletion_postprocessing.py <run_directory>
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================
# Nuclide groups to track
# ===========================================================================

# Primary fissile / fertile nuclides
FISSILE_NUCLIDES = ["U235", "Pu239", "Pu241"]
FERTILE_NUCLIDES = ["U238", "Pu240", "Pu242"]
POISON_NUCLIDES = ["Xe135", "Sm149", "Gd155", "Gd157"]
ACTINIDE_NUCLIDES = ["U234", "U235", "U236", "U238", "Np237",
                      "Pu238", "Pu239", "Pu240", "Pu241", "Pu242",
                      "Am241", "Am243", "Cm244"]
FP_NUCLIDES = ["Xe135", "Sm149", "Cs137", "Sr90", "I131", "Nd143"]


def run_depletion_postprocessing(run_dir, params):
    """
    Run full depletion post-processing.

    Parameters
    ----------
    run_dir : str
        Directory containing depletion_results.h5.
    params : dict
        Simulation parameters.

    Returns
    -------
    dict : Summary results.
    """
    import openmc.deplete

    print(f"\n{'=' * 80}")
    print("DEPLETION POST-PROCESSING")
    print(f"{'=' * 80}")
    print(f"Run directory: {run_dir}")

    results_path = os.path.join(run_dir, "depletion_results.h5")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found!")
        return None

    results = openmc.deplete.Results(results_path)

    # ==================================================================
    # 1. k-effective vs time/burnup
    # ==================================================================
    time_steps, keff_values = results.get_keff()
    keff_mean = np.array([k.nominal_value for k in keff_values])
    keff_std = np.array([k.std_dev for k in keff_values])

    # Time in days (OpenMC returns seconds)
    time_days = time_steps / 86400.0
    time_years = time_days / 365.25

    # Calculate burnup (MWd/MtU) if power and HM mass are known
    thermal_power_MW = params.get("thermal_power_MW", 15.0)
    total_HM_mass_kg = params.get("_total_HM_mass_kg", None)

    if total_HM_mass_kg and total_HM_mass_kg > 0:
        # Cumulative energy in MWd
        cumulative_energy_MWd = thermal_power_MW * time_days
        burnup_MWd_per_MtU = cumulative_energy_MWd / (total_HM_mass_kg / 1000.0)
    else:
        burnup_MWd_per_MtU = None

    # ==================================================================
    # 2. Find discharge burnup (where k_eff crosses 1.0)
    # ==================================================================
    discharge_burnup = None
    discharge_time_days = None
    discharge_time_years = None

    for i in range(len(keff_mean) - 1):
        if keff_mean[i] >= 1.0 and keff_mean[i + 1] < 1.0:
            # Linear interpolation
            frac = (keff_mean[i] - 1.0) / (keff_mean[i] - keff_mean[i + 1])
            discharge_time_days = time_days[i] + frac * (time_days[i + 1] - time_days[i])
            discharge_time_years = discharge_time_days / 365.25

            if burnup_MWd_per_MtU is not None:
                discharge_burnup = burnup_MWd_per_MtU[i] + frac * (
                    burnup_MWd_per_MtU[i + 1] - burnup_MWd_per_MtU[i]
                )
            break

    # ==================================================================
    # 3. Extract nuclide inventories
    # ==================================================================
    nuclide_data = {}

    # Collect all nuclides we want to track
    all_tracked = set(FISSILE_NUCLIDES + FERTILE_NUCLIDES + POISON_NUCLIDES +
                      ACTINIDE_NUCLIDES + FP_NUCLIDES)

    for nuc in all_tracked:
        try:
            _time, atoms = results.get_atoms("1", nuc)  # material id "1" = fuel
            nuclide_data[nuc] = atoms
        except Exception:
            # Try with different material IDs or naming
            try:
                # Fuel is typically the first material
                _time, atoms = results.get_atoms(run_dir, nuc, nuc_units="atom")
                nuclide_data[nuc] = atoms
            except Exception:
                pass

    # If material ID "1" doesn't work, try to find the fuel material
    if not nuclide_data:
        print("  Trying alternative material indexing...")
        try:
            # Get list of materials from first step
            mat_ids = list(results[0].index_mat.keys())
            print(f"  Available material IDs: {mat_ids}")

            for mat_id in mat_ids:
                for nuc in ["U235", "U238"]:
                    try:
                        _time, atoms = results.get_atoms(mat_id, nuc)
                        if np.any(atoms > 0):
                            print(f"  Found fuel in material ID: {mat_id}")
                            # Re-extract all nuclides with correct material ID
                            for nuc2 in all_tracked:
                                try:
                                    _time, atoms2 = results.get_atoms(mat_id, nuc2)
                                    nuclide_data[nuc2] = atoms2
                                except Exception:
                                    pass
                            break
                    except Exception:
                        continue
                if nuclide_data:
                    break
        except Exception as e:
            print(f"  Warning: Could not extract nuclide data: {e}")

    # ==================================================================
    # 4. Print summary
    # ==================================================================
    print(f"\n{'─' * 60}")
    print("  DEPLETION RESULTS SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Number of depletion steps: {len(keff_mean)}")
    print(f"  Total simulation time: {time_days[-1]:.1f} days ({time_years[-1]:.2f} years)")
    print(f"  Initial k_eff: {keff_mean[0]:.5f} ± {keff_std[0]:.5f}")
    print(f"  Final k_eff:   {keff_mean[-1]:.5f} ± {keff_std[-1]:.5f}")

    if burnup_MWd_per_MtU is not None:
        print(f"  Final burnup:  {burnup_MWd_per_MtU[-1]:.0f} MWd/MtU")

    if discharge_time_days is not None:
        print(f"\n  Discharge (k_eff = 1.0):")
        print(f"    Time:   {discharge_time_days:.1f} days ({discharge_time_years:.2f} years)")
        if discharge_burnup is not None:
            print(f"    Burnup: {discharge_burnup:.0f} MWd/MtU")
    else:
        if keff_mean[-1] > 1.0:
            print(f"\n  Core still supercritical at end of simulation (k = {keff_mean[-1]:.5f})")
        else:
            print(f"\n  Core subcritical from start (k_initial = {keff_mean[0]:.5f})")

    # Fissile inventory change
    if "U235" in nuclide_data:
        u235_initial = nuclide_data["U235"][0]
        u235_final = nuclide_data["U235"][-1]
        if u235_initial > 0:
            u235_depletion = (1.0 - u235_final / u235_initial) * 100
            print(f"\n  U-235 depletion: {u235_depletion:.1f}%")

    if "Pu239" in nuclide_data:
        pu239_final = nuclide_data["Pu239"][-1]
        if "U235" in nuclide_data and nuclide_data["U235"][0] > 0:
            pu_to_u_ratio = pu239_final / nuclide_data["U235"][0] * 100
            print(f"  Pu-239 buildup (% of initial U-235 atoms): {pu_to_u_ratio:.2f}%")

    print(f"{'─' * 60}")

    # ==================================================================
    # 5. Generate plots
    # ==================================================================
    print("\nGenerating depletion plots...")

    # Use burnup as x-axis if available, otherwise time
    if burnup_MWd_per_MtU is not None:
        x_data = burnup_MWd_per_MtU
        x_label = "Burnup (MWd/MtU)"
        x_label_short = "burnup"
    else:
        x_data = time_days
        x_label = "Time (days)"
        x_label_short = "time"

    # --- k_eff vs burnup/time ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.errorbar(x_data, keff_mean, yerr=keff_std, fmt="o-", capsize=3,
                markersize=5, linewidth=1.5, label="k-effective")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1, label="k = 1.0")

    if discharge_burnup is not None and burnup_MWd_per_MtU is not None:
        ax.axvline(discharge_burnup, color="green", linestyle=":", alpha=0.7,
                   label=f"Discharge: {discharge_burnup:.0f} MWd/MtU")
    elif discharge_time_days is not None:
        ax.axvline(discharge_time_days, color="green", linestyle=":", alpha=0.7,
                   label=f"Discharge: {discharge_time_days:.0f} days")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("k-effective", fontsize=12)
    ax.set_title("k-effective vs. Burnup", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(run_dir, f"depletion_keff_vs_{x_label_short}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # --- k_eff vs time (always plot this) ---
    if burnup_MWd_per_MtU is not None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        ax.errorbar(time_days, keff_mean, yerr=keff_std, fmt="o-", capsize=3,
                    markersize=5, linewidth=1.5)
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_xlabel("Time (days)", fontsize=12)
        ax.set_ylabel("k-effective", fontsize=12)
        ax.set_title("k-effective vs. Time", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add secondary x-axis for years
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()[0] / 365.25, ax.get_xlim()[1] / 365.25)
        ax2.set_xlabel("Time (years)", fontsize=11)

        save_path = os.path.join(run_dir, "depletion_keff_vs_time.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # --- Reactivity vs burnup/time ---
    reactivity_pcm = (keff_mean - 1.0) / keff_mean * 1e5

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(x_data, reactivity_pcm, "o-", markersize=5, linewidth=1.5)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Reactivity (pcm)", fontsize=12)
    ax.set_title("Excess Reactivity vs. Burnup", fontsize=14)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(run_dir, f"depletion_reactivity_vs_{x_label_short}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # --- Fissile nuclide inventories ---
    _plot_nuclide_group(x_data, x_label, time_steps, nuclide_data,
                        FISSILE_NUCLIDES, "Fissile Nuclide Inventories",
                        run_dir, "depletion_fissile_inventory")

    # --- Fertile nuclide inventories ---
    _plot_nuclide_group(x_data, x_label, time_steps, nuclide_data,
                        FERTILE_NUCLIDES, "Fertile Nuclide Inventories",
                        run_dir, "depletion_fertile_inventory")

    # --- Actinide buildup ---
    actinide_plot = [n for n in ACTINIDE_NUCLIDES if n not in ["U235", "U238"]]
    _plot_nuclide_group(x_data, x_label, time_steps, nuclide_data,
                        actinide_plot, "Transuranics & Minor Actinide Buildup",
                        run_dir, "depletion_actinide_buildup")

    # --- Fission product poisons ---
    _plot_nuclide_group(x_data, x_label, time_steps, nuclide_data,
                        POISON_NUCLIDES, "Fission Product Poisons",
                        run_dir, "depletion_fp_poisons")

    # --- Conversion ratio over time ---
    if "Pu239" in nuclide_data and "U235" in nuclide_data:
        pu239 = nuclide_data["Pu239"]
        u235 = nuclide_data["U235"]

        # Fissile inventory ratio: (U235 + Pu239 + Pu241) / initial_fissile
        fissile_initial = u235[0]
        if "Pu241" in nuclide_data:
            fissile_current = u235 + pu239 + nuclide_data["Pu241"]
        else:
            fissile_current = u235 + pu239
        fissile_ratio = fissile_current / fissile_initial

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.plot(x_data, fissile_ratio, "o-", markersize=5, linewidth=1.5, color="tab:green")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Fissile Inventory Ratio", fontsize=12)
        ax.set_title("Fissile Inventory Ratio vs. Burnup\n(U235 + Pu239 + Pu241) / Initial Fissile", fontsize=13)
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(run_dir, f"depletion_fissile_ratio_vs_{x_label_short}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # ==================================================================
    # 6. Save results
    # ==================================================================
    summary = {
        "n_steps": len(keff_mean),
        "time_days": time_days.tolist(),
        "time_years": time_years.tolist(),
        "keff_mean": keff_mean.tolist(),
        "keff_std": keff_std.tolist(),
        "burnup_MWd_per_MtU": burnup_MWd_per_MtU.tolist() if burnup_MWd_per_MtU is not None else None,
        "discharge_burnup_MWd_per_MtU": discharge_burnup,
        "discharge_time_days": discharge_time_days,
        "discharge_time_years": discharge_time_years,
        "initial_keff": float(keff_mean[0]),
        "final_keff": float(keff_mean[-1]),
        "thermal_power_MW": thermal_power_MW,
    }

    json_path = os.path.join(run_dir, "depletion_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  Summary saved to: {json_path}")

    # CSV for external plotting
    csv_path = os.path.join(run_dir, "depletion_keff_data.csv")
    header = "time_days,time_years,keff,keff_std"
    if burnup_MWd_per_MtU is not None:
        header += ",burnup_MWd_per_MtU"
        data = np.column_stack([time_days, time_years, keff_mean, keff_std, burnup_MWd_per_MtU])
    else:
        data = np.column_stack([time_days, time_years, keff_mean, keff_std])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"  CSV data saved to: {csv_path}")

    # Text report
    txt_path = os.path.join(run_dir, "depletion_results.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DEPLETION SIMULATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Thermal power: {thermal_power_MW} MW\n")
        f.write(f"Depletion steps: {len(keff_mean)}\n")
        f.write(f"Total time: {time_days[-1]:.1f} days ({time_years[-1]:.2f} years)\n\n")
        f.write(f"Initial k_eff: {keff_mean[0]:.5f} ± {keff_std[0]:.5f}\n")
        f.write(f"Final k_eff:   {keff_mean[-1]:.5f} ± {keff_std[-1]:.5f}\n\n")
        if burnup_MWd_per_MtU is not None:
            f.write(f"Final burnup: {burnup_MWd_per_MtU[-1]:.0f} MWd/MtU\n\n")
        if discharge_time_days is not None:
            f.write(f"Discharge (k=1.0): {discharge_time_days:.1f} days ({discharge_time_years:.2f} years)\n")
            if discharge_burnup is not None:
                f.write(f"Discharge burnup: {discharge_burnup:.0f} MWd/MtU\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write(f"{'Step':>5}  {'Time (d)':>10}  {'k_eff':>10}  {'± σ':>8}")
        if burnup_MWd_per_MtU is not None:
            f.write(f"  {'Burnup':>12}")
        f.write("\n" + "-" * 60 + "\n")
        for i in range(len(keff_mean)):
            f.write(f"{i:>5}  {time_days[i]:>10.1f}  {keff_mean[i]:>10.5f}  {keff_std[i]:>8.5f}")
            if burnup_MWd_per_MtU is not None:
                f.write(f"  {burnup_MWd_per_MtU[i]:>12.0f}")
            f.write("\n")
        f.write("=" * 80 + "\n")
    print(f"  Report saved to: {txt_path}")

    print(f"\n{'=' * 80}")
    print("DEPLETION POST-PROCESSING COMPLETE")
    print(f"{'=' * 80}\n")

    return summary


# ===========================================================================
# Plotting helper
# ===========================================================================

def _plot_nuclide_group(x_data, x_label, time_steps, nuclide_data,
                        nuclide_list, title, run_dir, filename_base):
    """Plot a group of nuclide inventories on the same axes."""

    available = [n for n in nuclide_list if n in nuclide_data and np.any(nuclide_data[n] > 0)]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    for nuc in available:
        atoms = nuclide_data[nuc]
        # Ensure same length as x_data
        n = min(len(x_data), len(atoms))
        ax.plot(x_data[:n], atoms[:n], "o-", markersize=4, linewidth=1.5, label=nuc)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Number of Atoms", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, ncol=min(4, len(available)))
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    save_path = os.path.join(run_dir, f"{filename_base}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ===========================================================================
# Standalone entry point
# ===========================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python depletion_postprocessing.py <run_directory>")
        sys.exit(1)

    run_dir = sys.argv[1]

    params_path = os.path.join(run_dir, "run_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        params = {}

    run_depletion_postprocessing(run_dir, params)
