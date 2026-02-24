"""
Depletion Post-Processing Script

Extracts and plots results from OpenMC depletion simulations:
  - k_eff vs. burnup and time
  - Nuclide inventories vs. burnup (driven by params["tracked_nuclides"])
  - Discharge burnup / cycle length estimates
  - Fissile inventory ratios
  - B-10 burnout from burnable poison material

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

# Default plot groups — used only if not specified in params
DEFAULT_PLOT_GROUPS = {
    "Fissile Actinides":    ["U235", "Pu239", "Pu241"],
    "Fertile Actinides":    ["U238", "U234", "U236", "Pu240", "Pu242"],
    "Minor Actinides":      ["Np237", "Np239", "Pu238",
                             "Am241", "Am243",
                             "Cm242", "Cm243", "Cm244", "Cm245", "Cm246"],
    "Xe/I Poisons":         ["Xe131", "Xe135", "Xe135_m1", "I135"],
    "Sm/Pm Poisons":        ["Sm149", "Sm151", "Sm152", "Pm147", "Pm149"],
    "Cs/Sr FPs":            ["Cs133", "Cs134", "Cs137", "Sr90"],
    "Nd/Eu FPs":            ["Nd143", "Nd145", "Nd147", "Eu153", "Eu154", "Eu155"],
    "Mo/Tc/Rh/Pd FPs":     ["Mo95", "Tc99", "Rh103", "Rh105", "Pd107"],
    "Kr FPs":               ["Kr83"],
    "Burnable Poison":      ["B10"],
}

def _find_material_id(results, params, mat_key, search_nuclide, label):
    """
    Locate a material ID in depletion results.

    Checks params[mat_key] first (saved by main.py volume calculation),
    then searches all material IDs for one containing search_nuclide.

    Args:
        results:        openmc.deplete.Results object
        params:         parameter dictionary
        mat_key:        key in params where the material ID was saved
                        e.g. "fuel_material_id" or "poison_material_id"
        search_nuclide: nuclide to look for when searching (e.g. "U235", "B10")
        label:          human-readable label for print messages

    Returns:
        str or None: material ID string, or None if not found
    """
    # Option 1 — saved ID from run_params.json
    if mat_key in params:
        mat_id = str(params[mat_key])
        try:
            _, atoms = results.get_atoms(mat_id, search_nuclide)
            if np.any(atoms > 0):
                print(f"   {label} material ID from run_params.json: {mat_id}")
                return mat_id
        except Exception:
            print(f"   WARNING: {mat_key}={mat_id} not found in results, searching...")

    # Option 2 — search all material IDs
    try:
        mat_ids = list(results[0].index_mat.keys())
        for mid in mat_ids:
            try:
                _, atoms = results.get_atoms(mid, search_nuclide)
                if np.any(atoms > 0):
                    print(f"   Found {label} in material ID: {mid}")
                    return mid
            except Exception:
                continue
    except Exception as e:
        print(f"   WARNING: Could not search material IDs for {label}: {e}")

    print(f"   WARNING: Could not locate {label} material in depletion results")
    return None

def _extract_nuclide_inventories(results, mat_id, nuclide_list, label):
    """
    Extract atom inventories for a list of nuclides from a single material.

    Args:
        results:      openmc.deplete.Results object
        mat_id:       material ID string
        nuclide_list: list of nuclide name strings
        label:        human-readable label for print messages

    Returns:
        dict: {nuclide_name: np.ndarray of atom counts per timestep}
              Only nuclides with at least one nonzero value are included.
    """
    if mat_id is None:
        return {}

    nuclide_data = {}
    failed = []

    for nuc in nuclide_list:
        try:
            _time, atoms = results.get_atoms(mat_id, nuc)
            if np.any(atoms > 0):
                nuclide_data[nuc] = np.array(atoms)
        except Exception:
            failed.append(nuc)

    if failed:
        print(f"   [{label}] Not found in chain/results: {failed}")

    print(f"   [{label}] Extracted {len(nuclide_data)} / {len(nuclide_list)} nuclides")
    return nuclide_data

def run_depletion_postprocessing(run_dir, params):
    """
    Run full depletion post-processing.

    Parameters
    ----------
    run_dir : str
        Directory containing depletion_results.h5.
    params : dict
        Simulation parameters (merged with run_params.json).

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
    # 1. k-effective vs time / burnup
    # ==================================================================

    time_steps, keff_values = results.get_keff()
    keff_mean = np.array([k.nominal_value for k in keff_values])
    keff_std  = np.array([k.std_dev       for k in keff_values])

    time_days  = time_steps / 86400.0
    time_years = time_days  / 365.25

    thermal_power_MW  = params.get("thermal_power_MW", 15.0)
    total_HM_mass_kg  = params.get("total_HM_mass_kg", None)
    total_B10_mass_kg = params.get("total_B10_mass_kg", None)

    if total_HM_mass_kg and total_HM_mass_kg > 0:
        cumulative_energy_MWd = thermal_power_MW * time_days
        burnup_MWd_per_MtU    = cumulative_energy_MWd / (total_HM_mass_kg / 1000.0)
    else:
        burnup_MWd_per_MtU = None

    x_data        = burnup_MWd_per_MtU if burnup_MWd_per_MtU is not None else time_days
    x_label       = "Burnup (MWd/MtU)"  if burnup_MWd_per_MtU is not None else "Time (days)"
    x_label_short = "burnup"            if burnup_MWd_per_MtU is not None else "time"

    # ==================================================================
    # 2. Discharge burnup (k_eff crosses 1.0)
    # ==================================================================

    discharge_burnup     = None
    discharge_time_days  = None
    discharge_time_years = None

    for i in range(len(keff_mean) - 1):
        if keff_mean[i] >= 1.0 and keff_mean[i + 1] < 1.0:
            frac = (keff_mean[i] - 1.0) / (keff_mean[i] - keff_mean[i + 1])
            discharge_time_days  = time_days[i] + frac * (time_days[i + 1] - time_days[i])
            discharge_time_years = discharge_time_days / 365.25
            if burnup_MWd_per_MtU is not None:
                discharge_burnup = burnup_MWd_per_MtU[i] + frac * (
                    burnup_MWd_per_MtU[i + 1] - burnup_MWd_per_MtU[i]
                )
            break

    # ==================================================================
    # 3. Locate materials and extract nuclide inventories
    # ==================================================================

    print("\n   Locating materials in depletion results...")

    fuel_mat_id   = _find_material_id(results, params,
                                       "fuel_material_id",   "U235", "Fuel")
    poison_mat_id = _find_material_id(results, params,
                                       "poison_material_id", "B10",  "Burnable poison")

    tracked_nuclides        = params.get("tracked_nuclides", ["U235", "U238", "Pu239", "B10"])
    poison_tracked_nuclides = params.get("poison_tracked_nuclides", ["B10"])

    # Exclude poison-specific nuclides from fuel search to avoid confusion
    fuel_nuclides = [n for n in tracked_nuclides if n not in poison_tracked_nuclides]

    print(f"\n   Extracting fuel inventories ({len(fuel_nuclides)} nuclides)...")
    fuel_data   = _extract_nuclide_inventories(results, fuel_mat_id,   fuel_nuclides,        "Fuel")

    print(f"\n   Extracting burnable poison inventories ({len(poison_tracked_nuclides)} nuclides)...")
    poison_data = _extract_nuclide_inventories(results, poison_mat_id, poison_tracked_nuclides, "Poison")

    # Merge for plotting — poison data keyed separately to avoid name collision
    # B10 from poison material is canonical; if also in fuel, prefer poison
    all_nuclide_data = {**fuel_data}
    for nuc, atoms in poison_data.items():
        all_nuclide_data[f"{nuc}_poison"] = atoms  # keep separate key
        all_nuclide_data[nuc] = atoms               # also overwrite top-level with poison value

    # ==================================================================
    # 4. Print summary
    # ==================================================================

    print(f"\n{'─' * 60}")
    print("  DEPLETION RESULTS SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Depletion steps:    {len(keff_mean)}")
    print(f"  Total time:         {time_days[-1]:.1f} days ({time_years[-1]:.2f} years)")
    print(f"  Initial k_eff:      {keff_mean[0]:.5f} ± {keff_std[0]:.5f}")
    print(f"  Final k_eff:        {keff_mean[-1]:.5f} ± {keff_std[-1]:.5f}")
    if burnup_MWd_per_MtU is not None:
        print(f"  Final burnup:       {burnup_MWd_per_MtU[-1]:.0f} MWd/MtU")
    if total_HM_mass_kg:
        print(f"  Initial HM mass:    {total_HM_mass_kg:.2f} kg")
    if total_B10_mass_kg:
        print(f"  Initial B-10 mass:  {total_B10_mass_kg:.4f} kg")

    if discharge_time_days is not None:
        print(f"\n  Discharge (k_eff = 1.0):")
        print(f"    Time:   {discharge_time_days:.1f} days ({discharge_time_years:.2f} years)")
        if discharge_burnup is not None:
            print(f"    Burnup: {discharge_burnup:.0f} MWd/MtU")
    elif keff_mean[-1] > 1.0:
        print(f"\n  Core still supercritical at end (k = {keff_mean[-1]:.5f})")
    else:
        print(f"\n  Core subcritical from start (k_initial = {keff_mean[0]:.5f})")

    for nuc, source_label in [("U235", "fuel"), ("Pu239", "fuel"), ("B10", "poison")]:
        if nuc in all_nuclide_data:
            initial = all_nuclide_data[nuc][0]
            final   = all_nuclide_data[nuc][-1]
            if initial > 0:
                pct = (1.0 - final / initial) * 100
                print(f"\n  {nuc} [{source_label}]: {initial:.4e} → {final:.4e} atoms  "
                      f"({pct:.1f}% depleted)")

    if "Pu239" in fuel_data and "U235" in fuel_data and fuel_data["U235"][0] > 0:
        pu_ratio = fuel_data["Pu239"][-1] / fuel_data["U235"][0] * 100
        print(f"\n  Pu-239 final (% of initial U-235 atoms): {pu_ratio:.2f}%")

    print(f"{'─' * 60}")

    # ==================================================================
    # 5. Generate plots
    # ==================================================================

    print("\nGenerating depletion plots...")

    # k_eff vs burnup/time
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
    plt.savefig(os.path.join(run_dir, f"depletion_keff_vs_{x_label_short}.png"), bbox_inches="tight")
    plt.close()

    # k_eff vs time with year secondary axis
    if burnup_MWd_per_MtU is not None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        ax.errorbar(time_days, keff_mean, yerr=keff_std, fmt="o-", capsize=3,
                    markersize=5, linewidth=1.5)
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_xlabel("Time (days)", fontsize=12)
        ax.set_ylabel("k-effective", fontsize=12)
        ax.set_title("k-effective vs. Time", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()[0] / 365.25, ax.get_xlim()[1] / 365.25)
        ax2.set_xlabel("Time (years)", fontsize=11)
        plt.savefig(os.path.join(run_dir, "depletion_keff_vs_time.png"), bbox_inches="tight")
        plt.close()

    # Reactivity (pcm)
    reactivity_pcm = (keff_mean - 1.0) / keff_mean * 1e5
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(x_data, reactivity_pcm, "o-", markersize=5, linewidth=1.5)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Reactivity (pcm)", fontsize=12)
    ax.set_title("Excess Reactivity vs. Burnup", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(run_dir, f"depletion_reactivity_vs_{x_label_short}.png"), bbox_inches="tight")
    plt.close()

    # Nuclide group plots — driven entirely by params["depletion_plot_groups"]
    plot_groups   = params.get("depletion_plot_groups", DEFAULT_PLOT_GROUPS)
    plotted_nuclides = set()

    for group_name, group_nuclides in plot_groups.items():
        available = [n for n in group_nuclides if n in all_nuclide_data]
        if available:
            _plot_nuclide_group(
                x_data, x_label, all_nuclide_data, available,
                group_name, run_dir,
                f"depletion_{group_name.lower().replace('/', '').replace(' ', '_')}"
            )
            plotted_nuclides.update(available)

    # Any tracked nuclides not covered by a plot group
    ungrouped = [
        n for n in tracked_nuclides
        if n in all_nuclide_data and n not in plotted_nuclides
    ]
    if ungrouped:
        _plot_nuclide_group(
            x_data, x_label, all_nuclide_data, ungrouped,
            "Other Tracked Nuclides", run_dir,
            "depletion_other_nuclides"
        )

    # Fissile inventory ratio
    fissile_present = [n for n in ["U235", "Pu239", "Pu241"] if n in fuel_data]
    if fissile_present and "U235" in fuel_data and fuel_data["U235"][0] > 0:
        fissile_initial = fuel_data["U235"][0]
        fissile_current = sum(fuel_data[n] for n in fissile_present)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.plot(x_data, fissile_current / fissile_initial, "o-",
                markersize=5, linewidth=1.5, color="tab:green")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Fissile Inventory Ratio", fontsize=12)
        ax.set_title(
            f"Fissile Inventory Ratio vs. Burnup\n"
            f"({' + '.join(fissile_present)}) / Initial U-235",
            fontsize=13
        )
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(run_dir, f"depletion_fissile_ratio_vs_{x_label_short}.png"),
                    bbox_inches="tight")
        plt.close()

    # B-10 burnout — two-panel: absolute atoms and fractional remaining
    if "B10" in all_nuclide_data:
        b10        = all_nuclide_data["B10"]
        b10_initial = b10[0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        ax1.plot(x_data, b10, "o-", markersize=4, linewidth=1.5, color="purple")
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel("B-10 Atoms", fontsize=12)
        ax1.set_title("B-10 Absolute Inventory (Burnable Poison)", fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        if b10_initial > 0:
            ax2.plot(x_data, b10 / b10_initial * 100, "o-", markersize=4,
                     linewidth=1.5, color="darkviolet")
            ax2.set_xlabel(x_label, fontsize=12)
            ax2.set_ylabel("Remaining B-10 (%)", fontsize=12)
            ax2.set_title("B-10 Fractional Burnout", fontsize=13)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "depletion_B10_burnout.png"), bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_B10_burnout.png")

    # ==================================================================
    # 6. Save results
    # ==================================================================

    summary = {
        "n_steps":                       len(keff_mean),
        "time_days":                     time_days.tolist(),
        "time_years":                    time_years.tolist(),
        "keff_mean":                     keff_mean.tolist(),
        "keff_std":                      keff_std.tolist(),
        "burnup_MWd_per_MtU":           burnup_MWd_per_MtU.tolist() if burnup_MWd_per_MtU is not None else None,
        "discharge_burnup_MWd_per_MtU": discharge_burnup,
        "discharge_time_days":           discharge_time_days,
        "discharge_time_years":          discharge_time_years,
        "initial_keff":                  float(keff_mean[0]),
        "final_keff":                    float(keff_mean[-1]),
        "thermal_power_MW":              thermal_power_MW,
        "total_HM_mass_kg":              total_HM_mass_kg,
        "total_B10_mass_kg":             total_B10_mass_kg,
        "fuel_nuclides_extracted":       list(fuel_data.keys()),
        "poison_nuclides_extracted":     list(poison_data.keys()),
    }

    with open(os.path.join(run_dir, "depletion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # CSV
    header = "time_days,time_years,keff,keff_std"
    cols   = [time_days, time_years, keff_mean, keff_std]
    if burnup_MWd_per_MtU is not None:
        header += ",burnup_MWd_per_MtU"
        cols.append(burnup_MWd_per_MtU)
    np.savetxt(os.path.join(run_dir, "depletion_keff_data.csv"),
               np.column_stack(cols), delimiter=",", header=header, comments="")

    # Text report
    txt_path = os.path.join(run_dir, "depletion_results.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DEPLETION SIMULATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Thermal power:    {thermal_power_MW} MW\n")
        if total_HM_mass_kg:
            f.write(f"Initial HM mass:  {total_HM_mass_kg:.2f} kg\n")
        if total_B10_mass_kg:
            f.write(f"Initial B-10:     {total_B10_mass_kg:.4f} kg\n")
        f.write(f"Depletion steps:  {len(keff_mean)}\n")
        f.write(f"Total time:       {time_days[-1]:.1f} days ({time_years[-1]:.2f} years)\n\n")
        f.write(f"Initial k_eff:    {keff_mean[0]:.5f} ± {keff_std[0]:.5f}\n")
        f.write(f"Final k_eff:      {keff_mean[-1]:.5f} ± {keff_std[-1]:.5f}\n\n")
        if burnup_MWd_per_MtU is not None:
            f.write(f"Final burnup:     {burnup_MWd_per_MtU[-1]:.0f} MWd/MtU\n\n")
        if discharge_time_days is not None:
            f.write(f"Discharge (k=1.0): {discharge_time_days:.1f} days "
                    f"({discharge_time_years:.2f} years)\n")
            if discharge_burnup is not None:
                f.write(f"Discharge burnup:  {discharge_burnup:.0f} MWd/MtU\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write(f"{'Step':>5}  {'Time (d)':>10}  {'k_eff':>10}  {'± σ':>8}")
        if burnup_MWd_per_MtU is not None:
            f.write(f"  {'Burnup (MWd/MtU)':>18}")
        f.write("\n" + "-" * 70 + "\n")
        for i in range(len(keff_mean)):
            f.write(f"{i:>5}  {time_days[i]:>10.1f}  {keff_mean[i]:>10.5f}  {keff_std[i]:>8.5f}")
            if burnup_MWd_per_MtU is not None:
                f.write(f"  {burnup_MWd_per_MtU[i]:>18.0f}")
            f.write("\n")

        # Isotopic summary — fuel
        f.write("\n" + "=" * 80 + "\n")
        f.write("FUEL ISOTOPIC INVENTORY SUMMARY (atoms)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Nuclide':<12}  {'Initial':>16}  {'Final':>16}  {'Change (%)':>12}\n")
        f.write("-" * 62 + "\n")
        for nuc in tracked_nuclides:
            if nuc in fuel_data:
                initial = fuel_data[nuc][0]
                final   = fuel_data[nuc][-1]
                pct     = (final - initial) / initial * 100 if initial > 0 else float('nan')
                f.write(f"{nuc:<12}  {initial:>16.4e}  {final:>16.4e}  {pct:>+12.2f}%\n")

        # Isotopic summary — poison
        if poison_data:
            f.write("\n" + "=" * 80 + "\n")
            f.write("BURNABLE POISON ISOTOPIC INVENTORY SUMMARY (atoms)\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Nuclide':<12}  {'Initial':>16}  {'Final':>16}  {'Change (%)':>12}\n")
            f.write("-" * 62 + "\n")
            for nuc, atoms in poison_data.items():
                initial = atoms[0]
                final   = atoms[-1]
                pct     = (final - initial) / initial * 100 if initial > 0 else float('nan')
                f.write(f"{nuc:<12}  {initial:>16.4e}  {final:>16.4e}  {pct:>+12.2f}%\n")

        f.write("=" * 80 + "\n")

    print(f"  Report saved to: {txt_path}")
    print(f"\n{'=' * 80}")
    print("DEPLETION POST-PROCESSING COMPLETE")
    print(f"{'=' * 80}\n")

    return summary

def _plot_nuclide_group(x_data, x_label, nuclide_data, nuclide_list,
                        title, run_dir, filename_base):
    available = [
        n for n in nuclide_list
        if n in nuclide_data and np.any(nuclide_data[n] > 0)
    ]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for nuc in available:
        atoms = nuclide_data[nuc]
        n     = min(len(x_data), len(atoms))
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
        print(f"Loaded parameters from run_params.json")
        print(f"Tracking {len(params.get('tracked_nuclides', []))} fuel nuclides")
        print(f"Tracking {len(params.get('poison_tracked_nuclides', []))} poison nuclides")
    else:
        print("WARNING: run_params.json not found, using defaults")
        params = {}

    run_depletion_postprocessing(run_dir, params)