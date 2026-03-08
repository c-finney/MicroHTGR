"""
Depletion Post-Processing Script

Extracts and plots results from OpenMC depletion simulations:
 - k_eff vs. burnup/time
 - Nuclide inventories vs. burnup (driven by params["tracked_nuclides"])
 - Discharge burnup and cycle length estimates
 - Fissile inventory ratios
 - B-10 burnout from burnable poison material
 - Conversion ratio vs. burnup

BeO reflector fluence analysis is handled by BeO_depletion_postprocessing.py.

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
import openmc
import openmc.deplete

from BeO_depletion_postprocessing import (
    extract_beo_peak_fluence,
    plot_and_save_beo_results,
)

# Default plot groups — used only if not specified in params
DEFAULT_PLOT_GROUPS = {
    "Fissile Actinides": ["U235",
                          "Pu239", "Pu241"],
    "Fertile Actinides": ["U238", "U234", "U236",
                          "Pu238", "Pu240", "Pu242"],
    "Minor Actinides":   ["Np237", "Np239",
                          "Am241", "Am243",
                          "Cm242", "Cm243", "Cm244", "Cm245", "Cm246"],
    "Xe/I Poisons":      ["Xe131", "Xe135", "Xe135_m1",
                          "I135"],
    "Sm/Pm Poisons":     ["Sm149", "Sm151", "Sm152",
                          "Pm147", "Pm149"],
    "Cs/Sr FPs":         ["Cs133", "Cs134", "Cs137",
                          "Sr90"],
    "Nd/Eu FPs":         ["Nd143", "Nd145", "Nd147",
                          "Eu153", "Eu154", "Eu155"],
    "Mo/Tc/Rh/Pd FPs":   ["Mo95",
                          "Tc99",
                          "Rh103", "Rh105",
                          "Pd107"],
    "Kr FPs":            ["Kr83"],
    "Boron Poisons":     ["B10"],
}

# ====================================================================================================
# FIND MATERIAL IDS
# ====================================================================================================

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

# ====================================================================================================
# NUCLIDE DATA EXTRACTION
# ====================================================================================================

def _extract_nuclide_inventories(results, mat_id, nuclide_list, label, symmetry_factor=1):
    """
    Extract atom inventories for a list of nuclides from a single material.

    Args:
        results:          openmc.deplete.Results object
        mat_id:           material ID string
        nuclide_list:     list of nuclide name strings
        label:            human-readable label for print messages
        symmetry_factor:  geometry multiplier (6 for 1/6 wedge, 1 for full core)

    Returns:
        dict: {nuclide_name: np.ndarray of atom counts per timestep}
              Only nuclides with at least one nonzero value are included.
              Atom counts are scaled to full core by symmetry_factor.
    """

    if mat_id is None:
        return {}

    nuclide_data = {}
    failed = []

    for nuc in nuclide_list:
        try:
            _time, atoms = results.get_atoms(mat_id, nuc)
            atoms = np.array(atoms) * symmetry_factor
            if np.any(atoms > 0):
                nuclide_data[nuc] = atoms
        except Exception:
            failed.append(nuc)

    if failed:
        print(f"   [{label}] Not found in chain/results: {failed}")

    scale_note = f" (×{symmetry_factor} for full core)" if symmetry_factor > 1 else ""
    print(f"   [{label}] Extracted {len(nuclide_data)} / {len(nuclide_list)} nuclides{scale_note}")
    return nuclide_data


def _extract_nuclide_inventories_multi(results, mat_ids, nuclide_list, label, symmetry_factor=1):
    """
    Extract atom inventories summed across multiple materials (spatial burnup zones).

    Used when use_spatial_burnup=True: each ring×axial band has its own material ID.
    Atoms from all zones are summed to give core-total inventories.

    Args:
        results:         openmc.deplete.Results object
        mat_ids:         list of material ID strings (all fuel zone materials)
        nuclide_list:    list of nuclide name strings
        label:           human-readable label for print messages
        symmetry_factor: geometry multiplier (6 for 1/6 wedge, 1 for full core)

    Returns:
        dict: {nuclide_name: np.ndarray of summed atom counts per timestep}
              Scaled to full core by symmetry_factor.
    """
    if not mat_ids:
        return {}

    # Determine expected number of timesteps from the first valid material
    n_steps = None
    for mid in mat_ids:
        try:
            _t, _a = results.get_atoms(str(mid), nuclide_list[0])
            n_steps = len(_a)
            break
        except Exception:
            continue
    if n_steps is None:
        print(f"   [{label}] WARNING: Could not determine timestep count from any material")
        return {}

    summed = {}   # nuc -> np.ndarray (n_steps,)
    found_mats = 0
    missing_mats = 0

    for mid in mat_ids:
        mid_str = str(mid)
        mat_found = False
        for nuc in nuclide_list:
            try:
                _t, atoms = results.get_atoms(mid_str, nuc)
                atoms = np.array(atoms, dtype=float)
                # Pad or trim to n_steps to keep arrays aligned
                if len(atoms) < n_steps:
                    atoms = np.concatenate([atoms, np.zeros(n_steps - len(atoms))])
                else:
                    atoms = atoms[:n_steps]
                summed[nuc] = summed.get(nuc, np.zeros(n_steps)) + atoms
                mat_found = True
            except Exception:
                pass
        if mat_found:
            found_mats += 1
        else:
            missing_mats += 1

    if missing_mats:
        print(f"   [{label}] {missing_mats}/{len(mat_ids)} zone materials had no tracked nuclides")

    # Apply symmetry factor and filter zeros
    nuclide_data = {}
    failed = []
    for nuc in nuclide_list:
        if nuc in summed and np.any(summed[nuc] > 0):
            nuclide_data[nuc] = summed[nuc] * symmetry_factor
        else:
            failed.append(nuc)

    if failed:
        print(f"   [{label}] Not found in any zone: {failed}")

    scale_note = f" (×{symmetry_factor} for full core)" if symmetry_factor > 1 else ""
    print(f"   [{label}] Summed {found_mats}/{len(mat_ids)} zones, "
          f"extracted {len(nuclide_data)}/{len(nuclide_list)} nuclides{scale_note}")
    return nuclide_data

# ====================================================================================================
# PEAK BURNUP EXTRACTION (spatial burnup)
# ====================================================================================================

def extract_peak_burnup(results, fuel_mat_ids_2d, fuel_mat_volumes,
                        time_days, burnup_MWd_per_MtU,
                        symmetry_factor=1):
    """
    Compute per-zone and peak burnup at every depletion timestep.

    Method
    ------
    Uses U-235 fractional depletion as a proxy for local burnup:

        f_zone[t]  = (N_U235_zone_0 - N_U235_zone[t]) / N_U235_zone_0
        f_core[t]  = (N_U235_core_0 - N_U235_core[t]) / N_U235_core_0

    Peak burnup:
        BU_peak[t] = max_zone(f_zone[t]) / f_core[t] * BU_avg[t]

    where BU_avg[t] = total cumulative energy / total HM mass (from burnup_MWd_per_MtU).

    Parameters
    ----------
    results             : openmc.deplete.Results
    fuel_mat_ids_2d     : list[list[int]] — [ring][axial_band] material IDs
    fuel_mat_volumes    : dict str(mat_id) -> float (cm3, simulated geometry)
    time_days           : np.ndarray  shape (n_steps,)
    burnup_MWd_per_MtU  : np.ndarray or None — core-average burnup per step
    symmetry_factor     : int (6 for 1/6 wedge, 1 for full core)

    Returns
    -------
    dict with keys:
        'zone_burnup_fraction'  : (n_rings, n_bands, n_steps) array — f_zone[t]
        'zone_labels'           : list of "(ring, band)" strings
        'peak_burnup_MWd_MtU'  : (n_steps,) array — peak zone burnup
        'peak_zone'             : (n_steps,) int array — flat zone index of peak
        'avg_burnup_MWd_MtU'   : (n_steps,) array — same as burnup_MWd_per_MtU
    None if burnup_MWd_per_MtU is not available or fuel_mat_ids_2d is missing.
    """

    if burnup_MWd_per_MtU is None or fuel_mat_ids_2d is None:
        return None

    n_rings = len(fuel_mat_ids_2d)
    n_bands = len(fuel_mat_ids_2d[0])
    n_steps = len(time_days)

    # Collect per-zone U235 atom arrays
    zone_u235 = np.zeros((n_rings, n_bands, n_steps))
    zone_ok    = np.zeros((n_rings, n_bands), dtype=bool)

    for ring_idx, row in enumerate(fuel_mat_ids_2d):
        for bax_idx, mat_id in enumerate(row):
            mid_str = str(mat_id)
            try:
                _, atoms = results.get_atoms(mid_str, "U235")
                atoms = np.array(atoms, dtype=float)
                if len(atoms) >= n_steps:
                    zone_u235[ring_idx, bax_idx, :] = atoms[:n_steps]
                else:
                    zone_u235[ring_idx, bax_idx, :len(atoms)] = atoms
                zone_ok[ring_idx, bax_idx] = True
            except Exception:
                pass

    n_good = int(np.sum(zone_ok))
    if n_good == 0:
        print("  WARNING: No U235 data found for any zone — cannot compute peak burnup")
        return None
    print(f"  Peak burnup: using U235 depletion from {n_good}/{n_rings * n_bands} zones")

    # Core-total U235 (sum all zones, scaled to full core)
    core_u235_total = zone_u235.sum(axis=(0, 1)) * symmetry_factor   # (n_steps,)
    u235_core_0 = core_u235_total[0]
    if u235_core_0 <= 0:
        print("  WARNING: zero initial U235 — cannot compute peak burnup")
        return None

    f_core = (u235_core_0 - core_u235_total) / u235_core_0           # (n_steps,)
    f_core = np.where(f_core <= 0, 1e-30, f_core)   # avoid divide-by-zero at step 0

    # Per-zone fractional depletion
    u235_zone_0 = zone_u235[:, :, 0]                                  # (n_rings, n_bands)
    u235_zone_0_safe = np.where(u235_zone_0 > 0, u235_zone_0, np.nan)

    # f_zone shape: (n_rings, n_bands, n_steps)
    zone_burnup_frac = ((u235_zone_0_safe[:, :, np.newaxis] - zone_u235)
                        / u235_zone_0_safe[:, :, np.newaxis])
    zone_burnup_frac = np.nan_to_num(zone_burnup_frac, nan=0.0)

    # Relative burnup of each zone vs. core average
    # BU_zone[t] = (f_zone[t] / f_core[t]) * BU_avg[t]
    with np.errstate(invalid='ignore'):
        zone_bu = (zone_burnup_frac / f_core[np.newaxis, np.newaxis, :]) * \
                   burnup_MWd_per_MtU[np.newaxis, np.newaxis, :]

    # Peak: maximum over all zones at each timestep
    zone_bu_flat = zone_bu.reshape(n_rings * n_bands, n_steps)        # flatten rings/bands
    only_ok = zone_ok.flatten()
    zone_bu_flat[~only_ok, :] = np.nan

    peak_burnup = np.nanmax(zone_bu_flat, axis=0)                     # (n_steps,)
    peak_zone_flat = np.nanargmax(zone_bu_flat, axis=0)               # flat index

    min_burnup = np.nanmin(zone_bu_flat, axis=0)                      # (n_steps,)
    min_zone_flat = np.nanargmin(zone_bu_flat, axis=0)                # flat index

    zone_labels = [f"ring{r}_band{b}"
                   for r in range(n_rings)
                   for b in range(n_bands)]

    print(f"  Peak burnup at final step: "
          f"{peak_burnup[-1]:.0f} MWd/MtU  "
          f"(avg = {burnup_MWd_per_MtU[-1]:.0f}  "
          f"peaking = {peak_burnup[-1]/max(burnup_MWd_per_MtU[-1],1):.2f}×  "
          f"zone = {zone_labels[peak_zone_flat[-1]]})")
    print(f"  Min  burnup at final step: "
          f"{min_burnup[-1]:.0f} MWd/MtU  "
          f"zone = {zone_labels[min_zone_flat[-1]]}")

    return {
        "zone_burnup_fraction": zone_burnup_frac,
        "zone_labels":          zone_labels,
        "peak_burnup_MWd_MtU": peak_burnup,
        "peak_zone":            peak_zone_flat,
        "min_burnup_MWd_MtU":  min_burnup,
        "min_zone":             min_zone_flat,
        "avg_burnup_MWd_MtU":  burnup_MWd_per_MtU,
    }


# ====================================================================================================
# NUCLIDE INVENTORY CSV EXPORT
# ====================================================================================================

def save_nuclide_inventory_csv(output_dir, time_days, time_years, burnup_MWd_per_MtU, fuel_data, poison_data, operational=None):
    """
    Save per-timestep nuclide atom inventories to CSV files.

    Writes two files:
      - nuclide_inventory_fuel.csv   — one column per fuel nuclide, rows = timesteps
      - nuclide_inventory_poison.csv — one column per poison nuclide, rows = timesteps
                                       (only if poison_data is non-empty)

    Index columns in both files:
      step, time_days, time_years[, burnup_MWd_per_MtU][, operational]

    Args:
        output_dir          : output directory
        time_days           : np.ndarray, shape (n_steps,)
        time_years          : np.ndarray, shape (n_steps,)
        burnup_MWd_per_MtU  : np.ndarray or None
        fuel_data           : dict {nuclide: np.ndarray}
        poison_data         : dict {nuclide: np.ndarray}
        operational         : np.ndarray of int (0/1) or None

    Returns:
        list of str: paths of files written
    """

    n_steps   = len(time_days)
    steps_col = np.arange(n_steps)

    # Build common index columns and header prefix
    index_cols   = [steps_col, time_days, time_years]
    index_header = "step,time_days,time_years"
    if burnup_MWd_per_MtU is not None:
        index_cols.append(burnup_MWd_per_MtU)
        index_header += ",burnup_MWd_per_MtU"
    if operational is not None:
        index_cols.append(operational[:n_steps])
        index_header += ",operational"

    written = []

    for label, data, filename in [
        ("Fuel",           fuel_data,   "nuclide_inventory_fuel.csv"),
        ("Burnable Poison", poison_data, "nuclide_inventory_poison.csv"),
    ]:
        if not data:
            print(f"  [{label}] No nuclide data to export, skipping {filename}")
            continue

        # Align all arrays to n_steps (trim if a nuclide array is longer)
        nuclides = sorted(data.keys())
        nuc_cols = []
        for nuc in nuclides:
            arr = data[nuc]
            if len(arr) >= n_steps:
                nuc_cols.append(arr[:n_steps])
            else:
                # Pad with NaN if shorter (shouldn't happen, but be safe)
                padded = np.full(n_steps, np.nan)
                padded[:len(arr)] = arr
                nuc_cols.append(padded)
                print(f"  [{label}] WARNING: {nuc} has {len(arr)} steps vs {n_steps}; padded with NaN")

        all_cols  = index_cols + nuc_cols
        header    = index_header + "," + ",".join(nuclides)
        out_path  = os.path.join(output_dir, filename)

        np.savetxt(
            out_path,
            np.column_stack(all_cols),
            delimiter=",",
            header=header,
            comments="",
            fmt=(["%.0f", "%.6f", "%.8f"]
                 + (["%.4f"] if burnup_MWd_per_MtU is not None else [])
                 + (["%.0f"] if operational is not None else [])
                 + ["%.6e"] * len(nuclides)),
        )

        print(f"  [{label}] Nuclide inventory saved → {out_path}  "
              f"({n_steps} steps × {len(nuclides)} nuclides)")
        written.append(out_path)

    return written

# ====================================================================================================
# CONVERSION RATIO CALCULATION
# ====================================================================================================

def calculate_conversion_ratio(fuel_data):
    """
    Calculate the approximate conversion ratio (CR) at each depletion timestep.

    For each step i → i+1:
      - U235_burned      = max(0, U235[i] - U235[i+1])
      - Pu239_burned     = max(0, Pu239[i] - Pu239[i+1])   (atoms of Pu-239 consumed)
      - Pu241_burned     = max(0, Pu241[i] - Pu241[i+1])
      - total_fissile_burned = U235_burned + Pu239_burned + Pu241_burned
      - Pu239_generated  = (Pu239[i+1] - Pu239[i]) + Pu239_burned
                         = gross Pu-239 production in the step
      - CR[i] = total_fissile_burned / max(1, Pu239_generated)

    Step 0 (BOL → first burnup point) uses all U-235 as the fissile source
    (Pu239 = 0 initially), giving a firm baseline CR.

    Parameters
    ----------
    fuel_data : dict {nuclide: np.ndarray}
        Must contain at least 'U235' and 'Pu239'; 'Pu241' optional.

    Returns
    -------
    dict with keys:
        'CR'               : (n_steps-1,) array — conversion ratio per step
        'U235_burned'      : (n_steps-1,) array — U-235 atoms burned per step
        'Pu239_burned'     : (n_steps-1,) array — Pu-239 atoms burned per step
        'Pu239_generated'  : (n_steps-1,) array — gross Pu-239 atoms generated per step
        'total_fissile_burned' : (n_steps-1,) array
    None if U235 or Pu239 are not in fuel_data.
    """
    if "U235" not in fuel_data or "Pu239" not in fuel_data:
        return None

    u235  = fuel_data["U235"]
    pu239 = fuel_data["Pu239"]
    n     = len(u235)

    if n < 2:
        return None

    pu241 = fuel_data.get("Pu241", np.zeros(n))

    # Align lengths
    min_n = min(len(u235), len(pu239), len(pu241))
    u235  = u235[:min_n]
    pu239 = pu239[:min_n]
    pu241 = pu241[:min_n]
    n     = min_n

    u235_burned  = np.maximum(0.0, u235[:-1]  - u235[1:])
    pu239_burned = np.maximum(0.0, pu239[:-1] - pu239[1:])
    pu241_burned = np.maximum(0.0, pu241[:-1] - pu241[1:])

    total_fissile_burned = u235_burned + pu239_burned + pu241_burned

    # Gross Pu-239 production = net change + atoms burned in this step
    pu239_net_change  = pu239[1:] - pu239[:-1]
    pu239_generated   = pu239_net_change + pu239_burned   # always ≥ 0 by construction

    # Guard against zero generation (early steps before Pu builds up)
    cr = np.where(pu239_generated > 0,
                  total_fissile_burned / pu239_generated,
                  np.nan)

    return {
        "CR":                   cr,
        "U235_burned":          u235_burned,
        "Pu239_burned":         pu239_burned,
        "Pu239_generated":      pu239_generated,
        "total_fissile_burned": total_fissile_burned,
    }


# ====================================================================================================
# PERFORM DEPLETION ANALYSIS PLOTTING AND SAVE RESULTS
# ====================================================================================================

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

    show_titles = params.get("show_titles", True)

    print(f"\n{'=' * 80}")
    print("DEPLETION POST-PROCESSING")
    print(f"{'=' * 80}")
    print(f"Run directory: {run_dir}")

    POSTPROCESSING_RESULTS_DIR = os.path.join(run_dir, "depletion_results")
    os.makedirs(POSTPROCESSING_RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(run_dir, "depletion_results.h5")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found!")
        return None

    results = openmc.deplete.Results(results_path)

    # ================================================================================
    # 0. GEOMETRY SYMMETRY FACTOR
    # ================================================================================

    is_wedge = params.get("use_1/6_geometry", False)
    symmetry_factor = 6 if is_wedge else 1

    # ================================================================================
    # 1. K-EFFECTIVE VS. TIME/BURNUP
    # ================================================================================

    time_steps, keff_values = results.get_keff()

    # Handle both older OpenMC (uncertainties objects with .nominal_value/.std_dev)
    # and newer OpenMC (plain numpy array of shape [n_steps, 2])
    if hasattr(keff_values[0], 'nominal_value'):
        keff_mean = np.array([k.nominal_value for k in keff_values])
        keff_std  = np.array([k.std_dev       for k in keff_values])
    else:
        # Newer API: keff_values is shape (n_steps, 2) — columns are [mean, std_dev]
        keff_values = np.array(keff_values)
        keff_mean   = keff_values[:, 0]
        keff_std    = keff_values[:, 1]

    time_days  = time_steps / 86400.0
    time_years = time_days  / 365.25

    # Operational flag: 1 if k_eff >= 1.0 at that step, 0 otherwise
    operational = (keff_mean >= 1.0).astype(int)
    op_indices = np.where(keff_mean >= 1.0)[0]
    last_operational_idx = int(op_indices[-1]) if len(op_indices) > 0 else 0

    thermal_power_MW  = params.get("thermal_power_MW", 10.0)
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

    # ================================================================================
    # 2. DISCHARGE BURNUP (k_eff crosses 1.0)
    # ================================================================================

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

    # ================================================================================
    # 3. LOCATE MATERIALS AND EXTRACT NUCLIDE INVENTORIES
    # ================================================================================

    print("\n   Locating materials in depletion results...")

    tracked_nuclides        = params.get("tracked_nuclides", ["U235", "U238", "Pu239", "B10"])
    poison_tracked_nuclides = params.get("poison_tracked_nuclides", ["B10"])

    # Exclude poison-specific nuclides from fuel search to avoid confusion
    fuel_nuclides = [n for n in tracked_nuclides if n not in poison_tracked_nuclides]

    # ---- Fuel: spatial burnup (multiple materials) vs. single material ----
    fuel_mat_ids_2d = params.get("fuel_mat_ids", None)

    if fuel_mat_ids_2d:
        # Spatial burnup: 2D list [ring][axial_band] of material IDs.
        # Flatten and sum across all zones → full-core inventory.
        all_fuel_ids = [str(mid)
                        for row in fuel_mat_ids_2d
                        for mid in row]
        # Deduplicate (use_spatial_burnup=False can share a single material)
        seen = set()
        unique_fuel_ids = []
        for mid in all_fuel_ids:
            if mid not in seen:
                seen.add(mid)
                unique_fuel_ids.append(mid)

        n_rings = len(fuel_mat_ids_2d)
        n_bands = len(fuel_mat_ids_2d[0]) if fuel_mat_ids_2d else 0
        print(f"\n   Spatial burnup: summing {len(unique_fuel_ids)} fuel zones "
              f"({n_rings} rings × {n_bands} axial bands)...")
        fuel_data = _extract_nuclide_inventories_multi(
            results, unique_fuel_ids, fuel_nuclides, "Fuel", symmetry_factor
        )
    else:
        # Non-spatial: single fuel material ID
        fuel_mat_id = _find_material_id(results, params,
                                        "fuel_material_id", "U235", "Fuel")
        print(f"\n   Extracting fuel inventories ({len(fuel_nuclides)} nuclides)...")
        fuel_data = _extract_nuclide_inventories(results, fuel_mat_id, fuel_nuclides,
                                                 "Fuel", symmetry_factor)

    poison_mat_id = _find_material_id(results, params,
                                       "poison_material_id", "B10",  "Burnable poison")
    print(f"\n   Extracting burnable poison inventories ({len(poison_tracked_nuclides)} nuclides)...")
    poison_data = _extract_nuclide_inventories(results, poison_mat_id, poison_tracked_nuclides,
                                                "Poison", symmetry_factor)

    # Merge for plotting — poison data keyed separately to avoid name collision
    # B10 from poison material is canonical; if also in fuel, prefer poison
    all_nuclide_data = {**fuel_data}
    for nuc, atoms in poison_data.items():
        all_nuclide_data[f"{nuc}_poison"] = atoms  # keep separate key
        all_nuclide_data[nuc] = atoms               # also overwrite top-level with poison value

    # ================================================================================
    # 4. PRINT SUMMARY
    # ================================================================================

    geom_label = "1/6 wedge" if is_wedge else "full core"

    print(f"\n{'─' * 60}")
    print("  DEPLETION RESULTS SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Geometry:           {geom_label}")
    print(f"  Depletion steps:    {len(keff_mean)}")
    print(f"  Total time:         {time_days[-1]:.1f} days ({time_years[-1]:.2f} years)")
    print(f"  Initial k_eff:      {keff_mean[0]:.5f} ± {keff_std[0]:.5f}")
    print(f"  Final k_eff:        {keff_mean[-1]:.5f} ± {keff_std[-1]:.5f}")
    if burnup_MWd_per_MtU is not None:
        print(f"  Final burnup:       {burnup_MWd_per_MtU[-1]:.0f} MWd/MtU")
    if total_HM_mass_kg:
        print(f"  Initial HM mass:    {total_HM_mass_kg:.2f} kg ")
    if total_B10_mass_kg:
        print(f"  Initial B-10 mass:  {total_B10_mass_kg:.4f} kg ")

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
            final   = all_nuclide_data[nuc][last_operational_idx]
            if initial > 0:
                pct = (1.0 - final / initial) * 100
                print(f"\n  {nuc} [{source_label}]: {initial:.4e} → {final:.4e} atoms  "
                      f"({pct:.1f}% depleted, at last operational step {last_operational_idx})  ")

    if "Pu239" in fuel_data and "U235" in fuel_data and fuel_data["U235"][0] > 0:
        pu_ratio = fuel_data["Pu239"][-1] / fuel_data["U235"][0] * 100
        print(f"\n  Pu-239 final (% of initial U-235 atoms): {pu_ratio:.2f}%")

    # BeO fluence summary (placeholder — filled after extraction below)
    print(f"{'─' * 60}")

    # ================================================================================
    # 5. PLOTTING
    # ================================================================================

    print("\nGenerating depletion plots...")

    # k_eff vs burnup/time
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.errorbar(x_data, keff_mean, yerr=keff_std, fmt="o-", capsize=3,
                markersize=5, linewidth=1.5, label="k-effective")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1, label="k = 1.0")
    if discharge_burnup is not None and burnup_MWd_per_MtU is not None:
        ax.axvline(discharge_burnup, color="green", linestyle=":", alpha=0.7,
                   label=f"Discharge: {discharge_burnup:.0f} MWd/MtU")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("k-effective", fontsize=12)
    if show_titles:
        ax.set_title("k-effective vs. Burnup", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR, f"depletion_keff_vs_{x_label_short}.png"), bbox_inches="tight")
    plt.close()

    # k_eff vs time with year secondary axis
    if burnup_MWd_per_MtU is not None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        ax.errorbar(time_days, keff_mean, yerr=keff_std, fmt="o-", capsize=3,
                    markersize=5, linewidth=1.5, label="k-effective")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1, label="k = 1.0")
        if discharge_time_years is not None and discharge_time_days is not None:
            ax.axvline(discharge_time_days, color="green", linestyle=":", alpha=0.7,
                       label=f"Discharge: {discharge_time_years:.2f} years")
        ax.set_xlabel("Time (days)", fontsize=12)
        ax.set_ylabel("k-effective", fontsize=12)
        if show_titles:
            ax.set_title("k-effective vs. Time", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()[0] / 365.25, ax.get_xlim()[1] / 365.25)
        ax2.set_xlabel("Time (years)", fontsize=11)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR, "depletion_keff_vs_time.png"), bbox_inches="tight")
        plt.close()

    # Reactivity (pcm)
    reactivity_pcm = (keff_mean - 1.0) / keff_mean * 1e5
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(x_data, reactivity_pcm, "o-", markersize=5, linewidth=1.5)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Reactivity (pcm)", fontsize=12)
    if show_titles:
        ax.set_title("Excess Reactivity vs. Burnup", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR, f"depletion_reactivity_vs_{x_label_short}.png"), bbox_inches="tight")
    plt.close()

    # Nuclide group plots — driven entirely by params["depletion_plot_groups"]
    plot_groups   = params.get("depletion_plot_groups", DEFAULT_PLOT_GROUPS)
    plotted_nuclides = set()

    for group_name, group_nuclides in plot_groups.items():
        available = [n for n in group_nuclides if n in all_nuclide_data]
        if available:
            _plot_nuclide_group(
                x_data, x_label, all_nuclide_data, available,
                group_name, POSTPROCESSING_RESULTS_DIR,
                f"depletion_{group_name.lower().replace('/', '').replace(' ', '_')}",
                is_wedge=is_wedge, show_titles=show_titles
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
            "Other Tracked Nuclides", POSTPROCESSING_RESULTS_DIR,
            "depletion_other_nuclides",
            is_wedge=is_wedge, show_titles=show_titles
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
        if show_titles:
            ax.set_title(
                f"Fissile Inventory Ratio vs. Burnup\n"
                f"({' + '.join(fissile_present)}) / Initial U-235",
                fontsize=13
            )
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR, f"depletion_fissile_ratio_vs_{x_label_short}.png"),
                    bbox_inches="tight")
        plt.close()

    # B-10 burnout — two-panel: absolute atoms and fractional remaining
    if "B10" in all_nuclide_data:
        b10         = all_nuclide_data["B10"]
        b10_initial = b10[0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        ax1.plot(x_data, b10, "o-", markersize=4, linewidth=1.5, color="purple")
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel("B-10 Atoms", fontsize=12)
        if show_titles:
            ax1.set_title("B-10 Absolute Inventory (Burnable Poison)", fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        if b10_initial > 0:
            ax2.plot(x_data, b10 / b10_initial * 100, "o-", markersize=4,
                     linewidth=1.5, color="darkviolet")
            ax2.set_xlabel(x_label, fontsize=12)
            ax2.set_ylabel("Remaining B-10 (%)", fontsize=12)
            if show_titles:
                ax2.set_title("B-10 Fractional Burnout", fontsize=13)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR, "depletion_B10_burnout.png"), bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_B10_burnout.png")

    # ================================================================================
    # 5b. PEAK BURNUP (spatial burnup only)
    # ================================================================================

    fuel_mat_volumes_raw = params.get("fuel_mat_volumes", {})
    fuel_mat_volumes_str = {str(k): float(v) for k, v in fuel_mat_volumes_raw.items()}

    peak_burnup_data = extract_peak_burnup(
        results            = results,
        fuel_mat_ids_2d    = fuel_mat_ids_2d,
        fuel_mat_volumes   = fuel_mat_volumes_str,
        time_days          = time_days,
        burnup_MWd_per_MtU = burnup_MWd_per_MtU,
        symmetry_factor    = symmetry_factor,
    )

    if peak_burnup_data is not None:
        peak_bu    = peak_burnup_data["peak_burnup_MWd_MtU"]
        min_bu     = peak_burnup_data["min_burnup_MWd_MtU"]
        avg_bu     = peak_burnup_data["avg_burnup_MWd_MtU"]
        peak_zone  = peak_burnup_data["peak_zone"]
        min_zone   = peak_burnup_data["min_zone"]
        zone_lbl   = peak_burnup_data["zone_labels"]

        # Peak / average / minimum burnup plot
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        ax.plot(x_data, avg_bu,  "o-", markersize=5, linewidth=1.5,
                color="black", label="Core-average burnup")
        ax.plot(x_data, peak_bu, "s--", markersize=5, linewidth=1.5,
                color="firebrick", label="Peak zone burnup")
        ax.plot(x_data, min_bu,  "^--", markersize=5, linewidth=1.5,
                color="steelblue", label="Minimum zone burnup")
        ax.fill_between(x_data, avg_bu, peak_bu, alpha=0.10, color="firebrick",
                        label="Peak-to-average margin")
        ax.fill_between(x_data, min_bu, avg_bu, alpha=0.10, color="steelblue",
                        label="Average-to-minimum margin")
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Burnup (MWd/MtU)", fontsize=12)
        if show_titles:
            ax.set_title("Peak / Average / Minimum Burnup\n(U-235 depletion proxy)", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_peak_burnup_vs_{x_label_short}.png"),
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_peak_burnup_vs_{x_label_short}.png")

        # Peak-to-average burnup ratio vs burnup
        with np.errstate(divide='ignore', invalid='ignore'):
            pf_burnup = np.where(avg_bu > 0, peak_bu / avg_bu, np.nan)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        ax.plot(x_data, pf_burnup, "o-", markersize=5, linewidth=1.5, color="darkorange")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Peak-to-Average Burnup Ratio", fontsize=12)
        if show_titles:
            ax.set_title("Burnup Peaking Factor vs. Burnup", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_burnup_peaking_vs_{x_label_short}.png"),
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_burnup_peaking_vs_{x_label_short}.png")

        # CSV: avg, peak, and min burnup per step
        peak_csv = os.path.join(POSTPROCESSING_RESULTS_DIR, "depletion_peak_burnup.csv")
        header = ("step,time_days,time_years"
                  + (",burnup_avg_MWd_MtU" if burnup_MWd_per_MtU is not None else "")
                  + ",burnup_peak_MWd_MtU,burnup_min_MWd_MtU,peak_to_avg_ratio,peak_zone,min_zone,operational")
        rows = []
        for i in range(len(time_days)):
            row = f"{i},{time_days[i]:.4f},{time_years[i]:.6f}"
            if burnup_MWd_per_MtU is not None:
                row += f",{avg_bu[i]:.2f}"
            pf_val = float(pf_burnup[i]) if not np.isnan(pf_burnup[i]) else 0.0
            op_val = int(operational[i])
            row += (f",{peak_bu[i]:.2f},{min_bu[i]:.2f},{pf_val:.4f},"
                    f"{zone_lbl[int(peak_zone[i])]},{zone_lbl[int(min_zone[i])]},{op_val}")
            rows.append(row)
        with open(peak_csv, "w") as f:
            f.write(header + "\n")
            for r in rows:
                f.write(r + "\n")
        print(f"  Saved: {peak_csv}")

    # ================================================================================
    # 5c. BEO PEAK FLUENCE (requires per-step statepoints)
    # ================================================================================

    beo_fluence_data = extract_beo_peak_fluence(
        run_dir       = run_dir,
        time_steps_s  = time_steps,           # seconds, from results.get_keff()
        keff_mean     = keff_mean,
        params        = params,
    )

    if beo_fluence_data is not None:
        plot_and_save_beo_results(
            beo_fluence_data, x_data, x_label, x_label_short,
            time_days, keff_mean, burnup_MWd_per_MtU,
            POSTPROCESSING_RESULTS_DIR, show_titles=show_titles,
        )

    # ================================================================================
    # 5d. CONVERSION RATIO
    # ================================================================================

    cr_data = calculate_conversion_ratio(fuel_data)

    if cr_data is not None:
        cr          = cr_data["CR"]
        pu239_gen   = cr_data["Pu239_generated"]
        fis_burned  = cr_data["total_fissile_burned"]
        n_cr        = len(cr)

        # x-axis mid-points (average of step start and end)
        if burnup_MWd_per_MtU is not None:
            x_cr = 0.5 * (burnup_MWd_per_MtU[:n_cr] + burnup_MWd_per_MtU[1:n_cr + 1])
        else:
            x_cr = 0.5 * (time_days[:n_cr] + time_days[1:n_cr + 1])

        # --- Plot: CR vs burnup/time ---
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        valid = ~np.isnan(cr)
        ax.plot(x_cr[valid], cr[valid], "o-", markersize=5, linewidth=1.5, color="tab:orange")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="CR = 1 (break-even)")
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Conversion Ratio", fontsize=12)
        if show_titles:
            ax.set_title("Conversion Ratio vs. Burnup", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_conversion_ratio_vs_{x_label_short}.png"),
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_conversion_ratio_vs_{x_label_short}.png")

        # --- Plot: Pu-239 generated and fissile burned per step ---
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        ax.plot(x_cr, fis_burned, "o-", markersize=4, linewidth=1.5,
                color="tab:red",   label="Total fissile burned")
        ax.plot(x_cr, pu239_gen,  "s-", markersize=4, linewidth=1.5,
                color="tab:blue",  label="Pu-239 generated (gross)")
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Atoms per step", fontsize=12)
        if show_titles:
            ax.set_title("Fissile Burned vs. Pu-239 Generated per Step", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_fissile_balance_vs_{x_label_short}.png"),
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_fissile_balance_vs_{x_label_short}.png")

        # --- CSV: conversion ratio per step ---
        cr_csv = os.path.join(POSTPROCESSING_RESULTS_DIR, "depletion_conversion_ratio.csv")
        cr_header = "step_mid"
        if burnup_MWd_per_MtU is not None:
            cr_header += ",burnup_mid_MWd_MtU"
        else:
            cr_header += ",time_mid_days"
        cr_header += ",U235_burned,Pu239_burned,Pu239_generated,total_fissile_burned,conversion_ratio"
        with open(cr_csv, "w") as f:
            f.write(cr_header + "\n")
            for i in range(n_cr):
                cr_val = float(cr[i]) if not np.isnan(cr[i]) else float('nan')
                f.write(
                    f"{i},{x_cr[i]:.4f},"
                    f"{cr_data['U235_burned'][i]:.6e},"
                    f"{cr_data['Pu239_burned'][i]:.6e},"
                    f"{pu239_gen[i]:.6e},"
                    f"{fis_burned[i]:.6e},"
                    f"{cr_val:.6f}\n"
                )
        print(f"  Saved: {cr_csv}")

        # Print summary
        valid_cr = cr[valid]
        if len(valid_cr) > 0:
            print(f"\n  Conversion Ratio Summary:")
            print(f"    Initial CR (step 1): {valid_cr[0]:.4f}")
            print(f"    Final   CR (last step): {valid_cr[-1]:.4f}")
            print(f"    Mean CR: {np.mean(valid_cr):.4f}")
    else:
        cr_data = None
        print("  Conversion ratio: skipped (U235 or Pu239 not in fuel_data)")

    # ================================================================================
    # 6. SAVE RESULTS
    # ================================================================================

    summary = {
        "n_steps":                       len(keff_mean),
        "geometry":                      "1/6 wedge" if is_wedge else "full core",
        "symmetry_factor":               symmetry_factor,
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
        "peak_burnup_MWd_per_MtU":      (peak_burnup_data["peak_burnup_MWd_MtU"].tolist()
                                         if peak_burnup_data is not None else None),
        "peak_burnup_final_MWd_per_MtU":(float(peak_burnup_data["peak_burnup_MWd_MtU"][-1])
                                         if peak_burnup_data is not None else None),
        "peak_burnup_zone_final":        (peak_burnup_data["zone_labels"][
                                              int(peak_burnup_data["peak_zone"][-1])]
                                         if peak_burnup_data is not None else None),
        # BeO peak fluence
        "beo_total_peak_fluence_n_cm2":  (beo_fluence_data["total_peak_fluence_n_cm2"]
                                          if beo_fluence_data is not None else None),
        "beo_shutdown_step_idx":         (beo_fluence_data["shutdown_step_idx"]
                                          if beo_fluence_data is not None else None),
        "beo_cumulative_fluence_n_cm2":  (beo_fluence_data["cumulative_fluence_n_cm2"].tolist()
                                          if beo_fluence_data is not None else None),
        "beo_peak_flux_per_step_n_cm2_s":(beo_fluence_data["peak_flux_per_step_n_cm2_s"].tolist()
                                          if beo_fluence_data is not None else None),
        # Conversion ratio
        "conversion_ratio":              (cr_data["CR"].tolist()
                                          if cr_data is not None else None),
        "conversion_ratio_initial":      (float(cr_data["CR"][~np.isnan(cr_data["CR"])][0])
                                          if cr_data is not None and np.any(~np.isnan(cr_data["CR"])) else None),
        "conversion_ratio_final":        (float(cr_data["CR"][~np.isnan(cr_data["CR"])][-1])
                                          if cr_data is not None and np.any(~np.isnan(cr_data["CR"])) else None),
    }

    with open(os.path.join(POSTPROCESSING_RESULTS_DIR, "depletion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ----- k-eff CSV Report -----

    header = "time_days,time_years,keff,keff_std"
    cols   = [time_days, time_years, keff_mean, keff_std]
    if burnup_MWd_per_MtU is not None:
        header += ",burnup_MWd_per_MtU"
        cols.append(burnup_MWd_per_MtU)
    header += ",operational"
    cols.append(operational)
    np.savetxt(os.path.join(POSTPROCESSING_RESULTS_DIR, "depletion_keff_data.csv"),
               np.column_stack(cols), delimiter=",", header=header, comments="")

    # ----- Nuclide Inventory CSV Report -----

    print("\nExporting nuclide inventory CSVs...")
    save_nuclide_inventory_csv(
        POSTPROCESSING_RESULTS_DIR, time_days, time_years, burnup_MWd_per_MtU,
        fuel_data, poison_data, operational=operational
    )

    # ----- Text Report -----

    txt_path = os.path.join(POSTPROCESSING_RESULTS_DIR, "depletion_results.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DEPLETION SIMULATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Geometry:         {geom_label}\n")
        f.write(f"Thermal power:    {thermal_power_MW} MW\n")
        if total_HM_mass_kg:
            f.write(f"Initial HM mass:  {total_HM_mass_kg:.2f} kg \n")
        if total_B10_mass_kg:
            f.write(f"Initial B-10:     {total_B10_mass_kg:.4f} kg \n")
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

        # Isotopic summary — Fuel (final values from last operational step)
        f.write("\n" + "=" * 80 + "\n")
        f.write("FUEL ISOTOPIC INVENTORY SUMMARY (atoms)\n")
        f.write(f"  Final values at last operational step: {last_operational_idx} "
                f"(t = {time_days[last_operational_idx]:.1f} days, "
                f"k = {keff_mean[last_operational_idx]:.5f})\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Nuclide':<12}  {'Initial':>16}  {'Final (op)':>16}  {'Change (%)':>12}\n")
        f.write("-" * 62 + "\n")
        for nuc in tracked_nuclides:
            if nuc in fuel_data:
                initial = fuel_data[nuc][0]
                final   = fuel_data[nuc][last_operational_idx]
                pct     = (final - initial) / initial * 100 if initial > 0 else float('nan')
                f.write(f"{nuc:<12}  {initial:>16.4e}  {final:>16.4e}  {pct:>+12.2f}%\n")

        # Isotopic summary — Poison (final values from last operational step)
        if poison_data:
            f.write("\n" + "=" * 80 + "\n")
            f.write("BURNABLE POISON ISOTOPIC INVENTORY SUMMARY (atoms)\n")
            f.write(f"  Final values at last operational step: {last_operational_idx} "
                    f"(t = {time_days[last_operational_idx]:.1f} days, "
                    f"k = {keff_mean[last_operational_idx]:.5f})\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Nuclide':<12}  {'Initial':>16}  {'Final (op)':>16}  {'Change (%)':>12}\n")
            f.write("-" * 62 + "\n")
            for nuc, atoms in poison_data.items():
                initial = atoms[0]
                final   = atoms[last_operational_idx]
                pct     = (final - initial) / initial * 100 if initial > 0 else float('nan')
                f.write(f"{nuc:<12}  {initial:>16.4e}  {final:>16.4e}  {pct:>+12.2f}%\n")

        f.write("=" * 80 + "\n")

        # BeO fluence report
        if beo_fluence_data is not None:
            f.write("\n" + "=" * 80 + "\n")
            f.write("BEO REFLECTOR PEAK FLUENCE SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total peak fluence: {beo_fluence_data['total_peak_fluence_n_cm2']:.4e} n/cm²\n")
            sd = beo_fluence_data['shutdown_step_idx']
            if sd < len(keff_mean):
                f.write(f"Reactor shutdown: step {sd}  "
                        f"(k_eff = {keff_mean[sd]:.4f} < 1.0,  "
                        f"t = {time_days[sd]:.1f} days)\n")
            else:
                f.write("Reactor remained supercritical throughout all steps\n")
            f.write(f"\n{'Step':>5}  {'Time (d)':>10}  {'k_eff':>8}  "
                    f"{'Peak Flux (n/cm²/s)':>22}  {'Step Fluence (n/cm²)':>22}  "
                    f"{'Cum. Fluence (n/cm²)':>22}\n")
            f.write("-" * 100 + "\n")
            pf_arr = beo_fluence_data['peak_flux_per_step_n_cm2_s']
            sf_arr = beo_fluence_data['step_fluence_n_cm2']
            cf_arr = beo_fluence_data['cumulative_fluence_n_cm2']
            for i in range(n_sp):
                pf = pf_arr[i] if not np.isnan(pf_arr[i]) else 0.0
                sf = sf_arr[i] if not np.isnan(sf_arr[i]) else 0.0
                f.write(f"{i:>5}  {time_days[i]:>10.1f}  {keff_mean[i]:>8.5f}  "
                        f"{pf:>22.4e}  {sf:>22.4e}  {cf_arr[i]:>22.4e}\n")
            f.write("=" * 80 + "\n")

    print(f"  Report saved to: {txt_path}")
    print(f"\n{'=' * 80}")
    print("DEPLETION POST-PROCESSING COMPLETE")
    print(f"{'=' * 80}\n")

    return summary

# ====================================================================================================
# NUCLIDE GROUP PLOTTING
# ====================================================================================================

def _plot_nuclide_group(x_data, x_label, nuclide_data, nuclide_list, title, output_dir, filename_base, is_wedge=False, show_titles=True):
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
    ylabel = "Number of Atoms"
    ax.set_ylabel(ylabel, fontsize=12)
    if show_titles:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, ncol=min(4, len(available)))
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    save_path = os.path.join(output_dir, f"{filename_base}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

# ====================================================================================================
# STANDALONE ENTRY POINT
# ====================================================================================================

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