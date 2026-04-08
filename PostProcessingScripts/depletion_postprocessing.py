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
    # As a module (PNG default, 300 dpi):
    from depletion_postprocessing import run_depletion_postprocessing
    run_depletion_postprocessing(run_dir, params)
    run_depletion_postprocessing(run_dir, params, pdf_output=True)  # vector PDF

    # Standalone:
    python depletion_postprocessing.py <run_directory>          # PNG (300 dpi)
    python depletion_postprocessing.py <run_directory> --pdf    # vector PDF
"""

import os
import sys
import csv
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import openmc
import openmc.deplete


# Default plot groups — used only if not specified in params

DEFAULT_PLOT_GROUPS = {
    "Fissile Actinides": ["U235",
                          "Pu239", "Pu241"],
    "Fertile Actinides": ["U238", "U234", "U236",
                          "Pu238", "Pu240", "Pu242"],
    "Minor Actinides":   ["Np237", "Np239",
                          "Am241", "Am243",
                          "Cm242", "Cm244"],
    "FP Poisons":        ["Xe131", "Xe135",
                          "I135",
                          "Pm147", "Pm149",
                          "Sm149", "Sm151", "Sm152"],
    "Other FPs":         ["Kr83",
                          "Sr90",
                          "Mo95",
                          "Tc99",
                          "Rh103",
                          "Cs133", "Cs137",
                          "Nd143", "Nd145",
                          "Eu153"],
    "Boron Poisons":     ["B10"]
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
# HEATING-TALLY BURNUP IN MWd/MtU (spatial burnup — primary method)
# ====================================================================================================

def extract_min_max_burnup_MWdMtU(run_dir, fuel_mat_ids_2d, fuel_mat_volumes,
                                   time_days, thermal_power_MW,
                                   total_HM_mass_kg, fuel_volume_full_core_cm3,
                                   symmetry_factor=1):
    """
    Compute per-zone min/max/avg burnup in MWd/MtU from zone_heating_per_step.json.

    Physics
    -------
    For each recorded depletion step s:
        BU_z[s] = P_total_MW * (H_z[s] / H_total[s]) * dt[s] * 1000 / M_HM_z_full_kg

    Cumulative per-zone burnup at each timestep t is the running sum of
    BU_z over all recorded steps up to and including t.

    For DepletionStudy (single-entry file), the fractions from the final
    statepoint are applied uniformly to all timesteps.

    Parameters
    ----------
    run_dir                   : str — directory containing zone_heating_per_step.json
    fuel_mat_ids_2d           : list[list[int]] — [ring][axial_band] material IDs
    fuel_mat_volumes          : dict str(mat_id) -> float (cm3, simulated geometry)
    time_days                 : np.ndarray  shape (n_steps,)
    thermal_power_MW          : float — full-core thermal power (MW)
    total_HM_mass_kg          : float — full-core HM mass (kg)
    fuel_volume_full_core_cm3 : float — full-core fuel volume (cm3)
    symmetry_factor           : int (6 for 1/6 wedge, 1 for full core)

    Returns
    -------
    dict or None:
        'zone_burnup_fraction'  : (n_rings, n_bands, n_steps) array — zone/avg ratio
        'zone_labels'           : list of "ring{r}_band{b}" strings
        'peak_burnup_MWd_MtU'  : (n_steps,) array
        'peak_zone'             : (n_steps,) int array
        'min_burnup_MWd_MtU'   : (n_steps,) array
        'min_zone'              : (n_steps,) int array
        'avg_burnup_MWd_MtU'   : (n_steps,) array
        'method'                : "heating-local tally"
    None if zone_heating_per_step.json is absent or empty.
    """
    heating_file = os.path.join(run_dir, "zone_heating_per_step.json")
    if not os.path.exists(heating_file):
        return None

    with open(heating_file) as f:
        step_records = json.load(f)

    if not step_records:
        return None

    n_rings  = len(fuel_mat_ids_2d)
    n_bands  = len(fuel_mat_ids_2d[0]) if n_rings > 0 else 0
    n_steps  = len(time_days)
    n_zones  = n_rings * n_bands

    # Build per-zone HM mass (full core, kg)
    fuel_volume_simulated = fuel_volume_full_core_cm3 / symmetry_factor
    zone_HM_mass = np.zeros(n_zones)
    for ring_idx, row in enumerate(fuel_mat_ids_2d):
        for bax_idx, mat_id in enumerate(row):
            flat_idx = ring_idx * n_bands + bax_idx
            V_sim = fuel_mat_volumes.get(str(mat_id), 0.0)
            if fuel_volume_simulated > 0:
                zone_HM_mass[flat_idx] = total_HM_mass_kg * V_sim / fuel_volume_simulated
            else:
                zone_HM_mass[flat_idx] = 0.0

    # Cumulative per-zone burnup per timestep — (n_zones, n_steps)
    zone_bu = np.zeros((n_zones, n_steps))

    # Map each recorded step to a time-index and accumulate
    # Strategy:
    #   Single-entry file  → apply to all steps uniformly
    #   Multi-entry file   → match each record's step_end_days to the closest time_days entry
    single_entry = (len(step_records) == 1)

    def _step_contribution(record):
        """Return per-zone burnup increment (MWd/MtU) for one recorded step."""
        H_total = record.get("H_total", 0.0)
        if H_total <= 0:
            return np.zeros(n_zones)
        dt      = record["dt_days"]
        H_zones = record.get("H_zones", {})
        contrib = np.zeros(n_zones)
        for ring_idx, row in enumerate(fuel_mat_ids_2d):
            for bax_idx, _ in enumerate(row):
                flat_idx = ring_idx * n_bands + bax_idx
                H_z = H_zones.get(f"{ring_idx}_{bax_idx}", 0.0)
                M_HM = zone_HM_mass[flat_idx]
                if M_HM > 0 and H_total > 0:
                    contrib[flat_idx] = thermal_power_MW * (H_z / H_total) * dt * 1000.0 / M_HM
        return contrib

    if single_entry:
        # Apply final-statepoint fractions uniformly: BU_z[t] = contrib_rate * t_cumulative
        record     = step_records[0]
        H_total    = record.get("H_total", 0.0)
        H_zones    = record.get("H_zones", {})
        for step_t in range(n_steps):
            dt_t = time_days[step_t] - (time_days[step_t - 1] if step_t > 0 else 0.0)
            for ring_idx, row in enumerate(fuel_mat_ids_2d):
                for bax_idx, _ in enumerate(row):
                    flat_idx = ring_idx * n_bands + bax_idx
                    H_z = H_zones.get(f"{ring_idx}_{bax_idx}", 0.0)
                    M_HM = zone_HM_mass[flat_idx]
                    if M_HM > 0 and H_total > 0:
                        incr = thermal_power_MW * (H_z / H_total) * dt_t * 1000.0 / M_HM
                        zone_bu[flat_idx, step_t:] += incr
    else:
        # Multi-step: match each record's step_end_days to the nearest time_days bin
        for record in step_records:
            contrib = _step_contribution(record)
            step_end = record["step_end_days"]
            # Find the first time index where time_days >= step_end
            t_idx = np.searchsorted(time_days, step_end, side='left')
            t_idx = min(t_idx, n_steps - 1)
            # Accumulate into this and all later timesteps
            zone_bu[:, t_idx:] += contrib[:, np.newaxis]

    # Core-average burnup at each step from zone contributions (weighted by zone HM mass)
    total_HM_zone_sum = zone_HM_mass.sum()
    if total_HM_zone_sum > 0:
        avg_bu = (zone_bu * zone_HM_mass[:, np.newaxis]).sum(axis=0) / total_HM_zone_sum
    else:
        avg_bu = np.zeros(n_steps)

    zone_ok = zone_HM_mass > 0
    zone_bu_masked = zone_bu.copy()
    zone_bu_masked[~zone_ok, :] = np.nan

    peak_burnup    = np.nanmax(zone_bu_masked, axis=0)
    min_burnup     = np.nanmin(zone_bu_masked, axis=0)
    peak_zone_flat = np.nanargmax(zone_bu_masked, axis=0)
    min_zone_flat  = np.nanargmin(zone_bu_masked, axis=0)

    zone_labels = [f"ring{r}_band{b}"
                   for r in range(n_rings)
                   for b in range(n_bands)]

    print(f"  Peak burnup at final step: "
          f"{peak_burnup[-1]:.0f} MWd/MtU  "
          f"(avg = {avg_bu[-1]:.0f}  "
          f"peaking = {peak_burnup[-1]/max(avg_bu[-1], 1):.2f}×  "
          f"zone = {zone_labels[peak_zone_flat[-1]]})")
    print(f"  Min  burnup at final step: "
          f"{min_burnup[-1]:.0f} MWd/MtU  "
          f"zone = {zone_labels[min_zone_flat[-1]]}")

    # Reshape zone_bu to (n_rings, n_bands, n_steps) for zone_burnup_fraction output
    zone_bu_3d = zone_bu.reshape(n_rings, n_bands, n_steps)
    with np.errstate(invalid='ignore'):
        avg_safe     = np.where(avg_bu > 0, avg_bu, np.nan)
        zone_bu_frac = zone_bu_3d / avg_safe[np.newaxis, np.newaxis, :]
    zone_bu_frac = np.nan_to_num(zone_bu_frac, nan=0.0)

    return {
        "zone_burnup_fraction": zone_bu_frac,
        "zone_labels":          zone_labels,
        "peak_burnup_MWd_MtU": peak_burnup,
        "peak_zone":            peak_zone_flat,
        "min_burnup_MWd_MtU":  min_burnup,
        "min_zone":             min_zone_flat,
        "avg_burnup_MWd_MtU":  avg_bu,
        "method":               "heating-local tally",
    }


# ====================================================================================================
# BURNUP IN %FIMA (spatial burnup — actinide inventory method)
# ====================================================================================================

# Comprehensive list of actinide nuclides to look for in depletion results.
# Any nuclide absent from the chain is skipped silently.
_ACTINIDE_NUCLIDES = [
    # Thorium (Z=90)
    "Th227", "Th228", "Th229", "Th230", "Th231", "Th232", "Th233", "Th234",
    # Protactinium (Z=91)
    "Pa231", "Pa232", "Pa233",
    # Uranium (Z=92)
    "U232", "U233", "U234", "U235", "U236", "U237", "U238", "U239", "U240",
    # Neptunium (Z=93)
    "Np235", "Np236", "Np237", "Np238", "Np239",
    # Plutonium (Z=94)
    "Pu236", "Pu237", "Pu238", "Pu239", "Pu240", "Pu241", "Pu242", "Pu243", "Pu244",
    # Americium (Z=95)
    "Am241", "Am242", "Am242_m1", "Am243", "Am244", "Am244_m1",
    # Curium (Z=96)
    "Cm241", "Cm242", "Cm243", "Cm244", "Cm245", "Cm246", "Cm247", "Cm248",
    # Berkelium (Z=97)
    "Bk249", "Bk250",
    # Californium (Z=98)
    "Cf249", "Cf250", "Cf251", "Cf252",
]


def extract_min_max_burnup_FIMA(results, fuel_mat_ids_2d,
                                 time_days, burnup_MWd_per_MtU,
                                 symmetry_factor=1):
    """
    Compute per-zone min/max/avg burnup in %FIMA from the actual actinide inventory.

    Physics
    -------
    Every fission event removes one HM atom from the fuel (fission products are
    not actinides).  Neutron capture + transmutation conserves HM atom count
    (U238 → Pu239 etc. all stay in the HM pool).  Therefore:

        %FIMA_z[t] = (N_HM_z[0] - N_HM_z[t]) / N_HM_z[0]  × 100%

    where N_HM_z[t] = sum of all tracked actinide atoms in zone z at time t.
    Isotopes are pulled from the depletion results for each specific burnup-zone
    material, so min and max are per-zone quantities.

    Core-average %FIMA is computed as an atom-weighted mean over all zones:

        avg_%FIMA[t] = (ΣN_HM_z[0] - ΣN_HM_z[t]) / ΣN_HM_z[0]  × 100%

    Parameters
    ----------
    results             : openmc.deplete.Results
    fuel_mat_ids_2d     : list[list[int]] — [ring][axial_band] material IDs
    time_days           : np.ndarray  shape (n_steps,)
    burnup_MWd_per_MtU  : np.ndarray or None — core-average MWd/MtU for x-axis reference
    symmetry_factor     : int (unused in ratio calc; kept for API consistency)

    Returns
    -------
    dict or None:
        'zone_burnup_fraction'  : (n_rings, n_bands, n_steps) — zone %FIMA / avg %FIMA
        'zone_labels'           : list of "ring{r}_band{b}" strings
        'peak_burnup_pct_FIMA'  : (n_steps,) array
        'peak_zone'             : (n_steps,) int array
        'min_burnup_pct_FIMA'   : (n_steps,) array
        'min_zone'              : (n_steps,) int array
        'avg_burnup_pct_FIMA'   : (n_steps,) array
        'avg_burnup_MWd_MtU'    : (n_steps,) array or None — reference for x-axis
        'tracked_actinides'     : list of str — nuclides found in at least one zone
        'method'                : "actinide inventory"
    None if no actinide data is found for any zone.
    """

    if fuel_mat_ids_2d is None:
        return None

    n_rings = len(fuel_mat_ids_2d)
    n_bands = len(fuel_mat_ids_2d[0])
    n_steps = len(time_days)

    # zone_HM[r, b, t] = total actinide atom count in zone (r, b) at step t
    zone_HM  = np.zeros((n_rings, n_bands, n_steps))
    zone_ok  = np.zeros((n_rings, n_bands), dtype=bool)
    found_nuclides = set()

    for ring_idx, row in enumerate(fuel_mat_ids_2d):
        for bax_idx, mat_id in enumerate(row):
            mat_str   = str(mat_id)
            zone_sum  = np.zeros(n_steps)
            any_found = False
            for nuc in _ACTINIDE_NUCLIDES:
                try:
                    _, atoms = results.get_atoms(mat_str, nuc)
                    atoms = np.array(atoms, dtype=float)
                    if len(atoms) >= n_steps:
                        zone_sum += atoms[:n_steps]
                    else:
                        zone_sum[:len(atoms)] += atoms
                    any_found = True
                    found_nuclides.add(nuc)
                except Exception:
                    pass
            if any_found and zone_sum[0] > 0:
                zone_HM[ring_idx, bax_idx, :] = zone_sum
                zone_ok[ring_idx, bax_idx]    = True

    n_good = int(np.sum(zone_ok))
    if n_good == 0:
        print("  WARNING: No actinide data found for any zone — cannot compute %FIMA")
        return None

    print(f"  %FIMA burnup: actinide inventory from {n_good}/{n_rings * n_bands} zones")
    print(f"  Tracked actinides found: {len(found_nuclides)}  "
          f"({', '.join(sorted(found_nuclides)[:8])}"
          f"{'...' if len(found_nuclides) > 8 else ''})")

    # --- Per-zone %FIMA = (N_HM_z[0] - N_HM_z[t]) / N_HM_z[0] * 100 ---
    N_HM_z0      = zone_HM[:, :, 0:1]                        # (n_rings, n_bands, 1)
    N_HM_z0_safe = np.where(N_HM_z0 > 0, N_HM_z0, np.nan)

    with np.errstate(invalid='ignore'):
        zone_FIMA_3d = (N_HM_z0_safe - zone_HM) / N_HM_z0_safe * 100.0
    zone_FIMA_3d = np.nan_to_num(zone_FIMA_3d, nan=0.0)

    # --- Core-average %FIMA (atom-weighted across all zones in simulated geometry) ---
    N_HM_core_0 = zone_HM[:, :, 0][zone_ok].sum()
    if N_HM_core_0 <= 0:
        print("  WARNING: zero initial HM atoms in core — cannot compute avg %FIMA")
        return None
    N_HM_core_t = zone_HM[zone_ok, :].sum(axis=0)            # (n_steps,)
    avg_FIMA    = (N_HM_core_0 - N_HM_core_t) / N_HM_core_0 * 100.0

    # --- Peak / min over zones ---
    zone_FIMA_flat = zone_FIMA_3d.reshape(n_rings * n_bands, n_steps)
    only_ok        = zone_ok.flatten()
    zone_FIMA_flat[~only_ok, :] = np.nan

    peak_FIMA      = np.nanmax(zone_FIMA_flat, axis=0)
    min_FIMA       = np.nanmin(zone_FIMA_flat, axis=0)
    peak_zone_flat = np.nanargmax(zone_FIMA_flat, axis=0)
    min_zone_flat  = np.nanargmin(zone_FIMA_flat, axis=0)

    zone_labels = [f"ring{r}_band{b}"
                   for r in range(n_rings)
                   for b in range(n_bands)]

    print(f"  Peak %FIMA at final step: "
          f"{peak_FIMA[-1]:.3f}%  "
          f"(avg = {avg_FIMA[-1]:.3f}%  "
          f"peaking = {peak_FIMA[-1]/max(avg_FIMA[-1], 1e-9):.2f}×  "
          f"zone = {zone_labels[peak_zone_flat[-1]]})")
    print(f"  Min  %FIMA at final step: "
          f"{min_FIMA[-1]:.3f}%  "
          f"zone = {zone_labels[min_zone_flat[-1]]}")

    # zone_burnup_fraction: zone %FIMA relative to core-average (for peaking factor)
    avg_FIMA_safe = np.where(avg_FIMA > 0, avg_FIMA, np.nan)
    with np.errstate(invalid='ignore'):
        zone_burnup_frac = zone_FIMA_3d / avg_FIMA_safe[np.newaxis, np.newaxis, :]
    zone_burnup_frac = np.nan_to_num(zone_burnup_frac, nan=0.0)

    return {
        "zone_burnup_fraction":  zone_burnup_frac,
        "zone_labels":           zone_labels,
        "peak_burnup_pct_FIMA":  peak_FIMA,
        "peak_zone":             peak_zone_flat,
        "min_burnup_pct_FIMA":   min_FIMA,
        "min_zone":              min_zone_flat,
        "avg_burnup_pct_FIMA":   avg_FIMA,
        "avg_burnup_MWd_MtU":    burnup_MWd_per_MtU,
        "tracked_actinides":     sorted(found_nuclides),
        "method":                "actinide inventory",
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

        print(f"  [{label}] Nuclide inventory saved → {os.path.basename(out_path)}  "
              f"({n_steps} steps × {len(nuclides)} nuclides)")
        written.append(out_path)

    return written

# ====================================================================================================
# CONVERSION RATIO CALCULATION
# ====================================================================================================

def calculate_conversion_ratio(fuel_data, time_days):
    """
    Calculate the conversion ratio (CR) at each depletion timestep.

    Method:
      fiss_per_day = (U235[0] - U235[1]) / dt[0]   (constant rate, calibrated at step 0
                                                      where Pu239 ≈ 0 so all fissile = U-235)
      Per step i → i+1:
        dt[i]               = time_days[i+1] - time_days[i]
        fissile_burned[i]   = fiss_per_day * dt[i]
        delta_U235[i]       = U235[i] - U235[i+1]
        Pu239_burned[i]     = fissile_burned[i] - delta_U235[i]
        delta_Pu239[i]      = Pu239[i+1] - Pu239[i]
        Pu239_generated[i]  = delta_Pu239[i] + Pu239_burned[i]
        CR[i]               = Pu239_generated[i] / fissile_burned[i]

    CR < 1 for a converter, CR = 1 at break-even, CR > 1 only for a breeder.

    Parameters
    ----------
    fuel_data : dict {nuclide: np.ndarray}
        Must contain 'U235' and 'Pu239'.
    time_days : array-like
        Time in days at each depletion step (length = n_steps).

    Returns
    -------
    dict with keys: 'CR', 'U235_burned', 'Pu239_burned', 'Pu239_generated',
                    'total_fissile_burned', 'dt_days'
    None if required nuclides are absent or data is too short.
    """
    if "U235" not in fuel_data or "Pu239" not in fuel_data:
        return None

    u235  = np.array(fuel_data["U235"],  dtype=float)
    pu239 = np.array(fuel_data["Pu239"], dtype=float)
    t     = np.array(time_days,          dtype=float)
    n     = min(len(u235), len(pu239), len(t))

    if n < 2:
        return None

    u235  = u235[:n]
    pu239 = pu239[:n]
    t     = t[:n]

    dt = t[1:] - t[:-1]           # step lengths in days

    # Constant fissile burn rate calibrated at step 0 (Pu239 ≈ 0 at BOL)
    delta_u235_step0 = float(u235[0] - u235[1])
    if delta_u235_step0 <= 0 or dt[0] <= 0:
        return None
    fiss_per_day = delta_u235_step0 / dt[0]

    # Per-step quantities
    fissile_burned  = fiss_per_day * dt              # varies with dt
    delta_u235      = u235[:-1] - u235[1:]           # U235 consumed this step
    pu239_burned    = fissile_burned - delta_u235    # remainder from Pu239
    delta_pu239     = pu239[1:] - pu239[:-1]         # net Pu239 change
    pu239_generated = delta_pu239 + pu239_burned     # gross Pu239 production

    cr = np.where(fissile_burned > 0,
                  pu239_generated / fissile_burned,
                  np.nan)

    return {
        "CR":                   cr,
        "U235_burned":          delta_u235,
        "Pu239_burned":         pu239_burned,
        "Pu239_generated":      pu239_generated,
        "total_fissile_burned": fissile_burned,
        "dt_days":              dt,
    }


# ====================================================================================================
# GAMMA SOURCES EOL CSV EXPORT
# ====================================================================================================

def export_gamma_sources_csv(results, gamma_sources, last_operational_idx,
                              symmetry_factor, output_dir):
    """
    Export EOL atom inventory for gamma-source isotopes summed across ALL materials.

    Searches every material ID present in the depletion results so isotopes
    confined to graphite, fuel, cladding, coolant, etc. are all captured.

    Parameters
    ----------
    results : openmc.deplete.Results
    gamma_sources : list of str
        Nuclide names from params["gamma_sources"].
    last_operational_idx : int
        Index of the last operational timestep.
    symmetry_factor : int
        Full-core multiplier (6 for 1/6 wedge, 1 for full core).
    output_dir : str
        Directory to write gamma_sources_EOL.csv into.

    Returns
    -------
    dict : {nuclide: float} — atoms at EOL for each found isotope.
    """
    print("\n  Exporting gamma sources EOL inventory...")

    try:
        all_mat_ids = list(results[0].index_mat.keys())
    except Exception as e:
        print(f"  WARNING: Could not retrieve material IDs from results: {e}")
        return {}

    found    = {}   # nuclide -> total atoms at last_operational_idx
    missing  = []

    # # >>>>>>>>>> HACK: extract at timestep 15 instead of last_operational_idx <<<<<<<<<<
    # _HACK_TIMESTEP = 15
    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    for nuc in gamma_sources:
        total = 0.0
        found_in_any = False
        for mid in all_mat_ids:
            try:
                _t, atoms = results.get_atoms(str(mid), nuc)
                atoms = np.array(atoms, dtype=float)
                if last_operational_idx < len(atoms) and atoms[last_operational_idx] > 0:  # HACK: comment out
                    total += atoms[last_operational_idx]                                    # HACK: comment out
                    found_in_any = True
                # if _HACK_TIMESTEP < len(atoms) and atoms[_HACK_TIMESTEP] > 0:               # HACK
                #     total += atoms[_HACK_TIMESTEP]                                           # HACK
                    # found_in_any = True
            except Exception:
                continue
        if found_in_any:
            found[nuc] = total * symmetry_factor
        else:
            missing.append(nuc)

    if missing:
        print(f"  WARNING: The following gamma_sources were not found in any material "
              f"and will be omitted from the CSV:\n    {missing}")

    # Write CSV — one row per isotope
    csv_path = os.path.join(output_dir, "gamma_sources_EOL.csv")  # HACK: still writes to same file
    with open(csv_path, "w") as f:
        f.write("isotope,atoms_at_EOL\n")
        for nuc in gamma_sources:
            if nuc in found:
                f.write(f"{nuc},{found[nuc]:.6e}\n")

    print(f"  Gamma sources EOL inventory saved → {os.path.basename(csv_path)}  "
          f"({len(found)}/{len(gamma_sources)} isotopes found, "
          f"step index {last_operational_idx})")
    return found


# ====================================================================================================
# CS DEPLETION STUDY HELPER FUNCTIONS
# ====================================================================================================

def _get_aro_keff_from_cs_csv(csv_path):
    """
    Extract the all-rods-out (ARO) k-effective from a critical_search*.csv file.

    The ARO state is the row where both bank_1 and bank_2 equal 0.0 (fully withdrawn).
    If no exact ARO row is found, returns the row with minimum total bank insertion.

    Parameters
    ----------
    csv_path : str
        Full path to the critical_search*.csv file.

    Returns
    -------
    (keff, keff_std) : tuple of float, or (None, None) on failure.
    """
    if not csv_path or not os.path.exists(csv_path):
        return None, None
    try:
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "keff":     float(row["keff"]),
                        "keff_std": float(row["keff_std"]),
                        "bank_1":   float(row["bank_1"]),
                        "bank_2":   float(row["bank_2"]),
                    })
                except (ValueError, KeyError):
                    continue
        if not rows:
            return None, None
        # Prefer the explicit ARO row (both banks at minimum insertion = 0.0)
        aro = [r for r in rows if r["bank_1"] == 0.0 and r["bank_2"] == 0.0]
        if aro:
            return aro[0]["keff"], aro[0]["keff_std"]
        # Fallback: row with lowest total insertion
        rows.sort(key=lambda r: r["bank_1"] + r["bank_2"])
        return rows[0]["keff"], rows[0]["keff_std"]
    except Exception:
        return None, None


def _get_last_th_coupler_files(step_dir):
    """
    Find the last-attempt, last-iteration th_coupler temperatures and heating CSVs
    in a CS depletion step directory.

    Naming convention:
      Attempt 1 : th_coupler_th_iter_NN_*.csv           (no attempt suffix)
      Attempt N : th_coupler_attemptN_th_iter_NN_*.csv

    Parameters
    ----------
    step_dir : str
        Full path to a cs_th_step*_t*d directory.

    Returns
    -------
    (temp_csv, heat_csv) : tuple of str or (None, None) if not found.
    """
    # Gather all temperature files, grouped by attempt number
    all_temp = glob.glob(os.path.join(step_dir, "th_coupler*_th_iter_*_temperatures.csv"))
    if not all_temp:
        return None, None

    def _attempt_iter(path):
        """Return (attempt_number, iter_number) for sorting."""
        fname = os.path.basename(path)
        # attempt number
        if "attempt" in fname:
            try:
                astr = fname.split("attempt")[1].split("_")[0]
                attempt = int(astr)
            except (IndexError, ValueError):
                attempt = 1
        else:
            attempt = 1
        # iter number
        try:
            istr = fname.split("th_iter_")[1].split("_")[0]
            it = int(istr)
        except (IndexError, ValueError):
            it = 0
        return attempt, it

    all_temp.sort(key=_attempt_iter)
    best_temp = all_temp[-1]  # highest (attempt, iter)

    # Corresponding heating file
    best_heat = best_temp.replace("_temperatures.csv", "_heating.csv")
    if not os.path.exists(best_heat):
        best_heat = None

    return best_temp, best_heat


# ====================================================================================================
# PERFORM DEPLETION ANALYSIS PLOTTING AND SAVE RESULTS
# ====================================================================================================

def run_depletion_postprocessing(run_dir, params, pdf_output=False):
    """
    Run full depletion post-processing.

    Parameters
    ----------
    run_dir : str
        Directory containing depletion_results.h5.
    params : dict
        Simulation parameters (merged with run_params.json).
    pdf_output : bool, optional
        If True, save figures as PDF (vector, no compression). Default False (PNG at 300 dpi).

    Returns
    -------
    dict : Summary results.
    """

    fig_fmt = "pdf" if pdf_output else "png"
    fig_dpi = None if pdf_output else 300

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

    # Operational flag — source depends on study mode:
    #   CSDepletionStudy : read from critical_search_depletion_log.json.
    #                      EOS keff (stored in depletion_results.h5) is always < 1 by
    #                      design — it uses the BOS rod position after the fuel has
    #                      burned for a full step.  BOS keff (≈ 1.0 while critical) is
    #                      the meaningful reactivity metric and lives in the log.
    #   DepletionStudy   : keff >= 1.0 from depletion results (ARO approximation).
    cs_log = None   # preserved for CSV section below
    if params.get("study_execution_mode") == "CSDepletionStudy":
        cs_log_path = os.path.join(run_dir, "critical_search_depletion_log.json")
        if os.path.exists(cs_log_path):
            with open(cs_log_path) as _f:
                cs_log = json.load(_f)
            # Log has one entry per step (step=1..N).
            # time_days index 0 = initial state (always operational).
            # time_days index i = end of step i → operational flag from log entry step=i.
            cs_operational = np.ones(len(keff_mean), dtype=int)
            for entry in cs_log:
                idx = entry["step"]   # 1-based step index == time_days index for that EOS
                if idx < len(cs_operational):
                    cs_operational[idx] = entry["operational"]
            operational = cs_operational
            print(f"   [Operational] Loaded from critical_search_depletion_log.json "
                  f"({int(operational.sum())} / {len(operational)} steps operational)")
        else:
            print("   [Operational] WARNING: CSDepletionStudy but no "
                  "critical_search_depletion_log.json found — falling back to keff >= 1.0")
            operational = (keff_mean >= 1.0).astype(int)
    else:
        operational = (keff_mean >= 1.0).astype(int)

    op_indices = np.where(operational == 1)[0]
    last_operational_idx = int(op_indices[-1]) if len(op_indices) > 0 else 0

    # BOS arrays (CSDepletionStudy only) — built once, reused in plots and CSV.
    # log entry step=i was run at step_start = time_days[i-1]; last time point has no BOS.
    bos_keff = bos_keff_std = bos_bank1 = bos_bank2 = None
    if cs_log:
        _n = len(time_days)
        bos_keff     = np.full(_n, np.nan)
        bos_keff_std = np.full(_n, np.nan)
        bos_bank1    = np.full(_n, np.nan)
        bos_bank2    = np.full(_n, np.nan)
        for _e in cs_log:
            _idx = _e["step"] - 1   # step=i BOS → time_days[i-1]
            if 0 <= _idx < _n:
                bos_keff[_idx]     = _e["critical_keff"]
                bos_keff_std[_idx] = _e["critical_keff_std"]
                bos_bank1[_idx]    = _e["bank_1_insertion"]
                bos_bank2[_idx]    = _e["bank_2_insertion"]

    # CSDepletionStudy: redefine operational using BOS keff >= 1.0 - k_tol.
    # The log's own operational flag may differ; the BOS keff criterion is authoritative.
    cs_bos_search_csv = {}   # bos time index → path of last critical_search*.csv
    if cs_log and bos_keff is not None:
        _k_tol = params.get("critical_search_k_tol", 0.0064)
        _k_threshold = 1.0 - _k_tol
        for i in range(len(operational)):
            if not np.isnan(bos_keff[i]):
                operational[i] = 1 if bos_keff[i] >= _k_threshold else 0
        # Build BOS index → search_csv mapping for ARO keff extraction
        for _e in cs_log:
            _bos_idx = _e["step"] - 1
            if 0 <= _bos_idx < len(time_days):
                cs_bos_search_csv[_bos_idx] = _e.get("search_csv", "")
        # Recompute derived indices after override
        op_indices = np.where(operational == 1)[0]
        last_operational_idx = int(op_indices[-1]) if len(op_indices) > 0 else 0
        print(f"   [Operational] Redefined by BOS keff >= {_k_threshold:.4f}: "
              f"{int(operational.sum())} / {len(operational)} steps operational "
              f"(last op idx={last_operational_idx}, "
              f"t={time_days[last_operational_idx]:.0f} d)")

    thermal_power_MW  = params.get("thermal_power_MW", 10.0)
    total_HM_mass_kg  = params.get("total_HM_mass_kg", None)
    total_B10_mass_kg = params.get("total_B10_mass_kg", None)

    if total_HM_mass_kg and total_HM_mass_kg > 0:
        cumulative_energy_MWd = thermal_power_MW * time_days
        burnup_MWd_per_MtU    = cumulative_energy_MWd / (total_HM_mass_kg / 1000.0)
    else:
        burnup_MWd_per_MtU = None

    x_data        = burnup_MWd_per_MtU if burnup_MWd_per_MtU is not None else time_days
    x_label       = "Core-Average Burnup (MWd/MtU)"  if burnup_MWd_per_MtU is not None else "Time (days)"
    x_label_short = "burnup"            if burnup_MWd_per_MtU is not None else "time"

    # Number of time steps to include in plots.
    # Include the last operational step AND the first non-operational step (as a boundary
    # context / to show the discharge crossing), then omit everything further.
    _first_nonop = None
    for _i in range(len(operational)):
        if operational[_i] == 0:
            _first_nonop = _i
            break
    if _first_nonop is not None:
        n_plot = _first_nonop + 1          # include first non-op as boundary
    else:
        n_plot = len(time_days)            # all steps operational

    # ================================================================================
    # 2. DISCHARGE BURNUP
    # ================================================================================
    # CSDepletionStudy : linearly interpolate between the ARO keff of the last
    #                    operational BOS and the first non-operational BOS (from each
    #                    step's critical_search*.csv) to find when ARO keff = 1.0.
    # DepletionStudy   : discharge = interpolated time where EOS keff crosses 1.0.

    discharge_burnup     = None
    discharge_time_days  = None
    discharge_time_years = None
    _cs_aro_keff_pts     = None   # ARO keff points for discharge interpolation plot

    if cs_log:
        # Find the 1→0 transition in the operational array
        _last_op_idx    = None
        _first_nonop_idx = None
        for i in range(len(operational) - 1):
            if operational[i] == 1 and operational[i + 1] == 0:
                _last_op_idx    = i
                _first_nonop_idx = i + 1
                break

        if _last_op_idx is not None and _first_nonop_idx is not None:
            # Retrieve ARO keff from the last critical_search*.csv for each boundary step.
            # bos_keff index alignment: bos_keff[i] = BOS keff at time_days[i], which
            # corresponds to cs_bos_search_csv[i].
            _aro_k_last, _aro_k_last_std = _get_aro_keff_from_cs_csv(
                cs_bos_search_csv.get(_last_op_idx, ""))
            _aro_k_nonop, _aro_k_nonop_std = _get_aro_keff_from_cs_csv(
                cs_bos_search_csv.get(_first_nonop_idx, ""))

            if (_aro_k_last is not None and _aro_k_nonop is not None
                    and _aro_k_last > _aro_k_nonop):
                # Linear interpolation: ARO keff = 1.0 between the two time points
                _frac = (_aro_k_last - 1.0) / (_aro_k_last - _aro_k_nonop)
                _frac = float(np.clip(_frac, 0.0, 1.0))
                discharge_time_days = (time_days[_last_op_idx]
                                       + _frac * (time_days[_first_nonop_idx]
                                                  - time_days[_last_op_idx]))
                discharge_time_years = discharge_time_days / 365.25
                if burnup_MWd_per_MtU is not None:
                    discharge_burnup = (burnup_MWd_per_MtU[_last_op_idx]
                                        + _frac * (burnup_MWd_per_MtU[_first_nonop_idx]
                                                   - burnup_MWd_per_MtU[_last_op_idx]))
                print(f"   [Discharge] ARO keff interpolation: "
                      f"k({time_days[_last_op_idx]:.0f} d)={_aro_k_last:.5f}, "
                      f"k({time_days[_first_nonop_idx]:.0f} d)={_aro_k_nonop:.5f}, "
                      f"frac={_frac:.4f} → {discharge_time_years:.3f} yr")
                # Store ARO keff points for plotting (x in both burnup/time and time-years)
                _cs_aro_keff_pts = {
                    'last_op_x':     x_data[_last_op_idx],
                    'nonop_x':       x_data[_first_nonop_idx],
                    'last_op_ty':    time_days[_last_op_idx] / 365.25,
                    'nonop_ty':      time_days[_first_nonop_idx] / 365.25,
                    'last_op_k':     _aro_k_last,
                    'last_op_k_std': _aro_k_last_std if _aro_k_last_std is not None else 0.0,
                    'nonop_k':       _aro_k_nonop,
                    'nonop_k_std':   _aro_k_nonop_std if _aro_k_nonop_std is not None else 0.0,
                }
            else:
                # Fallback if ARO keff not extractable
                discharge_time_days  = float(time_days[_first_nonop_idx])
                discharge_time_years = discharge_time_days / 365.25
                if burnup_MWd_per_MtU is not None:
                    discharge_burnup = float(burnup_MWd_per_MtU[_first_nonop_idx])
                print("   [Discharge] WARNING: ARO keff not found; using first non-op time.")
    else:
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

    tracked_nuclides        = params.get("tracked_nuclides", ["U235", "U238", "Pu238", "Pu239", "B10"])
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

    # ---- Poison: spatial burnup (multiple materials) vs. single material ----
    poison_mat_ids_2d = params.get("poison_mat_ids", None)

    if poison_mat_ids_2d:
        all_poison_ids = [str(mid)
                          for row in poison_mat_ids_2d
                          for mid in row]
        seen_p = set()
        unique_poison_ids = []
        for mid in all_poison_ids:
            if mid not in seen_p:
                seen_p.add(mid)
                unique_poison_ids.append(mid)
        n_prings = len(poison_mat_ids_2d)
        n_pbands = len(poison_mat_ids_2d[0]) if poison_mat_ids_2d else 0
        print(f"\n   Spatial burnup poison: summing {len(unique_poison_ids)} poison zones "
              f"({n_prings} rings × {n_pbands} axial bands)...")
        poison_data = _extract_nuclide_inventories_multi(
            results, unique_poison_ids, poison_tracked_nuclides, "Poison", symmetry_factor
        )
    else:
        poison_mat_id = _find_material_id(results, params,
                                          "poison_material_id", "B10", "Burnable poison")
        print(f"\n   Extracting burnable poison inventories ({len(poison_tracked_nuclides)} nuclides)...")
        poison_data = _extract_nuclide_inventories(results, poison_mat_id, poison_tracked_nuclides,
                                                   "Poison", symmetry_factor)

    # Graphite B10 — only extracted when deplete_graphite=True
    graphite_data = {}
    if params.get("deplete_graphite", False):
        graphite_mat_id = _find_material_id(results, params,
                                            "graphite_material_id", "B10", "Graphite")
        if graphite_mat_id:
            print(f"\n   Extracting graphite inventories (B10)...")
            graphite_data = _extract_nuclide_inventories(
                results, graphite_mat_id, ["B10"], "Graphite", symmetry_factor
            )

    # Merge for plotting — poison data keyed separately to avoid name collision
    # B10 from poison material is canonical; if also in fuel, prefer poison
    all_nuclide_data = {**fuel_data}
    for nuc, atoms in poison_data.items():
        all_nuclide_data[f"{nuc}_poison"] = atoms  # keep separate key
        all_nuclide_data[nuc] = atoms               # also overwrite top-level with poison value
    for nuc, atoms in graphite_data.items():
        all_nuclide_data[f"{nuc}_graphite"] = atoms  # keep separate key
        # If a poison source also exists, overwrite the top-level key with
        # the combined total so group plots show a single summed line.
        if f"{nuc}_poison" in all_nuclide_data:
            all_nuclide_data[nuc] = all_nuclide_data[f"{nuc}_poison"] + atoms
        else:
            all_nuclide_data[nuc] = atoms

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

    print(f"{'─' * 60}")

    # ================================================================================
    # 5. PLOTTING
    # ================================================================================

    print("\nGenerating depletion plots...")

    # Global style settings for all plots (sized for side-by-side on letter page)
    plt.rcParams.update({
        'font.size':        16,
        'axes.titlesize':   18,
        'axes.labelsize':   16,
        'xtick.labelsize':  16,
        'ytick.labelsize':  16,
        'legend.fontsize':  14,
        'lines.linewidth':  2.5,
        'lines.markersize': 6,
    })

    def _add_discharge_vline(ax, is_burnup):
        """Add a vertical discharge line if discharge data is available."""
        if is_burnup and discharge_burnup is not None:
            ax.axvline(discharge_burnup, color="green", linestyle=":", alpha=0.7,
                       label=f"Discharge: {discharge_burnup:.0f} MWd/MtU")
        elif not is_burnup and discharge_time_days is not None:
            ax.axvline(discharge_time_days, color="green", linestyle=":", alpha=0.7,
                       label=f"Discharge: {discharge_time_years:.2f} years")

    if cs_log:
        # CSDepletionStudy: plot BOS keff (≈ 1.0, meaningful) and EOS keff (always < 1)
        # as separate traces on the same axes. Truncate at last operational step.
        _bos_plot = bos_keff[:n_plot]
        valid_bos = ~np.isnan(_bos_plot)

        # vs. burnup / x_data
        fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
        ax.errorbar(x_data[:n_plot], keff_mean[:n_plot], yerr=keff_std[:n_plot],
                    fmt="o-", capsize=3, color="tab:blue", label="EOS k-effective")
        ax.errorbar(x_data[:n_plot][valid_bos], _bos_plot[valid_bos],
                    yerr=bos_keff_std[:n_plot][valid_bos], fmt="s--", capsize=3,
                    color="tab:orange", label="BOS k-effective (critical search)")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1, label="k = 1.0")
        _add_discharge_vline(ax, is_burnup=(burnup_MWd_per_MtU is not None))
        if _cs_aro_keff_pts is not None:
            ax.errorbar(
                [_cs_aro_keff_pts['last_op_x'], _cs_aro_keff_pts['nonop_x']],
                [_cs_aro_keff_pts['last_op_k'], _cs_aro_keff_pts['nonop_k']],
                yerr=[[_cs_aro_keff_pts['last_op_k_std'], _cs_aro_keff_pts['nonop_k_std']],
                      [_cs_aro_keff_pts['last_op_k_std'], _cs_aro_keff_pts['nonop_k_std']]],
                fmt='*--', capsize=3, color='purple', markersize=14,
                label='ARO k-eff (discharge interp.)', zorder=5)
        ax.set_xlabel(x_label)
        ax.set_ylabel("k-effective")
        if show_titles:
            ax.set_title("k-effective vs. Burnup (BOS & EOS)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_keff_vs_{x_label_short}.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()

        # vs. time (years primary, days secondary)
        _td_plot = time_days[:n_plot]
        _ty_plot = time_years[:n_plot]
        fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
        ax.errorbar(_ty_plot, keff_mean[:n_plot], yerr=keff_std[:n_plot],
                    fmt="o-", capsize=3, color="tab:blue", label="EOS k-effective")
        ax.errorbar(_ty_plot[valid_bos], _bos_plot[valid_bos],
                    yerr=bos_keff_std[:n_plot][valid_bos], fmt="s--", capsize=3,
                    color="tab:orange", label="BOS k-effective (critical search)")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1, label="k = 1.0")
        if discharge_time_years is not None:
            ax.axvline(discharge_time_years, color="green", linestyle=":", alpha=0.7,
                       label=f"Discharge: {discharge_time_years:.2f} yr")
        if _cs_aro_keff_pts is not None:
            ax.errorbar(
                [_cs_aro_keff_pts['last_op_ty'], _cs_aro_keff_pts['nonop_ty']],
                [_cs_aro_keff_pts['last_op_k'], _cs_aro_keff_pts['nonop_k']],
                yerr=[[_cs_aro_keff_pts['last_op_k_std'], _cs_aro_keff_pts['nonop_k_std']],
                      [_cs_aro_keff_pts['last_op_k_std'], _cs_aro_keff_pts['nonop_k_std']]],
                fmt='*--', capsize=3, color='purple', markersize=14,
                label='ARO k-eff (discharge interp.)', zorder=5)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("k-effective")
        if show_titles:
            ax.set_title("k-effective vs. Time (BOS & EOS)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if burnup_MWd_per_MtU is not None and total_HM_mass_kg:
            _bu_per_yr = thermal_power_MW * 365.25 / (total_HM_mass_kg / 1000.0)
            ax2 = ax.twiny()
            ax2.set_xlim(_ty_plot[0] * _bu_per_yr, _ty_plot[-1] * _bu_per_yr)
            ax2.set_xlabel("Core-Average Burnup (MWd/MtU)")
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_keff_vs_time.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()
        # No reactivity plot for CSDepletionStudy (EOS keff is always < 1 by design)

        # ================================================================================
        # 5d-ii. CS DEPLETION STUDY — CONTROL ROD BANK POSITION vs. TIME
        # ================================================================================
        # Only include operational steps; append a forced 0.0 point at discharge.
        _n_bank = (_last_op_idx + 1) if _last_op_idx is not None else n_plot
        _valid_bank = ~np.isnan(bos_bank1[:_n_bank])
        if np.any(_valid_bank):
            _ty_bank = time_years[:_n_bank][_valid_bank]
            _b1_vals = bos_bank1[:_n_bank][_valid_bank].copy()
            _b2_vals = bos_bank2[:_n_bank][_valid_bank].copy()
            # Append fully-withdrawn point at discharge (rods out at EOL)
            if discharge_time_years is not None:
                _ty_bank = np.append(_ty_bank, discharge_time_years)
                _b1_vals = np.append(_b1_vals, 0.0)
                _b2_vals = np.append(_b2_vals, 0.0)
            fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
            ax.plot(_ty_bank, _b1_vals, "o-", color="tab:blue", label="Bank 1 insertion")
            ax.plot(_ty_bank, _b2_vals, "s-", color="tab:red",  label="Bank 2 insertion")
            if discharge_time_years is not None:
                ax.axvline(discharge_time_years, color="green", linestyle=":", alpha=0.7,
                           label=f"Discharge: {discharge_time_years:.2f} yr")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Rod Insertion")
            if show_titles:
                ax.set_title("Control Rod Bank Insertion vs. Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            if burnup_MWd_per_MtU is not None and total_HM_mass_kg:
                _bu_per_yr = thermal_power_MW * 365.25 / (total_HM_mass_kg / 1000.0)
                ax2 = ax.twiny()
                ax2.set_xlim(_ty_bank[0] * _bu_per_yr, _ty_bank[-1] * _bu_per_yr)
                ax2.set_xlabel("Core-Average Burnup (MWd/MtU)")
            plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                      f"depletion_rod_position_vs_time.{fig_fmt}"),
                        bbox_inches="tight")
            plt.close()

        # ================================================================================
        # 5e. CS DEPLETION STUDY — THERMAL-HYDRAULIC PLOTS
        # ================================================================================
        # For each depletion step directory, read the last attempt's last iteration of
        # th_coupler temperatures and heating CSVs.  Step N (1-based) has its BOS at
        # time_days[N-1].  Collect max values and plot vs time (years) with burnup
        # (MWd/MtU) as the secondary x-axis.

        print("\n  Collecting TH data from CS depletion step directories...")

        # --- nc_htgr import for per-step MWe calculation ---
        _nc_htgr_avail = False
        _br = {}
        try:
            _nc_htgr_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "..", "HTGR-SCAPC"))
            if _nc_htgr_dir not in sys.path:
                sys.path.insert(0, _nc_htgr_dir)
            from nc_htgr import (compressor_T2 as _nc_comp_T2,
                                  turbine_T5    as _nc_turb_T5,
                                  helium_cp_gamma_R as _nc_he_cp)
            # Read Brayton parameters from nc_input.csv (same directory as nc_htgr.py)
            _nc_input_csv = os.path.join(_nc_htgr_dir, "nc_input.csv")
            _br_raw = {'P1_Pa': 2e6, 'T1_C': 30.0, 'pressure_ratio': 2.3,
                       'eta_c': 0.90, 'eta_t': 0.90, 'eps_recup': 0.90}
            if os.path.exists(_nc_input_csv):
                with open(_nc_input_csv, newline="") as _fh:
                    for _row in csv.reader(_fh):
                        if len(_row) >= 2 and not str(_row[0]).strip().startswith("#"):
                            _br_raw[_row[0].strip()] = _row[1].strip()
            _br = {
                'P1':        float(_br_raw.get('P1_Pa',         2e6)),
                'T1_C':      float(_br_raw.get('T1_C',          30.0)),
                'PR':        float(_br_raw.get('pressure_ratio', 2.3)),
                'eta_c':     float(_br_raw.get('eta_c',         0.90)),
                'eta_t':     float(_br_raw.get('eta_t',         0.90)),
                'eps_recup': float(_br_raw.get('eps_recup',     0.90)),
            }
            _nc_htgr_avail = True
            print(f"  [MWe] nc_htgr loaded. Brayton: PR={_br['PR']}, "
                  f"η_c={_br['eta_c']}, η_t={_br['eta_t']}, ε_recup={_br['eps_recup']}")
        except Exception as _nc_ex:
            print(f"  WARNING: could not import nc_htgr ({_nc_ex}); MWe calculation skipped.")

        _step_dirs_map = {}   # bos time index → step directory path
        for _e in cs_log:
            _bos_idx = _e["step"] - 1
            if "search_csv" in _e and _e["search_csv"]:
                _step_dirs_map[_bos_idx] = os.path.dirname(_e["search_csv"])

        _th_time_yr  = []   # BOS time in years for each collected step
        _th_burnup   = []   # core-average burnup at BOS time
        _th_T_cool   = []   # max coolant temperature [K]
        _th_T_comp   = []   # max compact temperature [K]
        _th_T_mat    = []   # max matrix temperature [K]
        _th_q_max    = []   # max avg channel heating [W]
        _th_mwe      = []   # net electrical output [MWe]

        _mwe_m_dot = float(params.get('th_m_dot_kg_s', 0.0097))
        _mwe_N_cool = int(params.get('th_N_cool_channels', 558))

        for _bos_idx in sorted(_step_dirs_map.keys()):
            if _bos_idx >= n_plot:
                break   # only include operational steps
            _sdir = _step_dirs_map[_bos_idx]
            _temp_csv, _heat_csv = _get_last_th_coupler_files(_sdir)
            if _temp_csv is None:
                continue
            try:
                _td = np.genfromtxt(_temp_csv, delimiter=",", names=True)
                _T_cool = float(np.nanmax(_td["T_coolant_K"]))
                _T_comp = float(np.nanmax(_td["T_compact_K"]))
                _T_mat  = float(np.nanmax(_td["T_matrix_K"]))
            except Exception as _ex:
                print(f"    WARNING: could not read {_temp_csv}: {_ex}")
                continue
            _q_max = np.nan
            if _heat_csv:
                try:
                    _hd = np.genfromtxt(_heat_csv, delimiter=",", names=True)
                    _q_max = float(np.nanmax(_hd["avg_channel_q_W"]))
                except Exception:
                    pass
            # Net electrical output via Brayton cycle math using dT_core from CSV.
            # The TH CSV was produced with a fixed cold inlet (T_in from nc_input.csv),
            # but in the actual cycle the recuperator pre-heats the core inlet to T3 > T_in.
            # Since helium cp is nearly constant, the core temperature rise
            # dT_core = T4_csv − T_in_csv is independent of the actual inlet temperature.
            # We therefore iterate the recuperator (T3 = T2 + eps*(T5−T2)) with
            # T4 = T3 + dT_core until T3 converges, then compute Wnet.
            _mwe = np.nan
            if _nc_htgr_avail:
                try:
                    # Temperature rise through heated core (K) — from bottom (outlet) to top (inlet)
                    _T4_csv   = float(_td["T_coolant_K"][0])   # first row: bottom of core = coolant outlet
                    _T_in_csv = float(_td["T_coolant_K"][-1])  # last row:  top of core   = coolant inlet
                    _dT_core  = _T4_csv - _T_in_csv            # K, independent of absolute T_in
                    _cp, _gamma, _ = _nc_he_cp()
                    _T1 = _br['T1_C'] + 273.15   # compressor inlet [K]
                    _T2 = _nc_comp_T2(_T1, _br['PR'], _gamma, _br['eta_c'])
                    # Iterate recuperator to find self-consistent T3 and T4
                    _T3 = _T2   # initial guess (no recuperation)
                    for _it in range(200):
                        _T4 = _T3 + _dT_core
                        _T5 = _nc_turb_T5(_T4, _br['PR'], _gamma, _br['eta_t'])
                        _T3_new = _T2 + _br['eps_recup'] * (_T5 - _T2)
                        if abs(_T3_new - _T3) < 1e-4:
                            _T3 = _T3_new
                            break
                        _T3 = 0.7 * _T3 + 0.3 * _T3_new
                    _T4   = _T3 + _dT_core
                    _T5   = _nc_turb_T5(_T4, _br['PR'], _gamma, _br['eta_t'])
                    _Wnet = _cp * (_T4 - _T5) - _cp * (_T2 - _T1)
                    _mwe  = _mwe_m_dot * _Wnet * _mwe_N_cool / 1e6
                except Exception as _me:
                    print(f"    WARNING: MWe calc failed for step {_bos_idx}: {_me}")
            _th_time_yr.append(time_years[_bos_idx])
            _th_burnup.append(burnup_MWd_per_MtU[_bos_idx] if burnup_MWd_per_MtU is not None else np.nan)
            _th_T_cool.append(_T_cool)
            _th_T_comp.append(_T_comp)
            _th_T_mat.append(_T_mat)
            _th_q_max.append(_q_max)
            _th_mwe.append(_mwe)

        if _th_time_yr:
            _th_time_yr = np.array(_th_time_yr)
            _th_burnup  = np.array(_th_burnup)
            _th_T_cool  = np.array(_th_T_cool)
            _th_T_comp  = np.array(_th_T_comp)
            _th_T_mat   = np.array(_th_T_mat)
            _th_q_max   = np.array(_th_q_max)
            _th_mwe     = np.array(_th_mwe)

            def _add_burnup_secondary_axis(ax_main, t_yr, bu):
                """Add burnup (MWd/MtU) as secondary x-axis above the primary time axis.
                Burnup is linearly proportional to time (constant power), so the secondary
                axis limits scale directly with the primary axis limits."""
                if bu is None or np.all(np.isnan(bu)) or total_HM_mass_kg is None:
                    return
                _bu_per_yr = thermal_power_MW * 365.25 / (total_HM_mass_kg / 1000.0)
                ax_top = ax_main.twiny()
                _lo, _hi = ax_main.get_xlim()
                ax_top.set_xlim(_lo * _bu_per_yr, _hi * _bu_per_yr)
                ax_top.set_xlabel("Core-Average Burnup (MWd/MtU)")

            # --- Plot 1: Max Temperature vs Time (years) ---
            fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
            ax.plot(_th_time_yr, _th_T_cool, "o-", color="tab:blue",   label="Max Coolant Temp")
            ax.plot(_th_time_yr, _th_T_comp, "s-", color="tab:red",    label="Max Compact Temp")
            ax.plot(_th_time_yr, _th_T_mat,  "^-", color="tab:orange", label="Max Matrix Temp")
            if discharge_time_years is not None:
                ax.axvline(discharge_time_years, color="green", linestyle=":", alpha=0.7,
                           label=f"Discharge: {discharge_time_years:.2f} yr")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Max Temperature (K)")
            if show_titles:
                ax.set_title("Max Temperature vs. Time (CS Depletion Study)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            _add_burnup_secondary_axis(ax, _th_time_yr, _th_burnup)
            plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                      f"cs_depletion_max_temp_vs_time.{fig_fmt}"),
                        bbox_inches="tight")
            plt.close()
            print(f"  Saved: cs_depletion_max_temp_vs_time.{fig_fmt}")

            # --- Plot 2: Max Avg Channel Heating vs Time (years) ---
            _q_valid = ~np.isnan(_th_q_max)
            if np.any(_q_valid):
                fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
                ax.plot(_th_time_yr[_q_valid], _th_q_max[_q_valid],
                        "o-", color="tab:purple", label="Max Avg Channel Heating")
                if discharge_time_years is not None:
                    ax.axvline(discharge_time_years, color="green", linestyle=":", alpha=0.7,
                               label=f"Discharge: {discharge_time_years:.2f} yr")
                ax.set_xlabel("Time (years)")
                ax.set_ylabel("Max Avg Channel Heating (W)")
                if show_titles:
                    ax.set_title("Max Avg Channel Heating vs. Time (CS Depletion Study)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                _add_burnup_secondary_axis(ax, _th_time_yr, _th_burnup)
                plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                          f"cs_depletion_max_heating_vs_time.{fig_fmt}"),
                            bbox_inches="tight")
                plt.close()
                print(f"  Saved: cs_depletion_max_heating_vs_time.{fig_fmt}")

            # --- Plot 3: Net Electrical Output (MWe) vs Time ---
            _mwe_valid = ~np.isnan(_th_mwe)
            if _nc_htgr_avail and np.any(_mwe_valid):
                fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
                ax.plot(_th_time_yr[_mwe_valid], _th_mwe[_mwe_valid],
                        "o-", color="tab:green", label="Net Electrical Output")
                if discharge_time_years is not None:
                    ax.axvline(discharge_time_years, color="green", linestyle=":", alpha=0.7,
                               label=f"Discharge: {discharge_time_years:.2f} yr")
                ax.set_xlabel("Time (years)")
                ax.set_ylabel("Net Electrical Output (MWe)")
                if show_titles:
                    ax.set_title("Net Electrical Output vs. Time (CS Depletion Study)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                _add_burnup_secondary_axis(ax, _th_time_yr, _th_burnup)
                plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                          f"cs_depletion_mwe_vs_time.{fig_fmt}"),
                            bbox_inches="tight")
                plt.close()
                print(f"  Saved: cs_depletion_mwe_vs_time.{fig_fmt}")

                # Save CSV
                _mwe_csv_path = os.path.join(POSTPROCESSING_RESULTS_DIR,
                                              "cs_depletion_mwe_vs_time.csv")
                with open(_mwe_csv_path, "w", newline="") as _fcsv:
                    _cw = csv.writer(_fcsv)
                    _cw.writerow(["time_years", "burnup_MWd_per_MtU", "net_electrical_MWe"])
                    for _ri in range(len(_th_time_yr)):
                        _cw.writerow([
                            f"{_th_time_yr[_ri]:.6f}",
                            f"{_th_burnup[_ri]:.2f}" if not np.isnan(_th_burnup[_ri]) else "",
                            f"{_th_mwe[_ri]:.5f}"    if not np.isnan(_th_mwe[_ri])    else "",
                        ])
                print(f"  Saved: cs_depletion_mwe_vs_time.csv")
        else:
            print("  WARNING: No TH data found in step directories; skipping TH plots.")

    else:
        # DepletionStudy: single EOS keff trace — truncate at last operational step.
        fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
        ax.errorbar(x_data[:n_plot], keff_mean[:n_plot], yerr=keff_std[:n_plot],
                    fmt="o-", capsize=3, label="k-effective")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1, label="k = 1.0")
        _add_discharge_vline(ax, is_burnup=(burnup_MWd_per_MtU is not None))
        ax.set_xlabel(x_label)
        ax.set_ylabel("k-effective")
        if show_titles:
            ax.set_title("k-effective vs. Burnup")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_keff_vs_{x_label_short}.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()

        if burnup_MWd_per_MtU is not None:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
            ax.errorbar(time_days[:n_plot], keff_mean[:n_plot], yerr=keff_std[:n_plot],
                        fmt="o-", capsize=3, label="k-effective")
            ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, linewidth=1, label="k = 1.0")
            _add_discharge_vline(ax, is_burnup=False)
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("k-effective")
            if show_titles:
                ax.set_title("k-effective vs. Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / 365.25, ax.get_xlim()[1] / 365.25)
            ax2.set_xlabel("Time (years)")
            plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                      f"depletion_keff_vs_time.{fig_fmt}"),
                        bbox_inches="tight")
            plt.close()

        reactivity_pcm = (keff_mean[:n_plot] - 1.0) / keff_mean[:n_plot] * 1e5
        fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
        ax.plot(x_data[:n_plot], reactivity_pcm, "o-")
        ax.axhline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Reactivity (pcm)")
        if show_titles:
            ax.set_title("Excess Reactivity vs. Burnup")
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_reactivity_vs_{x_label_short}.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()

    # Nuclide group plots — driven entirely by params["depletion_plot_groups"]
    # Slice x_data to operational range only.
    plot_groups = params.get("depletion_plot_groups", DEFAULT_PLOT_GROUPS)
    plotted_nuclides = set()

    for group_name, group_nuclides in plot_groups.items():
        available = [n for n in group_nuclides if n in all_nuclide_data]
        if available:
            _plot_nuclide_group(
                x_data[:n_plot], x_label, all_nuclide_data, available,
                group_name, POSTPROCESSING_RESULTS_DIR,
                f"depletion_{group_name.lower().replace('/', '').replace(' ', '_')}",
                is_wedge=is_wedge, show_titles=show_titles, fig_fmt=fig_fmt, fig_dpi=fig_dpi,
                n_plot=n_plot
            )
            plotted_nuclides.update(available)

    # Any tracked nuclides not covered by a plot group
    ungrouped = [
        n for n in tracked_nuclides
        if n in all_nuclide_data and n not in plotted_nuclides
    ]
    if ungrouped:
        _plot_nuclide_group(
            x_data[:n_plot], x_label, all_nuclide_data, ungrouped,
            "Other Tracked Nuclides", POSTPROCESSING_RESULTS_DIR,
            "depletion_other_nuclides",
            is_wedge=is_wedge, show_titles=show_titles, fig_fmt=fig_fmt, fig_dpi=fig_dpi,
            n_plot=n_plot
        )

    # Fissile inventory ratio
    fissile_present = [n for n in ["U235", "Pu239", "Pu241"] if n in fuel_data]
    if fissile_present and "U235" in fuel_data and fuel_data["U235"][0] > 0:
        fissile_initial = fuel_data["U235"][0]
        fissile_current = sum(fuel_data[n] for n in fissile_present)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=fig_dpi)
        ax.plot(x_data[:n_plot], (fissile_current / fissile_initial)[:n_plot],
                "o-", color="tab:green")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Fissile Inventory Ratio")
        if show_titles:
            ax.set_title(
                f"Fissile Inventory Ratio vs. Burnup\n"
                f"({' + '.join(fissile_present)}) / Initial U-235"
            )
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR, f"depletion_fissile_ratio_vs_{x_label_short}.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()

    # ---- B-10 burnout helpers ----
    def _save_fig(name):
        path = os.path.join(POSTPROCESSING_RESULTS_DIR, f"{name}.{fig_fmt}")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {name}.{fig_fmt}")

    def _b10_absolute(b10_array, label, filename, color):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=fig_dpi)
        ax.plot(x_data[:n_plot], b10_array[:n_plot], "o-", color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel("B-10 Atoms")
        if show_titles:
            ax.set_title(f"B-10 Absolute Inventory ({label})")
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        _save_fig(filename)

    def _b10_fractional(b10_array, b10_initial, label, filename, color):
        if b10_initial <= 0:
            return
        fig, ax = plt.subplots(figsize=(10, 5), dpi=fig_dpi)
        ax.plot(x_data[:n_plot], b10_array[:n_plot] / b10_initial * 100, "o-", color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Remaining B-10 (%)")
        if show_titles:
            ax.set_title(f"B-10 Fractional Burnout ({label})")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        _save_fig(filename)

    # B-10 burnout — burnable poison
    b10_p = all_nuclide_data.get("B10_poison") if "B10_poison" in all_nuclide_data \
            else all_nuclide_data.get("B10")
    if b10_p is not None:
        _b10_absolute(b10_p, "Burnable Poison",
                      "depletion_B10_burnout_poison_absolute", "purple")
        _b10_fractional(b10_p, b10_p[0], "Burnable Poison",
                        "depletion_B10_burnout_poison_fractional", "purple")

    # B-10 burnout — graphite
    b10_g = all_nuclide_data.get("B10_graphite")
    if b10_g is not None:
        _b10_absolute(b10_g, "Graphite",
                      "depletion_B10_burnout_graphite_absolute", "steelblue")
        _b10_fractional(b10_g, b10_g[0], "Graphite",
                        "depletion_B10_burnout_graphite_fractional", "steelblue")

    # B-10 combined — only when both sources present
    if b10_p is not None and b10_g is not None:
        b10_combined  = b10_p + b10_g
        b10_comb_init = b10_combined[0]

        # Absolute — all three traces
        fig, ax = plt.subplots(figsize=(10, 5), dpi=fig_dpi)
        ax.plot(x_data[:n_plot], b10_combined[:n_plot], "o-",  color="black",     label="Total")
        ax.plot(x_data[:n_plot], b10_p[:n_plot],        "s--", color="purple",    label="Burnable Poison")
        ax.plot(x_data[:n_plot], b10_g[:n_plot],        "^--", color="steelblue", label="Graphite")
        ax.set_xlabel(x_label)
        ax.set_ylabel("B-10 Atoms")
        if show_titles:
            ax.set_title("B-10 Absolute Inventory (Combined)")
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save_fig("depletion_B10_burnout_combined_absolute")

        # Fractional — all three traces normalised to combined initial
        if b10_comb_init > 0:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=fig_dpi)
            ax.plot(x_data[:n_plot], b10_combined[:n_plot] / b10_comb_init * 100,
                    "o-",  color="black",     label="Total")
            ax.plot(x_data[:n_plot], b10_p[:n_plot] / b10_comb_init * 100,
                    "s--", color="purple",    label="Burnable Poison")
            ax.plot(x_data[:n_plot], b10_g[:n_plot] / b10_comb_init * 100,
                    "^--", color="steelblue", label="Graphite")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Remaining B-10 (% of initial total)")
            if show_titles:
                ax.set_title("B-10 Fractional Burnout (Combined)")
            ax.set_ylim(0, 105)
            ax.legend()
            ax.grid(True, alpha=0.3)
            _save_fig("depletion_B10_burnout_combined_fractional")

        # By-source stacked area
        if b10_comb_init > 0:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=fig_dpi)
            ax.stackplot(x_data[:n_plot],
                         b10_p[:n_plot] / b10_comb_init * 100,
                         b10_g[:n_plot] / b10_comb_init * 100,
                         labels=["Burnable Poison", "Graphite"],
                         colors=["purple", "steelblue"], alpha=0.7)
            ax.set_xlabel(x_label)
            ax.set_ylabel("B-10 Source Fraction (%)")
            if show_titles:
                ax.set_title("B-10 Inventory by Source")
            ax.set_ylim(0, 105)
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)
            _save_fig("depletion_B10_burnout_combined_bysource")

    # ================================================================================
    # 5b. ZONE BURNUP IN MWd/MtU  (heating-local tally — primary method)
    # ================================================================================

    fuel_mat_volumes_raw = params.get("fuel_mat_volumes", {})
    fuel_mat_volumes_str = {str(k): float(v) for k, v in fuel_mat_volumes_raw.items()}

    mwdmtu_data = None
    if fuel_mat_ids_2d is not None:
        thermal_power_MW      = params.get("thermal_power_MW", 10.0)
        total_HM_mass_kg      = params.get("total_HM_mass_kg")
        fuel_volume_full_core = params.get("fuel_volume_full_core_cm3")
        if total_HM_mass_kg and fuel_volume_full_core:
            print("\n  --- MWd/MtU zone burnup (heating-local tally) ---")
            mwdmtu_data = extract_min_max_burnup_MWdMtU(
                run_dir                   = run_dir,
                fuel_mat_ids_2d           = fuel_mat_ids_2d,
                fuel_mat_volumes          = fuel_mat_volumes_str,
                time_days                 = time_days,
                thermal_power_MW          = thermal_power_MW,
                total_HM_mass_kg          = float(total_HM_mass_kg),
                fuel_volume_full_core_cm3 = float(fuel_volume_full_core),
                symmetry_factor           = symmetry_factor,
            )

    if mwdmtu_data is not None:
        peak_bu   = mwdmtu_data["peak_burnup_MWd_MtU"]
        min_bu    = mwdmtu_data["min_burnup_MWd_MtU"]
        avg_bu    = mwdmtu_data["avg_burnup_MWd_MtU"]
        peak_zone = mwdmtu_data["peak_zone"]
        min_zone  = mwdmtu_data["min_zone"]
        zone_lbl  = mwdmtu_data["zone_labels"]

        # Plot: peak / avg / min burnup in MWd/MtU (truncate at last operational step)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
        ax.plot(x_data[:n_plot], avg_bu[:n_plot],  "o-",  color="black",    label="Core-average burnup")
        ax.plot(x_data[:n_plot], peak_bu[:n_plot], "s--", color="firebrick", label="Peak zone burnup")
        ax.plot(x_data[:n_plot], min_bu[:n_plot],  "^--", color="steelblue", label="Minimum zone burnup")
        ax.fill_between(x_data[:n_plot], avg_bu[:n_plot], peak_bu[:n_plot], alpha=0.10, color="firebrick",
                        label="Peak-to-average margin")
        ax.fill_between(x_data[:n_plot], min_bu[:n_plot], avg_bu[:n_plot], alpha=0.10, color="steelblue",
                        label="Average-to-minimum margin")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Burnup (MWd/MtU)")
        if show_titles:
            ax.set_title("Peak / Average / Minimum Burnup (MWd/MtU)\n(heating-local tally)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_burnup_MWdMtU_vs_{x_label_short}.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_burnup_MWdMtU_vs_{x_label_short}.{fig_fmt}")

        # Peaking factor vs burnup (MWd/MtU basis)
        with np.errstate(divide='ignore', invalid='ignore'):
            pf_MWd = np.where(avg_bu > 0, peak_bu / avg_bu, np.nan)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=fig_dpi)
        ax.plot(x_data[:n_plot], pf_MWd[:n_plot], "o-", color="darkorange")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Peak-to-Average Burnup Ratio")
        if show_titles:
            ax.set_title("Burnup Peaking Factor vs. Burnup (MWd/MtU)")
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_burnup_peaking_MWdMtU_vs_{x_label_short}.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_burnup_peaking_MWdMtU_vs_{x_label_short}.{fig_fmt}")

        # CSV: MWd/MtU zone burnup per step
        mwdmtu_csv = os.path.join(POSTPROCESSING_RESULTS_DIR, "depletion_burnup_MWdMtU.csv")
        header_mwd = ("step,time_days,time_years"
                      + (",burnup_avg_MWd_MtU" if burnup_MWd_per_MtU is not None else "")
                      + ",burnup_peak_MWd_MtU,burnup_min_MWd_MtU,peak_to_avg_ratio"
                        ",peak_zone,min_zone,operational")
        rows_mwd = []
        for i in range(len(time_days)):
            row = f"{i},{time_days[i]:.4f},{time_years[i]:.6f}"
            if burnup_MWd_per_MtU is not None:
                row += f",{avg_bu[i]:.2f}"
            pf_val = float(pf_MWd[i]) if not np.isnan(pf_MWd[i]) else 0.0
            op_val = int(operational[i])
            row += (f",{peak_bu[i]:.2f},{min_bu[i]:.2f},{pf_val:.4f},"
                    f"{zone_lbl[int(peak_zone[i])]},{zone_lbl[int(min_zone[i])]},{op_val}")
            rows_mwd.append(row)
        with open(mwdmtu_csv, "w") as f:
            f.write(header_mwd + "\n")
            for r in rows_mwd:
                f.write(r + "\n")
        print(f"  Saved: {os.path.basename(mwdmtu_csv)}")

    # ================================================================================
    # 5c. ZONE BURNUP IN %FIMA  (actinide inventory method)
    # ================================================================================

    fima_data = None
    if fuel_mat_ids_2d is not None:
        print("\n  --- %FIMA zone burnup (actinide inventory) ---")
        fima_data = extract_min_max_burnup_FIMA(
            results            = results,
            fuel_mat_ids_2d    = fuel_mat_ids_2d,
            time_days          = time_days,
            burnup_MWd_per_MtU = burnup_MWd_per_MtU,
            symmetry_factor    = symmetry_factor,
        )

    if fima_data is not None:
        peak_FIMA      = fima_data["peak_burnup_pct_FIMA"]
        min_FIMA       = fima_data["min_burnup_pct_FIMA"]
        avg_FIMA       = fima_data["avg_burnup_pct_FIMA"]
        avg_bu_ref     = fima_data["avg_burnup_MWd_MtU"]   # x-axis: always MWd/MtU
        peak_zone_f    = fima_data["peak_zone"]
        min_zone_f     = fima_data["min_zone"]
        zone_lbl_f     = fima_data["zone_labels"]

        x_fima      = avg_bu_ref                            # x = core-average burnup MWd/MtU
        x_fima_lbl  = "Core-Average Burnup (MWd/MtU)"

        # Plot: peak / avg / min burnup in %FIMA vs. average burnup MWd/MtU
        fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
        _nf = min(n_plot, len(x_fima)) if x_fima is not None else n_plot
        ax.plot(x_fima[:_nf], avg_FIMA[:_nf],  "o-",  color="black",    label="Core-average burnup")
        ax.plot(x_fima[:_nf], peak_FIMA[:_nf], "s--", color="firebrick", label="Peak zone burnup")
        ax.plot(x_fima[:_nf], min_FIMA[:_nf],  "^--", color="steelblue", label="Minimum zone burnup")
        ax.fill_between(x_fima[:_nf], avg_FIMA[:_nf], peak_FIMA[:_nf], alpha=0.10, color="firebrick",
                        label="Peak-to-average margin")
        ax.fill_between(x_fima[:_nf], min_FIMA[:_nf], avg_FIMA[:_nf], alpha=0.10, color="steelblue",
                        label="Average-to-minimum margin")
        ax.set_xlabel(x_fima_lbl)
        ax.set_ylabel("Burnup (%FIMA)")
        if show_titles:
            bu_method_fima = fima_data.get("method", "actinide inventory")
            ax.set_title(f"Peak / Average / Minimum Burnup (%FIMA)\n({bu_method_fima})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_burnup_FIMA_vs_MWdMtU.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_burnup_FIMA_vs_MWdMtU.{fig_fmt}")

        # Peaking factor vs burnup (%FIMA basis)
        with np.errstate(divide='ignore', invalid='ignore'):
            pf_FIMA = np.where(avg_FIMA > 0, peak_FIMA / avg_FIMA, np.nan)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=fig_dpi)
        ax.plot(x_fima[:_nf], pf_FIMA[:_nf], "o-", color="darkorange")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel(x_fima_lbl)
        ax.set_ylabel("Peak-to-Average Burnup Ratio")
        if show_titles:
            ax.set_title("Burnup Peaking Factor vs. Burnup (%FIMA)")
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_burnup_peaking_FIMA_vs_MWdMtU.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_burnup_peaking_FIMA_vs_MWdMtU.{fig_fmt}")

        # CSV: %FIMA zone burnup per step
        fima_csv = os.path.join(POSTPROCESSING_RESULTS_DIR, "depletion_burnup_FIMA.csv")
        header_fima = ("step,time_days,time_years,burnup_avg_MWd_MtU"
                       ",burnup_avg_pct_FIMA,burnup_peak_pct_FIMA,burnup_min_pct_FIMA"
                       ",peak_to_avg_ratio,peak_zone,min_zone,operational")
        rows_fima = []
        for i in range(len(time_days)):
            pf_val = float(pf_FIMA[i]) if not np.isnan(pf_FIMA[i]) else 0.0
            op_val = int(operational[i])
            rows_fima.append(
                f"{i},{time_days[i]:.4f},{time_years[i]:.6f},"
                f"{avg_bu_ref[i]:.2f},{avg_FIMA[i]:.5f},"
                f"{peak_FIMA[i]:.5f},{min_FIMA[i]:.5f},{pf_val:.4f},"
                f"{zone_lbl_f[int(peak_zone_f[i])]},{zone_lbl_f[int(min_zone_f[i])]},{op_val}"
            )
        with open(fima_csv, "w") as f:
            f.write(header_fima + "\n")
            for r in rows_fima:
                f.write(r + "\n")
        print(f"  Saved: {os.path.basename(fima_csv)}")

    # ================================================================================
    # 5d. CONVERSION RATIO
    # ================================================================================

    cr_data = calculate_conversion_ratio(fuel_data, time_days)

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

        # --- Plot: CR vs burnup/time (truncate at last operational step) ---
        _n_cr_plot = min(n_cr, n_plot - 1)   # CR has n-1 midpoints for n timesteps
        fig, ax = plt.subplots(figsize=(12, 5), dpi=fig_dpi)
        valid = ~np.isnan(cr[:_n_cr_plot])
        ax.plot(x_cr[:_n_cr_plot][valid], cr[:_n_cr_plot][valid], "o-", color="tab:orange")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Conversion Ratio")
        if show_titles:
            ax.set_title("Conversion Ratio vs. Burnup")
        cr_max = float(np.nanmax(cr[:_n_cr_plot])) if np.any(valid) else 1.0
        ax.set_ylim(0, 1.5 * cr_max)
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_conversion_ratio_vs_{x_label_short}.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_conversion_ratio_vs_{x_label_short}.{fig_fmt}")

        # --- Plot: Pu-239 generated and fissile burned per step ---
        fig, ax = plt.subplots(figsize=(12, 5), dpi=fig_dpi)
        ax.plot(x_cr[:_n_cr_plot], fis_burned[:_n_cr_plot], "o-", color="tab:red",  label="Total fissile burned")
        ax.plot(x_cr[:_n_cr_plot], pu239_gen[:_n_cr_plot],  "s-", color="tab:blue", label="Pu-239 generated (gross)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Atoms per step")
        if show_titles:
            ax.set_title("Fissile Burned vs. Pu-239 Generated per Step")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.savefig(os.path.join(POSTPROCESSING_RESULTS_DIR,
                                  f"depletion_fissile_balance_vs_{x_label_short}.{fig_fmt}"),
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved: depletion_fissile_balance_vs_{x_label_short}.{fig_fmt}")

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
        print(f"  Saved: {os.path.basename(cr_csv)}")

        # Print summary
        valid_cr = cr[:_n_cr_plot][valid]
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
        "study_execution_mode":          params.get("study_execution_mode", "DepletionStudy"),
        "time_days":                     time_days.tolist(),
        "time_years":                    time_years.tolist(),
        "keff_mean":                     keff_mean.tolist(),
        "keff_std":                      keff_std.tolist(),
        "operational":                   operational.tolist(),
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
        "peak_burnup_MWd_per_MtU":      (mwdmtu_data["peak_burnup_MWd_MtU"].tolist()
                                         if mwdmtu_data is not None else None),
        "peak_burnup_final_MWd_per_MtU":(float(mwdmtu_data["peak_burnup_MWd_MtU"][-1])
                                         if mwdmtu_data is not None else None),
        "peak_burnup_zone_final":        (mwdmtu_data["zone_labels"][
                                              int(mwdmtu_data["peak_zone"][-1])]
                                         if mwdmtu_data is not None else None),
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
    # keff / keff_std are always EOS values from the depletion h5.
    # For CSDepletionStudy, BOS critical keff and rod positions are appended
    # from critical_search_depletion_log.json.
    #   Alignment: log entry step=i (BOS of step i) → time_days[i-1] (start of step i).
    #   The final time point has no BOS entry and is filled with NaN.

    header = "time_days,time_years,keff_eos,keff_eos_std" if cs_log else \
             "time_days,time_years,keff,keff_std"
    cols   = [time_days, time_years, keff_mean, keff_std]
    if burnup_MWd_per_MtU is not None:
        header += ",burnup_MWd_per_MtU"
        cols.append(burnup_MWd_per_MtU)
    header += ",operational"
    cols.append(operational)

    if cs_log:
        # Reuse BOS arrays computed earlier
        header += ",keff_bos,keff_bos_std,bank_1_insertion,bank_2_insertion"
        cols.extend([bos_keff, bos_keff_std, bos_bank1, bos_bank2])

    np.savetxt(os.path.join(POSTPROCESSING_RESULTS_DIR, "depletion_keff_data.csv"),
               np.column_stack(cols), delimiter=",", header=header, comments="")

    # ----- Nuclide Inventory CSV Report -----

    print("\nExporting nuclide inventory CSVs...")
    save_nuclide_inventory_csv(
        POSTPROCESSING_RESULTS_DIR, time_days, time_years, burnup_MWd_per_MtU,
        fuel_data, poison_data, operational=operational
    )

    if graphite_data:
        n_steps   = len(time_days)
        index_cols   = [np.arange(n_steps), time_days, time_years]
        index_header = "step,time_days,time_years"
        fmt_cols     = ["%.0f", "%.6f", "%.8f"]
        if burnup_MWd_per_MtU is not None:
            index_cols.append(burnup_MWd_per_MtU)
            index_header += ",burnup_MWd_per_MtU"
            fmt_cols.append("%.4f")
        if operational is not None:
            index_cols.append(operational[:n_steps])
            index_header += ",operational"
            fmt_cols.append("%.0f")
        nuclides  = sorted(graphite_data.keys())
        nuc_cols  = [graphite_data[n][:n_steps] for n in nuclides]
        out_path  = os.path.join(POSTPROCESSING_RESULTS_DIR, "nuclide_inventory_graphite.csv")
        np.savetxt(
            out_path,
            np.column_stack(index_cols + nuc_cols),
            delimiter=",",
            header=index_header + "," + ",".join(nuclides),
            comments="",
            fmt=fmt_cols + ["%.6e"] * len(nuclides),
        )
        print(f"  [Graphite] Nuclide inventory saved → {os.path.basename(out_path)}  "
              f"({n_steps} steps × {len(nuclides)} nuclides)")

    # ----- Gamma Sources EOL CSV -----
    gamma_sources = params.get("gamma_sources", [])
    if gamma_sources:
        export_gamma_sources_csv(
            results, gamma_sources, last_operational_idx,
            symmetry_factor, POSTPROCESSING_RESULTS_DIR
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
        # For CSDepletionStudy use BOS keff; otherwise use EOS keff from results.h5
        _rpt_keff = bos_keff if (cs_log and bos_keff is not None) else keff_mean
        _rpt_kstd = bos_keff_std if (cs_log and bos_keff_std is not None) else keff_std
        _keff_label = "BOS k_eff (crit. search)" if cs_log else "k_eff"

        # Header summary lines
        _k0 = _rpt_keff[0] if not np.isnan(_rpt_keff[0]) else keff_mean[0]
        _s0 = _rpt_kstd[0] if not np.isnan(_rpt_kstd[0]) else keff_std[0]
        _kl = _rpt_keff[last_operational_idx]
        _sl = _rpt_kstd[last_operational_idx]
        if np.isnan(_kl):
            _kl, _sl = keff_mean[last_operational_idx], keff_std[last_operational_idx]
        f.write(f"Initial {_keff_label}:  {_k0:.5f} ± {_s0:.5f}\n")
        f.write(f"Final   {_keff_label}:  {_kl:.5f} ± {_sl:.5f}  "
                f"(step {last_operational_idx}, t={time_days[last_operational_idx]:.0f} d)\n\n")
        if burnup_MWd_per_MtU is not None:
            f.write(f"Final burnup:     {burnup_MWd_per_MtU[last_operational_idx]:.0f} MWd/MtU\n\n")
        if discharge_time_days is not None:
            f.write(f"Discharge (k=1.0): {discharge_time_days:.1f} days "
                    f"({discharge_time_years:.2f} years)\n")
            if discharge_burnup is not None:
                f.write(f"Discharge burnup:  {discharge_burnup:.0f} MWd/MtU\n")

        _col_keff = "BOS k_eff" if cs_log else "k_eff"
        f.write("\n" + "-" * 75 + "\n")
        f.write(f"{'Step':>5}  {'Time (d)':>10}  {_col_keff:>10}  {'± σ':>8}")
        if burnup_MWd_per_MtU is not None:
            f.write(f"  {'Burnup (MWd/MtU)':>18}")
        f.write("\n" + "-" * 75 + "\n")
        for i in range(n_plot):
            _k = _rpt_keff[i]
            _s = _rpt_kstd[i]
            if np.isnan(_k):
                _k, _s = keff_mean[i], keff_std[i]
            f.write(f"{i:>5}  {time_days[i]:>10.1f}  {_k:>10.5f}  {_s:>8.5f}")
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

        # Isotopic summary — Graphite (final values from last operational step)
        if graphite_data:
            f.write("\n" + "=" * 80 + "\n")
            f.write("GRAPHITE ISOTOPIC INVENTORY SUMMARY (atoms)\n")
            f.write(f"  Final values at last operational step: {last_operational_idx} "
                    f"(t = {time_days[last_operational_idx]:.1f} days, "
                    f"k = {keff_mean[last_operational_idx]:.5f})\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Nuclide':<12}  {'Initial':>16}  {'Final (op)':>16}  {'Change (%)':>12}\n")
            f.write("-" * 62 + "\n")
            for nuc, atoms in graphite_data.items():
                initial = atoms[0]
                final   = atoms[last_operational_idx]
                pct     = (final - initial) / initial * 100 if initial > 0 else float('nan')
                f.write(f"{nuc:<12}  {initial:>16.4e}  {final:>16.4e}  {pct:>+12.2f}%\n")

        f.write("=" * 80 + "\n")

    print(f"  Report saved to: {os.path.basename(txt_path)}")
    print(f"\n{'=' * 80}")
    print("DEPLETION POST-PROCESSING COMPLETE")
    print(f"{'=' * 80}\n")

    return summary

# ====================================================================================================
# NUCLIDE GROUP PLOTTING
# ====================================================================================================

def _plot_nuclide_group(x_data, x_label, nuclide_data, nuclide_list, title, output_dir, filename_base, is_wedge=False, show_titles=True, fig_fmt="png", fig_dpi=300, n_plot=None):
    available = [
        n for n in nuclide_list
        if n in nuclide_data and np.any(nuclide_data[n] > 0)
    ]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(12, 6), **({} if fig_dpi is None else {"dpi": fig_dpi}))
    for nuc in available:
        atoms = nuclide_data[nuc]
        n     = min(len(x_data), len(atoms))
        if n_plot is not None:
            n = min(n, n_plot)
        ax.plot(x_data[:n], atoms[:n], "o-", label=nuc)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of Atoms")
    if show_titles:
        ax.set_title(title)
    ax.legend(ncol=min(4, len(available)))
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    save_path = os.path.join(output_dir, f"{filename_base}.{fig_fmt}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")

# ====================================================================================================
# STANDALONE ENTRY POINT
# ====================================================================================================

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pdf_output = "--pdf" in sys.argv

    if len(args) < 1:
        print("Usage: python depletion_postprocessing.py <run_directory> [--pdf]")
        sys.exit(1)

    run_dir = args[0]

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

    run_depletion_postprocessing(run_dir, params, pdf_output=pdf_output)