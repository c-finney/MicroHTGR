"""
BeO Reflector Depletion Post-Processing Script

Extracts and plots BeO reflector fluence results from OpenMC depletion simulations:
 - Peak flux per depletion step from per-step statepoints (beo_flux_radial tally)
 - Cumulative peak fluence vs. burnup/time
 - BeO fluence CSV export

Usage:
    # As a module:
    from BeO_depletion_postprocessing import run_BeO_depletion_postprocessing
    run_BeO_depletion_postprocessing(run_dir, params)

    # Standalone:
    python BeO_depletion_postprocessing.py <run_directory>
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import openmc
import openmc.deplete

# ====================================================================================================
# NORMALIZATION HELPER
# ====================================================================================================

def _beo_normalization_factor(sp, thermal_power_W):
    """
    Compute the source-particle normalization factor from a statepoint.

    Returns source_rate [particles/s] such that:
        physical_flux [n/cm²/s] = (raw_tally_flux [cm/source] / cell_volume [cm³])
                                  × source_rate [source/s]

    Tries the 'heating' global tally first, then falls back to the 'global_rates'
    fission tally using 200 MeV per fission.

    Returns None if neither tally is found.
    """
    # --- Heating-local tally (preferred) ---
    for t in sp.tallies.values():
        if t.name == 'heating':
            try:
                heating_eV = float(t.get_values(scores=['heating-local']).sum())
                if heating_eV > 0:
                    return thermal_power_W / (heating_eV * 1.602176634e-19)
            except Exception:
                pass

    # --- Global rates / fission tally (fallback) ---
    for t in sp.tallies.values():
        if t.name == 'global_rates':
            try:
                fission_raw = float(t.get_values(scores=['fission']).sum())
                if fission_raw > 0:
                    return thermal_power_W / (fission_raw * 3.2e-11)  # 200 MeV/fission
            except Exception:
                pass

    return None


# ====================================================================================================
# BEO PEAK FLUENCE EXTRACTION
# ====================================================================================================

def extract_beo_peak_fluence(run_dir, time_steps_s, keff_mean, params):
    """
    Extract peak BeO reflector fluence at every depletion timestep.

    For each per-step statepoint (openmc_simulation_n{i}.h5), the function:
      1. Reads the 'beo_flux_radial' CylindricalMesh tally.
      2. Normalises the raw (per-source-particle) flux to physical flux [n/cm²/s]
         using the total thermal power and the 'heating' global tally.
      3. Finds the peak flux density across all (r, φ, z) mesh cells.
      4. Multiplies by the step duration to obtain the step's fluence contribution.

    Total peak fluence is accumulated only while the reactor is operational
    (k_eff ≥ 1.0). The first step where k_eff < 1.0 is treated as the shutdown
    point; no further fluence is integrated after that.

    Parameters
    ----------
    run_dir       : str  — depletion run directory (contains openmc_simulation_n*.h5)
    time_steps_s  : np.ndarray  shape (n_steps+1,) — absolute times [s] from results.get_keff()
    keff_mean     : np.ndarray  shape (n_steps+1,) — k_eff at each depletion step
    params        : dict — simulation parameters

    Returns
    -------
    dict with keys:
        peak_flux_per_step_n_cm2_s : (n_sp,) array — physical peak flux at each step [n/cm²/s]
        step_fluence_n_cm2         : (n_sp,) array — peak fluence contribution per step [n/cm²]
        cumulative_fluence_n_cm2   : (n_steps+1,) array — cumulative fluence at each result time [n/cm²]
        total_peak_fluence_n_cm2   : float — total integrated peak fluence [n/cm²]
        shutdown_step_idx          : int — first step index where k_eff < 1.0 (n_steps+1 if never)
        n_statepoints_found        : int
    None if BeO tallies are disabled or no statepoints are found.
    """
    if not params.get("use_BeO_tallies", False) or not params.get("use_BeO_reflector", False):
        return None

    thermal_power_W = params.get("thermal_power_MW", 10.0) * 1e6

    # ---- Locate per-step statepoints ----------------------------------------
    sp_paths = sorted(
        glob.glob(os.path.join(run_dir, "openmc_simulation_n*.h5")),
        key=lambda f: int(os.path.basename(f)
                          .replace("openmc_simulation_n", "")
                          .replace(".h5", ""))
    )

    if not sp_paths:
        print("  BeO fluence: no openmc_simulation_n*.h5 statepoints found — skipping")
        return None

    n_sp      = len(sp_paths)
    n_results = len(keff_mean)   # n_timesteps + 1

    print(f"\n  BeO peak fluence: processing {n_sp} step statepoints...")

    peak_flux = np.full(n_sp, np.nan)    # [n/cm²/s]
    step_flu  = np.full(n_sp, np.nan)    # [n/cm²]

    for i, sp_path in enumerate(sp_paths):
        step_label = os.path.basename(sp_path).replace(".h5", "")
        try:
            sp = openmc.StatePoint(sp_path)

            # Find tally
            beo_tally = next((t for t in sp.tallies.values()
                              if t.name == 'beo_flux_radial'), None)
            if beo_tally is None:
                print(f"    [{step_label}] 'beo_flux_radial' tally not found — skipping")
                continue

            # Normalization factor
            norm = _beo_normalization_factor(sp, thermal_power_W)
            if norm is None or norm <= 0:
                print(f"    [{step_label}] Could not determine normalization — skipping")
                continue

            # Raw fast flux (E > 100 keV): sum(track_length)/n_source [cm/source] per mesh cell
            # Tally has [MeshFilter, EnergyFilter(1 bin)]; squeeze out the energy dimension.
            flux_raw = beo_tally.get_values(scores=['flux']).flatten()

            # Recover mesh geometry for cell volumes
            mesh     = beo_tally.filters[0].mesh
            r_grid   = np.asarray(mesh.r_grid)
            phi_grid = np.asarray(mesh.phi_grid)
            z_grid   = np.asarray(mesh.z_grid)

            n_r   = len(r_grid)   - 1
            n_phi = len(phi_grid) - 1
            n_z   = len(z_grid)   - 1

            # Cell volumes: V = 0.5*(r_out²–r_in²)·Δφ·Δz
            dr2  = np.diff(r_grid**2)          # (n_r,)
            dphi = np.diff(phi_grid)           # (n_phi,)
            dz   = np.diff(z_grid)             # (n_z,)
            vols = (0.5 * dr2[:, np.newaxis, np.newaxis]
                    * dphi[np.newaxis, :, np.newaxis]
                    * dz[np.newaxis, np.newaxis, :])   # (n_r, n_phi, n_z)

            # flux_raw is ordered (r, phi, z, energy); with 1 energy bin this is
            # equivalent to (r, phi, z) after reshape.
            flux_density = flux_raw.reshape(n_r, n_phi, n_z) / vols

            # Peak fast flux density [n/cm²/s] — peak is geometry-invariant (same in all sectors)
            peak_flux[i] = float(np.nanmax(flux_density)) * norm

            # Step duration: interval from time_steps_s[i] to time_steps_s[i+1]
            if i + 1 < n_results:
                dt_s = float(time_steps_s[i + 1] - time_steps_s[i])
                step_flu[i] = peak_flux[i] * dt_s

            print(f"    [{step_label}] k={keff_mean[i]:.4f}  "
                  f"peak_fast_flux={peak_flux[i]:.3e} n/cm²/s  "
                  f"Δfast_fluence={step_flu[i]:.3e} n/cm²")

        except Exception as e:
            print(f"    [{step_label}] ERROR: {e}")

    # ---- Load operational array from depletion_summary.json if available -----
    # Prefer the summary's operational array (consistent with depletion_postprocessing.py).
    # For CSDepletionStudy, falls back to BOS keff >= 1.0 - k_tol from the log.
    # Final fallback: keff < 1.0.
    operational_arr = None
    summary_path = os.path.join(run_dir, "depletion_results", "depletion_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as _sf:
                _summary = json.load(_sf)
            if "operational" in _summary and _summary["operational"] is not None:
                operational_arr = np.array(_summary["operational"], dtype=int)
                print(f"  BeO: loaded operational array from depletion_summary.json")
        except Exception as _e:
            print(f"  BeO: WARNING: could not read depletion_summary.json: {_e}")

    if operational_arr is None and params.get("study_execution_mode") == "CSDepletionStudy":
        # Build operational array from critical_search_depletion_log.json using
        # BOS keff >= 1.0 - k_tol criterion (mirrors depletion_postprocessing.py logic).
        _cs_log_path = os.path.join(run_dir, "critical_search_depletion_log.json")
        if os.path.exists(_cs_log_path):
            try:
                with open(_cs_log_path) as _f:
                    _cs_log = json.load(_f)
                _k_tol = params.get("critical_search_k_tol", 0.0064)
                _k_thresh = 1.0 - _k_tol
                _n = n_results
                _bos_keff = np.full(_n, np.nan)
                for _e in _cs_log:
                    _bi = _e["step"] - 1
                    if 0 <= _bi < _n:
                        _bos_keff[_bi] = _e["critical_keff"]
                operational_arr = np.ones(_n, dtype=int)
                for _i in range(_n):
                    if not np.isnan(_bos_keff[_i]):
                        operational_arr[_i] = 1 if _bos_keff[_i] >= _k_thresh else 0
                print(f"  BeO: built operational array from critical_search_depletion_log.json "
                      f"(k_thresh={_k_thresh:.4f})")
            except Exception as _e:
                print(f"  BeO: WARNING: could not read critical_search_depletion_log.json: {_e}")

    # ---- Find shutdown step ---------------------------------------------------
    shutdown_idx = n_results   # default: never shuts down
    if operational_arr is not None and len(operational_arr) >= n_results:
        for idx in range(n_results):
            if operational_arr[idx] == 0:
                shutdown_idx = idx
                break
    else:
        # Fallback: first step where keff < 1.0
        for idx in range(n_results):
            if keff_mean[idx] < 1.0:
                shutdown_idx = idx
                break

    # ---- Integrate cumulative fluence (only while operational) ---------------
    cumulative = np.zeros(n_results)
    running_total = 0.0

    for i in range(n_sp):
        if i >= shutdown_idx:
            break
        if np.isnan(step_flu[i]):
            continue
        running_total += step_flu[i]
        if i + 1 < n_results:
            cumulative[i + 1] = running_total

    # Fill any remaining result indices with the final total
    for j in range(n_results):
        if cumulative[j] == 0.0 and j > 0:
            cumulative[j] = cumulative[j - 1]

    total_fluence = running_total

    print(f"\n  BeO peak fast fluence summary (E > 100 keV):")
    print(f"    Total peak fast fluence:  {total_fluence:.4e} n/cm²")
    if shutdown_idx < n_results:
        print(f"    Shutdown at step {shutdown_idx}")
    else:
        print(f"    Reactor remained operational throughout all steps")

    return {
        "peak_flux_per_step_n_cm2_s": peak_flux,
        "step_fluence_n_cm2":         step_flu,
        "cumulative_fluence_n_cm2":   cumulative,
        "total_peak_fluence_n_cm2":   total_fluence,
        "shutdown_step_idx":          shutdown_idx,
        "n_statepoints_found":        n_sp,
    }


# ====================================================================================================
# BEO PLOTTING AND CSV EXPORT
# ====================================================================================================

def plot_and_save_beo_results(beo_fluence_data, x_data, x_label, x_label_short,
                              time_days, keff_mean, burnup_MWd_per_MtU,
                              output_dir, show_titles=True):
    """
    Generate BeO fluence plots and CSV from pre-computed fluence data.

    Parameters
    ----------
    beo_fluence_data   : dict returned by extract_beo_peak_fluence (must not be None)
    x_data             : np.ndarray — x-axis values (burnup or time)
    x_label            : str — x-axis label
    x_label_short      : str — short label used in filenames ('burnup' or 'time')
    time_days          : np.ndarray
    keff_mean          : np.ndarray
    burnup_MWd_per_MtU : np.ndarray or None
    output_dir         : str — directory to write plots and CSV into
    show_titles        : bool — whether to add titles to plots (default True)
    """
    beo_peak_flux = beo_fluence_data["peak_flux_per_step_n_cm2_s"]
    beo_step_flu  = beo_fluence_data["step_fluence_n_cm2"]
    beo_cum_flu   = beo_fluence_data["cumulative_fluence_n_cm2"]
    beo_sd_idx    = beo_fluence_data["shutdown_step_idx"]
    n_sp          = beo_fluence_data["n_statepoints_found"]

    # Truncate to operational range — non-operational steps are non-physical.
    n_plot_beo = min(beo_sd_idx, len(x_data), len(beo_cum_flu))
    if n_plot_beo == 0:
        n_plot_beo = len(x_data)   # fallback: plot everything if always operational

    # --- Plot: cumulative peak fluence vs. time/burnup (truncated at shutdown) ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(x_data[:n_plot_beo], beo_cum_flu[:n_plot_beo], "o-", markersize=5,
            linewidth=1.5, color="darkcyan", label="Cumulative peak fluence")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Cumulative Peak Fast Fluence (n/cm²) [E > 100 keV]", fontsize=12)
    if show_titles:
        ax.set_title("BeO Reflector Peak Fast Fluence vs. Burnup", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.savefig(os.path.join(output_dir,
                              f"depletion_beo_fluence_vs_{x_label_short}.png"),
                bbox_inches="tight")
    plt.close()
    print(f"  Saved: depletion_beo_fluence_vs_{x_label_short}.png")

    # --- Plot: peak flux per step (truncated at shutdown) ---
    _n_flux_plot = min(beo_sd_idx, n_sp)
    step_indices = np.arange(_n_flux_plot)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.bar(step_indices,
           np.where(np.isnan(beo_peak_flux[:_n_flux_plot]), 0, beo_peak_flux[:_n_flux_plot]),
           color="teal", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Depletion Step Index", fontsize=12)
    ax.set_ylabel("Peak Fast Flux (n/cm²/s) [E > 100 keV]", fontsize=12)
    if show_titles:
        ax.set_title("BeO Reflector Peak Fast Flux per Depletion Step", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.savefig(os.path.join(output_dir,
                              "depletion_beo_peak_flux_per_step.png"),
                bbox_inches="tight")
    plt.close()
    print(f"  Saved: depletion_beo_peak_flux_per_step.png")

    # --- CSV: step-by-step BeO fluence ---
    beo_csv = os.path.join(output_dir, "depletion_beo_fluence.csv")
    with open(beo_csv, "w") as f:
        header = "step_idx,time_days,keff"
        if burnup_MWd_per_MtU is not None:
            header += ",burnup_MWd_per_MtU"
        header += ",peak_fast_flux_n_cm2_s,step_fast_fluence_n_cm2,cumulative_fast_fluence_n_cm2,operational"
        f.write(header + "\n")
        for i in range(n_sp):
            t_d = float(time_days[i])
            k   = float(keff_mean[i])
            bu  = float(burnup_MWd_per_MtU[i]) if burnup_MWd_per_MtU is not None else None
            pf  = float(beo_peak_flux[i]) if not np.isnan(beo_peak_flux[i]) else 0.0
            sf  = float(beo_step_flu[i])  if not np.isnan(beo_step_flu[i])  else 0.0
            cf  = float(beo_cum_flu[i])
            op  = 1 if i < beo_sd_idx else 0
            row = f"{i},{t_d:.4f},{k:.6f}"
            if bu is not None:
                row += f",{bu:.2f}"
            row += f",{pf:.6e},{sf:.6e},{cf:.6e},{op}"
            f.write(row + "\n")
    print(f"  Saved: {beo_csv}")


# ====================================================================================================
# MODULE ENTRY POINT
# ====================================================================================================

def run_BeO_depletion_postprocessing(run_dir, params):
    """
    Run BeO reflector depletion post-processing.

    Parameters
    ----------
    run_dir : str
        Directory containing depletion_results.h5 and openmc_simulation_n*.h5 statepoints.
    params : dict
        Simulation parameters (merged with run_params.json by the caller).

    Returns
    -------
    dict : BeO fluence data returned by extract_beo_peak_fluence, or None.
    """
    print(f"\n{'=' * 80}")
    print("BEO REFLECTOR DEPLETION POST-PROCESSING")
    print(f"{'=' * 80}")
    print(f"Run directory: {run_dir}")

    results_path = os.path.join(run_dir, "depletion_results.h5")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found!")
        return None

    results = openmc.deplete.Results(results_path)
    time_steps, keff_values = results.get_keff()

    if hasattr(keff_values[0], 'nominal_value'):
        keff_mean = np.array([k.nominal_value for k in keff_values])
    else:
        keff_values = np.array(keff_values)
        keff_mean   = keff_values[:, 0]

    time_days  = time_steps / 86400.0

    total_HM_mass_kg = params.get("total_HM_mass_kg", None)
    thermal_power_MW = params.get("thermal_power_MW", 10.0)

    if total_HM_mass_kg and total_HM_mass_kg > 0:
        burnup_MWd_per_MtU = thermal_power_MW * time_days / (total_HM_mass_kg / 1000.0)
    else:
        burnup_MWd_per_MtU = None

    x_data        = burnup_MWd_per_MtU if burnup_MWd_per_MtU is not None else time_days
    x_label       = "Burnup (MWd/MtU)"  if burnup_MWd_per_MtU is not None else "Time (days)"
    x_label_short = "burnup"            if burnup_MWd_per_MtU is not None else "time"

    output_dir = os.path.join(run_dir, "depletion_results")
    os.makedirs(output_dir, exist_ok=True)

    show_titles = params.get("show_titles", True)

    beo_data = extract_beo_peak_fluence(run_dir, time_steps, keff_mean, params)
    if beo_data is not None:
        plot_and_save_beo_results(
            beo_data, x_data, x_label, x_label_short,
            time_days, keff_mean, burnup_MWd_per_MtU,
            output_dir, show_titles=show_titles,
        )
    else:
        print("BeO fluence data not available (check use_BeO_tallies and use_BeO_reflector params).")

    return beo_data


# ====================================================================================================
# STANDALONE ENTRY POINT
# ====================================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python BeO_depletion_postprocessing.py <run_directory>")
        sys.exit(1)

    run_dir = sys.argv[1]

    params_path = os.path.join(run_dir, "run_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = json.load(f)
        print(f"Loaded parameters from run_params.json")
    else:
        print("WARNING: run_params.json not found, using defaults")
        params = {}

    # Load depletion results for time/keff data
    results_path = os.path.join(run_dir, "depletion_results.h5")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found!")
        sys.exit(1)

    results = openmc.deplete.Results(results_path)
    time_steps, keff_values = results.get_keff()

    if hasattr(keff_values[0], 'nominal_value'):
        keff_mean = np.array([k.nominal_value for k in keff_values])
    else:
        keff_values = np.array(keff_values)
        keff_mean   = keff_values[:, 0]

    time_days  = time_steps / 86400.0
    time_years = time_days  / 365.25

    total_HM_mass_kg = params.get("total_HM_mass_kg", None)
    thermal_power_MW = params.get("thermal_power_MW", 10.0)

    if total_HM_mass_kg and total_HM_mass_kg > 0:
        burnup_MWd_per_MtU = thermal_power_MW * time_days / (total_HM_mass_kg / 1000.0)
    else:
        burnup_MWd_per_MtU = None

    x_data        = burnup_MWd_per_MtU if burnup_MWd_per_MtU is not None else time_days
    x_label       = "Burnup (MWd/MtU)"  if burnup_MWd_per_MtU is not None else "Time (days)"
    x_label_short = "burnup"            if burnup_MWd_per_MtU is not None else "time"

    output_dir = os.path.join(run_dir, "depletion_results")
    os.makedirs(output_dir, exist_ok=True)

    show_titles = params.get("show_titles", True)

    beo_data = extract_beo_peak_fluence(run_dir, time_steps, keff_mean, params)
    if beo_data is not None:
        plot_and_save_beo_results(
            beo_data, x_data, x_label, x_label_short,
            time_days, keff_mean, burnup_MWd_per_MtU,
            output_dir, show_titles=show_titles,
        )
    else:
        print("BeO fluence data not available (check use_BeO_tallies and use_BeO_reflector params).")
