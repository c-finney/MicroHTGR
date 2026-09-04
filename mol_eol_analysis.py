"""
MOL/EOL Analysis Script

Performs Middle-of-Life (MOL) and End-of-Life (EOL) analysis on a completed
OpenMC depletion run without re-running the full depletion:

  1. Reactivity coefficient studies (FTC, MTC, ITC) at a specific burnup step.
  2. Heat map extraction at that burnup step, with control rods tuned to
     reach criticality via a binary search over insertion fraction.
  3. Leakage spectrum extraction at the same conditions.

The script reads depleted material compositions directly from depletion_results.h5,
injects them into a fresh model build (preserving geometry and rod positions),
and runs new eigenvalue calculations.

Usage
-----
  # As a module (from the depletion run directory):
  from mol_eol_analysis import run_mol_eol_analysis
  run_mol_eol_analysis(depletion_run_dir, step_idx=-1, step_label="EOL",
                       output_base_dir=output_dir)

  # Standalone:
  python mol_eol_analysis.py <depletion_run_dir> [step_idx] [step_label]
      step_idx:   integer step index in depletion_results.h5, or -1 for EOL (default)
      step_label: label string for output directories, e.g. "MOL" or "EOL" (default: "EOL")

Workflow
--------
  1. Load run_params.json from depletion run → recover all simulation params.
  2. Read depletion_results.h5 → extract atom counts per zone at step_idx.
  3. Reconstruct material compositions as {(ring_idx, bax_idx): {nuc: atoms/cm3}}.
  4. For each analysis type, call build_model() then inject depleted compositions
     before export_to_xml().
  5. Run eigenvalue OpenMC → post-process results.
"""

import os
import sys
import glob
import json
import copy
import shutil
import subprocess
import csv
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — script can be called from anywhere
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Single-channel thermal-hydraulics / Brayton cycle solver (nc_htgr.py).
# Vendored into ThermalHydraulics/ — see NOTICE.md for provenance.
_NC_HTGR_DIR = os.path.join(SCRIPT_DIR, "ThermalHydraulics")
if os.path.isdir(_NC_HTGR_DIR):
    sys.path.insert(0, _NC_HTGR_DIR)

# Heating-profile post-processing helpers
_POST_PROC_DIR = os.path.join(SCRIPT_DIR, "PostProcessingScripts")
if os.path.isdir(_POST_PROC_DIR):
    sys.path.insert(0, _POST_PROC_DIR)

import openmc
import openmc.deplete

# Cross-section library location comes from config.py, which reads it from the
# OPENMC_CROSS_SECTIONS environment variable (see README.md for setup).
import config as _cfg

cross_sections_path = _cfg.require_cross_sections()
os.environ['OPENMC_CROSS_SECTIONS'] = cross_sections_path
openmc.config['cross_sections'] = cross_sections_path


# ============================================================================
# STEP 1 — RECONSTRUCT DEPLETED MATERIAL COMPOSITIONS
# ============================================================================

def reconstruct_depleted_materials(depletion_run_dir, step_idx, h5_path=None):
    """
    Read depletion_results.h5 and return atom-density maps for all fuel zones.

    Parameters
    ----------
    depletion_run_dir : str
        Directory containing run_params.json (and optionally depletion_results.h5).
    step_idx : int
        Depletion step index.  -1 = final (EOL) step.
    h5_path : str, optional
        Explicit path to the depletion HDF5 file.  If omitted, defaults to
        ``depletion_run_dir/depletion_results.h5``.

    Returns
    -------
    depleted : dict
        Maps (ring_idx, bax_idx) -> {nuclide_str: atoms_per_cm3 (float)}
        Only nuclides with positive atom counts are included.
    fuel_mat_ids_2d : list[list[int]]
        Original 2D material-ID array [ring][axial_band] from run_params.json.
    params : dict
        Full run_params.json dictionary (used to rebuild the model).
    step_time_days : float
        Physical time (days) at the requested step.
    """

    params_path = os.path.join(depletion_run_dir, "run_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"run_params.json not found in {depletion_run_dir}")
    with open(params_path) as f:
        run_params = json.load(f)

    if h5_path is None:
        h5_path = os.path.join(depletion_run_dir, "depletion_results.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Depletion HDF5 not found: {h5_path}")

    results = openmc.deplete.Results(h5_path)
    n_steps = len(results)

    if step_idx < 0:
        step_idx = n_steps + step_idx      # -1 → last step
    step_idx = max(0, min(step_idx, n_steps - 1))

    # Time at this step
    times, _ = results.get_keff()
    step_time_days = float(times[step_idx]) / 86400.0

    fuel_mat_ids_2d = run_params.get("fuel_mat_ids", None)
    fuel_mat_volumes = {str(k): v
                        for k, v in run_params.get("fuel_mat_volumes", {}).items()}

    if fuel_mat_ids_2d is None:
        raise ValueError("run_params.json has no 'fuel_mat_ids' — was use_spatial_burnup=True?")

    print(f"\n  Depletion results: {n_steps} steps, reading step {step_idx} "
          f"({step_time_days:.1f} days)")
    print(f"  Fuel material grid: {len(fuel_mat_ids_2d)} rings × "
          f"{len(fuel_mat_ids_2d[0])} axial bands")

    # Collect all unique material IDs tracked in the depletion results
    try:
        tracked_mat_ids = set(results[0].index_mat.keys())
    except Exception:
        tracked_mat_ids = set()

    # Build set of nuclides available in the cross-sections library so we
    # can skip depletion-chain products (e.g. Ag109_m1) that have no
    # transport data and would crash OpenMC at run time.
    try:
        xs_library = openmc.data.DataLibrary.from_xml()
        xs_nuclides = {
            m for entry in xs_library.libraries
            if entry['type'] == 'neutron'
            for m in entry['materials']
        }
    except Exception:
        xs_nuclides = None   # fallback: no filtering

    depleted = {}   # (ring_idx, bax_idx) -> {nuc: atoms/cm3}

    # Build the global nuclide list once (index_nuc is keyed by nuclide name,
    # not by material ID).
    try:
        all_nuclides = list(results[0].index_nuc.keys())
    except Exception:
        all_nuclides = []

    n_skipped_xs = 0
    for ring_idx, row in enumerate(fuel_mat_ids_2d):
        for bax_idx, mat_id in enumerate(row):
            mat_id_str = str(mat_id)
            if mat_id_str not in tracked_mat_ids:
                continue

            vol_cm3 = float(fuel_mat_volumes.get(mat_id_str, 0.0))
            if vol_cm3 <= 0:
                print(f"  WARNING: zero volume for material {mat_id_str}, skipping")
                continue

            nuc_densities = {}
            for nuc in all_nuclides:
                # Skip nuclides absent from the transport library
                if xs_nuclides is not None and nuc not in xs_nuclides:
                    n_skipped_xs += 1
                    continue
                try:
                    _, atoms = results.get_atoms(mat_id_str, nuc)
                    n_at = float(atoms[step_idx])
                    if n_at > 0:
                        nuc_densities[nuc] = n_at / vol_cm3   # atoms/cm3
                except Exception:
                    pass

            if nuc_densities:
                depleted[(ring_idx, bax_idx)] = nuc_densities

    if n_skipped_xs:
        print(f"  Skipped {n_skipped_xs} nuclide-zone entries absent from XS library")

    print(f"  Reconstructed {len(depleted)} / "
          f"{len(fuel_mat_ids_2d) * len(fuel_mat_ids_2d[0])} fuel zones")

    # ── Burnable poison (B4C) — spatial burnup (2D grid) ──────────────────
    # Each ring × burnup-band has its own depletable poison clone.
    # We reconstruct per-zone densities using the same pattern as fuel,
    # storing them under ('poison', ring_idx, bax_idx) keys.
    poison_mat_ids_2d  = run_params.get("poison_mat_ids", None)
    poison_mat_volumes = {str(k): v
                          for k, v in run_params.get("poison_mat_volumes", {}).items()}

    if poison_mat_ids_2d and poison_mat_volumes:
        n_poison_reconstructed = 0
        for p_ring_idx, p_row in enumerate(poison_mat_ids_2d):
            for p_bax_idx, p_mat_id in enumerate(p_row):
                p_mat_id_str = str(p_mat_id)
                if p_mat_id_str not in tracked_mat_ids:
                    continue
                p_vol_cm3 = float(poison_mat_volumes.get(p_mat_id_str, 0.0))
                if p_vol_cm3 <= 0:
                    continue
                p_densities = {}
                for nuc in all_nuclides:
                    if xs_nuclides is not None and nuc not in xs_nuclides:
                        continue
                    try:
                        _, atoms = results.get_atoms(p_mat_id_str, nuc)
                        n_at = float(atoms[step_idx])
                        if n_at > 0:
                            p_densities[nuc] = n_at / p_vol_cm3
                    except Exception:
                        pass
                if p_densities:
                    depleted[('poison', p_ring_idx, p_bax_idx)] = p_densities
                    n_poison_reconstructed += 1
        print(f"  Reconstructed {n_poison_reconstructed} / "
              f"{sum(len(r) for r in poison_mat_ids_2d)} poison zones "
              f"({len(poison_mat_ids_2d)} rings × {len(poison_mat_ids_2d[0])} axial bands)")
    else:
        print(f"  WARNING: poison_mat_ids / poison_mat_volumes not found in run_params.json "
              f"— burnable poison compositions NOT reconstructed")

    # ── Depletable graphite ────────────────────────────────────────────────
    graphite_mat_id_str = str(run_params.get("graphite_material_id", ""))
    graphite_vol_cm3    = float(run_params.get("graphite_volume_simulated_cm3", 0.0))

    if graphite_mat_id_str and graphite_mat_id_str in tracked_mat_ids and graphite_vol_cm3 > 0:
        graphite_densities = {}
        for nuc in all_nuclides:
            if xs_nuclides is not None and nuc not in xs_nuclides:
                continue
            try:
                _, atoms = results.get_atoms(graphite_mat_id_str, nuc)
                n_at = float(atoms[step_idx])
                if n_at > 0:
                    graphite_densities[nuc] = n_at / graphite_vol_cm3
            except Exception:
                pass
        if graphite_densities:
            depleted['graphite'] = graphite_densities
            print(f"  Reconstructed depletable graphite (mat {graphite_mat_id_str}): "
                  f"{len(graphite_densities)} nuclides")

    return depleted, fuel_mat_ids_2d, run_params, step_time_days


# ============================================================================
# STEP 2 — INJECT DEPLETED COMPOSITIONS INTO A FRESHLY BUILT MODEL
# ============================================================================

def _inject_depleted_materials(fuel_clones, depleted, model=None, poison_clones=None):
    """
    Overwrite fuel-clone nuclide compositions with depleted atom densities.
    Also injects depleted burnable-poison (B4C) compositions if present.

    Called AFTER build_model() returns fuel_clones but BEFORE
    model.export_to_xml() is called, so the modified compositions are
    written to the XML files used by the OpenMC run.

    Parameters
    ----------
    fuel_clones : list[list[openmc.Material]]
        From build_model() — [ring_idx][bax_idx].
    depleted : dict
        From reconstruct_depleted_materials() — (ring, bax) -> {nuc: dens},
        plus ('poison', ring, bax) keys for per-zone burnable poison.
    model : openmc.model.Model, optional
        The freshly-built model (unused for poison when poison_clones provided).
    poison_clones : list[list[openmc.Material]], optional
        From build_model() — [ring_idx][bax_idx] poison materials.
        When provided, per-zone poison compositions are injected directly.
    """
    def _overwrite_mat(mat, nuc_densities):
        total_density = sum(nuc_densities.values())   # atoms/cm3
        mat._nuclides = []
        if hasattr(mat, '_elements'):
            mat._elements = []
        for nuc, n_dens in nuc_densities.items():
            mat.add_nuclide(nuc, n_dens / total_density, 'ao')
        mat.set_density('atom/cm3', total_density)

    injected = 0
    for ring_idx, ring_fuels in enumerate(fuel_clones):
        for bax_idx, mat in enumerate(ring_fuels):
            # With non-spatial burnup all rings share bax_idx=0
            effective_bax = bax_idx if len(ring_fuels) > 1 else 0
            key = (ring_idx, effective_bax)
            if key not in depleted:
                continue
            nuc_densities = depleted[key]
            if not nuc_densities:
                continue
            _overwrite_mat(mat, nuc_densities)
            injected += 1

    print(f"  Injected depleted compositions into {injected} fuel material clones")

    # ── Burnable poison — per-zone injection ───────────────────────────────
    if poison_clones is not None:
        poison_injected = 0
        for p_ring_idx, p_ring in enumerate(poison_clones):
            for p_bax_idx, p_mat in enumerate(p_ring):
                key = ('poison', p_ring_idx, p_bax_idx)
                if key not in depleted:
                    continue
                nuc_densities = depleted[key]
                if not nuc_densities:
                    continue
                _overwrite_mat(p_mat, nuc_densities)
                poison_injected += 1
        print(f"  Injected depleted compositions into {poison_injected} poison material clones")

    # ── Depletable graphite ────────────────────────────────────────────────
    if 'graphite' in depleted and model is not None:
        graphite_mat = next(
            (m for m in model.materials if m.name == "Graphite"), None
        )
        if graphite_mat is not None:
            _overwrite_mat(graphite_mat, depleted['graphite'])
            print(f"  Injected depleted composition into Graphite material")
        else:
            print(f"  WARNING: 'graphite' key in depleted but no Graphite "
                  f"material found in model — graphite NOT injected")


def _assert_new_dir(run_dir, depletion_run_dir=None):
    """
    Abort if run_dir would overwrite an existing simulation directory.

    Specifically prevents any MOL/EOL run from writing into the original
    depletion run directory, which would corrupt materials.xml / geometry.xml.
    """
    run_real = os.path.realpath(run_dir)

    if depletion_run_dir is not None:
        dep_real = os.path.realpath(depletion_run_dir)
        if run_real == dep_real:
            raise RuntimeError(
                f"MOL/EOL run_dir equals the depletion_run_dir!\n"
                f"  run_dir          = {run_dir}\n"
                f"  depletion_run_dir = {depletion_run_dir}\n"
                "Refusing to overwrite original run files."
            )
        # Also block writing *inside* the depletion run dir unless it's a
        # recognised analysis subfolder (mol_eol_analysis_* or th_coupler*
        # or critical_search*).
        _ALLOWED_PREFIXES = ("mol_eol_analysis", "th_coupler", "critical_search", "cs_th")
        if run_real.startswith(dep_real + os.sep):
            tail = os.path.relpath(run_real, dep_real)
            if not any(tail.startswith(pfx) for pfx in _ALLOWED_PREFIXES):
                raise RuntimeError(
                    f"run_dir is inside the depletion_run_dir but not in a "
                    f"recognised analysis subfolder "
                    f"({', '.join(_ALLOWED_PREFIXES)}).\n"
                    f"  run_dir = {run_dir}\n"
                    "Please use a separate output_base_dir."
                )

def _run_eigenvalue_with_depleted(params, depleted, run_dir, depletion_run_dir=None):
    """
    Build model, inject depleted materials, export XML, and run OpenMC.

    The run_dir must be a fresh directory separate from depletion_run_dir
    to avoid overwriting the original simulation files.

    Returns
    -------
    n_trisos : int
    fuel_clones : list[list[openmc.Material]]
    """
    from main_simulation import build_model

    _assert_new_dir(run_dir, depletion_run_dir)
    os.makedirs(run_dir, exist_ok=True)
    model, n_trisos, m_colors, fuel_clones, poison_clones = build_model(params, run_dir)

    print(f"\n  Injecting depleted compositions...")
    _inject_depleted_materials(fuel_clones, depleted, model=model, poison_clones=poison_clones)

    model.export_to_xml()

    shutil.copy2(cross_sections_path, os.path.join(run_dir, 'cross_sections.xml'))

    openmc_output_file = os.path.join(run_dir, 'openmc_output.txt')
    with open(openmc_output_file, 'w', buffering=1) as outf:
        process = subprocess.Popen(
            ['openmc'],
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env={**os.environ, 'OMP_NUM_THREADS': '128'}
        )
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
            outf.write(line)
            outf.flush()
        rc = process.wait()

    if rc != 0:
        raise RuntimeError(f"OpenMC failed (return code {rc}) in {run_dir}")

    return n_trisos, fuel_clones


def _read_keff(run_dir):
    """Read the Combined k-effective and its std_dev from the statepoint in run_dir.

    sp.keff reads the 'k_combined' dataset from the HDF5 statepoint, which is
    the combined (analog + track-length + collision) estimator — the same value
    reported as 'Combined k-effective' in OpenMC's terminal output.
    """
    import glob
    sp_files = sorted(glob.glob(os.path.join(run_dir, "statepoint.*.h5")))
    if not sp_files:
        raise FileNotFoundError(f"No statepoint file in {run_dir}")
    sp = openmc.StatePoint(sp_files[-1])
    # sp.keff == sp.k_combined (the 'k_combined' HDF5 dataset)
    return float(sp.keff.n), float(sp.keff.s)


# ============================================================================
# STEP 3 — CRITICAL ROD INSERTION BINARY SEARCH
# ============================================================================

# ---------------------------------------------------------------------------
# CSV writer — one row per trial eigenvalue solve
# ---------------------------------------------------------------------------
 
def _write_search_csv(csv_path, rows):
    """
    Write / overwrite the search summary CSV.
 
    rows : list of dicts with keys:
        stage, iteration, bank_1, bank_2, keff, keff_std, delta_pcm
    """
    fieldnames = ["stage", "iteration", "bank_1", "bank_2",
                  "keff", "keff_std", "delta_pcm"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
 
 
# ---------------------------------------------------------------------------
# Interpolation search (regula falsi + Illinois anti-stagnation)
# ---------------------------------------------------------------------------
#
# k_eff is MONOTONICALLY DECREASING with insertion fraction:
#   insertion = 0  ->  rods out  ->  highest k_eff
#   insertion = 1  ->  rods in   ->  lowest  k_eff
#
# We maintain a bracket [lo, hi] where k(lo) > k_target > k(hi).
# The Illinois variant prevents one-sided stagnation by halving the
# "old" endpoint's k value when the new point falls on the same side
# twice in a row.
#
# Fallback to bisection if the interpolated point is outside the bracket
# (can happen with noisy Monte Carlo k_eff estimates).
 
def _interpolation_next(lo, k_lo, hi, k_hi, k_target,
                        last_side, same_side_count):
    """
    Compute the next trial insertion using regula falsi + Illinois.
 
    Parameters
    ----------
    lo, hi           : float  -- current bracket endpoints (insertion fractions)
    k_lo, k_hi       : float  -- k_eff at lo and hi respectively
    k_target         : float  -- target k_eff (typically 1.0)
    last_side        : str or None -- "lo" or "hi", which side the last new
                       point landed on
    same_side_count  : int    -- how many consecutive times the new point has
                       been on the same side
 
    Returns
    -------
    mid : float  -- next trial insertion fraction
    """
    dk = k_lo - k_hi
    if abs(dk) < 1e-9:
        # Degenerate bracket -- fall back to bisection
        return 0.5 * (lo + hi)
 
    # Illinois: if the new point has been on the same side twice in a row,
    # halve the "stale" endpoint's effective k to force the interpolation
    # to move further into the bracket.
    k_lo_eff = k_lo
    k_hi_eff = k_hi
    if same_side_count >= 2:
        if last_side == "hi":
            k_lo_eff = k_target + 0.5 * (k_lo - k_target)
        else:
            k_hi_eff = k_target + 0.5 * (k_hi - k_target)
 
    mid = lo + (k_target - k_lo_eff) / (k_hi_eff - k_lo_eff) * (hi - lo)
 
    # Safety clamp -- noisy MC k_eff can push interpolation outside bracket
    margin = 0.02 * (hi - lo)
    mid = max(lo + margin, min(hi - margin, mid))
 
    return mid
 
 
def find_critical_rod_insertion(
    params,
    depleted,
    output_dir,
    depletion_run_dir=None,
    k_target=1.0,
    k_tol=0.003,
    max_iter=20,
    prev_result=None,
):
    """
    Two-stage interpolation search to find the critical rod insertion fraction.
 
    Stage 0 -- quick check: run with bank 1 fully inserted, bank 2 at 0.
    Stage 1 -- interpolation search:
      * If stage-0 k < k_target: search bank 1 in [0, 1] with bank 2 = 0.
      * If stage-0 k > k_target: fix bank 1 = 1.0, search bank 2 in [0, 1].
    Bank 3 is always left at 0.
 
    Trial run subdirectories are left intact inside output_dir.
    Cleanup (shutil.rmtree of the entire output_dir) is the caller's
    responsibility so that the CSV can be copied out first.
 
    Parameters
    ----------
    params : dict
        Simulation parameters (deep-copied; original unchanged).
    depleted : dict
        Depleted material compositions from reconstruct_depleted_materials().
        Pass {} for BOL (no injection).
    output_dir : str
        Root directory for the search.  Sub-directories for trial runs are
        created here.  The CSV is written here as critical_search_summary.csv.
    depletion_run_dir : str or None
        Passed to _assert_new_dir to prevent overwriting the depletion directory.
    k_target : float
        Target k_eff (default 1.0).
    k_tol : float
        Convergence tolerance on |k_eff - k_target|.
    max_iter : int
        Maximum interpolation-search iterations per stage.
    prev_result : dict or None
        Result dict returned by a previous call to this function (e.g. from the
        preceding depletion timestep).  When provided, the search uses warm-start
        brackets derived from the previous critical insertion, reducing the number
        of trial runs needed:

        * Previous stage was "bank1" (bank 2 remained at 0):
          Stage 0 is skipped.  Two bracketing runs are made at bank_1 = 0.0
          and bank_1 = prev_result["critical_bank_1"].  The search then
          proceeds in [0, prev_b1] instead of [0, 1].

        * Previous stage was "bank2" (bank 1 stayed at 1.0):
          Stage 0 (bank_1=1, bank_2=0) is still run.  The upper bracket for
          bank 2 is set to prev_result["critical_bank_2"] instead of 1.0.

        In both cases, if the warm hint does not provide a valid bracket (the
        core reactivity increased unexpectedly), the algorithm falls back to the
        normal full-range bracket automatically.
 
    Returns
    -------
    dict with keys:
        critical_bank_1    : float
        critical_bank_2    : float
        critical_insertion : float   -- insertion of the active search bank
        search_stage       : str     -- "bank1" or "bank2"
        critical_keff      : float
        critical_keff_std  : float
        critical_run_dir   : str     -- directory of the converged run
        converged          : bool
        n_iterations       : int
        csv_path           : str     -- path to the summary CSV inside output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "critical_search_summary.csv")
    csv_rows = []   # accumulated across both stages
 
    print(f"\n{'─' * 70}")
    print(f"  CRITICAL ROD SEARCH  (interpolation + Illinois fallback)")
    print(f"  Target: k_eff = {k_target:.4f}  +/-  {k_tol:.4f}")
    print(f"  Output: {output_dir}")
    print(f"{'─' * 70}")
 
    # Reduced-particle settings for all search iterations
    search_params = copy.deepcopy(params)
    search_params["total_batches"]       = params.get("critical_search_batches",  50)
    search_params["inactive_batches"]    = params.get("critical_search_inactive",  25)
    search_params["particles"]           = params.get(
        "critical_search_particles",
        max(50_000, params.get("particles", 100_000) // 2)
    )
    search_params["make_geometry_plots"] = False
    search_params["use_mesh_tallies"]    = False
    search_params["use_BeO_tallies"]     = False
    search_params["use_leakage_tallies"] = False
    search_params["use_global_tallies"]  = False
 
    def _run_trial(stage_label, iteration, b1, b2, trial_dir):
        """Build, run, read k_eff.  Returns (keff, keff_std).
        Trial directory is left on disk -- caller cleans up the whole tree."""
        p = copy.deepcopy(search_params)
        p["bank_1_insertion"] = b1
        p["bank_2_insertion"] = b2
        p["bank_3_insertion"] = 0.0
 
        _run_eigenvalue_with_depleted(p, depleted, trial_dir,
                                      depletion_run_dir=depletion_run_dir)
        k, k_std = _read_keff(trial_dir)
 
        csv_rows.append({
            "stage":     stage_label,
            "iteration": iteration,
            "bank_1":    round(b1,    6),
            "bank_2":    round(b2,    6),
            "keff":      round(k,     6),
            "keff_std":  round(k_std, 6),
            "delta_pcm": round((k - k_target) * 1e5, 1),
        })
        # Overwrite CSV after every trial so a partial run is still readable
        _write_search_csv(csv_path, csv_rows)
 
        return k, k_std
 
    # Extract warm-start hints from a previous critical search result (if any).
    # As fuel depletes (loses reactivity), the critical insertion fraction
    # decreases — meaning the previous step's critical insertion is a valid
    # upper bracket for the current search, tightening the initial interval.
    _prev_stage   = prev_result.get("search_stage") if prev_result else None
    _prev_b1_hint = prev_result.get("critical_bank_1") if prev_result else None
    _prev_b2_hint = prev_result.get("critical_bank_2") if prev_result else None

    # ── Bank-1 warm-start path (stage 0 skipped) ───────────────────────────────────
    # When the previous step converged in the bank-1 stage (bank 2 stayed at
    # 0), the critical bank-1 insertion is expected to be <= prev_b1_hint at
    # the current (more depleted) step.  Use prev_b1_hint directly as the
    # upper bracket and skip the expensive stage-0 run at bank_1=1.0.
    if (_prev_stage == "bank1"
            and _prev_b1_hint is not None
            and 0.0 < _prev_b1_hint <= 1.0):

        active_bank  = "bank_1"
        fixed_b2     = 0.0
        search_stage = "bank1"
        print(f"\n  Warm-start (bank1 stage): skipping stage 0; "
              f"previous critical bank_1 = {_prev_b1_hint:.4f}")

        # Iter 0: fully withdrawn — establishes the high-k (lo) bracket end
        lo_dir = os.path.join(output_dir, "stage1_iter00_b1_0.0000")
        print(f"\n  Iter  0: bank_1 = 0.0000  [bracketing rods-out end]")
        k_at_0, k_at_0_std = _run_trial(search_stage, 0,
                                         b1=0.0, b2=fixed_b2,
                                         trial_dir=lo_dir)
        print(f"         k_eff = {k_at_0:.5f} +/- {k_at_0_std:.5f}   "
              f"(delta = {(k_at_0 - k_target)*1e5:+.0f} pcm)")

        if k_at_0 < k_target:
            print(f"\n  WARNING: k_eff < k_target with all rods withdrawn. "
                  f"Core is subcritical. Returning rods-out result.")
            print(f"  CSV summary -> {csv_path}")
            return {
                "critical_bank_1":    0.0,
                "critical_bank_2":    0.0,
                "critical_insertion": 0.0,
                "search_stage":       search_stage,
                "critical_keff":      k_at_0,
                "critical_keff_std":  k_at_0_std,
                "critical_run_dir":   lo_dir,
                "converged":          False,
                "n_iterations":       1,
                "csv_path":           csv_path,
            }

        # Iter 1: previous critical insertion — should give k < k_target
        hint_dir = os.path.join(output_dir,
                                f"stage1_iter01_b1_{_prev_b1_hint:.4f}")
        print(f"\n  Iter  1: bank_1 = {_prev_b1_hint:.4f}  "
              f"[warm-start upper bracket]")
        k_at_hint, k_at_hint_std = _run_trial(search_stage, 1,
                                               b1=_prev_b1_hint, b2=fixed_b2,
                                               trial_dir=hint_dir)
        print(f"         k_eff = {k_at_hint:.5f} +/- {k_at_hint_std:.5f}   "
              f"(delta = {(k_at_hint - k_target)*1e5:+.0f} pcm)")

        if abs(k_at_hint - k_target) < k_tol:
            print(f"\n  Converged at warm-start bracket: bank_1 = "
                  f"{_prev_b1_hint:.4f}, k = {k_at_hint:.5f} +/- {k_at_hint_std:.5f}")
            print(f"  CSV summary -> {csv_path}")
            return {
                "critical_bank_1":    _prev_b1_hint,
                "critical_bank_2":    0.0,
                "critical_insertion": _prev_b1_hint,
                "search_stage":       search_stage,
                "critical_keff":      k_at_hint,
                "critical_keff_std":  k_at_hint_std,
                "critical_run_dir":   hint_dir,
                "converged":          True,
                "n_iterations":       2,
                "csv_path":           csv_path,
            }

        if k_at_hint > k_target:
            # Warm hint did not bracket (core still supercritical at prev_b1).
            # Extend to bank_1=1.0 for a valid upper bracket.
            print(f"\n  Warm hint gave k > k_target; "
                  f"extending upper bracket to bank_1 = 1.0")
            ext_dir = os.path.join(output_dir, "stage1_iter02_b1_1.0000")
            print(f"\n  Iter  2: bank_1 = 1.0000  [extended upper bracket]")
            k_at_1, k_at_1_std = _run_trial(search_stage, 2,
                                             b1=1.0, b2=fixed_b2,
                                             trial_dir=ext_dir)
            print(f"         k_eff = {k_at_1:.5f} +/- {k_at_1_std:.5f}   "
                  f"(delta = {(k_at_1 - k_target)*1e5:+.0f} pcm)")
            lo, k_lo   = 0.0, k_at_0
            hi, k_hi   = 1.0, k_at_1
            iter_offset = 3
        else:
            # Warm hint is a valid upper bracket: search in [0.0, prev_b1]
            lo, k_lo   = 0.0, k_at_0
            hi, k_hi   = _prev_b1_hint, k_at_hint
            iter_offset = 2

    else:
        # ── Stage 0: bank 1 fully inserted, bank 2 out ─────────────────────────────
        s0_dir = os.path.join(output_dir, "stage0")
        print(f"\n  Stage 0: bank 1 = 1.0, bank 2 = 0.0")
        k0, k0_std = _run_trial("stage0", 0, b1=1.0, b2=0.0, trial_dir=s0_dir)
        print(f"  k_eff = {k0:.5f} +/- {k0_std:.5f}   "
              f"(delta = {(k0 - k_target)*1e5:+.0f} pcm)")

        if abs(k0 - k_target) < k_tol:
            print(f"\n  Converged at stage 0: bank 1 = 1.0, bank 2 = 0.0")
            print(f"  CSV summary -> {csv_path}")
            return {
                "critical_bank_1":    1.0,
                "critical_bank_2":    0.0,
                "critical_insertion": 1.0,
                "search_stage":       "bank1",
                "critical_keff":      k0,
                "critical_keff_std":  k0_std,
                "critical_run_dir":   s0_dir,
                "converged":          True,
                "n_iterations":       1,
                "csv_path":           csv_path,
            }

        # ── Stage 1: choose which bank to search ────────────────────────────────────
        if k0 < k_target:
            active_bank  = "bank_1"
            fixed_b2     = 0.0
            search_stage = "bank1"
            print(f"\n  k < k_target with bank 1 full -> interpolation search on "
                  f"bank 1  (bank 2 = 0.0)")
            # Need k at insertion=0 to open the high-k end of the bracket
            lo_dir = os.path.join(output_dir, "stage1_iter00_b1_0.0000")
            print(f"\n  Iter  0: bank_1 = 0.0000  [bracketing rods-out end]")
            k_at_0, k_at_0_std = _run_trial(search_stage, 0,
                                             b1=0.0, b2=fixed_b2,
                                             trial_dir=lo_dir)
            print(f"         k_eff = {k_at_0:.5f} +/- {k_at_0_std:.5f}   "
                  f"(delta = {(k_at_0 - k_target)*1e5:+.0f} pcm)")

            if k_at_0 < k_target:
                print(f"\n  WARNING: k_eff < k_target with all rods withdrawn. "
                      f"Core is subcritical. Returning rods-out result.")
                print(f"  CSV summary -> {csv_path}")
                return {
                    "critical_bank_1":    0.0,
                    "critical_bank_2":    0.0,
                    "critical_insertion": 0.0,
                    "search_stage":       search_stage,
                    "critical_keff":      k_at_0,
                    "critical_keff_std":  k_at_0_std,
                    "critical_run_dir":   lo_dir,
                    "converged":          False,
                    "n_iterations":       2,
                    "csv_path":           csv_path,
                }

            lo, k_lo   = 0.0, k_at_0   # high k side (rods out)
            hi, k_hi   = 1.0, k0       # low  k side (rods in)
            iter_offset = 1

        else:
            active_bank  = "bank_2"
            fixed_b1     = 1.0
            search_stage = "bank2"
            print(f"\n  k > k_target with bank 1 full -> interpolation search on "
                  f"bank 2  (bank 1 = 1.0)")

            # Determine upper bracket for bank 2: use warm-start hint if available
            if (_prev_stage == "bank2"
                    and _prev_b2_hint is not None
                    and 0.0 < _prev_b2_hint <= 1.0):
                # Warm-start: previous critical bank_2 should give k < k_target
                # (core less reactive now), so use it as the upper bracket.
                hint_dir = os.path.join(output_dir,
                                        f"stage1_iter00_b2_{_prev_b2_hint:.4f}")
                print(f"\n  Warm-start (bank2 stage): using previous critical "
                      f"bank_2 = {_prev_b2_hint:.4f} as upper bracket")
                print(f"\n  Iter  0: bank_2 = {_prev_b2_hint:.4f}  "
                      f"[warm-start upper bracket]")
                k_at_hint, k_at_hint_std = _run_trial(search_stage, 0,
                                                       b1=fixed_b1,
                                                       b2=_prev_b2_hint,
                                                       trial_dir=hint_dir)
                print(f"         k_eff = {k_at_hint:.5f} +/- {k_at_hint_std:.5f}   "
                      f"(delta = {(k_at_hint - k_target)*1e5:+.0f} pcm)")

                if abs(k_at_hint - k_target) < k_tol:
                    print(f"\n  Converged at warm-start bracket: bank_2 = "
                          f"{_prev_b2_hint:.4f}, "
                          f"k = {k_at_hint:.5f} +/- {k_at_hint_std:.5f}")
                    print(f"  CSV summary -> {csv_path}")
                    return {
                        "critical_bank_1":    1.0,
                        "critical_bank_2":    _prev_b2_hint,
                        "critical_insertion": _prev_b2_hint,
                        "search_stage":       search_stage,
                        "critical_keff":      k_at_hint,
                        "critical_keff_std":  k_at_hint_std,
                        "critical_run_dir":   hint_dir,
                        "converged":          True,
                        "n_iterations":       2,
                        "csv_path":           csv_path,
                    }

                if k_at_hint > k_target:
                    # Hint didn't bracket — extend to full bank_2 insertion.
                    # Use the hint point as the lo (supercritical) bracket end
                    # so the search stays in [prev_b2, 1.0] rather than [0, 1.0].
                    print(f"\n  Warm hint gave k > k_target; "
                          f"extending bracket to bank_2 = 1.0")
                    ext_dir = os.path.join(output_dir, "stage1_iter01_b2_1.0000")
                    print(f"\n  Iter  1: bank_2 = 1.0000  [extended upper bracket]")
                    k_at_1, k_at_1_std = _run_trial(search_stage, 1,
                                                     b1=fixed_b1, b2=1.0,
                                                     trial_dir=ext_dir)
                    print(f"         k_eff = {k_at_1:.5f} +/- {k_at_1_std:.5f}   "
                          f"(delta = {(k_at_1 - k_target)*1e5:+.0f} pcm)")

                    if k_at_1 > k_target:
                        print(f"\n  WARNING: k_eff > k_target with all rods inserted. "
                              f"Cannot suppress to k_target. Returning all-rods-in result.")
                        print(f"  CSV summary -> {csv_path}")
                        return {
                            "critical_bank_1":    1.0,
                            "critical_bank_2":    1.0,
                            "critical_insertion": 1.0,
                            "search_stage":       search_stage,
                            "critical_keff":      k_at_1,
                            "critical_keff_std":  k_at_1_std,
                            "critical_run_dir":   ext_dir,
                            "converged":          False,
                            "n_iterations":       3,
                            "csv_path":           csv_path,
                        }

                    # Tight bracket: [prev_b2_hint, 1.0] — avoids re-exploring [0, prev_b2]
                    lo, k_lo   = _prev_b2_hint, k_at_hint
                    hi, k_hi   = 1.0, k_at_1
                    iter_offset = 2
                else:
                    # Warm hint is a valid upper bracket: search in [0.0, prev_b2]
                    lo, k_lo   = 0.0, k0
                    hi, k_hi   = _prev_b2_hint, k_at_hint
                    iter_offset = 1

            else:
                # Normal: need k at bank_2=1 to open the low-k end of the bracket
                hi_dir = os.path.join(output_dir, "stage1_iter00_b2_1.0000")
                print(f"\n  Iter  0: bank_2 = 1.0000  [bracketing all-rods-in end]")
                k_at_1, k_at_1_std = _run_trial(search_stage, 0,
                                                  b1=fixed_b1, b2=1.0,
                                                  trial_dir=hi_dir)
                print(f"         k_eff = {k_at_1:.5f} +/- {k_at_1_std:.5f}   "
                      f"(delta = {(k_at_1 - k_target)*1e5:+.0f} pcm)")

                if k_at_1 > k_target:
                    print(f"\n  WARNING: k_eff > k_target with all rods inserted. "
                          f"Cannot suppress to k_target. Returning all-rods-in result.")
                    print(f"  CSV summary -> {csv_path}")
                    return {
                        "critical_bank_1":    1.0,
                        "critical_bank_2":    1.0,
                        "critical_insertion": 1.0,
                        "search_stage":       search_stage,
                        "critical_keff":      k_at_1,
                        "critical_keff_std":  k_at_1_std,
                        "critical_run_dir":   hi_dir,
                        "converged":          False,
                        "n_iterations":       2,
                        "csv_path":           csv_path,
                    }

                lo, k_lo   = 0.0, k0      # high k side (bank_2 out)
                hi, k_hi   = 1.0, k_at_1  # low  k side (bank_2 in)
                iter_offset = 1
 
    # ── Interpolation search loop ───────────────────────────────────────────
    best_dir        = None
    best_ins        = None
    best_k          = None
    best_std        = None
    converged       = False
    last_side       = None
    same_side_count = 0
 
    for i in range(max_iter):
        iter_num = i + iter_offset
 
        mid = _interpolation_next(lo, k_lo, hi, k_hi, k_target,
                                  last_side, same_side_count)
 
        if active_bank == "bank_1":
            b1_trial, b2_trial = mid, fixed_b2
            label_suffix = f"b1_{mid:.4f}"
        else:
            b1_trial, b2_trial = fixed_b1, mid
            label_suffix = f"b2_{mid:.4f}"
 
        trial_dir = os.path.join(output_dir,
                                 f"stage1_iter{iter_num:02d}_{label_suffix}")
 
        print(f"\n  Iter {iter_num:2d}: {active_bank} = {mid:.4f}  "
              f"[{lo:.4f}, {hi:.4f}]")
        k, k_std = _run_trial(search_stage, iter_num,
                              b1=b1_trial, b2=b2_trial,
                              trial_dir=trial_dir)
        print(f"         k_eff = {k:.5f} +/- {k_std:.5f}   "
              f"(delta = {(k - k_target)*1e5:+.0f} pcm)")
 
        if best_k is None or abs(k - k_target) < abs(best_k - k_target):
            best_k   = k
            best_std = k_std
            best_ins = mid
            best_dir = trial_dir
 
        if abs(k - k_target) < k_tol:
            converged = True
            print(f"\n  Converged: {active_bank} = {mid:.4f}, "
                  f"k = {k:.5f} +/- {k_std:.5f}")
            break
 
        # k decreases with insertion:
        #   k > k_target -> need more absorption -> raise lo
        #   k < k_target -> need less absorption -> lower hi
        if k > k_target:
            new_side = "lo"
            same_side_count = same_side_count + 1 if last_side == "lo" else 1
            lo, k_lo = mid, k
        else:
            new_side = "hi"
            same_side_count = same_side_count + 1 if last_side == "hi" else 1
            hi, k_hi = mid, k
 
        last_side = new_side
 
    if not converged:
        print(f"\n  WARNING: Did not converge in {max_iter} iterations.")
        print(f"  Best: {active_bank} = {best_ins:.4f}, k = {best_k:.5f}")
 
    if search_stage == "bank1":
        final_b1, final_b2 = best_ins, 0.0
    else:
        final_b1, final_b2 = 1.0, best_ins
 
    print(f"\n  CSV summary -> {csv_path}")
 
    return {
        "critical_bank_1":    final_b1,
        "critical_bank_2":    final_b2,
        "critical_insertion": best_ins,
        "search_stage":       search_stage,
        "critical_keff":      best_k,
        "critical_keff_std":  best_std,
        "critical_run_dir":   best_dir,
        "converged":          converged,
        "n_iterations":       i + 1 + iter_offset,
        "csv_path":           csv_path,
    }

# ============================================================================
# STEP 3b — T/H COUPLER
# ============================================================================

def _extract_heating_csv_from_statepoint(run_dir, params):
    """
    Extract axial heating profile from the statepoint in run_dir and write a
    neutronics CSV compatible with nc_htgr's NeutronicsTable.

    Returns
    -------
    csv_path : str  — path to the written neutronics CSV, or None on failure
    q_avg    : np.ndarray  — average_channel_q_W values (n_zones,), or None
    z_centers_cm : np.ndarray  — axial zone centres in cm, or None
    """
    try:
        import gc
        from heating_profile_extraction import (
            get_normalization_factor,
            extract_mesh_heating,
        )
    except ImportError as exc:
        print(f"  WARNING: cannot import heating_profile_extraction: {exc}")
        return None, None, None

    sp_files = sorted(glob.glob(os.path.join(run_dir, "statepoint.*.h5")))
    if not sp_files:
        print(f"  WARNING: no statepoint in {run_dir}")
        return None, None, None
    sp_path = sp_files[-1]

    thermal_power_MW = params.get("thermal_power_MW", 10.0)
    symmetry_factor  = 6 if params.get("use_1/6_geometry", True) else 1

    try:
        source_per_sec = get_normalization_factor(sp_path, thermal_power_MW)
    except Exception as exc:
        print(f"  WARNING: normalization factor failed: {exc}")
        return None, None, None

    try:
        data = extract_mesh_heating(sp_path, source_per_sec, params, symmetry_factor,
                                    tally_name='mesh_heating')
    except Exception as exc:
        print(f"  WARNING: mesh heating extraction failed: {exc}")
        return None, None, None

    z_centers_cm   = data["z_centers"]              # (nz,) in cm
    heating_2d     = data["heating_2d"]             # (n_channels, nz)
    n_channels     = data["n_channels"]

    nonzero_mask = heating_2d.sum(axis=1) > 0
    if nonzero_mask.sum() == 0:
        print("  WARNING: all heating channels are zero")
        return None, None, None

    hottest_idx     = int(np.argmax(heating_2d.sum(axis=1)))
    hottest_profile = heating_2d[hottest_idx, :]
    avg_profile     = heating_2d[nonzero_mask, :].mean(axis=0)
    nz_idx          = np.where(nonzero_mask)[0]
    coldest_idx     = int(nz_idx[np.argmin(heating_2d.sum(axis=1)[nz_idx])])
    coldest_profile = heating_2d[coldest_idx, :]

    csv_path = os.path.join(run_dir, "neutronics_th.csv")
    header   = "z_center_cm,hottest_channel_q_W,average_channel_q_W,coldest_channel_q_W"
    np.savetxt(
        csv_path,
        np.column_stack([z_centers_cm, hottest_profile, avg_profile, coldest_profile]),
        delimiter=",", header=header, comments="", fmt="%.6e",
    )
    return csv_path, avg_profile, z_centers_cm


def _nc_htgr_temps(params, neutronics_csv_path):
    """
    Run the nc_htgr average-channel solver for the given neutronics heating
    profile and return interpolated temperature arrays aligned to OpenMC's
    n_ax_zones axial zones.

    Returns (T_coolant_z_K, T_compact_z_K, T_matrix_z_K) as numpy arrays,
    or (None, None, None) on failure.
    """
    try:
        from nc_htgr import (
            NeutronicsTable, integrated_cycle_with_channel,
            read_key_value_csv, parse_inputs_from_deck,
        )
    except ImportError as exc:
        print(f"  WARNING: cannot import nc_htgr: {exc}")
        return None, None, None

    n_ax       = int(params["n_ax_zones"])
    core_h_cm  = float(params["core_height"])
    refl_t_cm  = float(params["reflector_thickness"])
    L_heated_m = core_h_cm  * 0.01
    L_m        = (core_h_cm + 2.0 * refl_t_cm) * 0.01
    L_unheated = 0.5 * (L_m - L_heated_m)      # unheated entry/exit length [m]

    # Load all TH/Brayton inputs from nc_input.csv.
    _nc_input_path = os.path.join(_NC_HTGR_DIR, "nc_input.csv")
    _deck = read_key_value_csv(_nc_input_path) if os.path.exists(_nc_input_path) else {}
    ch, br = parse_inputs_from_deck(_deck)

    try:
        ntable = NeutronicsTable(neutronics_csv_path,
                                 N_fuel_channels=ch.N_fuel_channels)
    except Exception as exc:
        print(f"  WARNING: NeutronicsTable load failed: {exc}")
        return None, None, None

    # Override geometry-derived fields (computed from OpenMC params).
    ch.L              = L_m
    ch.L_heated       = L_heated_m
    ch.D_cool         = 2.0 * float(params["coolant_radius"]) * 0.01
    ch.D_compact      = 2.0 * float(params["compact_radius"]) * 0.01
    ch.pitch          = float(params["fuel_to_coolant_distance"]) * 0.01
    ch.packing_fraction = float(params["triso_pf"])

    # Fields not present in nc_input.csv — set by the coupling context.
    ch.axial_shape       = "neutronics_table"
    ch.neutronics_file   = neutronics_csv_path
    ch.channel_case      = "average"
    ch._neutronics_table = ntable
    ch._Q_per_channel_W  = ntable.Q_per_channel["average"]
    ch.L_heated          = ntable.L_heated

    try:
        # integrated_cycle_with_channel iterates the recuperator to find the
        # self-consistent core inlet temperature T3, then runs the full channel
        # solve at that inlet.  This replaces the fixed cold-inlet approach.
        channel_df, _, cycle_summary = integrated_cycle_with_channel(ch, br)
        print(f"    [TH] Brayton T3={cycle_summary['T3_K']:.1f} K  "
              f"T4={cycle_summary['T4_K']:.1f} K  "
              f"η_th={cycle_summary['eta_th']:.3f}")
    except Exception as exc:
        print(f"  WARNING: nc_htgr channel solve failed: {exc}")
        return None, None, None

    # Map nc_htgr z-nodes to OpenMC axial zone centres.
    axial_section_h_cm = core_h_cm / n_ax
    # Zone centres from bottom (index 0) to top (index n_ax-1)
    z_centers_cm = np.linspace(0.5 * axial_section_h_cm,
                               core_h_cm - 0.5 * axial_section_h_cm, n_ax)

    if ch.flow_upward:
        # Inlet at bottom; z_nc increases from bottom to top
        frac_from_inlet = z_centers_cm / core_h_cm
    else:
        # Inlet at top; z_nc increases from top to bottom
        frac_from_inlet = (core_h_cm - z_centers_cm) / core_h_cm

    z_nc_m = L_unheated + frac_from_inlet * L_heated_m     # nc_htgr positions [m]
    z_nc_m = np.clip(z_nc_m, 0.0, L_m)

    ch_z   = channel_df["z_m"].values
    T_bulk_K    = np.interp(z_nc_m, ch_z, channel_df["T_bulk_C"].values       + 273.15)
    T_fhw_K     = np.interp(z_nc_m, ch_z, channel_df["T_fuel_hole_wall_C"].values + 273.15)
    T_compact_K = np.interp(z_nc_m, ch_z, channel_df["T_compact_center_C"].values + 273.15)

    return T_bulk_K, T_compact_K, T_fhw_K


def th_coupler(
    params,
    depleted,
    output_dir,
    depletion_run_dir=None,
    bank_1=None,
    bank_2=None,
):
    """
    Iterative thermal-hydraulic coupler.

    Runs eigenvalue simulations with the current temperature profile, extracts
    the axial heating profile via the mesh_heating tally, feeds it into the
    nc_htgr single-channel solver to obtain updated temperature arrays, and
    repeats until both k_eff and the heating profile are converged.

    Convergence criteria (all must be satisfied):
      1. |k_new - k_prev| < th_coupler_k_tol  (default 0.0064, 1 beta U-235)
      2. max(|q_new - q_prev|) / max(q_new) < th_coupler_q_tol_frac  (default 0.05)
      3. At least th_coupler_min_iter iterations have been completed  (default 4)

    After th_coupler_max_iter iterations the loop breaks regardless (default 10).

    Only global tallies and the mesh_heating tally are enabled — all other
    tallies (leakage, BeO, zone_heating_local) are suppressed.

    Parameters
    ----------
    params : dict
        Simulation parameters (deep-copied; original unchanged).
        Rod positions are taken from params unless overridden by bank_1/bank_2.
    depleted : dict
        Depleted material compositions from reconstruct_depleted_materials().
        Pass {} for BOL (no injection).
    output_dir : str
        Root directory.  Iteration subdirectories are created here and deleted
        on completion; only the summary CSV survives.
    depletion_run_dir : str or None
        Passed to _assert_new_dir to prevent overwriting the depletion directory.
    bank_1 : float or None
        Override for bank_1_insertion.  None → use params value.
    bank_2 : float or None
        Override for bank_2_insertion.  None → use params value.

    Returns
    -------
    dict with keys:
        converged          : bool
        n_iterations       : int
        final_keff         : float
        final_keff_std     : float
        converged_params   : dict  — params deep-copy with _th_*_z arrays set
        csv_path           : str   — summary CSV path
    """
    os.makedirs(output_dir, exist_ok=True)

    k_tol        = float(params.get("th_coupler_k_tol",      0.0064))
    q_tol        = float(params.get("th_coupler_q_tol_frac", 0.05))
    min_iter     = int(params.get("th_coupler_min_iter",      4))
    max_iter     = int(params.get("th_coupler_max_iter",      10))
    ignore_keff  = bool(params.get("th_coupler_ignore_keff",  False))
    min_keff     = 1.0 - k_tol   # valid keff range: [1.0 - k_tol, 1.0 + k_tol]
    max_keff     = 1.0 + k_tol
    th_batches   = int(params.get("th_coupler_batches",       50))
    th_inactive  = int(params.get("th_coupler_inactive",      20))
    th_particles = int(params.get("th_coupler_particles",     50_000))

    print(f"\n{'─' * 70}")
    print(f"  TH COUPLER  (mesh heating + nc_htgr single-channel)")
    if ignore_keff:
        print(f"  k_tol={k_tol}  q_tol_frac={q_tol}  ignore_keff=True  "
              f"min_iter={min_iter}  max_iter={max_iter}")
        print(f"  NOTE: keff validity check disabled — converging on heating profile only")
    else:
        print(f"  k_tol={k_tol}  q_tol_frac={q_tol}  "
              f"keff_valid=[{min_keff:.4f}, {max_keff:.4f}]  "
              f"min_iter={min_iter}  max_iter={max_iter}")
    print(f"  Output: {output_dir}")
    print(f"{'─' * 70}")

    # Build base params for all iterations — absolute minimum tally set:
    #   use_heating_tally        → single un-filtered heating-local tally (normalization)
    #   use_mesh_heating_tally   → active-core-only 'mesh_heating' tally (axial profile).
    #                              Covers reactor_bottom→reactor_top only, so it is
    #                              cheaper than the full-core version: fewer bins to
    #                              score and a smaller statepoint per iteration.
    # All other tally groups (flux/fission, full-core, leakage, BeO) are disabled.
    base_params = copy.deepcopy(params)
    base_params["total_batches"]             = th_batches
    base_params["inactive_batches"]          = th_inactive
    base_params["particles"]                 = th_particles
    base_params["make_geometry_plots"]       = False
    base_params["use_global_tallies"]        = False
    base_params["use_heating_tally"]         = True   # normalization tally
    base_params["use_mesh_tallies"]          = False
    base_params["use_mesh_heating_tally"]    = True   # active-core mesh heating (axial profile)
    base_params["use_mesh_heating_full_tally"] = False
    base_params["use_BeO_tallies"]           = False
    base_params["use_leakage_tallies"]       = False

    if bank_1 is not None:
        base_params["bank_1_insertion"] = bank_1
    if bank_2 is not None:
        base_params["bank_2_insertion"] = bank_2

    csv_path  = os.path.join(output_dir, "th_coupler_summary.csv")
    csv_rows  = []
    iter_dirs = []

    prev_k      = None
    prev_q      = None
    best_params = copy.deepcopy(base_params)  # carries _th_*_z
    final_keff  = float("nan")
    final_std       = float("nan")
    converged       = False
    keff_true_iters = []   # keff values from iterations with valid deltas (it >= 1)

    def _write_summary_csv():
        # dk_ok   = |Δkeff| < k_tol  (rate of change has stabilised, not absolute value)
        # dq_ok   = Δq/q_max < q_tol (heating profile has stabilised)
        # keff_valid = keff ∈ [1.0 - k_tol, 1.0 + k_tol]  (absolute criticality check)
        # q_max_diff  = max(|q_new - q_prev|)  — numerator of q_max_change_frac
        # q_curr_max  = max(q_new)              — denominator of q_max_change_frac
        fieldnames = ["iteration", "keff", "keff_std", "delta_k",
                      "q_max_diff", "q_curr_max", "q_max_change_frac",
                      "dk_ok", "dq_ok", "keff_valid"]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    def _is_monotonic(vals):
        """Return True if vals (len >= 2) is strictly monotone increasing or decreasing."""
        if len(vals) < 2:
            return False
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        return all(d > 0 for d in diffs) or all(d < 0 for d in diffs)

    def _write_iter_profiles(it, z_cm, q_avg_W, T_cool_K, T_comp_K, T_mat_K):
        """Write per-iteration heating and temperature profile CSVs."""
        if z_cm is None:
            return
        # Heating profile
        heat_path = os.path.join(output_dir, f"th_iter_{it:02d}_heating.csv")
        with open(heat_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["z_center_cm", "avg_channel_q_W"])
            for z, q in zip(z_cm, q_avg_W):
                w.writerow([round(float(z), 4), round(float(q), 6)])
        # Temperature profile
        if T_cool_K is not None:
            temp_path = os.path.join(output_dir, f"th_iter_{it:02d}_temperatures.csv")
            with open(temp_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["z_center_cm", "T_coolant_K", "T_compact_K", "T_matrix_K"])
                for z, tc, tp, tm in zip(z_cm, T_cool_K, T_comp_K, T_mat_K):
                    w.writerow([round(float(z), 4),
                                round(float(tc), 4),
                                round(float(tp), 4),
                                round(float(tm), 4)])

    for it in range(max_iter + 1):  # +1: iteration 0 is the base; max_iter additional iters follow
        iter_dir = os.path.join(output_dir, f"th_iter_{it:02d}")
        iter_dirs.append(iter_dir)

        it_params = copy.deepcopy(best_params)

        print(f"\n  Iter {it:2d}: running eigenvalue  "
              f"(b1={it_params['bank_1_insertion']:.4f}, "
              f"b2={it_params['bank_2_insertion']:.4f})")

        _run_eigenvalue_with_depleted(it_params, depleted, iter_dir,
                                      depletion_run_dir=depletion_run_dir)
        k, k_std = _read_keff(iter_dir)
        final_keff = k
        final_std  = k_std

        # Extract heating profile → nc_htgr temperatures
        neutronics_csv, q_avg, z_centers_cm = \
            _extract_heating_csv_from_statepoint(iter_dir, it_params)

        if neutronics_csv is not None:
            T_cool_z, T_comp_z, T_mat_z = _nc_htgr_temps(it_params, neutronics_csv)
        else:
            T_cool_z = T_comp_z = T_mat_z = None

        # Save per-iteration heating and temperature profiles
        _write_iter_profiles(it, z_centers_cm, q_avg, T_cool_z, T_comp_z, T_mat_z)

        # Compute convergence metrics
        delta_k = abs(k - prev_k) if prev_k is not None else float("nan")
        if prev_q is not None and q_avg is not None and q_avg.max() > 0:
            q_max_diff    = float(np.max(np.abs(q_avg - prev_q)))  # max change across all zones
            q_curr_max    = float(q_avg.max())                     # peak q (denominator)
            q_change_frac = q_max_diff / q_curr_max
        else:
            q_max_diff    = float("nan")
            q_curr_max    = float("nan")
            q_change_frac = float("nan")

        # Track keff for iterations that have valid deltas (all except it == 0)
        if not np.isnan(delta_k):
            keff_true_iters.append(k)

        conv_k     = (not np.isnan(delta_k)       and delta_k       < k_tol)
        conv_q     = (not np.isnan(q_change_frac) and q_change_frac < q_tol)
        keff_valid = True if ignore_keff else (min_keff <= k <= max_keff)

        if ignore_keff:
            print(f"         k_eff = {k:.5f} ± {k_std:.5f}   "
                  f"Δk = {delta_k:.5f}   Δq/q_max = "
                  f"{q_change_frac:.4f}   (keff check disabled)"
                  if not np.isnan(q_change_frac)
                  else f"         k_eff = {k:.5f} ± {k_std:.5f}   "
                       f"Δk = {delta_k:.5f}   (keff check disabled)")
        else:
            print(f"         k_eff = {k:.5f} ± {k_std:.5f}   "
                  f"Δk = {delta_k:.5f}   Δq/q_max = "
                  f"{q_change_frac:.4f}   keff_valid={keff_valid}"
                  if not np.isnan(q_change_frac)
                  else f"         k_eff = {k:.5f} ± {k_std:.5f}   "
                       f"Δk = {delta_k:.5f}   keff_valid={keff_valid}")

        _fmt = lambda v: round(v, 6) if not np.isnan(v) else "nan"
        csv_rows.append({
            "iteration":         it,
            "keff":              round(k,     6),
            "keff_std":          round(k_std, 6),
            "delta_k":           _fmt(delta_k),
            "q_max_diff":        _fmt(q_max_diff),
            "q_curr_max":        _fmt(q_curr_max),
            "q_max_change_frac": _fmt(q_change_frac),
            "dk_ok":             conv_k,
            "dq_ok":             conv_q,
            "keff_valid":        keff_valid,
        })
        _write_summary_csv()

        # Update temperatures for next iteration
        if T_cool_z is not None:
            best_params["_th_coolant_z"] = T_cool_z.tolist()
            best_params["_th_compact_z"] = T_comp_z.tolist()
            best_params["_th_matrix_z"]  = T_mat_z.tolist()

        prev_k = k
        if q_avg is not None:
            prev_q = q_avg.copy()

        # Iteration 0 has no deltas (nan), so don't count it toward min_iter.
        # past_min is True once we have completed at least min_iter iterations
        # that each have a valid Δk and Δq to evaluate.
        past_min = (it >= min_iter)

        if past_min and conv_k and conv_q and keff_valid:
            # Full convergence: Δk, Δq, and keff all within bounds (or ignored).
            converged = True
            if ignore_keff:
                print(f"\n  TH Coupler converged at iteration {it}  "
                      f"(Δk={delta_k:.5f}, Δq/q_max={q_change_frac:.4f}, keff check disabled)")
            else:
                print(f"\n  TH Coupler converged at iteration {it}  "
                      f"(Δk={delta_k:.5f}, Δq/q_max={q_change_frac:.4f}, "
                      f"keff={k:.5f} ∈ [{min_keff:.4f}, {max_keff:.4f}])")
            break

        if not ignore_keff and past_min and conv_q and not keff_valid:
            # Early exit only when keff has been monotonically drifting (all
            # increasing or all decreasing) over the last 4 true iterations.
            # If keff is oscillating it may still self-correct — don't bail.
            # Skipped entirely when ignore_keff=True.
            recent4 = keff_true_iters[-4:]
            if _is_monotonic(recent4):
                print(f"\n  WARNING: TH Coupler early exit at iteration {it}: "
                      f"q converged (Δq/q_max={q_change_frac:.4f}) but "
                      f"keff={k:.5f} outside [{min_keff:.4f}, {max_keff:.4f}] "
                      f"and monotonically {'increasing' if recent4[-1] > recent4[0] else 'decreasing'} "
                      f"over last {len(recent4)} true iterations. "
                      f"Caller should re-run critical rod search with updated temps.")
                break
            else:
                print(f"         (q converged but keff={k:.5f} out of range; "
                      f"keff not monotonic over last {len(recent4)} true iters — continuing)")

    if not converged:
        print(f"\n  WARNING: TH Coupler did not converge in {max_iter + 1} iterations "
              f"(1 base + {max_iter} additional).")
        print(f"  Best k_eff = {final_keff:.5f}")

    print(f"\n  Summary CSV -> {csv_path}")

    # Clean up iteration subdirectories
    for d in iter_dirs:
        try:
            shutil.rmtree(d)
        except Exception as e:
            print(f"  WARNING: could not delete {d}: {e}")

    # Restore the original simulation-level settings in best_params before
    # returning.  The TH coupler overrides these keys in base_params for
    # efficiency (fewer tallies, fixed batch counts), but converged_params is
    # meant to carry only the temperature profile (_th_*_z) forward into the
    # next critical-search / depletion build — not the reduced tally flags.
    # Without this restore, current_th_params becomes permanently poisoned
    # (use_BeO_tallies=False, etc.) from the first TH coupler call onward,
    # so no BeO / mesh / leakage tallies are ever written to tallies.xml or
    # run_params.json for any subsequent depletion step.
    _tally_and_sim_keys = (
        "use_global_tallies",
        "use_mesh_tallies",
        "use_BeO_tallies",
        "use_leakage_tallies",
        "use_heating_tally",
        "use_mesh_heating_tally",
        "use_mesh_heating_full_tally",
        "total_batches",
        "inactive_batches",
        "particles",
        "make_geometry_plots",
    )
    for _key in _tally_and_sim_keys:
        if _key in params:
            best_params[_key] = params[_key]

    return {
        "converged":        converged,
        "final_keff_valid": (min_keff <= final_keff <= max_keff) if not np.isnan(final_keff) else False,
        "n_iterations":     len(csv_rows),
        "final_keff":       final_keff,
        "final_keff_std":   final_std,
        "converged_params": best_params,
        "csv_path":         csv_path,
    }


# ============================================================================
# STEP 4 — FULL-TALLY EIGENVALUE RUN WITH DEPLETED MATERIALS
# ============================================================================

def run_full_tally_eigenvalue(params, depleted, run_dir, label="", depletion_run_dir=None):
    """
    Run a full-tally eigenvalue simulation with depleted materials.

    Enables mesh tallies, global tallies, and leakage tallies so that
    all post-processing scripts (heat map, tally plotter, leakage spectrum)
    can be applied to the resulting statepoint.

    Returns
    -------
    n_trisos : int
    keff     : float
    keff_std : float
    """
    full_params = copy.deepcopy(params)
    full_params["use_mesh_tallies"]    = True
    full_params["use_global_tallies"]  = True
    full_params["use_leakage_tallies"] = True
    full_params["make_geometry_plots"] = False
    full_params["total_batches"]       = 500
    full_params["inactive_batches"]    = 100

    print(f"\n{'=' * 70}")
    if label:
        print(f"  FULL-TALLY EIGENVALUE RUN — {label}")
    else:
        print(f"  FULL-TALLY EIGENVALUE RUN")
    print(f"{'=' * 70}")

    n_trisos, _ = _run_eigenvalue_with_depleted(full_params, depleted, run_dir,
                                                depletion_run_dir=depletion_run_dir)
    keff, keff_std = _read_keff(run_dir)

    print(f"\n  k_eff = {keff:.5f} ± {keff_std:.5f}")
    return n_trisos, keff, keff_std


# ============================================================================
# STEP 5 — REACTIVITY COEFFICIENT STUDY AT A GIVEN BURNUP STEP
# ============================================================================

def run_mol_eol_reactivity_coefficients(
    depletion_run_dir,
    step_idx,
    step_label,
    output_base_dir,
    delta_T_values=None,
    coefficients=None,
    bank_1_override=None,
    bank_2_override=None,
):
    """
    Compute FTC/MTC/ITC at a specific depletion step using depleted materials.

    Parameters
    ----------
    depletion_run_dir : str
        Directory containing depletion_results.h5 and run_params.json.
    step_idx : int
        Depletion step index (-1 = EOL).
    step_label : str
        Human-readable label ("MOL", "EOL", etc.) used in directory names.
    output_base_dir : str
        Root directory for all output sub-directories.
    delta_T_values : list[float], optional
        Temperature perturbations in K. Default: [50, 100, 150].
    coefficients : list[str], optional
        Subset of ["FTC", "MTC", "ITC"]. Default: all three.
    bank_1_override : float or None
        If given, override bank 1 insertion for all runs.
    bank_2_override : float or None
        If given, override bank 2 insertion for all runs.
        Bank 3 is always left at 0.

    Returns
    -------
    dict : Results from run_reactivity_coefficients().
    """
    if delta_T_values is None:
        delta_T_values = [50.0, 100.0, 150.0]
    if coefficients is None:
        coefficients = ["FTC", "MTC", "ITC"]

    print(f"\n{'=' * 80}")
    print(f"MOL/EOL REACTIVITY COEFFICIENTS — {step_label}")
    print(f"{'=' * 80}")

    depleted, fuel_mat_ids_2d, run_params, step_time_days = \
        reconstruct_depleted_materials(depletion_run_dir, step_idx)

    print(f"  Step time: {step_time_days:.1f} days")

    # Merge stored run_params with config defaults for any missing keys
    import config as cfg
    merged = {**cfg.params, **run_params}

    if bank_1_override is not None:
        merged["bank_1_insertion"] = bank_1_override
        print(f"  Rod insertion override: bank 1 = {bank_1_override:.4f}")
    if bank_2_override is not None:
        merged["bank_2_insertion"] = bank_2_override
        print(f"  Rod insertion override: bank 2 = {bank_2_override:.4f}")
    merged["bank_3_insertion"] = 0.0

    # Disable heavy options for reactivity perturbation runs
    merged["make_geometry_plots"] = False
    merged["use_mesh_tallies"]    = False
    merged["use_BeO_tallies"]     = False

    rc_output_dir = os.path.join(output_base_dir,
                                  f"reactivity_coefficients_{step_label}")
    os.makedirs(rc_output_dir, exist_ok=True)

    # ---- Reference run (rods at nominal position) ----
    ref_dir = os.path.join(rc_output_dir, "reference_nominal")
    print(f"\n  Running reference case → {ref_dir}")
    _run_eigenvalue_with_depleted(merged, depleted, ref_dir,
                                  depletion_run_dir=depletion_run_dir)
    k_ref, k_ref_std = _read_keff(ref_dir)
    print(f"  Reference k_eff = {k_ref:.5f} ± {k_ref_std:.5f}")

    # ---- Perturbed cases ----
    from reactivity_coefficients import (
        PERTURBATION_BUILDERS, COEFF_LABELS,
        _alpha, _alpha_uncertainty, _reactivity_pcm, _print_summary,
    )
    from reactivity_coefficients_postprocessing import (
        save_results as _save_results,
        plot_results as _plot_results,
    )

    rho_ref = _reactivity_pcm(k_ref)
    all_results = {}

    for coeff_name in coefficients:
        builder = PERTURBATION_BUILDERS[coeff_name]
        label   = COEFF_LABELS[coeff_name]

        print(f"\n{'─' * 70}")
        print(f"  Computing {label}")
        print(f"{'─' * 70}")

        coeff_dir = os.path.join(rc_output_dir, f"perturbation_{coeff_name}")
        os.makedirs(coeff_dir, exist_ok=True)
        case_results = []

        for dT in delta_T_values:
            for sign, sign_label in [(+1, "pos"), (-1, "neg")]:
                actual_dT = sign * dT
                case_label = f"{coeff_name}_dT_{sign_label}{int(dT)}K"
                case_dir = os.path.join(coeff_dir, case_label)

                pert_params = builder(merged, actual_dT)
                pert_params["make_geometry_plots"] = False
                pert_params["use_mesh_tallies"]    = False
                pert_params["use_BeO_tallies"]     = False

                # Skip if already run
                try:
                    k_pert, k_pert_std = _read_keff(case_dir)
                    print(f"  [{case_label}] Already computed — k = {k_pert:.5f}")
                except Exception:
                    print(f"  [{case_label}] Running ΔT = {actual_dT:+.0f} K…")
                    _run_eigenvalue_with_depleted(pert_params, depleted, case_dir,
                                                  depletion_run_dir=depletion_run_dir)
                    k_pert, k_pert_std = _read_keff(case_dir)
                    print(f"  [{case_label}] k = {k_pert:.5f} ± {k_pert_std:.5f}")

                alpha_val = _alpha(k_ref, k_pert, actual_dT)
                alpha_unc = _alpha_uncertainty(k_ref, k_ref_std, k_pert,
                                               k_pert_std, actual_dT)
                case_results.append({
                    "delta_T":        actual_dT,
                    "k_pert":         k_pert,
                    "k_pert_std":     k_pert_std,
                    "rho_pert":       _reactivity_pcm(k_pert),
                    "alpha_pcm_per_K": alpha_val,
                    "alpha_std":      alpha_unc,
                    "case_dir":       case_dir,
                })

        # Central-difference estimates
        central_diff = []
        for dT in delta_T_values:
            pos = next((c for c in case_results if c["delta_T"] == +dT), None)
            neg = next((c for c in case_results if c["delta_T"] == -dT), None)
            if pos and neg:
                alpha_cd  = _alpha(neg["k_pert"], pos["k_pert"], 2 * dT)
                alpha_unc = _alpha_uncertainty(neg["k_pert"], neg["k_pert_std"],
                                               pos["k_pert"], pos["k_pert_std"],
                                               2 * dT)
                central_diff.append({
                    "delta_T":                dT,
                    "alpha_central_pcm_per_K": alpha_cd,
                    "alpha_central_std":       alpha_unc,
                })

        if central_diff:
            avg_alpha     = np.mean([c["alpha_central_pcm_per_K"] for c in central_diff])
            avg_alpha_std = (np.sqrt(np.sum([c["alpha_central_std"]**2
                                             for c in central_diff]))
                             / len(central_diff))
        else:
            avg_alpha     = np.mean([c["alpha_pcm_per_K"] for c in case_results])
            avg_alpha_std = (np.sqrt(np.sum([c["alpha_std"]**2
                                             for c in case_results]))
                             / len(case_results))

        all_results[coeff_name] = {
            "label":                   label,
            "cases":                   case_results,
            "central_difference":      central_diff,
            "average_alpha_pcm_per_K": avg_alpha,
            "average_alpha_std":       avg_alpha_std,
        }

    _print_summary(all_results, k_ref, k_ref_std, rho_ref)
    _save_results(all_results, k_ref, k_ref_std, rho_ref, rc_output_dir)
    _plot_results(all_results, k_ref, rc_output_dir)

    # Save step metadata
    meta = {
        "step_label":       step_label,
        "step_idx":         step_idx,
        "step_time_days":   step_time_days,
        "rod_insertion":    merged.get("bank_3_insertion", 0.0),
        "reference_k_eff":  k_ref,
        "reference_k_std":  k_ref_std,
    }
    with open(os.path.join(rc_output_dir, "mol_eol_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Reactivity coefficient results saved to: {rc_output_dir}")
    return all_results


# ============================================================================
# STEP 6 — HEAT MAP AND LEAKAGE SPECTRUM EXTRACTION
# ============================================================================

def run_mol_eol_heat_map(
    depletion_run_dir,
    step_idx,
    step_label,
    output_base_dir,
    rod_mode="critical",
    fixed_bank_1=None,
    fixed_bank_2=None,
    k_tol=0.003,
    max_search_iter=20,
):
    """
    Extract heat map and leakage spectrum at a specific burnup step.

    Parameters
    ----------
    depletion_run_dir : str
        Directory containing depletion_results.h5 and run_params.json.
    step_idx : int
        Depletion step index (-1 = EOL).
    step_label : str
        Human-readable label ("MOL", "EOL").
    output_base_dir : str
        Root directory for all output sub-directories.
    rod_mode : str
        One of:
          "rods_out"   — all banks at 0.
          "all_in"     — bank 1 = 1.0, bank 2 = 1.0.
          "critical"   — run critical rod search first, then full-tally run.
          "fixed"      — use fixed_bank_1 / fixed_bank_2 values.
    fixed_bank_1 : float or None
        Bank 1 insertion for rod_mode="fixed".
    fixed_bank_2 : float or None
        Bank 2 insertion for rod_mode="fixed".
    k_tol : float
        Convergence tolerance for the criticality search.
    max_search_iter : int
        Maximum iterations for the binary search.

    Returns
    -------
    dict with keys: run_dir, keff, keff_std, rod_mode, rod_insertion.
    """
    print(f"\n{'=' * 80}")
    print(f"MOL/EOL HEAT MAP — {step_label} | rod_mode = {rod_mode}")
    print(f"{'=' * 80}")

    depleted, fuel_mat_ids_2d, run_params, step_time_days = \
        reconstruct_depleted_materials(depletion_run_dir, step_idx)

    import config as cfg
    merged = {**cfg.params, **run_params}
    merged["make_geometry_plots"] = False

    # ---- Determine rod insertion for this run ----
    critical_search_result = None

    if rod_mode == "rods_out":
        merged["bank_1_insertion"] = 0.0
        merged["bank_2_insertion"] = 0.0
        merged["bank_3_insertion"] = 0.0
        rod_insertion = 0.0
        label_suffix  = "rods_out"

    elif rod_mode == "all_in":
        merged["bank_1_insertion"] = 1.0
        merged["bank_2_insertion"] = 1.0
        merged["bank_3_insertion"] = 0.0
        rod_insertion = 1.0
        label_suffix  = "all_rods_in"

    elif rod_mode == "fixed":
        if fixed_bank_1 is None and fixed_bank_2 is None:
            raise ValueError("rod_mode='fixed' requires at least one of "
                             "fixed_bank_1 or fixed_bank_2 to be set")
        b1 = fixed_bank_1 if fixed_bank_1 is not None else merged.get("bank_1_insertion", 0.0)
        b2 = fixed_bank_2 if fixed_bank_2 is not None else 0.0
        merged["bank_1_insertion"] = b1
        merged["bank_2_insertion"] = b2
        merged["bank_3_insertion"] = 0.0
        rod_insertion = b1  # used for label/metadata; b2 stored separately
        label_suffix  = f"b1_{b1:.4f}_b2_{b2:.4f}"

    elif rod_mode == "critical":
        # First, quick run with rods out to check if supercritical
        rods_out_params = copy.deepcopy(merged)
        rods_out_params["bank_1_insertion"] = 0.0
        rods_out_params["bank_2_insertion"] = 0.0
        rods_out_params["bank_3_insertion"] = 0.0
        rods_out_params["make_geometry_plots"] = False
        rods_out_params["use_mesh_tallies"]    = False
        rods_out_params["use_BeO_tallies"]     = False
        rods_out_params["use_leakage_tallies"] = False
        rods_out_params["use_global_tallies"]  = False

        quick_dir = os.path.join(output_base_dir,
                                  f"heatmap_{step_label}_criticality_check")
        print(f"\n  Quick k_eff check (rods out)...")
        _run_eigenvalue_with_depleted(rods_out_params, depleted, quick_dir,
                                      depletion_run_dir=depletion_run_dir)
        k_check, k_check_std = _read_keff(quick_dir)
        print(f"  k_eff (rods out) = {k_check:.5f} ± {k_check_std:.5f}")

        if abs(k_check - 1.0) < k_tol and k_check > 1.0:
            # Already critical — no search needed
            print(f"  Core is already critical with rods out — skipping search")
            rod_insertion = 0.0
            merged["bank_1_insertion"] = 0.0
            merged["bank_2_insertion"] = 0.0
            merged["bank_3_insertion"] = 0.0
            label_suffix = "critical_rods_out"
        elif k_check < 1.0:
            print(f"  Core is subcritical with rods out — cannot insert rods to reach k=1")
            print(f"  Proceeding with rods_out configuration")
            rod_insertion = 0.0
            merged["bank_1_insertion"] = 0.0
            merged["bank_2_insertion"] = 0.0
            merged["bank_3_insertion"] = 0.0
            label_suffix = "rods_out_subcritical"
        else:
            # Supercritical — search for critical insertion
            search_dir = os.path.join(output_base_dir,
                                       f"heatmap_{step_label}_rod_search")
            critical_search_result = find_critical_rod_insertion(
                merged, depleted, search_dir,
                k_target=1.0,
                k_tol=k_tol,
                max_iter=max_search_iter,
            )
            merged["bank_1_insertion"] = critical_search_result["critical_bank_1"]
            merged["bank_2_insertion"] = critical_search_result["critical_bank_2"]
            merged["bank_3_insertion"] = 0.0
            rod_insertion = critical_search_result["critical_insertion"]
            label_suffix = (f"critical_b1_{critical_search_result['critical_bank_1']:.4f}"
                            f"_b2_{critical_search_result['critical_bank_2']:.4f}")
    else:
        raise ValueError(f"Unknown rod_mode: '{rod_mode}'. "
                         "Use 'rods_out', 'all_in', 'fixed', or 'critical'.")

    # ---- Full-tally run ----
    full_run_dir = os.path.join(output_base_dir,
                                 f"heatmap_{step_label}_{label_suffix}")

    n_trisos, keff, keff_std = run_full_tally_eigenvalue(
        merged, depleted, full_run_dir,
        label=f"{step_label} | {rod_mode} | ins={rod_insertion:.4f}"
    )

    # ---- Post-processing: heat map ----
    try:
        from heating_profile_extraction import run_heating_profile_extraction
        print("\n  Running heating profile extraction...")
        run_heating_profile_extraction(full_run_dir, merged)
    except Exception as e:
        print(f"  WARNING: Heating profile extraction failed: {e}")
        import traceback; traceback.print_exc()

    # ---- Post-processing: leakage spectrum ----
    try:
        from leakage_spectrum import run_leakage_analysis
        print("\n  Running leakage spectrum analysis...")
        run_leakage_analysis(full_run_dir, merged)
    except Exception as e:
        print(f"  WARNING: Leakage spectrum analysis failed: {e}")
        import traceback; traceback.print_exc()

    # ---- Post-processing: tally plotter ----
    try:
        from tally_plotter import run_tally_plots
        print("\n  Running tally plots...")
        run_tally_plots(full_run_dir, merged)
    except Exception as e:
        print(f"  WARNING: Tally plotting failed: {e}")

    # ---- Save metadata ----
    meta = {
        "step_label":       step_label,
        "step_idx":         step_idx,
        "step_time_days":   step_time_days,
        "rod_mode":         rod_mode,
        "rod_insertion":    rod_insertion,
        "k_eff":            keff,
        "k_eff_std":        keff_std,
        "full_run_dir":     full_run_dir,
    }
    if critical_search_result is not None:
        meta["critical_search"] = critical_search_result

    meta_path = os.path.join(full_run_dir, "mol_eol_heatmap_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\n  Metadata saved to: {meta_path}")

    return meta

# ============================================================================
# ORCHESTRATOR — RUN ALL ANALYSES FOR ONE BURNUP STEP
# ============================================================================

def run_mol_eol_analysis(
    depletion_run_dir,
    step_idx=-1,
    step_label="EOL",
    output_base_dir=None,
    run_reactivity_study=True,
    run_heat_map=True,
    run_leakage=True,
    run_all_rods_in=True,
    delta_T_values=None,
    coefficients=None,
    k_tol=0.003,
    max_search_iter=20,
    critical_bank_1_override=None,
    critical_bank_2_override=None,
):
    """
    Orchestrate the full MOL/EOL analysis suite for one depletion step.

    Runs (in order, all optional):
      0. Critical rod search (shared) — finds the critical insertion once.
      1. Reactivity coefficient study (FTC, MTC, ITC) at critical insertion.
      2. Heat map + leakage at the same critical insertion (no re-search).
      3. Heat map + leakage with all rods fully inserted.

    Parameters
    ----------
    depletion_run_dir : str
        Original depletion run directory.
    step_idx : int
        Depletion step.  -1 = EOL (default).
    step_label : str
        Human-readable label ("MOL", "EOL") used in directory/file names.
    output_base_dir : str or None
        If None, creates a sub-directory inside depletion_run_dir.
    run_reactivity_study : bool
    run_heat_map : bool
        If True, runs both "critical" and (optionally) "all_in" heat maps.
    run_leakage : bool
        Leakage spectrum is extracted inside run_mol_eol_heat_map automatically.
        This flag enables/disables the standalone leakage-only pass.
    run_all_rods_in : bool
        If True and run_heat_map is True, also runs a separate "all_in" heat map.
    delta_T_values, coefficients : see run_mol_eol_reactivity_coefficients().
    k_tol, max_search_iter : see find_critical_rod_insertion().
    critical_bank_1_override : float or None
        If provided, skip the critical rod search and use this value directly
        for bank 1 insertion.
    critical_bank_2_override : float or None
        If provided alongside critical_bank_1_override, also override bank 2.

    Returns
    -------
    dict : Summary of all analyses performed.
    """
    if output_base_dir is None:
        output_base_dir = os.path.join(depletion_run_dir,
                                        f"mol_eol_analysis_{step_label}")
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"\n{'#' * 80}")
    print(f"#  MOL/EOL ANALYSIS — {step_label}")
    print(f"#  Depletion run: {depletion_run_dir}")
    print(f"#  Output:        {output_base_dir}")
    print(f"{'#' * 80}\n")

    summary = {
        "step_label":          step_label,
        "step_idx":            step_idx,
        "depletion_run_dir":   depletion_run_dir,
        "output_base_dir":     output_base_dir,
    }

    # ── Find critical rod insertion ONCE ────────────────────────────────────
    # All power-condition analyses (RC study + heat map) share this position.
    # The critical rod search is the "expensive" step; running it once avoids
    # redundant work and guarantees consistency between analyses.
    # If override values are provided, skip the search entirely.
    critical_bank_1 = 0.0   # fall back to rods-out if search skipped
    critical_bank_2 = 0.0

    _cs_log_path = os.path.join(depletion_run_dir, "critical_search_depletion_log.json")

    if critical_bank_1_override is not None:
        critical_bank_1 = float(critical_bank_1_override)
        critical_bank_2 = float(critical_bank_2_override) if critical_bank_2_override is not None else 0.0
        summary["critical_rod_search"] = {
            "skipped": True,
            "critical_bank_1": critical_bank_1,
            "critical_bank_2": critical_bank_2,
        }
        print(f"\n{'─' * 70}")
        print(f"  CRITICAL ROD SEARCH — skipped (using supplied values)")
        print(f"  Bank 1 = {critical_bank_1:.4f},  Bank 2 = {critical_bank_2:.4f}")
        print(f"{'─' * 70}")
    elif os.path.exists(_cs_log_path) and (run_heat_map or run_reactivity_study):
        # CS depletion run — read the stored critical rod position for this step
        # from critical_search_depletion_log.json instead of re-running the search.
        # The log is a list of per-step entries (1-based "step" field) written by
        # run_coupled_depletion().  H5 index i (0-based) maps to log entry i.
        with open(_cs_log_path) as _f:
            _cs_log = json.load(_f)
        _log_idx = step_idx if step_idx >= 0 else len(_cs_log) + step_idx
        _log_idx = max(0, min(_log_idx, len(_cs_log) - 1))
        _log_entry = _cs_log[_log_idx]
        critical_bank_1 = float(_log_entry["bank_1_insertion"])
        critical_bank_2 = float(_log_entry["bank_2_insertion"])
        summary["critical_rod_search"] = {
            "from_cs_log":    True,
            "log_step":       _log_entry["step"],
            "critical_bank_1": critical_bank_1,
            "critical_bank_2": critical_bank_2,
            "critical_keff":   _log_entry.get("critical_keff"),
            "critical_keff_std": _log_entry.get("critical_keff_std"),
            "converged":       _log_entry.get("converged"),
        }
        print(f"\n{'─' * 70}")
        print(f"  CRITICAL ROD SEARCH — skipped (CS depletion log found)")
        print(f"  Step {_log_entry['step']}:  "
              f"bank_1 = {critical_bank_1:.4f},  bank_2 = {critical_bank_2:.4f}"
              + (f"  (k = {_log_entry['critical_keff']:.5f})"
                 if _log_entry.get("critical_keff") is not None else ""))
        print(f"{'─' * 70}")
    elif run_heat_map or run_reactivity_study:
        print(f"\n{'─' * 70}")
        print(f"  CRITICAL ROD SEARCH — shared pre-step")
        print(f"{'─' * 70}")
        try:
            # Reconstruct depleted compositions and build merged params here
            # so the search reuses the same data that the sub-runs will use.
            import config as cfg
            _dep, _, _rp, _ = reconstruct_depleted_materials(
                depletion_run_dir, step_idx)
            _params_search = {**cfg.params, **_rp}
            _params_search["make_geometry_plots"] = False
            _params_search["use_mesh_tallies"]    = False
            _params_search["use_BeO_tallies"]     = False
            _params_search["use_leakage_tallies"] = False
            _params_search["use_global_tallies"]  = False

            _search_dir = os.path.join(output_base_dir, "critical_rod_search")
            _crit = find_critical_rod_insertion(
                _params_search, _dep, _search_dir,
                k_target=1.0,
                k_tol=k_tol,
                max_iter=max_search_iter,
            )
            critical_bank_1 = _crit["critical_bank_1"]
            critical_bank_2 = _crit["critical_bank_2"]
            summary["critical_rod_search"] = _crit
            print(f"\n  Critical position: bank 1 = {critical_bank_1:.4f}, "
                  f"bank 2 = {critical_bank_2:.4f}  "
                  f"(k = {_crit['critical_keff']:.5f} ± {_crit['critical_keff_std']:.5f})")
        except Exception as e:
            print(f"\n  WARNING: Critical rod search failed ({e}); "
                  f"falling back to rods-out for all analyses.")
            import traceback; traceback.print_exc()
            critical_bank_1 = 0.0
            critical_bank_2 = 0.0
            summary["critical_rod_search"] = {"error": str(e)}

    # 1. Reactivity coefficients at critical rod position (power conditions)
    if run_reactivity_study:
        try:
            rc_results = run_mol_eol_reactivity_coefficients(
                depletion_run_dir=depletion_run_dir,
                step_idx=step_idx,
                step_label=step_label,
                output_base_dir=output_base_dir,
                delta_T_values=delta_T_values,
                coefficients=coefficients,
                bank_1_override=critical_bank_1,
                bank_2_override=critical_bank_2,
            )
            summary["reactivity_coefficients"] = {
                name: {
                    "average_alpha_pcm_per_K": res["average_alpha_pcm_per_K"],
                    "average_alpha_std":        res["average_alpha_std"],
                }
                for name, res in rc_results.items()
            }
        except Exception as e:
            print(f"\nERROR in reactivity coefficient study: {e}")
            import traceback; traceback.print_exc()
            summary["reactivity_coefficients"] = {"error": str(e)}

    # 2. Heat map — critical rod position (reuse insertion found above)
    if run_heat_map:
        try:
            hm_critical = run_mol_eol_heat_map(
                depletion_run_dir=depletion_run_dir,
                step_idx=step_idx,
                step_label=step_label,
                output_base_dir=output_base_dir,
                rod_mode="fixed",
                fixed_bank_1=critical_bank_1,
                fixed_bank_2=critical_bank_2,
            )
            summary["heat_map_critical"] = hm_critical
        except Exception as e:
            print(f"\nERROR in heat map (critical): {e}")
            import traceback; traceback.print_exc()
            summary["heat_map_critical"] = {"error": str(e)}

    # 3. Heat map — all rods fully inserted
    if run_heat_map and run_all_rods_in:
        try:
            hm_all_in = run_mol_eol_heat_map(
                depletion_run_dir=depletion_run_dir,
                step_idx=step_idx,
                step_label=step_label,
                output_base_dir=output_base_dir,
                rod_mode="all_in",
            )
            summary["heat_map_all_rods_in"] = hm_all_in
        except Exception as e:
            print(f"\nERROR in heat map (all rods in): {e}")
            import traceback; traceback.print_exc()
            summary["heat_map_all_rods_in"] = {"error": str(e)}

    # Save overall summary
    summary_path = os.path.join(output_base_dir, "mol_eol_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n{'#' * 80}")
    print(f"#  MOL/EOL ANALYSIS COMPLETE — {step_label}")
    print(f"#  Summary: {summary_path}")
    print(f"{'#' * 80}\n")

    return summary


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MOL/EOL analysis on a completed OpenMC depletion run."
    )
    parser.add_argument("depletion_run_dir",
                        help="Directory containing depletion_results.h5")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "CriticalSearch"],
                        help="Run mode: 'full' runs all analyses (default); "
                             "'CriticalSearch' runs only the critical rod search "
                             "and prints the result")
    parser.add_argument("--step", type=int, default=-1,
                        help="Depletion step index (-1 = EOL, default)")
    parser.add_argument("--label", type=str, default="EOL",
                        help="Step label (MOL, EOL, etc.)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <depletion_run_dir>/mol_eol_analysis_<label>)")
    parser.add_argument("--no-rc",  action="store_true",
                        help="Skip reactivity coefficient study (full mode only)")
    parser.add_argument("--no-hm",  action="store_true",
                        help="Skip heat map extraction (full mode only)")
    parser.add_argument("--no-rods-in", action="store_true",
                        help="Skip the 'all rods in' heat map run (full mode only)")
    parser.add_argument("--k-tol", type=float, default=0.003,
                        help="Criticality search tolerance (default 0.003)")
    parser.add_argument("--max-iter", type=int, default=20,
                        help="Max iterations for criticality search (default 20)")
    parser.add_argument("--bank1", type=float, default=None,
                        help="Skip critical search and use this bank 1 insertion fraction "
                             "(full mode only; also sets --bank2 if provided)")
    parser.add_argument("--bank2", type=float, default=None,
                        help="Bank 2 insertion fraction to use with --bank1 (default 0.0)")

    args = parser.parse_args()

    if args.mode == "CriticalSearch":
        # ── Standalone critical rod search ──────────────────────────────────
        output_base = args.output or os.path.join(
            args.depletion_run_dir,
            f"mol_eol_analysis_{args.label}"
        )
        os.makedirs(output_base, exist_ok=True)

        print(f"\n{'#' * 80}")
        print(f"#  CRITICAL ROD SEARCH — {args.label}")
        print(f"#  Depletion run: {args.depletion_run_dir}")
        print(f"#  Output:        {output_base}")
        print(f"{'#' * 80}\n")

        import config as cfg
        dep, _, rp, step_time_days = reconstruct_depleted_materials(
            args.depletion_run_dir, args.step)
        search_params = {**cfg.params, **rp}
        search_params["make_geometry_plots"] = False
        search_params["use_mesh_tallies"]    = False
        search_params["use_BeO_tallies"]     = False
        search_params["use_leakage_tallies"] = False
        search_params["use_global_tallies"]  = False

        search_dir = os.path.join(output_base, "critical_rod_search")
        result = find_critical_rod_insertion(
            search_params, dep, search_dir,
            k_target=1.0,
            k_tol=args.k_tol,
            max_iter=args.max_iter,
        )

        print(f"\n{'─' * 70}")
        print(f"  CRITICAL SEARCH RESULT — {args.label}  ({step_time_days:.1f} days)")
        print(f"  Bank 1 insertion : {result['critical_bank_1']:.4f}")
        print(f"  Bank 2 insertion : {result['critical_bank_2']:.4f}")
        print(f"  Search stage     : {result['search_stage']}")
        print(f"  k_eff            : {result['critical_keff']:.5f} "
              f"± {result['critical_keff_std']:.5f}")
        print(f"  Converged        : {result['converged']}  "
              f"({result['n_iterations']} iterations)")
        print(f"{'─' * 70}\n")

        result_path = os.path.join(output_base, "critical_search_result.json")
        with open(result_path, "w") as f:
            json.dump({**result, "step_label": args.label,
                       "step_time_days": step_time_days}, f, indent=2, default=float)
        print(f"  Result saved to: {result_path}")

    else:
        # ── Full MOL/EOL analysis suite ──────────────────────────────────────
        run_mol_eol_analysis(
            depletion_run_dir        = args.depletion_run_dir,
            step_idx                 = args.step,
            step_label               = args.label,
            output_base_dir          = args.output,
            run_reactivity_study     = not args.no_rc,
            run_heat_map             = not args.no_hm,
            run_all_rods_in          = not args.no_rods_in,
            k_tol                    = args.k_tol,
            max_search_iter          = args.max_iter,
            critical_bank_1_override = args.bank1,
            critical_bank_2_override = args.bank2,
        )
