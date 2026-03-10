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
import json
import copy
import subprocess
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — script can be called from anywhere
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(SCRIPT_DIR)   # MicroHTGR/
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import openmc
import openmc.deplete

cross_sections_path = '/home/cade/Desktop/OpenMC/CrossSections/cross_sections.xml'
os.environ['OPENMC_CROSS_SECTIONS'] = cross_sections_path
openmc.config['cross_sections'] = cross_sections_path


# ============================================================================
# STEP 1 — RECONSTRUCT DEPLETED MATERIAL COMPOSITIONS
# ============================================================================

def reconstruct_depleted_materials(depletion_run_dir, step_idx):
    """
    Read depletion_results.h5 and return atom-density maps for all fuel zones.

    Parameters
    ----------
    depletion_run_dir : str
        Directory containing depletion_results.h5 and run_params.json.
    step_idx : int
        Depletion step index.  -1 = final (EOL) step.

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

    h5_path = os.path.join(depletion_run_dir, "depletion_results.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"depletion_results.h5 not found in {depletion_run_dir}")

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

    # ── Burnable poison (B4C) ──────────────────────────────────────────────
    # The poison depletes heavily (B-10 burns up).  Injecting the depleted
    # composition is essential for a correct post-depletion keff.
    poison_mat_id_str = str(run_params.get("poison_material_id", ""))
    poison_vol_cm3    = float(run_params.get("poison_volume_simulated_cm3", 0.0))

    if poison_mat_id_str and poison_mat_id_str in tracked_mat_ids and poison_vol_cm3 > 0:
        poison_densities = {}
        for nuc in all_nuclides:
            if xs_nuclides is not None and nuc not in xs_nuclides:
                continue
            try:
                _, atoms = results.get_atoms(poison_mat_id_str, nuc)
                n_at = float(atoms[step_idx])
                if n_at > 0:
                    poison_densities[nuc] = n_at / poison_vol_cm3
            except Exception:
                pass
        if poison_densities:
            depleted['poison'] = poison_densities
            print(f"  Reconstructed burnable poison (mat {poison_mat_id_str}): "
                  f"{len(poison_densities)} nuclides")
        else:
            print(f"  WARNING: no poison nuclide data found for mat {poison_mat_id_str}")
    else:
        print(f"  WARNING: burnable poison not found in depletion results "
              f"(id={poison_mat_id_str!r}, vol={poison_vol_cm3:.1f} cm3)")

    return depleted, fuel_mat_ids_2d, run_params, step_time_days


# ============================================================================
# STEP 2 — INJECT DEPLETED COMPOSITIONS INTO A FRESHLY BUILT MODEL
# ============================================================================

def _inject_depleted_materials(fuel_clones, depleted, model=None):
    """
    Overwrite fuel-clone nuclide compositions with depleted atom densities.
    Also injects the depleted burnable-poison (B4C) composition if present.

    Called AFTER build_model() returns fuel_clones but BEFORE
    model.export_to_xml() is called, so the modified compositions are
    written to the XML files used by the OpenMC run.

    Parameters
    ----------
    fuel_clones : list[list[openmc.Material]]
        From build_model() — [ring_idx][bax_idx].
    depleted : dict
        From reconstruct_depleted_materials() — (ring, bax) -> {nuc: dens},
        plus optional 'poison' key for the burnable poison material.
    model : openmc.model.Model, optional
        The freshly-built model; used to locate the B4C_Poison material.
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

    # ── Burnable poison ────────────────────────────────────────────────────
    if 'poison' in depleted and model is not None:
        poison_mat = next(
            (m for m in model.materials if m.name == "B4C_Poison"), None
        )
        if poison_mat is not None:
            _overwrite_mat(poison_mat, depleted['poison'])
            print(f"  Injected depleted composition into B4C_Poison material")
        else:
            print(f"  WARNING: 'poison' key in depleted but no B4C_Poison "
                  f"material found in model — poison NOT injected")
    elif 'poison' in depleted and model is None:
        print(f"  WARNING: 'poison' key in depleted but model not passed — "
              f"B4C_Poison NOT injected")


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
        # Also block writing *inside* the depletion run dir unless it's the
        # dedicated mol_eol_analysis_* subdirectory pattern.
        if run_real.startswith(dep_real + os.sep):
            tail = os.path.relpath(run_real, dep_real)
            if not tail.startswith("mol_eol_analysis"):
                raise RuntimeError(
                    f"MOL/EOL run_dir is inside the depletion_run_dir "
                    f"but not in a 'mol_eol_analysis_*' subfolder.\n"
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
    from MicroHTGRNeutronics_INL_HTGTR_Inspired import build_model

    _assert_new_dir(run_dir, depletion_run_dir)
    os.makedirs(run_dir, exist_ok=True)
    model, n_trisos, m_colors, fuel_clones = build_model(params, run_dir)

    print(f"\n  Injecting depleted compositions...")
    _inject_depleted_materials(fuel_clones, depleted, model=model)

    model.export_to_xml()

    import shutil
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
    """Read k_eff and std_dev from the statepoint in run_dir."""
    import glob
    sp_files = sorted(glob.glob(os.path.join(run_dir, "statepoint.*.h5")))
    if not sp_files:
        raise FileNotFoundError(f"No statepoint file in {run_dir}")
    sp = openmc.StatePoint(sp_files[-1])
    return float(sp.keff.n), float(sp.keff.s)


# ============================================================================
# STEP 3 — CRITICAL ROD INSERTION BINARY SEARCH
# ============================================================================

def find_critical_rod_insertion(
    params,
    depleted,
    output_dir,
    depletion_run_dir=None,
    k_target=1.0,
    k_tol=0.003,
    max_iter=20,
):
    """
    Two-stage binary search to find the critical rod position.

    Stage 0 — quick check: run with bank 1 fully inserted, bank 2 at 0.
    Stage 1 — binary search:
      * If stage-0 k < k_target: search bank 1 in [0, 1] with bank 2 = 0.
      * If stage-0 k > k_target: fix bank 1 = 1.0, search bank 2 in [0, 1].
    Bank 3 is always left at 0 (unused legacy bank).

    Parameters
    ----------
    params : dict
        Simulation parameters (will be deep-copied; original is unchanged).
    depleted : dict
        Depleted material compositions from reconstruct_depleted_materials().
    output_dir : str
        Root directory for binary-search sub-runs.
    k_target : float
        Target k_eff (default 1.0 — critical).
    k_tol : float
        Convergence tolerance on |k_eff - k_target|.
    max_iter : int
        Maximum binary-search iterations per stage.

    Returns
    -------
    dict with keys:
        critical_bank_1    : float   — final bank 1 insertion fraction
        critical_bank_2    : float   — final bank 2 insertion fraction
        critical_insertion : float   — insertion of the active search bank
        search_stage       : str     — "bank1" or "bank2"
        critical_keff      : float
        critical_keff_std  : float
        critical_run_dir   : str
        converged          : bool
        n_iterations       : int
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'─' * 70}")
    print(f"  CRITICAL ROD SEARCH — 2-stage (bank 1 then bank 2)")
    print(f"  Target: k_eff = {k_target:.4f}  ±  {k_tol:.4f}")
    print(f"{'─' * 70}")

    # Reduced particle settings for all search iterations
    search_params = copy.deepcopy(params)
    search_params["total_batches"]       = 100
    search_params["inactive_batches"]    = 40
    search_params["particles"]           = max(50_000, params.get("particles", 100_000) // 2)
    search_params["make_geometry_plots"] = False
    search_params["use_mesh_tallies"]    = False
    search_params["use_BeO_tallies"]     = False
    search_params["use_leakage_tallies"] = False
    search_params["use_global_tallies"]  = False

    # ── Stage 0: bank 1 fully inserted, bank 2 out ──────────────────────────
    stage0_params = copy.deepcopy(search_params)
    stage0_params["bank_1_insertion"] = 1.0
    stage0_params["bank_2_insertion"] = 0.0
    stage0_params["bank_3_insertion"] = 0.0
    stage0_dir = os.path.join(output_dir, "stage0_bank1_full_bank2_out")

    print(f"\n  Stage 0: bank 1 = 1.0, bank 2 = 0.0")
    _run_eigenvalue_with_depleted(stage0_params, depleted, stage0_dir,
                                  depletion_run_dir=depletion_run_dir)
    k0, k0_std = _read_keff(stage0_dir)
    print(f"  k_eff = {k0:.5f} ± {k0_std:.5f}   "
          f"(Δ = {(k0 - k_target)*1e5:+.0f} pcm)")

    if abs(k0 - k_target) < k_tol:
        print(f"\n  ✓ Converged at stage 0: bank 1 = 1.0, bank 2 = 0.0")
        return {
            "critical_bank_1":   1.0,
            "critical_bank_2":   0.0,
            "critical_insertion": 1.0,
            "search_stage":      "bank1",
            "critical_keff":     k0,
            "critical_keff_std": k0_std,
            "critical_run_dir":  stage0_dir,
            "converged":         True,
            "n_iterations":      1,
        }

    # ── Stage 1: choose which bank to search ────────────────────────────────
    if k0 < k_target:
        # Subcritical even with bank 1 fully inserted → search bank 1 in [0,1]
        active_bank  = "bank_1"
        fixed_b1     = None     # will be set per iteration
        fixed_b2     = 0.0
        search_stage = "bank1"
        print(f"\n  k < 1 with bank 1 full → binary search on bank 1 (bank 2 = 0.0)")
    else:
        # Supercritical with bank 1 fully inserted → fix bank 1, search bank 2
        active_bank  = "bank_2"
        fixed_b1     = 1.0
        fixed_b2     = None     # will be set per iteration
        search_stage = "bank2"
        print(f"\n  k > 1 with bank 1 full → binary search on bank 2 (bank 1 = 1.0)")

    lo, hi = 0.0, 1.0
    best_dir = None
    best_ins = None
    best_k   = None
    best_std = None
    converged = False

    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        iter_params = copy.deepcopy(search_params)
        iter_params["bank_3_insertion"] = 0.0
        if active_bank == "bank_1":
            iter_params["bank_1_insertion"] = mid
            iter_params["bank_2_insertion"] = fixed_b2
        else:
            iter_params["bank_1_insertion"] = fixed_b1
            iter_params["bank_2_insertion"] = mid

        case_dir = os.path.join(output_dir,
                                f"search_{search_stage}_iter{i+1:02d}_ins{mid:.4f}")
        print(f"\n  Iter {i+1:2d}: {active_bank} = {mid:.4f}  [{lo:.4f}, {hi:.4f}]")

        _run_eigenvalue_with_depleted(iter_params, depleted, case_dir,
                                      depletion_run_dir=depletion_run_dir)
        k, k_std = _read_keff(case_dir)
        print(f"         k_eff = {k:.5f} ± {k_std:.5f}   "
              f"(Δ = {(k - k_target)*1e5:+.0f} pcm)")

        if best_k is None or abs(k - k_target) < abs(best_k - k_target):
            best_k   = k
            best_std = k_std
            best_ins = mid
            best_dir = case_dir

        if abs(k - k_target) < k_tol:
            converged = True
            print(f"\n  ✓ Converged: {active_bank} = {mid:.4f}, "
                  f"k = {k:.5f} ± {k_std:.5f}")
            break

        # k > target → more absorption needed → increase insertion
        if k > k_target:
            lo = mid
        else:
            hi = mid

    if not converged:
        print(f"\n  WARNING: Did not converge in {max_iter} iterations.")
        print(f"  Best: {active_bank} = {best_ins:.4f}, k = {best_k:.5f}")

    if search_stage == "bank1":
        final_b1, final_b2 = best_ins, 0.0
    else:
        final_b1, final_b2 = 1.0, best_ins

    return {
        "critical_bank_1":   final_b1,
        "critical_bank_2":   final_b2,
        "critical_insertion": best_ins,
        "search_stage":      search_stage,
        "critical_keff":     best_k,
        "critical_keff_std": best_std,
        "critical_run_dir":  best_dir,
        "converged":         converged,
        "n_iterations":      i + 1,
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
    full_params["inactive_batches"]    = 200

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

        if abs(k_check - 1.0) < k_tol:
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
