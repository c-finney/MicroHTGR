"""
Reactivity Coefficient Calculation via Direct Perturbation Method

Calculates the following temperature reactivity coefficients:
  - Fuel Temperature Coefficient (FTC / Doppler coefficient)
  - Moderator Temperature Coefficient (MTC)
  - Isothermal Temperature Coefficient (ITC)

Method: For each coefficient, the simulation is re-run at perturbed temperatures
and the reactivity difference is computed as:
    α = Δρ / ΔT = [(k₂ - k₁) / (k₁ · k₂)] / (T₂ - T₁)   [pcm/K]

Usage:
    # As a module (from the main simulation script):
    from reactivity_coefficients import run_reactivity_coefficients
    run_reactivity_coefficients(params, core_rings, base_run_dir, output_base_dir)

    # Standalone:
    python reactivity_coefficients.py
"""

import os
import sys
import copy
import json
import math
import numpy as np
from datetime import datetime

from reactivity_coefficients_postprocessing import save_results as _save_results, plot_results as _plot_results

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reactivity_pcm(k):
    """Convert k-effective to reactivity in pcm."""
    return (k - 1.0) / k * 1e5


def _alpha(k1, k2, dT):
    """
    Compute temperature coefficient α = Δρ/ΔT in pcm/K.

    Parameters
    ----------
    k1, k2 : float
        k-effective at temperatures T1 and T2 = T1 + dT.
    dT : float
        Temperature perturbation (K).

    Returns
    -------
    float : reactivity coefficient in pcm/K.
    """
    rho1 = _reactivity_pcm(k1)
    rho2 = _reactivity_pcm(k2)
    return (rho2 - rho1) / dT


def _alpha_uncertainty(k1, k1_std, k2, k2_std, dT):
    """
    Propagated 1σ uncertainty on α = Δρ/ΔT.

    Uses standard error propagation assuming k1 and k2 are independent.
    """
    # ∂ρ/∂k = 1/k² (in pcm units, multiply by 1e5)
    sig_rho1 = k1_std / (k1 ** 2) * 1e5
    sig_rho2 = k2_std / (k2 ** 2) * 1e5
    sig_alpha = math.sqrt(sig_rho1 ** 2 + sig_rho2 ** 2) / abs(dT)
    return sig_alpha


def _extract_keff(run_dir):
    """Extract k-effective from statepoint in a completed run directory."""
    import openmc

    for f in os.listdir(run_dir):
        if f.startswith("statepoint") and f.endswith(".h5"):
            sp = openmc.StatePoint(os.path.join(run_dir, f))
            return sp.keff.nominal_value, sp.keff.std_dev
    raise FileNotFoundError(f"No statepoint file found in {run_dir}")


# ---------------------------------------------------------------------------
# Temperature perturbation builders
# ---------------------------------------------------------------------------

def _build_fuel_perturbed_params(base_params, delta_T):
    """
    Perturb FUEL temperatures only (Doppler coefficient).

    Shifts compact_min, compact_max by delta_T.
    Everything else stays at nominal.
    """
    p = copy.deepcopy(base_params)
    p["compact_min"] += delta_T
    p["compact_max"] += delta_T
    return p


def _build_moderator_perturbed_params(base_params, delta_T):
    """
    Perturb MODERATOR / GRAPHITE temperatures only (MTC).

    Shifts matrix_min, matrix_max, and reflector_min/reflector_max (if present)
    by delta_T.  Fuel compact and coolant temperatures stay at nominal.
    """
    p = copy.deepcopy(base_params)
    p["matrix_min"] += delta_T
    p["matrix_max"] += delta_T
    if "reflector_min" in p:
        p["reflector_min"] += delta_T
    if "reflector_max" in p:
        p["reflector_max"] += delta_T
    return p


def _build_isothermal_perturbed_params(base_params, delta_T):
    """
    Perturb ALL temperatures uniformly (ITC).

    Shifts coolant, compact, matrix, and reflector temperatures by delta_T.
    """
    p = copy.deepcopy(base_params)
    p["coolant_inlet"] += delta_T
    p["coolant_outlet"] += delta_T
    p["compact_min"] += delta_T
    p["compact_max"] += delta_T
    p["matrix_min"] += delta_T
    p["matrix_max"] += delta_T
    if "reflector_min" in p:
        p["reflector_min"] += delta_T
    if "reflector_max" in p:
        p["reflector_max"] += delta_T
    return p


# Mapping from coefficient name to perturbation builder
PERTURBATION_BUILDERS = {
    "FTC": _build_fuel_perturbed_params,
    "MTC": _build_moderator_perturbed_params,
    "ITC": _build_isothermal_perturbed_params,
}

COEFF_LABELS = {
    "FTC": "Fuel Temperature Coefficient (Doppler)",
    "MTC": "Moderator Temperature Coefficient",
    "ITC": "Isothermal Temperature Coefficient",
}

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_reactivity_coefficients(
    params,
    base_run_dir,
    output_base_dir,
    delta_T_values=None,
    coefficients=None,
    run_simulation_fn=None,
    run_post_processing_fn=None,
):
    """
    Calculate reactivity coefficients via direct perturbation.

    For each requested coefficient the code:
      1. Accepts or runs a reference (nominal) case.
      2. Runs perturbed cases at T_nom ± ΔT for each ΔT in delta_T_values.
      3. Computes α = Δρ / ΔT at each perturbation and reports the average.

    Parameters
    ----------
    params : dict
        Nominal simulation parameters (from config.py).
    core_rings : list
        Core ring layout (from config.py).
    base_run_dir : str
        Path to a completed nominal (reference) run directory.  If the
        directory contains a valid statepoint the reference k_eff is read
        from it; otherwise the nominal case is re-run.
    output_base_dir : str
        Root directory under which perturbed-case directories are created.
    delta_T_values : list[float], optional
        List of temperature perturbations (K) to apply.  Both +ΔT and -ΔT
        are run for each value to allow central-difference estimates.
        Default: [50, 100, 150].
    coefficients : list[str], optional
        Which coefficients to compute.  Any subset of ["FTC", "MTC", "ITC"].
        Default: all three.
    run_simulation_fn : callable, optional
        The function that runs an OpenMC simulation.  Signature must be
        ``run_simulation_fn(params, core_rings, run_dir) -> n_trisos``.
        If None, the function is imported from the main script.
    run_post_processing_fn : callable, optional
        Post-processing function with signature
        ``run_post_processing_fn(run_dir, params, n_trisos)``.

    Returns
    -------
    dict : Nested dictionary of results keyed by coefficient name.
    """

    if delta_T_values is None:
        delta_T_values = [50.0, 100.0, 150.0]
    if coefficients is None:
        coefficients = ["FTC", "MTC", "ITC"]

    # ------------------------------------------------------------------
    # Import simulation driver if not provided
    # ------------------------------------------------------------------
    if run_simulation_fn is None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, SCRIPT_DIR)
        from MicroHTGRNeutronics_INL_HTGTR_Inspired import run_simulation as _run_sim
        from MicroHTGRNeutronics_INL_HTGTR_Inspired import run_post_processing as _run_pp
        run_simulation_fn = _run_sim
        if run_post_processing_fn is None:
            run_post_processing_fn = _run_pp

    # ------------------------------------------------------------------
    # 1. Reference case
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("REACTIVITY COEFFICIENT CALCULATION — DIRECT PERTURBATION")
    print("=" * 80)

    # Try to read reference k_eff from existing run
    try:
        k_ref, k_ref_std = _extract_keff(base_run_dir)
        print(f"\nReference case found in: {base_run_dir}")
        print(f"  k_ref = {k_ref:.5f} ± {k_ref_std:.5f}")
    except (FileNotFoundError, Exception) as e:
        print(f"\nNo valid reference run found ({e}).  Running nominal case…")
        ref_dir = os.path.join(output_base_dir, "reference_nominal")
        n_trisos = run_simulation_fn(params, ref_dir)
        if run_post_processing_fn:
            run_post_processing_fn(ref_dir, params, n_trisos)
        k_ref, k_ref_std = _extract_keff(ref_dir)
        base_run_dir = ref_dir
        print(f"  k_ref = {k_ref:.5f} ± {k_ref_std:.5f}")

    rho_ref = _reactivity_pcm(k_ref)

    # ------------------------------------------------------------------
    # 2. Perturbed cases
    # ------------------------------------------------------------------
    all_results = {}

    for coeff_name in coefficients:
        builder = PERTURBATION_BUILDERS[coeff_name]
        label = COEFF_LABELS[coeff_name]

        print(f"\n{'─' * 80}")
        print(f"  Computing {label}")
        print(f"{'─' * 80}")

        coeff_dir = os.path.join(output_base_dir, f"perturbation_{coeff_name}")
        os.makedirs(coeff_dir, exist_ok=True)

        case_results = []

        # Run both positive and negative perturbations
        for dT in delta_T_values:
            for sign, sign_label in [(+1, "pos"), (-1, "neg")]:
                actual_dT = sign * dT
                case_label = f"{coeff_name}_dT_{sign_label}{int(dT)}K"
                case_dir = os.path.join(coeff_dir, case_label)

                perturbed_params = builder(params, actual_dT)

                # Check if already computed
                try:
                    k_pert, k_pert_std = _extract_keff(case_dir)
                    print(f"  [{case_label}] Already computed — k = {k_pert:.5f}")
                except Exception:
                    print(f"  [{case_label}] Running ΔT = {actual_dT:+.0f} K …")
                    n_trisos = run_simulation_fn(perturbed_params, case_dir)
                    if run_post_processing_fn:
                        run_post_processing_fn(case_dir, perturbed_params, n_trisos)
                    k_pert, k_pert_std = _extract_keff(case_dir)
                    print(f"  [{case_label}] k = {k_pert:.5f} ± {k_pert_std:.5f}")

                alpha_val = _alpha(k_ref, k_pert, actual_dT)
                alpha_unc = _alpha_uncertainty(k_ref, k_ref_std, k_pert, k_pert_std, actual_dT)

                case_results.append({
                    "delta_T": actual_dT,
                    "k_pert": k_pert,
                    "k_pert_std": k_pert_std,
                    "rho_pert": _reactivity_pcm(k_pert),
                    "alpha_pcm_per_K": alpha_val,
                    "alpha_std": alpha_unc,
                    "case_dir": case_dir,
                })

        # Central-difference estimates (pair +dT and -dT)
        central_diff_results = []
        for dT in delta_T_values:
            pos = next((c for c in case_results if c["delta_T"] == +dT), None)
            neg = next((c for c in case_results if c["delta_T"] == -dT), None)
            if pos and neg:
                alpha_cd = _alpha(neg["k_pert"], pos["k_pert"], 2 * dT)
                alpha_cd_unc = _alpha_uncertainty(
                    neg["k_pert"], neg["k_pert_std"],
                    pos["k_pert"], pos["k_pert_std"],
                    2 * dT,
                )
                central_diff_results.append({
                    "delta_T": dT,
                    "alpha_central_pcm_per_K": alpha_cd,
                    "alpha_central_std": alpha_cd_unc,
                })

        # Average central-difference coefficient
        if central_diff_results:
            avg_alpha = np.mean([c["alpha_central_pcm_per_K"] for c in central_diff_results])
            avg_alpha_std = np.sqrt(np.sum([c["alpha_central_std"] ** 2 for c in central_diff_results])) / len(central_diff_results)
        else:
            avg_alpha = np.mean([c["alpha_pcm_per_K"] for c in case_results])
            avg_alpha_std = np.sqrt(np.sum([c["alpha_std"] ** 2 for c in case_results])) / len(case_results)

        all_results[coeff_name] = {
            "label": label,
            "cases": case_results,
            "central_difference": central_diff_results,
            "average_alpha_pcm_per_K": avg_alpha,
            "average_alpha_std": avg_alpha_std,
        }

    # ------------------------------------------------------------------
    # 3. Report
    # ------------------------------------------------------------------
    _print_summary(all_results, k_ref, k_ref_std, rho_ref)
    _save_results(all_results, k_ref, k_ref_std, rho_ref, output_base_dir)
    _plot_results(all_results, k_ref, output_base_dir)

    return all_results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_summary(all_results, k_ref, k_ref_std, rho_ref):
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 80)
    print("REACTIVITY COEFFICIENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"Reference k_eff = {k_ref:.5f} ± {k_ref_std:.5f}  (ρ = {rho_ref:.1f} pcm)")

    for name, res in all_results.items():
        print(f"\n{'─' * 60}")
        print(f"  {res['label']}")
        print(f"{'─' * 60}")

        print(f"  {'ΔT (K)':>10}  {'k_pert':>10}  {'ρ_pert (pcm)':>14}  {'α (pcm/K)':>12}  {'± σ':>10}")
        for c in res["cases"]:
            print(
                f"  {c['delta_T']:>+10.0f}  {c['k_pert']:>10.5f}  "
                f"{c['rho_pert']:>14.1f}  {c['alpha_pcm_per_K']:>12.3f}  "
                f"{c['alpha_std']:>10.3f}"
            )

        if res["central_difference"]:
            print(f"\n  Central-difference estimates:")
            print(f"  {'±ΔT (K)':>10}  {'α_cd (pcm/K)':>14}  {'± σ':>10}")
            for cd in res["central_difference"]:
                print(
                    f"  {cd['delta_T']:>10.0f}  "
                    f"{cd['alpha_central_pcm_per_K']:>14.3f}  "
                    f"{cd['alpha_central_std']:>10.3f}"
                )

        print(f"\n  ▶ Average α = {res['average_alpha_pcm_per_K']:.3f} ± {res['average_alpha_std']:.3f} pcm/K")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Standalone usage example.

    Expects the simulation modules (config.py, materials.py, assembly.py,
    trisos.py, MicroHTGRNeutronics_INL_HTGTR_Inspired.py) to be on the
    Python path or in the same directory.
    """
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, SCRIPT_DIR)

    import config as cfg
    from MicroHTGRNeutronics_INL_HTGTR_Inspired import run_simulation, run_post_processing

    now = datetime.now()
    run_name = f"htgr_reactivity_coeffs_{now.strftime('%m.%d.%Y_%H.%M.%S')}"

    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    OUTPUT_BASE = os.path.join(PARENT_DIR, "MicroHTGR_Output")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    output_dir = os.path.join(OUTPUT_BASE, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # If you have an existing reference run, point to it here:
    # base_run_dir = "/path/to/existing/nominal/run"
    # Otherwise, set to the output_dir and it will run a fresh reference case.
    base_run_dir = output_dir

    print(f"\nOutput directory: {output_dir}")
    print(f"Reference run directory: {base_run_dir}\n")

    results = run_reactivity_coefficients(
        params=cfg.params,
        base_run_dir=base_run_dir,
        output_base_dir=output_dir,
        delta_T_values=[50.0, 100.0, 150.0],
        coefficients=["FTC", "MTC", "ITC"],
        run_simulation_fn=run_simulation,
        run_post_processing_fn=run_post_processing,
    )

    print("\nDone.")