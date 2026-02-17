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
import matplotlib.pyplot as plt
from datetime import datetime

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

    Shifts matrix_min, matrix_max, reflector_min, reflector_max by delta_T.
    Fuel compact and coolant temperatures stay at nominal.
    """
    p = copy.deepcopy(base_params)
    p["matrix_min"] += delta_T
    p["matrix_max"] += delta_T
    p["reflector_min"] += delta_T
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
    p["reflector_min"] += delta_T
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
    core_rings,
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
        n_trisos = run_simulation_fn(params, core_rings, ref_dir)
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
                    n_trisos = run_simulation_fn(perturbed_params, core_rings, case_dir)
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


def _save_results(all_results, k_ref, k_ref_std, rho_ref, output_dir):
    """Save results to JSON and text files."""

    # --- JSON ---
    json_data = {
        "reference": {
            "k_eff": k_ref,
            "k_eff_std": k_ref_std,
            "reactivity_pcm": rho_ref,
        },
        "coefficients": {},
    }

    for name, res in all_results.items():
        json_data["coefficients"][name] = {
            "label": res["label"],
            "average_alpha_pcm_per_K": res["average_alpha_pcm_per_K"],
            "average_alpha_std": res["average_alpha_std"],
            "cases": [
                {
                    "delta_T": c["delta_T"],
                    "k_pert": c["k_pert"],
                    "k_pert_std": c["k_pert_std"],
                    "rho_pert": c["rho_pert"],
                    "alpha_pcm_per_K": c["alpha_pcm_per_K"],
                    "alpha_std": c["alpha_std"],
                }
                for c in res["cases"]
            ],
            "central_difference": res["central_difference"],
        }

    json_path = os.path.join(output_dir, "reactivity_coefficients.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # --- Human-readable text ---
    txt_path = os.path.join(output_dir, "reactivity_coefficients.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("REACTIVITY COEFFICIENTS — DIRECT PERTURBATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Reference k_eff = {k_ref:.5f} ± {k_ref_std:.5f}\n")
        f.write(f"Reference reactivity = {rho_ref:.1f} pcm\n\n")

        for name, res in all_results.items():
            f.write(f"{res['label']}:\n")
            f.write(f"  α = {res['average_alpha_pcm_per_K']:.3f} ± {res['average_alpha_std']:.3f} pcm/K\n\n")
            for c in res["cases"]:
                f.write(
                    f"  ΔT = {c['delta_T']:+.0f} K  |  "
                    f"k = {c['k_pert']:.5f}  |  "
                    f"α = {c['alpha_pcm_per_K']:.3f} pcm/K\n"
                )
            f.write("\n")
        f.write("=" * 80 + "\n")
    print(f"Results saved to: {txt_path}")


def _plot_results(all_results, k_ref, output_dir):
    """Generate publication-quality plots of reactivity vs. temperature perturbation."""

    for name, res in all_results.items():
        cases = res["cases"]
        dTs = np.array([c["delta_T"] for c in cases])
        rhos = np.array([c["rho_pert"] for c in cases])
        k_stds = np.array([c["k_pert_std"] for c in cases])
        ks = np.array([c["k_pert"] for c in cases])

        rho_ref = _reactivity_pcm(k_ref)
        delta_rho = rhos - rho_ref

        # Propagated uncertainty on Δρ
        sig_rho_ref = 0  # reference uncertainty already embedded in individual α uncertainties
        sig_rhos = k_stds / (ks ** 2) * 1e5

        # Sort by ΔT for clean line plot
        sort_idx = np.argsort(dTs)
        dTs = dTs[sort_idx]
        delta_rho = delta_rho[sort_idx]
        sig_rhos = sig_rhos[sort_idx]

        # Linear fit
        coeffs = np.polyfit(dTs, delta_rho, 1)
        fit_line = np.polyval(coeffs, dTs)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.errorbar(dTs, delta_rho, yerr=sig_rhos, fmt="o", capsize=4, capthick=1.2,
                     markersize=8, label="Perturbed cases", zorder=3)
        ax.plot(dTs, fit_line, "--", color="tab:red", linewidth=1.5,
                label=f"Linear fit: α = {coeffs[0]:.3f} pcm/K", zorder=2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")

        ax.set_xlabel("Temperature Perturbation ΔT (K)", fontsize=12)
        ax.set_ylabel("Δρ (pcm)", fontsize=12)
        ax.set_title(f"{res['label']}\nα = {res['average_alpha_pcm_per_K']:.3f} ± {res['average_alpha_std']:.3f} pcm/K",
                      fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, f"reactivity_coefficient_{name}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {save_path}")

    # --- Combined bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    names = list(all_results.keys())
    alphas = [all_results[n]["average_alpha_pcm_per_K"] for n in names]
    stds = [all_results[n]["average_alpha_std"] for n in names]
    labels = [all_results[n]["label"].split("(")[0].strip() for n in names]

    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(labels, alphas, yerr=stds, capsize=5, color=colors[:len(names)],
                  edgecolor="black", linewidth=0.8, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("α (pcm/K)", fontsize=12)
    ax.set_title("Temperature Reactivity Coefficients", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val, std in zip(bars, alphas, stds):
        y_pos = bar.get_height() + std + 0.1
        if val < 0:
            y_pos = bar.get_height() - std - 0.3
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    save_path = os.path.join(output_dir, "reactivity_coefficients_summary.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Summary plot saved: {save_path}")


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
        core_rings=cfg.core_rings,
        base_run_dir=base_run_dir,
        output_base_dir=output_dir,
        delta_T_values=[50.0, 100.0, 150.0],
        coefficients=["FTC", "MTC", "ITC"],
        run_simulation_fn=run_simulation,
        run_post_processing_fn=run_post_processing,
    )

    print("\nDone.")