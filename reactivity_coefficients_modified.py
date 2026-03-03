"""
Reactivity Coefficient Calculation via Direct Perturbation Method

Calculates:
  - Fuel Temperature Coefficient (FTC): perturbs compact temperatures only
  - Moderator Temperature Coefficient (MTC): perturbs graphite matrix + reflector temperatures only
  - Isothermal Temperature Coefficient (ITC): perturbs all temperatures uniformly

Method:
  For each coefficient, the simulation is run at T_ref ± ΔT. The reactivity
  coefficient is then computed via central difference:

      α = (ρ₊ - ρ₋) / (T₊ - T₋)      [pcm/K]

  where ρ = (k - 1) / k × 10⁵

Usage:
  - As a standalone driver:
      python reactivity_coefficients.py

  - Post-process an existing study:
      python reactivity_coefficients.py <study_directory> <delta_T>

  - Integrated into the main simulation by adding to config.py:
      reactivity_study = True
      reactivity_delta_T = 50.0   # Kelvin
"""

import os
import sys
import json
import copy
import numpy as np
from datetime import datetime

# ====================================================================================================
# HELPER FUNCTIONS
# ====================================================================================================

def keff_to_reactivity_pcm(k):
    """Convert k-effective to reactivity in pcm."""
    return (k - 1.0) / k * 1e5


def extract_keff(run_dir):
    """
    Extract k-effective from a completed OpenMC run directory.

    Returns:
        tuple: (keff, keff_std) or (None, None) on failure
    """
    try:
        import openmc
        for f in os.listdir(run_dir):
            if f.startswith('statepoint') and f.endswith('.h5'):
                sp = openmc.StatePoint(os.path.join(run_dir, f))
                return sp.keff.nominal_value, sp.keff.std_dev
    except Exception as e:
        print(f"  ERROR extracting k-eff from {run_dir}: {e}")
    return None, None

# ====================================================================================================
# TEMPERATURE PERTURBATION FUNCTIONS
# ====================================================================================================

def perturb_fuel_temperatures(params, delta_T):
    """
    Apply a uniform shift to fuel compact temperatures only.

    Perturbed parameters:
        compact_min, compact_max
    """
    p = copy.deepcopy(params)
    p["compact_min"] += delta_T
    p["compact_max"] += delta_T
    return p

def perturb_moderator_temperatures(params, delta_T):
    """
    Apply a uniform shift to moderator/reflector temperatures only.

    Perturbed parameters:
        matrix_min, matrix_max, reflector_min, reflector_max
    """
    p = copy.deepcopy(params)
    p["matrix_min"] += delta_T
    p["matrix_max"] += delta_T
    p["reflector_min"] += delta_T
    p["reflector_max"] += delta_T
    return p

def perturb_isothermal_temperatures(params, delta_T):
    """
    Apply a uniform shift to ALL temperature parameters (fuel, moderator,
    coolant, reflector) — i.e. the entire system temperature moves together.
    """
    p = copy.deepcopy(params)
    p["coolant_inlet"] += delta_T
    p["coolant_outlet"] += delta_T
    p["compact_min"] += delta_T
    p["compact_max"] += delta_T
    p["matrix_min"] += delta_T
    p["matrix_max"] += delta_T
    p["reflector_min"] += delta_T
    p["reflector_max"] += delta_T
    return p

# Map of coefficient type -> perturbation function and readable name
COEFF_REGISTRY = {
    "FTC": {
        "name": "Fuel Temperature Coefficient",
        "perturb_fn": perturb_fuel_temperatures,
        "description": "Perturbs compact_min, compact_max",
    },
    "MTC": {
        "name": "Moderator Temperature Coefficient",
        "perturb_fn": perturb_moderator_temperatures,
        "description": "Perturbs matrix_min/max, reflector_min/max",
    },
    "ITC": {
        "name": "Isothermal Temperature Coefficient",
        "perturb_fn": perturb_isothermal_temperatures,
        "description": "Perturbs all temperatures uniformly",
    },
}

# ====================================================================================================
# CONFIGURE PERTURBATION STUDY CASES
# ====================================================================================================

def setup_perturbation_cases(base_params, delta_T, base_dir,
                             coefficients=("FTC", "MTC", "ITC"),
                             run_baseline=True):
    """
    Set up directory structure and parameter sets for all perturbation cases.

    For each coefficient type, creates:
        <coeff>_plus_<dT>K/    (T_ref + dT)
        <coeff>_minus_<dT>K/   (T_ref - dT)

    Plus one shared baseline directory.

    Parameters
    ------------
    base_params : dict
        Reference simulation parameters.
    delta_T : float
        Temperature perturbation magnitude in Kelvin.
    base_dir : str
        Root output directory for the study.
    coefficients : tuple of str
        Which coefficients to compute ("FTC", "MTC", "ITC").
    run_baseline : bool
        If True, include a shared baseline case.

    Returns
    --------
    list of dict
        Each dict describes a case:
        {"label", "coeff", "variant", "params", "run_dir", "delta_T"}
    """
    os.makedirs(base_dir, exist_ok=True)
    cases = []

    # Shared baseline (same for all coefficients since ref temps are identical)
    if run_baseline:
        cases.append({
            "label": "Baseline (T_ref)",
            "coeff": "baseline",
            "variant": "baseline",
            "params": copy.deepcopy(base_params),
            "run_dir": os.path.join(base_dir, "baseline"),
            "delta_T": 0.0,
        })

    for coeff_key in coefficients:
        entry = COEFF_REGISTRY[coeff_key]
        perturb_fn = entry["perturb_fn"]

        # Plus perturbation
        cases.append({
            "label": f"{coeff_key} +{delta_T:.0f} K",
            "coeff": coeff_key,
            "variant": "plus",
            "params": perturb_fn(base_params, +delta_T),
            "run_dir": os.path.join(base_dir, f"{coeff_key}_plus_{delta_T:.0f}K"),
            "delta_T": +delta_T,
        })

        # Minus perturbation
        cases.append({
            "label": f"{coeff_key} -{delta_T:.0f} K",
            "coeff": coeff_key,
            "variant": "minus",
            "params": perturb_fn(base_params, -delta_T),
            "run_dir": os.path.join(base_dir, f"{coeff_key}_minus_{delta_T:.0f}K"),
            "delta_T": -delta_T,
        })

    return cases

# ====================================================================================================
# RUN PERTURBTION STUDIES
# ====================================================================================================

def run_perturbation_study(base_params, delta_T, base_dir,
                           coefficients=("FTC", "MTC", "ITC"),
                           run_simulation_fn=None,
                           skip_existing=True):
    """
    Run the full direct-perturbation reactivity coefficient study.

    Parameters
    ------------
    base_params : dict
        Reference simulation parameters.
    delta_T : float
        Temperature perturbation in Kelvin (positive value).
    base_dir : str
        Root output directory.
    coefficients : tuple of str
        Which coefficients to compute.
    run_simulation_fn : callable
        Function with signature run_simulation(params, run_dir).
        If None, imports from MicroHTGRNeutronics_INL_HTGTR_Inspired.
    skip_existing : bool
        If True, skip cases where a statepoint file already exists.

    Returns
    --------
    dict : Mapping from coefficient key to result dict.
    """
    if run_simulation_fn is None:
        from MicroHTGRNeutronics_INL_HTGTR_Inspired import run_simulation
        run_simulation_fn = run_simulation

    cases = setup_perturbation_cases(
        base_params, delta_T, base_dir,
        coefficients=coefficients, run_baseline=True
    )

    # ----- Run all cases -----
    n_cases = len(cases)
    print(f"\n{'='*80}")
    print(f"REACTIVITY COEFFICIENT STUDY - Direct Perturbation")
    print(f"dT = +/-{delta_T:.0f} K | Coefficients: {', '.join(coefficients)}")
    print(f"Output directory: {base_dir}")
    print(f"Total cases to run: {n_cases}")
    print(f"{'='*80}\n")

    for i, case in enumerate(cases):
        run_dir = case["run_dir"]
        label = case["label"]

        # Check for existing results
        has_statepoint = False
        if os.path.isdir(run_dir):
            for f in os.listdir(run_dir):
                if f.startswith("statepoint") and f.endswith(".h5"):
                    has_statepoint = True
                    break

        if skip_existing and has_statepoint:
            print(f"[{i+1}/{n_cases}] {label} - SKIPPED (statepoint exists)")
            continue

        print(f"\n[{i+1}/{n_cases}] Running: {label}")
        print(f"  Directory: {run_dir}")

        # Log perturbed temperatures for traceability
        p = case["params"]
        print(f"  Fuel temps:      {p['compact_min']:.1f} - {p['compact_max']:.1f} K")
        print(f"  Matrix temps:    {p['matrix_min']:.1f} - {p['matrix_max']:.1f} K")
        print(f"  Coolant temps:   {p['coolant_inlet']:.1f} - {p['coolant_outlet']:.1f} K")
        print(f"  Reflector temps: {p['reflector_min']:.1f} - {p['reflector_max']:.1f} K")

        try:
            run_simulation_fn(case["params"], run_dir)
        except Exception as e:
            print(f"  FAILED: {e}")

    # ----- Extract results and compute coefficients -----
    return postprocess_perturbation_study(base_dir, delta_T, coefficients)

# ====================================================================================================
# POST-PROCESSING: EXTRACT K-EFF VALUES AND COMPUTE COEFFICIENTS
# ====================================================================================================

def postprocess_perturbation_study(base_dir, delta_T, coefficients=("FTC", "MTC", "ITC")):
    """
    Post-process a completed perturbation study directory.

    Can be called standalone without re-running simulations.

    Parameters
    ------------
    base_dir : str
        Root directory containing baseline/, FTC_plus_*/, FTC_minus_*/, etc.
    delta_T : float
        The dT used in the study (K).
    coefficients : tuple of str
        Which coefficients to process.

    Returns
    --------
    dict : Mapping from coefficient key to result dict with:
        alpha_pcm_per_K, alpha_std_pcm_per_K, k_plus, k_minus, k_baseline, etc.
    """
    print(f"\n{'='*80}")
    print("REACTIVITY COEFFICIENT RESULTS")
    print(f"{'='*80}")

    # Extract baseline k-eff
    baseline_dir = os.path.join(base_dir, "baseline")
    k_base, k_base_std = extract_keff(baseline_dir)

    if k_base is None:
        print("ERROR: Could not extract baseline k-eff!")
        print(f"  Looked in: {baseline_dir}")
        return None

    rho_base = keff_to_reactivity_pcm(k_base)
    print(f"\nBaseline: k_eff = {k_base:.5f} +/- {k_base_std:.5f}  (rho = {rho_base:.1f} pcm)")

    results = {}

    for coeff_key in coefficients:
        entry = COEFF_REGISTRY[coeff_key]
        name = entry["name"]

        dir_plus = os.path.join(base_dir, f"{coeff_key}_plus_{delta_T:.0f}K")
        dir_minus = os.path.join(base_dir, f"{coeff_key}_minus_{delta_T:.0f}K")

        k_plus, k_plus_std = extract_keff(dir_plus)
        k_minus, k_minus_std = extract_keff(dir_minus)

        if k_plus is None or k_minus is None:
            print(f"\n{name} ({coeff_key}): INCOMPLETE - missing results")
            if k_plus is None:
                print(f"  Missing: {dir_plus}")
            if k_minus is None:
                print(f"  Missing: {dir_minus}")
            results[coeff_key] = None
            continue

        rho_plus = keff_to_reactivity_pcm(k_plus)
        rho_minus = keff_to_reactivity_pcm(k_minus)

        # Central difference: alpha = (rho_plus - rho_minus) / (2 * dT)
        alpha = (rho_plus - rho_minus) / (2.0 * delta_T)

        # Propagate uncertainty: sigma_alpha = sqrt(sigma_rho+^2 + sigma_rho-^2) / (2 dT)
        # where sigma_rho = sigma_k / k^2 * 1e5
        sigma_rho_plus = k_plus_std / (k_plus ** 2) * 1e5
        sigma_rho_minus = k_minus_std / (k_minus ** 2) * 1e5
        alpha_std = np.sqrt(sigma_rho_plus**2 + sigma_rho_minus**2) / (2.0 * delta_T)

        results[coeff_key] = {
            "name": name,
            "alpha_pcm_per_K": alpha,
            "alpha_std_pcm_per_K": alpha_std,
            "k_baseline": k_base,
            "k_baseline_std": k_base_std,
            "k_plus": k_plus,
            "k_plus_std": k_plus_std,
            "k_minus": k_minus,
            "k_minus_std": k_minus_std,
            "rho_plus_pcm": rho_plus,
            "rho_minus_pcm": rho_minus,
            "delta_T_K": delta_T,
        }

        print(f"\n{name} ({coeff_key}):")
        print(f"  k(+{delta_T:.0f}K) = {k_plus:.5f} +/- {k_plus_std:.5f}  ->  rho = {rho_plus:.1f} pcm")
        print(f"  k(-{delta_T:.0f}K) = {k_minus:.5f} +/- {k_minus_std:.5f}  ->  rho = {rho_minus:.1f} pcm")
        print(f"  delta_rho = {rho_plus - rho_minus:.1f} pcm over {2*delta_T:.0f} K")
        print(f"  alpha = {alpha:.3f} +/- {alpha_std:.3f} pcm/K")

        # Flag unexpected signs
        if coeff_key == "FTC" and alpha > 0:
            print(f"  WARNING: Positive FTC - expected negative (Doppler feedback)")
        elif coeff_key == "ITC" and alpha > 0:
            print(f"  WARNING: Positive ITC - check moderator/fuel balance")

    # Verify ITC ~ FTC + MTC (additivity check)
    if all(k in results and results[k] is not None for k in ("FTC", "MTC", "ITC")):
        ftc = results["FTC"]["alpha_pcm_per_K"]
        mtc = results["MTC"]["alpha_pcm_per_K"]
        itc = results["ITC"]["alpha_pcm_per_K"]
        sum_check = ftc + mtc
        deviation = abs(itc - sum_check)

        print(f"\n{'-'*60}")
        print(f"Additivity Check:")
        print(f"  FTC + MTC = {ftc:.3f} + {mtc:.3f} = {sum_check:.3f} pcm/K")
        print(f"  ITC       = {itc:.3f} pcm/K")
        print(f"  Difference: {deviation:.3f} pcm/K", end="")
        if deviation < max(abs(itc) * 0.15, 0.5):
            print("  (OK - within expected cross-coupling tolerance)")
        else:
            print("  (NOTE: Significant cross-coupling effects present)")

    # ----- Save results -----
    _save_results(results, base_dir, delta_T, k_base, k_base_std)

    print(f"\n{'='*80}\n")
    return results

# ====================================================================================================
# SAVE REACTIVITY COEFFICIENT CALCULATION RESULTS
# ====================================================================================================

def _save_results(results, base_dir, delta_T, k_base, k_base_std):
    """Save results to text report and JSON."""

    # ----- Text report -----
    report_path = os.path.join(base_dir, "reactivity_coefficients.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REACTIVITY COEFFICIENT RESULTS - Direct Perturbation Method\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Perturbation magnitude: dT = +/-{delta_T:.0f} K\n")
        f.write(f"Baseline k-eff: {k_base:.5f} +/- {k_base_std:.5f}\n\n")

        for key, res in results.items():
            if res is None:
                f.write(f"{key}: INCOMPLETE\n\n")
                continue
            f.write(f"{res['name']} ({key}):\n")
            f.write(f"  k(+{delta_T:.0f}K) = {res['k_plus']:.5f} +/- {res['k_plus_std']:.5f}\n")
            f.write(f"  k(-{delta_T:.0f}K) = {res['k_minus']:.5f} +/- {res['k_minus_std']:.5f}\n")
            f.write(f"  alpha = {res['alpha_pcm_per_K']:.3f} +/- {res['alpha_std_pcm_per_K']:.3f} pcm/K\n\n")

        # Additivity check
        if all(k in results and results[k] is not None for k in ("FTC", "MTC", "ITC")):
            ftc = results["FTC"]["alpha_pcm_per_K"]
            mtc = results["MTC"]["alpha_pcm_per_K"]
            itc = results["ITC"]["alpha_pcm_per_K"]
            f.write(f"Additivity Check:\n")
            f.write(f"  FTC + MTC = {ftc + mtc:.3f} pcm/K\n")
            f.write(f"  ITC       = {itc:.3f} pcm/K\n")
            f.write(f"  Difference: {abs(itc - ftc - mtc):.3f} pcm/K\n\n")

        f.write("=" * 80 + "\n")

    print(f"\nReport saved: {report_path}")

    # ----- JSON for programmatic access -----
    json_path = os.path.join(base_dir, "reactivity_coefficients.json")
    json_data = {
        "delta_T_K": delta_T,
        "k_baseline": k_base,
        "k_baseline_std": k_base_std,
        "coefficients": {},
    }
    for key, res in results.items():
        if res is not None:
            json_data["coefficients"][key] = {k: v for k, v in res.items()}

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"JSON saved:   {json_path}")

# ====================================================================================================
# PLOTTING FUNCTIONS
# ====================================================================================================

def plot_reactivity_coefficients(results, base_dir):
    """Generate a bar chart comparing the three coefficients."""
    import matplotlib.pyplot as plt

    labels = []
    values = []
    errors = []
    colors_list = []

    color_map = {"FTC": "#2196F3", "MTC": "#4CAF50", "ITC": "#FF9800"}

    for key in ("FTC", "MTC", "ITC"):
        if key in results and results[key] is not None:
            labels.append(key)
            values.append(results[key]["alpha_pcm_per_K"])
            errors.append(results[key]["alpha_std_pcm_per_K"])
            colors_list.append(color_map.get(key, "gray"))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=errors, capsize=6, color=colors_list,
                  edgecolor='black', linewidth=0.8, width=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Reactivity Coefficient (pcm/K)", fontsize=12)
    ax.set_title("Reactivity Coefficients - Direct Perturbation Method", fontsize=13)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.grid(axis='y', alpha=0.3)

    # Annotate values on bars
    for bar, val, err in zip(bars, values, errors):
        y_pos = bar.get_height()
        offset = max(abs(val) * 0.05, 0.3)
        ax.text(bar.get_x() + bar.get_width() / 2,
                y_pos + (offset if y_pos >= 0 else -offset),
                f"{val:.2f} +/- {err:.2f}",
                ha='center', va='bottom' if y_pos >= 0 else 'top', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(base_dir, "reactivity_coefficients_barplot.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved:   {save_path}")

# ====================================================================================================
# STANDALONE REACTIVITY COEFFICIENT CALCULATION
# ====================================================================================================

if __name__ == "__main__":
    """
    Usage:
        # Run the full study (simulations + post-processing):
        python reactivity_coefficients.py

        # Post-process an existing study:
        python reactivity_coefficients.py <study_directory> <delta_T>

        # Post-process specific coefficients:
        python reactivity_coefficients.py <study_directory> <delta_T> FTC,MTC
    """
    if len(sys.argv) >= 3:
        # Post-process mode
        study_dir = sys.argv[1]
        dT = float(sys.argv[2])
        coeffs = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("FTC", "MTC", "ITC")

        results = postprocess_perturbation_study(study_dir, dT, coeffs)
        if results:
            plot_reactivity_coefficients(results, study_dir)

    else:
        # Full run mode - import config and run
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, SCRIPT_DIR)

        import config as cfg
        from MicroHTGRNeutronics_INL_HTGTR_Inspired import run_simulation

        # Study configuration — can be overridden in config.py
        delta_T = getattr(cfg, 'reactivity_delta_T', 50.0)
        coefficients = getattr(cfg, 'reactivity_coefficients', ("FTC", "MTC", "ITC"))

        now = datetime.now()
        PARENT_DIR = os.path.dirname(SCRIPT_DIR)
        OUTPUT_BASE = os.path.join(PARENT_DIR, "MicroHTGR_Output")
        base_dir = os.path.join(
            OUTPUT_BASE,
            f"htgr_run_{now.strftime('%m.%d.%Y_%H.%M.%S')}_ReactivityCoefficients"
        )

        results = run_perturbation_study(
            base_params=cfg.params,
            delta_T=delta_T,
            base_dir=base_dir,
            coefficients=coefficients,
            run_simulation_fn=run_simulation,
        )

        if results:
            plot_reactivity_coefficients(results, base_dir)

        print(f"\nStudy complete. Results in: {base_dir}")
