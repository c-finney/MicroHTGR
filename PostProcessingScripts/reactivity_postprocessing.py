"""
Reactivity Coefficient Post-Processing Script

Extracts, summarizes, and plots results from reactivity coefficient
perturbation studies. Called automatically by reactivity_coefficients.py
after all perturbed cases have been run.

Usage:
    # As a module:
    from reactivity_postprocessing import run_reactivity_postprocessing
    run_reactivity_postprocessing(all_results, k_ref, k_ref_std, output_dir)

    # Standalone:
    python reactivity_postprocessing.py <reactivity_study_directory>
"""

import os
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================================================
# HELPER FUNCTION
# ====================================================================================================

def reactivity_pcm(k):
    """Convert k-effective to reactivity in pcm."""
    return (k - 1.0) / k * 1e5

# ====================================================================================================
# SUMMARY PRINTING
# ====================================================================================================

def print_summary(all_results, k_ref, k_ref_std, rho_ref):
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

# ====================================================================================================
# SAVE RESULTS TO JSON AND TEXT FILES
# ====================================================================================================

def save_results(all_results, k_ref, k_ref_std, rho_ref, output_dir):
    
    # ----- JSON FILE -----

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

    # ----- Text File -----

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

# ====================================================================================================
# PLOTTING FUNCTION
# ====================================================================================================

def plot_results(all_results, k_ref, output_dir):
    """Generate publication-quality plots of reactivity vs. temperature perturbation."""

    for name, res in all_results.items():
        cases = res["cases"]
        dTs = np.array([c["delta_T"] for c in cases])
        rhos = np.array([c["rho_pert"] for c in cases])
        k_stds = np.array([c["k_pert_std"] for c in cases])
        ks = np.array([c["k_pert"] for c in cases])

        rho_ref_val = reactivity_pcm(k_ref)
        delta_rho = rhos - rho_ref_val

        # Propagated uncertainty on Δρ
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

# ====================================================================================================
# POST-PROCESSING ENTRY POINT FUNCTION
# ====================================================================================================

def run_reactivity_postprocessing(all_results, k_ref, k_ref_std, output_dir):
    """
    Run full post-processing: print summary, save files, generate plots.

    Parameters
    ----------
    all_results : dict
        Nested results dictionary from reactivity_coefficients.py.
    k_ref : float
        Reference k-effective.
    k_ref_std : float
        Reference k-effective standard deviation.
    output_dir : str
        Directory to save outputs.
    """
    rho_ref = reactivity_pcm(k_ref)

    print_summary(all_results, k_ref, k_ref_std, rho_ref)
    save_results(all_results, k_ref, k_ref_std, rho_ref, output_dir)
    plot_results(all_results, k_ref, output_dir)

# ====================================================================================================
# STANDALON ENTY POINT FUNCTION: RE-PROCESS FROM SAVED JSON
# ====================================================================================================

def reprocess_from_json(json_path, output_dir=None):
    """
    Re-run plotting and reporting from a previously saved JSON file.

    Parameters
    ----------
    json_path : str
        Path to reactivity_coefficients.json.
    output_dir : str, optional
        Directory to save outputs. Defaults to the directory containing the JSON.
    """
    if output_dir is None:
        output_dir = os.path.dirname(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    k_ref = data["reference"]["k_eff"]
    k_ref_std = data["reference"]["k_eff_std"]

    # Reconstruct all_results dict from JSON
    all_results = {}
    for name, coeff_data in data["coefficients"].items():
        all_results[name] = {
            "label": coeff_data["label"],
            "cases": coeff_data["cases"],
            "central_difference": coeff_data["central_difference"],
            "average_alpha_pcm_per_K": coeff_data["average_alpha_pcm_per_K"],
            "average_alpha_std": coeff_data["average_alpha_std"],
        }

    print(f"\nRe-processing from: {json_path}")
    run_reactivity_postprocessing(all_results, k_ref, k_ref_std, output_dir)

# ====================================================================================================
# STANDALONE ENTRY POINT
# ====================================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reactivity_postprocessing.py <reactivity_study_directory>")
        print("\nRe-generates plots and reports from reactivity_coefficients.json")
        sys.exit(1)

    study_dir = sys.argv[1]
    json_file = os.path.join(study_dir, "reactivity_coefficients.json")

    if not os.path.exists(json_file):
        print(f"ERROR: {json_file} not found.")
        sys.exit(1)

    reprocess_from_json(json_file, study_dir)
