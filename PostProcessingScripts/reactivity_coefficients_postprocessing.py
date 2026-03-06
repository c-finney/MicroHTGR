"""
Reactivity Coefficient Post-Processing Script

Handles plotting and saving of figures and text/JSON files produced by
reactivity_coefficients.py.

Usage:
    # Imported automatically by reactivity_coefficients.py
    from reactivity_coefficients_postprocessing import save_results, plot_results

    # Standalone (re-plot from an existing JSON):
    python reactivity_coefficients_postprocessing.py <output_base_dir>
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


# Short bar-chart labels for the summary plot
_BAR_LABELS = {
    "FTC": "Fuel",
    "MTC": "Moderator",
    "ITC": "Temperature",
}


def _reactivity_pcm(k):
    """Convert k-effective to reactivity in pcm."""
    return (k - 1.0) / k * 1e5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_results(all_results, k_ref, k_ref_std, rho_ref, output_dir):
    """Save results to JSON and human-readable text files."""

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


def plot_results(all_results, k_ref, output_dir):
    """Generate publication-quality plots of reactivity vs. temperature perturbation."""

    for name, res in all_results.items():
        cases = res["cases"]
        dTs = np.array([c["delta_T"] for c in cases])
        rhos = np.array([c["rho_pert"] for c in cases])
        k_stds = np.array([c["k_pert_std"] for c in cases])
        ks = np.array([c["k_pert"] for c in cases])

        rho_ref = _reactivity_pcm(k_ref)
        delta_rho = rhos - rho_ref

        sig_rhos = k_stds / (ks ** 2) * 1e5

        sort_idx = np.argsort(dTs)
        dTs = dTs[sort_idx]
        delta_rho = delta_rho[sort_idx]
        sig_rhos = sig_rhos[sort_idx]

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
        ax.set_title(
            f"{res['label']}\n"
            f"α = {res['average_alpha_pcm_per_K']:.3f} ± {res['average_alpha_std']:.3f} pcm/K",
            fontsize=13,
        )
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
    labels = [_BAR_LABELS.get(n, all_results[n]["label"].split("(")[0].strip()) for n in names]

    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(labels, alphas, yerr=stds, capsize=5, color=colors[:len(names)],
                  edgecolor="black", linewidth=0.8, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("α (pcm/K)", fontsize=12)
    ax.set_title("Temperature Reactivity Coefficients", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

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
# Standalone entry point — re-plot from an existing JSON
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reactivity_coefficients_postprocessing.py <output_base_dir>")
        print("\nRe-generates plots and text files from an existing reactivity_coefficients.json.")
        sys.exit(1)

    output_dir = sys.argv[1]
    json_path = os.path.join(output_dir, "reactivity_coefficients.json")

    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found.")
        sys.exit(1)

    with open(json_path, "r") as f:
        data = json.load(f)

    k_ref     = data["reference"]["k_eff"]
    k_ref_std = data["reference"]["k_eff_std"]
    rho_ref   = data["reference"]["reactivity_pcm"]

    # Rebuild all_results structure expected by plot_results / save_results
    all_results = {}
    for name, coeff in data["coefficients"].items():
        all_results[name] = {
            "label":                   coeff["label"],
            "average_alpha_pcm_per_K": coeff["average_alpha_pcm_per_K"],
            "average_alpha_std":       coeff["average_alpha_std"],
            "cases":                   coeff["cases"],
            "central_difference":      coeff.get("central_difference", []),
        }

    print(f"\nRe-generating outputs in: {output_dir}")
    save_results(all_results, k_ref, k_ref_std, rho_ref, output_dir)
    plot_results(all_results, k_ref, output_dir)
    print("\nDone.")
