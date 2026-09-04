"""
Control Rod Worth Post-Processing Script

Converts a control rod insertion parametric study into integral and differential
rod worth curves.

Run a ParametricStudy sweeping a rod bank insertion fraction, e.g.

    "parametric_param":  "bank_3_insertion",
    "parametric_values": [0.0, 0.05, 0.10, ..., 1.00],

then point this script at the resulting study directory. It reads
``parametric_study_results/parametric_study_results.csv`` and computes:

  * Reactivity at each insertion,  rho = (k - 1) / k          [pcm]
  * Integral worth relative to the fully withdrawn reference,
        W_int(x) = rho(x) - rho(0)                            [pcm]
  * Differential worth by forward difference,
        W_diff(x) = [rho(x+dx) - rho(x)] / dx                 [pcm per insertion fraction]

Monte Carlo uncertainties are propagated throughout. The reactivity standard
deviation follows from rho = 1 - 1/k, so  sigma_rho = sigma_k / k^2.

Usage
-----
    # As a module:
    from rod_worth_postprocessing import run_rod_worth_postprocessing
    run_rod_worth_postprocessing(parametric_dir)

    # Standalone:
    python rod_worth_postprocessing.py <parametric_study_directory> [param_name]

Writes ``rod_worth.csv`` and ``rod_worth.png`` into the study's
``parametric_study_results`` directory.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====================================================================================================
# REACTIVITY HELPERS
# ====================================================================================================


def rho_pcm(k):
    """Reactivity in pcm from k-effective."""
    return ((k - 1.0) / k) * 1e5


def rho_sd_pcm(k, sd_k):
    """
    Standard deviation of the reactivity in pcm.

    rho = 1 - 1/k, so d(rho)/dk = 1/k^2 and sigma_rho = sigma_k / k^2.
    """
    return (sd_k / (k * k)) * 1e5


# ====================================================================================================
# ROD WORTH CALCULATION
# ====================================================================================================


def run_rod_worth_postprocessing(parametric_dir, param_name=None, make_plot=True):
    """
    Compute integral and differential rod worth from a parametric study.

    Args:
        parametric_dir (str): Parametric study directory, or the
            parametric_study_results directory inside it.
        param_name (str, optional): Which swept parameter to analyse. Defaults to
            the only parameter present, and errors if the file holds more than one.
        make_plot (bool): Also write rod_worth.png alongside the CSV.

    Returns:
        pandas.DataFrame: insertion fraction, integral and differential worth,
        each with its standard deviation.
    """
    results_dir = parametric_dir
    if os.path.basename(os.path.normpath(results_dir)) != "parametric_study_results":
        candidate = os.path.join(parametric_dir, "parametric_study_results")
        if os.path.isdir(candidate):
            results_dir = candidate

    csv_path = os.path.join(results_dir, "parametric_study_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No parametric_study_results.csv found at {csv_path}. "
            "Point this script at a completed ParametricStudy directory."
        )

    df = pd.read_csv(csv_path)

    # Resolve which swept parameter to analyse.
    available = sorted(df["parameter_name"].unique())
    if param_name is None:
        if len(available) != 1:
            raise ValueError(
                f"{csv_path} contains multiple parameters {available}; "
                "pass param_name to choose one."
            )
        param_name = available[0]
    elif param_name not in available:
        raise ValueError(f"Parameter {param_name!r} not in {csv_path}. Available: {available}")

    g = (df[df["parameter_name"] == param_name]
         .sort_values("parameter_value")
         .reset_index(drop=True)
         .copy())

    if len(g) < 2:
        raise ValueError(
            f"Need at least two insertion values to compute rod worth; got {len(g)}."
        )

    g["rho_pcm"] = rho_pcm(g["keff"])
    g["rho_sd_pcm"] = rho_sd_pcm(g["keff"], g["keff_std"])

    # Reference is the case closest to fully withdrawn.
    base = g.loc[(g["parameter_value"] - 0.0).abs().idxmin()]

    g["integral_worth_pcm"] = g["rho_pcm"] - base["rho_pcm"]
    g["integral_worth_sd_pcm"] = np.sqrt(g["rho_sd_pcm"] ** 2 + base["rho_sd_pcm"] ** 2)

    # Forward difference; the last row has no successor and is left as NaN.
    dx = g["parameter_value"].shift(-1) - g["parameter_value"]
    g["differential_worth_pcm_per_frac"] = (g["rho_pcm"].shift(-1) - g["rho_pcm"]) / dx
    g["differential_worth_sd_pcm_per_frac"] = (
        np.sqrt(g["rho_sd_pcm"].shift(-1) ** 2 + g["rho_sd_pcm"] ** 2) / dx.abs()
    )

    out = g[["parameter_value",
             "integral_worth_pcm", "integral_worth_sd_pcm",
             "differential_worth_pcm_per_frac", "differential_worth_sd_pcm_per_frac"]].rename(
        columns={"parameter_value": "insertion_frac"})

    csv_out = os.path.join(results_dir, "rod_worth.csv")
    out.to_csv(csv_out, index=False)
    print(f"  Rod worth CSV written: {csv_out}")

    total = out["integral_worth_pcm"].abs().max()
    print(f"  Parameter analysed:   {param_name}")
    print(f"  Insertion points:     {len(out)}")
    print(f"  Total integral worth: {total:,.0f} pcm")

    if make_plot:
        _plot_rod_worth(out, param_name, results_dir)

    return out


# ====================================================================================================
# PLOTTING
# ====================================================================================================


def _plot_rod_worth(out, param_name, results_dir):
    """Write a two-panel integral and differential rod worth figure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.errorbar(out["insertion_frac"], out["integral_worth_pcm"],
                 yerr=out["integral_worth_sd_pcm"],
                 marker="o", linewidth=1.5, capsize=3)
    ax1.set_ylabel("Integral worth (pcm)")
    ax1.set_title(f"Control Rod Worth — {param_name}")
    ax1.grid(alpha=0.3)

    valid = out.dropna(subset=["differential_worth_pcm_per_frac"])
    ax2.errorbar(valid["insertion_frac"], valid["differential_worth_pcm_per_frac"],
                 yerr=valid["differential_worth_sd_pcm_per_frac"],
                 marker="s", linewidth=1.5, capsize=3, color="tab:orange")
    ax2.set_xlabel("Insertion fraction")
    ax2.set_ylabel("Differential worth (pcm per unit insertion)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_out = os.path.join(results_dir, "rod_worth.png")
    fig.savefig(png_out, dpi=200)
    plt.close(fig)
    print(f"  Rod worth plot written: {png_out}")


# ====================================================================================================
# STANDALONE ENTRY POINT
# ====================================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("ERROR: a parametric study directory is required.")
        sys.exit(1)

    run_rod_worth_postprocessing(
        sys.argv[1],
        param_name=sys.argv[2] if len(sys.argv) > 2 else None,
    )
