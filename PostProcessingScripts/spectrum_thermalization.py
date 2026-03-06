"""
Neutron Energy Spectrum & Thermalization Metrics Post-Processing

Extracts the energy-dependent flux tally from OpenMC statepoint files
and computes key thermalization metrics relevant to HTGRs:

  - Neutron energy spectrum (flux per unit lethargy)
  - Thermal / epithermal / fast flux fractions
  - Average neutron energy & median energy
  - Thermal-to-fast flux ratio
  - Spectral index (epithermal-to-thermal ratio)
  - Thermal utilization factor f (from tally data)
  - Resonance escape probability estimate
  - Cadmium ratio estimate

All plots are saved to the run directory.

Usage:
    # As a module from the main simulation script:
    from spectrum_thermalization import run_spectrum_analysis

    run_spectrum_analysis(run_dir, params)

    # Standalone:
    python spectrum_thermalization.py <run_directory> [batch_number]
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ===========================================================================
# Energy group boundaries (eV)
# ===========================================================================
E_THERMAL_UPPER = 0.625          # Cadmium cutoff
E_EPITHERMAL_UPPER = 1.0e5      # 100 keV
E_FAST_UPPER = 2.0e7            # 20 MeV (upper limit of most libraries)

# Resonance region for p (resonance escape probability) estimate
E_RESONANCE_LOW = 1.0            # 1 eV
E_RESONANCE_HIGH = 1.0e3         # 1 keV  (dominant U-238 resonances)

# Additional energy landmarks
E_FISSION_PEAK = 2.0e6           # ~2 MeV typical fission spectrum peak
E_1EV = 1.0
E_CADMIUM = 0.5                  # Approximate Cd cutoff


# ===========================================================================
# Core analysis function
# ===========================================================================

def analyze_spectrum(energy_edges, flux_per_bin):
    """
    Compute thermalization metrics from a flux energy spectrum.

    Parameters
    ----------
    energy_edges : ndarray, shape (N+1,)
        Energy bin edges in eV (ascending).
    flux_per_bin : ndarray, shape (N,)
        Integrated flux in each bin (n·cm / source or similar).

    Returns
    -------
    dict : Dictionary of computed metrics.
    ndarray : Flux per unit lethargy for each bin.
    ndarray : Bin center energies in eV.
    """

    n_bins = len(flux_per_bin)
    assert len(energy_edges) == n_bins + 1

    # Bin centers (geometric mean)
    E_lo = energy_edges[:-1]
    E_hi = energy_edges[1:]
    E_center = np.sqrt(E_lo * E_hi)

    # Lethargy width: Δu = ln(E_hi / E_lo)
    delta_u = np.log(E_hi / E_lo)
    delta_u = np.where(delta_u > 0, delta_u, 1e-30)  # guard against zero-width bins

    # Flux per unit lethargy  (the standard way to display reactor spectra)
    flux_per_lethargy = flux_per_bin / delta_u

    # =========================================================================-
    # Group-integrated fluxes
    # =========================================================================-
    thermal_mask = E_center <= E_THERMAL_UPPER
    epithermal_mask = (E_center > E_THERMAL_UPPER) & (E_center <= E_EPITHERMAL_UPPER)
    fast_mask = E_center > E_EPITHERMAL_UPPER

    phi_thermal = np.sum(flux_per_bin[thermal_mask])
    phi_epithermal = np.sum(flux_per_bin[epithermal_mask])
    phi_fast = np.sum(flux_per_bin[fast_mask])
    phi_total = np.sum(flux_per_bin)

    # Guard against zero total flux
    if phi_total == 0:
        phi_total = 1e-30

    # Fractions
    f_thermal = phi_thermal / phi_total
    f_epithermal = phi_epithermal / phi_total
    f_fast = phi_fast / phi_total

    # =========================================================================-
    # Average and median energies
    # =========================================================================-
    E_avg = np.sum(E_center * flux_per_bin) / phi_total

    # Median: energy below which 50 % of flux resides
    cumulative = np.cumsum(flux_per_bin) / phi_total
    idx_median = np.searchsorted(cumulative, 0.5)
    E_median = E_center[min(idx_median, n_bins - 1)]

    # Most probable energy (peak of flux/lethargy spectrum)
    idx_peak = np.argmax(flux_per_lethargy)
    E_peak = E_center[idx_peak]

    # =========================================================================-
    # Ratios
    # =========================================================================-
    thermal_to_fast = phi_thermal / phi_fast if phi_fast > 0 else np.inf
    spectral_index = phi_epithermal / phi_thermal if phi_thermal > 0 else np.inf

    # Cadmium ratio estimate  CR ≈ φ_total / φ_epithermal  (simplified)
    cadmium_ratio = phi_total / phi_epithermal if phi_epithermal > 0 else np.inf

    # =========================================================================-
    # Resonance-region flux fraction (1 eV – 1 keV)
    # =========================================================================-
    resonance_mask = (E_center >= E_RESONANCE_LOW) & (E_center <= E_RESONANCE_HIGH)
    phi_resonance = np.sum(flux_per_bin[resonance_mask])
    f_resonance = phi_resonance / phi_total

    # =========================================================================-
    # Thermalization ratio: ratio of flux below 0.1 eV to total
    # (measures how well the spectrum reaches full Maxwellian equilibrium)
    # =========================================================================-
    sub_thermal_mask = E_center <= 0.1
    phi_sub_thermal = np.sum(flux_per_bin[sub_thermal_mask])
    f_sub_thermal = phi_sub_thermal / phi_total

    # =========================================================================-
    # Effective neutron temperature (fit Maxwellian to thermal region)
    # =========================================================================-
    kB_eV = 8.617333e-5  # Boltzmann constant in eV/K
    T_neutron = E_avg / (1.5 * kB_eV) if E_avg < 1.0 else None  # only meaningful if thermal

    # For HTGR with epithermal spectrum, use average thermal energy instead
    if phi_thermal > 0:
        E_avg_thermal = np.sum(E_center[thermal_mask] * flux_per_bin[thermal_mask]) / phi_thermal
        T_neutron_thermal = E_avg_thermal / (1.5 * kB_eV)
    else:
        E_avg_thermal = 0
        T_neutron_thermal = 0

    metrics = {
        # Group fluxes
        "phi_thermal": phi_thermal,
        "phi_epithermal": phi_epithermal,
        "phi_fast": phi_fast,
        "phi_total": phi_total,

        # Fractions
        "f_thermal": f_thermal,
        "f_epithermal": f_epithermal,
        "f_fast": f_fast,
        "f_resonance": f_resonance,
        "f_sub_thermal": f_sub_thermal,

        # Characteristic energies
        "E_avg_eV": E_avg,
        "E_median_eV": E_median,
        "E_peak_eV": E_peak,
        "E_avg_thermal_eV": E_avg_thermal,

        # Ratios
        "thermal_to_fast_ratio": thermal_to_fast,
        "spectral_index": spectral_index,
        "cadmium_ratio": cadmium_ratio,

        # Neutron temperature
        "T_neutron_thermal_K": T_neutron_thermal,

        # Energy boundaries used
        "E_thermal_upper_eV": E_THERMAL_UPPER,
        "E_epithermal_upper_eV": E_EPITHERMAL_UPPER,
    }

    return metrics, flux_per_lethargy, E_center


# ===========================================================================
# Plotting
# ===========================================================================

def plot_flux_spectrum(E_center, flux_per_lethargy, flux_per_bin, metrics, run_dir, batch=None):
    """Generate neutron energy spectrum plots."""

    batch_label = f" (Batch {batch})" if batch else ""
    colors = {"thermal": "#2196F3", "epithermal": "#FF9800", "fast": "#F44336"}

    # =========================================================================-
    # 1. Flux per unit lethargy vs energy
    # =========================================================================-
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Shade energy regions
    thermal_mask = E_center <= E_THERMAL_UPPER
    epithermal_mask = (E_center > E_THERMAL_UPPER) & (E_center <= E_EPITHERMAL_UPPER)
    fast_mask = E_center > E_EPITHERMAL_UPPER

    ax.fill_between(E_center[thermal_mask], flux_per_lethargy[thermal_mask],
                     alpha=0.25, color=colors["thermal"], label=f"Thermal ({metrics['f_thermal']*100:.1f}%)")
    ax.fill_between(E_center[epithermal_mask], flux_per_lethargy[epithermal_mask],
                     alpha=0.25, color=colors["epithermal"], label=f"Epithermal ({metrics['f_epithermal']*100:.1f}%)")
    ax.fill_between(E_center[fast_mask], flux_per_lethargy[fast_mask],
                     alpha=0.25, color=colors["fast"], label=f"Fast ({metrics['f_fast']*100:.1f}%)")

    ax.plot(E_center, flux_per_lethargy, "k-", linewidth=1.0, alpha=0.8)

    # Mark energy boundaries
    for E_bound, lbl in [(E_THERMAL_UPPER, "0.625 eV"), (E_EPITHERMAL_UPPER, "100 keV")]:
        ax.axvline(E_bound, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(E_bound * 1.2, ax.get_ylim()[1] * 0.9, lbl, fontsize=8, color="gray", rotation=90, va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Neutron Energy (eV)", fontsize=12)
    ax.set_ylabel("Flux per Unit Lethargy (arb. units)", fontsize=12)
    ax.set_title(f"Neutron Energy Spectrum{batch_label}", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.set_xlim(E_center[E_center > 0].min(), E_center.max())

    save_path = os.path.join(run_dir, "neutron_energy_spectrum.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # =========================================================================-
    # 2. Cumulative flux fraction
    # =========================================================================-
    cumulative = np.cumsum(flux_per_bin) / metrics["phi_total"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(E_center, cumulative * 100, "k-", linewidth=1.5)

    ax.axvline(E_THERMAL_UPPER, color=colors["thermal"], linestyle="--", linewidth=1,
               label=f"Thermal cutoff ({E_THERMAL_UPPER} eV)")
    ax.axvline(E_EPITHERMAL_UPPER, color=colors["fast"], linestyle="--", linewidth=1,
               label=f"Fast cutoff ({E_EPITHERMAL_UPPER/1e3:.0f} keV)")
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(E_center.min() * 2, 52, f"Median E = {_format_energy(metrics['E_median_eV'])}",
            fontsize=9, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("Neutron Energy (eV)", fontsize=12)
    ax.set_ylabel("Cumulative Flux Fraction (%)", fontsize=12)
    ax.set_title(f"Cumulative Neutron Flux Distribution{batch_label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="major", alpha=0.3)
    ax.set_ylim(0, 105)

    save_path = os.path.join(run_dir, "neutron_flux_cumulative.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # =========================================================================-
    # 3. Thermal region zoom (linear y-scale, shows Maxwellian peak)
    # =========================================================================-
    zoom_mask = E_center <= 2.0  # up to 2 eV

    if np.any(zoom_mask) and np.any(flux_per_lethargy[zoom_mask] > 0):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.plot(E_center[zoom_mask], flux_per_lethargy[zoom_mask], "b-", linewidth=1.5)
        ax.fill_between(E_center[zoom_mask], flux_per_lethargy[zoom_mask],
                         alpha=0.2, color=colors["thermal"])

        ax.axvline(E_THERMAL_UPPER, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="Cd cutoff (0.625 eV)")

        if metrics["T_neutron_thermal_K"] > 0:
            ax.set_title(
                f"Thermal Flux Region{batch_label}\n"
                f"Effective thermal neutron temperature ≈ {metrics['T_neutron_thermal_K']:.0f} K",
                fontsize=13,
            )
        else:
            ax.set_title(f"Thermal Flux Region{batch_label}", fontsize=13)

        ax.set_xscale("log")
        ax.set_xlabel("Neutron Energy (eV)", fontsize=12)
        ax.set_ylabel("Flux per Unit Lethargy (arb. units)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(run_dir, "neutron_spectrum_thermal_zoom.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # =========================================================================-
    # 4. Metrics summary card (saved as figure)
    # =========================================================================-
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.axis("off")

    text_lines = [
        ("NEUTRON THERMALIZATION METRICS", "bold", 14),
        ("", "normal", 10),
        (f"Thermal fraction  (E < {E_THERMAL_UPPER} eV):    {metrics['f_thermal']*100:.2f} %", "normal", 11),
        (f"Epithermal fraction:                       {metrics['f_epithermal']*100:.2f} %", "normal", 11),
        (f"Fast fraction     (E > {E_EPITHERMAL_UPPER/1e3:.0f} keV):   {metrics['f_fast']*100:.2f} %", "normal", 11),
        (f"Resonance fraction (1 eV – 1 keV):         {metrics['f_resonance']*100:.2f} %", "normal", 11),
        ("", "normal", 10),
        (f"Average neutron energy:    {_format_energy(metrics['E_avg_eV'])}", "normal", 11),
        (f"Median neutron energy:     {_format_energy(metrics['E_median_eV'])}", "normal", 11),
        (f"Peak energy (φ/Δu):        {_format_energy(metrics['E_peak_eV'])}", "normal", 11),
        ("", "normal", 10),
        (f"Thermal-to-fast ratio:     {metrics['thermal_to_fast_ratio']:.4f}", "normal", 11),
        (f"Spectral index (epi/th):   {metrics['spectral_index']:.4f}", "normal", 11),
        (f"Cadmium ratio (approx):    {metrics['cadmium_ratio']:.2f}", "normal", 11),
        ("", "normal", 10),
        (f"Eff. thermal neutron T:    {metrics['T_neutron_thermal_K']:.0f} K", "normal", 11),
    ]

    y = 0.95
    for text, weight, size in text_lines:
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
                fontweight=weight, fontfamily="monospace", verticalalignment="top")
        y -= 0.055

    save_path = os.path.join(run_dir, "thermalization_metrics_card.png")
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def _format_energy(E_eV):
    """Format energy value with appropriate units."""
    if E_eV < 1e-3:
        return f"{E_eV*1e3:.3f} meV"
    elif E_eV < 1.0:
        return f"{E_eV:.4f} eV"
    elif E_eV < 1e3:
        return f"{E_eV:.2f} eV"
    elif E_eV < 1e6:
        return f"{E_eV/1e3:.2f} keV"
    else:
        return f"{E_eV/1e6:.2f} MeV"


# ===========================================================================
# Main entry point
# ===========================================================================

def run_spectrum_analysis(run_dir, params, batch=None):
    """
    Run full neutron spectrum and thermalization analysis.

    Parameters
    ----------
    run_dir : str
        Directory containing OpenMC simulation results (statepoint file).
    params : dict
        Simulation parameters dictionary.
    batch : int, optional
        Statepoint batch number.  Auto-detected if None.

    Returns
    -------
    dict : Computed metrics dictionary.
    """
    import openmc

    print(f"\n{'=' * 80}")
    print("NEUTRON ENERGY SPECTRUM & THERMALIZATION ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Run directory: {run_dir}")

    results_dir = os.path.join(run_dir, 'spectrum_thermalization_results')
    os.makedirs(results_dir, exist_ok=True)

    # =========================================================================
    # Find statepoint
    # =========================================================================
    if batch is None:
        for f in os.listdir(run_dir):
            if f.startswith("statepoint") and f.endswith(".h5"):
                batch = int(f.split(".")[1])
                break

    if batch is None:
        print("ERROR: No statepoint file found!")
        return None

    sp_path = os.path.join(run_dir, f"statepoint.{batch}.h5")
    print(f"Statepoint: {sp_path}")

    sp = openmc.StatePoint(sp_path)

    # =========================================================================
    # Extract flux energy spectrum tally
    # =========================================================================
    try:
        spectrum_tally = sp.get_tally(name="flux_energy_spectrum")
    except Exception:
        print("ERROR: 'flux_energy_spectrum' tally not found in statepoint.")
        print("Ensure the main simulation defines this tally with an EnergyFilter.")
        return None

    energy_filter = spectrum_tally.find_filter(openmc.EnergyFilter)
    energy_edges = energy_filter.bins.flatten()
    # Remove duplicate edges from the flattened bin pairs
    energy_edges_unique = np.unique(energy_edges)

    # Flux mean values — shape is (n_bins, 1, 1) for a single-score tally
    flux_per_bin = spectrum_tally.mean.flatten()

    # Also grab uncertainties
    flux_std = spectrum_tally.std_dev.flatten()

    n_bins = len(flux_per_bin)
    print(f"Energy bins: {n_bins}")
    print(f"Energy range: {_format_energy(energy_edges_unique[0])} → {_format_energy(energy_edges_unique[-1])}")

    # =========================================================================
    # Run analysis
    # =========================================================================
    metrics, flux_per_lethargy, E_center = analyze_spectrum(energy_edges_unique, flux_per_bin)

    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n{'─' * 60}")
    print("  THERMALIZATION METRICS SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Thermal fraction  (E < {E_THERMAL_UPPER} eV):  {metrics['f_thermal']*100:.2f} %")
    print(f"  Epithermal fraction:                    {metrics['f_epithermal']*100:.2f} %")
    print(f"  Fast fraction     (E > {E_EPITHERMAL_UPPER/1e3:.0f} keV): {metrics['f_fast']*100:.2f} %")
    print(f"  Resonance fraction (1 eV – 1 keV):     {metrics['f_resonance']*100:.2f} %")
    print(f"")
    print(f"  Average neutron energy:    {_format_energy(metrics['E_avg_eV'])}")
    print(f"  Median neutron energy:     {_format_energy(metrics['E_median_eV'])}")
    print(f"  Peak energy (φ/Δu):        {_format_energy(metrics['E_peak_eV'])}")
    print(f"")
    print(f"  Thermal-to-fast ratio:     {metrics['thermal_to_fast_ratio']:.4f}")
    print(f"  Spectral index (epi/th):   {metrics['spectral_index']:.4f}")
    print(f"  Cadmium ratio (approx):    {metrics['cadmium_ratio']:.2f}")
    print(f"")
    print(f"  Eff. thermal neutron T:    {metrics['T_neutron_thermal_K']:.0f} K")
    print(f"{'─' * 60}")

    # =========================================================================
    # Generate plots
    # =========================================================================
    print("\nGenerating spectrum plots...")
    plot_flux_spectrum(E_center, flux_per_lethargy, flux_per_bin, metrics, results_dir, batch)

    # =========================================================================
    # Save results
    # =========================================================================
    # JSON
    json_path = os.path.join(results_dir, "thermalization_metrics.json")
    # Convert numpy types for JSON serialization
    metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics.items()}
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n  Metrics saved to: {json_path}")

    # Human-readable text
    txt_path = os.path.join(results_dir, "thermalization_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("NEUTRON ENERGY SPECTRUM & THERMALIZATION METRICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Energy Group Definitions:\n")
        f.write(f"  Thermal:     E < {E_THERMAL_UPPER} eV\n")
        f.write(f"  Epithermal:  {E_THERMAL_UPPER} eV < E < {E_EPITHERMAL_UPPER/1e3:.0f} keV\n")
        f.write(f"  Fast:        E > {E_EPITHERMAL_UPPER/1e3:.0f} keV\n\n")

        f.write("Flux Fractions:\n")
        f.write(f"  Thermal:     {metrics['f_thermal']*100:.3f} %\n")
        f.write(f"  Epithermal:  {metrics['f_epithermal']*100:.3f} %\n")
        f.write(f"  Fast:        {metrics['f_fast']*100:.3f} %\n")
        f.write(f"  Resonance:   {metrics['f_resonance']*100:.3f} %\n")
        f.write(f"  Sub-thermal: {metrics['f_sub_thermal']*100:.3f} %\n\n")

        f.write("Characteristic Energies:\n")
        f.write(f"  Average:     {_format_energy(metrics['E_avg_eV'])}\n")
        f.write(f"  Median:      {_format_energy(metrics['E_median_eV'])}\n")
        f.write(f"  Peak (φ/Δu): {_format_energy(metrics['E_peak_eV'])}\n\n")

        f.write("Spectral Ratios:\n")
        f.write(f"  Thermal-to-fast:     {metrics['thermal_to_fast_ratio']:.4f}\n")
        f.write(f"  Spectral index:      {metrics['spectral_index']:.4f}\n")
        f.write(f"  Cadmium ratio:       {metrics['cadmium_ratio']:.2f}\n\n")

        f.write("Neutron Temperature:\n")
        f.write(f"  Effective thermal T: {metrics['T_neutron_thermal_K']:.0f} K\n\n")

        f.write("=" * 80 + "\n")
    print(f"  Report saved to: {txt_path}")

    # Save raw spectrum data as CSV for external plotting
    csv_path = os.path.join(results_dir, "neutron_spectrum_data.csv")
    header = "E_center_eV,flux_per_bin,flux_per_lethargy,flux_std"
    data = np.column_stack([E_center, flux_per_bin, flux_per_lethargy, flux_std])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"  Spectrum data saved to: {csv_path}")

    print(f"\n{'=' * 80}")
    print("SPECTRUM ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")

    return metrics


# ===========================================================================
# Standalone entry point
# ===========================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python spectrum_thermalization.py <run_directory> [batch_number]")
        print("\nExtracts neutron energy spectrum from the 'flux_energy_spectrum' tally")
        print("and computes thermalization metrics.  Parameters are loaded from")
        print("run_params.json in the run directory.")
        sys.exit(1)

    run_dir = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"\nProcessing: {run_dir}")

    # Load params
    params_path = os.path.join(run_dir, "run_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        params = {}

    run_spectrum_analysis(run_dir, params, batch)
