"""
Burnup Estimation Post-Processing Script

Estimates fuel cycle length and extracts k_eff and leakage fraction from statepoint files.
Uranium mass and fuel volume are read directly from run_params.json, which is populated
analytically during model construction (no stochastic volume calculation required).

Usage:
    from burnup_estimation import run_burnup_estimation
    run_burnup_estimation(run_dir, params)

    # Standalone:
    python burnup_estimation.py <run_directory>
"""

import openmc
import numpy as np
import os
import sys
import json

# ====================================================================================================
# PERFORM BURNUP ANALYSIS AND SAVE RESULTS
# ====================================================================================================

def run_burnup_estimation(run_dir, params, burnup_limit=160_000):
    """
    Estimate fuel cycle length and extract simulation results.

    Uranium mass and fuel volume are read from run_params.json, which is written
    analytically during model construction in simulation.py.

    Args:
        run_dir:       Directory containing simulation results and run_params.json
        params:        Parameter dictionary (merged params + run_params.json entries)
        burnup_limit:  Maximum burnup limit in MWd/MtU (default 160,000)

    Returns:
        dict: Dictionary containing all calculated values, or None on failure
    """

    print(f"\n{'='*80}")
    print("SIMULATION RESULTS EXTRACTION")
    print(f"{'='*80}")

    # ================================================================================
    # 1. EXTRACT K-EFFECTIVE AND LEAKAGE FRACTION FROM OUTPUT FILE
    # ================================================================================

    keff          = None
    keff_std      = None
    leakage_fraction = None
    leakage_std   = None

    output_file = os.path.join(run_dir, 'openmc_output.txt')

    if not os.path.exists(output_file):
        print("ERROR: openmc_output.txt not found!")
        return None

    with open(output_file, 'r') as f:
        content = f.read()

    for line in content.split('\n'):
        if 'Combined k-effective' in line and '=' in line:
            parts = line.split('=')[1].strip().split('+/-')
            keff = float(parts[0].strip())
            if len(parts) > 1:
                keff_std = float(parts[1].strip())

        if 'Leakage Fraction' in line and '=' in line:
            parts = line.split('=')[1].strip().split('+/-')
            leakage_fraction = float(parts[0].strip())
            if len(parts) > 1:
                leakage_std = float(parts[1].strip())

    if keff is None:
        print("ERROR: Could not parse k-effective from openmc_output.txt")
        return None

    if leakage_fraction is None:
        print("WARNING: Could not parse leakage fraction, defaulting to 0.0")
        leakage_fraction = 0.0
        leakage_std      = 0.0

    print(f"\nSimulation Results:")
    print(f"   k-effective:      {keff:.5f} ± {keff_std:.5f}")
    print(f"   Leakage fraction: {leakage_fraction:.5f} ± {leakage_std:.5f} "
          f"({leakage_fraction*100:.2f}%)")

    # ================================================================================
    # 2. READ URANIUM MASS FROM run_params.json
    # ================================================================================

    # params is already the merged dict (cfg.params + run_params.json), so
    # total_HM_mass_kg should be present directly.  Fall back gracefully if not.

    total_HM_mass_kg        = params.get("total_HM_mass_kg",          None)
    fuel_volume_simulated   = params.get("fuel_volume_simulated_cm3", None)
    fuel_volume_full_core   = params.get("fuel_volume_full_core_cm3", None)
    n_trisos                = params.get("n_trisos",                   None)
    use_homogenized_fuel    = params.get("use_homogenized_fuel",       False)
    use_spatial_burnup      = params.get("use_spatial_burnup",         True)
    geometry_factor         = 6 if params.get("use_1/6_geometry", False) else 1

    if total_HM_mass_kg is None or total_HM_mass_kg <= 0:
        # Last-resort: try loading run_params.json directly from the run directory
        params_path = os.path.join(run_dir, 'run_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                saved = json.load(f)
            total_HM_mass_kg      = saved.get("total_HM_mass_kg",          None)
            fuel_volume_simulated = saved.get("fuel_volume_simulated_cm3", None)
            fuel_volume_full_core = saved.get("fuel_volume_full_core_cm3", None)
            n_trisos              = saved.get("n_trisos",                   None)
            use_homogenized_fuel  = saved.get("use_homogenized_fuel",       False)
            use_spatial_burnup    = saved.get("use_spatial_burnup",         True)

    if total_HM_mass_kg is None or total_HM_mass_kg <= 0:
        print("\nERROR: total_HM_mass_kg not found in run_params.json.")
        print("Make sure the simulation completed successfully and run_params.json")
        print("was written by simulation.py before calling post-processing.")
        return None

    print(f"\nFuel Inventory (from run_params.json):")
    print(f"   Fuel representation:          "
          f"{'Homogenized compact' if use_homogenized_fuel else 'Explicit TRISO'}")
    print(f"   Spatial burnup tracking:      {use_spatial_burnup}")
    if n_trisos is not None and not use_homogenized_fuel:
        print(f"   TRISOs per compact per zone:  {n_trisos:,}")
    if fuel_volume_simulated is not None:
        print(f"   Fuel kernel volume (simulated): {fuel_volume_simulated:.4f} cm³")
    if fuel_volume_full_core is not None:
        print(f"   Fuel kernel volume (full core): {fuel_volume_full_core:.4f} cm³")
        if geometry_factor > 1:
            print(f"   Geometry factor:              {geometry_factor} (1/{geometry_factor} symmetry)")
    print(f"   Total uranium mass:           {total_HM_mass_kg:.4f} kg")

    # ================================================================================
    # 3. FUEL CYCLE LENGTH ESTIMATE
    # ================================================================================

    print(f"\n{'='*80}")
    print("FUEL CYCLE LENGTH ESTIMATE")
    print(f"{'='*80}")

    thermal_power_MW = params.get("thermal_power_MW", params.get("thermal_power", 10.0))

    # Total energy available from fuel
    total_energy_MWd = (total_HM_mass_kg / 1000.0) * burnup_limit

    # Cycle length at 100% capacity factor
    cycle_length_days_100pct  = total_energy_MWd / thermal_power_MW
    cycle_length_years_100pct = cycle_length_days_100pct / 365.25

    # Cycle length at 90% capacity factor
    capacity_factor           = 0.90
    cycle_length_days_90pct   = cycle_length_days_100pct * capacity_factor
    cycle_length_years_90pct  = cycle_length_days_90pct / 365.25

    print(f"Maximum TRISO burnup limit:  {burnup_limit:,} MWd/MtU")
    print(f"Reactor thermal power:       {thermal_power_MW} MWth")
    print(f"Total available energy:      {total_energy_MWd:.1f} MWd")
    print(f"")
    print(f"Cycle length (100% capacity factor):")
    print(f"   {cycle_length_days_100pct:.1f} days  ({cycle_length_years_100pct:.2f} years)")
    print(f"")
    print(f"Cycle length (90% capacity factor):")
    print(f"   {cycle_length_days_90pct:.1f} days  ({cycle_length_years_90pct:.2f} years)")

    # ================================================================================
    # 4. SPECIFIC POWER DENSITY
    # ================================================================================

    specific_power_kW_per_kgU = (thermal_power_MW * 1000.0) / total_HM_mass_kg

    print(f"\nSpecific Power:  {specific_power_kW_per_kgU:.2f} kW/kgU")
    print(f"{'='*80}")

    # ================================================================================
    # 5. TRISO PARTICLE COUNT
    # ================================================================================

    kernel_radius_cm   = params.get('kernel_radius', 0.021485)
    kernel_volume_cm3  = (4.0 / 3.0) * np.pi * kernel_radius_cm**3
    uco_density_g_cm3  = params.get("kernel_density", 10820) / 1000.0
    u_mass_fraction    = 238.0 / 268.0
    m_U_per_kernel_g   = kernel_volume_cm3 * uco_density_g_cm3 * u_mass_fraction

    total_trisos_full_core = int((total_HM_mass_kg * 1000.0) / m_U_per_kernel_g)

    print(f"\n{'='*80}")
    print(f"TRISO PARTICLE COUNT  (full core)")
    print(f"{'='*80}")
    print(f"   Kernel radius:          {kernel_radius_cm * 1e4:.2f} μm")
    print(f"   Kernel volume:          {kernel_volume_cm3:.6e} cm³")
    print(f"   U mass per kernel:      {m_U_per_kernel_g * 1e6:.4f} μg")
    print(f"   Total TRISO particles:  {total_trisos_full_core:,}")
    print(f"{'='*80}\n")

    # ================================================================================
    # 6. SAVE RESULTS TO TEXT FILE
    # ================================================================================

    results_file = os.path.join(run_dir, 'simulation_results.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SIMULATION RESULTS AND BURNUP ESTIMATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("Simulation Results:\n")
        f.write(f"   k-effective:      {keff:.5f} ± {keff_std:.5f}\n")
        if leakage_std is not None and leakage_std > 0:
            f.write(f"   Leakage fraction: {leakage_fraction:.5f} ± {leakage_std:.5f} "
                    f"({leakage_fraction*100:.2f}%)\n\n")
        else:
            f.write(f"   Leakage fraction: {leakage_fraction:.5f} "
                    f"({leakage_fraction*100:.2f}%)\n\n")

        f.write("Fuel Inventory:\n")
        f.write(f"   Fuel representation:   "
                f"{'Homogenized compact' if use_homogenized_fuel else 'Explicit TRISO'}\n")
        f.write(f"   Spatial burnup:        {use_spatial_burnup}\n")
        if n_trisos is not None and not use_homogenized_fuel:
            f.write(f"   TRISOs/compact/zone:   {n_trisos:,}\n")
        if fuel_volume_full_core is not None:
            f.write(f"   Fuel kernel volume:    {fuel_volume_full_core:.4f} cm³  (full core)\n")
        if geometry_factor > 1:
            f.write(f"   Geometry factor:       {geometry_factor}  (1/{geometry_factor} symmetry)\n")
        f.write(f"   Uranium mass:          {total_HM_mass_kg:.4f} kg\n\n")

        f.write("TRISO Particle Count (full core):\n")
        f.write(f"   Kernel radius:         {kernel_radius_cm * 1e4:.2f} μm\n")
        f.write(f"   U mass per kernel:     {m_U_per_kernel_g * 1e6:.4f} μg\n")
        f.write(f"   Total TRISO particles: {total_trisos_full_core:,}\n\n")

        f.write("Fuel Cycle Length Estimate:\n")
        f.write(f"   Burnup limit:          {burnup_limit:,} MWd/MtU\n")
        f.write(f"   Thermal power:         {thermal_power_MW} MWth\n")
        f.write(f"   Total energy:          {total_energy_MWd:.1f} MWd\n\n")
        f.write(f"   100% capacity factor:  {cycle_length_days_100pct:.1f} days  "
                f"({cycle_length_years_100pct:.2f} years)\n")
        f.write(f"    90% capacity factor:  {cycle_length_days_90pct:.1f} days  "
                f"({cycle_length_years_90pct:.2f} years)\n\n")

        f.write(f"Specific Power:  {specific_power_kW_per_kgU:.2f} kW/kgU\n")
        f.write("=" * 80 + "\n")

    print(f"Results saved to: {results_file}\n")

    # ================================================================================
    # 7. RETURN ALL CALCULATED RESULTS
    # ================================================================================

    return {
        'keff':                       keff,
        'keff_std':                   keff_std,
        'leakage_fraction':           leakage_fraction,
        'leakage_std':                leakage_std,
        'total_HM_mass_kg':           total_HM_mass_kg,
        'fuel_volume_simulated_cm3':  fuel_volume_simulated,
        'fuel_volume_full_core_cm3':  fuel_volume_full_core,
        'total_trisos_full_core':     total_trisos_full_core,
        'cycle_length_days_100pct':   cycle_length_days_100pct,
        'cycle_length_days_90pct':    cycle_length_days_90pct,
        'specific_power_kW_per_kgU':  specific_power_kW_per_kgU,
    }

# ====================================================================================================
# STANDALONE ENTRY POINT
# ====================================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python burnup_estimation.py <run_directory>")
        print("\nThe run directory must contain run_params.json (written by simulation.py)")
        print("and openmc_output.txt (written during the OpenMC run).")
        sys.exit(1)

    run_dir = sys.argv[1]
    print(f"\nProcessing: {run_dir}")

    params_path = os.path.join(run_dir, 'run_params.json')
    if not os.path.exists(params_path):
        print(f"ERROR: run_params.json not found in {run_dir}")
        sys.exit(1)

    with open(params_path, 'r') as f:
        params = json.load(f)

    run_burnup_estimation(run_dir, params)