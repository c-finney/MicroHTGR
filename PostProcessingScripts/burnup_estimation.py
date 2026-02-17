"""
Burnup Estimation Post-Processing Script
Estimates fuel cycle length and extracts k_eff and leakage fraction from statepoint files.
"""

import openmc
import numpy as np
import os
import glob


def run_burnup_estimation(run_dir, params, n_trisos=None, burnup_limit=160_000, batch_number=None):
    """
    Estimate fuel cycle length and extract simulation results.
    
    Uranium mass is extracted directly from OpenMC volume calculations if available.
    If no volume calculation exists, only k_eff and leakage are reported.
    
    Args:
        run_dir: Directory containing simulation results
        params: Dictionary of reactor parameters
        n_trisos: Number of TRISO particles (optional, for legacy compatibility)
        burnup_limit: Maximum burnup limit in MWd/MtU (default 160,000)
        batch_number: Batch number for statepoint file (auto-detected if None)
    
    Returns:
        dict: Dictionary containing all calculated values
    """
    
    print(f"\n{'='*80}")
    print("SIMULATION RESULTS EXTRACTION")
    print(f"{'='*80}")
    
    # =========================================================================
    # 1. EXTRACT k_eff AND LEAKAGE FROM STATEPOINT
    # =========================================================================
    
    # Find the statepoint file
    sp_file = None
    for f in os.listdir(run_dir):
        if f.startswith('statepoint') and f.endswith('.h5'):
            sp_file = os.path.join(run_dir, f)
            break
    
    if sp_file is None:
        print("ERROR: No statepoint file found!")
        return None
    
    sp = openmc.StatePoint(sp_file)
    
    # Extract k_eff
    keff = sp.keff.nominal_value
    keff_std = sp.keff.std_dev
    
    # Extract leakage fraction - correct method
    leakage_fraction = None
    leakage_std = None
    
    # Read from the main OpenMC output (where openmc.run() prints)
    # This is typically captured in the console output or redirected files
    output_candidates = [
        os.path.join(run_dir, 'tallies.out'),
        os.path.join(run_dir, 'output.txt'),
        os.path.join(run_dir, 'out'),
        os.path.join(run_dir, 'openmc_output.txt'),
    ]
    
    # Also check for any .out or .log files
    try:
        for file in os.listdir(run_dir):
            if file.endswith('.out') or file.endswith('.log'):
                output_candidates.append(os.path.join(run_dir, file))
    except:
        pass
    
    for output_file in output_candidates:
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    content = f.read()
                    # Look for the leakage fraction line
                    for line in content.split('\n'):
                        if 'Leakage Fraction' in line and '=' in line:
                            # Parse line like: "Leakage Fraction            = 0.23736 +/- 0.00050"
                            parts = line.split('=')
                            if len(parts) > 1:
                                values = parts[1].strip().split('+/-')
                                leakage_fraction = float(values[0].strip())
                                if len(values) > 1:
                                    leakage_std = float(values[1].strip())
                                print(f"Extracted leakage from {os.path.basename(output_file)}: {leakage_fraction:.5f}")
                                break
                if leakage_fraction is not None:
                    break
            except Exception as e:
                continue
    
    # If output wasn't captured to a file, inform user
    if leakage_fraction is None:
        print("WARNING: Could not find leakage fraction in output files!")
        print("Checked files:", [os.path.basename(f) for f in output_candidates if os.path.exists(f)])
        print("\nSetting leakage to 0.0 for now (calculation will continue)")
        leakage_fraction = 0.0
        leakage_std = 0.0
    
    print(f"\nSimulation Results:")
    print(f"   k-effective: {keff:.5f} ± {keff_std:.5f}")
    if leakage_std is not None:
        print(f"   Leakage fraction: {leakage_fraction:.5f} ± {leakage_std:.5f} ({leakage_fraction*100:.2f}%)")
    else:
        print(f"   Leakage fraction: {leakage_fraction:.5f} ({leakage_fraction*100:.2f}%)")
    
    # =========================================================================
    # 2. GET URANIUM MASS FROM OPENMC VOLUME CALCULATION
    # =========================================================================
    
    import warnings
    
    # Suppress OpenMC ID warnings when reloading materials
    warnings.filterwarnings('ignore', category=openmc.IDWarning)
    
    total_HM_mass_kg = None
    total_fuel_volume_cm3 = 0.0
    
    # Look for volume calculation files
    volume_calc_path = os.path.join(run_dir, 'volume_*.h5')
    volume_files = glob.glob(volume_calc_path)
    
    if volume_files:
        try:
            vol_calc_file = volume_files[0]
            print(f"\nVolume calculation .h5 file found: {os.path.basename(vol_calc_file)}")
            
            vol_calc = openmc.VolumeCalculation.from_hdf5(vol_calc_file)
            
            print("\nDomain volumes:")
            
            # Iterate through volumes correctly
            for domain_id, vol_var in vol_calc.volumes.items():
                # Extract nominal value and standard deviation from Variable object
                vol = vol_var.nominal_value
                vol_std = vol_var.std_dev
                
                # Try to get domain name
                domain_name = f"Domain {domain_id}"
                
                # Check if this matches the fuel material by checking params or trying common names
                # For now, assume all volumes in the calculation are fuel (since we specify fuel in vol_calc)
                print(f"   {domain_name}: {vol:.2f} ± {vol_std:.2f} cm³")
                total_fuel_volume_cm3 += vol
            
            # If we found fuel volumes, estimate mass using UCO density
            if total_fuel_volume_cm3 > 0:
                # Account for 1/6 geometry
                geometry_factor = 6 if params.get("use_1/6_geometry", False) else 1
                total_fuel_volume_full_core = total_fuel_volume_cm3 * geometry_factor
                
                # UCO density ~10.97 g/cm³, U mass fraction ~0.888
                uco_density = params.get("kernel_density", 10970) / 1000  # g/cm³
                u_mass_fraction = 238.0 / 268.0  # U in UCO
                
                total_HM_mass_kg = (total_fuel_volume_full_core * uco_density * u_mass_fraction) / 1000
                
                print(f"\nTotal fuel volume (simulated): {total_fuel_volume_cm3:.2f} cm^3")
                if geometry_factor > 1:
                    print(f"Total fuel volume (full core): {total_fuel_volume_full_core:.2f} cm^3\n")
                print(f"UCO density: {uco_density:.3f} g/cm³")
                print(f"Uranium mass fraction in UCO: {u_mass_fraction:.3f}")
                print(f"Estimated uranium mass: {total_HM_mass_kg:.2f} kg")
                print(f"{'='*80}")
                    
        except Exception as e:
            print(f"WARNING: Could not read volume calculation: {e}")
            import traceback
            traceback.print_exc()
            total_HM_mass_kg = None  # Explicitly set to None on error
    else:
        print("\nNo volume calculation file found (volume_*.h5)")
    
    # Reset warnings
    warnings.filterwarnings('default', category=openmc.IDWarning)
    
    # =========================================================================
    # CHECK IF WE HAVE MASS DATA - EARLY EXIT IF NOT
    # =========================================================================
    
    # If we couldn't extract mass directly, inform user and exit early
    if total_HM_mass_kg is None or total_HM_mass_kg <= 0:
        print("\n" + "="*80)
        print("WARNING: Could not extract uranium mass from volume calculation.")
        print("="*80)
        print("\nSkipping burnup estimation (no mass data available).")
        
        # Still save k_eff and leakage results
        results_path = os.path.join(run_dir, 'simulation_results.txt')
        with open(results_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SIMULATION RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"k-effective: {keff:.5f} ± {keff_std:.5f}\n")
            if leakage_std is not None and leakage_std > 0:
                f.write(f"Leakage fraction: {leakage_fraction:.5f} ± {leakage_std:.5f} ({leakage_fraction*100:.2f}%)\n")
            else:
                f.write(f"Leakage fraction: {leakage_fraction:.5f} ({leakage_fraction*100:.2f}%)\n")
            f.write(f"\nNote: Burnup estimation requires stochastic volume calculation.\n")
            f.write("="*80 + "\n")
        
        print(f"\nResults saved to: {results_path}")
        
        return {
            'keff': keff,
            'keff_std': keff_std,
            'leakage_fraction': leakage_fraction,
            'leakage_std': leakage_std,
            'total_HM_mass_kg': None,
            'cycle_length_days_100pct': None,
            'cycle_length_days_90pct': None
        }
    
    # If we get here, we have valid mass data - continue with burnup calculations
    print(f"\n{'='*80}")
    print("FUEL CYCLE LENGTH ESTIMATE")
    print(f"{'='*80}")
    
    # =========================================================================
    # 3. CALCULATE FUEL CYCLE LENGTH
    # =========================================================================
    
    max_burnup_MWd_per_MtU = burnup_limit
    thermal_power_MW = params.get("thermal_power", 15)
    
    # Total energy available from fuel (MWd)
    total_energy_MWd = (total_HM_mass_kg / 1000) * max_burnup_MWd_per_MtU
    
    # Theoretical fuel cycle length at 100% capacity factor
    cycle_length_days_100pct = total_energy_MWd / thermal_power_MW
    cycle_length_years_100pct = cycle_length_days_100pct / 365.25
    
    # Realistic cycle length with 90% capacity factor
    capacity_factor = 0.90
    cycle_length_days_90pct = cycle_length_days_100pct * capacity_factor
    cycle_length_years_90pct = cycle_length_days_90pct / 365.25
    
    print(f"Maximum TRISO burnup limit: {max_burnup_MWd_per_MtU:,} MWd/MtU")
    print(f"Reactor thermal power: {thermal_power_MW} MWth")
    print(f"Total available energy: {total_energy_MWd:.1f} MWd")
    print("")
    print(f"Cycle length (100% capacity factor):")
    print(f"   {cycle_length_days_100pct:.1f} days ({cycle_length_years_100pct:.2f} years)")
    print("")
    print(f"Cycle length (90% capacity factor):")
    print(f"   {cycle_length_days_90pct:.1f} days ({cycle_length_years_90pct:.2f} years)")
    
    # =========================================================================
    # 4. SPECIFIC POWER DENSITY
    # =========================================================================
    
    specific_power_kW_per_kgU = (thermal_power_MW * 1000) / total_HM_mass_kg
    
    print(f"\nSpecific Power:")
    print(f"  {specific_power_kW_per_kgU:.1f} kW/kgU")
    
    print(f"{'='*80}")
    
    # =========================================================================
    # 5. CALCULATE KERNEL PROPERTIES AND TRISO COUNT
    # =========================================================================
    
    # Calculate mass per kernel
    kernel_radius_cm = params.get('kernel_radius', 2.125e-2)  # cm
    kernel_volume_cm3 = (4/3) * np.pi * kernel_radius_cm**3
    uco_density_g_cm3 = params.get("kernel_density", 10970) / 1000  # g/cm³
    kernel_mass_g = kernel_volume_cm3 * uco_density_g_cm3
    u_mass_fraction = 238.0 / 268.0
    m_U_per_kernel_g = kernel_mass_g * u_mass_fraction
    
    # Calculate total number of TRISO particles
    total_trisos_full_core = int((total_HM_mass_kg * 1000) / m_U_per_kernel_g)
    
    print(f"\n{'='*80}")
    print(f"TRISO PARTICLE COUNT")
    print(f"{'='*80}")
    print(f"  Mass per kernel: {m_U_per_kernel_g*1e6:.2f} μg U")
    print(f"  Total TRISO particles: {total_trisos_full_core:,}")
    print(f"{'='*80}\n")
    
    # =========================================================================
    # 6. SAVE RESULTS TO FILE
    # =========================================================================
    
    results_file = os.path.join(run_dir, 'simulation_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SIMULATION RESULTS AND BURNUP ESTIMATION\n")
        f.write("="*80 + "\n")
        
        f.write("Simulation Results:\n")
        f.write(f"   k-effective: {keff:.5f} ± {keff_std:.5f}\n")
        if leakage_std is not None:
            f.write(f"   Leakage fraction: {leakage_fraction:.5f} ± {leakage_std:.5f} ({leakage_fraction*100:.2f}%)\n\n")
        else:
            f.write(f"   Leakage fraction: {leakage_fraction:.5f} ({leakage_fraction*100:.2f}%)\n\n")
        
        f.write("Kernel Properties:\n")
        f.write(f"   Kernel radius: {kernel_radius_cm*1e4:.1f} μm\n")
        f.write(f"   U mass per kernel: {m_U_per_kernel_g*1e6:.2f} μg\n\n")
        
        f.write("Core Inventory:\n")
        if params.get("use_1/6_geometry", False):
            f.write("   (1/6 geometry - values scaled to full core)\n")
        f.write(f"   Total TRISO particles: {total_trisos_full_core:,}\n")
        f.write(f"   Total uranium mass: {total_HM_mass_kg:.2f} kg\n\n")
        
        f.write("Fuel Cycle Length Estimate:\n")
        f.write(f"   Maximum burnup limit: {max_burnup_MWd_per_MtU:,} MWd/MtU\n")
        f.write(f"   Thermal power: {thermal_power_MW} MWth\n")
        f.write(f"   Total energy: {total_energy_MWd:.1f} MWd\n\n")
        
        f.write(f"   100% capacity factor: {cycle_length_days_100pct:.1f} days ({cycle_length_years_100pct:.2f} years)\n")
        f.write(f"   90% capacity factor: {cycle_length_days_90pct:.1f} days ({cycle_length_years_90pct:.2f} years)\n\n")
        
        f.write(f"Specific Power: {specific_power_kW_per_kgU:.1f} kW/kgU\n")
        f.write("="*80 + "\n")
    
    print(f"Results saved to: {results_file}\n")
    
    # Return all calculated values
    return {
        'keff': keff,
        'keff_std': keff_std,
        'leakage_fraction': leakage_fraction,
        'leakage_std': leakage_std,
        'total_HM_mass_kg': total_HM_mass_kg,
        'total_trisos': total_trisos_full_core,
        'cycle_length_days_100pct': cycle_length_days_100pct,
        'cycle_length_days_90pct': cycle_length_days_90pct,
        'specific_power_kW_per_kgU': specific_power_kW_per_kgU
    }


if __name__ == "__main__":
    # Standalone usage
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python burnup_estimation.py <run_directory>")
        print("\nThe script will load parameters from run_params.json in the run directory.")
        print("Uranium mass is extracted from OpenMC volume calculation if available.")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    print(f"\nProcessing: {run_dir}")
    
    # Load run_params.json
    params_path = os.path.join(run_dir, 'run_params.json')
    
    if os.path.exists(params_path):
        print(f"\nLoading parameters from run_params.json...")
        with open(params_path, 'r') as f:
            params = json.load(f)
    else:
        print(f"ERROR: run_params.json not found in {run_dir}")
        print("This file is created automatically when running simulations.")
        sys.exit(1)
    
    run_burnup_estimation(run_dir, params)