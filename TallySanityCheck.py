import openmc
import numpy as np
import os

BASE_DIR = '/home/cade/Desktop/OpenMC/SeniorDesign/MicroHTGR_Output/htgr_run_00.37.19_01.20.2026'
batch_number = 100
target_power_MW = 5.0  # target reactor power

sp_path = os.path.join(BASE_DIR, f'statepoint.{batch_number}.h5')
sp = openmc.StatePoint(sp_path)

# ============================================================
# MANUAL POWER NORMALIZATION
# ============================================================

# Get heating tally for normalization
heating_tally = sp.get_tally(name='heating')
heating_rate_ev = heating_tally.mean[0, 0, 0]  # eV/source particle

# Convert to J/source
joule_per_ev = 1.60218e-19
heating_rate_j = heating_rate_ev * joule_per_ev  # J/source

# Calculate source rate [source particles/sec]
power_watts = target_power_MW * 1e6
source_per_sec = power_watts / heating_rate_j

print(f"\n{'='*70}")
print(f"MANUAL POWER NORMALIZATION")
print(f"{'='*70}")
print(f"Heating rate: {heating_rate_ev:.3e} eV/source")
print(f"Heating rate: {heating_rate_j:.3e} J/source")
print(f"Target power: {target_power_MW:.3f} MW")
print(f"Source rate: {source_per_sec:.3e} source particles/s")
print(f"{'='*70}\n")

# ============================================================
# GET GLOBAL TALLIES AND SCALE
# ============================================================

global_tally = sp.get_tally(name='global_rates')
scores = global_tally.scores

flux_idx = scores.index('flux')
fission_idx = scores.index('fission')
nu_fission_idx = scores.index('nu-fission')

mean = global_tally.mean[0, 0, :]

# Per source particle values
total_flux_per_source = mean[flux_idx]
total_fission_per_source = mean[fission_idx]
total_nu_fission_per_source = mean[nu_fission_idx]

# Scale to physical units (per second)
total_flux = total_flux_per_source * source_per_sec
total_fission_rate = total_fission_per_source * source_per_sec
total_nu_fission = total_nu_fission_per_source * source_per_sec

# ============================================================
# VERIFY POWER CALCULATION
# ============================================================

# Calculate power from fission rate (verification)
energy_per_fission = 200e6 * joule_per_ev  # 200 MeV in Joules
power_from_fission_watts = total_fission_rate * energy_per_fission
power_from_fission_MW = power_from_fission_watts / 1e6

# Calculate power from heating tally (should match exactly)
power_from_heating_MW = heating_rate_ev * 1e-6 * source_per_sec * joule_per_ev  # MW

print(f"{'='*70}")
print(f"GLOBAL REACTION RATES (Batch {batch_number})")
print(f"{'='*70}")
print(f"k-effective: {sp.keff.nominal_value:.5f} ± {sp.keff.std_dev:.5f}")
print(f"\nPer Source Particle (before normalization):")
print(f"  Flux: {total_flux_per_source:.3e} n·cm/source")
print(f"  Fission: {total_fission_per_source:.3e} fissions/source")
print(f"  Nu-Fission: {total_nu_fission_per_source:.3e} neutrons/source")
print(f"\nPhysical Units (after normalization):")
print(f"  Total Flux: {total_flux:.3e} n/(cm² · s)")
print(f"  Total Fission Rate: {total_fission_rate:.3e} fissions/s")
print(f"  Total Nu-Fission Rate: {total_nu_fission:.3e} neutrons/s")
print(f"\nPower Verification:")
print(f"  Target power: {target_power_MW:.3f} MW")
print(f"  Power from fission (200 MeV est.): {power_from_fission_MW:.3f} MW")
print(f"  Power from heating tally: {power_from_heating_MW:.3f} MW")
print(f"{'='*70}\n")

# ============================================================
# CHECK MESH TALLY TOTALS
# ============================================================

mesh_tally = sp.get_tally(name='mesh_rates')
mesh_flux_per_source = mesh_tally.mean[:, 0, flux_idx]
mesh_fission_per_source = mesh_tally.mean[:, 0, fission_idx]

# Scale mesh tallies
mesh_flux = mesh_flux_per_source * source_per_sec
mesh_fission = mesh_fission_per_source * source_per_sec

print(f"{'='*70}")
print(f"MESH TALLY STATISTICS:")
print(f"{'='*70}")
print(f"Number of mesh cells: {len(mesh_flux)}")
print(f"\nPer Source Particle:")
print(f"  Sum of mesh flux: {mesh_flux_per_source.sum():.3e} n·cm/source")
print(f"  Sum of mesh fission: {mesh_fission_per_source.sum():.3e} fissions/source")
print(f"  Average flux per cell: {mesh_flux_per_source.mean():.3e} n/(cm²)/source")
print(f"\nPhysical Units:")
print(f"  Sum of mesh flux: {mesh_flux.sum():.3e} n·cm/s")
print(f"  Sum of mesh fission: {mesh_fission.sum():.3e} fissions/s")
print(f"  Average flux per cell: {mesh_flux.mean():.3e} n/(cm²·s)")
print(f"  Min flux: {mesh_flux.min():.3e} n/(cm²·s)")
print(f"  Max flux: {mesh_flux.max():.3e} n/(cm²·s)")
print(f"  Min fission: {mesh_fission.min():.3e} fissions/s")
print(f"  Max fission: {mesh_fission.max():.3e} fissions/s")
print(f"{'='*70}\n")

# Save normalization factor for use in plotting script
normalization_data = {
    'source_per_sec': source_per_sec,
    'heating_rate_ev': heating_rate_ev,
    'target_power_MW': target_power_MW,
    'batch': batch_number
}

np.save(os.path.join(BASE_DIR, 'normalization_factor.npy'), normalization_data)
print(f"Normalization factor saved to: {os.path.join(BASE_DIR, 'normalization_factor.npy')}")