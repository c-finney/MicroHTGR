import openmc
import numpy as np
import os
import pandas as pd
import re

# Main parametric study directory
PARAMETRIC_DIR = '/home/cade/Desktop/OpenMC/SeniorDesign/MicroHTGR_Output/htgr_run_00.15.33_01.21.2026_ParametricStudy_boron_ppm'
batch_number = 50

# ============================================================
# FIND ALL CASE DIRECTORIES
# ============================================================

case_dirs = []
for item in os.listdir(PARAMETRIC_DIR):
    item_path = os.path.join(PARAMETRIC_DIR, item)
    if os.path.isdir(item_path) and 'Case' in item:
        case_dirs.append(item_path)

case_dirs.sort()

print(f"Found {len(case_dirs)} case directories")

# ============================================================
# EXTRACT DATA FROM EACH CASE
# ============================================================

results = []

for case_dir in case_dirs:
    case_name = os.path.basename(case_dir)
    
    # Parse parameter name and value from directory name
    # Expected format: paramname_Case_XX_value
    match = re.match(r'(.+?)_Case_\d+_([\d.]+)', case_name)
    
    if not match:
        print(f"Warning: Could not parse {case_name}, skipping...")
        continue
    
    param_name = match.group(1)
    param_value = float(match.group(2))
    
    sp_path = os.path.join(case_dir, f'statepoint.{batch_number}.h5')
    
    if not os.path.exists(sp_path):
        print(f"Warning: No statepoint file found in {case_name}, skipping...")
        continue
    
    try:
        sp = openmc.StatePoint(sp_path)
        
        # Get keff
        keff = sp.keff.nominal_value
        keff_std = sp.keff.std_dev
        
        # Get heating tally
        heating_tally = sp.get_tally(name='heating')
        heating_rate_ev = heating_tally.mean[0, 0, 0]  # eV/source
        
        # Get global tallies
        global_tally = sp.get_tally(name='global_rates')
        scores = global_tally.scores
        
        flux_idx = scores.index('flux')
        fission_idx = scores.index('fission')
        nu_fission_idx = scores.index('nu-fission')
        
        mean = global_tally.mean[0, 0, :]
        
        flux_per_source = mean[flux_idx]
        fission_per_source = mean[fission_idx]
        nu_fission_per_source = mean[nu_fission_idx]
        
        # Store results
        results.append({
            'parameter_name': param_name,
            'parameter_value': param_value,
            'keff': keff,
            'keff_std': keff_std,
            'flux_per_source': flux_per_source,
            'fission_per_source': fission_per_source,
            'nu_fission_per_source': nu_fission_per_source,
            'heating_ev_per_source': heating_rate_ev
        })
        
        print(f"Processed {case_name}: param={param_value}, keff={keff:.5f}")
        
    except Exception as e:
        print(f"Error processing {case_name}: {str(e)}")
        continue

# ============================================================
# CREATE DATAFRAME AND SAVE TO CSV
# ============================================================

df = pd.DataFrame(results)

# Sort by parameter value
df = df.sort_values('parameter_value')

# Save to CSV
output_csv = os.path.join(PARAMETRIC_DIR, 'parametric_study_results.csv')
df.to_csv(output_csv, index=False)

print(f"\n{'='*80}")
print(f"PARAMETRIC STUDY SUMMARY")
print(f"{'='*80}")
print(f"Total cases processed: {len(results)}")
print(f"Results saved to: {output_csv}")
print(f"\nPreview of results:")
print(df.to_string(index=False))
print(f"{'='*80}")