"""
Parametric Study Post-Processing Script
Extracts and summarizes results from all cases in a parametric study.
"""

import openmc
import numpy as np
import os
import pandas as pd
import re
import matplotlib.pyplot as plt

def extract_case_results(case_dir, batch_number=None):
    """
    Extract results from a single case directory.
    
    Parameters:
    -----------
    case_dir : str
        Path to case directory
    batch_number : int, optional
        Batch number for statepoint file. If None, finds automatically.
    
    Returns:
    --------
    dict : Dictionary of extracted results, or None if extraction fails
    """
    
    # Find statepoint file if batch not specified
    if batch_number is None:
        for f in os.listdir(case_dir):
            if f.startswith('statepoint') and f.endswith('.h5'):
                batch_number = int(f.split('.')[1])
                break
    
    if batch_number is None:
        return None
    
    sp_path = os.path.join(case_dir, f'statepoint.{batch_number}.h5')
    
    if not os.path.exists(sp_path):
        return None
    
    try:
        sp = openmc.StatePoint(sp_path)
        
        # Get heating tally
        heating_tally = sp.get_tally(name='heating')
        heating_rate_ev = heating_tally.mean[0, 0, 0]
        
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
        
        keff = None
        keff_std = None
        leakage_fraction = None
        leakage_std = None

        output_file = os.path.join(case_dir, 'openmc_output.txt')

        if not os.path.exists(output_file):
            print("ERROR: openmc_output.txt not found!")
            return None

        with open(output_file, 'r') as f:
            content = f.read()

        for line in content.split('\n'):
            if 'k-effective (Collision)' in line and '=' in line:
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
            leakage_std = 0.0
        
        return {
            'keff': keff,
            'keff_std': keff_std,
            'leakage_fraction': leakage_fraction,
            'leakage_std': leakage_std,
            'flux_per_source': flux_per_source,
            'fission_per_source': fission_per_source,
            'nu_fission_per_source': nu_fission_per_source,
            'heating_ev_per_source': heating_rate_ev
        }
        
    except Exception as e:
        print(f"  Error extracting results: {str(e)}")
        return None

def run_parametric_postprocessing(parametric_dir, batch_number=None):
    """
    Run post-processing for an entire parametric study.
    
    Parameters:
    -----------
    parametric_dir : str
        Path to parametric study directory
    batch_number : int, optional
        Batch number for statepoint files
    
    Returns:
    --------
    pd.DataFrame : Results dataframe
    """
    
    print(f"\n{'='*80}")
    print("PARAMETRIC STUDY POST-PROCESSING")
    print(f"{'='*80}")
    print(f"Directory: {parametric_dir}")
    
    POSTPROCESSING_RESULTS_DIR = os.path.join(parametric_dir, "parametric_study_results")
    os.makedirs(POSTPROCESSING_RESULTS_DIR, exist_ok=True)

    # Find all case directories
    case_dirs = []
    for item in os.listdir(parametric_dir):
        item_path = os.path.join(parametric_dir, item)
        if os.path.isdir(item_path) and 'Case' in item:
            case_dirs.append(item_path)
    
    case_dirs.sort()
    
    print(f"Found {len(case_dirs)} case directories")
    
    if len(case_dirs) == 0:
        print("No case directories found!")
        return None
    
    # Extract data from each case
    results = []
    
    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir)
        
        # Parse parameter name and value from directory name
        # Expected format: paramname_Case_XX_value
        match = re.match(r'(.+?)_Case_\d+_([\d.eE+-]+)', case_name)
        
        if not match:
            print(f"  Warning: Could not parse {case_name}, skipping...")
            continue
        
        param_name = match.group(1)
        try:
            param_value = float(match.group(2))
        except ValueError:
            print(f"  Warning: Could not parse value from {case_name}, skipping...")
            continue
        
        print(f"  Processing {case_name}...", end=" ")
        
        case_results = extract_case_results(case_dir, batch_number)
        
        if case_results is None:
            print("FAILED")
            continue
        
        case_results['parameter_name'] = param_name
        case_results['parameter_value'] = param_value
        case_results['case_dir'] = case_dir
        
        results.append(case_results)
        print(f"k_eff = {case_results['keff']:.5f}")
    
    if len(results) == 0:
        print("No results extracted!")
        return None
    
    # Create dataframe
    df = pd.DataFrame(results)
    
    # Sort by parameter value
    df = df.sort_values('parameter_value')
    
    # Save to CSV
    output_csv = os.path.join(POSTPROCESSING_RESULTS_DIR, 'parametric_study_results.csv')
    df.to_csv(output_csv, index=False)
    
    # Generate plots
    generate_parametric_plots(df, POSTPROCESSING_RESULTS_DIR, show_titles=True)
    
    # Print summary
    print(f"\n{'='*80}")
    print("PARAMETRIC STUDY SUMMARY")
    print(f"{'='*80}")
    print(f"Total cases processed: {len(results)}")
    print(f"Results saved to: {output_csv}")
    print(f"\nResults:")
    
    # Print compact table
    display_cols = ['parameter_value', 'keff', 'keff_std', 'leakage_fraction', "leakage_std"]
    print(df[display_cols].to_string(index=False))
    
    print(f"\nk_eff range: {df['keff'].min():.5f} to {df['keff'].max():.5f}")
    print(f"{'='*80}")
    
    return df

def generate_parametric_plots(df, output_dir, show_titles=True):
    """
    Generate plots for parametric study results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    output_dir : str
        Directory to save plots
    """
    
    param_name = df['parameter_name'].iloc[0]
    param_values = df['parameter_value'].values
    keff_values = df['keff'].values
    keff_std = df['keff_std'].values
    leakage = df['leakage_fraction'].values
    
    # Plot k_eff vs parameter
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.errorbar(param_values, keff_values, yerr=keff_std, 
                fmt='o-', capsize=3, capthick=1, markersize=8)
    ax.set_xlabel(param_name.replace('_', ' ') + " (cm)")
    ax.set_ylabel('k-effective')
    if show_titles:
        ax.set_title(f'k-effective vs {param_name.replace("_", " ")}')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at k=1
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='k = 1.0')
    ax.legend()
    
    save_path = os.path.join(output_dir, f'parametric_keff_vs_{param_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Plot leakage vs parameter
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(param_values, leakage * 100, 'o-', markersize=8)
    ax.set_xlabel(param_name.replace('_', ' ') + " (cm)")
    ax.set_ylabel('Leakage Fraction (%)')
    if show_titles:
        ax.set_title(f'Neutron Leakage vs {param_name.replace("_", " ")}')
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, f'parametric_leakage_vs_{param_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Plot reactivity vs parameter
    reactivity = (keff_values - 1) / keff_values * 1e5  # in pcm
    reactivity_std = keff_std / (keff_values**2) * 1e5
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.errorbar(param_values, reactivity, yerr=reactivity_std,
                fmt='o-', capsize=3, capthick=1, markersize=8)
    ax.set_xlabel(param_name.replace('_', ' ') + " (cm)")
    ax.set_ylabel('Reactivity (pcm)')
    if show_titles:
        ax.set_title(f'Reactivity vs {param_name.replace("_", " ")}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='ρ = 0')
    ax.legend()
    
    save_path = os.path.join(output_dir, f'parametric_reactivity_vs_{param_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parametric_postprocessing.py <parametric_study_directory> [batch_number]")
        sys.exit(1)
    
    parametric_dir = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    run_parametric_postprocessing(parametric_dir, batch)