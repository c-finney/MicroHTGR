"""
Post-Processing Scripts Package

This package contains scripts for post-processing OpenMC simulation results:
- burnup_estimation: Calculates fuel cycle length, k_eff, and leakage
- tally_plotter: Generates flux and fission rate plots
- parametric_postprocessing: Summarizes parametric study results
- spectrum_thermalization: Neutron energy spectrum and thermalization metrics
- reactivity_coefficients_postprocessing: Reactivity coefficient plotting and reporting
- depletion_postprocessing: Depletion simulation results and nuclide tracking
- leakage_spectrum: Neutron leakage energy spectra at reflector boundaries
"""

from .burnup_estimation import run_burnup_estimation
from .tally_plotter import run_tally_plots
from .parametric_postprocessing import run_parametric_postprocessing
from .spectrum_thermalization import run_spectrum_analysis
from .reactivity_coefficients_postprocessing import save_results, plot_results
from .depletion_postprocessing import run_depletion_postprocessing
from .leakage_spectrum import run_leakage_analysis

__all__ = [
    'run_burnup_estimation',
    'run_tally_plots',
    'run_parametric_postprocessing',
    'run_spectrum_analysis',
    'save_results',
    'plot_results',
    'run_depletion_postprocessing',
    'run_leakage_analysis',
]
