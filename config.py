# ====================================================================================================
# GLOBAL PARAMETERS
# ====================================================================================================

# Unless otherwise stated all lengths are in cm, densities in kg/m3, and temperatures in Kelvin

params = {
    # ----- Fuel Kernel -----
    "fuel_type": "UCO",           # No other fuel forms currently modeled yet
    "enrichment": 0.1975,         # U-235 atom fraction
    "kernel_radius": 0.021485,
    "kernel_density": 10820, 

    # ----- TRISO Layers -----
    "buffer_thickness": 0.01,
    "ipyc_thickness": 0.004,
    "sic_thickness": 0.0035,
    "opyc_thickness": 0.004,
    "buffer_density": 1050,
    "pyc_density": 1900,
    "sic_density": 3203,

    # ----- Fuel Compact  -----
    "compact_radius": 0.635,
    "compact_height": 4.93,
    "triso_pf": 0.30,           # Packing fraction of triso particles in fuel compact

    # ----- Coolant Channel -----
    "coolant_radius": 0.8,
    "coolant_density": 2.873, 

    # ----- Graphite Moderator/Reflector -----
    "boron_ppm": 1.1,
    "matrix_density": 1850,

    # ----- Fuel Lattice -----
    "fuel_to_coolant_distance": 2.5,

    # ----- Core Dimensions/Geometry -----
    "core_radius": 90.0,
    "core_height": 237.9,
    "reflector_thickness": 79.3, 
    "n_ax_zones": 50,
    "use_1/6_geometry": True,

    # ----- Burnable Poison -----
    "B10_enrichment_poison": 0.3,
    "B10_wt_percent_poison": 0.001,
    "B4C_density_poison": 2380,

    # ----- Control Rods -----
    "control_radius": 2.54,                  # Radius for reflector assembly control rod absorber material
    "fuel_assembly_control_radius": 2.54,    # Radius for fuel assembly control rod absorber material
    "sheath_thickness": 0.1, 
    "guide_tube_thickness": 0.2,  
    "bank_1_insertion": 0.0,                 # Fractional control rod insertion for bank 1 (0-1.0)
    "bank_2_insertion": 0.0,                 # Fractional control rod insertion for bank 2 (0-1.0)
    # "bank_1_insertion": 1.0,                 # Critical control rod insertion for bank 1 (0-1.0)
    # "bank_2_insertion": 0.734336,            # Critical control rod insertion for bank 2 (0-1.0)
    "bank_3_insertion": 0.0,                 # Fractional control rod insertion for bank 3 (0-1.0)
    "B10_enrichment_control": 0.9,
    "B10_wt_percent_control": 0.687,
    "B4C_density_control": 2380,
    "Incoloy800H_density": 7940,

    # ----- Secondary Shutdown System -----
    "secondary_SD_rods_inserted": False,       # True = all SS rods fully inserted, False = all removed
    "b4c_ss_pf": 0.55,                        # B4C sphere volume fraction in secondary shutdown system
    "b4c_ss_sphere_radius": 0.5,              # B4C sphere radius for secondary shutdown system
    "use_homogenized_SS_rods": False,         # True = homogenized B4C+He mix, False = explicit B4C spheres

    # ----- Beryllium Reflector -----
    "use_BeO_reflector": True,
    "BeO_inner_radius": 70,                  # If none defaults to lattice extent (defined as (n_rings-1)*bundle_pitch+bundle_pitch/4)
    "BeO_thickness": 20.0,                   # Will not exceed core radius if inner radius + thickness > core radius
    "BeO_density": 3010,

    # ----- Core Layout -----
    "core_rings": [
        ["rr", "f", "f"] * 6,
        ["f", "fc2"] * 6,
        ["fss"] * 6,
        ["fcp1"],
    ],
    # Core Ring Assembly Options:
    #    "f"              — Fueled assembly with no control rods or burnable poison rods
    #    "fp"             — Fueled assembly with 6 burnable poison rods on the outer corners of the assembly
    #    "fpa"            — Fueled assembly with 1 burnable poison rod in the center of the assembly
    #    "fc1/fc2/fc3"    — Fueled assembly with 1 central control rod with bank number 1, 2, or 3
    #    "fcp1/fcp2/fcp3" — Fueled assembly with 1 central control rod with bank number 1, 2, or 3 and 6 burnable poison rods on the outer corners of the assembly
    #    "fss"            — Fueled assembly with 1 central secondary shutdown rod
    #    "fssp"           — Fueled assembly with 1 central secondary shutdown rod and 6 burnable poison rods on the outer corners
    #    "rr"             — Reflector block with no control rods
    #    "r1/r2/r3"       — Reflector block with 1 central control rod with bank number 1, 2, or 3
    #    "ra1/ra2/ra3"    — Alt reflector block with 3 control rods in hexagonal ring with bank number 1, 2, or 3 (ONLY WORKS FOR 1/6 GEOMETRY)
    #    "rss"            — Reflector block with 1 central secondary shutdown rod
    #    "rssa"           — Alt reflector block with 3 secondary shutdown rods in hexagonal ring (ONLY WORKS FOR 1/6 GEOMETRY)

    # ----- Temperature Profile -----
    "coolant_inlet": 573.15,
    "coolant_outlet": 1023.15,
    "compact_min": 973.15,
    "compact_max": 1173.15,
    "matrix_min": 903.15,
    "matrix_max": 1083.15,
    # NOTE: reflector temps are no longer separate parameters.
    # Top/bottom axial reflectors use matrix_max/matrix_min respectively.
    # Radial reflector cells use the matrix axial profile (T_matrix_z).

    # ----- Tally Configuration -----
    "n_XY_mesh_zones_full_core": 200,
    "use_global_tallies": True,          # Full tally set: flux spectrum, heating-local, global rates
    "use_mesh_tallies": True,
    "use_mesh_heating_tally": False,     # Active-core-only mesh heating tally used by TH coupler iterations (cheaper than full mesh)
    "use_mesh_heating_full_tally": False, # Full-core mesh heating tally without flux/fission scores (placeholder, currently unused)
    "use_leakage_tallies": True,
    "use_BeO_tallies": True,

    # ----- OpenMC Monte Carlo Settings -----
    "total_batches": 50,
    "inactive_batches": 20,
    "particles": 50_000,

    # ----- Geometry Plots -----
    "make_geometry_plots": True,
    "plot_threads": 128,         # OpenMP threads used by openmc --plot

    # ----- Spatial Burnup Resolution -----
    "ax_zones_per_burnup_region": 10, # Number of axial zones per burnup region (must be an integer and factor of n_ax_zones), note there are as many radial zones as there are rings in core_rings
    "use_spatial_burnup": True,       # Adds axial and radial burnup zones and heating-local tallies for each zone to calculate min/max burnup

    # ----- RPT Homogenization (Reactivity Equivalent Physical Transform) -----
    # Set use_homogenized_fuel=True to activate the two-region RPT model.
    # rpt_radius must be calibrated first via study_execution_mode="RPTCalibration".
    #
    # Physics: inner cylinder of radius rpt_radius is filled with a homogenized
    # mixture of all TRISO layers + proportional graphite at effective packing
    # fraction pf_inner = triso_pf * (compact_radius / rpt_radius)^2.
    # The outer annulus (rpt_radius < r < compact_radius) is pure graphite.
    #
    # Calibration scan range: [compact_radius*sqrt(triso_pf), compact_radius]
    #   Lower bound: maximum self-shielding (all TRISO volume in inner cylinder)
    #   Upper bound: flat homogenization (no benefit over simple mixing)
    "use_homogenized_fuel": True,
    "rpt_radius": 0.44636,                   # Calibrated inner cylinder radius (cm). None = not yet calibrated.
    "rpt_calibration_max_iter": 20,       # Maximum Illinois interpolation iterations in RPTCalibration study.
    "rpt_calibration_k_tol": 0.0002,       # k_eff convergence tolerance for RPTCalibration study.

    # ----- Parametric Study Configuration -----
    "parametric_param": "BeO_thickness",
    "parametric_values": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    # "parametric_param": "bank_3_insertion",
    # "parametric_values": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    #                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],

    # ----- Reactivity Coefficient Study Configuration -----
    "reactivity_delta_T_values": [50.0, 100.0, 150.0],
    "reactivity_coefficients": ["FTC", "MTC", "ITC"],

    # ----- Thermal-Hydraulics Coupler Convergence Settings -----
    "th_coupler_k_tol": 0.002,        # k convergence tolerance (1 beta for U-235)
    "th_coupler_q_tol_frac": 0.05,     # max |Δq|/max(q) tolerance for heating profile
    "th_coupler_ignore_keff": False,   # if True, skip k_eff convergence check and converge on heating profile only
    "th_coupler_min_iter": 4,          # minimum iterations before checking convergence
    "th_coupler_max_iter": 10,         # maximum iterations before breaking
    "th_coupler_batches": 50,          # active batches per th_coupler eigenvalue run
    "th_coupler_inactive": 20,         # inactive batches per th_coupler eigenvalue run
    "th_coupler_particles": 50_000,    # particles per th_coupler eigenvalue run

    # ----- nc_htgr Channel Parameters (used by th_coupler) -----
    # Computed from geometry: L_m = (core_height + 2*reflector_thickness)*0.01
    # L_heated_m = core_height*0.01, D_cool_m = 2*coolant_radius*0.01,
    # D_compact_m = 2*compact_radius*0.01, pitch_m = fuel_to_coolant_distance*0.01,
    # packing_fraction = triso_pf  — all derived automatically from params above.
    "th_m_dot_kg_s": 0.0097,           # per-channel helium mass flow rate [kg/s]
    "th_P_in_Pa": 4.6e6,               # core inlet helium pressure [Pa]
    "th_N_fuel_channels": 1218,        # total fuel channels in full core
    "th_N_cool_channels": 558,         # total coolant channels in full core
    "th_D_fuel_hole_m": 0.0127,        # fuel hole bore diameter [m]
    "th_roughness_m": 1.0e-5,          # coolant channel wall roughness [m]
    "th_emiss_compact": 0.85,          # compact outer surface emissivity
    "th_emiss_fuel_hole": 0.85,        # fuel hole wall emissivity
    "th_k_compact_eff_W_mK": 6.0,      # effective compact thermal conductivity [W/m·K]
    "th_flow_upward": False,           # False = downward flow (inlet at top)
    "th_N_nodes": 200,                 # nc_htgr axial node count per channel solve
    "th_n_fuel_adj_to_cool": 6,        # fuel compacts sharing heat with each coolant channel
    "th_n_cool_adj_to_fuel": 3,        # coolant channels shared by each fuel compact
    "th_graphite_k_model": "pcea_table",  # graphite conductivity model for nc_htgr

    # ----- Critical Rod Search Configuration -----
    "critical_search_k_tol": 0.002,
    "critical_search_max_iter": 10,
    "critical_search_batches": 50,
    "critical_search_inactive": 20,
    "critical_search_particles": 50_000,

    # ----- Graphite Depletion -----
    "deplete_graphite": True,               # Deplete graphite as a single lumped core-averaged material
    "graphite_volume_particles": 10_000_000, # Stochastic volume calculation samples (auto-enabled when deplete_graphite=True)

    # ----- Depletion Study Configuration -----
    "thermal_power_MW": 10.0,
    # "depletion_chain_file": "/home/cade/Desktop/OpenMC/CrossSections/chain_endfb81_thermal.xml",
    "depletion_chain_file": "/home/cade/Desktop/OpenMC/CrossSections/chain_casl_pwr.xml",
    "use_reduced_chain_file": False,
    # "depletion_timesteps_days": [1, 3, 3, 3, 10, 10, 10, 30, 30, 30, 60, 60, 60, 120, 120, 120, 180, 180, 180], # Normal depletion time steps
    "depletion_timesteps_days": [1, 3, 3, 3, 10, 10, 10, 30, 30, 30, 60, 60, 60, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90], # CS depletion time steps
    # "depletion_timesteps_days": [1, 3, 3, 3],  # Testing timme steps
    "depletion_integrator": "PredictorIntegrator",
    # Integrator options:
    #    "PredictorIntegrator" — simplest, one transport solve per step
    #    "CECMIntegrator"      — CE/CM predictor-corrector (more accurate, 2× cost)
    #    "CF4Integrator"       — 4th order (most accurate, 4× cost)
    #    "LEQIIntegrator"      — LE/QI (good accuracy, 2× cost)

    # ----- Depletion Restart Configuration -----
    "restart_depletion": False,
    "restart_run_dir": "/home/cade/Desktop/OpenMC/SeniorDesign/MicroHTGR_Output/htgr_run_02.23.2026_14.00.15_Depletion",
    "restart_timesteps_days": [120], # Remaining timesteps to run (replaces original list)
    # To use restart:
    #    1. Set restart_depletion = True
    #    2. Set restart_run_dir to the path of the failed run directory
    #    3. Set restart_timesteps_days to whatever timesteps remain
    #    4. Set study_execution_mode = "DepletionStudy"
    #    5. Run — results will be appended to the existing depletion_results.h5

    # ----- Depletion Nuclide Tracking -----
    "tracked_nuclides": [
        # --- Fissile Actinides ---
        "U235",
        "Pu239", "Pu241",

        # --- Fertile Actinides ---
        "U238", "U234", "U236",
        "Pu238", "Pu240", "Pu242",

        # --- Minor Actinides ---
        "Np237", "Np239",
        "Am241", "Am243",
        "Cm242", "Cm244",

        # --- FP Poisons ---
        "Xe131", "Xe135",
        "I235",
        "Pm147", "Pm149",
        "Sm149", "Sm151", "Sm152",

        # --- Other FPs ---
        "Kr83",
        "Sr90",
        "Mo95",
        "Tc99",
        "Rho103",
        "Cs134", "Cs137",
        "Ce144",
        "Nd143", "Nd145",
        "Eu152", "Eu154", "Eu155",

        # --- Burnable Poison ---
        "B10"
    ],
    "poison_tracked_nuclides": ["B10"],    # Nuclides to search for in the burnable poison material exclusively

    # ----- Depletion Post-Processing Plot Groups -----
    "depletion_plot_groups": {
            "Fissile Actinides": ["U235",
                                  "Pu239", "Pu241"],
            "Fertile Actinides": ["U238", "U234", "U236",
                                  "Pu238", "Pu240", "Pu242"],
            "Minor Actinides":   ["Np237", "Np239",
                                  "Am241", "Am243",
                                  "Cm242", "Cm244"],
            "FP Poisons":        ["Xe131", "Xe135",
                                  "I135",
                                  "Pm147", "Pm149",
                                  "Sm149", "Sm151", "Sm152"],
            "Other FPs":         ["Kr83",
                                  "Sr90",
                                  "Mo95",
                                  "Tc99",
                                  "Rh103",
                                  "Cs134", "Cs137",
                                  "Ce144",
                                  "Nd143", "Nd145",
                                  "Eu152", "Eu154", "Eu155"],
            "Boron Poisons":     ["B10"]
    },

    # ----- Depletion EOL Gamma Sources of Interest -----
    "gamma_sources": [
        "H3",
        "C14",
        "Na22",
        "Cl36",
        "Mn54",
        "Co57", "Co60",
        "Ni59", "Ni63",
        "Sr90",
        "Cs134", "Cs137",
        "Ce144",
        "Eu152", "Eu154", "Eu155", 
        "U238",
        "Pu238", "Pu239",
        "Am241", "Am243"
    ],

    # ----- Study Execution Mode Configuration -----
    "study_execution_mode": "CSDepletionStudy",
    # Study Execution Mode Options:
    #    "SingleStudy"      — Singular steady state monte carlo simulation of specified core layout
    #    "ParametricStudy"  — Creates multiple steady state monte carlo simulations varying a single core parameter
    #    "ReactivityStudy"  — Calculates reactivity coefficients via multiple steady state monte carlo simulations and specified temperature perturbations
    #    "CriticalSearch"   — Performs critical rod position search, first inserting bank 1 all the way and then inserting bank 2 until critical
    #    "DepletionStudy"   — Performs all rods out depletion run on specified core layout using specified depletion timesteps
    #    "CSDepletionStudy" — Performs depletion run on specified core layout moving control rods to obtain criticality at each depletion timestep
    #    "RPTCalibration"   — Finds the RPT inner radius (rpt_radius) that matches explicit-TRISO k_eff.
    #                         Runs one explicit-TRISO reference, brackets with r_min/r_max endpoints, then
    #                         iterates with Illinois regula falsi until |Δk| < rpt_calibration_k_tol.
    #                         After running, set rpt_radius in this file to the reported optimal value.
    "run_post_processing": True, # Note this controls individual study post-processing for ParametricStudy runs as parametric post-processing is always run
    "show_titles": True,         # Determines whether titles should be shown on plots generated during study
}