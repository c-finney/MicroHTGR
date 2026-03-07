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
    "use_1/6_geometry": False,

    # ----- Burnable Poison -----
    "B10_enrichment_poison": 0.3,
    "B10_wt_percent_poison": 0.001,
    "B4C_density_poison": 2380,

    # ----- Control Rods -----
    "control_radius": 2.54,                  # Radius for reflector assembly control rods
    "fuel_assembly_control_radius": 2.54,    # Radius for circular control rods in fuel assemblies
    "sheath_thickness": 0.1, 
    "guide_tube_thickness": 0.2,  
    "bank_1_insertion": 0.0,                 # Fractional control rod insertion for bank 1 (0-1.0)
    "bank_2_insertion": 0.0,                 # Fractional control rod insertion for bank 2 (0-1.0)
    "bank_3_insertion": 0.0,                 # Fractional control rod insertion for bank 3 (0-1.0)
    "secondary_SD_rods_inserted": False,     # True = all SS rods fully inserted, False = all removed
    "B10_enrichment_control": 0.6,
    "B10_wt_percent_control": 0.448,
    "B4C_density_control": 2380,
    "Incoloy800H_density": 7940,

    # ----- Beryllium Reflector -----
    "use_beryllium_reflector": False,
    "BeO_inner_radius": 70,                # If none defaults to lattice extent (defined as (n_rings-1)*bundle_pitch+bundle_pitch/4)
    "BeO_thickness": 20.0,                   # Will not exceed core radius if inner radius + thickness > core radius
    "BeO_density": 3010,

    # ----- Core Layout -----
    "core_rings": [
        ["rr", "f", "f"] * 6,
        ["f", "fc1"] + ["f", "fc2"] * 5,
        ["fss"] * 6,
        ["fssp"],
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
    "reflector_min": 903.15,
    "reflector_max": 968.15,

    # ----- Tally Configuration -----
    "n_XY_mesh_zones_full_core": 500,
    "use_global_tallies": True,
    "use_mesh_tallies": True,
    "use_leakage_tallies": True,
    "use_BeO_tallies": True,

    # ----- OpenMC Monte Carlo Settings -----
    "total_batches": 20,
    "inactive_batches": 10,
    "particles": 100_000,

    # ----- Stochastic Volume Calculation Settings -----
    "calculate_fuel_volume": False,
    "volume_samples": 1_000_000_000,

    # ----- Parametric Study Configuration -----
    "parametric_param": "BeO_thickness",
    "parametric_values": [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],

    # ----- Reactivity Coefficient Study Configuration -----
    "reactivity_delta_T_values": [50.0, 100.0, 150.0],
    "reactivity_coefficients": ["FTC", "MTC", "ITC"],

    # ----- Depletion Study Configuration -----
    "thermal_power_MW": 10.0,
    # "depletion_chain_file": "/home/cade/Desktop/OpenMC/CrossSections/chain_endfb81_thermal.xml",
    "depletion_chain_file": "/home/cade/Desktop/OpenMC/CrossSections/chain_casl_pwr.xml",
    "use_reduced_chain_file": True,
    "depletion_timesteps_days": [10, 10, 10, 30, 30, 30, 60, 60, 60, 120, 120, 120, 180, 180, 180],
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
        "Pu240", "Pu242",
        
        # --- Minor Actinides ---
        "Np237", "Np239", "Pu238",
        "Am241", "Am243",
        "Cm242", "Cm243", "Cm244", "Cm245", "Cm246",

        # --- Xe/I Poisons ---
        "Xe131", "Xe135", "Xe135_m1",
        "I135",
        
        # --- Sm/Pm Poisons ---
        "Sm149", "Sm151", "Sm152",
        "Pm147", "Pm149",
        
        # --- Cs/Sr Fission Products ---
        "Cs133", "Cs134", "Cs137",
        "Sr90",

        # --- Nd/Eu Fission Products ---
        "Nd143", "Nd145", "Nd147",
        "Eu153", "Eu154", "Eu155",
        
        # --- Mo/Tc/Rh/Pd Fission Products ---   
        "Mo95",
        "Tc99",
        "Rh103", "Rh105",
        "Pd107",
        
        # ----- Kr Fission Products -----
        "Kr83",

        # --- Burnable Poison ---
        "B10",
    ],
    "poison_tracked_nuclides": ["B10"],    # Nuclides to search for in the burnable poison material exclusively

    # ----- Depletion Post-Processing Plot Groups -----
    "depletion_plot_groups": {
        "Fissile Actinides":    ["U235",
                                 "Pu239", "Pu241"],
        "Fertile Actinides":    ["U238", "U234", "U236",
                                 "Pu238", "Pu240", "Pu242"],
        "Minor Actinides":      ["Np237", "Np239",
                                 "Am241", "Am243",
                                 "Cm242", "Cm243", "Cm244", "Cm245", "Cm246"],
        "Xe/I Poisons":         ["Xe131", "Xe135", "Xe135_m1",
                                 "I135"],
        "Sm/Pm Poisons":        ["Sm149", "Sm151", "Sm152",
                                 "Pm147", "Pm149"],
        "Cs/Sr FPs":            ["Cs133", "Cs134", "Cs137",
                                 "Sr90"],
        "Nd/Eu FPs":            ["Nd143", "Nd145", "Nd147",
                                 "Eu153", "Eu154", "Eu155"],
        "Mo/Tc/Rh/Pd FPs":      ["Mo95",
                                 "Tc99",
                                 "Rh103", "Rh105",
                                 "Pd107"],
        "Kr FPs":               ["Kr83"],
        "Boron Poisons":        ["B10"],
    },

    # ----- Study Execution Mode Configuration -----
    "study_execution_mode": "SingleStudy",
    # Study Execution Mode Options:
    #    "SingleStudy"     — Singular steady state monte carlo simulation of specified core layout 
    #    "ParametricStudy" — Creates multiple steady state monte carlo simulations varying a single core parameter
    #    "ReactivityStudy" — Calculates reactivity coefficients via multiple steady state monte carlo simulations and specified temperature perturbations
    #    "DepletionStudy"  — Performs depletion run on specified core layout using specified depletion timesteps
    "run_post_processing": True, # Note this controls individual study post-processing for ParametricStudy runs as parametric post-processing is always run

}