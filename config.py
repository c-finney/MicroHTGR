# ====================================================================================================
# GLOBAL PARAMETERS
# ====================================================================================================

# Unless otherwise stated all lengths are in cm, densities in kg/m3, and temperatures in Kelvin

params = {
    # ----- Fuel Kernel -----
    "fuel_type": "UCO",                        # No other fuel forms currently modeled yet
    "enrichment": 0.1975,                      # U-235 atom fraction
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
    "triso_pf": 0.30,                          # Packing fraction of triso particles in fuel compact

    # ----- Coolant Channel -----
    "n_coolant_channels_per_block": 18,        # Number of coolant channels per assembly
    "coolant_radius": 0.8,
    "coolant_density": 2.873, 

    # ----- Graphite Moderator/Reflector -----
    "boron_ppm": 1.1,
    "matrix_density": 1850,

    # ----- Fuel Lattice -----
    "fuel_to_coolant_distance": 2.5,

    # ----- Core Dimensions -----
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
    "control_radius": 2.54,                    # Radius for reflector assembly control rods
    "fuel_assembly_control_radius": 2.54,      # Radius for circular control rods in fuel assemblies
    "sheath_thickness": 0.1, 
    "guide_tube_thickness": 0.2,  
    "bank_1_insertion": 0.8,                   # Fractional control rod insertion for bank 1 (0-1.0)
    "bank_2_insertion": 0.8,                   # Fractional control rod insertion for bank 2 (0-1.0)
    "bank_3_insertion": 0.8,                   # Fractional control rod insertion for bank 3 (0-1.0)
    "B10_enrichment_control": 0.6,
    "B10_wt_percent_control": 0.001,
    "B4C_density_control": 2380,
    "Incoloy800H_density": 7940,

    # ----- Core Layout -----
    "core_rings": [
        ["r3", "f", "f"] * 6,
        ["f", "fc2"] * 6,
        ["fpa"] * 6,
        ["fcp1"],
    ],
    # Core Ring Assembly Options:
    #     "f"              — Fueled assembly with no control rods or burnable poison rods
    #     "fp"             — Fueled assembly with 6 burnable poison rods on the outer corners of the assembly
    #     "fpa"            — Fueled assembly with 1 burnable poison rod in the center of the assembly
    #     "fc1/fc2/fc3"    — Fueled assembly with 1 central control rod with bank number 1, 2, or 3
    #     "fcp1/fcp2/fcp3" — Fueled assembly with 1 central control rod with bank number 1, 2, or 3 and 6 burnable poison rods on the outer corners of the assembly
    #     "rr"               Reflector block with no control rods
    #     "r1/r2/r3"       — Reflector block with 1 central control rod with bank number 1, 2, or 3
    #     "ra1/ra2/ra3"    — Alt reflector block with 3 control rods in hexagonal ring with bank number 1, 2, or 3 (ONLY WORKS FOR 1/6 GEOMETRY)

    # ----- Temperature Profile -----
    "coolant_inlet": 573.15,
    "coolant_outlet": 1023.15,
    "compact_min": 973.15,
    "compact_max": 1173.15,
    "matrix_min": 903.15,
    "matrix_max": 1083.15,
    "reflector_min": 903.15,
    "reflector_max": 968.15,

    # ----- OpenMC Monte Carlo Settings -----
    "total_batches": 50,
    "inactive_batches": 25,
    "particles": 100_000,

    # ----- Stochastic Volume Calculation Settings -----
    "calculate_fuel_volume": True,
    "volume_samples": 1_000_000_000,

    # ----- Parametric Study Configuration -----
    "parametric_param": "bank_1_insertion",
    "parametric_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

    # ----- Reactivity Coefficient Study Configuration -----
    "reactivity_delta_T_values": [50.0, 100.0, 150.0],
    "reactivity_coefficients": ["FTC", "MTC", "ITC"],

    # ----- Depletion Study Configuration -----
    "thermal_power_MW": 15.0,
    "depletion_chain_file": "/home/cade/Desktop/OpenMC/CrossSections/chain_endfb81_thermal.xml",
    "use_reduced_chain_file": True,
    "depletion_timesteps_days": [10, 10, 10, 30, 30, 30, 60, 60, 60, 120, 120, 120, 120, 120, 120, 180, 180, 180],
    "depletion_integrator": "PredictorIntegrator",
    # Integrator options:
    #     "PredictorIntegrator" — simplest, one transport solve per step
    #     "CECMIntegrator"      — CE/CM predictor-corrector (more accurate, 2× cost)
    #     "CF4Integrator"       — 4th order (most accurate, 4× cost)
    #     "LEQIIntegrator"      — LE/QI (good accuracy, 2× cost)

    # ----- Depletion Nuclide Tracking -----
    "tracked_nuclides": [
        # --- Actinides ---
        "U234",  "U235",  "U236",  "U238",
        "Np237", "Np239",
        "Pu238", "Pu239", "Pu240", "Pu241", "Pu242",
        "Am241", "Am243",
        "Cm242", "Cm243", "Cm244", "Cm245", "Cm246",

        # --- Fission Products ---
        "Kr83",
        "Sr90",
        "Mo95",
        "Tc99",
        "Rh103", "Rh105",
        "Pd107",
        "I135",
        "Xe131", "Xe135", "Xe135_m1",
        "Cs133", "Cs134", "Cs137",
        "Nd143", "Nd145", "Nd147",
        "Pm147", "Pm149",
        "Sm149", "Sm151", "Sm152",
        "Eu153", "Eu154", "Eu155",

        # --- Structural / Moderator ---
        "O16",
        "Si28", "Si29", "Si30",

        # --- Burnable Poison ---
        "B10",
    ],

    # ----- Study Execution Mode Configuration -----
    "run_post_processing": True,
    "study_execution_mode": "SingleStudy"
    # Study Execution Mode Options:
    #     "SingleStudy"     — Singular steady state monte carlo simulation of specified core layout 
    #     "ParametricStudy" — Creates multiple steady state monte carlo simulations varying a single core parameter
    #     "ReactivityStudy" — Calculates reactivity coefficients via multiple steady state monte carlo simulations and specified temperature perturbations
    #     "DepletionStudy"  — Performs depletion run on specified core layout using specified depletion timesteps
}