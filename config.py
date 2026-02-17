# ====================================================================================================
# GLOBAL PARAMETERS (Most Major Design Variables Live Here)
# ====================================================================================================

# Unless otherwise stated all length dimensions are in cm and densities in kg/m3

params = {
    # ----- Fuel Kernel -----
    "fuel_type": "UCO",
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

    # ----- Hexagonal Lattice -----
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
    "control_insertion": 0.0,                  # Fractional control rod insertion (0-1.0)
    "B10_enrichment_control": 0.6,
    "B10_wt_percent_control": 0.001,
    "B4C_density_control": 2380,
    "Incoloy800H_density": 7940,

    # ----- Temperatures in Kelvin -----
    "coolant_inlet": 573.15,
    "coolant_outlet": 1023.15,
    "compact_min": 973.15,
    "compact_max": 1173.15,
    "matrix_min": 903.15,
    "matrix_max": 1083.15,
    "reflector_min": 903.15,
    "reflector_max": 968.15,

    # ----- OpenMC Settings -----
    "total_batches": 10,
    "inactive_batches": 2,
    "particles": 100_000,

    # ----- Stochastic Volume Calculation Settings -----
    "calculate_fuel_volume": True,
    "volume_samples": 1_000_000_000
}

core_rings = [
    ["r", "f", "f"] * 6,
    ["f", "fc"] * 6,
    ["f"] * 6,
    ["fcp"],
]

# ====================================================================================================
# SINGLE PARAMETRIC STUDY CONNFIGURATION
# ====================================================================================================

parametric_param = None
parametric_values = None

# parametric_param = "triso_pf"
# parametric_values = [0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]

# parametric_param = "fuel_to_coolant_distance"
# parametric_values = [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
# parametric_values = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

# parametric_param = "enrichment"
# parametric_values = [0.075, 0.10, 0.125, 0.15, 0.175, 0.1975]

# parametric_param = "boron_ppm"
# parametric_values = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

# ====================================================================================================
# GRID SEARCH STUDY CONNFIGURATION
# ====================================================================================================

# parametric_param_1 = "triso_pf"
# parametric_values_1 = [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]

# parametric_param_2 = "bundle_pitch"
# parametric_values_2 = [16, 17, 18, 19, 20, 21, 22, 23, 24]