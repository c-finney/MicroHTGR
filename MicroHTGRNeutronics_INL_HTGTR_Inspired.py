import os
import math
import shutil
import openmc
import numpy as np
import openmc.deplete
from datetime import datetime
import sys
import subprocess

# Add parent directory to path to find modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config as cfg
import materials as mats
import assembly as asm
import trisos

# Add PostProcessingScripts to path
POST_PROCESSING_DIR = os.path.join(SCRIPT_DIR, "PostProcessingScripts")
if os.path.exists(POST_PROCESSING_DIR):
    sys.path.insert(0, POST_PROCESSING_DIR)

cross_sections_path = '/home/cade/Desktop/OpenMC/CrossSections/cross_sections.xml'
os.environ['OPENMC_CROSS_SECTIONS'] = cross_sections_path

# ====================================================================================================
# PRE-PROCESSING FUNCTIONS
# ====================================================================================================

def save_params(run_dir, params):
    """
    Save simulation parameters to JSON file at start of run.
    
    Args:
        run_dir: Directory to save to
        params: Simulation parameters dictionary
    """
    import json
    
    # Filter to JSON-serializable types
    params_serializable = {}
    for k, v in params.items():
        if isinstance(v, (int, float, bool, str, list)):
            params_serializable[k] = v
        elif isinstance(v, np.ndarray):
            params_serializable[k] = v.tolist()
        elif v is None:
            params_serializable[k] = None
    
    params_path = os.path.join(run_dir, 'run_params.json')
    with open(params_path, 'w') as f:
        json.dump(params_serializable, f, indent=2)
    
    print(f"Saved parameters to: {params_path}")

# ====================================================================================================
# MAIN SIMULATION FUNCTION
# ====================================================================================================

def run_simulation(params, core_rings, run_dir):
    """
    Run the OpenMC simulation.
    
    Returns:
        n_trisos: Number of TRISO particles per axial zone
    """

    # ====================================================================================================
    # CREATE RUN DIRECTORY AND INITIALIZE MODEL
    # ====================================================================================================

    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    # Save params to run directory for post-processing
    save_params(run_dir, params)

    # Save cross_sections.xml file to run directory
    shutil.copy2(cross_sections_path, os.path.join(run_dir, 'cross_sections.xml'))

    model = openmc.model.Model()
    
    # ====================================================================================================    
    # INITIALIZE TEMPERATURE PROFILES
    # ====================================================================================================

    # Helper function for cosine temperature distribution
    def cosine_temp_profile(T_min, T_max, n_zones):
        """
        Generate cosine temperature distribution (peaked in center).
        T(z) = T_min + (T_max - T_min) * cos²(π * (z - 0.5))
        where z goes from 0 to 1 along the core height.
        """
        z_normalized = np.linspace(0, 1, n_zones)
        # Cosine squared distribution - peaks at center (z=0.5)
        cos_profile = np.cos(np.pi * (z_normalized - 0.5))**2
        temps = T_min + (T_max - T_min) * cos_profile
        return temps
    
    # Coolant: Linear distribution (hottest at outlet)
    T_coolant_z = np.linspace(params["coolant_inlet"], params["coolant_outlet"], params["n_ax_zones"])

    # Fuel compact: Cosine distribution (hottest in center)
    T_compact_z = cosine_temp_profile(params["compact_min"], params["compact_max"], params["n_ax_zones"])

    # Graphite matrix: Cosine distribution (hottest in center)
    T_matrix_z = cosine_temp_profile(params["matrix_min"], params["matrix_max"], params["n_ax_zones"])

    # Radial reflector: Cosine distribution (hottest in center)
    T_reflector_z = cosine_temp_profile(params["reflector_min"], params["reflector_max"], params["n_ax_zones"])

    # Top and bottom reflectors: Use minimum reflector temperature (uniform)
    T_reflector_axial = params["reflector_min"]
    
    # ====================================================================================================
    # DEFINE AXIAL COORDINATES 
    # ====================================================================================================
    
    reactor_bottom = 0.0
    reactor_top = reactor_bottom + params["core_height"]

    axial_section_height = params["core_height"] / params["n_ax_zones"]

    axial_coords = np.linspace(reactor_bottom, reactor_top, params["n_ax_zones"] + 1)

    # ====================================================================================================
    # CREATE TRISO LATTICE
    # ====================================================================================================

    triso_lattice, n_trisos = trisos.create_triso_lattice(
        params = params,
        mats = mats,
        axial_section_height = axial_section_height
    )

    # ====================================================================================================
    # CREATE ASSEMBLIES
    # ====================================================================================================

    assemblies, m_colors, bundle_pitch = asm.create_assembly_univs(
        params = params,
        mats = mats,
        T_coolant_z = T_coolant_z,
        T_compact_z = T_compact_z,
        T_matrix_z = T_matrix_z,
        triso_lattice = triso_lattice,
        axial_coords = axial_coords,
        reactor_bottom = reactor_bottom,
        reactor_top = reactor_top
    )

    # ====================================================================================================
    # CREATE CORE LATTICE
    # ====================================================================================================

    core_lattice = asm.build_core_lattice(
        assemblies = assemblies,
        core_rings = core_rings,
        bundle_pitch = bundle_pitch
    )

    # ====================================================================================================
    # FULL CORE AND OUTER PERMANENT REFLECTOR CREATION
    # ====================================================================================================

    # Create axially-segmented outer reflector cells
    outer_refl_cells = []

    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        z_mid = 0.5 * (z_min + z_max)
        min_z_plane = openmc.ZPlane(z0=z_min)
        max_z_plane = openmc.ZPlane(z0=z_max)
        
        # Get temperature for this axial zone
        T_reflector = T_reflector_z[idx]
        
        # Clone graphite material and set temperature
        graphite_clone = mats.graphite.clone()
        m_colors[graphite_clone] = 'darkblue'
        graphite_clone.temperature = T_reflector
        
        # Create cell for this axial slice of outer reflector
        outer_refl_cell = openmc.Cell(fill=graphite_clone)
        outer_refl_cells.append(outer_refl_cell)

    # Create universe containing all axial slices
    core_outer_univ = openmc.Universe(cells=outer_refl_cells)
    core_lattice.outer = core_outer_univ

    # Define core boundary
    core_cyl = openmc.ZCylinder(r=params["core_radius"], boundary_type='vacuum')
    min_z = openmc.ZPlane(z0=reactor_bottom)
    max_z = openmc.ZPlane(z0=reactor_top)

    if params["use_1/6_geometry"]:
        # Plane 1: along x-axis (angle = 0°)
        plane_1 = openmc.Plane(a=0, b=1, c=0, d=0, boundary_type='reflective')  # y = 0

        # Plane 2: at 60° from x-axis
        angle_deg = 60.0
        angle_rad = np.radians(angle_deg)
        # Normal vector for plane at 60°: perpendicular to the radial direction
        plane_2 = openmc.Plane(
            a=-np.sin(angle_rad), 
            b=np.cos(angle_rad), 
            c=0, 
            d=0, 
            boundary_type='reflective'
        )

        # Define the 1/6 geometry wedge region
        # The wedge is defined as: inside cylinder AND between the two reflective planes
        wedge_region = (
            -core_cyl & 
            +min_z & 
            -max_z & 
            +plane_1 &  # Above first plane (y > 0 side)
            -plane_2    # Below second plane (60° side)
        )

        core_cell = openmc.Cell(fill=core_lattice, region=wedge_region)

    else:
        core_cell = openmc.Cell(fill=core_lattice, region=-core_cyl & +min_z & -max_z)

    # Top and bottom reflectors with uniform temperature
    top_refl_z = reactor_top + params["reflector_thickness"]
    bottom_refl_z = reactor_bottom - params["reflector_thickness"]
    top_refl = openmc.ZPlane(z0=top_refl_z, boundary_type='vacuum')
    bottom_refl = openmc.ZPlane(z0=bottom_refl_z, boundary_type='vacuum')

    # Clone graphite for top/bottom reflectors and set temperature
    graphite_top = mats.graphite.clone()
    m_colors[graphite_top] = 'darkblue'
    graphite_top.temperature = T_reflector_axial
    
    graphite_bottom = mats.graphite.clone()
    m_colors[graphite_bottom] = 'darkblue'
    graphite_bottom.temperature = T_reflector_axial

    top_refl_cell = openmc.Cell(
        fill=graphite_top, 
        region=-core_cyl & +max_z & -top_refl & +plane_1 & -plane_2
    )

    bottom_refl_cell = openmc.Cell(
        fill=graphite_bottom, 
        region=-core_cyl & +bottom_refl & -min_z & +plane_1 & -plane_2
    )

    geometry = openmc.Geometry([core_cell, top_refl_cell, bottom_refl_cell])
    model.geometry = geometry

    # ====================================================================================================
    # GEOMETRY PLOT GENERATION
    # ====================================================================================================

    m_colors[mats.fuel] = 'palegreen'
    m_colors[mats.buffer] = 'sandybrown'
    m_colors[mats.pyc] = 'orange'
    m_colors[mats.sic] = 'yellow'
    m_colors[mats.graphite] = 'darkblue'
    m_colors[mats.b4c_poison] = 'purple'
    if params["control_insertion"] > 0:
        m_colors[mats.b4c_control] = 'black'
    m_colors[mats.incoloy800H] = 'gray'

    sin60 = np.sin(np.radians(60))

    if params["use_1/6_geometry"]:
        plot1_y_width = params["core_radius"] * sin60
        plot1_x_origin = params["core_radius"] / 2
        plot1_y_origin = params["core_radius"] * sin60 / 2

        plot3_x_width = params["core_radius"]
        plot3_x_origin = params["core_radius"] / 2

        plot5_x_width = params["core_radius"]
        plot5_y_width = sin60 * params["core_radius"]
        plot5_x_origin = params["core_radius"] / 2
        plot5_y_origin = params["core_radius"] * sin60 / 2
        plot5_y_pixels = 866

    else:
        plot1_y_width = 2 * params["core_radius"]
        plot1_x_origin = 0.0
        plot1_y_origin = 0.0

        plot3_x_width = 2 * params["core_radius"]
        plot3_x_origin = 0.0

        plot5_x_width = 2 * params["core_radius"]
        plot5_y_width = 2 * params["core_radius"]
        plot5_x_origin = 0.0
        plot5_y_origin = 0.0
        plot5_y_pixels = 1000

    plot1 = openmc.Plot()
    plot1.filename = 'Core_YZ_Material'
    plot1.width = (plot1_y_width, params["core_height"] + 2 * params["reflector_thickness"])
    plot1.basis = 'yz'
    plot1.origin = (plot1_x_origin, plot1_y_origin, params["core_height"] / 2)
    plot1.pixels = (800, 1200)
    plot1.color_by = 'material'
    plot1.colors = m_colors

    plot2 = openmc.Plot()
    plot2.filename = 'Core_YZ_Cell'
    plot2.width = plot1.width
    plot2.basis = plot1.basis
    plot2.origin = plot1.origin
    plot2.pixels = plot1.pixels
    plot2.color_by = 'cell'

    plot3 = openmc.Plot()
    plot3.filename = 'Core_XZ_Material'
    plot3.width = (plot3_x_width, params["core_height"] + 2 * params["reflector_thickness"])
    plot3.basis = 'xz'
    plot3.origin = (plot3_x_origin, 0.0, params["core_height"] / 2.0)
    plot3.pixels = (800, 1200)
    plot3.color_by = 'material'
    plot3.colors = m_colors
    
    plot4 = openmc.Plot()
    plot4.filename = 'Core_XZ_Cell'
    plot4.width = plot3.width
    plot4.basis = plot3.basis
    plot4.origin = plot3.origin
    plot4.pixels = plot3.pixels
    plot4.color_by = 'cell'

    plot5 = openmc.Plot()
    plot5.filename = 'Core_XY_Material'
    plot5.width = (plot5_x_width, plot5_y_width)
    plot5.basis = 'xy'
    plot5.origin = (plot5_x_origin, plot5_y_origin, (params["core_height"] / 2) + (axial_section_height / 2))
    plot5.pixels = (1000, plot5_y_pixels)
    plot5.color_by = 'material'
    plot5.colors = m_colors

    plot6 = openmc.Plot()
    plot6.filename = 'Core_XY_Cell'
    plot6.width = plot5.width
    plot6.basis = plot5.basis
    plot6.origin = plot5.origin
    plot6.pixels = plot5.pixels
    plot6.color_by = 'cell'

    model.plots = openmc.Plots([plot1, plot2, plot3, plot4, plot5, plot6])

    # ====================================================================================================
    # TALLY CREATION
    # ====================================================================================================

    tallies = openmc.Tallies()

    # ----- Energy Spectrum Tallies -----
    # Create filter for fuel and energy bins
    fuel_filter = openmc.MaterialFilter(mats.fuel)
    energy_bins = np.logspace(-9, 7, 200)
    energy_filter = openmc.EnergyFilter(energy_bins)

    flux_spectrum_tally = openmc.Tally(name="flux_energy_spectrum")
    flux_spectrum_tally.scores = ["flux"]
    flux_spectrum_tally.filters = [energy_filter]

    fission_tally = openmc.Tally(name="fission")
    fission_tally.scores = ["fission"]

    heating_tally = openmc.Tally(name="heating")
    heating_tally.scores = ["heating-local"]

    tallies += [flux_spectrum_tally, fission_tally, heating_tally]

    # ----- Spatial Mesh Tallies -----
    
    if params["use_1/6_geometry"]:
        mesh_x_min = 0.0
        mesh_x_max = params["core_radius"]
        mesh_nx = 250

        mesh_y_min = 0.0
        mesh_y_max = params["core_radius"] * sin60 
        mesh_ny = 217

    else:
        mesh_x_min = -params["core_radius"]
        mesh_x_max = params["core_radius"]
        mesh_nx = 500

        mesh_y_min = -params["core_radius"]
        mesh_y_max = params["core_radius"]
        mesh_ny = 500

    # Active Core Region Mesh
    mesh = openmc.RegularMesh()
    mesh.dimension = [mesh_nx, mesh_ny, params["n_ax_zones"]]
    mesh.lower_left = [mesh_x_min, mesh_y_min, reactor_bottom]
    mesh.upper_right = [mesh_x_max, mesh_y_max, reactor_top]
    mesh_filter = openmc.MeshFilter(mesh)

    mesh_tally_active = openmc.Tally(name='mesh_rates')
    mesh_tally_active.filters = [mesh_filter]
    mesh_tally_active.scores = ['flux', 'fission', 'nu-fission']

    # Full Core Mesh (including reflectors)
    n_reflector_zones = 33
    n_total_zones = n_reflector_zones + params["n_ax_zones"] + n_reflector_zones

    mesh_full = openmc.RegularMesh()
    mesh_full.dimension = [mesh_nx, mesh_ny, n_total_zones]

    mesh_bottom = reactor_bottom - params["reflector_thickness"]
    mesh_top = reactor_top + params["reflector_thickness"]

    mesh_full.lower_left = [mesh_x_min, mesh_y_min, mesh_bottom]
    mesh_full.upper_right = [mesh_x_max, mesh_y_max, mesh_top]
    mesh_full_filter = openmc.MeshFilter(mesh_full)

    mesh_tally_full = openmc.Tally(name='mesh_rates_full')
    mesh_tally_full.filters = [mesh_full_filter]
    mesh_tally_full.scores = ['flux', 'fission', 'nu-fission']

    # ----- Global tally for total rates -----
    global_tally = openmc.Tally(name='global_rates')
    global_tally.scores = ['flux', 'fission', 'nu-fission']

    tallies += [mesh_tally_active, mesh_tally_full, global_tally]

    model.tallies = tallies

    # ====================================================================================================
    # MONTE CARLO SETTINGS
    # ====================================================================================================

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.batches = params.get("total_batches", 300)
    settings.inactive = params.get("inactive_batches", 50)
    settings.particles = params.get("particles", 100_000)
    settings.temperature = {
        'method': 'interpolation',
        'range': (293.0, 1800.0),
        'tolerance': 100.0
    }

    if params["use_1/6_geometry"]:
        phi_dist = openmc.stats.Uniform(a = 0.0, b = np.pi / 2)
    else:
        phi_dist = openmc.stats.Uniform(a = 0.0, b = 2 * np.pi)
    r_dist = openmc.stats.Uniform(a = 0.0, b = params["core_radius"])
    z_dist = openmc.stats.Uniform(a = reactor_bottom, b = reactor_top)
    source = openmc.IndependentSource()
    source.space = openmc.stats.CylindricalIndependent(
        r = r_dist,
        phi = phi_dist,
        z = z_dist,
        origin = (0.0, 0.0, 0.0)  # center of the cylinder
    )
    settings.source = source

    # ====================================================================================================
    # STOCHASTIC VOLUME CALCULATION SETTINGS
    # ====================================================================================================
    
    if params["calculate_fuel_volume"]:        
        vol_calc = openmc.VolumeCalculation(
            domains=[mats.fuel],
            samples=params.get("volume_samples", 1_000_000),
            lower_left=[mesh_x_min, mesh_y_min, reactor_bottom],
            upper_right=[mesh_x_max, mesh_y_max, reactor_top]
        )
        settings.volume_calculations = [vol_calc]

    # ====================================================================================================
    # RUN OPENMC
    # ====================================================================================================

    model.settings = settings

    all_mats = model.geometry.get_all_materials()
    model.materials = openmc.Materials(all_mats.values())
    model.export_to_xml()

    openmc.plot_geometry(output=False, cwd=run_dir)

    if params["calculate_fuel_volume"]:
        print("\nRunning stochastic volume calculation for fuel...\n")
        
        # Run the volume calculation
        openmc.calculate_volumes(
            cwd = run_dir,
            threads = 24,
            output = True
        )
        
        # Load results and print
        vol_calc_results = openmc.VolumeCalculation.from_hdf5(
            os.path.join(run_dir, 'volume_1.h5')
        )
        
        for domain_id, vol_var in vol_calc_results.volumes.items():
            # Extract nominal value and standard deviation from Variable object
            vol = vol_var.nominal_value
                        
            # Calculate and print mass estimate for fuel
            # Account for 1/6 geometry
            geometry_factor = 6 if params["use_1/6_geometry"] else 1
            total_vol = vol * geometry_factor
            
            # UCO: density ~10.97 g/cm³, U mass fraction ~0.888
            uco_density = params["kernel_density"] / 1000  # g/cm³
            u_mass_fraction = 238.0 / 268.0
            total_u_mass_kg = (total_vol * uco_density * u_mass_fraction) / 1000
            
            print(f"\nTotal fuel volume (full core): {total_vol:.4f} cm³")
            print(f"Estimated uranium mass: {total_u_mass_kg:.2f} kg\n")
    
    else:
        print("\nSkipping volume calculation.\n")

    openmc_output_file = os.path.join(run_dir, 'openmc_output.txt')

    with open(openmc_output_file, 'w', buffering=1) as outf:
        process = subprocess.Popen(
            ['openmc'],
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env={**os.environ, 'OMP_NUM_THREADS': '24'}
        )
        
        # Read and display output line-by-line in real-time
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
            outf.write(line)
            outf.flush()
        
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"OpenMC failed with return code {return_code}")
    
    return n_trisos

# ====================================================================================================
# POST-PROCESSING FUNCTIONS
# ====================================================================================================

def update_run_info(run_dir, n_trisos):
    """
    Update run_params.json with n_trisos after TRISO creation.
    
    Args:
        run_dir: Directory containing run_params.json
        n_trisos: Number of TRISO particles per axial zone
    """
    import json
    
    params_path = os.path.join(run_dir, 'run_params.json')
    
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    
    data['n_trisos'] = n_trisos
    
    with open(params_path, 'w') as f:
        json.dump(data, f, indent=2)

def run_post_processing(run_dir, params, n_trisos):
    """
    Run all post-processing scripts for a completed simulation.
    
    Args:
        run_dir: Directory containing simulation results
        params: Simulation parameters
        n_trisos: Number of TRISO particles per axial zone
    """
    print(f"{'='*80}")
    print("RUNNING POST-PROCESSING")
    print(f"{'='*80}\n")
    
    # Update run_params.json with n_trisos
    update_run_info(run_dir, n_trisos)
    
    # Try to import post-processing modules
    try:
        from burnup_estimation import run_burnup_estimation
        print("Running burnup estimation...")
        run_burnup_estimation(run_dir, params, n_trisos)
    except ImportError as e:
        print(f"Warning: Could not import burnup_estimation: {e}")
    except Exception as e:
        print(f"Warning: Burnup estimation failed: {e}")
    
    try:
        from tally_plotter import run_tally_plots
        print("Running tally plotting...")
        run_tally_plots(run_dir, params)
    except ImportError as e:
        print(f"Warning: Could not import tally_plotter: {e}")
    except Exception as e:
        print(f"Warning: Tally plotting failed: {e}")
    
    try:
        from spectrum_thermalization import run_spectrum_analysis
        print("Running spectrum & thermalization analysis...")
        run_spectrum_analysis(run_dir, params)
    except ImportError as e:
        print(f"Warning: Could not import spectrum_thermalization: {e}")
    except Exception as e:
        print(f"Warning: Spectrum analysis failed: {e}")

    print(f"{'='*80}")
    print("POST-PROCESSING COMPLETE")
    print(f"{'='*80}")

def run_parametric_post_processing(parametric_dir):
    """
    Run parametric study post-processing.
    
    Args:
        parametric_dir: Directory containing all parametric study cases
    """
    try:
        from parametric_postprocessing import run_parametric_postprocessing
        run_parametric_postprocessing(parametric_dir)
    except ImportError as e:
        print(f"Warning: Could not import parametric_postprocessing: {e}")
    except Exception as e:
        print(f"Warning: Parametric post-processing failed: {e}")

# ====================================================================================================
# STUDY EXECUTION 
# ====================================================================================================

if __name__ == "__main__":
    # ----- Create base directory structure -----
    now = datetime.now()
    run_name = f"htgr_run_{now.strftime('%m.%d.%Y_%H.%M.%S')}"
    
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    OUTPUT_BASE = os.path.join(PARENT_DIR, "MicroHTGR_Output")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    BASE_DIR = os.path.join(OUTPUT_BASE, run_name)

    run_parametric_study = cfg.parametric_param is not None and len(cfg.parametric_values) > 0
    
    # ----- Run Parametric Study -----
    if cfg.params["study_execution_mode"] == "ParametricStudy":
        # Add "_ParametricStudy" suffix to base run folder
        BASE_DIR = os.path.join(OUTPUT_BASE, run_name + "_ParametricStudy" + f"_{cfg.parametric_param}")
        os.makedirs(BASE_DIR, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PARAMETRIC STUDY: {cfg.parametric_param}")
        print(f"Values: {cfg.parametric_values}")
        print(f"Base Directory: {BASE_DIR}")
        print(f"{'='*80}")
        
        # Iteratively run simulation for values in parametric study
        for i, val in enumerate(cfg.parametric_values):
            caseNum = i + 1
            caseNumFormatted = f"{caseNum:0{len(str(len(cfg.parametric_values)))+1}d}"

            runName = f"{cfg.parametric_param}_Case_{caseNumFormatted}_{val}"
            
            # Create run-specific directory for current value
            run_dir = os.path.join(BASE_DIR, runName)

            print(f"\n{'='*80}")
            print(f"Runing Case {caseNumFormatted}: {cfg.parametric_param} = {val}")
            print(f"Run Directory: {run_dir}")
            print(f"{'='*80}\n")

            # Create temporary copy of params and modify the current specified parameter
            params_copy = cfg.params.copy()
            params_copy[cfg.parametric_param] = val

            n_trisos = run_simulation(params_copy, cfg.core_rings, run_dir)

            run_post_processing(run_dir, params_copy, n_trisos)
        
        run_parametric_post_processing(BASE_DIR)

        print(f"\n{'='*80}")
        print("PARAMETRIC STUDY COMPLETE")
        print(f"Results Directory: {BASE_DIR}")
        print(f"{'='*80}\n")
    
    # ----- Run Reactivity Study -----
    elif cfg.params["study_execution_mode"] == "Reactivity Study":
        from reactivity_coefficients import run_reactivity_coefficients

        BASE_DIR_RC = os.path.join(OUTPUT_BASE, run_name + "_ReactivityCoeffs")
        os.makedirs(BASE_DIR_RC, exist_ok=True)

        run_reactivity_coefficients(
            params = cfg.params,
            core_rings = cfg.core_rings,
            base_run_dir = BASE_DIR,
            output_base_dir = BASE_DIR_RC,
            delta_T_values = [50.0, 100.0, 150.0],
            coefficients = ["FTC", "MTC", "ITC"],
            run_simulation_fn = run_simulation,
            run_post_processing_fn = run_post_processing,
        )

    # ----- Run Single Run -----   
    elif cfg.params["study_execution_mode"] == "SingleRun":
        # Add "_SingleRun" suffix to base run folder
        BASE_DIR = os.path.join(OUTPUT_BASE, run_name + "_SingleRun")
        
        print(f"\n{'='*80}")
        print("SINGLE RUN MODE")
        print(f"Run directory: {BASE_DIR}")
        print(f"{'='*80}\n")
        
        n_trisos = run_simulation(cfg.params, cfg.core_rings, BASE_DIR)

        run_post_processing(BASE_DIR, cfg.params, n_trisos)
        
        print(f"\n{'='*80}")
        print("SIMULATION COMPLETE")
        print(f"Results Directory: {BASE_DIR}")
        print(f"{'='*80}\n")