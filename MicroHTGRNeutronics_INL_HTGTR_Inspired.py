import os
import math
import shutil
import openmc
import numpy as np
import openmc.deplete
from datetime import datetime
import sys
import subprocess
import json

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
openmc.config['cross_sections'] = cross_sections_path

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
    
    print(f"\nSaved parameters to: {params_path}")

# ====================================================================================================
# REDUCED DEPLETION CHAIN FUNCTION
# ====================================================================================================

def build_reduced_chain(full_chain_file, reduced_chain_file, tracked_nuclides):
    """
    Build a reduced depletion chain containing only the specified nuclides.

    The reduced chain preserves all decay/transmutation pathways between the
    tracked nuclides; everything else is pruned by openmc.deplete.Chain.reduce().

    Args:
        full_chain_file:    Path to the full ENDF/B chain XML file.
        reduced_chain_file: Path where the reduced chain XML will be written.
        tracked_nuclides:   List of nuclide name strings (OpenMC format, e.g. "Xe135_m1").

    Returns:
        reduced_chain_file: Path to the reduced chain XML (for passing to CoupledOperator).
    """
    print(f"\nBuilding reduced depletion chain...")
    print(f"Full chain file:    {full_chain_file}")
    print(f"Reduced chain file: {reduced_chain_file}")
    print(f"Tracking {len(tracked_nuclides)} nuclides")

    chain = openmc.deplete.Chain.from_xml(full_chain_file)

    chain_nuclide_names = {nuc.name for nuc in chain.nuclides}
    missing = [n for n in tracked_nuclides if n not in chain_nuclide_names]
    if missing:
        print(f"\nWARNING: Nuclides not found in full chain (will be skipped):\n    {missing}")

    reduced_chain = chain.reduce(tracked_nuclides)

    os.makedirs(os.path.dirname(reduced_chain_file), exist_ok=True)
    reduced_chain.export_to_xml(reduced_chain_file)

    return reduced_chain_file

# ==================================================================
# MODEL BUILDING FUNCTION
# ==================================================================

def build_model(params, run_dir):
    """
    Build the complete OpenMC model (geometry, materials, tallies, settings).

    This is the shared model construction used by both run_simulation() and
    run_depletion_simulation().

    Args:
        params: Simulation parameters dictionary
        run_dir: Directory for output files

    Returns:
        tuple: (model, n_trisos, m_colors)
            - model: openmc.model.Model ready to export or deplete
            - n_trisos: Number of TRISO particles per axial zone
            - m_colors: Material color dictionary for plotting
    """

    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    # Save params to run directory for post-processing
    save_params(run_dir, params)

    # Save cross_sections.xml file to run directory
    shutil.copy2(cross_sections_path, os.path.join(run_dir, 'cross_sections.xml'))

    model = openmc.model.Model()
    
    # ==================================================================    
    # INITIALIZE TEMPERATURE PROFILES
    # ==================================================================

    def cosine_temp_profile(T_min, T_max, n_zones):
        """
        Generate cosine temperature distribution (peaked in center).
        T(z) = T_min + (T_max - T_min) * cos²(π * (z - 0.5))
        """
        z_normalized = np.linspace(0, 1, n_zones)
        cos_profile = np.cos(np.pi * (z_normalized - 0.5))**2
        temps = T_min + (T_max - T_min) * cos_profile
        return temps
    
    T_coolant_z = np.linspace(params["coolant_inlet"], params["coolant_outlet"], params["n_ax_zones"])
    T_compact_z = cosine_temp_profile(params["compact_min"], params["compact_max"], params["n_ax_zones"])
    T_matrix_z = cosine_temp_profile(params["matrix_min"], params["matrix_max"], params["n_ax_zones"])
    T_reflector_z = cosine_temp_profile(params["reflector_min"], params["reflector_max"], params["n_ax_zones"])
    T_reflector_axial = params["reflector_min"]
    
    # ==================================================================
    # DEFINE AXIAL COORDINATES 
    # ==================================================================
    
    reactor_bottom = 0.0
    reactor_top = reactor_bottom + params["core_height"]
    axial_section_height = params["core_height"] / params["n_ax_zones"]
    axial_coords = np.linspace(reactor_bottom, reactor_top, params["n_ax_zones"] + 1)

    # ==================================================================
    # CREATE TRISO LATTICE
    # ==================================================================

    triso_lattice, n_trisos = trisos.create_triso_lattice(
        params = params,
        mats = mats,
        axial_section_height = axial_section_height
    )

    # ==================================================================
    # CREATE ASSEMBLIES
    # ==================================================================

    assemblies, m_colors, bundle_pitch = asm.create_assembly_univs(
        params = params,
        mats = mats,
        T_coolant_z = T_coolant_z,
        T_compact_z = T_compact_z,
        T_matrix_z = T_matrix_z,
        T_reflector_z = T_reflector_z,
        triso_lattice = triso_lattice,
        axial_coords = axial_coords,
        reactor_bottom = reactor_bottom,
        reactor_top = reactor_top
    )

    # ==================================================================
    # CREATE CORE LATTICE
    # ==================================================================

    core_lattice = asm.build_core_lattice(
        assemblies = assemblies,
        core_rings = params["core_rings"],
        bundle_pitch = bundle_pitch
    )

    # ==================================================================
    # FULL CORE AND OUTER PERMANENT REFLECTOR CREATION
    # ==================================================================

    outer_refl_cells = []
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        T_reflector = T_reflector_z[idx]
        graphite_clone = mats.graphite.clone()
        m_colors[graphite_clone] = 'darkblue'
        graphite_clone.temperature = T_reflector
        outer_refl_cell = openmc.Cell(fill=graphite_clone)
        outer_refl_cells.append(outer_refl_cell)

    core_outer_univ = openmc.Universe(cells=outer_refl_cells)
    core_lattice.outer = core_outer_univ

    core_cyl = openmc.ZCylinder(r=params["core_radius"], boundary_type='vacuum')
    min_z = openmc.ZPlane(z0=reactor_bottom)
    max_z = openmc.ZPlane(z0=reactor_top)

    if params["use_1/6_geometry"]:
        plane_1 = openmc.Plane(a=0, b=1, c=0, d=0, boundary_type='reflective')
        angle_deg = 60.0
        angle_rad = np.radians(angle_deg)
        plane_2 = openmc.Plane(
            a=-np.sin(angle_rad), 
            b=np.cos(angle_rad), 
            c=0, 
            d=0, 
            boundary_type='reflective'
        )
        wedge_region = (
            -core_cyl & 
            +min_z & 
            -max_z & 
            +plane_1 &
            -plane_2
        )
        core_cell = openmc.Cell(fill=core_lattice, region=wedge_region)
    else:
        core_cell = openmc.Cell(fill=core_lattice, region=-core_cyl & +min_z & -max_z)

    # Top and bottom reflectors
    top_refl_z = reactor_top + params["reflector_thickness"]
    bottom_refl_z = reactor_bottom - params["reflector_thickness"]
    top_refl = openmc.ZPlane(z0=top_refl_z, boundary_type='vacuum')
    bottom_refl = openmc.ZPlane(z0=bottom_refl_z, boundary_type='vacuum')

    graphite_top = mats.graphite.clone()
    m_colors[graphite_top] = 'darkblue'
    graphite_top.temperature = T_reflector_axial
    
    graphite_bottom = mats.graphite.clone()
    m_colors[graphite_bottom] = 'darkblue'
    graphite_bottom.temperature = T_reflector_axial

    if params["use_1/6_geometry"]:
        top_refl_cell = openmc.Cell(
            fill=graphite_top, 
            region=-core_cyl & +max_z & -top_refl & +plane_1 & -plane_2
        )
        bottom_refl_cell = openmc.Cell(
            fill=graphite_bottom, 
            region=-core_cyl & +bottom_refl & -min_z & +plane_1 & -plane_2
        )
    else:
        top_refl_cell = openmc.Cell(
            fill=graphite_top, 
            region=-core_cyl & +max_z & -top_refl
        )
        bottom_refl_cell = openmc.Cell(
            fill=graphite_bottom, 
            region=-core_cyl & +bottom_refl & -min_z
        )

    geometry = openmc.Geometry([core_cell, top_refl_cell, bottom_refl_cell])
    model.geometry = geometry

    # ==================================================================
    # GEOMETRY PLOT GENERATION
    # ==================================================================

    m_colors[mats.fuel] = 'palegreen'
    m_colors[mats.buffer] = 'sandybrown'
    m_colors[mats.pyc] = 'orange'
    m_colors[mats.sic] = 'yellow'
    m_colors[mats.graphite] = 'darkblue'
    m_colors[mats.b4c_poison] = 'purple'
    if params["bank_1_insertion"] > 0 or params["bank_2_insertion"] > 0 or params["bank_3_insertion"] > 0:
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

    # ==================================================================
    # TALLY CREATION
    # ==================================================================

    tallies = openmc.Tallies()

    # ----- Global Tallies -----

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

    # ----- Mesh Tallies -----

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

    mesh = openmc.RegularMesh()
    mesh.dimension = [mesh_nx, mesh_ny, params["n_ax_zones"]]
    mesh.lower_left = [mesh_x_min, mesh_y_min, reactor_bottom]
    mesh.upper_right = [mesh_x_max, mesh_y_max, reactor_top]
    mesh_filter = openmc.MeshFilter(mesh)

    mesh_tally_active = openmc.Tally(name='mesh_rates')
    mesh_tally_active.filters = [mesh_filter]
    mesh_tally_active.scores = ['flux', 'fission', 'nu-fission']

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

    global_tally = openmc.Tally(name='global_rates')
    global_tally.scores = ['flux', 'fission', 'nu-fission']

    tallies += [mesh_tally_active, mesh_tally_full, global_tally]

    # ----- Leakage Spectrum Tallies -----

    leakage_energy_bins   = np.logspace(-9, 7, 200)
    leakage_energy_filter = openmc.EnergyFilter(leakage_energy_bins)

    # Surface filters
    radial_surf_filter  = openmc.SurfaceFilter(core_cyl)
    axial_top_surf_filter = openmc.SurfaceFilter(top_refl)
    axial_bot_surf_filter = openmc.SurfaceFilter(bottom_refl)

    radial_current_tally = openmc.Tally(name='radial_leakage_current')
    radial_current_tally.filters = [radial_surf_filter, leakage_energy_filter]
    radial_current_tally.scores  = ['current']

    axial_top_current_tally = openmc.Tally(name='axial_top_leakage_current')
    axial_top_current_tally.filters = [axial_top_surf_filter, leakage_energy_filter]
    axial_top_current_tally.scores  = ['current']

    axial_bot_current_tally = openmc.Tally(name='axial_bot_leakage_current')
    axial_bot_current_tally.filters = [axial_bot_surf_filter, leakage_energy_filter]
    axial_bot_current_tally.scores  = ['current']

    tallies += [radial_current_tally, axial_top_current_tally, axial_bot_current_tally]

    model.tallies = tallies

    # ==================================================================
    # MONTE CARLO SETTINGS
    # ==================================================================

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
        origin = (0.0, 0.0, 0.0)
    )
    settings.source = source

    # Stochastic volume calculation
    if params.get("calculate_fuel_volume", False):        
        vol_calc = openmc.VolumeCalculation(
            domains=[mats.fuel, mats.b4c_poison],
            samples=params.get("volume_samples", 1_000_000),
            lower_left=[mesh_x_min, mesh_y_min, reactor_bottom],
            upper_right=[mesh_x_max, mesh_y_max, reactor_top]
        )
        settings.volume_calculations = [vol_calc]

    model.settings = settings

    # Finalize materials
    all_mats = model.geometry.get_all_materials()
    model.materials = openmc.Materials(all_mats.values())

    active_ids = set(all_mats.keys())
    m_colors_active = {mat: color for mat, color in m_colors.items()
                       if mat.id in active_ids}

    for plot in model.plots:
        if plot.color_by == 'material':
            plot.colors = m_colors_active

    return model, n_trisos, m_colors

# ====================================================================================================
# MAIN SIMULATION FUNCTION
# ====================================================================================================

def run_simulation(params, run_dir):
    """
    Build and run an eigenvalue OpenMC simulation.
    
    Returns:
        n_trisos: Number of TRISO particles per axial zone
    """

    model, n_trisos, m_colors = build_model(params, run_dir)
    model.export_to_xml()

    openmc.plot_geometry(output=False, cwd=run_dir)

    # Stochastic volume calculation
    if params.get("calculate_fuel_volume", False):
        print("\nRunning stochastic volume calculation for fuel and burnable poison...\n")

        openmc.calculate_volumes(
            cwd=run_dir,
            threads=24,
            output=True
        )

        vol_calc_results = openmc.VolumeCalculation.from_hdf5(
            os.path.join(run_dir, 'volume_1.h5')
        )

        geometry_factor = 6 if params["use_1/6_geometry"] else 1
        fuel_volume_simulated = 0.0
        poison_volume_simulated = 0.0

        for domain_id, vol_var in vol_calc_results.volumes.items():
            vol = vol_var.nominal_value
            vol_std = vol_var.std_dev

            if domain_id == mats.fuel.id:
                fuel_volume_simulated += vol
                print(f"\nFuel domain {domain_id}: {vol:.4f} ± {vol_std:.4f} cm³")
            elif domain_id == mats.b4c_poison.id:
                poison_volume_simulated += vol
                print(f"B4C poison domain {domain_id}: {vol:.4f} ± {vol_std:.4f} cm³")

        # Fuel reporting
        total_fuel_volume_full_core = fuel_volume_simulated * geometry_factor
        uco_density_g_cm3 = params["kernel_density"] / 1000.0
        u_mass_fraction = 238.0 / 268.0
        total_HM_mass_kg = (total_fuel_volume_full_core * uco_density_g_cm3 * u_mass_fraction) / 1000.0

        # B4C poison reporting
        total_poison_volume_full_core = poison_volume_simulated * geometry_factor
        b4c_density_g_cm3 = params["B4C_density_poison"] / 1000.0
        b10_enrichment = params["B10_enrichment_poison"]
        mass_10 = openmc.data.atomic_mass('B10')
        mass_11 = openmc.data.atomic_mass('B11')
        b10_mass_fraction = (b10_enrichment * mass_10) / (
            b10_enrichment * mass_10 + (1.0 - b10_enrichment) * mass_11
        )
        total_B10_mass_kg = (total_poison_volume_full_core * b4c_density_g_cm3 * b10_mass_fraction) / 1000.0

        print(f"\nFuel volume (simulated geometry):        {fuel_volume_simulated:.4f} cm³")
        print(f"Fuel volume (full core):                 {total_fuel_volume_full_core:.4f} cm³")
        print(f"Estimated uranium mass:                  {total_HM_mass_kg:.2f} kg")
        print(f"\nB4C poison volume (simulated geometry):  {poison_volume_simulated:.4f} cm³")
        print(f"B4C poison volume (full core):           {total_poison_volume_full_core:.4f} cm³")
        print(f"Estimated B-10 mass:                     {total_B10_mass_kg:.4f} kg\n")

        # Save to run_params.json
        params_path = os.path.join(run_dir, 'run_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                saved_params = json.load(f)
        else:
            saved_params = {}

        saved_params['n_trisos']                    = n_trisos
        saved_params['fuel_material_id']            = mats.fuel.id
        saved_params['poison_material_id']          = mats.b4c_poison.id
        saved_params['fuel_volume_simulated_cm3']   = fuel_volume_simulated
        saved_params['fuel_volume_full_core_cm3']   = total_fuel_volume_full_core
        saved_params['total_HM_mass_kg']            = total_HM_mass_kg
        saved_params['total_HM_mass_kg']           = total_HM_mass_kg
        saved_params['poison_volume_simulated_cm3'] = poison_volume_simulated
        saved_params['poison_volume_full_core_cm3'] = total_poison_volume_full_core
        saved_params['total_B10_mass_kg']           = total_B10_mass_kg
        saved_params['total_B10_mass_kg']          = total_B10_mass_kg

        with open(params_path, 'w') as f:
            json.dump(saved_params, f, indent=2)

        print(f"   Volume results saved to run_params.json\n")

    else:
        print("\nSkipping volume calculation.\n")

    # Run OpenMC
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
# DEPLETION SIMULATION FUNCTION
# ====================================================================================================

def run_depletion_simulation(params, run_dir):
    """
    Build model and run a coupled depletion simulation.

    Uses OpenMC's CoupledOperator with the specified integrator to
    deplete fuel over the configured timesteps at constant power.

    A reduced depletion chain is generated from params["tracked_nuclides"] before
    the first run and written to params["depletion_chain_reduced_file"].
    On subsequent runs the reduced chain is reused automatically.

    If params["restart_depletion"] is True, the model is loaded from the
    original run directory's XML files (preserving material IDs that match
    depletion_results.h5) and previous results are passed to the
    CoupledOperator via prev_results.

    Returns:
        n_trisos: Number of TRISO particles per axial zone
    """

    print(f"\n{'=' * 80}")
    print("DEPLETION SIMULATION")
    print(f"{'=' * 80}")

    is_restart = params.get("restart_depletion", False)

    # ==================================================================
    # RESTART PATH — load existing model from original run directory
    # ==================================================================
    #
    # Rebuilding the model from scratch assigns new material IDs, which
    # breaks the mapping between depletion_results.h5 and the model.
    # Instead, load the XML files that were written during the original
    # run so that material IDs are guaranteed to match.
    # ==================================================================

    if is_restart:
        restart_dir = params.get("restart_run_dir", run_dir)
        prev_h5 = os.path.join(restart_dir, "depletion_results.h5")

        # Validate restart directory contents
        required_files = {
            "depletion_results.h5": prev_h5,
            "materials.xml":       os.path.join(restart_dir, "materials.xml"),
            "geometry.xml":        os.path.join(restart_dir, "geometry.xml"),
            "settings.xml":        os.path.join(restart_dir, "settings.xml"),
            "run_params.json":     os.path.join(restart_dir, "run_params.json"),
        }
        for label, path in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Cannot restart: {label} not found in {restart_dir}"
                )

        print(f"\n{'=' * 80}")
        print("RESTART MODE — loading model from original run directory")
        print(f"Restart directory: {restart_dir}")
        print(f"{'=' * 80}")

        os.chdir(restart_dir)

        # Load previous depletion results FIRST — we need these to
        # restore correct material compositions before building the model.
        #
        # During the failed run, the operator likely updated materials.xml
        # with step N+1 compositions before crashing, while
        # depletion_results.h5 only recorded through step N. Loading
        # materials.xml directly would give us stale/wrong compositions.
        # export_to_materials() writes the step-N compositions from the
        # HDF5 file back into a corrected materials XML.
        prev_results = openmc.deplete.Results(prev_h5)
        n_completed = len(prev_results) - 1   # Results includes the t=0 entry
        print(f"  Completed depletion steps: {n_completed}")

        # Write corrected materials.xml from last completed step.
        # export_to_materials reads from `path`, updates compositions
        # from the HDF5 results, and writes back to the same file.
        # We copy the original first so we don't overwrite it.
        corrected_materials_path = os.path.join(restart_dir, "materials_restart.xml")
        shutil.copy2(
            os.path.join(restart_dir, "materials.xml"),
            corrected_materials_path
        )
        prev_results.export_to_materials(-1, path=corrected_materials_path)
        print(f"  Exported step-{n_completed} compositions to: {corrected_materials_path}")

        # Clear OpenMC's global ID registries AFTER export_to_materials
        # (which internally creates Material objects) and BEFORE we load
        # the model, so that from_xml() can cleanly claim the correct IDs.
        for cls in [openmc.Material, openmc.Cell, openmc.Universe,
                    openmc.Surface, openmc.Lattice]:
            if hasattr(cls, 'used_ids'):
                cls.used_ids.clear()
        if hasattr(openmc, 'reset_auto_ids'):
            openmc.reset_auto_ids()

        # Load the model from corrected materials (preserves material IDs
        # while ensuring compositions match depletion_results.h5)
        materials = openmc.Materials.from_xml(corrected_materials_path)
        geometry = openmc.Geometry.from_xml(
            os.path.join(restart_dir, "geometry.xml"),
            materials = materials
        )
        settings = openmc.Settings.from_xml(
            os.path.join(restart_dir, "settings.xml")
        )

        model = openmc.model.Model(
            geometry = geometry,
            materials = materials,
            settings = settings
        )

        # Load saved run parameters for volumes and material IDs
        with open(required_files["run_params.json"], 'r') as f:
            saved_params = json.load(f)

        n_trisos = saved_params.get("n_trisos", 0)
        fuel_mat_id = saved_params["fuel_material_id"]
        poison_mat_id = saved_params["poison_material_id"]
        fuel_volume = saved_params["fuel_volume_simulated_cm3"]
        poison_volume = saved_params["poison_volume_simulated_cm3"]

        # Set volumes on the loaded materials (CoupledOperator needs them)
        for mat in materials:
            if mat.id == fuel_mat_id:
                mat.volume = fuel_volume
                print(f"  Fuel material (id={mat.id}): volume = {fuel_volume:.4f} cm³")
            elif mat.id == poison_mat_id:
                mat.volume = poison_volume
                print(f"  Poison material (id={mat.id}): volume = {poison_volume:.4f} cm³")

        # Determine remaining timesteps
        restart_ts = params.get("restart_timesteps_days", None)
        if restart_ts is not None and len(restart_ts) > 0:
            timesteps_days = restart_ts
            print(f"  Using user-specified restart timesteps: {timesteps_days}")
        else:
            original_ts = params.get("depletion_timesteps_days", [30] * 12)
            timesteps_days = original_ts[n_completed:]
            if len(timesteps_days) == 0:
                print("  All original timesteps already completed — nothing to do.")
                return n_trisos
            print(f"  Original timesteps ({len(original_ts)}): {original_ts}")
            print(f"  Remaining timesteps ({len(timesteps_days)}): {timesteps_days}")

        # Use chain file from restart directory if it exists, otherwise regenerate
        reduced_chain_in_dir = os.path.join(restart_dir, "chain_reduced.xml")
        if os.path.exists(reduced_chain_in_dir):
            chain_file = reduced_chain_in_dir
            print(f"  Using existing reduced chain: {chain_file}")
        else:
            full_chain_file = params.get("depletion_chain_file", None)
            if full_chain_file is None or not os.path.exists(full_chain_file):
                raise FileNotFoundError(f"Depletion chain file not found: {full_chain_file}")
            chain_file = full_chain_file
            print(f"  Using full chain file: {chain_file}")

    # ==================================================================
    # FRESH RUN PATH — build model from scratch
    # ==================================================================

    else:
        prev_results = None
        n_completed = 0

        # Force volume calculation on for depletion (operator needs fuel volume)
        depletion_params = params.copy()
        depletion_params["calculate_fuel_volume"] = True

        model, n_trisos, m_colors = build_model(depletion_params, run_dir)

        # ==================================================================
        # EXPORT MODEL AND RUN STOCHASTIC VOLUME CALCULATION
        # ==================================================================

        model.export_to_xml()

        openmc.plot_geometry(output=False, cwd=run_dir)

        print("\nRunning stochastic volume calculation for fuel and burnable poison...")
        print(f"Samples: {depletion_params.get('volume_samples', 1_000_000):,}\n")

        openmc.calculate_volumes(
            cwd = run_dir,
            threads = 24,
            output = True
        )

        vol_calc_results = openmc.VolumeCalculation.from_hdf5(
            os.path.join(run_dir, 'volume_1.h5')
        )

        # ==================================================================
        # SET FUEL AND POISON MATERIAL VOLUMES FROM STOCHASTIC CALCULATION
        # ==================================================================

        geometry_factor = 6 if params["use_1/6_geometry"] else 1
        fuel_volume_simulated = 0.0
        poison_volume_simulated = 0.0

        for domain_id, vol_var in vol_calc_results.volumes.items():
            vol = vol_var.nominal_value
            vol_std = vol_var.std_dev

            if domain_id == mats.fuel.id:
                fuel_volume_simulated += vol
                print(f"\nFuel domain {domain_id}: {vol:.4f} ± {vol_std:.4f} cm³")
            elif domain_id == mats.b4c_poison.id:
                poison_volume_simulated += vol
                print(f"B4C poison domain {domain_id}: {vol:.4f} ± {vol_std:.4f} cm³")

        if fuel_volume_simulated <= 0:
            raise RuntimeError("Stochastic volume calculation returned zero fuel volume. "
                               "Check that the volume calculation bounds overlap the fuel region.")
        if poison_volume_simulated <= 0:
            raise RuntimeError("Stochastic volume calculation returned zero B4C poison volume. "
                               "Check that the volume calculation bounds overlap the poison region.")

        # Set volumes on depletable materials
        mats.fuel.volume = fuel_volume_simulated
        mats.b4c_poison.volume = poison_volume_simulated

        # Fuel mass reporting
        total_fuel_volume_full_core = fuel_volume_simulated * geometry_factor
        uco_density_g_cm3 = params["kernel_density"] / 1000.0
        u_mass_fraction = 238.0 / 268.0
        total_HM_mass_kg = (total_fuel_volume_full_core * uco_density_g_cm3 * u_mass_fraction) / 1000.0

        # B4C poison mass reporting
        total_poison_volume_full_core = poison_volume_simulated * geometry_factor
        b4c_density_g_cm3 = params["B4C_density_poison"] / 1000.0
        mass_10 = openmc.data.atomic_mass('B10')
        mass_11 = openmc.data.atomic_mass('B11')
        b10_enrichment = params["B10_enrichment_poison"]
        b10_atom_fraction = b10_enrichment
        b10_mass_fraction = (b10_atom_fraction * mass_10) / (
            b10_atom_fraction * mass_10 + (1 - b10_atom_fraction) * mass_11
        )
        total_B10_mass_kg = (total_poison_volume_full_core * b4c_density_g_cm3 * b10_mass_fraction) / 1000.0

        print(f"\nFuel volume (simulated geometry):        {fuel_volume_simulated:.4f} cm³")
        print(f"Fuel volume (full core):                 {total_fuel_volume_full_core:.4f} cm³")
        print(f"Estimated uranium mass:                  {total_HM_mass_kg:.2f} kg")
        print(f"\nB4C poison volume (simulated geometry):  {poison_volume_simulated:.4f} cm³")
        print(f"B4C poison volume (full core):           {total_poison_volume_full_core:.4f} cm³")
        print(f"Estimated B-10 mass:                     {total_B10_mass_kg:.4f} kg")

        # Store masses in params for post-processing
        params["total_HM_mass_kg"] = total_HM_mass_kg
        params["total_B10_mass_kg"] = total_B10_mass_kg

        # Update run_params.json with depletion-specific info
        params_path = os.path.join(run_dir, 'run_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                saved_params = json.load(f)
        else:
            saved_params = {}
        saved_params['n_trisos'] = n_trisos
        saved_params['fuel_volume_simulated_cm3'] = fuel_volume_simulated
        saved_params['fuel_volume_full_core_cm3'] = total_fuel_volume_full_core
        saved_params['total_HM_mass_kg'] = total_HM_mass_kg
        saved_params['poison_volume_simulated_cm3'] = poison_volume_simulated
        saved_params['poison_volume_full_core_cm3'] = total_poison_volume_full_core
        saved_params['total_B10_mass_kg'] = total_B10_mass_kg
        saved_params['fuel_material_id'] = mats.fuel.id
        saved_params['poison_material_id'] = mats.b4c_poison.id
        with open(params_path, 'w') as f:
            json.dump(saved_params, f, indent=2)

        model.export_to_xml()

        # ==================================================================
        # CONFIGURE DEPLETION CHAIN
        # ==================================================================

        full_chain_file = params.get("depletion_chain_file", None)
        if full_chain_file is None or not os.path.exists(full_chain_file):
            raise FileNotFoundError(f"Depletion chain file not found: {full_chain_file}")

        # Always generate reduced chain into the run directory
        reduced_chain_file = os.path.join(run_dir, "chain_reduced.xml")

        if params["use_reduced_chain_file"] and len(params["tracked_nuclides"]) > 0:
            chain_file = build_reduced_chain(
                full_chain_file    = full_chain_file,
                reduced_chain_file = reduced_chain_file,
                tracked_nuclides   = params["tracked_nuclides"]
            )
        else:
            print("\nUsing full depletion chain file.")
            chain_file = full_chain_file

        timesteps_days = params.get("depletion_timesteps_days", [30] * 12)

    # ==================================================================
    # CONFIGURE DEPLETION TIMESTEPS AND POWER (shared by both paths)
    # ==================================================================

    thermal_power_W = params.get("thermal_power_MW", 15.0) * 1e6

    # Scale power for 1/6 geometry (operator sees only the simulated fraction)
    if params["use_1/6_geometry"]:
        operator_power_W = thermal_power_W / 6.0
        print(f"\n1/6 geometry: scaling power from {thermal_power_W/1e6:.1f} MW to {operator_power_W/1e6:.3f} MW")
    else:
        operator_power_W = thermal_power_W

    print(f"\nDepletion chain file: {chain_file}")
    print(f"Thermal power: {thermal_power_W / 1e6:.1f} MW (full core)")
    print(f"Operator power: {operator_power_W / 1e6:.3f} MW (simulated geometry)")
    print(f"Number of timesteps: {len(timesteps_days)}")
    print(f"Total depletion time: {sum(timesteps_days):.0f} days ({sum(timesteps_days)/365.25:.2f} years)")
    print(f"Timesteps (days): {timesteps_days}")
    if is_restart:
        print(f"RESTART: appending {len(timesteps_days)} steps to {n_completed} previously completed steps")

    # ==================================================================
    # CREATE OPERATOR AND INTEGRATOR
    # ==================================================================
    
    operator = openmc.deplete.CoupledOperator(
        model,
        chain_file = chain_file,
        normalization_mode = "fission-q",
        prev_results = prev_results
    )

    integrator_name = params.get("depletion_integrator", "PredictorIntegrator")

    integrator_map = {
        "PredictorIntegrator": openmc.deplete.PredictorIntegrator,
        "CECMIntegrator": openmc.deplete.CECMIntegrator,
        "CF4Integrator": openmc.deplete.CF4Integrator,
        "EPCRK4Integrator": openmc.deplete.EPCRK4Integrator,
        "LEQIIntegrator": openmc.deplete.LEQIIntegrator,
        "SICELIIntegrator": openmc.deplete.SICELIIntegrator,
        "SILEQIIntegrator": openmc.deplete.SILEQIIntegrator,
    }

    IntegratorClass = integrator_map.get(integrator_name)
    if IntegratorClass is None:
        print(f"  WARNING: Unknown integrator '{integrator_name}', using PredictorIntegrator")
        IntegratorClass = openmc.deplete.PredictorIntegrator

    print(f"Integrator: {IntegratorClass.__name__}")

    integrator = IntegratorClass(
        operator,
        timesteps_days,
        power = operator_power_W,
        timestep_units = 'd'
    )

    # ==================================================================
    # RUN DEPLETION
    # ==================================================================

    print(f"\n{'=' * 80}")
    if is_restart:
        print("STARTING DEPLETION RESTART CALCULATION")
    else:
        print("STARTING DEPLETION CALCULATION")
    print(f"{'=' * 80}\n")

    integrator.integrate()

    print(f"\n{'=' * 80}")
    print("DEPLETION CALCULATION COMPLETE")
    print(f"{'=' * 80}\n")

    return n_trisos

# ====================================================================================================
# POST-PROCESSING FUNCTIONS
# ====================================================================================================

def update_run_info(run_dir, n_trisos):
    """
    Update run_params.json with n_trisos after TRISO creation.
    """
    
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
    """
    print(f"{'='*80}")
    print("RUNNING POST-PROCESSING")
    print(f"{'='*80}\n")
    
    update_run_info(run_dir, n_trisos)

    params_path = os.path.join(run_dir, 'run_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            saved_params = json.load(f)
        merged_params = {**params, **saved_params}
    else:
        print("WARNING: run_params.json not found, post-processing may be missing runtime data")
        merged_params = params
    
    try:
        from burnup_estimation import run_burnup_estimation
        print("Running burnup estimation...")
        run_burnup_estimation(run_dir, merged_params, n_trisos)
    except ImportError as e:
        print(f"Warning: Could not import burnup_estimation: {e}")
    except Exception as e:
        print(f"Warning: Burnup estimation failed: {e}")
    
    try:
        from tally_plotter import run_tally_plots
        print("Running tally plotting...")
        run_tally_plots(run_dir, merged_params)
    except ImportError as e:
        print(f"Warning: Could not import tally_plotter: {e}")
    except Exception as e:
        print(f"Warning: Tally plotting failed: {e}")
    
    try:
        from spectrum_thermalization import run_spectrum_analysis
        print("Running spectrum & thermalization analysis...")
        run_spectrum_analysis(run_dir, merged_params)
    except ImportError as e:
        print(f"Warning: Could not import spectrum_thermalization: {e}")
    except Exception as e:
        print(f"Warning: Spectrum analysis failed: {e}")

    try:
        from leakage_spectrum import run_leakage_analysis
        print("Running radial leakage spectrum analysis...")
        run_leakage_analysis(run_dir, merged_params)
    except ImportError as e:
        print(f"Warning: Could not import leakage_spectrum: {e}")
    except Exception as e:
        print(f"Warning: Leakage spectrum analysis failed: {e}")

    print(f"{'='*80}")
    print("POST-PROCESSING COMPLETE")
    print(f"{'='*80}")

def run_depletion_post_processing(run_dir, params):
    """
    Run depletion-specific post-processing.
    """
    try:
        from depletion_postprocessing import run_depletion_postprocessing
        print("Running depletion post-processing...")
        run_depletion_postprocessing(run_dir, params)
    except ImportError as e:
        print(f"Warning: Could not import depletion_postprocessing: {e}")
    except Exception as e:
        print(f"Warning: Depletion post-processing failed: {e}")
        import traceback
        traceback.print_exc()

def run_parametric_post_processing(parametric_dir):
    """
    Run parametric study post-processing.
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
    
    # ----- Run Parametric Study -----
    if cfg.params["study_execution_mode"] == "ParametricStudy":
        BASE_DIR = os.path.join(OUTPUT_BASE, run_name + "_ParametricStudy" + f"_{cfg.params["parametric_param"]}")
        os.makedirs(BASE_DIR, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PARAMETRIC STUDY: {cfg.params["parametric_param"]}")
        print(f"Values: {cfg.params["parametric_values"]}")
        print(f"Base Directory: {BASE_DIR}")
        print(f"{'='*80}")
        
        for i, val in enumerate(cfg.params["parametric_values"]):
            caseNum = i + 1
            caseNumFormatted = f"{caseNum:0{len(str(len(cfg.params["parametric_values"])))+1}d}"
            runName = f"{cfg.params["parametric_param"]}_Case_{caseNumFormatted}_{val}"
            run_dir = os.path.join(BASE_DIR, runName)

            print(f"\n{'='*80}")
            print(f"Runing Case {caseNumFormatted}: {cfg.params["parametric_param"]} = {val}")
            print(f"Run Directory: {run_dir}")
            print(f"{'='*80}\n")

            params_copy = cfg.params.copy()
            params_copy[cfg.params["parametric_param"]] = val

            n_trisos = run_simulation(params_copy, run_dir)

            if cfg.params["run_post_processing"]:
                run_post_processing(run_dir, params_copy, n_trisos)
        
        run_parametric_post_processing(BASE_DIR)

        print(f"\n{'='*80}")
        print("PARAMETRIC STUDY COMPLETE")
        print(f"Results Directory: {BASE_DIR}")
        print(f"{'='*80}\n")
    
    # ----- Run Reactivity Study -----
    elif cfg.params["study_execution_mode"] == "ReactivityStudy":
        from reactivity_coefficients import run_reactivity_coefficients

        BASE_DIR_RC = os.path.join(OUTPUT_BASE, run_name + "_ReactivityCoeffs")
        os.makedirs(BASE_DIR_RC, exist_ok=True)

        run_reactivity_coefficients(
            params = cfg.params,
            base_run_dir = BASE_DIR,
            output_base_dir = BASE_DIR_RC,
            delta_T_values = cfg.params["reactivity_delta_T_values"],
            coefficients = cfg.params["reactivity_coefficients"],
            run_simulation_fn = run_simulation,
            run_post_processing_fn = run_post_processing if cfg.params["run_post_processing"] else None,
        )

    # ----- Run Depletion -----
    elif cfg.params["study_execution_mode"] == "DepletionStudy":

        # If restarting, run inside the original directory instead of creating a new one
        if cfg.params.get("restart_depletion", False) and cfg.params.get("restart_run_dir"):
            BASE_DIR = cfg.params["restart_run_dir"]
            print(f"\n{'='*80}")
            print("DEPLETION RESTART MODE")
            print(f"Restarting in original run directory: {BASE_DIR}")
            print(f"{'='*80}")
        else:
            BASE_DIR = os.path.join(OUTPUT_BASE, run_name + "_Depletion")
            print(f"\n{'='*80}")
            print("DEPLETION RUN MODE")
            print(f"Run directory: {BASE_DIR}")
            print(f"{'='*80}")

        n_trisos = run_depletion_simulation(cfg.params, BASE_DIR)

        # Run depletion-specific post-processing
        run_depletion_post_processing(BASE_DIR, cfg.params)

        print(f"\n{'='*80}")
        print("DEPLETION RUN COMPLETE")
        print(f"Results Directory: {BASE_DIR}")
        print(f"{'='*80}")

    # ----- Run Single Run -----   
    elif cfg.params["study_execution_mode"] == "SingleStudy":
        BASE_DIR = os.path.join(OUTPUT_BASE, run_name + "_SingleStudy")
        
        print(f"\n{'='*80}")
        print("SINGLE RUN MODE")
        print(f"Run directory: {BASE_DIR}")
        print(f"{'='*80}")
        
        n_trisos = run_simulation(cfg.params, BASE_DIR)

        if cfg.params["run_post_processing"]:
            run_post_processing(BASE_DIR, cfg.params, n_trisos)
        
        print(f"\n{'='*80}")
        print("SIMULATION COMPLETE")
        print(f"Results Directory: {BASE_DIR}")
        print(f"{'='*80}\n")

    else:
        print(f"\nERROR: Unknown study_execution_mode: '{cfg.params['study_execution_mode']}'")
        print("Valid modes: SingleStudy, ParametricStudy, ReactivityStudy, DepletionStudy")
        sys.exit(1)
