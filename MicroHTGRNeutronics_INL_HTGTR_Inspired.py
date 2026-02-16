import os
import math
import shutil
import openmc
import numpy as np
import openmc.deplete
from datetime import datetime
import config as cfg
import materials as mats
import assembly as asm
import trisos

cross_sections_path = '/home/cade/Desktop/OpenMC/CrossSections/cross_sections.xml'
os.environ['OPENMC_CROSS_SECTIONS'] = cross_sections_path

# ====================================================================================================
# FUEL CYCLE LENGTH ESTIMATION FUNCTION
# ====================================================================================================

def estimate_fuel_cycle_length(params, n_trisos_per_zone, run_dir, burnup_limit):
    """
    Estimate fuel cycle length based on core fuel inventory and burnup limit.
    
    Args:
        params: Dictionary of reactor parameters
        n_trisos_per_zone: Number of TRISO particles per axial zone
        run_dir: Directory where results will be saved
    
    Returns:
        cycle_length_days: Estimated fuel cycle length in days
        total_HM_mass_kg: Total heavy metal mass in kg
    """
    
    print(f"\n{'='*80}")
    print("FUEL CYCLE LENGTH ESTIMATION")
    print(f"{'='*80}")
    
    # ----- 1. Calculate volume and mass of one fuel kernel -----
    
    V_kernel_cm3 = (4/3) * np.pi * params["kernel_radius"]**3
    
    # Mass of UCO in one kernel (grams)
    # Density is in kg/m³, convert to g/cm³
    rho_UCO_g_cm3 = params["kernel_density"] / 1000
    m_UCO_per_kernel_g = V_kernel_cm3 * rho_UCO_g_cm3
    
    # UCO composition: approximately U₁C₀.₅O₁.₅
    # Molecular weights: U=238, C=12, O=16
    # MW_UCO ≈ 238 + 0.5*12 + 1.5*16 = 268 g/mol
    # Mass fraction of U in UCO ≈ 238/268 ≈ 0.888
    
    U_mass_fraction = 238.0 / 268.0
    m_U_per_kernel_g = m_UCO_per_kernel_g * U_mass_fraction
    
    print(f"\nKernel Properties:")
    print(f"  Kernel radius: {params['kernel_radius']*1e4:.1f} μm")
    print(f"  Kernel volume: {V_kernel_cm3:.2e} cm³")
    print(f"  UCO density: {rho_UCO_g_cm3:.2f} g/cm³")
    print(f"  UCO mass per kernel: {m_UCO_per_kernel_g*1e6:.2f} μg")
    print(f"  U mass fraction in UCO: {U_mass_fraction:.3f}")
    print(f"  U mass per kernel: {m_U_per_kernel_g*1e6:.2f} μg")
    
    # ----- 2. Count fuel channels in core -----
    
    # From the lattice definition in your code:
    # ring0 = [f] (center assembly with fuel)
    # ring1 = [f] * 6
    # ring2 = [f] * 12
    # ring3 = [f] * 18
    # Total assemblies = 1 + 6 + 12 + 18 = 37
    
    n_assemblies = params["n_fuel_assemblies_per_core"]
    
    # Count fuel channels per assembly from your lattice definition
    # ring4 = (d + [c] + [f]) * 6, where d = [f]*2  → (2f + c + f)*6 = 18f + 6c
    # ring3 = ([c] + d) * 6 = (c + 2f)*6 = 12f + 6c
    # ring2 = ([f] + [c]) * 6 = 6f + 6c
    # ring1 = [f] * 6 = 6f
    # Total per assembly: 18f + 12f + 6f + 6f = 42 fuel channels
    
    fuel_channels_per_assembly = 42
    
    # ----- 3. Calculate total TRISO and HM mass -----
    
    total_fuel_channels = n_assemblies * fuel_channels_per_assembly
    total_trisos = n_trisos_per_zone * params["n_ax_zones"] * total_fuel_channels
    
    total_U_mass_g = total_trisos * m_U_per_kernel_g
    total_HM_mass_kg = total_U_mass_g / 1000
    
    print(f"\nCore Inventory:")
    print(f"  Number of fuel assemblies: {n_assemblies}")
    print(f"  Fuel channels per assembly: {fuel_channels_per_assembly}")
    print(f"  Total fuel channels: {total_fuel_channels:,}")
    print(f"  TRISOs per axial zone: {n_trisos_per_zone:,}")
    print(f"  Number of axial zones: {params['n_ax_zones']}")
    print(f"  Total TRISO particles: {total_trisos:,}")
    print(f"  Total uranium mass: {total_HM_mass_kg:.2f} kg")
    
    # ----- 4. Calculate fuel cycle length -----
    
    max_burnup_MWd_per_MtU = burnup_limit  # TRISO damage limit
    thermal_power_MW = 15  # Your reactor thermal power
    
    # Total energy available from fuel (MWd)
    total_energy_MWd = (total_HM_mass_kg / 1000) * max_burnup_MWd_per_MtU
    
    # Theoretical fuel cycle length (days) at 100% capacity factor
    cycle_length_days_100pct = total_energy_MWd / thermal_power_MW
    cycle_length_years_100pct = cycle_length_days_100pct / 365.25
    
    # Realistic cycle length with 90% capacity factor
    capacity_factor = 0.90
    cycle_length_days_90pct = cycle_length_days_100pct * capacity_factor
    cycle_length_years_90pct = cycle_length_days_90pct / 365.25
    
    print(f"\nFuel Cycle Length Estimate:")
    print(f"  Maximum TRISO burnup limit: {max_burnup_MWd_per_MtU:,} MWd/MtU")
    print(f"  Reactor thermal power: {thermal_power_MW} MWth")
    print(f"  Total available energy: {total_energy_MWd:.1f} MWd")
    print(f"  ")
    print(f"  Cycle length (100% capacity factor):")
    print(f"    {cycle_length_days_100pct:.1f} days ({cycle_length_years_100pct:.2f} years)")
    print(f"  ")
    print(f"  Cycle length (90% capacity factor):")
    print(f"    {cycle_length_days_90pct:.1f} days ({cycle_length_years_90pct:.2f} years)")
    
    # ----- 5. Specific power density -----
    
    specific_power_kW_per_kgU = (thermal_power_MW * 1000) / total_HM_mass_kg
    
    print(f"\nSpecific Power:")
    print(f"  {specific_power_kW_per_kgU:.1f} kW/kgU")
    
    print(f"{'='*80}\n")
    
    # ----- 6. Save results to file -----
    
    results_file = os.path.join(run_dir, 'fuel_cycle_estimate.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FUEL CYCLE LENGTH ESTIMATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("Kernel Properties:\n")
        f.write(f"  Kernel radius: {params['kernel_radius']*1e4:.1f} μm\n")
        f.write(f"  U mass per kernel: {m_U_per_kernel_g*1e6:.2f} μg\n\n")
        
        f.write("Core Inventory:\n")
        f.write(f"  Total TRISO particles: {total_trisos:,}\n")
        f.write(f"  Total uranium mass: {total_HM_mass_kg:.2f} kg\n\n")
        
        f.write("Fuel Cycle Length Estimate:\n")
        f.write(f"  Maximum burnup limit: {max_burnup_MWd_per_MtU:,} MWd/MtU\n")
        f.write(f"  Thermal power: {thermal_power_MW} MWth\n")
        f.write(f"  Total energy: {total_energy_MWd:.1f} MWd\n\n")
        
        f.write(f"  100% capacity factor: {cycle_length_days_100pct:.1f} days ({cycle_length_years_100pct:.2f} years)\n")
        f.write(f"  90% capacity factor: {cycle_length_days_90pct:.1f} days ({cycle_length_years_90pct:.2f} years)\n\n")
        
        f.write(f"Specific Power: {specific_power_kW_per_kgU:.1f} kW/kgU\n")
        f.write("="*80 + "\n")
    
    return cycle_length_days_90pct, total_HM_mass_kg

# ====================================================================================================
# MAIN SIMULATION FUNCTION
# ====================================================================================================

def run_simulation(params, core_rings, run_dir):
    # ====================================================================================================
    # CREATE RUN DIRECTORY AND INITIALIZE MODEL
    # ====================================================================================================

    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    # Saves cross_sections.xml file to run directory
    # Useful to ensure used cross sections are correct for each simulation
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

    triso_lattice = trisos.create_triso_lattice(
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

    # The outer permanent reflector (region outside core lattice but inside core_cyl)
    # needs axial temperature zones matching the reflector temperature profile

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

        top_refl_cell = openmc.Cell(fill=graphite_top, region=-core_cyl & +max_z & -top_refl)

        bottom_refl_cell = openmc.Cell(fill=graphite_bottom, region=-core_cyl & +bottom_refl & -min_z)

        geometry = openmc.Geometry([core_cell, top_refl_cell, bottom_refl_cell])
        model.geometry = geometry

    # ====================================================================================================
    # 10. GEOMETRY PLOT GENERATION
    # ====================================================================================================

    m_colors[mats.fuel] = 'palegreen'
    m_colors[mats.buffer] = 'sandybrown'
    m_colors[mats.pyc] = 'orange'
    m_colors[mats.sic] = 'yellow'
    m_colors[mats.graphite] = 'darkblue'
    m_colors[mats.b4c_poison] = 'purple'
    m_colors[mats.b4c_control] = 'black'
    m_colors[mats.incoloy800H] = 'gray'

    plot1 = openmc.Plot()
    plot1.filename = 'Core_YZ_Material'
    plot1.width = (2 * params["core_radius"], 2 * params["core_height"])
    plot1.basis = 'yz'
    plot1.origin = (0.0, 0.0, params["core_height"] / 2.0)
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
    plot3.width = (2 * params["core_radius"], 2 * params["core_height"])
    plot3.basis = 'xz'
    plot3.origin = (0.0, 0.0, params["core_height"] / 2.0)
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
    plot5.filename = 'Bundle_XY_Material'
    plot5.width = (bundle_pitch, bundle_pitch)
    plot5.basis = 'xy'
    plot5.origin = (0.0, 0.0, axial_section_height / 4.0)
    plot5.pixels = (2000, 2000)
    plot5.color_by = 'material'
    plot5.colors = m_colors

    plot6 = openmc.Plot()
    plot6.filename = 'Bundle_XY_Cell'
    plot6.width = plot5.width
    plot6.basis = plot5.basis
    plot6.origin = plot5.origin
    plot6.pixels = plot5.pixels
    plot6.color_by = 'cell'

    plot7 = openmc.Plot()
    plot7.filename = 'Core_XY_Material'
    plot7.width = (2 * params["core_radius"], 2 * params["core_radius"])
    plot7.basis = 'xy'
    plot7.origin = (0.0, 0.0, axial_section_height / 4.0)
    plot7.pixels = (1000, 1000)
    plot7.color_by = 'material'
    plot7.colors = m_colors

    plot8 = openmc.Plot()
    plot8.filename = 'Core_XY_Cell'
    plot8.width = plot7.width
    plot8.basis = plot7.basis
    plot8.origin = plot7.origin
    plot8.pixels = plot7.pixels
    plot8.color_by = 'cell'

    model.plots = openmc.Plots([plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8])

    # ====================================================================================================
    # 11. TALLY CREATION
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

    # ----- Active Core Region Spatial Mesh Tallies -----
    # Create a cylindrical mesh for active core region using Cartesian mesh that covers the cylindrical active core region
    mesh = openmc.RegularMesh()
    mesh.dimension = [500, 500, params["n_ax_zones"]]
    mesh.lower_left = [-params["core_radius"], -params["core_radius"], reactor_bottom]
    mesh.upper_right = [params["core_radius"], params["core_radius"], reactor_top]
    mesh_filter = openmc.MeshFilter(mesh)

    # Mesh tally for spatial distributions
    mesh_tally_active = openmc.Tally(name='mesh_rates')
    mesh_tally_active.filters = [mesh_filter]
    mesh_tally_active.scores = ['flux', 'fission', 'nu-fission']

    # ----- Full Core Spatial Mesh Tallies -----
    # Use coarser axial resolution for reflectors
    n_reflector_zones = 33  # Zones in each reflector
    n_total_zones = n_reflector_zones + params["n_ax_zones"] + n_reflector_zones

    mesh_full = openmc.RegularMesh()
    mesh_full.dimension = [500, 500, n_total_zones]

    mesh_bottom = reactor_bottom - params["reflector_thickness"]
    mesh_top = reactor_top + params["reflector_thickness"]

    mesh_full.lower_left = [-params["core_radius"], -params["core_radius"], mesh_bottom]
    mesh_full.upper_right = [params["core_radius"], params["core_radius"], mesh_top]
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
    # 12. MONTE CARLO SETTINGS
    # ====================================================================================================

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.batches = 50
    settings.inactive = 10
    settings.particles = 100_000
    settings.temperature = {
        'method': 'interpolation',
        'range': (293.0, 1800.0),
        'tolerance': 100.0
    }

    r_dist = openmc.stats.Uniform(a = 0.0, b = params["core_radius"])
    phi_dist = openmc.stats.Uniform(a = 0.0, b = 2*np.pi)
    z_dist = openmc.stats.Uniform(a = reactor_bottom, b = reactor_top)
    source = openmc.IndependentSource()
    source.space = openmc.stats.CylindricalIndependent(
        r = r_dist,
        phi = phi_dist,
        z = z_dist,
        origin = (0.0, 0.0, 0.0)  # center of the cylinder
    )
    settings.source = source
    model.settings = settings

    # ====================================================================================================
    # 13. RUN OPENMC
    # ====================================================================================================

    all_mats = model.geometry.get_all_materials()
    model.materials = openmc.Materials(all_mats.values())
    model.export_to_xml()

    openmc.plot_geometry(output=False, cwd=run_dir)

    openmc.run(
        cwd=run_dir,
        threads=24,
        output=True
    )

    return n_trisos

# ====================================================================================================
# STUDY EXECUTION 
# ====================================================================================================

if __name__ == "__main__":
    # ----- Create base directory structure -----
    now = datetime.now()
    run_name = f"htgr_run_{now.strftime('%m.%d.%Y_%H.%M.%S')}"
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    
    OUTPUT_BASE = os.path.join(PARENT_DIR, "MicroHTGR_Output")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    BASE_DIR = os.path.join(OUTPUT_BASE, run_name)

    run_parametric_study = cfg.parametric_param is not None and len(cfg.parametric_values) > 0
    
    # ----- Run Parametric Study -----
    if run_parametric_study:
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

            run_simulation(params_copy, cfg.core_rings, run_dir)
        
        print(f"\n{'='*80}")
        print("PARAMETRIC STUDY COMPLETE")
        print(f"Results Directory: {BASE_DIR}")
        print(f"{'='*80}\n")
     
    # ----- Run Single Study -----   
    else:
        # Add "_SingleRun" suffix to base run folder
        BASE_DIR = os.path.join(OUTPUT_BASE, run_name + "_SingleRun")
        
        print(f"\n{'='*80}")
        print("SINGLE RUN MODE")
        print(f"Run directory: {BASE_DIR}")
        print(f"{'='*80}\n")
        
        n_trisos = run_simulation(cfg.params, cfg.core_rings, BASE_DIR)

        estimate_fuel_cycle_length(cfg.params, n_trisos, BASE_DIR, 160_000)
        
        print(f"\n{'='*80}")
        print("SIMULATION COMPLETE")
        print(f"Results Directory: {BASE_DIR}")
        print(f"{'='*80}\n")