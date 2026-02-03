import os
import math
import shutil
import openmc
import numpy as np
import openmc.deplete
from datetime import datetime

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

def run_simulation(params, run_dir):
    # ====================================================================================================
    # 1. CREATE RUN DIRECTORY AND INITIALIZE MODEL
    # ====================================================================================================

    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    # Saves cross_sections.xml file to run directory
    # Useful to ensure used cross sections are correct for each simulation
    shutil.copy2(cross_sections_path, os.path.join(run_dir, 'cross_sections.xml'))

    model = openmc.model.Model()

    # ====================================================================================================
    # 2. MATERIAL DEFINITIONS (Material Constants And Compositions)
    # ====================================================================================================

    materials = openmc.Materials()

    # ----- Fuel Kernel -----
    fuel = openmc.Material(name="Fuel")
    fuel.add_nuclide("U235", params["enrichment"])
    fuel.add_nuclide("U238", 1.0 - params["enrichment"])
    fuel.add_element("C", 1.0)
    fuel.add_element('O', 0.50)
    fuel.set_density("kg/m3", params["kernel_density"])
    fuel.depletable = True

    # ----- TRISO Layers -----
    buffer = openmc.Material(name="Buffer") # Carbon buffer layer
    buffer.add_element("C", 1.0)
    buffer.set_density("kg/m3", params["buffer_density"])

    pyc = openmc.Material(name="PyC") # Pyrolytic carbon layer (inner and outer)
    pyc.add_element("C", 1.0)
    pyc.set_density("kg/m3", params["pyc_density"])

    sic = openmc.Material(name="SiC") # Silicon carbide layer
    sic.add_element("Si", 1.0)
    sic.add_element("C", 1.0)
    sic.set_density("kg/m3", params["sic_density"])

    # ----- Moderator/Reflector -----
    graphite = openmc.Material(name="Graphite")
    boron_mass_fraction = params["boron_ppm"] / 1e6
    A_carbon = 12.011
    A_boron = 10.811
    boron_atom_fraction = boron_mass_fraction * A_carbon / A_boron
    graphite.add_element("C", 1.0 - boron_atom_fraction)
    graphite.add_element("B", boron_atom_fraction)
    graphite.set_density("kg/m3", params["matrix_density"])

    # ----- Helium Coolant -----
    helium = openmc.Material(name="Helium")
    helium.add_nuclide("He4", 1.0)
    helium.set_density("kg/m3", params["coolant_density"])

    # ----- Boron Carbide Burnable Poison -----
    b4c_poison = openmc.Material(name="B4C_Poison")
    enrichment_10_poison = params["B10_enrichment_poison"]
    mass_10 = openmc.data.atomic_mass('B10')
    mass_11 = openmc.data.atomic_mass('B11')

    # number of atoms in one gram of boron mixture
    n_10_poison = enrichment_10_poison / mass_10
    n_11_poison = (1.0 - enrichment_10_poison) / mass_11
    total_n_poison = n_10_poison + n_11_poison
    grams_10_poison = n_10_poison / total_n_poison
    grams_11_poison = n_11_poison / total_n_poison

    # now, figure out how much carbon needs to be in the poison to get
    # an overall specified B10 weight percent
    total_b10_weight_percent_poison = params["B10_wt_percent_poison"]
    total_mass_poison = grams_10_poison / total_b10_weight_percent_poison
    carbon_mass_poison = total_mass_poison - grams_10_poison - grams_11_poison

    b4c_poison.add_nuclide('B10', grams_10_poison / total_mass_poison, 'wo')
    b4c_poison.add_nuclide('B11', grams_11_poison / total_mass_poison, 'wo')
    b4c_poison.add_element('C', carbon_mass_poison / total_mass_poison, 'wo')
    b4c_poison.set_density('kg/m3', params["B4C_density_poison"])

    # ----- Boron Carbide Control Rod -----
    b4c_control = openmc.Material(name="B4C_Control")
    enrichment_10_control = params["B10_enrichment_control"]

    # number of atoms in one gram of boron mixture
    n_10_control = enrichment_10_control / mass_10
    n_11_control = (1.0 - enrichment_10_control) / mass_11
    total_n_control = n_10_control + n_11_control
    grams_10_control = n_10_control / total_n_control
    grams_11_control = n_11_control / total_n_control

    # now, figure out how much carbon needs to be in the control rod to get
    # an overall specified B10 weight percent
    total_b10_weight_percent_control = params["B10_wt_percent_control"]
    total_mass_control = grams_10_control / total_b10_weight_percent_control
    carbon_mass_control = total_mass_control - grams_10_control - grams_11_control

    b4c_control.add_nuclide('B10', grams_10_control / total_mass_control, 'wo')
    b4c_control.add_nuclide('B11', grams_11_control / total_mass_control, 'wo')
    b4c_control.add_element('C', carbon_mass_control / total_mass_control, 'wo')
    b4c_control.set_density('kg/m3', params["B4C_density_control"])

    # ----- Incoloy 800H -----
    incoloy800H = openmc.Material(name='Incoloy 800H')
    incoloy800H.set_density('kg/m3', params["Incoloy800H_density"])

    incoloy800H.add_element('Ni', 32.5, 'wo')
    incoloy800H.add_element('Cr', 21.0, 'wo')
    incoloy800H.add_element('Al', 0.40, 'wo')
    incoloy800H.add_element('Ti', 0.40, 'wo')
    incoloy800H.add_element('C', 0.08, 'wo')
    incoloy800H.add_element('Mn', 1.0, 'wo')
    incoloy800H.add_element('Si', 0.50, 'wo')
    incoloy800H.add_element('S', 0.015, 'wo')
    incoloy800H.add_element('Cu', 0.50, 'wo')
    incoloy800H.add_element('Fe', 43.605, 'wo')

    materials += [fuel, buffer, pyc, sic, graphite, helium, b4c_poison, b4c_control, incoloy800H]
    materials.export_to_xml()

    # ----- Initialize Temperature Profiles -----

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
    # 3. TRISO PARTICLE CREATION
    # ====================================================================================================

    # Creates model of TRISO fuel particle in the following order:
    # Fuel Kernel > Carbon Buffer Layer > Inner PyC Layer > SiC Layer > Outer PyC Layer

    r_kernel = params["kernel_radius"]
    r_buffer = r_kernel + params["buffer_thickness"]
    r_ipyc   = r_buffer + params["ipyc_thickness"]
    r_sic    = r_ipyc   + params["sic_thickness"]
    r_opyc   = r_sic    + params["opyc_thickness"]

    s_fuel   = openmc.Sphere(r = r_kernel)
    s_buffer = openmc.Sphere(r = r_buffer)
    s_ipyc   = openmc.Sphere(r = r_ipyc)
    s_sic    = openmc.Sphere(r = r_sic)
    s_opyc   = openmc.Sphere(r = r_opyc)

    c_triso_fuel   = openmc.Cell(name = 'c_triso_fuel'     , fill = fuel,     region = -s_fuel)
    c_triso_buffer = openmc.Cell(name = 'c_triso_c_buffer' , fill = buffer,   region = +s_fuel & -s_buffer)
    c_triso_ipyc   = openmc.Cell(name = 'c_triso_pyc_inner', fill = pyc,      region = +s_buffer & -s_ipyc)
    c_triso_sic    = openmc.Cell(name = 'c_triso_sic'      , fill = sic,      region = +s_ipyc & -s_sic)
    c_triso_opyc   = openmc.Cell(name = 'c_triso_pyc_outer', fill = pyc,      region = +s_sic & -s_opyc)
    c_triso_matrix = openmc.Cell(name = 'c_triso_matrix'   , fill = graphite, region = +s_opyc)

    triso_universe = openmc.Universe(cells=[c_triso_fuel, c_triso_buffer, c_triso_ipyc, c_triso_sic, c_triso_opyc, c_triso_matrix])

    # ====================================================================================================
    # 4. FUEL COMPACT AND COOLANT CHANNEL LATTICE CREATION
    # ====================================================================================================

    reactor_bottom = 0.0
    reactor_top = reactor_bottom + params["core_height"]

    axial_section_height = params["core_height"] / params["n_ax_zones"]

    # Superimposed TRISO search lattice
    triso_lattice_shape = (4, 4, int(axial_section_height / 0.5))

    fuel_cyl = openmc.ZCylinder(r = params["compact_radius"])
    coolant_cyl = openmc.ZCylinder(r = params["coolant_radius"])
    poison_cyl = openmc.ZCylinder(r = params["compact_radius"])  # Same size as fuel compact

    # Create a TRISO lattice for one axial section (copied into each axial zones)
    # Center the TRISO region on the origin so it fills lattice cells appropriately
    zmin_local = -0.5 * axial_section_height
    zmax_local =  0.5 * axial_section_height
    min_z = openmc.ZPlane(z0 = zmin_local)
    max_z = openmc.ZPlane(z0 = zmax_local)

    # Region in which TRISOs are generated
    triso_region = -fuel_cyl & +min_z & -max_z

    rand_spheres = openmc.model.pack_spheres(radius=r_opyc, region=triso_region, pf=params["triso_pf"])

    print(f"Number of TRISOs created per axial zone: {len(rand_spheres)}")

    # Hard boundary filter (critical at high PF)
    # CURRENTLY NOT WORKING
    # At high PF (0.35+) this filter will not remove all unsafe TRISOs, some remain outside the lattice
    llc, urc = triso_region.bounding_box

    def valid_triso(c):
        x, y, z = c
        return (
            x*x + y*y <= (params["compact_radius"] - r_opyc)**2 and
            zmin_local + r_opyc <= z <= zmax_local - r_opyc and
            llc[0] + r_opyc <= x <= urc[0] - r_opyc and
            llc[1] + r_opyc <= y <= urc[1] - r_opyc and
            llc[2] + r_opyc <= z <= urc[2] - r_opyc
        )

    safe_trisos = [c for c in rand_spheres if valid_triso(c)]

    # Calculate actual achieved PF
    n_trisos = len(safe_trisos)
    V_triso = (4/3) * np.pi * r_opyc**3
    V_compact = np.pi * params["compact_radius"]**2 * axial_section_height
    actual_pf = n_trisos * V_triso / V_compact

    print(f"Number of safe TRISOs per axial zone: {n_trisos}")
    print(f"Requested TRISO PF: {params['triso_pf']:.3f}")
    print(f"Achieved TRISO PF: {actual_pf:.3f}\n")

    random_trisos = [openmc.model.TRISO(r_opyc, triso_universe, i) for i in safe_trisos]

    # Insert TRISOs into a lattice to accelerate point location queries
    pitch = (urc - llc) / triso_lattice_shape
    triso_lattice = openmc.model.create_triso_lattice(random_trisos, llc, pitch, triso_lattice_shape, graphite)

    axial_coords = np.linspace(reactor_bottom, reactor_top, params["n_ax_zones"] + 1)
    fuel_lattice_univs = []
    poison_lattice_univs = []

    m_colors = {}

    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        # Use the middle of the axial section to compute the temperature and density (for TH coupling)
        ax_pos = 0.5 * (z_min + z_max)
        min_z_plane = openmc.ZPlane(z0=z_min)
        max_z_plane = openmc.ZPlane(z0=z_max)

        # ----- Get temperatures for this axial zone -----
        T_coolant = T_coolant_z[idx]
        T_compact = T_compact_z[idx]
        T_matrix = T_matrix_z[idx]

        # Fuel channel cells
        fuel_ch_cell = openmc.Cell(region=-fuel_cyl, fill=triso_lattice)
        fuel_ch_cell.temperature = T_compact

        fuel_ch_matrix_cell = openmc.Cell(region=+fuel_cyl, fill=graphite)
        fuel_ch_matrix_cell.temperature = T_matrix

        # Poison channel cells
        poison_ch_cell = openmc.Cell(region=-poison_cyl, fill=b4c_poison)
        poison_ch_cell.temperature = T_matrix

        poison_ch_matrix_cell = openmc.Cell(region=+poison_cyl, fill=graphite)
        poison_ch_matrix_cell.temperature = T_matrix

        # Graphite reflector cells
        graphite_cell = openmc.Cell(fill=graphite)
        graphite_cell.temperature = T_matrix

        # Create fluid cells and clone the material to set unique densities (for TH coupling)
        coolant_matrix_cell = openmc.Cell(region=+coolant_cyl, fill=graphite)
        coolant_matrix_cell.temperature = T_matrix

        # Create coolant cell with single material (not distributed)
        coolant_helium = helium.clone()
        coolant_helium.temperature = T_coolant
        m_colors[coolant_helium] = 'red'
        
        coolant_cell = openmc.Cell(region=-coolant_cyl, fill=coolant_helium)

        # Define a universe for each type of channel (fuel, coolant, poison, and graphite)
        f = openmc.Universe(cells=[fuel_ch_cell, fuel_ch_matrix_cell])
        c = openmc.Universe(cells=[coolant_cell, coolant_matrix_cell])
        p = openmc.Universe(cells=[poison_ch_cell, poison_ch_matrix_cell])
        g = openmc.Universe(cells=[graphite_cell])

        d = [f] * 2

        # Standard fuel assembly rings
        ring0 = [g]
        ring1 = [f] * 6
        ring2 = ([f] + [c]) * 6
        ring3 = ([c] + d) * 6
        ring4 = (d + [c] + [f]) * 6        

        fuel_lattice_univs.append([ring4, ring3, ring2, ring1, ring0])

        # Poison assembly rings (outer 6 corners of ring4 replaced with poison)
        ring4_poison = []
        for i, univ in enumerate((d + [c] + [f]) * 6):
            # Replace the outermost fuel channel in each sector (position 2 in each sector)
            if i % 4 == 0:
                ring4_poison.append(p)
            else:
                ring4_poison.append(univ)

        poison_lattice_univs.append([ring4_poison, ring3, ring2, ring1, ring0])

    # ====================================================================================================
    # 5. CONTROL ROD CREATION (FOR REFLECTOR ASSEMBLIES)
    # ====================================================================================================

    # Calculate control rod geometry
    # Control rod consists of:
    # - Inner B4C absorber
    # - Incoloy sheath around the absorber
    # - Incoloy guide tube that extends from top of B4C to top reflector
    # - Helium inside guide tube when control rod is withdrawn

    r_b4c = params["control_radius"] - params["sheath_thickness"]
    r_sheath_outer = params["control_radius"]
    r_guide_inner = r_sheath_outer
    r_guide_outer = r_guide_inner + params["guide_tube_thickness"]

    # Control rod insertion depth (fraction of core height)
    control_insertion_depth = params["control_insertion"] * params["core_height"]

    # Create control rod universes for each axial zone
    control_rod_univs = []
    
    # Top of reactor (including top reflector)
    top_reflector_top = reactor_top + params["reflector_thickness"]
    
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        z_mid = 0.5 * (z_min + z_max)
        T_matrix = T_matrix_z[idx]
        T_coolant = T_coolant_z[idx]
        
        # Determine what's present at this axial position
        # Control rod extends from top down by control_insertion_depth
        control_bottom = reactor_top - control_insertion_depth
        
        control_cyl_b4c = openmc.ZCylinder(r=r_b4c)
        control_cyl_sheath_outer = openmc.ZCylinder(r=r_sheath_outer)
        control_cyl_guide_outer = openmc.ZCylinder(r=r_guide_outer)
        
        if z_mid >= control_bottom:
            # Control rod is present here (B4C + sheath + guide tube)
            b4c_cell = openmc.Cell(fill=b4c_control, region=-control_cyl_b4c)
            b4c_cell.temperature = T_matrix
            
            sheath_cell = openmc.Cell(fill=incoloy800H, 
                                     region=+control_cyl_b4c & -control_cyl_sheath_outer)
            sheath_cell.temperature = T_matrix
            
            guide_tube_cell = openmc.Cell(fill=incoloy800H,
                                         region=+control_cyl_sheath_outer & -control_cyl_guide_outer)
            guide_tube_cell.temperature = T_matrix
            
            # Graphite outside guide tube
            matrix_cell = openmc.Cell(fill=graphite, region=+control_cyl_guide_outer)
            matrix_cell.temperature = T_matrix
            
            control_univ = openmc.Universe(cells=[b4c_cell, sheath_cell, guide_tube_cell, matrix_cell])
        else:
            # Guide tube only with helium inside (control rod withdrawn)
            guide_helium = helium.clone()
            guide_helium.temperature = T_coolant
            m_colors[guide_helium] = 'red'
            
            guide_helium_cell = openmc.Cell(fill=guide_helium, region=-control_cyl_sheath_outer)
            
            guide_tube_cell = openmc.Cell(fill=incoloy800H,
                                         region=+control_cyl_sheath_outer & -control_cyl_guide_outer)
            guide_tube_cell.temperature = T_matrix
            
            # Graphite outside guide tube
            matrix_cell = openmc.Cell(fill=graphite, region=+control_cyl_guide_outer)
            matrix_cell.temperature = T_matrix
            
            control_univ = openmc.Universe(cells=[guide_helium_cell, guide_tube_cell, matrix_cell])
        
        control_rod_univs.append(control_univ)
    
    # ====================================================================================================
    # 5B. FUEL ASSEMBLY CONTROL ROD PARAMETERS
    # ====================================================================================================
    
    # For fuel assemblies, control rods are circular B4C cylinders in the center with sheath and guide tube.
    # When withdrawn, the guide tube remains and helium coolant fills the inside.
    # The inner rings (ring0, ring1, ring2) are replaced with graphite so the
    # circular control rod doesn't overlap with fuel or coolant channels.
    
    # Calculate the radii for the fuel assembly control rod (same logic as reflector control rods)
    r_b4c_fuel = params["fuel_assembly_control_radius"] - params["sheath_thickness"]
    r_sheath_outer_fuel = params["fuel_assembly_control_radius"]
    r_guide_inner_fuel = r_sheath_outer_fuel
    r_guide_outer_fuel = r_guide_inner_fuel + params["guide_tube_thickness"]

    # ====================================================================================================
    # 6. FUEL ASSEMBLY CREATION (Four Types)
    # ====================================================================================================

    bundle_pitch = 5 * params["fuel_to_coolant_distance"] * math.sqrt(3.0)

    # ----- 6.1 Standard Fuel Assembly -----
    fuel_assembly_lat = openmc.HexLattice(name = "Fuel Lattice")
    fuel_assembly_lat.orientation = 'x'
    fuel_assembly_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_lat.universes = fuel_lattice_univs

    graphite_outer_cell = openmc.Cell(fill=graphite)
    inf_graphite_universe = openmc.Universe(cells=[graphite_outer_cell])
    fuel_assembly_lat.outer = inf_graphite_universe

    hex_prism_fuel = openmc.model.hexagonal_prism(bundle_pitch / math.sqrt(3.0), 'x')
    min_z = openmc.ZPlane(z0 = reactor_bottom)
    max_z = openmc.ZPlane(z0 = reactor_top)

    fuel_assembly_cell = openmc.Cell(fill=fuel_assembly_lat, region=hex_prism_fuel & +min_z & -max_z)
    fuel_assembly_univ = openmc.Universe(cells=[fuel_assembly_cell])

    # ----- 6.2 Fuel Assembly with Poison Rods -----
    fuel_assembly_poison_lat = openmc.HexLattice(name="Fuel Lattice with Poison")
    fuel_assembly_poison_lat.orientation = 'x'
    fuel_assembly_poison_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_poison_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_poison_lat.universes = poison_lattice_univs
    fuel_assembly_poison_lat.outer = inf_graphite_universe

    fuel_assembly_poison_cell = openmc.Cell(fill=fuel_assembly_poison_lat, 
                                           region=hex_prism_fuel & +min_z & -max_z)
    fuel_assembly_poison_univ = openmc.Universe(cells=[fuel_assembly_poison_cell])

    # ----- 6.3 Fuel Assembly with Central CIRCULAR Control Rod (FIXED) -----
    # Create the assembly with ring0, ring1, ring2 replaced by graphite
    # Then overlay a cylindrical control rod region on the graphite center
    # Control rod includes B4C absorber, Incoloy sheath, and guide tube (same as reflector)
    
    # Create modified lattice universes with graphite in inner rings
    fuel_control_lattice_univs = []
    
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        T_matrix = T_matrix_z[idx]
        T_coolant = T_coolant_z[idx]
        T_compact = T_compact_z[idx]
        
        # Get the outer rings from the standard fuel lattice
        base_univs = fuel_lattice_univs[idx]
        ring4 = base_univs[0]  # Outermost ring
        ring3 = base_univs[1]
        
        # Create graphite universe for inner rings (ring0, ring1, ring2)
        graphite_inner_cell = openmc.Cell(fill=graphite)
        graphite_inner_cell.temperature = T_matrix
        g_inner = openmc.Universe(cells=[graphite_inner_cell])
        
        # Replace ring0, ring1, ring2 with graphite
        ring2_graphite = [g_inner] * 12
        ring1_graphite = [g_inner] * 6
        ring0_graphite = [g_inner]
        
        control_assembly_univs = [ring4, ring3, ring2_graphite, ring1_graphite, ring0_graphite]
        fuel_control_lattice_univs.append(control_assembly_univs)
    
    # Create a new hex lattice for the controlled fuel assembly with graphite inner rings
    fuel_assembly_control_lat = openmc.HexLattice(name="Fuel Lattice for Control Assembly")
    fuel_assembly_control_lat.orientation = 'x'
    fuel_assembly_control_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_control_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_control_lat.universes = fuel_control_lattice_univs
    fuel_assembly_control_lat.outer = inf_graphite_universe
    
    # Create the cylindrical control rod surfaces for fuel assemblies
    fuel_control_cyl_b4c = openmc.ZCylinder(r=r_b4c_fuel)
    fuel_control_cyl_sheath_outer = openmc.ZCylinder(r=r_sheath_outer_fuel)
    fuel_control_cyl_guide_outer = openmc.ZCylinder(r=r_guide_outer_fuel)
    
    # Create axially-segmented cells for the control rod region
    fuel_assembly_control_cells = []
    
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        z_mid = 0.5 * (z_min + z_max)
        T_matrix = T_matrix_z[idx]
        T_coolant = T_coolant_z[idx]
        
        min_z_plane = openmc.ZPlane(z0=z_min)
        max_z_plane = openmc.ZPlane(z0=z_max)
        
        # Determine if control rod is inserted at this axial position
        control_bottom = reactor_top - control_insertion_depth
        
        # Region for this axial slice
        axial_region = +min_z_plane & -max_z_plane
        
        if z_mid >= control_bottom:
            # Control rod INSERTED - B4C + sheath + guide tube
            b4c_cell = openmc.Cell(fill=b4c_control, 
                                  region=-fuel_control_cyl_b4c & axial_region & hex_prism_fuel)
            b4c_cell.temperature = T_matrix
            
            sheath_cell = openmc.Cell(fill=incoloy800H,
                                     region=+fuel_control_cyl_b4c & -fuel_control_cyl_sheath_outer & axial_region & hex_prism_fuel)
            sheath_cell.temperature = T_matrix
            
            guide_tube_cell = openmc.Cell(fill=incoloy800H,
                                         region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region & hex_prism_fuel)
            guide_tube_cell.temperature = T_matrix
            
            fuel_assembly_control_cells.extend([b4c_cell, sheath_cell, guide_tube_cell])
        else:
            # Control rod WITHDRAWN - Guide tube with helium inside
            control_helium = helium.clone()
            control_helium.temperature = T_coolant
            m_colors[control_helium] = 'red'
            
            helium_cell = openmc.Cell(fill=control_helium, 
                                     region=-fuel_control_cyl_sheath_outer & axial_region & hex_prism_fuel)
            
            guide_tube_cell = openmc.Cell(fill=incoloy800H,
                                         region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region & hex_prism_fuel)
            guide_tube_cell.temperature = T_matrix
            
            fuel_assembly_control_cells.extend([helium_cell, guide_tube_cell])
    
    # Create the hex lattice cell (outside the control rod guide tube)
    fuel_lattice_cell = openmc.Cell(fill=fuel_assembly_control_lat, 
                                   region=+fuel_control_cyl_guide_outer & hex_prism_fuel & +min_z & -max_z)
    
    # Combine all cells into the fuel assembly with control rod universe
    all_control_assembly_cells = fuel_assembly_control_cells + [fuel_lattice_cell]
    fuel_assembly_control_univ = openmc.Universe(cells=all_control_assembly_cells)

    # ----- 6.4 Fuel Assembly with Poison Rods AND Central CIRCULAR Control Rod (FIXED) -----
    # Similar to 6.3 but using the poison lattice for outer rings
    # Control rod includes B4C absorber, Incoloy sheath, and guide tube (same as reflector)
    
    # Create modified lattice universes with graphite in inner rings and poison in outer ring
    fuel_control_poison_lattice_univs = []
    
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        T_matrix = T_matrix_z[idx]
        
        # Get the outer rings from the poison lattice
        poison_univs = poison_lattice_univs[idx]
        ring4_poison = poison_univs[0]  # Outermost ring with poison
        ring3 = poison_univs[1]
        
        # Create graphite universe for inner rings (ring0, ring1, ring2)
        graphite_inner_cell = openmc.Cell(fill=graphite)
        graphite_inner_cell.temperature = T_matrix
        g_inner = openmc.Universe(cells=[graphite_inner_cell])
        
        # Replace ring0, ring1, ring2 with graphite
        ring2_graphite = [g_inner] * 12
        ring1_graphite = [g_inner] * 6
        ring0_graphite = [g_inner]
        
        control_poison_assembly_univs = [ring4_poison, ring3, ring2_graphite, ring1_graphite, ring0_graphite]
        fuel_control_poison_lattice_univs.append(control_poison_assembly_univs)
    
    fuel_assembly_control_poison_lat = openmc.HexLattice(name="Fuel Lattice with Poison for Control Assembly")
    fuel_assembly_control_poison_lat.orientation = 'x'
    fuel_assembly_control_poison_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_control_poison_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_control_poison_lat.universes = fuel_control_poison_lattice_univs
    fuel_assembly_control_poison_lat.outer = inf_graphite_universe
    
    # Create axially-segmented cells for the control rod region (with poison lattice)
    fuel_assembly_control_poison_cells = []
    
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        z_mid = 0.5 * (z_min + z_max)
        T_matrix = T_matrix_z[idx]
        T_coolant = T_coolant_z[idx]
        
        min_z_plane = openmc.ZPlane(z0=z_min)
        max_z_plane = openmc.ZPlane(z0=z_max)
        
        control_bottom = reactor_top - control_insertion_depth
        axial_region = +min_z_plane & -max_z_plane
        
        if z_mid >= control_bottom:
            # Control rod INSERTED - B4C + sheath + guide tube
            b4c_cell = openmc.Cell(fill=b4c_control, 
                                  region=-fuel_control_cyl_b4c & axial_region & hex_prism_fuel)
            b4c_cell.temperature = T_matrix
            
            sheath_cell = openmc.Cell(fill=incoloy800H,
                                     region=+fuel_control_cyl_b4c & -fuel_control_cyl_sheath_outer & axial_region & hex_prism_fuel)
            sheath_cell.temperature = T_matrix
            
            guide_tube_cell = openmc.Cell(fill=incoloy800H,
                                         region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region & hex_prism_fuel)
            guide_tube_cell.temperature = T_matrix
            
            fuel_assembly_control_poison_cells.extend([b4c_cell, sheath_cell, guide_tube_cell])
        else:
            # Control rod WITHDRAWN - Guide tube with helium inside
            control_helium = helium.clone()
            control_helium.temperature = T_coolant
            m_colors[control_helium] = 'red'
            
            helium_cell = openmc.Cell(fill=control_helium, 
                                     region=-fuel_control_cyl_sheath_outer & axial_region & hex_prism_fuel)
            
            guide_tube_cell = openmc.Cell(fill=incoloy800H,
                                         region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region & hex_prism_fuel)
            guide_tube_cell.temperature = T_matrix
            
            fuel_assembly_control_poison_cells.extend([helium_cell, guide_tube_cell])
    
    # Create the poison hex lattice cell (outside the control rod guide tube)
    poison_lattice_cell = openmc.Cell(fill=fuel_assembly_control_poison_lat, 
                                     region=+fuel_control_cyl_guide_outer & hex_prism_fuel & +min_z & -max_z)
    
    # Combine all cells
    all_control_poison_assembly_cells = fuel_assembly_control_poison_cells + [poison_lattice_cell]
    fuel_assembly_control_poison_univ = openmc.Universe(cells=all_control_poison_assembly_cells)

    # ====================================================================================================
    # 7. REFLECTOR ASSEMBLY CREATION (WITH CENTRAL CONTROL ROD)
    # ====================================================================================================

    hex_prism_refl = openmc.model.hexagonal_prism(bundle_pitch / math.sqrt(3.0), 'x')

    # Create reflector assembly with same axial zones as fuel assemblies
    # Plus additional zones for top reflector where guide tubes extend
    reflector_lattice_univs = []

    # Main core region - control rods with sheath and guide tube
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        reflector_lattice_univs.append([[control_rod_univs[idx]]])

    # Create reflector assembly lattice (similar structure to fuel assemblies)
    reflector_assembly_lat = openmc.HexLattice(name="Reflector Lattice with Control")
    reflector_assembly_lat.orientation = 'x'
    reflector_assembly_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    reflector_assembly_lat.pitch = (bundle_pitch, axial_section_height)
    reflector_assembly_lat.universes = reflector_lattice_univs

    # Outer universe for reflector lattice
    graphite_outer_refl = openmc.Cell(fill=graphite)
    graphite_outer_refl.temperature = params["reflector_min"]
    inf_graphite_refl_universe = openmc.Universe(cells=[graphite_outer_refl])
    reflector_assembly_lat.outer = inf_graphite_refl_universe

    # Reflector assembly cell
    min_z = openmc.ZPlane(z0=reactor_bottom)
    max_z = openmc.ZPlane(z0=reactor_top)

    reflector_assembly_cell = openmc.Cell(fill=reflector_assembly_lat, region=hex_prism_refl & +min_z & -max_z)
    reflector_assembly_univ = openmc.Universe(cells=[reflector_assembly_cell])

    # ====================================================================================================
    # 8. CORE LATTICE CREATION
    # ====================================================================================================

    f = fuel_assembly_univ
    r = reflector_assembly_univ
    fp = fuel_assembly_poison_univ
    fc = fuel_assembly_control_univ
    fcp = fuel_assembly_control_poison_univ

    ring0 = [fcp]
    ring1 = [fp] * 6
    ring2 = ([f] + [f]) * 6
    # ring3 = [f] * 18
    ring3 = ([r] + [f] + [f]) * 6
    # ring4 = ([r] + [f] + [f] + [f]) * 6
    # ring4 = ([f] + [f] + [f] + [f]) * 6

    # core_lattice_univs = [ring4, ring3, ring2, ring1, ring0]
    core_lattice_univs = [ring3, ring2, ring1, ring0]
    # core_lattice_univs = [ring2, ring1, ring0]

    core_lattice = openmc.HexLattice(name="Core Lattice")
    core_lattice.center = (0.0, 0.0)
    core_lattice.pitch = (bundle_pitch,)
    core_lattice.universes = core_lattice_univs

    # ====================================================================================================
    # 9. FULL CORE AND OUTER PERMANENT REFLECTOR CREATION (UPDATED WITH TEMPERATURE ZONES)
    # ====================================================================================================

    # The outer permanent reflector (region outside core lattice but inside core_cyl)
    # needs axial temperature zones matching the reflector temperature profile

    # Create axially-segmented outer reflector cells
    outer_refl_cells = []

    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        min_z_plane = openmc.ZPlane(z0=z_min)
        max_z_plane = openmc.ZPlane(z0=z_max)
        
        # Get temperature for this axial zone
        T_reflector = T_reflector_z[idx]
        
        # Clone graphite material and set temperature
        graphite_clone = graphite.clone()
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
        graphite_top = graphite.clone()
        m_colors[graphite_top] = 'darkblue'
        graphite_top.temperature = T_reflector_axial
        
        graphite_bottom = graphite.clone()
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
        graphite_top = graphite.clone()
        m_colors[graphite_top] = 'darkblue'
        graphite_top.temperature = T_reflector_axial
        
        graphite_bottom = graphite.clone()
        m_colors[graphite_bottom] = 'darkblue'
        graphite_bottom.temperature = T_reflector_axial

        top_refl_cell = openmc.Cell(fill=graphite_top, region=-core_cyl & +max_z & -top_refl)

        bottom_refl_cell = openmc.Cell(fill=graphite_bottom, region=-core_cyl & +bottom_refl & -min_z)

        geometry = openmc.Geometry([core_cell, top_refl_cell, bottom_refl_cell])
        model.geometry = geometry

    # ====================================================================================================
    # 10. GEOMETRY PLOT GENERATION
    # ====================================================================================================

    m_colors[fuel] = 'palegreen'
    m_colors[buffer] = 'sandybrown'
    m_colors[pyc] = 'orange'
    m_colors[sic] = 'yellow'
    m_colors[graphite] = 'darkblue'
    m_colors[b4c_poison] = 'purple'
    m_colors[b4c_control] = 'black'
    m_colors[incoloy800H] = 'gray'

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
    fuel_filter = openmc.MaterialFilter(fuel)
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

    model.export_to_xml()

    openmc.plot_geometry(output=False, cwd=run_dir)

    openmc.run(
        cwd=run_dir,
        threads=24,
        output=True
    )

    return n_trisos

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
    "n_fuel_assemblies_per_core": 31,

    # ----- Core Dimensions -----
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
    "control_radius": 5.08,                    # Radius for reflector assembly control rods
    "fuel_assembly_control_radius": 5.08,      # Radius for circular control rods in fuel assemblies
    "sheath_thickness": 0.3, 
    "guide_tube_thickness": 0.5,  
    "control_insertion": 1.0,                  # Fractional control rod insertion (0-1.0)
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
    "reflector_max": 968.15
}

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

    run_parametric_study = parametric_param is not None and len(parametric_values) > 0
    
    # ----- Run Parametric Study -----
    if run_parametric_study:
        # Add "_ParametricStudy" suffix to base run folder
        BASE_DIR = os.path.join(OUTPUT_BASE, run_name + "_ParametricStudy" + f"_{parametric_param}")
        os.makedirs(BASE_DIR, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PARAMETRIC STUDY: {parametric_param}")
        print(f"Values: {parametric_values}")
        print(f"Base Directory: {BASE_DIR}")
        print(f"{'='*80}")
        
        # Iteratively run simulation for values in parametric study
        for i, val in enumerate(parametric_values):
            caseNum = i + 1
            caseNumFormatted = f"{caseNum:0{len(str(len(parametric_values)))+1}d}"

            runName = f"{parametric_param}_Case_{caseNumFormatted}_{val}"
            
            # Create run-specific directory for current value
            run_dir = os.path.join(BASE_DIR, runName)

            print(f"\n{'='*80}")
            print(f"Runing Case {caseNumFormatted}: {parametric_param} = {val}")
            print(f"Run Directory: {run_dir}")
            print(f"{'='*80}\n")

            # Create temporary copy of params and modify the current specified parameter
            params_copy = params.copy()
            params_copy[parametric_param] = val

            run_simulation(params_copy, run_dir)
        
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
        
        n_trisos = run_simulation(params, BASE_DIR)

        estimate_fuel_cycle_length(params, n_trisos, BASE_DIR, 160_000)
        
        print(f"\n{'='*80}")
        print("SIMULATION COMPLETE")
        print(f"Results Directory: {BASE_DIR}")
        print(f"{'='*80}\n")