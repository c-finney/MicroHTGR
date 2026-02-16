import math
import openmc
import numpy as np

def create_assembly_univs(params, mats, T_coolant_z, T_compact_z, T_matrix_z, triso_lattice, axial_coords, reactor_bottom, reactor_top):
    """
    Create all fuel assembly variants and reflector assemblies.
    
    Returns:
        dict: Dictionary mapping assembly codes to OpenMC Universe objects
        dict: Material colors dictionary for plotting
    """
    
    m_colors = {}
    
    axial_section_height = params["core_height"] / params["n_ax_zones"]
    bundle_pitch = 5 * params["fuel_to_coolant_distance"] * math.sqrt(3.0)
    
    # Control rod insertion depth
    control_insertion_depth = params["control_insertion"] * params["core_height"]
    
    # ================================================================================================
    # GEOMETRY SURFACES
    # ================================================================================================
    
    fuel_cyl = openmc.ZCylinder(r=params["compact_radius"])
    coolant_cyl = openmc.ZCylinder(r=params["coolant_radius"])
    poison_cyl = openmc.ZCylinder(r=params["compact_radius"])
    
    hex_prism_fuel = openmc.model.hexagonal_prism(bundle_pitch / math.sqrt(3.0), 'x')
    hex_prism_refl = openmc.model.hexagonal_prism(bundle_pitch / math.sqrt(3.0), 'x')
    
    min_z = openmc.ZPlane(z0=reactor_bottom)
    max_z = openmc.ZPlane(z0=reactor_top)
    
    # Reflector control rod radii
    r_b4c = params["control_radius"] - params["sheath_thickness"]
    r_sheath_outer = params["control_radius"]
    r_guide_outer = r_sheath_outer + params["guide_tube_thickness"]
    
    # Fuel assembly control rod radii
    r_b4c_fuel = params["fuel_assembly_control_radius"] - params["sheath_thickness"]
    r_sheath_outer_fuel = params["fuel_assembly_control_radius"]
    r_guide_outer_fuel = r_sheath_outer_fuel + params["guide_tube_thickness"]
    
    # Fuel assembly control rod surfaces
    fuel_control_cyl_b4c = openmc.ZCylinder(r=r_b4c_fuel)
    fuel_control_cyl_sheath_outer = openmc.ZCylinder(r=r_sheath_outer_fuel)
    fuel_control_cyl_guide_outer = openmc.ZCylinder(r=r_guide_outer_fuel)
    
    # ================================================================================================
    # LATTICE UNIVERSES FOR EACH AXIAL ZONE
    # ================================================================================================
    
    fuel_lattice_univs = []
    poison_lattice_univs = []
    poison_lattice_univs_alt = []
    control_rod_univs = []
    
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        z_mid = 0.5 * (z_min + z_max)
        min_z_plane = openmc.ZPlane(z0=z_min)
        max_z_plane = openmc.ZPlane(z0=z_max)
        
        T_coolant = T_coolant_z[idx]
        T_compact = T_compact_z[idx]
        T_matrix = T_matrix_z[idx]
        
        # ----- Fuel channel cells -----
        fuel_ch_cell = openmc.Cell(region=-fuel_cyl, fill=triso_lattice)
        fuel_ch_cell.temperature = T_compact
        
        fuel_ch_matrix_cell = openmc.Cell(region=+fuel_cyl, fill=mats.graphite)
        fuel_ch_matrix_cell.temperature = T_matrix
        
        # ----- Poison channel cells -----
        poison_ch_cell = openmc.Cell(region=-poison_cyl, fill=mats.b4c_poison)
        poison_ch_cell.temperature = T_matrix
        
        poison_ch_matrix_cell = openmc.Cell(region=+poison_cyl, fill=mats.graphite)
        poison_ch_matrix_cell.temperature = T_matrix
        
        # ----- Graphite cell -----
        graphite_cell = openmc.Cell(fill=mats.graphite)
        graphite_cell.temperature = T_matrix
        
        # ----- Coolant cells -----
        coolant_matrix_cell = openmc.Cell(region=+coolant_cyl, fill=mats.graphite)
        coolant_matrix_cell.temperature = T_matrix
        
        coolant_helium = mats.helium.clone()
        coolant_helium.temperature = T_coolant
        m_colors[coolant_helium] = 'red'
        
        coolant_cell = openmc.Cell(region=-coolant_cyl, fill=coolant_helium)
        
        # ----- Channel universes -----
        f = openmc.Universe(cells=[fuel_ch_cell, fuel_ch_matrix_cell])
        c = openmc.Universe(cells=[coolant_cell, coolant_matrix_cell])
        p = openmc.Universe(cells=[poison_ch_cell, poison_ch_matrix_cell])
        g = openmc.Universe(cells=[graphite_cell])
        
        d = [f] * 2
        
        # ----- Standard fuel assembly rings -----
        ring0 = [g]
        ring1 = [f] * 6
        ring2 = ([f] + [c]) * 6
        ring3 = ([c] + d) * 6
        ring4 = (d + [c] + [f]) * 6
        
        fuel_lattice_univs.append([ring4, ring3, ring2, ring1, ring0])
        
        # ----- Poison assembly rings -----
        ring4_poison = []
        for i, univ in enumerate((d + [c] + [f]) * 6):
            if i % 4 == 0:
                ring4_poison.append(p)
            else:
                ring4_poison.append(univ)
        
        poison_lattice_univs.append([ring4_poison, ring3, ring2, ring1, ring0])
        poison_lattice_univs_alt.append([ring4, ring3, ring2, ring1, [p]])
        
        # ----- Control rod universes (for reflector assemblies) -----
        control_bottom = reactor_top - control_insertion_depth
        
        control_cyl_b4c = openmc.ZCylinder(r=r_b4c)
        control_cyl_sheath_outer = openmc.ZCylinder(r=r_sheath_outer)
        control_cyl_guide_outer = openmc.ZCylinder(r=r_guide_outer)
        
        if z_mid >= control_bottom:
            b4c_cell = openmc.Cell(fill=mats.b4c_control, region=-control_cyl_b4c)
            b4c_cell.temperature = T_matrix
            
            sheath_cell = openmc.Cell(fill=mats.incoloy800H,
                                     region=+control_cyl_b4c & -control_cyl_sheath_outer)
            sheath_cell.temperature = T_matrix
            
            guide_tube_cell = openmc.Cell(fill=mats.incoloy800H,
                                         region=+control_cyl_sheath_outer & -control_cyl_guide_outer)
            guide_tube_cell.temperature = T_matrix
            
            matrix_cell = openmc.Cell(fill=mats.graphite, region=+control_cyl_guide_outer)
            matrix_cell.temperature = T_matrix
            
            control_univ = openmc.Universe(cells=[b4c_cell, sheath_cell, guide_tube_cell, matrix_cell])
        
        else:
            guide_helium = mats.helium.clone()
            guide_helium.temperature = T_coolant
            m_colors[guide_helium] = 'red'
            
            guide_helium_cell = openmc.Cell(fill=guide_helium, region=-control_cyl_sheath_outer)
            
            guide_tube_cell = openmc.Cell(fill=mats.incoloy800H,
                                         region=+control_cyl_sheath_outer & -control_cyl_guide_outer)
            guide_tube_cell.temperature = T_matrix
            
            matrix_cell = openmc.Cell(fill=mats.graphite, region=+control_cyl_guide_outer)
            matrix_cell.temperature = T_matrix
            
            control_univ = openmc.Universe(cells=[guide_helium_cell, guide_tube_cell, matrix_cell])
        
        control_rod_univs.append(control_univ)
    
    # ================================================================================================
    # STANDARD FUEL ASSEMBLY
    # ================================================================================================
    
    fuel_assembly_lat = openmc.HexLattice(name="Fuel Lattice")
    fuel_assembly_lat.orientation = 'x'
    fuel_assembly_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_lat.universes = fuel_lattice_univs
    
    graphite_outer_cell = openmc.Cell(fill=mats.graphite)
    inf_graphite_universe = openmc.Universe(cells=[graphite_outer_cell])
    fuel_assembly_lat.outer = inf_graphite_universe
    
    fuel_assembly_cell = openmc.Cell(fill=fuel_assembly_lat, region=hex_prism_fuel & +min_z & -max_z)
    fuel_assembly_univ = openmc.Universe(cells=[fuel_assembly_cell])
    
    # ================================================================================================
    # FUEL ASSEMBLY WITH 6 EDGE POISON RODS
    # ================================================================================================
    
    fuel_assembly_poison_lat = openmc.HexLattice(name="Fuel Lattice with Poison")
    fuel_assembly_poison_lat.orientation = 'x'
    fuel_assembly_poison_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_poison_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_poison_lat.universes = poison_lattice_univs
    fuel_assembly_poison_lat.outer = inf_graphite_universe
    
    fuel_assembly_poison_cell = openmc.Cell(fill=fuel_assembly_poison_lat,
                                           region=hex_prism_fuel & +min_z & -max_z)
    fuel_assembly_poison_univ = openmc.Universe(cells=[fuel_assembly_poison_cell])
    
    # ================================================================================================
    # FUEL ASSEMBLY WITH SINGULAR CENTRAL POISON ROD
    # ================================================================================================
    
    fuel_assembly_poison_lat_alt = openmc.HexLattice(name="Fuel Lattice with Poison Alt")
    fuel_assembly_poison_lat_alt.orientation = 'x'
    fuel_assembly_poison_lat_alt.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_poison_lat_alt.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_poison_lat_alt.universes = poison_lattice_univs_alt
    fuel_assembly_poison_lat_alt.outer = inf_graphite_universe
    
    fuel_assembly_poison_cell_alt = openmc.Cell(fill=fuel_assembly_poison_lat_alt,
                                               region=hex_prism_fuel & +min_z & -max_z)
    fuel_assembly_poison_univ_alt = openmc.Universe(cells=[fuel_assembly_poison_cell_alt])
    
    # ================================================================================================
    # FUEL ASSEMBLY WITH CENTRAL CONTROL ROD
    # ================================================================================================
    
    fuel_control_lattice_univs = []
    
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        T_matrix = T_matrix_z[idx]
        
        base_univs = fuel_lattice_univs[idx]
        ring4 = base_univs[0]
        ring3 = base_univs[1]
        ring2 = base_univs[2]
        
        graphite_inner_cell = openmc.Cell(fill=mats.graphite)
        graphite_inner_cell.temperature = T_matrix
        g_inner = openmc.Universe(cells=[graphite_inner_cell])
        
        ring1_graphite = [g_inner] * 6
        ring0_graphite = [g_inner]
        
        fuel_control_lattice_univs.append([ring4, ring3, ring2, ring1_graphite, ring0_graphite])
    
    fuel_assembly_control_lat = openmc.HexLattice(name="Fuel Lattice for Control Assembly")
    fuel_assembly_control_lat.orientation = 'x'
    fuel_assembly_control_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_control_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_control_lat.universes = fuel_control_lattice_univs
    fuel_assembly_control_lat.outer = inf_graphite_universe
    
    fuel_assembly_control_cells = []
    
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        z_mid = 0.5 * (z_min + z_max)
        min_z_plane = openmc.ZPlane(z0=z_min)
        max_z_plane = openmc.ZPlane(z0=z_max)

        T_matrix = T_matrix_z[idx]
        T_coolant = T_coolant_z[idx]
        
        control_bottom = reactor_top - control_insertion_depth

        axial_region = +min_z_plane & -max_z_plane
        
        if z_mid >= control_bottom:
            b4c_cell = openmc.Cell(fill=mats.b4c_control,
                                  region=-fuel_control_cyl_b4c & axial_region & hex_prism_fuel)
            b4c_cell.temperature = T_matrix
            
            sheath_cell = openmc.Cell(fill=mats.incoloy800H,
                                     region=+fuel_control_cyl_b4c & -fuel_control_cyl_sheath_outer & axial_region & hex_prism_fuel)
            sheath_cell.temperature = T_matrix
            
            guide_tube_cell = openmc.Cell(fill=mats.incoloy800H,
                                         region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region & hex_prism_fuel)
            guide_tube_cell.temperature = T_matrix
            
            fuel_assembly_control_cells.extend([b4c_cell, sheath_cell, guide_tube_cell])
        else:
            control_helium = mats.helium.clone()
            control_helium.temperature = T_coolant
            m_colors[control_helium] = 'red'
            
            helium_cell = openmc.Cell(fill=control_helium,
                                     region=-fuel_control_cyl_sheath_outer & axial_region & hex_prism_fuel)
            
            guide_tube_cell = openmc.Cell(fill=mats.incoloy800H,
                                         region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region & hex_prism_fuel)
            guide_tube_cell.temperature = T_matrix
            
            fuel_assembly_control_cells.extend([helium_cell, guide_tube_cell])
    
    fuel_lattice_cell = openmc.Cell(fill=fuel_assembly_control_lat,
                                   region=+fuel_control_cyl_guide_outer & hex_prism_fuel & +min_z & -max_z)
    
    all_control_assembly_cells = fuel_assembly_control_cells + [fuel_lattice_cell]
    fuel_assembly_control_univ = openmc.Universe(cells=all_control_assembly_cells)
    
    # ================================================================================================
    # FUEL ASSEMBLY WITH CONTROL ROD AND POISON RODS
    # ================================================================================================
    
    fuel_control_poison_lattice_univs = []
    
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        T_matrix = T_matrix_z[idx]
        
        poison_univs = poison_lattice_univs[idx]
        ring4_poison = poison_univs[0]
        ring3 = poison_univs[1]
        ring2 = poison_univs[2]
        
        graphite_inner_cell = openmc.Cell(fill=mats.graphite)
        graphite_inner_cell.temperature = T_matrix
        g_inner = openmc.Universe(cells=[graphite_inner_cell])
        
        ring1_graphite = [g_inner] * 6
        ring0_graphite = [g_inner]
        
        fuel_control_poison_lattice_univs.append([ring4_poison, ring3, ring2, ring1_graphite, ring0_graphite])
    
    fuel_assembly_control_poison_lat = openmc.HexLattice(name="Fuel Lattice with Poison for Control Assembly")
    fuel_assembly_control_poison_lat.orientation = 'x'
    fuel_assembly_control_poison_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_control_poison_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_control_poison_lat.universes = fuel_control_poison_lattice_univs
    fuel_assembly_control_poison_lat.outer = inf_graphite_universe
    
    fuel_assembly_control_poison_cells = []
    
    for idx, (z_min, z_max) in enumerate(zip(axial_coords[0:-1], axial_coords[1:])):
        z_mid = 0.5 * (z_min + z_max)
        min_z_plane = openmc.ZPlane(z0=z_min)
        max_z_plane = openmc.ZPlane(z0=z_max)

        T_matrix = T_matrix_z[idx]
        T_coolant = T_coolant_z[idx]
        
        control_bottom = reactor_top - control_insertion_depth
        
        axial_region = +min_z_plane & -max_z_plane
        
        if z_mid >= control_bottom:
            b4c_cell = openmc.Cell(fill=mats.b4c_control,
                                  region=-fuel_control_cyl_b4c & axial_region & hex_prism_fuel)
            b4c_cell.temperature = T_matrix
            
            sheath_cell = openmc.Cell(fill=mats.incoloy800H,
                                     region=+fuel_control_cyl_b4c & -fuel_control_cyl_sheath_outer & axial_region & hex_prism_fuel)
            sheath_cell.temperature = T_matrix
            
            guide_tube_cell = openmc.Cell(fill=mats.incoloy800H,
                                         region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region & hex_prism_fuel)
            guide_tube_cell.temperature = T_matrix
            
            fuel_assembly_control_poison_cells.extend([b4c_cell, sheath_cell, guide_tube_cell])
        else:
            control_helium = mats.helium.clone()
            control_helium.temperature = T_coolant
            m_colors[control_helium] = 'red'
            
            helium_cell = openmc.Cell(fill=control_helium,
                                     region=-fuel_control_cyl_sheath_outer & axial_region & hex_prism_fuel)
            
            guide_tube_cell = openmc.Cell(fill=mats.incoloy800H,
                                         region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region & hex_prism_fuel)
            guide_tube_cell.temperature = T_matrix
            
            fuel_assembly_control_poison_cells.extend([helium_cell, guide_tube_cell])
    
    poison_lattice_cell = openmc.Cell(fill=fuel_assembly_control_poison_lat,
                                     region=+fuel_control_cyl_guide_outer & hex_prism_fuel & +min_z & -max_z)
    
    all_control_poison_assembly_cells = fuel_assembly_control_poison_cells + [poison_lattice_cell]
    fuel_assembly_control_poison_univ = openmc.Universe(cells=all_control_poison_assembly_cells)
    
    # ================================================================================================
    # REFLECTOR ASSEMBLY WITH CONTROL ROD
    # ================================================================================================
    
    reflector_lattice_univs = []
    for idx in range(len(axial_coords) - 1):
        reflector_lattice_univs.append([[control_rod_univs[idx]]])
    
    reflector_assembly_lat = openmc.HexLattice(name="Reflector Lattice with Control")
    reflector_assembly_lat.orientation = 'x'
    reflector_assembly_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    reflector_assembly_lat.pitch = (bundle_pitch, axial_section_height)
    reflector_assembly_lat.universes = reflector_lattice_univs
    
    graphite_outer_refl = openmc.Cell(fill=mats.graphite)
    graphite_outer_refl.temperature = params["reflector_min"]
    inf_graphite_refl_universe = openmc.Universe(cells=[graphite_outer_refl])
    reflector_assembly_lat.outer = inf_graphite_refl_universe
    
    reflector_assembly_cell = openmc.Cell(fill=reflector_assembly_lat, region=hex_prism_refl & +min_z & -max_z)
    reflector_assembly_univ = openmc.Universe(cells=[reflector_assembly_cell])
    
    # ================================================================================================
    # RETURN ASSEMBLY DICTIONARY
    # ================================================================================================
    
    assemblies = {
        "f": fuel_assembly_univ,
        "r": reflector_assembly_univ,
        "fp": fuel_assembly_poison_univ,
        "fpa": fuel_assembly_poison_univ_alt,
        "fc": fuel_assembly_control_univ,
        "fcp": fuel_assembly_control_poison_univ,
    }
    
    return assemblies, m_colors, bundle_pitch

def build_core_lattice(assemblies, core_rings, bundle_pitch):
    """
    Build the core lattice from ring definitions.
    
    Args:
        assemblies: Dictionary mapping assembly codes to Universe objects
        core_rings: List of rings, each ring is a list of assembly codes
        bundle_pitch: Pitch between assemblies
    
    Returns:
        openmc.HexLattice: The core lattice
    """
    
    core_lattice_univs = []
    
    for ring_def in core_rings:
        ring = [assemblies[code] for code in ring_def]
        core_lattice_univs.append(ring)
    
    core_lattice = openmc.HexLattice(name="Core Lattice")
    core_lattice.center = (0.0, 0.0)
    core_lattice.pitch = (bundle_pitch,)
    core_lattice.universes = core_lattice_univs
    
    return core_lattice