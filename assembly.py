import math
import openmc
import numpy as np

# ====================================================================================================
# ASSEMBLY UNIVERSES BUILDER FUNCTIONS
# ====================================================================================================

def build_bank_assemblies(
    bank_id, control_insertion_depth,
    params, mats, T_coolant_z, T_matrix_z,
    axial_coords, reactor_bottom, reactor_top,
    axial_section_height, bundle_pitch,
    fuel_lattice_univs, poison_lattice_univs,
    hex_prism_fuel, hex_prism_refl,
    min_z, max_z,
    fuel_control_cyl_b4c, fuel_control_cyl_sheath_outer, fuel_control_cyl_guide_outer,
    inf_graphite_universe, inf_graphite_refl_universe,
    m_colors
):
    """
    Build the three bank-dependent assembly types (r, fc, fcp) for a single
    control rod bank insertion depth.

    Returns:
        dict with keys "r", "ra" "fc", "fcp" mapped to OpenMC Universe objects
    """

    r_b4c = params["control_radius"] - params["sheath_thickness"]
    r_sheath_outer = params["control_radius"]
    r_guide_outer = r_sheath_outer + params["guide_tube_thickness"]

    control_bottom = reactor_top - control_insertion_depth

    # ==================================================================
    # REFLECTOR ASSEMBLY WITH CONTROL ROD (r)
    # ==================================================================

    control_rod_univs = []

    for idx, (z_min, z_max) in enumerate(zip(axial_coords[:-1], axial_coords[1:])):
        z_mid = 0.5 * (z_min + z_max)
        T_matrix = T_matrix_z[idx]
        T_coolant = T_coolant_z[idx]

        control_cyl_b4c = openmc.ZCylinder(r=r_b4c)
        control_cyl_sheath_outer = openmc.ZCylinder(r=r_sheath_outer)
        control_cyl_guide_outer = openmc.ZCylinder(r=r_guide_outer)

        if z_mid >= control_bottom:
            b4c_cell = openmc.Cell(fill=mats.b4c_control, region=-control_cyl_b4c)
            b4c_cell.temperature = T_matrix

            sheath_cell = openmc.Cell(
                fill=mats.incoloy800H,
                region=+control_cyl_b4c & -control_cyl_sheath_outer)
            sheath_cell.temperature = T_matrix

            guide_tube_cell = openmc.Cell(
                fill=mats.incoloy800H,
                region=+control_cyl_sheath_outer & -control_cyl_guide_outer)
            guide_tube_cell.temperature = T_matrix

            matrix_cell = openmc.Cell(fill=mats.graphite, region=+control_cyl_guide_outer)
            matrix_cell.temperature = T_matrix

            control_univ = openmc.Universe(
                cells=[b4c_cell, sheath_cell, guide_tube_cell, matrix_cell])
        else:
            guide_helium = mats.helium.clone()
            guide_helium.temperature = T_coolant
            m_colors[guide_helium] = 'red'

            guide_helium_cell = openmc.Cell(fill=guide_helium, region=-control_cyl_sheath_outer)

            guide_tube_cell = openmc.Cell(
                fill=mats.incoloy800H,
                region=+control_cyl_sheath_outer & -control_cyl_guide_outer)
            guide_tube_cell.temperature = T_matrix

            matrix_cell = openmc.Cell(fill=mats.graphite, region=+control_cyl_guide_outer)
            matrix_cell.temperature = T_matrix

            control_univ = openmc.Universe(
                cells=[guide_helium_cell, guide_tube_cell, matrix_cell])

        control_rod_univs.append(control_univ)

    reflector_lattice_univs = [[[u]] for u in control_rod_univs]

    reflector_assembly_lat = openmc.HexLattice(
        name=f"Reflector Lattice with Control Bank {bank_id}")
    reflector_assembly_lat.orientation = 'x'
    reflector_assembly_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    reflector_assembly_lat.pitch = (bundle_pitch, axial_section_height)
    reflector_assembly_lat.universes = reflector_lattice_univs
    reflector_assembly_lat.outer = inf_graphite_refl_universe

    reflector_assembly_cell = openmc.Cell(
        fill=reflector_assembly_lat, region=hex_prism_refl & +min_z & -max_z)
    reflector_assembly_univ = openmc.Universe(cells=[reflector_assembly_cell])

    # ==================================================================
    # REFLECTOR ASSEMBLY WITH 6 CONTROL RODS IN HEX RING (ra)
    # ==================================================================

    graphite_center_cell = openmc.Cell(fill=mats.graphite)
    graphite_center_cell.temperature = params["reflector_min"]
    graphite_center_univ = openmc.Universe(cells=[graphite_center_cell])

    reflector_alt_lattice_univs = []
    for idx in range(len(axial_coords) - 1):
        ring1 = [graphite_center_univ, graphite_center_univ, graphite_center_univ, control_rod_univs[idx], control_rod_univs[idx], control_rod_univs[idx]]
        ring0 = [graphite_center_univ]
        reflector_alt_lattice_univs.append([ring1, ring0])

    reflector_alt_assembly_lat = openmc.HexLattice(
        name=f"Reflector Alt Lattice with 6 Control Rods Bank {bank_id}")
    reflector_alt_assembly_lat.orientation = 'y'
    reflector_alt_assembly_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    reflector_alt_assembly_lat.pitch = (bundle_pitch / 3.0, axial_section_height)
    reflector_alt_assembly_lat.universes = reflector_alt_lattice_univs
    reflector_alt_assembly_lat.outer = inf_graphite_refl_universe

    reflector_alt_assembly_cell = openmc.Cell(
        fill=reflector_alt_assembly_lat, region=hex_prism_refl & +min_z & -max_z)
    reflector_alt_assembly_univ = openmc.Universe(cells=[reflector_alt_assembly_cell])

    # ==================================================================
    # HELPER: build the inner-ring graphite substitution lattice universes
    # (shared between fc and fcp, differing only in ring4)
    # ==================================================================

    def make_control_lattice_univs(source_lattice_univs, lat_name):
        result = []
        for idx in range(len(axial_coords) - 1):
            T_matrix = T_matrix_z[idx]
            base = source_lattice_univs[idx]
            ring4, ring3, ring2 = base[0], base[1], base[2]

            graphite_inner_cell = openmc.Cell(fill=mats.graphite)
            graphite_inner_cell.temperature = T_matrix
            g_inner = openmc.Universe(cells=[graphite_inner_cell])

            result.append([ring4, ring3, ring2, [g_inner] * 6, [g_inner]])
        return result

    # ==================================================================
    # HELPER: build axial control-rod cells for fuel assemblies
    # ==================================================================

    def make_fuel_control_cells(hex_region_with_outer_cyl):
        cells = []
        for idx, (z_min, z_max) in enumerate(zip(axial_coords[:-1], axial_coords[1:])):
            z_mid = 0.5 * (z_min + z_max)
            T_matrix = T_matrix_z[idx]
            T_coolant = T_coolant_z[idx]

            min_z_plane = openmc.ZPlane(z0=z_min)
            max_z_plane = openmc.ZPlane(z0=z_max)
            axial_region = +min_z_plane & -max_z_plane & hex_region_with_outer_cyl

            if z_mid >= control_bottom:
                b4c = openmc.Cell(
                    fill=mats.b4c_control,
                    region=-fuel_control_cyl_b4c & axial_region)
                b4c.temperature = T_matrix

                sheath = openmc.Cell(
                    fill=mats.incoloy800H,
                    region=+fuel_control_cyl_b4c & -fuel_control_cyl_sheath_outer & axial_region)
                sheath.temperature = T_matrix

                guide = openmc.Cell(
                    fill=mats.incoloy800H,
                    region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region)
                guide.temperature = T_matrix

                cells.extend([b4c, sheath, guide])
            else:
                ctrl_helium = mats.helium.clone()
                ctrl_helium.temperature = T_coolant
                m_colors[ctrl_helium] = 'red'

                he_cell = openmc.Cell(
                    fill=ctrl_helium,
                    region=-fuel_control_cyl_sheath_outer & axial_region)

                guide = openmc.Cell(
                    fill=mats.incoloy800H,
                    region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region)
                guide.temperature = T_matrix

                cells.extend([he_cell, guide])

        return cells

    # ==================================================================
    # FUEL ASSEMBLY WITH CENTRAL CONTROL ROD  (fc)
    # ==================================================================

    fc_lattice_univs = make_control_lattice_univs(
        fuel_lattice_univs, f"Fuel Lattice for Control Assembly Bank {bank_id}")

    fc_lat = openmc.HexLattice(name=f"Fuel Lattice for Control Assembly Bank {bank_id}")
    fc_lat.orientation = 'x'
    fc_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fc_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fc_lat.universes = fc_lattice_univs
    fc_lat.outer = inf_graphite_universe

    hex_inner = +fuel_control_cyl_guide_outer & hex_prism_fuel & +min_z & -max_z
    fc_lattice_cell = openmc.Cell(fill=fc_lat, region=hex_inner)

    fc_control_cells = make_fuel_control_cells(hex_prism_fuel)
    fc_univ = openmc.Universe(cells=fc_control_cells + [fc_lattice_cell])

    # ==================================================================
    # FUEL ASSEMBLY WITH CONTROL ROD AND POISON RODS  (fcp)
    # ==================================================================

    fcp_lattice_univs = make_control_lattice_univs(
        poison_lattice_univs, f"Fuel Lattice with Poison for Control Assembly Bank {bank_id}")

    fcp_lat = openmc.HexLattice(
        name=f"Fuel Lattice with Poison for Control Assembly Bank {bank_id}")
    fcp_lat.orientation = 'x'
    fcp_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fcp_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fcp_lat.universes = fcp_lattice_univs
    fcp_lat.outer = inf_graphite_universe

    fcp_lattice_cell = openmc.Cell(fill=fcp_lat, region=hex_inner)

    fcp_control_cells = make_fuel_control_cells(hex_prism_fuel)
    fcp_univ = openmc.Universe(cells=fcp_control_cells + [fcp_lattice_cell])

    return {
        "r": reflector_assembly_univ,
        "ra": reflector_alt_assembly_univ,
        "fc": fc_univ,
        "fcp": fcp_univ,
    }

def create_assembly_univs(params, mats, T_coolant_z, T_compact_z, T_matrix_z, T_reflector_z, triso_lattice, axial_coords, reactor_bottom, reactor_top):
    """
    Create all fuel assembly variants and reflector assemblies.

    Returns:
        dict: Dictionary mapping assembly codes to OpenMC Universe objects.
              Bank-dependent types use suffixes 1/2/3  (r1, r2, r3,
              fc1, fc2, fc3, fcp1, fcp2, fcp3).
              Bank-independent types have no suffix   (f, fp, fpa).
        dict: Material colors dictionary for plotting
        float: Bundle pitch
    """

    m_colors = {}

    axial_section_height = params["core_height"] / params["n_ax_zones"]
    bundle_pitch = 5 * params["fuel_to_coolant_distance"] * math.sqrt(3.0)

    # ==================================================================
    # SHARED SURFACES
    # ==================================================================

    fuel_cyl = openmc.ZCylinder(r=params["compact_radius"])
    coolant_cyl = openmc.ZCylinder(r=params["coolant_radius"])
    poison_cyl = openmc.ZCylinder(r=params["compact_radius"])

    hex_prism_fuel = openmc.model.hexagonal_prism(bundle_pitch / math.sqrt(3.0), 'x')
    hex_prism_refl = openmc.model.hexagonal_prism(bundle_pitch / math.sqrt(3.0), 'x')

    min_z = openmc.ZPlane(z0=reactor_bottom)
    max_z = openmc.ZPlane(z0=reactor_top)

    # Fuel assembly control rod surfaces (shared across all banks — same radius)
    r_b4c_fuel = params["fuel_assembly_control_radius"] - params["sheath_thickness"]
    r_sheath_outer_fuel = params["fuel_assembly_control_radius"]
    r_guide_outer_fuel = r_sheath_outer_fuel + params["guide_tube_thickness"]

    fuel_control_cyl_b4c = openmc.ZCylinder(r=r_b4c_fuel)
    fuel_control_cyl_sheath_outer = openmc.ZCylinder(r=r_sheath_outer_fuel)
    fuel_control_cyl_guide_outer = openmc.ZCylinder(r=r_guide_outer_fuel)

    # ==================================================================
    # AXIAL ZONE UNIVERSES  (bank-independent)
    # ==================================================================

    fuel_lattice_univs = []
    poison_lattice_univs = []
    poison_lattice_univs_alt = []

    for idx, (z_min, z_max) in enumerate(zip(axial_coords[:-1], axial_coords[1:])):
        T_coolant = T_coolant_z[idx]
        T_compact = T_compact_z[idx]
        T_matrix = T_matrix_z[idx]

        # Fuel channel
        fuel_ch_cell = openmc.Cell(region=-fuel_cyl, fill=triso_lattice)
        fuel_ch_cell.temperature = T_compact
        fuel_ch_matrix_cell = openmc.Cell(region=+fuel_cyl, fill=mats.graphite)
        fuel_ch_matrix_cell.temperature = T_matrix

        # Poison channel
        poison_ch_cell = openmc.Cell(region=-poison_cyl, fill=mats.b4c_poison)
        poison_ch_cell.temperature = T_matrix
        poison_ch_matrix_cell = openmc.Cell(region=+poison_cyl, fill=mats.graphite)
        poison_ch_matrix_cell.temperature = T_matrix

        # Graphite
        graphite_cell = openmc.Cell(fill=mats.graphite)
        graphite_cell.temperature = T_matrix

        # Coolant
        coolant_matrix_cell = openmc.Cell(region=+coolant_cyl, fill=mats.graphite)
        coolant_matrix_cell.temperature = T_matrix
        coolant_helium = mats.helium.clone()
        coolant_helium.temperature = T_coolant
        m_colors[coolant_helium] = 'red'
        coolant_cell = openmc.Cell(region=-coolant_cyl, fill=coolant_helium)

        f = openmc.Universe(cells=[fuel_ch_cell, fuel_ch_matrix_cell])
        c = openmc.Universe(cells=[coolant_cell, coolant_matrix_cell])
        p = openmc.Universe(cells=[poison_ch_cell, poison_ch_matrix_cell])
        g = openmc.Universe(cells=[graphite_cell])
        d = [f] * 2

        ring0 = [g]
        ring1 = [f] * 6
        ring2 = ([f] + [c]) * 6
        ring3 = ([c] + d) * 6
        ring4 = (d + [c] + [f]) * 6

        fuel_lattice_univs.append([ring4, ring3, ring2, ring1, ring0])

        ring4_poison = []
        for i, univ in enumerate((d + [c] + [f]) * 6):
            ring4_poison.append(p if i % 4 == 0 else univ)

        poison_lattice_univs.append([ring4_poison, ring3, ring2, ring1, ring0])
        poison_lattice_univs_alt.append([ring4, ring3, ring2, ring1, [p]])

    # ==================================================================
    # BANK-INDEPENDENT OUTER UNIVERSES
    # ==================================================================

    graphite_outer_cell = openmc.Cell(fill=mats.graphite)
    inf_graphite_universe = openmc.Universe(cells=[graphite_outer_cell])

    graphite_outer_refl = openmc.Cell(fill=mats.graphite)
    graphite_outer_refl.temperature = params["reflector_min"]
    inf_graphite_refl_universe = openmc.Universe(cells=[graphite_outer_refl])

    # Standard fuel assembly (f)
    fuel_assembly_lat = openmc.HexLattice(name="Fuel Lattice")
    fuel_assembly_lat.orientation = 'x'
    fuel_assembly_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_lat.universes = fuel_lattice_univs
    fuel_assembly_lat.outer = inf_graphite_universe

    fuel_assembly_cell = openmc.Cell(
        fill=fuel_assembly_lat, region=hex_prism_fuel & +min_z & -max_z)
    fuel_assembly_univ = openmc.Universe(cells=[fuel_assembly_cell])

    # Fuel assembly with 6 edge poison rods (fp)
    fuel_assembly_poison_lat = openmc.HexLattice(name="Fuel Lattice with Poison")
    fuel_assembly_poison_lat.orientation = 'x'
    fuel_assembly_poison_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_poison_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_poison_lat.universes = poison_lattice_univs
    fuel_assembly_poison_lat.outer = inf_graphite_universe

    fuel_assembly_poison_cell = openmc.Cell(
        fill=fuel_assembly_poison_lat, region=hex_prism_fuel & +min_z & -max_z)
    fuel_assembly_poison_univ = openmc.Universe(cells=[fuel_assembly_poison_cell])

    # Fuel assembly with singular central poison rod (fpa)
    fuel_assembly_poison_lat_alt = openmc.HexLattice(name="Fuel Lattice with Poison Alt")
    fuel_assembly_poison_lat_alt.orientation = 'x'
    fuel_assembly_poison_lat_alt.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fuel_assembly_poison_lat_alt.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fuel_assembly_poison_lat_alt.universes = poison_lattice_univs_alt
    fuel_assembly_poison_lat_alt.outer = inf_graphite_universe

    fuel_assembly_poison_cell_alt = openmc.Cell(
        fill=fuel_assembly_poison_lat_alt, region=hex_prism_fuel & +min_z & -max_z)
    fuel_assembly_poison_univ_alt = openmc.Universe(cells=[fuel_assembly_poison_cell_alt])

    # Pure graphite reflector block (rr)    
    rr_lattice_univs = []
    for idx in range(len(axial_coords) - 1):
        T_reflector = T_reflector_z[idx]

        graphite_rr_cell = openmc.Cell(fill=mats.graphite)
        graphite_rr_cell.temperature = T_reflector
        graphite_rr_univ = openmc.Universe(cells=[graphite_rr_cell])

        rr_lattice_univs.append([[graphite_rr_univ]])

    rr_assembly_lat = openmc.HexLattice(name="Pure Graphite Reflector Lattice")
    rr_assembly_lat.orientation = 'x'
    rr_assembly_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    rr_assembly_lat.pitch = (bundle_pitch, axial_section_height)
    rr_assembly_lat.universes = rr_lattice_univs
    rr_assembly_lat.outer = inf_graphite_refl_universe

    rr_assembly_cell = openmc.Cell(
        fill=rr_assembly_lat, region=hex_prism_refl & +min_z & -max_z)
    rr_assembly_univ = openmc.Universe(cells=[rr_assembly_cell])

    # ==================================================================
    # BANK-DEPENDENT ASSEMBLIES  (r, fc, fcp) × 3 banks
    # ==================================================================

    bank_keys = [
        ("bank_1_insertion", 1),
        ("bank_2_insertion", 2),
        ("bank_3_insertion", 3),
    ]

    assemblies = {
        "f":   fuel_assembly_univ,
        "fp":  fuel_assembly_poison_univ,
        "fpa": fuel_assembly_poison_univ_alt,
        "rr":  rr_assembly_univ,
    }

    for param_key, bank_id in bank_keys:
        insertion_depth = params[param_key] * params["core_height"]

        bank_assemblies = build_bank_assemblies(
            bank_id=bank_id,
            control_insertion_depth=insertion_depth,
            params=params,
            mats=mats,
            T_coolant_z=T_coolant_z,
            T_matrix_z=T_matrix_z,
            axial_coords=axial_coords,
            reactor_bottom=reactor_bottom,
            reactor_top=reactor_top,
            axial_section_height=axial_section_height,
            bundle_pitch=bundle_pitch,
            fuel_lattice_univs=fuel_lattice_univs,
            poison_lattice_univs=poison_lattice_univs,
            hex_prism_fuel=hex_prism_fuel,
            hex_prism_refl=hex_prism_refl,
            min_z=min_z,
            max_z=max_z,
            fuel_control_cyl_b4c=fuel_control_cyl_b4c,
            fuel_control_cyl_sheath_outer=fuel_control_cyl_sheath_outer,
            fuel_control_cyl_guide_outer=fuel_control_cyl_guide_outer,
            inf_graphite_universe=inf_graphite_universe,
            inf_graphite_refl_universe=inf_graphite_refl_universe,
            m_colors=m_colors,
        )

        # Store as r1/r2/r3, fc1/fc2/fc3, fcp1/fcp2/fcp3
        for assembly_type, univ in bank_assemblies.items():
            assemblies[f"{assembly_type}{bank_id}"] = univ

    return assemblies, m_colors, bundle_pitch

# ====================================================================================================
# CORE LATTICE BUILDER FUNCTION
# ====================================================================================================

def build_core_lattice(assemblies, core_rings, bundle_pitch):
    """
    Build the core lattice from ring definitions.

    Args:
        assemblies: Dictionary mapping assembly codes to Universe objects.
                    Bank-dependent types use suffixes: r1/r2/r3, fc1/fc2/fc3, fcp1/fcp2/fcp3.
        core_rings: List of rings, each ring is a list of assembly codes.
        bundle_pitch: Pitch between assemblies.

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