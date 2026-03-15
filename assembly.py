import math
import types
import openmc
import numpy as np

# ====================================================================================================
# CONTROL ROD ASSEMBLY UNIVERSES BUILDER FUNCTION
# ====================================================================================================

def build_bank_assemblies(
    bank_id: int,
    control_insertion_depth: float,
    params: dict,
    mats: types.ModuleType,
    T_coolant_z: list[float],
    T_matrix_z: list[float],
    axial_coords: list[float],
    reactor_bottom: float,
    reactor_top: float,
    axial_section_height: float,
    bundle_pitch: float,
    fuel_lattice_univs: list,
    poison_lattice_univs: list,
    hex_prism_fuel: openmc.Region,
    hex_prism_refl: openmc.Region,
    min_z: openmc.ZPlane,
    max_z: openmc.ZPlane,
    fuel_control_cyl_b4c: openmc.ZCylinder,
    fuel_control_cyl_sheath_outer: openmc.ZCylinder,
    fuel_control_cyl_guide_outer: openmc.ZCylinder,
    inf_graphite_universe: openmc.Universe,
    inf_graphite_refl_universe: openmc.Universe,
    m_colors: dict,
) -> dict[str, openmc.Universe]:
    """
    Build the four bank-dependent assembly types (r, ra, fc, fcp) for a single
    control rod bank insertion depth.

    Control rod geometry is defined by a continuous ZPlane at the exact insertion
    depth rather than snapping to axial zone midpoints.  This allows fractional
    insertions like 0.81 and 0.83 to be geometrically distinct even when they
    fall within the same axial zone.  Per-zone temperatures are still assigned
    from T_matrix_z and T_coolant_z so the axial temperature profile is preserved.

    For each axial zone there are three cases:
      - Zone fully below control_bottom  → entirely B4C inserted
      - Zone fully above control_bottom  → entirely helium (withdrawn)
      - Zone straddles control_bottom    → split into two sub-regions at the exact plane

    Args:
        bank_id (int): Control bank identifier (1, 2, or 3)
        control_insertion_depth (float): Distance inserted from reactor top in cm
        params (dict): Dictionary of reactor/simulation parameters
        mats (types.ModuleType): Reactor materials module
        T_coolant_z (list[float]): Coolant temperature per axial zone in K
        T_matrix_z (list[float]): Graphite matrix temperature per axial zone in K
        axial_coords (list[float]): Z-coordinates of axial zone boundaries in cm
        reactor_bottom (float): Bottom of the active core in cm
        reactor_top (float): Top of the active core in cm
        axial_section_height (float): Height of one axial zone in cm
        bundle_pitch (float): Center-to-center distance between assemblies in cm
        fuel_lattice_univs (list): Per-zone hex lattice universe lists for fuel assemblies
        poison_lattice_univs (list): Per-zone hex lattice universe lists for fuel+poison assemblies
        hex_prism_fuel (openmc.Region): Hexagonal prism region for fuel assemblies
        hex_prism_refl (openmc.Region): Hexagonal prism region for reflector assemblies
        min_z (openmc.ZPlane): Bottom bounding plane of the core
        max_z (openmc.ZPlane): Top bounding plane of the core
        fuel_control_cyl_b4c (openmc.ZCylinder): Inner B4C cylinder surface for fuel assembly control rod
        fuel_control_cyl_sheath_outer (openmc.ZCylinder): Outer sheath surface for fuel assembly control rod
        fuel_control_cyl_guide_outer (openmc.ZCylinder): Outer guide tube surface for fuel assembly control rod
        inf_graphite_universe (openmc.Universe): Infinite graphite universe for fuel assembly outer fill
        inf_graphite_refl_universe (openmc.Universe): Infinite graphite universe for reflector assembly outer fill
        m_colors (dict): Material color dictionary updated in-place for plotting

    Returns:
        dict[str, openmc.Universe]: Assembly type codes mapped to their Universe objects.
            Keys: "r" (reflector with central control rod), "ra" (reflector with 6 control rods),
                  "fc" (fuel with central control rod), "fcp" (fuel with control and poison rods)
    """

    r_b4c = params["control_radius"] - params["sheath_thickness"]
    r_sheath_outer = params["control_radius"]
    r_guide_outer = r_sheath_outer + params["guide_tube_thickness"]

    # Exact insertion plane in continuous geometry — independent of axial mesh
    control_bottom = reactor_top - control_insertion_depth
    control_bottom_plane = openmc.ZPlane(z0=control_bottom)

    # ==================================================================
    # REFLECTOR ASSEMBLY WITH CONTROL ROD (r)
    # ==================================================================

    control_rod_univs = []

    for idx, (z_min, z_max) in enumerate(zip(axial_coords[:-1], axial_coords[1:])):
        T_matrix = T_matrix_z[idx]
        T_coolant = T_coolant_z[idx]

        # Fresh cylinders per zone (each universe is self-contained)
        control_cyl_b4c = openmc.ZCylinder(r=r_b4c)
        control_cyl_sheath_outer = openmc.ZCylinder(r=r_sheath_outer)
        control_cyl_guide_outer = openmc.ZCylinder(r=r_guide_outer)

        min_z_plane = openmc.ZPlane(z0=z_min)
        max_z_plane = openmc.ZPlane(z0=z_max)
        axial_region = +min_z_plane & -max_z_plane

        zone_fully_inserted  = control_bottom <= z_min   # rod tip at or below zone bottom
        zone_fully_withdrawn = control_bottom >= z_max   # rod tip at or above zone top

        if zone_fully_inserted:
            b4c_cell = openmc.Cell(fill=mats.b4c_control, region=-control_cyl_b4c & axial_region)
            b4c_cell.temperature = T_matrix

            sheath_cell = openmc.Cell(
                fill=mats.incoloy800H,
                region=+control_cyl_b4c & -control_cyl_sheath_outer & axial_region)
            sheath_cell.temperature = T_matrix

            guide_tube_cell = openmc.Cell(
                fill=mats.incoloy800H,
                region=+control_cyl_sheath_outer & -control_cyl_guide_outer & axial_region)
            guide_tube_cell.temperature = T_matrix

            matrix_cell = openmc.Cell(fill=mats.graphite, region=+control_cyl_guide_outer & axial_region)
            matrix_cell.temperature = T_matrix

            control_univ = openmc.Universe(
                cells=[b4c_cell, sheath_cell, guide_tube_cell, matrix_cell])

        elif zone_fully_withdrawn:
            guide_helium_cell = openmc.Cell(fill=mats.helium, region=-control_cyl_sheath_outer & axial_region)
            guide_helium_cell.temperature = T_coolant
            m_colors[mats.helium] = 'red'

            guide_tube_cell = openmc.Cell(
                fill=mats.incoloy800H,
                region=+control_cyl_sheath_outer & -control_cyl_guide_outer & axial_region)
            guide_tube_cell.temperature = T_matrix

            matrix_cell = openmc.Cell(fill=mats.graphite, region=+control_cyl_guide_outer & axial_region)
            matrix_cell.temperature = T_matrix

            control_univ = openmc.Universe(
                cells=[guide_helium_cell, guide_tube_cell, matrix_cell])

        else:
            # Rod tip falls inside this zone — split at the exact plane
            # Rod body occupies above control_bottom (rod inserts from the top)
            rod_region = +control_bottom_plane & -max_z_plane
            # Gap (no rod) occupies below control_bottom
            gap_region = +min_z_plane & -control_bottom_plane

            b4c_cell = openmc.Cell(fill=mats.b4c_control, region=-control_cyl_b4c & rod_region)
            b4c_cell.temperature = T_matrix

            sheath_ins = openmc.Cell(
                fill=mats.incoloy800H,
                region=+control_cyl_b4c & -control_cyl_sheath_outer & rod_region)
            sheath_ins.temperature = T_matrix

            guide_ins = openmc.Cell(
                fill=mats.incoloy800H,
                region=+control_cyl_sheath_outer & -control_cyl_guide_outer & rod_region)
            guide_ins.temperature = T_matrix

            matrix_ins = openmc.Cell(fill=mats.graphite, region=+control_cyl_guide_outer & rod_region)
            matrix_ins.temperature = T_matrix

            he_cell = openmc.Cell(fill=mats.helium, region=-control_cyl_sheath_outer & gap_region)
            he_cell.temperature = T_coolant
            m_colors[mats.helium] = 'red'

            guide_wit = openmc.Cell(
                fill=mats.incoloy800H,
                region=+control_cyl_sheath_outer & -control_cyl_guide_outer & gap_region)
            guide_wit.temperature = T_matrix

            matrix_wit = openmc.Cell(fill=mats.graphite, region=+control_cyl_guide_outer & gap_region)
            matrix_wit.temperature = T_matrix

            control_univ = openmc.Universe(
                cells=[b4c_cell, sheath_ins, guide_ins, matrix_ins,
                       he_cell, guide_wit, matrix_wit])

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

    # Copies outer rings from source_lattice_univs and replaces the two inner rings with graphite for control rod clearance
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

    # Builds per-axial-zone control rod cells for a fuel assembly, splitting geometry at the exact control_bottom_plane.
    def make_fuel_control_cells(hex_region_with_outer_cyl):
        cells = []
        for idx, (z_min, z_max) in enumerate(zip(axial_coords[:-1], axial_coords[1:])):
            T_matrix = T_matrix_z[idx]
            T_coolant = T_coolant_z[idx]

            min_z_plane = openmc.ZPlane(z0=z_min)
            max_z_plane = openmc.ZPlane(z0=z_max)
            axial_region = +min_z_plane & -max_z_plane & hex_region_with_outer_cyl

            zone_fully_inserted  = control_bottom <= z_min
            zone_fully_withdrawn = control_bottom >= z_max

            if zone_fully_inserted:
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

            elif zone_fully_withdrawn:
                he_cell = openmc.Cell(
                    fill=mats.helium,
                    region=-fuel_control_cyl_sheath_outer & axial_region)
                he_cell.temperature = T_coolant
                m_colors[mats.helium] = 'red'

                guide = openmc.Cell(
                    fill=mats.incoloy800H,
                    region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & axial_region)
                guide.temperature = T_matrix

                cells.extend([he_cell, guide])

            else:
                # Rod tip inside this zone — split at the exact plane
                # Rod body occupies above control_bottom (rod inserts from the top)
                rod_region = +control_bottom_plane & -max_z_plane & hex_region_with_outer_cyl
                # Gap (no rod) occupies below control_bottom
                gap_region = +min_z_plane & -control_bottom_plane & hex_region_with_outer_cyl

                b4c = openmc.Cell(
                    fill=mats.b4c_control,
                    region=-fuel_control_cyl_b4c & rod_region)
                b4c.temperature = T_matrix

                sheath_ins = openmc.Cell(
                    fill=mats.incoloy800H,
                    region=+fuel_control_cyl_b4c & -fuel_control_cyl_sheath_outer & rod_region)
                sheath_ins.temperature = T_matrix

                guide_ins = openmc.Cell(
                    fill=mats.incoloy800H,
                    region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & rod_region)
                guide_ins.temperature = T_matrix

                he_cell = openmc.Cell(
                    fill=mats.helium,
                    region=-fuel_control_cyl_sheath_outer & gap_region)
                he_cell.temperature = T_coolant
                m_colors[mats.helium] = 'red'

                guide_wit = openmc.Cell(
                    fill=mats.incoloy800H,
                    region=+fuel_control_cyl_sheath_outer & -fuel_control_cyl_guide_outer & gap_region)
                guide_wit.temperature = T_matrix

                cells.extend([b4c, sheath_ins, guide_ins, he_cell, guide_wit])

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

# ====================================================================================================
# SECONDARY SHUTDOWN SYSTEM ASSEMBLY UNIVERSES BUILDER FUNCTION
# ====================================================================================================

def build_ss_assemblies(
    ss_inserted: bool,
    params: dict,
    mats: types.ModuleType,
    T_coolant_z: list[float],
    T_matrix_z: list[float],
    axial_coords: list[float],
    reactor_bottom: float,
    reactor_top: float,
    axial_section_height: float,
    bundle_pitch: float,
    fuel_lattice_univs: list,
    poison_lattice_univs: list,
    hex_prism_fuel: openmc.Region,
    hex_prism_refl: openmc.Region,
    min_z: openmc.ZPlane,
    max_z: openmc.ZPlane,
    inf_graphite_universe: openmc.Universe,
    inf_graphite_refl_universe: openmc.Universe,
    m_colors: dict,
) -> dict[str, openmc.Universe]:
    """
    Build the four secondary shutdown rod assembly types: rss, rssa, fss, fssp.

    Unlike control rod assemblies there is no bank insertion logic — SS rods are
    either fully inserted (ss_inserted=True) or fully removed (ss_inserted=False).

    The SS rod material is a 55% B4C / 45% He homogeneous mixture (mats.b4c_ss).
    There is no Incoloy sheath or guide tube; the bored hole has radius
    control_radius + guide_tube_thickness, matching the control rod guide-outer
    diameter so graphite block hole dimensions are identical.

    Args:
        ss_inserted (bool): If True, rods are fully inserted; if False, fully withdrawn
        params (dict): Dictionary of reactor/simulation parameters
        mats (types.ModuleType): Reactor materials module
        T_coolant_z (list[float]): Coolant temperature per axial zone in K
        T_matrix_z (list[float]): Graphite matrix temperature per axial zone in K
        axial_coords (list[float]): Z-coordinates of axial zone boundaries in cm
        reactor_bottom (float): Bottom of the active core in cm
        reactor_top (float): Top of the active core in cm
        axial_section_height (float): Height of one axial zone in cm
        bundle_pitch (float): Center-to-center distance between assemblies in cm
        fuel_lattice_univs (list): Per-zone hex lattice universe lists for fuel assemblies
        poison_lattice_univs (list): Per-zone hex lattice universe lists for fuel+poison assemblies
        hex_prism_fuel (openmc.Region): Hexagonal prism region for fuel assemblies
        hex_prism_refl (openmc.Region): Hexagonal prism region for reflector assemblies
        min_z (openmc.ZPlane): Bottom bounding plane of the core
        max_z (openmc.ZPlane): Top bounding plane of the core
        inf_graphite_universe (openmc.Universe): Infinite graphite universe for fuel assembly outer fill
        inf_graphite_refl_universe (openmc.Universe): Infinite graphite universe for reflector assembly outer fill
        m_colors (dict): Material color dictionary updated in-place for plotting

    Returns:
        dict[str, openmc.Universe]: Assembly type codes mapped to their Universe objects.
            Keys: "rss" (reflector with central SS rod), "rssa" (reflector with 6 SS rods),
                  "fss" (fuel with central SS rod), "fssp" (fuel with SS rod and edge poison rods)
    """

    r_ss_refl = params["control_radius"] + params["guide_tube_thickness"]
    r_ss_fuel = params["fuel_assembly_control_radius"] + params["guide_tube_thickness"]

    ss_cyl_refl = openmc.ZCylinder(r=r_ss_refl)
    ss_cyl_fuel = openmc.ZCylinder(r=r_ss_fuel)

    # ==================================================================
    # REFLECTOR ASSEMBLY WITH SS ROD (rss)
    # ==================================================================

    ss_rod_univs = []
    for idx in range(len(axial_coords) - 1):
        T_matrix  = T_matrix_z[idx]
        T_coolant = T_coolant_z[idx]

        if ss_inserted:
            ss_cell = openmc.Cell(fill=mats.b4c_ss, region=-ss_cyl_refl)
            ss_cell.temperature = T_matrix
        else:
            ss_cell = openmc.Cell(fill=mats.helium, region=-ss_cyl_refl)
            ss_cell.temperature = T_coolant
            m_colors[mats.helium] = 'red'

        graphite_cell = openmc.Cell(fill=mats.graphite, region=+ss_cyl_refl)
        graphite_cell.temperature = T_matrix

        ss_rod_univs.append(openmc.Universe(cells=[ss_cell, graphite_cell]))

    rss_lat = openmc.HexLattice(name="Reflector Lattice with SS Rod")
    rss_lat.orientation = 'x'
    rss_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    rss_lat.pitch = (bundle_pitch, axial_section_height)
    rss_lat.universes = [[[u]] for u in ss_rod_univs]
    rss_lat.outer = inf_graphite_refl_universe

    rss_univ = openmc.Universe(cells=[
        openmc.Cell(fill=rss_lat, region=hex_prism_refl & +min_z & -max_z)])

    # ==================================================================
    # ALT REFLECTOR ASSEMBLY WITH 6 SS RODS IN HEX RING (rssa)
    # ==================================================================

    graphite_center_cell = openmc.Cell(fill=mats.graphite)
    graphite_center_cell.temperature = params["reflector_min"]
    graphite_center_univ = openmc.Universe(cells=[graphite_center_cell])

    rssa_lattice_univs = []
    for idx in range(len(axial_coords) - 1):
        ring1 = [graphite_center_univ, graphite_center_univ, graphite_center_univ,
                 ss_rod_univs[idx], ss_rod_univs[idx], ss_rod_univs[idx]]
        rssa_lattice_univs.append([ring1, [graphite_center_univ]])

    rssa_lat = openmc.HexLattice(name="Reflector Alt Lattice with 6 SS Rods")
    rssa_lat.orientation = 'y'
    rssa_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    rssa_lat.pitch = (bundle_pitch / 3.0, axial_section_height)
    rssa_lat.universes = rssa_lattice_univs
    rssa_lat.outer = inf_graphite_refl_universe

    rssa_univ = openmc.Universe(cells=[
        openmc.Cell(fill=rssa_lat, region=hex_prism_refl & +min_z & -max_z)])

    # Copies outer rings from source_lattice_univs and replaces the two inner rings with graphite for SS rod clearance.
    def make_ss_fuel_lattice_univs(source_lattice_univs):
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

    # Builds per-axial-zone SS rod cells for a fuel assembly, filling with b4c_ss or helium based on ss_inserted.
    def make_fuel_ss_cells(hex_region):
        cells = []
        for idx, (z_min, z_max) in enumerate(zip(axial_coords[:-1], axial_coords[1:])):
            T_matrix  = T_matrix_z[idx]
            T_coolant = T_coolant_z[idx]
            axial_region = (+openmc.ZPlane(z0=z_min) & -openmc.ZPlane(z0=z_max)
                            & hex_region)
            if ss_inserted:
                ss_cell = openmc.Cell(fill=mats.b4c_ss,
                                      region=-ss_cyl_fuel & axial_region)
                ss_cell.temperature = T_matrix
                cells.append(ss_cell)
            else:
                ss_he_cell = openmc.Cell(fill=mats.helium, region=-ss_cyl_fuel & axial_region)
                ss_he_cell.temperature = T_coolant
                m_colors[mats.helium] = 'red'
                cells.append(ss_he_cell)
        return cells

    # ==================================================================
    # FUEL ASSEMBLY WITH CENTRAL SS ROD (fss)
    # ==================================================================

    hex_inner_ss = +ss_cyl_fuel & hex_prism_fuel & +min_z & -max_z

    fss_lat = openmc.HexLattice(name="Fuel Lattice for SS Assembly")
    fss_lat.orientation = 'x'
    fss_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fss_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fss_lat.universes = make_ss_fuel_lattice_univs(fuel_lattice_univs)
    fss_lat.outer = inf_graphite_universe

    fss_univ = openmc.Universe(
        cells=make_fuel_ss_cells(hex_prism_fuel)
              + [openmc.Cell(fill=fss_lat, region=hex_inner_ss)])

    # ==================================================================
    # FUEL ASSEMBLY WITH SS ROD AND EDGE POISON RODS (fssp)
    # ==================================================================

    fssp_lat = openmc.HexLattice(name="Fuel Lattice with Poison for SS Assembly")
    fssp_lat.orientation = 'x'
    fssp_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
    fssp_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
    fssp_lat.universes = make_ss_fuel_lattice_univs(poison_lattice_univs)
    fssp_lat.outer = inf_graphite_universe

    fssp_univ = openmc.Universe(
        cells=make_fuel_ss_cells(hex_prism_fuel)
              + [openmc.Cell(fill=fssp_lat, region=hex_inner_ss)])

    return {
        "rss":  rss_univ,
        "rssa": rssa_univ,
        "fss":  fss_univ,
        "fssp": fssp_univ,
    }

# ====================================================================================================
# ALL ASSEMBLY UNIVERSES BUILDER FUNCTION
# ====================================================================================================

def create_assembly_univs(
    params: dict,
    mats: types.ModuleType,
    T_coolant_z: list[float],
    T_compact_z: list[float],
    T_matrix_z: list[float],
    T_reflector_z: list[float],
    ring_triso_lattices: dict,
    axial_coords: list[float],
    reactor_bottom: float,
    reactor_top: float,
) -> tuple[dict[str, openmc.Universe], dict, float]:
    """
    Create all fuel assembly variants and reflector assemblies.

    Fuel-containing assembly types are built once per core ring so that each
    ring uses its own fuel material clones (enabling spatial burnup tracking).
    The ring-specific variants are keyed as "{type}_ring{ring_idx}" (e.g.
    "f_ring0", "fc1_ring2").  Reflector-only types (rr, r1/2/3, ra1/2/3,
    rss, rssa) are ring-independent and stored without a ring suffix.

    The ring_triso_lattices dict holds the fill object for each fuel compact
    cell.  Each value may be either:
      - an openmc.RectLattice  (explicit TRISO, use_homogenized_fuel=False)
      - an openmc.Universe     (single homogenized cell, use_homogenized_fuel=True)

    assembly.py treats both identically — the fill object is placed directly
    into the fuel channel cell region, so no branching on fuel type is needed
    here.  The distinction is entirely handled upstream in simulation.py and
    trisos.py.

    Args:
        params (dict): Dictionary of reactor/simulation parameters
        mats (types.ModuleType): Reactor materials module
        T_coolant_z (list[float]): Coolant temperature per axial zone in K
        T_compact_z (list[float]): Fuel compact temperature per axial zone in K
        T_matrix_z (list[float]): Graphite matrix temperature per axial zone in K
        T_reflector_z (list[float]): Reflector temperature per axial zone in K
        ring_triso_lattices (dict): Nested dict {ring_idx: {ax_idx: fill_object}} where
            fill_object is either an openmc.RectLattice (explicit TRISO) or openmc.Universe
            (homogenized RPT) — one fill object per (core ring, axial zone)
        axial_coords (list[float]): Z-coordinates of axial zone boundaries in cm
        reactor_bottom (float): Bottom of the active core in cm
        reactor_top (float): Top of the active core in cm

    Returns:
        tuple: (assemblies, m_colors, bundle_pitch)
            - assemblies (dict[str, openmc.Universe]): Assembly type codes mapped to Universe objects
            - m_colors (dict): Material color dictionary for plotting
            - bundle_pitch (float): Center-to-center distance between assemblies in cm
    """

    m_colors = {}

    n_rings = len(params["core_rings"])
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

    r_b4c_fuel = params["fuel_assembly_control_radius"] - params["sheath_thickness"]
    r_sheath_outer_fuel = params["fuel_assembly_control_radius"]
    r_guide_outer_fuel = r_sheath_outer_fuel + params["guide_tube_thickness"]

    fuel_control_cyl_b4c = openmc.ZCylinder(r=r_b4c_fuel)
    fuel_control_cyl_sheath_outer = openmc.ZCylinder(r=r_sheath_outer_fuel)
    fuel_control_cyl_guide_outer = openmc.ZCylinder(r=r_guide_outer_fuel)

    # ==================================================================
    # BANK-INDEPENDENT OUTER UNIVERSES
    # ==================================================================

    graphite_outer_cell = openmc.Cell(fill=mats.graphite)
    inf_graphite_universe = openmc.Universe(cells=[graphite_outer_cell])

    graphite_outer_refl = openmc.Cell(fill=mats.graphite)
    graphite_outer_refl.temperature = params["reflector_min"]
    inf_graphite_refl_universe = openmc.Universe(cells=[graphite_outer_refl])

    bank_keys = [
        ("bank_1_insertion", 1),
        ("bank_2_insertion", 2),
        ("bank_3_insertion", 3),
    ]

    assemblies = {}

    # ==================================================================
    # PER-RING FUEL LATTICE UNIVERSES AND FUEL ASSEMBLY VARIANTS
    # ==================================================================

    for ring_idx in range(n_rings):

        fuel_lattice_univs = []
        poison_lattice_univs = []
        poison_lattice_univs_alt = []

        for idx, (z_min, z_max) in enumerate(zip(axial_coords[:-1], axial_coords[1:])):
            T_coolant = T_coolant_z[idx]
            T_compact = T_compact_z[idx]
            T_matrix = T_matrix_z[idx]

            # Retrieve the fill object for this ring and axial zone.
            # This is either a TRISO RectLattice (explicit) or a Universe wrapping a homogenized compact cell
            # Both are valid fills for the fuel channel cell below
            this_fuel_fill = ring_triso_lattices[ring_idx][idx]

            # Fuel Cannel
            fuel_ch_cell = openmc.Cell(region=-fuel_cyl, fill=this_fuel_fill)
            fuel_ch_cell.temperature = T_compact
            fuel_ch_matrix_cell = openmc.Cell(region=+fuel_cyl, fill=mats.graphite)
            fuel_ch_matrix_cell.temperature = T_matrix

            # Poison Channel
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
            coolant_cell = openmc.Cell(region=-coolant_cyl, fill=mats.helium)
            coolant_cell.temperature = T_coolant
            m_colors[mats.helium] = 'red'

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

        # ---- Standard fuel assembly (f_ring{ring_idx}) ----
        fuel_assembly_lat = openmc.HexLattice(name=f"Fuel Lattice Ring {ring_idx}")
        fuel_assembly_lat.orientation = 'x'
        fuel_assembly_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
        fuel_assembly_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
        fuel_assembly_lat.universes = fuel_lattice_univs
        fuel_assembly_lat.outer = inf_graphite_universe

        fuel_assembly_cell = openmc.Cell(
            fill=fuel_assembly_lat, region=hex_prism_fuel & +min_z & -max_z)
        assemblies[f"f_ring{ring_idx}"] = openmc.Universe(cells=[fuel_assembly_cell])

        # ---- Fuel assembly with 6 edge poison rods (fp_ring{ring_idx}) ----
        fuel_assembly_poison_lat = openmc.HexLattice(name=f"Fuel Lattice with Poison Ring {ring_idx}")
        fuel_assembly_poison_lat.orientation = 'x'
        fuel_assembly_poison_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
        fuel_assembly_poison_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
        fuel_assembly_poison_lat.universes = poison_lattice_univs
        fuel_assembly_poison_lat.outer = inf_graphite_universe

        fuel_assembly_poison_cell = openmc.Cell(
            fill=fuel_assembly_poison_lat, region=hex_prism_fuel & +min_z & -max_z)
        assemblies[f"fp_ring{ring_idx}"] = openmc.Universe(cells=[fuel_assembly_poison_cell])

        # ---- Fuel assembly with central poison rod (fpa_ring{ring_idx}) ----
        fuel_assembly_poison_lat_alt = openmc.HexLattice(name=f"Fuel Lattice with Poison Alt Ring {ring_idx}")
        fuel_assembly_poison_lat_alt.orientation = 'x'
        fuel_assembly_poison_lat_alt.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
        fuel_assembly_poison_lat_alt.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
        fuel_assembly_poison_lat_alt.universes = poison_lattice_univs_alt
        fuel_assembly_poison_lat_alt.outer = inf_graphite_universe

        fuel_assembly_poison_cell_alt = openmc.Cell(
            fill=fuel_assembly_poison_lat_alt, region=hex_prism_fuel & +min_z & -max_z)
        assemblies[f"fpa_ring{ring_idx}"] = openmc.Universe(cells=[fuel_assembly_poison_cell_alt])

        # ---- Bank-dependent fuel assemblies (fc, fcp) per ring ----
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

            for assembly_type, univ in bank_assemblies.items():
                if assembly_type in ('fc', 'fcp'):
                    assemblies[f"{assembly_type}{bank_id}_ring{ring_idx}"] = univ
                else:
                    assemblies[f"{assembly_type}{bank_id}"] = univ

        # ---- Secondary shutdown rod fuel assemblies (fss, fssp) per ring ----
        ss_assemblies = build_ss_assemblies(
            ss_inserted=params.get("secondary_SD_rods_inserted", False),
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
            inf_graphite_universe=inf_graphite_universe,
            inf_graphite_refl_universe=inf_graphite_refl_universe,
            m_colors=m_colors,
        )

        for atype, univ in ss_assemblies.items():
            if atype in ('fss', 'fssp'):
                assemblies[f"{atype}_ring{ring_idx}"] = univ
            else:
                assemblies[atype] = univ

    # ==================================================================
    # PURE GRAPHITE REFLECTOR BLOCK (rr) — ring-independent
    # ==================================================================

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
    assemblies["rr"] = openmc.Universe(cells=[rr_assembly_cell])

    return assemblies, m_colors, bundle_pitch

# ====================================================================================================
# CORE LATTICE BUILDER FUNCTION
# ====================================================================================================

def build_core_lattice(
    assemblies: dict[str, openmc.Universe],
    core_rings: list[list[str]],
    bundle_pitch: float,
) -> openmc.HexLattice:
    """
    Build the core hex lattice from ring definitions.

    Args:
        assemblies (dict[str, openmc.Universe]): Assembly type codes mapped to Universe objects.
            Bank-dependent types use suffixes: r1/r2/r3, fc1/fc2/fc3, fcp1/fcp2/fcp3.
        core_rings (list[list[str]]): List of rings from outermost to innermost, each ring
            is a list of assembly type code strings
        bundle_pitch (float): Center-to-center distance between assemblies in cm

    Returns:
        openmc.HexLattice: The assembled core hex lattice
    """

    core_lattice_univs = []

    for ring_idx, ring_def in enumerate(core_rings):
        ring = []
        for code in ring_def:
            ring_key = f"{code}_ring{ring_idx}"
            if ring_key in assemblies:
                ring.append(assemblies[ring_key])
            elif code in assemblies:
                ring.append(assemblies[code])
            else:
                raise KeyError(
                    f"Assembly type '{code}' not found in assemblies dict "
                    f"(tried '{ring_key}' and '{code}')"
                )
        core_lattice_univs.append(ring)

    core_lattice = openmc.HexLattice(name="Core Lattice")
    core_lattice.center = (0.0, 0.0)
    core_lattice.pitch = (bundle_pitch,)
    core_lattice.universes = core_lattice_univs

    return core_lattice