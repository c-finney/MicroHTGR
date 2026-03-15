import types
import openmc
import numpy as np

# ====================================================================================================
# TRISO LATTICE BUILDER FUNCTIONS
# ====================================================================================================

def generate_triso_positions(
    params: dict,
    axial_section_height: float
) -> tuple[list[tuple[float, float, float]], int, float, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """
    Generate TRISO sphere positions for one axial section.

    Call this once and pass the returned data to build_triso_lattice_for_material()
    to build multiple lattices with different fuel materials at the same positions.

    Args:
        params (dict): Dictionary of reactor parameters
        axial_section_height (float): Height of one axial zone in cm

    Returns:
        tuple: (safe_trisos, n_trisos, r_opyc, llc, pitch, triso_lattice_shape)
            - safe_trisos (list): List of (x, y, z) sphere centers for TRISOS within constraints
            - n_trisos (int): Number of accepted TRISO positions
            - r_opyc (float): Outer PyC radius in cm
            - llc (np.ndarray): Lower-left corner of the bounding box
            - pitch (np.ndarray): Lattice cell pitch
            - triso_lattice_shape (tuple): (nx, ny, nz) tuple for the search lattice
    """

    r_kernel = params["kernel_radius"]
    r_buffer = r_kernel + params["buffer_thickness"]
    r_ipyc   = r_buffer + params["ipyc_thickness"]
    r_sic    = r_ipyc   + params["sic_thickness"]
    r_opyc   = r_sic    + params["opyc_thickness"]

    fuel_cyl = openmc.ZCylinder(r=params["compact_radius"])

    zmin_local = -0.5 * axial_section_height
    zmax_local =  0.5 * axial_section_height
    min_z_local = openmc.ZPlane(z0=zmin_local)
    max_z_local = openmc.ZPlane(z0=zmax_local)

    triso_region = -fuel_cyl & +min_z_local & -max_z_local

    rand_spheres = openmc.model.pack_spheres(radius=r_opyc, region=triso_region, pf=params["triso_pf"])

    llc, urc = triso_region.bounding_box

    # Screening helper function to see if TRISO is within correct bounding box
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

    n_trisos = len(safe_trisos)
    V_triso = (4/3) * np.pi * r_opyc**3
    V_compact = np.pi * params["compact_radius"]**2 * axial_section_height
    actual_pf = n_trisos * V_triso / V_compact

    print(f"\nNumber of TRISOs created per axial zone: {len(rand_spheres)}")
    print(f"Number of safe TRISOs per axial zone: {n_trisos}")
    print(f"Requested TRISO PF: {params['triso_pf']:.3f}")
    print(f"Achieved TRISO PF: {actual_pf:.3f}")

    triso_lattice_shape = (4, 4, int(axial_section_height / 0.5))
    pitch = (urc - llc) / np.array(triso_lattice_shape)

    return safe_trisos, n_trisos, r_opyc, llc, pitch, triso_lattice_shape

def build_triso_lattice_for_material(
    fuel_material: openmc.Material,
    mats: types.ModuleType,
    params: dict,
    safe_trisos: list[tuple[float, float, float]],
    r_opyc: float,
    llc: np.ndarray,
    pitch: np.ndarray,
    triso_lattice_shape: tuple[int, int, int],
) -> openmc.RectLattice:
    """
    Build a TRISO search lattice using the given fuel material at pre-generated positions.

    Used to create multiple lattices with identical geometry but distinct fuel material
    instances (e.g., one per core ring per axial zone for spatial burnup tracking).

    Args:
        fuel_material (openmc.Material): Material to use for the fuel kernel
        mats (types.ModuleType): Reactor materials module (used for buffer, pyc, sic, and graphite)
        params (dict): Dictionary of reactor/simulation parameters
        safe_trisos (list): List of (x, y, z) sphere centers for TRISOS within constraints
        r_opyc (float): Outer PyC radius in cm
        llc (np.ndarray): Lower-left corner of the bounding box
        pitch (np.ndarray): Lattice cell pitch
        triso_lattice_shape (tuple): (nx, ny, nz) tuple for the search lattice

    Returns:
        openmc.model.create_triso_lattice result (an openmc.RectLattice)
    """

    r_kernel = params["kernel_radius"]
    r_buffer = r_kernel + params["buffer_thickness"]
    r_ipyc   = r_buffer + params["ipyc_thickness"]
    r_sic    = r_ipyc   + params["sic_thickness"]

    s_fuel   = openmc.Sphere(r=r_kernel)
    s_buffer = openmc.Sphere(r=r_buffer)
    s_ipyc   = openmc.Sphere(r=r_ipyc)
    s_sic    = openmc.Sphere(r=r_sic)
    s_opyc   = openmc.Sphere(r=r_opyc)

    c_triso_fuel   = openmc.Cell(name='c_triso_fuel',      fill=fuel_material, region=-s_fuel)
    c_triso_buffer = openmc.Cell(name='c_triso_c_buffer',  fill=mats.buffer,   region=+s_fuel & -s_buffer)
    c_triso_ipyc   = openmc.Cell(name='c_triso_pyc_inner', fill=mats.pyc,      region=+s_buffer & -s_ipyc)
    c_triso_sic    = openmc.Cell(name='c_triso_sic',       fill=mats.sic,      region=+s_ipyc & -s_sic)
    c_triso_opyc   = openmc.Cell(name='c_triso_pyc_outer', fill=mats.pyc,      region=+s_sic & -s_opyc)
    c_triso_matrix = openmc.Cell(name='c_triso_matrix',    fill=mats.graphite, region=+s_opyc)

    triso_universe = openmc.Universe(cells=[
        c_triso_fuel, c_triso_buffer, c_triso_ipyc,
        c_triso_sic, c_triso_opyc, c_triso_matrix
    ])

    random_trisos = [openmc.model.TRISO(r_opyc, triso_universe, center) for center in safe_trisos]

    triso_lattice = openmc.model.create_triso_lattice(
        random_trisos, llc, pitch, triso_lattice_shape, mats.graphite
    )

    return triso_lattice

def build_homogenized_compact_fill(
    fuel_material: openmc.Material,
    params: dict,
    mats: types.ModuleType,
) -> openmc.Universe:
    """
    Return a two-region annular Universe implementing the Reactivity-Equivalent
    Physical Transform (RPT) method for homogenization.

    This is a drop-in replacement for the TRISO search lattice returned by
    build_triso_lattice_for_material(). The Universe contains:
        - Inner cylinder (r < r_rpt):              fuel_material (RPT inner material)
        - Outer annulus  (r_rpt < r < r_compact):  pure graphite
    where r_rpt = params["rpt_radius"] is the calibrated RPT radius (cm) determined empirically.

    The RPT inner material (from materials.make_rpt_inner_material) contains all
    TRISO layers + proportional graphite at effective pf_inner = pf*(r_compact/r_rpt)^2.
    As r_rpt is reduced toward r_compact*sqrt(pf), the concentration increases
    and self-shielding improves.  The calibrated r_rpt is the value at which
    k_eff(RPT) = k_eff(explicit TRISO reference).

    The outer annulus cell uses region=+inner_cyl (unbounded outward).  The
    parent fuel channel cell in assembly.py clips it at r_compact via its own
    -fuel_cyl region, so no coincident surface is introduced here.

    Args:
        fuel_material (openmc.Material): RPT inner material from materials.make_rpt_inner_material()
        params (dict): Dictionary of reactor/simulation parameters
        mats (types.ModuleType): Reactor materials module (used for graphite annulus cell)

    Returns:
        RPT_universe (openmc.Universe): Universe containing the two-region RPT fuel compact
    """
    r_rpt = params["rpt_radius"]

    inner_cyl    = openmc.ZCylinder(r=r_rpt)
    inner_cell   = openmc.Cell(fill=fuel_material, region=-inner_cyl)
    annulus_cell = openmc.Cell(fill=mats.graphite,  region=+inner_cyl)

    RPT_universe = openmc.Universe(cells=[inner_cell, annulus_cell])

    return RPT_universe

def create_triso_lattice(
    params: dict,
    mats: types.ModuleType,
    axial_section_height: float,
) -> tuple[openmc.RectLattice, int]:
    """
    Create a TRISO particle lattice for fuel compacts using mats.fuel.

    Convenience wrapper around generate_triso_positions() and
    build_triso_lattice_for_material(). For spatial burnup tracking with
    per-region fuel material clones, call those two functions directly instead.

    Args:
        params (dict): Dictionary of reactor/simulation parameters
        mats (types.ModuleType): Materials module containing fuel, buffer, pyc, sic, graphite
        axial_section_height (float): Height of one axial zone in cm

    Returns:
        tuple: (triso_lattice, n_trisos)
            - triso_lattice (openmc.RectLattice): OpenMC lattice containing TRISO particles
            - n_trisos (int): Number of TRISO particles per axial zone
    """

    safe_trisos, n_trisos, r_opyc, llc, pitch, triso_lattice_shape = generate_triso_positions(
        params, axial_section_height
    )

    triso_lattice = build_triso_lattice_for_material(
        mats.fuel, mats, params, safe_trisos, r_opyc, llc, pitch, triso_lattice_shape
    )

    return triso_lattice, n_trisos