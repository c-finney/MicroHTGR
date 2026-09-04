"""
B4C Sphere Lattice Construction for the Reserve Shutdown System
===============================================================

Builds the explicit packed-sphere model of the reserve (secondary) shutdown
system, in which B4C spheres fall by gravity into dedicated core channels.

Mirrors the structure of ``trisos.py``: positions are generated once for a target
packing fraction and channel radius, then filled into an OpenMC lattice. Modelling
the spheres explicitly rather than as a homogenised B4C/He mixture captures the
self-shielding of the absorber, which is significant at the packing fractions used
here. Set ``use_homogenized_SS_rods`` in ``config.py`` to fall back to the
homogenised treatment for faster scoping runs.
"""

import types
import openmc
import numpy as np

# ====================================================================================================
# B4C SPHERE LATTICE BUILDER FUNCTIONS
# ====================================================================================================

def generate_b4c_sphere_positions(
    params: dict,
    axial_section_height: float,
    r_ss: float
) -> tuple[list[tuple[float, float, float]], int, float, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """
    Generate B4C sphere positions for one axial section of a secondary shutdown rod.

    Call this once and pass the returned data to build_b4c_sphere_lattice() to build
    the sphere lattice.

    Args:
        params (dict): Dictionary of reactor parameters
        axial_section_height (float): Height of one axial zone in cm
        r_ss (float): Radius of the SS rod cylinder in cm

    Returns:
        tuple: (safe_spheres, n_spheres, r_b4c, llc, pitch, lattice_shape)
            - safe_spheres (list): List of (x, y, z) sphere centers within constraints
            - n_spheres (int): Number of accepted sphere positions
            - r_b4c (float): B4C sphere radius in cm
            - llc (np.ndarray): Lower-left corner of the bounding box
            - pitch (np.ndarray): Lattice cell pitch
            - lattice_shape (tuple): (nx, ny, nz) tuple for the search lattice
    """

    r_b4c = params["b4c_ss_sphere_radius"]
    pf    = params["b4c_ss_pf"]

    ss_cyl      = openmc.ZCylinder(r=r_ss)
    zmin_local  = -0.5 * axial_section_height
    zmax_local  =  0.5 * axial_section_height
    min_z_local = openmc.ZPlane(z0=zmin_local)
    max_z_local = openmc.ZPlane(z0=zmax_local)

    sphere_region = -ss_cyl & +min_z_local & -max_z_local

    rand_spheres = openmc.model.pack_spheres(radius=r_b4c, region=sphere_region, pf=pf,
                                             initial_pf=0.1)

    llc, urc = sphere_region.bounding_box

    # pack_spheres clips sphere centers to the container boundary via repel_spheres.
    # Floating point in that clipping can leave centers with x²+y² = (r_ss-r_b4c)² + ε.
    # Relax the checks by eps so those boundary spheres are accepted, not discarded.
    eps = 1e-9
    def valid_sphere(c):
        x, y, z = c
        return (
            x*x + y*y <= (r_ss - r_b4c)**2 + eps and
            zmin_local + r_b4c - eps <= z <= zmax_local - r_b4c + eps and
            llc[0] + r_b4c - eps <= x <= urc[0] - r_b4c + eps and
            llc[1] + r_b4c - eps <= y <= urc[1] - r_b4c + eps and
            llc[2] + r_b4c - eps <= z <= urc[2] - r_b4c + eps
        )

    safe_spheres = [c for c in rand_spheres if valid_sphere(c)]
    n_spheres    = len(safe_spheres)

    V_sphere  = (4/3) * np.pi * r_b4c**3
    V_rod     = np.pi * r_ss**2 * axial_section_height
    actual_pf = n_spheres * V_sphere / V_rod

    print(f"\nNumber of B4C spheres created per axial zone: {len(rand_spheres)}")
    print(f"Number of safe B4C spheres per axial zone: {n_spheres}")
    print(f"Requested B4C sphere PF: {pf:.3f}")
    print(f"Achieved B4C sphere PF: {actual_pf:.3f}")

    n_z           = max(1, int(axial_section_height / (2 * r_b4c)))
    lattice_shape = (4, 4, n_z)
    # Expand the lattice bounds by eps so sphere faces sitting exactly on the
    # bounding-box edge don't trigger the "TRISO outside lattice" warning.
    lattice_llc = llc - eps
    pitch       = (urc + eps - lattice_llc) / np.array(lattice_shape)

    return safe_spheres, n_spheres, r_b4c, lattice_llc, pitch, lattice_shape


def build_b4c_sphere_lattice(
    mats: types.ModuleType,
    params: dict,
    safe_spheres: list[tuple[float, float, float]],
    r_b4c: float,
    llc: np.ndarray,
    pitch: np.ndarray,
    lattice_shape: tuple[int, int, int],
) -> openmc.RectLattice:
    """
    Build a B4C sphere search lattice using the given sphere positions.

    Each sphere is filled with b4c_control material; the matrix between spheres is helium.

    Args:
        mats (types.ModuleType): Reactor materials module (used for b4c_control and helium)
        params (dict): Dictionary of reactor/simulation parameters
        safe_spheres (list): List of (x, y, z) sphere centers within constraints
        r_b4c (float): B4C sphere radius in cm
        llc (np.ndarray): Lower-left corner of the bounding box
        pitch (np.ndarray): Lattice cell pitch
        lattice_shape (tuple): (nx, ny, nz) tuple for the search lattice

    Returns:
        openmc.model.create_triso_lattice result (an openmc.RectLattice)
    """

    s_b4c = openmc.Sphere(r=r_b4c)

    c_b4c = openmc.Cell(name='c_b4c_sphere',    fill=mats.b4c_control, region=-s_b4c)
    c_he  = openmc.Cell(name='c_b4c_sphere_he', fill=mats.helium,      region=+s_b4c)

    b4c_sphere_universe = openmc.Universe(cells=[c_b4c, c_he])

    spheres = [openmc.model.TRISO(r_b4c, b4c_sphere_universe, center) for center in safe_spheres]

    lattice = openmc.model.create_triso_lattice(
        spheres, llc, pitch, lattice_shape, mats.helium
    )

    return lattice


def create_b4c_sphere_lattice(
    params: dict,
    mats: types.ModuleType,
    axial_section_height: float,
    r_ss: float,
) -> tuple[openmc.RectLattice, int]:
    """
    Create a B4C sphere lattice for one axial zone of a secondary shutdown rod.

    Convenience wrapper around generate_b4c_sphere_positions() and
    build_b4c_sphere_lattice(). The returned lattice is centered at z=0
    (local coordinates: -axial_section_height/2 to +axial_section_height/2)
    and is intended to be reused across all axial zones via a per-zone Universe
    placed inside an axial HexLattice (which handles z-translation to local frame).

    Args:
        params (dict): Dictionary of reactor/simulation parameters
        mats (types.ModuleType): Materials module containing b4c_control and helium
        axial_section_height (float): Height of one axial zone in cm
        r_ss (float): Radius of the SS rod cylinder in cm

    Returns:
        tuple: (b4c_lattice, n_spheres)
            - b4c_lattice (openmc.RectLattice): OpenMC lattice containing B4C spheres
            - n_spheres (int): Number of B4C spheres per axial zone
    """

    safe_spheres, n_spheres, r_b4c, llc, pitch, lattice_shape = generate_b4c_sphere_positions(
        params, axial_section_height, r_ss
    )

    b4c_lattice = build_b4c_sphere_lattice(mats, params, safe_spheres, r_b4c, llc, pitch, lattice_shape)

    return b4c_lattice, n_spheres
