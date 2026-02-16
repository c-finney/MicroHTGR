import openmc
import numpy as np

def create_triso_lattice(params, mats, axial_section_height):    
    # ====================================================================================================
    # TRISO PARTICLE CREATION
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

    c_triso_fuel   = openmc.Cell(name = 'c_triso_fuel'     , fill = mats.fuel,     region = -s_fuel)
    c_triso_buffer = openmc.Cell(name = 'c_triso_c_buffer' , fill = mats.buffer,   region = +s_fuel & -s_buffer)
    c_triso_ipyc   = openmc.Cell(name = 'c_triso_pyc_inner', fill = mats.pyc,      region = +s_buffer & -s_ipyc)
    c_triso_sic    = openmc.Cell(name = 'c_triso_sic'      , fill = mats.sic,      region = +s_ipyc & -s_sic)
    c_triso_opyc   = openmc.Cell(name = 'c_triso_pyc_outer', fill = mats.pyc,      region = +s_sic & -s_opyc)
    c_triso_matrix = openmc.Cell(name = 'c_triso_matrix'   , fill = mats.graphite, region = +s_opyc)

    triso_universe = openmc.Universe(cells=[c_triso_fuel, c_triso_buffer, c_triso_ipyc, c_triso_sic, c_triso_opyc, c_triso_matrix])

    # ====================================================================================================
    # FUEL COMPACT LATTICE CREATION
    # ====================================================================================================

    # Superimposed TRISO search lattice
    triso_lattice_shape = (4, 4, int(axial_section_height / 0.5))

    fuel_cyl = openmc.ZCylinder(r = params["compact_radius"])

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
    # CURRENTLY NOT WORKING (at high PF (0.35+) this filter will not remove all unsafe TRISOs, some remain outside the lattice)
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

    triso_lattice = openmc.model.create_triso_lattice(random_trisos, llc, pitch, triso_lattice_shape, mats.graphite)

    return triso_lattice
