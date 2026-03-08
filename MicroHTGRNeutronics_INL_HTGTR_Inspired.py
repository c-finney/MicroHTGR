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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    Spatial burnup and fuel representation are controlled by two parameters:

        use_spatial_burnup (bool):
            True  — one fuel material clone per (ring × burnup band); enables
                    radial and axial burnup variation. Required for depletion
                    accuracy. Each burnup region gets its own TRISO lattice
                    (or homogenized cell if use_homogenized_fuel=True).
            False — single shared fuel material for the whole core. Minimal
                    memory and cell count. Useful for k-eff scoping runs.

        use_homogenized_fuel (bool):
            True  — each fuel compact is represented as a single homogenized
                    cell (volume-averaged TRISO layers + matrix graphite).
                    Eliminates all TRISO lattice geometry. ~30,000x fewer cells
                    per compact. Typical Δk vs. explicit TRISO: 50–200 pcm.
                    Standard approach for burnup / depletion studies.
            False — explicit TRISO particle lattice geometry. Reference accuracy
                    for resonance self-shielding. High memory and cell count.

    Args:
        params: Simulation parameters dictionary
        run_dir: Directory for output files

    Returns:
        tuple: (model, n_trisos, m_colors, fuel_clones)
            - model:       openmc.model.Model ready to export or deplete
            - n_trisos:    Number of TRISO particles per axial zone
                           (0 when use_homogenized_fuel=True)
            - m_colors:    Material color dictionary for plotting
            - fuel_clones: list[list[openmc.Material]] — fuel_clones[ring][bax]
    """

    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    # Save params to run directory for post-processing
    save_params(run_dir, params)

    # Save cross_sections.xml file to run directory
    shutil.copy2(cross_sections_path, os.path.join(run_dir, 'cross_sections.xml'))

    model = openmc.model.Model()

    # Read feature flags (default to safe/legacy values if absent)
    use_spatial_burnup    = params.get("use_spatial_burnup",    True)
    use_homogenized_fuel  = params.get("use_homogenized_fuel",  False)

    print(f"\nFuel representation:")
    print(f"  use_spatial_burnup:   {use_spatial_burnup}")
    print(f"  use_homogenized_fuel: {use_homogenized_fuel}")
    
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
    # GENERATE TRISO POSITIONS OR SKIP IF HOMOGENIZED
    # ==================================================================

    if use_homogenized_fuel:
        # No TRISO geometry — positions are never needed.
        # Set sentinel values so downstream volume calculations still work.
        safe_trisos         = []
        n_trisos            = 0
        r_opyc              = None
        llc                 = None
        triso_pitch         = None
        triso_lattice_shape = None
        print("\nUsing Homogenized Fuel: Skipping TRISO position generation.")
    else:
        safe_trisos, n_trisos, r_opyc, llc, triso_pitch, triso_lattice_shape = \
            trisos.generate_triso_positions(
                params=params,
                axial_section_height=axial_section_height
            )

    # ==================================================================
    # CREATE FUEL MATERIAL CLONES
    # ==================================================================

    n_rings          = len(params["core_rings"])
    n_ax             = params["n_ax_zones"]

    if use_spatial_burnup:
        zones_per_region = params.get("ax_zones_per_burnup_region", 1)
        n_burnup_ax      = math.ceil(n_ax / zones_per_region)
    else:
        # Non-spatial: entire core is one burnup band
        zones_per_region = n_ax
        n_burnup_ax      = 1

    print(f"\nSpatial burnup tracking: {n_rings} rings × {n_burnup_ax} burnup bands "
          f"({'spatial' if use_spatial_burnup else 'non-spatial, single shared material'}) "
          f"= {n_rings * n_burnup_ax if use_spatial_burnup else 1} fuel material region(s)")

    if use_homogenized_fuel:
        # Two-region RPT model: inner cylinder (r < rpt_radius) filled with homogenized
        # mixture of all TRISO layers + proportional graphite at pf_inner = pf*(r_compact/rpt_radius)^2.
        # The outer graphite annulus is added in build_homogenized_compact_fill().
        r_rpt = params.get("rpt_radius")
        if r_rpt is None:
            raise ValueError(
                "use_homogenized_fuel=True requires 'rpt_radius' to be set in params. "
                "Run study_execution_mode='RPTCalibration' first to determine the value, "
                "then set rpt_radius in config.py."
            )

        if use_spatial_burnup:
            fuel_clones = []
            for ring_idx in range(n_rings):
                ring_fuels = []
                for bax_idx in range(n_burnup_ax):
                    f = mats.make_rpt_inner_material(
                        params, r_rpt, name=f"RPTInner_ring{ring_idx}_bax{bax_idx}"
                    )
                    ring_fuels.append(f)
                fuel_clones.append(ring_fuels)
        else:
            # Single shared RPT inner material for all rings and axial zones
            f_single = mats.make_rpt_inner_material(params, r_rpt, name="RPTInner_global")
            fuel_clones = [[f_single] for _ in range(n_rings)]

    else:
        # Explicit TRISO: clone the kernel-only fuel material as before
        if use_spatial_burnup:
            fuel_clones = []
            for ring_idx in range(n_rings):
                ring_fuels = []
                for bax_idx in range(n_burnup_ax):
                    f = mats.fuel.clone()
                    f.name = f"Fuel_ring{ring_idx}_bax{bax_idx}"
                    f.depletable = True
                    ring_fuels.append(f)
                fuel_clones.append(ring_fuels)
        else:
            # Single shared fuel material for all rings and axial zones
            f_single = mats.fuel.clone()
            f_single.name = "Fuel_global"
            f_single.depletable = True
            fuel_clones = [[f_single] for _ in range(n_rings)]

    # ==================================================================
    # BUILD FUEL FILLS (TRISO LATTICES OR HOMOGENIZED UNIVERSES)
    # ==================================================================
    #
    # ring_triso_lattices[ring_idx][ax_idx] holds the object used as the
    # fill for a fuel compact cell in assembly.py.  For explicit TRISO this
    # is an openmc.RectLattice; for homogenized fuel it is an openmc.Universe
    # wrapping a single material cell.  assembly.py uses the fill object
    # identically in both cases.

    if use_homogenized_fuel:
        # Build one homogenized-compact universe per (ring, burnup band)
        # and expand to full axial-zone mapping — fast, no lattice construction.

        if use_spatial_burnup:
            n_lattices = n_rings * n_burnup_ax
            print(f"Building {n_lattices} homogenized compact universes "
                  f"({n_rings} rings × {n_burnup_ax} burnup bands)...")

            burnup_fills = {r: {} for r in range(n_rings)}
            for ring_idx in range(n_rings):
                for bax_idx in range(n_burnup_ax):
                    burnup_fills[ring_idx][bax_idx] = trisos.build_homogenized_compact_fill(
                        fuel_material = fuel_clones[ring_idx][bax_idx],
                        params        = params,
                        mats          = mats,
                    )
        else:
            # Single shared homogenized universe for the whole core
            print("Building 1 shared homogenized compact universe (non-spatial burnup)...")
            shared_fill = trisos.build_homogenized_compact_fill(
                fuel_material = fuel_clones[0][0],
                params        = params,
                mats          = mats,
            )
            burnup_fills = {r: {0: shared_fill} for r in range(n_rings)}

        ring_triso_lattices = {
            ring_idx: {
                ax_idx: burnup_fills[ring_idx][ax_idx // zones_per_region]
                for ax_idx in range(n_ax)
            }
            for ring_idx in range(n_rings)
        }
        print("Done building homogenized compact universes.")

    else:
        # Explicit TRISO path — build one lattice per (ring, burnup band),
        # parallelised with threads as before.

        if use_spatial_burnup:
            n_lattices = n_rings * n_burnup_ax
        else:
            n_lattices = 1  # single shared lattice

        n_workers = min(n_lattices, os.cpu_count() or 4)
        print(f"Building {n_lattices} TRISO lattice(s) using {n_workers} thread(s)...")

        def _build_one(ring_idx, bax_idx):
            return ring_idx, bax_idx, trisos.build_triso_lattice_for_material(
                fuel_material       = fuel_clones[ring_idx][bax_idx],
                mats                = mats,
                params              = params,
                safe_trisos         = safe_trisos,
                r_opyc              = r_opyc,
                llc                 = llc,
                pitch               = triso_pitch,
                triso_lattice_shape = triso_lattice_shape,
            )

        if use_spatial_burnup:
            burnup_triso_lattices = {r: {} for r in range(n_rings)}
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(_build_one, ring_idx, bax_idx)
                    for ring_idx in range(n_rings)
                    for bax_idx in range(n_burnup_ax)
                ]
                for i, fut in enumerate(as_completed(futures), 1):
                    ring_idx, bax_idx, lattice = fut.result()
                    burnup_triso_lattices[ring_idx][bax_idx] = lattice
                    if i % 10 == 0 or i == n_lattices:
                        print(f"  {i}/{n_lattices} lattices built")
        else:
            # Build exactly one lattice and reuse it everywhere
            _, _, shared_lattice = _build_one(0, 0)
            burnup_triso_lattices = {r: {0: shared_lattice} for r in range(n_rings)}
            print(f"  1/1 lattices built (shared across all rings and axial zones)")

        print("Done building TRISO lattices.")

        # Expand burnup-band mapping to full axial-zone mapping
        ring_triso_lattices = {
            ring_idx: {
                ax_idx: burnup_triso_lattices[ring_idx][ax_idx // zones_per_region]
                for ax_idx in range(n_ax)
            }
            for ring_idx in range(n_rings)
        }

    # ==================================================================
    # SET FUEL KERNEL VOLUMES ANALYTICALLY
    # ==================================================================
    #
    # For homogenized fuel the "fuel kernel volume" in a burnup band is
    # the volume of all fuel kernels within the homogenized compact
    # cylinders in that band, computed from the kernel volume fraction:
    #   vf_kernel = pf * (r_kernel / r_opyc)^3
    #   V_kernel_in_compact = vf_kernel * pi * r_compact^2 * h_compact
    #
    # For explicit TRISO the existing n_trisos * V_kernel formula is used.

    def _n_fuel_compacts_for_code(code):
        if not code.startswith('f'):
            return 0
        if 'cp' in code:
            return 30
        if 'ss' in code and code.endswith('p'):
            return 30
        if 'c' in code or 'ss' in code:
            return 36
        if code == 'fp':
            return 36
        return 42   # "f" or "fpa"

    V_kernel        = (4.0 / 3.0) * np.pi * params["kernel_radius"]**3
    geometry_factor = 6 if params["use_1/6_geometry"] else 1

    if use_homogenized_fuel:
        # The RPT inner material fills a cylinder of radius rpt_radius, so its
        # volume per compact per zone is V_inner = pi * rpt_radius^2 * h.
        r_rpt = params["rpt_radius"]
        V_inner_cross_section = np.pi * r_rpt**2

        if use_spatial_burnup:
            for ring_idx, ring_def in enumerate(params["core_rings"]):
                n_fuel_compacts = sum(_n_fuel_compacts_for_code(c) for c in ring_def)
                V_inner_per_zone = (V_inner_cross_section
                                    * axial_section_height * n_fuel_compacts
                                    / geometry_factor)
                for bax_idx in range(n_burnup_ax):
                    band_start   = bax_idx * zones_per_region
                    actual_zones = min(zones_per_region, n_ax - band_start)
                    fuel_clones[ring_idx][bax_idx].volume = actual_zones * V_inner_per_zone
        else:
            total_volume = 0.0
            for ring_idx, ring_def in enumerate(params["core_rings"]):
                n_fuel_compacts = sum(_n_fuel_compacts_for_code(c) for c in ring_def)
                V_inner_per_zone = (V_inner_cross_section
                                    * axial_section_height * n_fuel_compacts
                                    / geometry_factor)
                actual_zones = min(zones_per_region, n_ax)
                total_volume += actual_zones * V_inner_per_zone
            fuel_clones[0][0].volume = total_volume

    else:
        if use_spatial_burnup:
            for ring_idx, ring_def in enumerate(params["core_rings"]):
                n_fuel_compacts = sum(_n_fuel_compacts_for_code(c) for c in ring_def)
                V_per_zone = (n_trisos * V_kernel * n_fuel_compacts) / geometry_factor
                for bax_idx in range(n_burnup_ax):
                    band_start   = bax_idx * zones_per_region
                    actual_zones = min(zones_per_region, n_ax - band_start)
                    fuel_clones[ring_idx][bax_idx].volume = actual_zones * V_per_zone
        else:
            total_volume = 0.0
            for ring_idx, ring_def in enumerate(params["core_rings"]):
                n_fuel_compacts = sum(_n_fuel_compacts_for_code(c) for c in ring_def)
                V_per_zone = (n_trisos * V_kernel * n_fuel_compacts) / geometry_factor
                actual_zones = min(zones_per_region, n_ax)
                total_volume += actual_zones * V_per_zone
            fuel_clones[0][0].volume = total_volume

    # Set poison volume analytically
    def _n_poison_compacts_for_code(code):
        if not code.startswith('f'):
            return 0
        if code == 'fpa':
            return 1
        if 'p' in code:
            return 6
        return 0

    n_poison_compacts = sum(
        _n_poison_compacts_for_code(c)
        for ring_def in params["core_rings"]
        for c in ring_def
    )
    V_poison_column  = np.pi * params["compact_radius"]**2 * params["core_height"]
    poison_volume    = n_poison_compacts * V_poison_column / geometry_factor
    mats.b4c_poison.volume = poison_volume

    # ==================================================================
    # CREATE ASSEMBLIES
    # ==================================================================

    assemblies, m_colors, bundle_pitch = asm.create_assembly_univs(
        params             = params,
        mats               = mats,
        T_coolant_z        = T_coolant_z,
        T_compact_z        = T_compact_z,
        T_matrix_z         = T_matrix_z,
        T_reflector_z      = T_reflector_z,
        ring_triso_lattices = ring_triso_lattices,
        axial_coords       = axial_coords,
        reactor_bottom     = reactor_bottom,
        reactor_top        = reactor_top
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
    # FULL CORE AND RADIAL REFLECTOR CREATION
    # ==================================================================

    T_refl_avg = 0.5 * (params["reflector_min"] + params["reflector_max"])
    m_colors[mats.graphite] = 'darkblue'
    outer_graphite_cell = openmc.Cell(fill=mats.graphite)
    outer_graphite_cell.temperature = T_refl_avg
    core_lattice.outer = openmc.Universe(cells=[outer_graphite_cell])

    core_cyl = openmc.ZCylinder(r=params["core_radius"], boundary_type='vacuum')
    min_z = openmc.ZPlane(z0=reactor_bottom)
    max_z = openmc.ZPlane(z0=reactor_top)

    # Wedge planes (defined for both full and 1/6 so downstream code can reference them)
    plane_1 = None
    plane_2 = None
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

    def _apply_wedge(region):
        """Intersect region with wedge planes if using 1/6 geometry."""
        if params["use_1/6_geometry"]:
            return region & +plane_1 & -plane_2
        return region

    # Hex lattice circumscribed radius (outer edge of outermost assemblies)
    n_core_rings = len(params["core_rings"])
    lattice_extent_r = (n_core_rings - 1) * bundle_pitch + bundle_pitch / 4
    lattice_cyl = openmc.ZCylinder(r=lattice_extent_r)

    # Z-planes for axial zoning (reuse min_z / max_z at boundaries)
    z_planes = []
    for i, z in enumerate(axial_coords):
        if i == 0:
            z_planes.append(min_z)
        elif i == len(axial_coords) - 1:
            z_planes.append(max_z)
        else:
            z_planes.append(openmc.ZPlane(z0=z))

    core_cell = openmc.Cell(
        fill=core_lattice,
        region=_apply_wedge(-lattice_cyl & +min_z & -max_z))

    # ==================================================================
    # BERYLLIUM OXIDE REFLECTOR  (optional annular BeO region)
    # ==================================================================

    inner_graphite_cells = []
    beo_cells = []
    outer_graphite_cells = []

    use_beo = params["use_BeO_reflector"] and params["BeO_thickness"] > 0.0
    if use_beo:
        beo_inner_r = params["BeO_inner_radius"]
        if beo_inner_r is None:
            beo_inner_r = lattice_extent_r
        beo_thickness = params["BeO_thickness"]
        beo_outer_r = min(beo_inner_r + beo_thickness, params["core_radius"])

        print(f"\nBeO reflector enabled:")
        print(f"  Lattice extent:  {lattice_extent_r:.2f} cm  "
              f"({n_core_rings} rings)")

        beo_inner_cyl = openmc.ZCylinder(r=beo_inner_r)

        need_inner_graphite = beo_inner_r > lattice_extent_r
        if need_inner_graphite:
            print(f"  Inner graphite:  {lattice_extent_r:.2f} -> "
                  f"{beo_inner_r:.2f} cm")
            for idx in range(len(axial_coords) - 1):
                T_refl = T_reflector_z[idx]
                region = (+lattice_cyl & -beo_inner_cyl
                          & +z_planes[idx] & -z_planes[idx + 1])
                cell = openmc.Cell(fill=mats.graphite, region=_apply_wedge(region))
                cell.temperature = T_refl
                inner_graphite_cells.append(cell)
        else:
            print(f"  No inner graphite (BeO starts at lattice edge)")

        print(f"  BeO annulus:     {beo_inner_r:.2f} -> "
              f"{beo_outer_r:.2f} cm  "
              f"(thickness = {beo_outer_r - beo_inner_r:.2f} cm)")

        need_outer_graphite = beo_outer_r < params["core_radius"]
        if need_outer_graphite:
            beo_outer_cyl = openmc.ZCylinder(r=beo_outer_r)
            beo_outer_surf = beo_outer_cyl
            print(f"  Outer graphite:  {beo_outer_r:.2f} -> "
                  f"{params['core_radius']:.2f} cm")
        else:
            beo_outer_surf = core_cyl
            print(f"  BeO extends to core boundary (no outer graphite)")

        m_colors[mats.beo] = 'lightblue'
        for idx in range(len(axial_coords) - 1):
            T_refl = T_reflector_z[idx]
            region = (+beo_inner_cyl & -beo_outer_surf
                      & +z_planes[idx] & -z_planes[idx + 1])
            cell = openmc.Cell(fill=mats.beo, region=_apply_wedge(region))
            cell.temperature = T_refl
            beo_cells.append(cell)

        if need_outer_graphite:
            for idx in range(len(axial_coords) - 1):
                T_refl = T_reflector_z[idx]
                region = (+beo_outer_cyl & -core_cyl
                          & +z_planes[idx] & -z_planes[idx + 1])
                cell = openmc.Cell(fill=mats.graphite, region=_apply_wedge(region))
                cell.temperature = T_refl
                outer_graphite_cells.append(cell)

    else:
        for idx in range(len(axial_coords) - 1):
            T_refl = T_reflector_z[idx]
            region = (+lattice_cyl & -core_cyl
                      & +z_planes[idx] & -z_planes[idx + 1])
            cell = openmc.Cell(fill=mats.graphite, region=_apply_wedge(region))
            cell.temperature = T_refl
            outer_graphite_cells.append(cell)

    top_refl_z = reactor_top + params["reflector_thickness"]
    bottom_refl_z = reactor_bottom - params["reflector_thickness"]
    top_refl = openmc.ZPlane(z0=top_refl_z, boundary_type='vacuum')
    bottom_refl = openmc.ZPlane(z0=bottom_refl_z, boundary_type='vacuum')

    top_refl_cell = openmc.Cell(
        fill=mats.graphite,
        region=_apply_wedge(-core_cyl & +max_z & -top_refl))
    top_refl_cell.temperature = T_reflector_axial
    bottom_refl_cell = openmc.Cell(
        fill=mats.graphite,
        region=_apply_wedge(-core_cyl & +bottom_refl & -min_z))
    bottom_refl_cell.temperature = T_reflector_axial

    all_geometry_cells = ([core_cell]
                          + inner_graphite_cells
                          + beo_cells
                          + outer_graphite_cells
                          + [top_refl_cell, bottom_refl_cell])
    geometry = openmc.Geometry(all_geometry_cells)
    model.geometry = geometry

    # ==================================================================
    # GEOMETRY PLOT GENERATION
    # ==================================================================

    # Assign plotting colors
    for ring_fuels in fuel_clones:
        for f in ring_fuels:
            m_colors[f] = 'palegreen'
    m_colors[mats.buffer] = 'sandybrown'
    m_colors[mats.pyc] = 'orange'
    m_colors[mats.sic] = 'yellow'
    m_colors[mats.graphite] = 'darkblue'
    m_colors[mats.b4c_poison] = 'purple'
    m_colors[mats.b4c_ss] = 'orange'
    if params["bank_1_insertion"] > 0 or params["bank_2_insertion"] > 0 or params["bank_3_insertion"] > 0:
        m_colors[mats.b4c_control] = 'black'
    m_colors[mats.incoloy800H] = 'gray'
    m_colors[mats.beo] = 'lightblue'

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
    energy_bins = np.logspace(-9, 7, 200)
    energy_filter = openmc.EnergyFilter(energy_bins)

    # ----- Global Tallies -----

    if params["use_global_tallies"]:
        flux_spectrum_tally = openmc.Tally(name="flux_energy_spectrum")
        flux_spectrum_tally.scores = ["flux"]
        flux_spectrum_tally.filters = [energy_filter]

        heating_tally = openmc.Tally(name="heating")
        heating_tally.scores = ["heating-local"]
        
        global_tally = openmc.Tally(name='global_rates')
        global_tally.scores = ['flux', 'fission', 'nu-fission']

        tallies += [flux_spectrum_tally, heating_tally, global_tally]

    # ----- Mesh Tallies -----

    if params["use_mesh_tallies"]:
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
            mesh_nx = params["n_XY_mesh_zones_full_core"]
            mesh_y_min = -params["core_radius"]
            mesh_y_max = params["core_radius"]
            mesh_ny = params["n_XY_mesh_zones_full_core"]

        # --- Flux/Fission Mesh Tally (active core only) ---
        mesh = openmc.RegularMesh()
        mesh.dimension = [mesh_nx, mesh_ny, params["n_ax_zones"]]
        mesh.lower_left = [mesh_x_min, mesh_y_min, reactor_bottom]
        mesh.upper_right = [mesh_x_max, mesh_y_max, reactor_top]
        mesh_filter = openmc.MeshFilter(mesh)

        mesh_tally_active = openmc.Tally(name='mesh_rates')
        mesh_tally_active.filters = [mesh_filter]
        mesh_tally_active.scores = ['flux', 'fission']

        # --- Heating Mesh Tally (active core only) ---
        mesh_heating_tally = openmc.Tally(name='mesh_heating')
        mesh_heating_tally.filters = [mesh_filter]
        mesh_heating_tally.scores = ['heating-local']

        n_reflector_zones = 33
        n_total_zones = n_reflector_zones + params["n_ax_zones"] + n_reflector_zones

        # --- Flux/Fission Mesh Tally (full core) ---
        mesh_full = openmc.RegularMesh()
        mesh_full.dimension = [mesh_nx, mesh_ny, n_total_zones]
        mesh_bottom = reactor_bottom - params["reflector_thickness"]
        mesh_top = reactor_top + params["reflector_thickness"]
        mesh_full.lower_left = [mesh_x_min, mesh_y_min, mesh_bottom]
        mesh_full.upper_right = [mesh_x_max, mesh_y_max, mesh_top]
        mesh_full_filter = openmc.MeshFilter(mesh_full)

        mesh_tally_full = openmc.Tally(name='mesh_rates_full')
        mesh_tally_full.filters = [mesh_full_filter]
        mesh_tally_full.scores = ['flux', 'fission']

        # --- Heating Mesh Tally (full core) ---
        mesh_heating_tally_full = openmc.Tally(name='mesh_heating_full')
        mesh_heating_tally_full.filters = [mesh_full_filter]
        mesh_heating_tally_full.scores = ['heating-local']

        tallies += [mesh_tally_active, mesh_heating_tally, mesh_tally_full, mesh_heating_tally_full]

    # ----- Neutron Leakage Tallies -----

    if params["use_leakage_tallies"]:
        radial_surf_filter    = openmc.SurfaceFilter(core_cyl)
        axial_top_surf_filter = openmc.SurfaceFilter(top_refl)
        axial_bot_surf_filter = openmc.SurfaceFilter(bottom_refl)

        radial_current_tally = openmc.Tally(name='radial_leakage_current')
        radial_current_tally.filters = [radial_surf_filter, energy_filter]
        radial_current_tally.scores  = ['current']

        axial_top_current_tally = openmc.Tally(name='axial_top_leakage_current')
        axial_top_current_tally.filters = [axial_top_surf_filter, energy_filter]
        axial_top_current_tally.scores  = ['current']

        axial_bot_current_tally = openmc.Tally(name='axial_bot_leakage_current')
        axial_bot_current_tally.filters = [axial_bot_surf_filter, energy_filter]
        axial_bot_current_tally.scores  = ['current']

        tallies += [radial_current_tally, axial_top_current_tally, axial_bot_current_tally]

    # ----- BeO Reflector Tallies -----

    if params["use_BeO_tallies"] and params["use_BeO_reflector"]:
        if len(beo_cells) > 0:
            beo_inner_r = params["BeO_inner_radius"] if params["BeO_inner_radius"] is not None else lattice_extent_r
            beo_outer_r = min(beo_inner_r + params["BeO_thickness"], params["core_radius"])

            num_bins = round(params["n_XY_mesh_zones_full_core"] / 180 * params["BeO_thickness"])

            if params["use_1/6_geometry"]:
                phi_grid = np.linspace(0, np.pi / 3, 7)
            else:
                phi_grid = np.linspace(0, 2 * np.pi, 13)

            beo_cyl_mesh = openmc.CylindricalMesh(
                r_grid   = np.linspace(beo_inner_r, beo_outer_r, num_bins + 1),
                z_grid   = axial_coords,
                phi_grid = phi_grid
            )

            beo_mesh_filter = openmc.MeshFilter(beo_cyl_mesh)

            beo_flux_tally = openmc.Tally(name='beo_flux_radial')
            beo_flux_tally.filters = [beo_mesh_filter]
            beo_flux_tally.scores = ['flux']

            tallies += [beo_flux_tally]
            print(f"\nBeO flux tally enabled:")
            print(f"  Radial bins:    {num_bins} from r = {beo_inner_r:.2f} to {beo_outer_r:.2f} cm")
            print(f"  Axial bins:     {len(axial_coords) - 1} (reusing core axial zones)")
            print(f"  Azimuthal bins: {'6 over 60-degree wedge (1/6 geometry)' if params['use_1/6_geometry'] else '12 over full core'}")
        else:
            print("\nWARNING: use_BeO_tallies=True but no BeO cells were created.")
    elif params["use_BeO_tallies"]:
        print("\nWARNING: BeO flux tallies skipped — no BeO reflector was used.")

    model.tallies = tallies

    # ==================================================================
    # MONTE CARLO SETTINGS
    # ==================================================================

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.batches = params["total_batches"]
    settings.inactive = params["inactive_batches"]
    settings.particles = params["particles"]
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

    model.settings = settings

    # Exclude the base fuel template — only the depletable clones should be in the model
    explicit_mats = [m for m in mats.materials if m is not mats.fuel]
    for ring_fuels in fuel_clones:
        explicit_mats.extend(ring_fuels)

    # Homogenized fuel uses temporary constituent materials internally via
    # mix_materials(); those are not tracked separately and do not need to
    # be added here — mix_materials returns a single combined material.
    model.materials = openmc.Materials(explicit_mats)

    for plot in model.plots:
        if plot.color_by == 'material':
            plot.colors = m_colors

    return model, n_trisos, m_colors, fuel_clones

# ====================================================================================================
# RPT CALIBRATION HELPERS
# ====================================================================================================

def _extract_keff(run_dir):
    """
    Read the final k_eff and its standard deviation from the statepoint in run_dir.

    Returns:
        (k_eff_mean, k_eff_std): floats
    """
    import glob
    sp_files = sorted(glob.glob(os.path.join(run_dir, "statepoint.*.h5")))
    if not sp_files:
        raise FileNotFoundError(f"No statepoint file found in {run_dir}")
    sp = openmc.StatePoint(sp_files[-1])
    return float(sp.keff.n), float(sp.keff.s)


def run_rpt_calibration(params, output_base_dir, run_simulation_fn):
    """
    Find the RPT inner radius (rpt_radius) that matches explicit-TRISO k_eff.

    Algorithm
    ---------
    1. Run an explicit-TRISO reference simulation (use_homogenized_fuel=False)
       to obtain k_eff_ref.
    2. Scan rpt_calibration_n_points values of r_rpt linearly from
         r_min = compact_radius * sqrt(triso_pf)   (maximum self-shielding)
       to
         r_max = compact_radius                    (flat homogenization baseline)
    3. Interpolate linearly between the two bracketing points to estimate the
       r_rpt at which k_eff(RPT) = k_eff_ref.
    4. Save results to rpt_calibration_results.json and print a summary table.

    Because k_eff is monotonically increasing as r_rpt decreases, the scan
    is guaranteed to find at most one crossing.

    After running, set  rpt_radius = <optimal value>  in config.py, then use
    study_execution_mode = "SingleStudy" or "DepletionStudy" with
    use_homogenized_fuel = True.

    Args:
        params:            Simulation parameters dictionary (from config.py).
        output_base_dir:   Root directory for all calibration run sub-directories.
        run_simulation_fn: Callable matching the signature of run_simulation().

    Returns:
        dict: Full calibration results (also saved as JSON).
    """
    import copy

    r_compact = params["compact_radius"]
    pf        = params["triso_pf"]
    r_min     = r_compact * pf**0.5         # minimum r_rpt (pf_inner = 1)
    r_max     = r_compact                    # maximum r_rpt (flat homogenization)
    n_pts     = params.get("rpt_calibration_n_points", 8)

    print(f"\n{'='*80}")
    print("RPT CALIBRATION STUDY")
    print(f"  compact_radius  = {r_compact:.4f} cm")
    print(f"  triso_pf        = {pf:.3f}")
    print(f"  r_rpt scan      = [{r_min:.4f}, {r_max:.4f}] cm  ({n_pts} points)")
    print(f"  Output dir      = {output_base_dir}")
    print(f"{'='*80}")

    os.makedirs(output_base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: explicit-TRISO reference
    # ------------------------------------------------------------------
    print(f"\n--- Step 1: Explicit-TRISO reference run ---")
    ref_params = copy.copy(params)
    ref_params["use_homogenized_fuel"] = False
    ref_params["make_geometry_plots"]  = False
    ref_dir = os.path.join(output_base_dir, "rpt_ref_explicit_TRISO")

    run_simulation_fn(ref_params, ref_dir)
    k_eff_ref, k_eff_ref_std = _extract_keff(ref_dir)

    print(f"\nReference k_eff (explicit TRISO): {k_eff_ref:.5f} ± {k_eff_ref_std:.5f}")

    # ------------------------------------------------------------------
    # Step 2: scan r_rpt values with early exit on bracket
    #
    # k_eff is monotonically DECREASING as r_rpt increases (r_min → r_max):
    #   r_min → pf_inner = 1  → maximum self-shielding → highest k_eff
    #   r_max → pf_inner = pf → flat homogenization   → lowest  k_eff
    #
    # As soon as two consecutive points straddle k_eff_ref we have a
    # bracket and can interpolate immediately — no further runs needed.
    # ------------------------------------------------------------------
    r_rpt_values    = np.linspace(r_min, r_max, n_pts)
    rpt_results     = []
    optimal_r_rpt   = None
    optimal_delta_k = None
    bracket_found   = False

    for i, r_rpt in enumerate(r_rpt_values):
        pf_inner = pf * (r_compact / r_rpt)**2
        print(f"\n--- RPT Case {i+1}/{n_pts}: r_rpt = {r_rpt:.4f} cm  "
              f"(pf_inner = {pf_inner:.4f}) ---")

        rpt_params = copy.copy(params)
        rpt_params["use_homogenized_fuel"] = True
        rpt_params["rpt_radius"]           = r_rpt
        rpt_params["make_geometry_plots"]  = False

        case_dir = os.path.join(output_base_dir,
                                f"rpt_case_{i+1:02d}_r{r_rpt:.4f}cm")
        run_simulation_fn(rpt_params, case_dir)

        k_eff, k_eff_std = _extract_keff(case_dir)
        delta_k_pcm = (k_eff - k_eff_ref) * 1e5
        print(f"  k_eff = {k_eff:.5f} ± {k_eff_std:.5f}  "
              f"(Δk = {delta_k_pcm:+.0f} pcm vs. reference)")

        rpt_results.append({
            "r_rpt":       float(r_rpt),
            "pf_inner":    float(pf_inner),
            "k_eff":       float(k_eff),
            "k_eff_std":   float(k_eff_std),
            "delta_k_pcm": float(delta_k_pcm),
        })

        # Early exit: check if the last two points bracket k_eff_ref
        if len(rpt_results) >= 2:
            k_prev = rpt_results[-2]["k_eff"]
            r_prev = rpt_results[-2]["r_rpt"]
            if (k_prev - k_eff_ref) * (k_eff - k_eff_ref) <= 0:
                # Linear interpolation within the bracket
                optimal_r_rpt   = r_prev + (k_eff_ref - k_prev) / (k_eff - k_prev) * (r_rpt - r_prev)
                optimal_delta_k = 0.0
                bracket_found   = True
                remaining       = n_pts - i - 1
                print(f"\n  >>> Bracket found between cases {i} and {i+1}. "
                      f"Interpolated r_rpt = {optimal_r_rpt:.4f} cm")
                if remaining > 0:
                    print(f"  Early exit: skipping remaining {remaining} case(s).")
                break

    # ------------------------------------------------------------------
    # Step 3: fallback if no bracket was found in the scan range
    # ------------------------------------------------------------------
    if not bracket_found:
        best = min(rpt_results, key=lambda r: abs(r["k_eff"] - k_eff_ref))
        optimal_r_rpt   = best["r_rpt"]
        optimal_delta_k = best["delta_k_pcm"]
        print(f"\nWARNING: No bracketing crossing found across the full scan range "
              f"[{r_min:.4f}, {r_max:.4f}] cm.")
        print(f"  Nearest point ({optimal_r_rpt:.4f} cm, Δk = {optimal_delta_k:+.0f} pcm) "
              f"used as best estimate.")
        print(f"  Consider increasing rpt_calibration_n_points or checking whether "
              f"the explicit-TRISO k_eff lies within the RPT model's achievable range.")

    # ------------------------------------------------------------------
    # Step 4: save results and print summary
    # ------------------------------------------------------------------
    calibration_results = {
        "reference_k_eff":         k_eff_ref,
        "reference_k_eff_std":     k_eff_ref_std,
        "r_compact":               r_compact,
        "triso_pf":                pf,
        "r_rpt_min":               float(r_min),
        "r_rpt_max":               float(r_max),
        "n_points":                n_pts,
        "rpt_cases":               rpt_results,
        "optimal_r_rpt":           float(optimal_r_rpt),
        "optimal_delta_k_pcm":     float(optimal_delta_k) if optimal_delta_k is not None else None,
    }

    results_path = os.path.join(output_base_dir, "rpt_calibration_results.json")
    with open(results_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)

    print(f"\n{'='*80}")
    print("RPT CALIBRATION SUMMARY")
    print(f"{'='*80}")
    print(f"Reference k_eff (explicit TRISO): {k_eff_ref:.5f} ± {k_eff_ref_std:.5f}\n")
    print(f"  {'r_rpt (cm)':>12}  {'pf_inner':>9}  {'k_eff':>10}  {'±':>1}  {'std':>8}  {'Δk (pcm)':>10}")
    print(f"  {'-'*60}")
    for r in rpt_results:
        print(f"  {r['r_rpt']:>12.4f}  {r['pf_inner']:>9.4f}  "
              f"{r['k_eff']:>10.5f}  ±  {r['k_eff_std']:>8.5f}  "
              f"{r['delta_k_pcm']:>+10.0f}")
    print(f"\n  >>> Estimated optimal r_rpt = {optimal_r_rpt:.4f} cm")
    print(f"\n  Set the following in config.py:")
    print(f"    \"rpt_radius\": {optimal_r_rpt:.4f},")
    print(f"\n  Full results saved to: {results_path}")
    print(f"{'='*80}\n")

    return calibration_results


# ====================================================================================================
# MAIN SIMULATION FUNCTION
# ====================================================================================================

def run_simulation(params, run_dir):
    """
    Build and run an eigenvalue OpenMC simulation.
    
    Returns:
        n_trisos: Number of TRISO particles per axial zone
                  (0 when use_homogenized_fuel=True)
    """

    model, n_trisos, m_colors, fuel_clones = build_model(params, run_dir)
    model.export_to_xml()

    if params.get("make_geometry_plots", False):
        n_plot_threads = str(params.get("plot_threads", os.cpu_count()))
        old_omp = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = n_plot_threads
        try:
            openmc.plot_geometry(output=False, cwd=run_dir)
        finally:
            if old_omp is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = old_omp

    geometry_factor           = 6 if params["use_1/6_geometry"] else 1
    poison_volume             = mats.b4c_poison.volume
    total_poison_volume       = poison_volume * geometry_factor
    
    # Deduplicate by material ID before summing
    # When use_spatial_burnup=False all rings share the same material object so we must not double-count it
    seen_ids = set()
    fuel_volume_simulated = 0.0
    for ri in range(len(fuel_clones)):
        for ai in range(len(fuel_clones[ri])):
            mat = fuel_clones[ri][ai]
            if mat.id not in seen_ids:
                seen_ids.add(mat.id)
                fuel_volume_simulated += mat.volume
    total_fuel_volume = fuel_volume_simulated * geometry_factor

    uco_density_g_cm3  = params["kernel_density"] / 1000.0
    u_mass_fraction    = 238.0 / 268.0
    total_HM_mass_kg   = total_fuel_volume * uco_density_g_cm3 * u_mass_fraction / 1000.0

    b4c_density_g_cm3  = params["B4C_density_poison"] / 1000.0
    b10_enrichment     = params["B10_enrichment_poison"]
    mass_10            = openmc.data.atomic_mass('B10')
    mass_11            = openmc.data.atomic_mass('B11')
    b10_mass_fraction  = (b10_enrichment * mass_10) / (
        b10_enrichment * mass_10 + (1.0 - b10_enrichment) * mass_11
    )
    total_B10_mass_kg  = total_poison_volume * b4c_density_g_cm3 * b10_mass_fraction / 1000.0

    print(f"\nFuel volume   (simulated geometry): {fuel_volume_simulated:.4f} cm³")
    print(f"Fuel volume   (full core):          {total_fuel_volume:.4f} cm³")
    print(f"Uranium mass  (full core):          {total_HM_mass_kg:.2f} kg")
    print(f"B4C poison    (simulated geometry): {poison_volume:.4f} cm³")
    print(f"B4C poison    (full core):          {total_poison_volume:.4f} cm³")
    print(f"B-10 mass     (full core):          {total_B10_mass_kg:.4f} kg")

    params_path = os.path.join(run_dir, 'run_params.json')
    saved_params = json.load(open(params_path)) if os.path.exists(params_path) else {}
    saved_params.update({
        'n_trisos':                    n_trisos,
        'use_homogenized_fuel':        params.get("use_homogenized_fuel", False),
        'use_spatial_burnup':          params.get("use_spatial_burnup", True),
        'poison_material_id':          mats.b4c_poison.id,
        'fuel_volume_simulated_cm3':   fuel_volume_simulated,
        'fuel_volume_full_core_cm3':   total_fuel_volume,
        'total_HM_mass_kg':            total_HM_mass_kg,
        'poison_volume_simulated_cm3': poison_volume,
        'poison_volume_full_core_cm3': total_poison_volume,
        'total_B10_mass_kg':           total_B10_mass_kg,
        'fuel_mat_volumes': {
            str(fuel_clones[ri][ai].id): fuel_clones[ri][ai].volume
            for ri in range(len(fuel_clones))
            for ai in range(len(fuel_clones[ri]))
        },
        'fuel_mat_ids': [
            [fuel_clones[ri][ai].id for ai in range(len(fuel_clones[ri]))]
            for ri in range(len(fuel_clones))
        ],
    })
    with open(params_path, 'w') as f:
        json.dump(saved_params, f, indent=2)
    print(f"Volume data saved to run_params.json\n")

    openmc_output_file = os.path.join(run_dir, 'openmc_output.txt')

    with open(openmc_output_file, 'w', buffering=1) as outf:
        process = subprocess.Popen(
            ['openmc'],
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env={**os.environ, 'OMP_NUM_THREADS': '128'}
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
                  (0 when use_homogenized_fuel=True)
    """

    print(f"\n{'=' * 80}")
    print("DEPLETION SIMULATION")
    print(f"{'=' * 80}")

    is_restart = params["restart_depletion"]

    # ==================================================================
    # RESTART PATH
    # ==================================================================

    if is_restart:
        restart_dir = params["restart_run_dir"]
        prev_h5 = os.path.join(restart_dir, "depletion_results.h5")

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

        for cls in [openmc.Material, openmc.Cell, openmc.Universe,
                    openmc.Surface, openmc.Lattice]:
            if hasattr(cls, 'used_ids'):
                cls.used_ids.clear()
        if hasattr(openmc, 'reset_auto_ids'):
            openmc.reset_auto_ids()

        materials = openmc.Materials.from_xml(
            os.path.join(restart_dir, "materials.xml")
        )
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

        with open(required_files["run_params.json"], 'r') as f:
            saved_params = json.load(f)

        n_trisos      = saved_params["n_trisos"]
        poison_mat_id = saved_params["poison_material_id"]
        poison_volume = saved_params["poison_volume_simulated_cm3"]

        fuel_mat_volumes = {
            int(k): v for k, v in saved_params.get("fuel_mat_volumes", {}).items()
        }

        n_fuel_vols_set = 0
        for mat in materials:
            if mat.id in fuel_mat_volumes:
                mat.volume = fuel_mat_volumes[mat.id]
                n_fuel_vols_set += 1
            elif mat.id == poison_mat_id:
                mat.volume = poison_volume
                print(f"Poison material (id={mat.id}): volume = {poison_volume:.4f} cm³")
        print(f"Set volumes on {n_fuel_vols_set} fuel material clones from saved run_params.json")

        prev_results = openmc.deplete.Results(prev_h5)
        n_completed = len(prev_results) - 1
        print(f"Completed depletion steps: {n_completed}")

        restart_ts = params["restart_timesteps_days"]
        if restart_ts is not None and len(restart_ts) > 0:
            timesteps_days = restart_ts
            print(f"Using user-specified restart timesteps: {timesteps_days}")
        else:
            original_ts = params["depletion_timesteps_days"]
            timesteps_days = original_ts[n_completed:]
            if len(timesteps_days) == 0:
                print("All original timesteps already completed — nothing to do.")
                return n_trisos
            print(f"Original timesteps ({len(original_ts)}): {original_ts}")
            print(f"Remaining timesteps ({len(timesteps_days)}): {timesteps_days}")

        reduced_chain_in_dir = os.path.join(restart_dir, "chain_reduced.xml")
        if os.path.exists(reduced_chain_in_dir):
            chain_file = reduced_chain_in_dir
            print(f"Using existing reduced chain: {chain_file}")
        else:
            full_chain_file = params["depletion_chain_file"]
            if full_chain_file is None or not os.path.exists(full_chain_file):
                raise FileNotFoundError(f"Depletion chain file not found: {full_chain_file}")
            chain_file = full_chain_file
            print(f"  Using full chain file: {chain_file}")

    # ==================================================================
    # FRESH RUN PATH
    # ==================================================================

    else:
        prev_results = None
        n_completed = 0

        model, n_trisos, m_colors, fuel_clones = build_model(params, run_dir)
        model.export_to_xml()

        if params.get("make_geometry_plots", False):
            n_plot_threads = str(params.get("plot_threads", os.cpu_count() or 4))
            old_omp = os.environ.get("OMP_NUM_THREADS")
            os.environ["OMP_NUM_THREADS"] = n_plot_threads
            try:
                openmc.plot_geometry(output=False, cwd=run_dir)
            finally:
                if old_omp is None:
                    os.environ.pop("OMP_NUM_THREADS", None)
                else:
                    os.environ["OMP_NUM_THREADS"] = old_omp

        geometry_factor       = 6 if params["use_1/6_geometry"] else 1
        poison_volume         = mats.b4c_poison.volume
        total_poison_volume   = poison_volume * geometry_factor

        # Deduplicate by material ID before summing
        # When use_spatial_burnup=False all rings share the same material object so we must not double-count it
        seen_ids = set()
        fuel_volume_simulated = 0.0
        for ri in range(len(fuel_clones)):
            for ai in range(len(fuel_clones[ri])):
                mat = fuel_clones[ri][ai]
                if mat.id not in seen_ids:
                    seen_ids.add(mat.id)
                    fuel_volume_simulated += mat.volume
        total_fuel_volume = fuel_volume_simulated * geometry_factor

        uco_density_g_cm3  = params["kernel_density"] / 1000.0
        u_mass_fraction    = 238.0 / 268.0
        total_HM_mass_kg   = total_fuel_volume * uco_density_g_cm3 * u_mass_fraction / 1000.0

        b4c_density_g_cm3  = params["B4C_density_poison"] / 1000.0
        b10_enrichment     = params["B10_enrichment_poison"]
        mass_10            = openmc.data.atomic_mass('B10')
        mass_11            = openmc.data.atomic_mass('B11')
        b10_mass_fraction  = (b10_enrichment * mass_10) / (
            b10_enrichment * mass_10 + (1.0 - b10_enrichment) * mass_11
        )
        total_B10_mass_kg  = total_poison_volume * b4c_density_g_cm3 * b10_mass_fraction / 1000.0

        print(f"\nFuel volume   (simulated geometry): {fuel_volume_simulated:.4f} cm³")
        print(f"Fuel volume   (full core):          {total_fuel_volume:.4f} cm³")
        print(f"Uranium mass  (full core):          {total_HM_mass_kg:.2f} kg")
        print(f"  ({len(params['core_rings'])} rings × {params['n_ax_zones']} axial zones, "
              f"{sum(len(r) for r in fuel_clones)} fuel material regions)")
        print(f"B4C poison    (simulated geometry): {poison_volume:.4f} cm³")
        print(f"B4C poison    (full core):          {total_poison_volume:.4f} cm³")
        print(f"B-10 mass     (full core):          {total_B10_mass_kg:.4f} kg")

        params["total_HM_mass_kg"]  = total_HM_mass_kg
        params["total_B10_mass_kg"] = total_B10_mass_kg

        params_path  = os.path.join(run_dir, 'run_params.json')
        saved_params = json.load(open(params_path)) if os.path.exists(params_path) else {}
        saved_params.update({
            'n_trisos':                    n_trisos,
            'use_homogenized_fuel':        params.get("use_homogenized_fuel", False),
            'use_spatial_burnup':          params.get("use_spatial_burnup", True),
            'poison_material_id':          mats.b4c_poison.id,
            'fuel_volume_simulated_cm3':   fuel_volume_simulated,
            'fuel_volume_full_core_cm3':   total_fuel_volume,
            'total_HM_mass_kg':            total_HM_mass_kg,
            'poison_volume_simulated_cm3': poison_volume,
            'poison_volume_full_core_cm3': total_poison_volume,
            'total_B10_mass_kg':           total_B10_mass_kg,
            'fuel_mat_volumes': {
                str(fuel_clones[ri][ai].id): fuel_clones[ri][ai].volume
                for ri in range(len(fuel_clones))
                for ai in range(len(fuel_clones[ri]))
            },
            'fuel_mat_ids': [
                [fuel_clones[ri][ai].id for ai in range(len(fuel_clones[ri]))]
                for ri in range(len(fuel_clones))
            ],
        })
        with open(params_path, 'w') as f:
            json.dump(saved_params, f, indent=2)

        # Configure depletion chain
        full_chain_file = params["depletion_chain_file"]
        if full_chain_file is None or not os.path.exists(full_chain_file):
            raise FileNotFoundError(f"Depletion chain file not found: {full_chain_file}")

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

        timesteps_days = params["depletion_timesteps_days"]

    # ==================================================================
    # CONFIGURE TIMESTEPS AND POWER
    # ==================================================================

    thermal_power_W = params["thermal_power_MW"] * 1e6

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

    integrator_name = params["depletion_integrator"]

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
        run_burnup_estimation(run_dir, merged_params)
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

    try:
        from heating_profile_extraction import run_heating_profile_extraction
        print("Running heating profile extraction...")
        run_heating_profile_extraction(run_dir, merged_params)
    except ImportError as e:
        print(f"Warning: Could not import heating_profile_extraction: {e}")
    except Exception as e:
        print(f"Warning: Heating profile extraction failed: {e}")

    print(f"{'='*80}")
    print("POST-PROCESSING COMPLETE")
    print(f"{'='*80}")

def run_depletion_post_processing(run_dir, params):
    """
    Run depletion-specific post-processing.
    """

    params_path = os.path.join(run_dir, 'run_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            saved_params = json.load(f)
        merged_params = {**params, **saved_params}
    else:
        merged_params = params
    
    try:
        from depletion_postprocessing import run_depletion_postprocessing
        print("Running depletion post-processing...")
        run_depletion_postprocessing(run_dir, merged_params)
    except ImportError as e:
        print(f"Warning: Could not import depletion_postprocessing: {e}")
    except Exception as e:
        print(f"Warning: Depletion post-processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    if params['use_BeO_reflector'] and params['BeO_thickness'] > 0.0:
        try:
            from BeO_depletion_postprocessing import run_BeO_depletion_postprocessing
            run_BeO_depletion_postprocessing(run_dir, merged_params)
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
    now = datetime.now()
    run_name = f"htgr_run_{now.strftime('%m.%d.%Y_%H.%M.%S')}"
    
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    OUTPUT_BASE = os.path.join(PARENT_DIR, "MicroHTGR_Output")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    BASE_DIR = os.path.join(OUTPUT_BASE, run_name)
    
    if cfg.params["study_execution_mode"] == "ParametricStudy":
        BASE_DIR = os.path.join(OUTPUT_BASE, run_name + "_ParametricStudy" + f"_{cfg.params['parametric_param']}")
        os.makedirs(BASE_DIR, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PARAMETRIC STUDY: {cfg.params['parametric_param']}")
        print(f"Values: {cfg.params['parametric_values']}")
        print(f"Base Directory: {BASE_DIR}")
        print(f"{'='*80}")
        
        for i, val in enumerate(cfg.params["parametric_values"]):
            caseNum = i + 1
            caseNumFormatted = f"{caseNum:0{len(str(len(cfg.params['parametric_values'])))+1}d}"
            runName = f"{cfg.params['parametric_param']}_Case_{caseNumFormatted}_{val}"
            run_dir = os.path.join(BASE_DIR, runName)

            print(f"\n{'='*80}")
            print(f"Running Case {caseNumFormatted}: {cfg.params['parametric_param']} = {val}")
            print(f"Run Directory: {run_dir}")
            print(f"{'='*80}")

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
    
    elif cfg.params["study_execution_mode"] == "RPTCalibration":
        BASE_DIR_RPT = os.path.join(OUTPUT_BASE, run_name + "_RPTCalibration")
        os.makedirs(BASE_DIR_RPT, exist_ok=True)

        run_rpt_calibration(
            params           = cfg.params,
            output_base_dir  = BASE_DIR_RPT,
            run_simulation_fn = run_simulation,
        )

        print(f"\n{'='*80}")
        print("RPT CALIBRATION COMPLETE")
        print(f"Results Directory: {BASE_DIR_RPT}")
        print(f"{'='*80}\n")

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

    elif cfg.params["study_execution_mode"] == "DepletionStudy":

        if cfg.params["restart_depletion"] and cfg.params["restart_run_dir"] is not None:
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

    # ----- Run Single Study -----   

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
        print("Valid modes: SingleStudy, ParametricStudy, ReactivityStudy, DepletionStudy, RPTCalibration")
        sys.exit(1)