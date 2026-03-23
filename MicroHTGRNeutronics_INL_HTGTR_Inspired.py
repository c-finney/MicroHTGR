import os
import math
import shutil
import openmc
import numpy as np
import openmc.deplete
from datetime import datetime
import h5py
import sys
import subprocess
import json
import glob
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to find modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import modules for config parameters, materials, TRISO creation, and assembly/core creation
import config as cfg
import materials as mats
import trisos
import assembly as asm

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
        if isinstance(v, (int, float, bool, str, list, dict)):
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

# ====================================================================================================
# MODEL BUILDING FUNCTION
# ====================================================================================================

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
        tuple: (model, n_trisos, m_colors, fuel_clones, poison_clones)
            - model:         openmc.model.Model ready to export or deplete
            - n_trisos:      Number of TRISO particles per axial zone
                             (0 when use_homogenized_fuel=True)
            - m_colors:      Material color dictionary for plotting
            - fuel_clones:   list[list[openmc.Material]] — fuel_clones[ring][bax]
            - poison_clones: list[list[openmc.Material]] — poison_clones[ring][bax]
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
        # Build one homogenized-compact universe per (ring, burnup band), parallelized

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
        # Explicit TRISO path — build one lattice per (ring, burnup band), parallelized

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
                    clone = fuel_clones[ring_idx][bax_idx]
                    clone.volume = actual_zones * V_inner_per_zone
                    if actual_zones * V_inner_per_zone == 0:
                        clone.depletable = False
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
            if total_volume == 0:
                fuel_clones[0][0].depletable = False

    else:
        if use_spatial_burnup:
            for ring_idx, ring_def in enumerate(params["core_rings"]):
                n_fuel_compacts = sum(_n_fuel_compacts_for_code(c) for c in ring_def)
                V_per_zone = (n_trisos * V_kernel * n_fuel_compacts) / geometry_factor
                for bax_idx in range(n_burnup_ax):
                    band_start   = bax_idx * zones_per_region
                    actual_zones = min(zones_per_region, n_ax - band_start)
                    clone = fuel_clones[ring_idx][bax_idx]
                    clone.volume = actual_zones * V_per_zone
                    if actual_zones * V_per_zone == 0:
                        clone.depletable = False
        else:
            total_volume = 0.0
            for ring_idx, ring_def in enumerate(params["core_rings"]):
                n_fuel_compacts = sum(_n_fuel_compacts_for_code(c) for c in ring_def)
                V_per_zone = (n_trisos * V_kernel * n_fuel_compacts) / geometry_factor
                actual_zones = min(zones_per_region, n_ax)
                total_volume += actual_zones * V_per_zone
            fuel_clones[0][0].volume = total_volume
            if total_volume == 0:
                fuel_clones[0][0].depletable = False

    # ==================================================================
    # CREATE BURNABLE POISON MATERIAL CLONES
    # ==================================================================
    # Mirrors the fuel_clones logic above: one B4C_Poison clone per
    # (core ring × burnup band) so each radial/axial zone depletes
    # independently.

    if use_spatial_burnup:
        poison_clones = []
        for ring_idx in range(n_rings):
            ring_poisons = []
            for bax_idx in range(n_burnup_ax):
                p = mats.b4c_poison.clone()
                p.name = f"B4C_Poison_ring{ring_idx}_bax{bax_idx}"
                p.depletable = True
                ring_poisons.append(p)
            poison_clones.append(ring_poisons)
    else:
        # Non-spatial: single shared poison material for the whole core
        p_single = mats.b4c_poison.clone()
        p_single.name = "B4C_Poison_global"
        p_single.depletable = True
        poison_clones = [[p_single] for _ in range(n_rings)]

    # Expand to full axial-zone mapping (same pattern as ring_triso_lattices)
    ring_poison_mats = {
        ring_idx: {
            ax_idx: poison_clones[ring_idx][ax_idx // zones_per_region]
            for ax_idx in range(n_ax)
        }
        for ring_idx in range(n_rings)
    }

    # Set poison compact volumes analytically (per-clone, per-band)
    def _n_poison_compacts_for_code(code):
        if not code.startswith('f'):
            return 0
        if code == 'fpa':
            return 1
        if 'p' in code:
            return 6
        return 0

    V_poison_zone = np.pi * params["compact_radius"]**2 * axial_section_height

    if use_spatial_burnup:
        for ring_idx, ring_def in enumerate(params["core_rings"]):
            n_pois = sum(_n_poison_compacts_for_code(c) for c in ring_def)
            V_per_zone = n_pois * V_poison_zone / geometry_factor
            for bax_idx in range(n_burnup_ax):
                band_start   = bax_idx * zones_per_region
                actual_zones = min(zones_per_region, n_ax - band_start)
                clone = poison_clones[ring_idx][bax_idx]
                clone.volume = actual_zones * V_per_zone
                # Rings with no poison compacts get volume=0 — mark as non-depletable
                # so OpenMC does not try to track them (avoids divide-by-zero crash)
                if actual_zones * V_per_zone == 0:
                    clone.depletable = False
    else:
        total_pois_vol = 0.0
        for ring_def in params["core_rings"]:
            n_pois = sum(_n_poison_compacts_for_code(c) for c in ring_def)
            V_poison_column = np.pi * params["compact_radius"]**2 * params["core_height"]
            total_pois_vol += n_pois * V_poison_column / geometry_factor
        poison_clones[0][0].volume = total_pois_vol
        if total_pois_vol == 0:
            poison_clones[0][0].depletable = False

    # ==================================================================
    # CREATE ASSEMBLIES
    # ==================================================================

    assemblies, m_colors, bundle_pitch = asm.create_assembly_univs(
        params              = params,
        mats                = mats,
        T_coolant_z         = T_coolant_z,
        T_compact_z         = T_compact_z,
        T_matrix_z          = T_matrix_z,
        T_reflector_z       = T_reflector_z,
        ring_triso_lattices = ring_triso_lattices,
        ring_poison_mats    = ring_poison_mats,
        axial_coords        = axial_coords,
        reactor_bottom      = reactor_bottom,
        reactor_top         = reactor_top
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

    # Assign plotting colors (skip zero-volume clones — not in materials.xml)
    for ring_fuels in fuel_clones:
        for f in ring_fuels:
            if f.volume:
                m_colors[f] = 'palegreen'
    m_colors[mats.buffer] = 'sandybrown'
    m_colors[mats.pyc] = 'orange'
    m_colors[mats.sic] = 'yellow'
    m_colors[mats.graphite] = 'darkblue'
    for ring_pois in poison_clones:
        for pmat in ring_pois:
            if pmat.volume:
                m_colors[pmat] = 'purple'
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

    # ----- Zone Heating Tally (per-zone burnup via heating-local) -----
    if use_spatial_burnup:
        _zone_mats = [fc for ring in fuel_clones for fc in ring if fc.volume]
        _zone_mat_filter = openmc.MaterialFilter(_zone_mats)
        zone_heating_tally = openmc.Tally(name="zone_heating_local")
        zone_heating_tally.filters = [_zone_mat_filter]
        zone_heating_tally.scores = ["heating-local"]
        tallies.append(zone_heating_tally)
        print(f"\nZone heating tally added: {len(_zone_mats)} fuel zone materials")

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

    # Exclude the base fuel and poison templates — only the depletable clones should be in the model
    explicit_mats = [m for m in mats.materials if m is not mats.fuel and m is not mats.b4c_poison]
    for ring_fuels in fuel_clones:
        explicit_mats.extend(f for f in ring_fuels if f.volume)
    for ring_poisons in poison_clones:
        explicit_mats.extend(p for p in ring_poisons if p.volume)

    # Homogenized fuel uses temporary constituent materials internally via
    # mix_materials(); those are not tracked separately and do not need to
    # be added here — mix_materials returns a single combined material.
    model.materials = openmc.Materials(explicit_mats)

    for plot in model.plots:
        if plot.color_by == 'material':
            plot.colors = m_colors

    return model, n_trisos, m_colors, fuel_clones, poison_clones

# ====================================================================================================
# RPT CALIBRATION HELPERS
# ====================================================================================================

def _extract_keff(run_dir):
    """Read the Combined k-effective and its std_dev from the statepoint in run_dir.

    sp.keff reads the 'k_combined' HDF5 dataset — the combined (analog +
    track-length + collision) estimator, identical to 'Combined k-effective'
    in OpenMC terminal output.

    Returns:
        (k_eff_mean, k_eff_std): floats
    """
    sp_files = sorted(glob.glob(os.path.join(run_dir, "statepoint.*.h5")))
    if not sp_files:
        raise FileNotFoundError(f"No statepoint file found in {run_dir}")
    sp = openmc.StatePoint(sp_files[-1])
    # sp.keff == sp.k_combined (the 'k_combined' HDF5 dataset)
    return float(sp.keff.n), float(sp.keff.s)


def run_rpt_calibration(params, output_base_dir, run_simulation_fn):
    """
    Find the RPT inner radius (rpt_radius) that matches explicit-TRISO k_eff.

    Algorithm
    ---------
    1. Run an explicit-TRISO reference simulation (use_homogenized_fuel=False)
       to obtain k_eff_ref.
    2. Establish an initial bracket by running both endpoints:
         r_lo = compact_radius * sqrt(triso_pf)   (maximum self-shielding → highest k_eff)
         r_hi = compact_radius                    (flat homogenization   → lowest  k_eff)
    3. Iterate with regula falsi + Illinois anti-stagnation until
       |k_eff - k_eff_ref| < rpt_calibration_k_tol or rpt_calibration_max_iter is reached.
    4. Save results to rpt_calibration_results.json and print a summary table.

    k_eff is monotonically DECREASING as r_rpt increases (r_lo → r_hi), so a
    single bracket is guaranteed to exist if k_eff_ref lies within the RPT
    model's achievable range.

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
    r_lo      = r_compact * pf**0.5         # lower bound (pf_inner = 1, highest k_eff)
    r_hi      = r_compact                    # upper bound (flat homogenization, lowest k_eff)
    k_tol     = params.get("rpt_calibration_k_tol",   0.005)
    max_iter  = params.get("rpt_calibration_max_iter", 20)

    print(f"\n{'='*80}")
    print("RPT CALIBRATION STUDY")
    print(f"  compact_radius  = {r_compact:.4f} cm")
    print(f"  triso_pf        = {pf:.3f}")
    print(f"  r_rpt bracket   = [{r_lo:.4f}, {r_hi:.4f}] cm")
    print(f"  k_tol           = {k_tol}")
    print(f"  max_iter        = {max_iter}")
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
    # Step 2: initial bracket runs at r_lo and r_hi
    # ------------------------------------------------------------------
    rpt_results = []
    case_counter = [0]

    def _run_rpt_case(r_rpt, label):
        case_counter[0] += 1
        pf_inner = pf * (r_compact / r_rpt)**2
        print(f"\n--- RPT Case {case_counter[0]} ({label}): r_rpt = {r_rpt:.4f} cm  "
              f"(pf_inner = {pf_inner:.4f}) ---")
        rpt_p = copy.copy(params)
        rpt_p["use_homogenized_fuel"] = True
        rpt_p["rpt_radius"]           = r_rpt
        rpt_p["make_geometry_plots"]  = False
        case_dir = os.path.join(output_base_dir,
                                f"rpt_case_{case_counter[0]:02d}_r{r_rpt:.4f}cm")
        run_simulation_fn(rpt_p, case_dir)
        k, k_std = _extract_keff(case_dir)
        dk_pcm = (k - k_eff_ref) * 1e5
        print(f"  k_eff = {k:.5f} ± {k_std:.5f}  (Δk = {dk_pcm:+.0f} pcm vs. reference)")
        rpt_results.append({
            "r_rpt":       float(r_rpt),
            "pf_inner":    float(pf_inner),
            "k_eff":       float(k),
            "k_eff_std":   float(k_std),
            "delta_k_pcm": float(dk_pcm),
        })
        return k, k_std

    print(f"\n--- Step 2: Initial bracket runs ---")
    k_lo, _ = _run_rpt_case(r_lo, "lower bound")
    k_hi, _ = _run_rpt_case(r_hi, "upper bound")

    optimal_r_rpt = None
    converged     = False

    # Verify k_eff_ref lies within the achievable range
    if (k_lo - k_eff_ref) * (k_hi - k_eff_ref) >= 0:
        best = min(rpt_results, key=lambda r: abs(r["k_eff"] - k_eff_ref))
        optimal_r_rpt = best["r_rpt"]
        print(f"\nWARNING: k_eff_ref = {k_eff_ref:.5f} is outside the RPT model's "
              f"achievable range [{k_hi:.5f}, {k_lo:.5f}].")
        print(f"  Using nearest endpoint r_rpt = {optimal_r_rpt:.4f} cm as best estimate.")
        print(f"  Check whether the explicit-TRISO k_eff lies within the RPT model's achievable range.")
    else:
        # ------------------------------------------------------------------
        # Step 3: Illinois regula falsi iteration
        #
        # Bracket invariant: k(r_lo_b) > k_eff_ref > k(r_hi_b)
        # ------------------------------------------------------------------
        r_lo_b, k_lo_b = r_lo, k_lo
        r_hi_b, k_hi_b = r_hi, k_hi
        last_side       = None
        same_side_count = 0

        print(f"\n--- Step 3: Illinois interpolation search "
              f"(k_tol = {k_tol}, max_iter = {max_iter}) ---")

        for _ in range(max_iter):
            dk = k_lo_b - k_hi_b
            if abs(dk) < 1e-9:
                r_mid = 0.5 * (r_lo_b + r_hi_b)
            else:
                # Illinois: halve stale endpoint's residual after two same-side hits
                k_lo_eff = k_lo_b
                k_hi_eff = k_hi_b
                if same_side_count >= 2:
                    if last_side == "lo":   # new pts keep landing on lo side → stale hi
                        k_hi_eff = k_eff_ref + 0.5 * (k_hi_b - k_eff_ref)
                    else:                   # new pts keep landing on hi side → stale lo
                        k_lo_eff = k_eff_ref + 0.5 * (k_lo_b - k_eff_ref)
                r_mid = r_lo_b + (k_eff_ref - k_lo_eff) / (k_hi_eff - k_lo_eff) * (r_hi_b - r_lo_b)
                # Safety clamp against noisy MC k_eff pushing interpolation outside bracket
                margin = 0.02 * (r_hi_b - r_lo_b)
                r_mid  = max(r_lo_b + margin, min(r_hi_b - margin, r_mid))

            k_mid, _ = _run_rpt_case(r_mid, "iter")

            if abs(k_mid - k_eff_ref) < k_tol:
                optimal_r_rpt = r_mid
                converged     = True
                print(f"\n  >>> Converged: |Δk| = {abs(k_mid - k_eff_ref):.5f} < {k_tol}")
                break

            # Update bracket and track same-side count for Illinois
            if (k_mid - k_eff_ref) > 0:
                new_side = "lo"
                r_lo_b, k_lo_b = r_mid, k_mid
            else:
                new_side = "hi"
                r_hi_b, k_hi_b = r_mid, k_mid

            same_side_count = same_side_count + 1 if new_side == last_side else 1
            last_side = new_side

        if not converged:
            best = min(rpt_results, key=lambda r: abs(r["k_eff"] - k_eff_ref))
            optimal_r_rpt = best["r_rpt"]
            print(f"\nWARNING: Did not converge within {max_iter} iterations.")
            print(f"  Best estimate: r_rpt = {optimal_r_rpt:.4f} cm  "
                  f"(Δk = {best['delta_k_pcm']:+.0f} pcm)")

    # ------------------------------------------------------------------
    # Step 4: save results and print summary
    # ------------------------------------------------------------------
    optimal_result  = next((r for r in rpt_results if r["r_rpt"] == optimal_r_rpt), None)
    optimal_delta_k = optimal_result["delta_k_pcm"] if optimal_result else None

    calibration_results = {
        "reference_k_eff":         k_eff_ref,
        "reference_k_eff_std":     k_eff_ref_std,
        "r_compact":               r_compact,
        "triso_pf":                pf,
        "r_rpt_min":               float(r_lo),
        "r_rpt_max":               float(r_hi),
        "k_tol":                   k_tol,
        "max_iter":                max_iter,
        "converged":               converged,
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
    print(f"\n  >>> Estimated optimal r_rpt = {optimal_r_rpt:.4f} cm  (converged: {converged})")
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

    model, n_trisos, m_colors, fuel_clones, poison_clones = build_model(params, run_dir)
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

    geometry_factor = 6 if params["use_1/6_geometry"] else 1

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

    seen_pids = set()
    poison_volume_simulated = 0.0
    for ri in range(len(poison_clones)):
        for ai in range(len(poison_clones[ri])):
            mat = poison_clones[ri][ai]
            if mat.id not in seen_pids:
                seen_pids.add(mat.id)
                poison_volume_simulated += mat.volume
    total_poison_volume = poison_volume_simulated * geometry_factor

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
    print(f"B4C poison    (simulated geometry): {poison_volume_simulated:.4f} cm³")
    print(f"B4C poison    (full core):          {total_poison_volume:.4f} cm³")
    print(f"B-10 mass     (full core):          {total_B10_mass_kg:.4f} kg")

    params_path = os.path.join(run_dir, 'run_params.json')
    saved_params = json.load(open(params_path)) if os.path.exists(params_path) else {}
    saved_params.update({
        'n_trisos':                    n_trisos,
        'use_homogenized_fuel':        params.get("use_homogenized_fuel", False),
        'use_spatial_burnup':          params.get("use_spatial_burnup", True),
        'fuel_volume_simulated_cm3':   fuel_volume_simulated,
        'fuel_volume_full_core_cm3':   total_fuel_volume,
        'total_HM_mass_kg':            total_HM_mass_kg,
        'poison_volume_simulated_cm3': poison_volume_simulated,
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
        'poison_mat_volumes': {
            str(poison_clones[ri][ai].id): poison_clones[ri][ai].volume
            for ri in range(len(poison_clones))
            for ai in range(len(poison_clones[ri]))
        },
        'poison_mat_ids': [
            [poison_clones[ri][ai].id for ai in range(len(poison_clones[ri]))]
            for ri in range(len(poison_clones))
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
# ZONE HEATING HELPER (per-zone burnup from heating-local tally)
# ====================================================================================================

def _read_zone_heating_entry(run_dir, fuel_mat_ids_2d):
    """
    Read the latest statepoint in run_dir and return a per-zone heating dict.

    Returns a dict:
        {
          "H_zones": {"ring_bax": float, ...},  # eV/src per zone
          "H_total": float                       # eV/src total model heating (from 'heating' tally)
        }
    or None on any failure.

    H_total uses the global 'heating' tally (no material filter) so the zone
    fractions account for gamma heating deposited in moderator/reflector.
    Falls back to sum of zone values if the global tally is unavailable.
    """
    sp_files = sorted(glob.glob(os.path.join(run_dir, "statepoint.*.h5")))
    if not sp_files:
        return None
    try:
        sp = openmc.StatePoint(sp_files[-1])
        zone_tally = sp.get_tally(name="zone_heating_local")

        # Read material filter bins and mean values directly from numpy arrays
        # to avoid get_pandas_dataframe() failures on some OpenMC versions.
        mat_filter = next(
            f for f in zone_tally.filters
            if isinstance(f, openmc.MaterialFilter)
        )
        # bins may be Material objects or plain ints depending on OpenMC version
        mat_bins = [m.id if hasattr(m, 'id') else int(m) for m in mat_filter.bins]
        # zone_tally.mean shape: (n_materials, n_nuclides, n_scores)
        means = zone_tally.mean[:, 0, 0]   # scalar per material bin

        # Global heating (no filter) for normalisation denominator
        try:
            h_tally = sp.get_tally(name="heating")
            H_total = float(h_tally.mean.flat[0])
        except Exception:
            H_total = None

        H_zones   = {}
        H_zone_sum = 0.0
        mat_id_to_idx = {mid: i for i, mid in enumerate(mat_bins)}
        for ring_idx, row in enumerate(fuel_mat_ids_2d):
            for bax_idx, mat_id in enumerate(row):
                key = f"{ring_idx}_{bax_idx}"
                idx = mat_id_to_idx.get(mat_id)
                H_z = float(means[idx]) if idx is not None else 0.0
                H_zones[key] = H_z
                H_zone_sum  += H_z

        if H_total is None or H_total <= 0:
            H_total = H_zone_sum

        return {"H_zones": H_zones, "H_total": H_total}

    except Exception as e:
        print(f"  WARNING: Could not read zone heating from statepoint: {e}")
        return None


def _append_zone_heating_step(run_dir, fuel_mat_ids_2d,
                               step_idx, step_start_days, step_end_days):
    """
    Read zone heating from the latest statepoint and append an entry to
    zone_heating_per_step.json in run_dir.

    Called once per depletion step (CSDepletionStudy) or once after all
    steps (DepletionStudy) immediately after integrator.integrate().
    """
    heating_file = os.path.join(run_dir, "zone_heating_per_step.json")

    entry = _read_zone_heating_entry(run_dir, fuel_mat_ids_2d)
    if entry is None:
        print("  WARNING: zone heating not recorded for this step (no statepoint data).")
        return

    record = {
        "step_idx":        step_idx,
        "step_start_days": step_start_days,
        "step_end_days":   step_end_days,
        "dt_days":         step_end_days - step_start_days,
        "H_total":         entry["H_total"],
        "H_zones":         entry["H_zones"],
    }

    if os.path.exists(heating_file):
        with open(heating_file) as f:
            existing = json.load(f)
    else:
        existing = []

    existing.append(record)
    with open(heating_file, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Zone heating recorded → zone_heating_per_step.json  "
          f"(step {step_idx}, dt={record['dt_days']:.1f} d, "
          f"H_total={entry['H_total']:.4e})")


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

        n_trisos = saved_params["n_trisos"]

        fuel_mat_volumes = {
            int(k): v for k, v in saved_params.get("fuel_mat_volumes", {}).items()
        }
        poison_mat_volumes = {
            int(k): v for k, v in saved_params.get("poison_mat_volumes", {}).items()
        }

        graphite_mat_id    = saved_params.get("graphite_material_id")
        graphite_vol_saved = saved_params.get("graphite_volume_simulated_cm3")

        n_fuel_vols_set   = 0
        n_poison_vols_set = 0
        for mat in materials:
            if mat.id in fuel_mat_volumes:
                mat.volume = fuel_mat_volumes[mat.id]
                n_fuel_vols_set += 1
            elif mat.id in poison_mat_volumes:
                mat.volume = poison_mat_volumes[mat.id]
                n_poison_vols_set += 1
            elif graphite_mat_id is not None and mat.id == graphite_mat_id and graphite_vol_saved is not None:
                mat.volume = graphite_vol_saved
                print(f"Graphite material (id={mat.id}): volume = {graphite_vol_saved:.4f} cm³")
        print(f"Set volumes on {n_fuel_vols_set} fuel material clones from saved run_params.json")
        print(f"Set volumes on {n_poison_vols_set} poison material clones from saved run_params.json")

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

        model, n_trisos, m_colors, fuel_clones, poison_clones = build_model(params, run_dir)
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

        geometry_factor = 6 if params["use_1/6_geometry"] else 1

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

        seen_pids = set()
        poison_volume_simulated = 0.0
        for ri in range(len(poison_clones)):
            for ai in range(len(poison_clones[ri])):
                mat = poison_clones[ri][ai]
                if mat.id not in seen_pids:
                    seen_pids.add(mat.id)
                    poison_volume_simulated += mat.volume
        total_poison_volume = poison_volume_simulated * geometry_factor

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

        # --- Stochastic volume calculation for graphite depletion ---
        # This calculation is expensive and must only run once per study.
        # If a previously computed volume is already saved in run_params.json
        # (e.g. after a partial run or restart), load it directly.
        graphite_vol_simulated = None
        graphite_vol_full      = None
        if params.get("deplete_graphite", False):
            _saved_g_vol = None
            _rp_path_check = os.path.join(run_dir, "run_params.json")
            if os.path.exists(_rp_path_check):
                _rp_check = json.load(open(_rp_path_check))
                _saved_g_vol = _rp_check.get("graphite_volume_simulated_cm3")

            if _saved_g_vol is not None:
                graphite_vol_simulated = float(_saved_g_vol)
                graphite_vol_full      = graphite_vol_simulated * geometry_factor
                mats.graphite.volume   = graphite_vol_simulated
                print(f"\nGraphite volume loaded from run_params.json (skipping stochastic calc): "
                      f"{graphite_vol_simulated:.4f} cm³")
            else:
                n_vol_particles = params.get("graphite_volume_particles", 1_000_000)
                core_r   = params["core_radius"]
                refl_t   = params["reflector_thickness"]
                core_h   = params["core_height"]
                lower_left  = [-core_r, -core_r, -refl_t]
                upper_right = [ core_r,  core_r,  core_h + refl_t]

                print(f"\nRunning stochastic volume calculation for graphite "
                      f"({n_vol_particles:,} particles)...")
                vol_calc = openmc.VolumeCalculation(
                    [mats.graphite], n_vol_particles, lower_left, upper_right
                )
                model.settings.volume_calculations = [vol_calc]
                model.settings.export_to_xml()
                openmc.calculate_volumes(output=True, cwd=run_dir)

                vol_h5_path = os.path.join(run_dir, "volume_1.h5")
                with h5py.File(vol_h5_path, 'r') as vf:
                    mat_id_str = str(mats.graphite.id)
                    mat_key = next(
                        (k for k in vf.keys() if mat_id_str in k),
                        None
                    )
                    if mat_key is None:
                        raise KeyError(
                            f"Graphite material (id={mats.graphite.id}) not found in {vol_h5_path}. "
                            f"Available keys: {list(vf.keys())}"
                        )
                    graphite_vol_simulated = float(vf[mat_key]['volume'][0])
                graphite_vol_full = graphite_vol_simulated * geometry_factor
                mats.graphite.volume = graphite_vol_simulated
                model.materials.export_to_xml()

                # Clear volume_calculations so they don't re-run during depletion
                model.settings.volume_calculations = []
                model.settings.export_to_xml()

                print(f"Graphite volume (simulated geometry): {graphite_vol_simulated:.4f} cm³")
                print(f"Graphite volume (full core):          {graphite_vol_full:.4f} cm³")

        print(f"\nFuel volume   (simulated geometry): {fuel_volume_simulated:.4f} cm³")
        print(f"Fuel volume   (full core):          {total_fuel_volume:.4f} cm³")
        print(f"Uranium mass  (full core):          {total_HM_mass_kg:.2f} kg")
        print(f"  ({len(params['core_rings'])} rings × {params['n_ax_zones']} axial zones, "
              f"{sum(len(r) for r in fuel_clones)} fuel material regions)")
        print(f"B4C poison    (simulated geometry): {poison_volume_simulated:.4f} cm³")
        print(f"B4C poison    (full core):          {total_poison_volume:.4f} cm³")
        print(f"  ({len(poison_clones)} rings × {len(poison_clones[0])} burnup bands, "
              f"{sum(len(r) for r in poison_clones)} poison material regions)")
        print(f"B-10 mass     (full core):          {total_B10_mass_kg:.4f} kg")

        params["total_HM_mass_kg"]  = total_HM_mass_kg
        params["total_B10_mass_kg"] = total_B10_mass_kg

        params_path  = os.path.join(run_dir, 'run_params.json')
        saved_params = json.load(open(params_path)) if os.path.exists(params_path) else {}
        saved_params.update({
            'n_trisos':                    n_trisos,
            'use_homogenized_fuel':        params.get("use_homogenized_fuel", False),
            'use_spatial_burnup':          params.get("use_spatial_burnup", True),
            'fuel_volume_simulated_cm3':   fuel_volume_simulated,
            'fuel_volume_full_core_cm3':   total_fuel_volume,
            'total_HM_mass_kg':            total_HM_mass_kg,
            'poison_volume_simulated_cm3': poison_volume_simulated,
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
            'poison_mat_volumes': {
                str(poison_clones[ri][ai].id): poison_clones[ri][ai].volume
                for ri in range(len(poison_clones))
                for ai in range(len(poison_clones[ri]))
            },
            'poison_mat_ids': [
                [poison_clones[ri][ai].id for ai in range(len(poison_clones[ri]))]
                for ri in range(len(poison_clones))
            ],
        })
        if params.get("deplete_graphite", False) and graphite_vol_simulated is not None:
            saved_params['graphite_material_id']          = mats.graphite.id
            saved_params['graphite_volume_simulated_cm3'] = graphite_vol_simulated
            saved_params['graphite_volume_full_core_cm3'] = graphite_vol_full
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

    # Record zone heating fractions from the final statepoint.
    # For multi-step DepletionStudy only the last step's statepoint survives;
    # the postprocessing will apply those fractions uniformly across all steps.
    if params.get("use_spatial_burnup", True):
        if is_restart:
            _fuel_ids_2d = saved_params.get("fuel_mat_ids", None)
        else:
            _fuel_ids_2d = [
                [fuel_clones[ri][ai].id for ai in range(len(fuel_clones[ri]))]
                for ri in range(len(fuel_clones))
            ]
        if _fuel_ids_2d is not None:
            _total_days = sum(timesteps_days)
            _append_zone_heating_step(
                run_dir         = run_dir,
                fuel_mat_ids_2d = _fuel_ids_2d,
                step_idx        = len(timesteps_days) - 1,
                step_start_days = _total_days - timesteps_days[-1],
                step_end_days   = _total_days,
            )

    print(f"\n{'=' * 80}")
    print("DEPLETION CALCULATION COMPLETE")
    print(f"{'=' * 80}\n")

    return n_trisos

# ====================================================================================================
# DEPLETION SIMULATION FUNCTION
# ====================================================================================================

def run_critical_search_depletion_simulation(params, run_dir):
    """
    Build model and run a coupled depletion simulation where a criticality
    search is performed before every depletion timestep.

    Workflow
    --------
    For each timestep i in params["depletion_timesteps_days"]:

      1.  Load depleted materials from the previous step's depletion_results.h5
          (or use fresh fuel for step 0).
      2.  Build a fresh model with those materials injected.
      3.  Run a two-stage binary search (bank 1 then bank 2) to find the
          critical rod insertion fraction for the current burnup state.
      4.  Rebuild the model again at the critical rod position.
      5.  Run a single-timestep CoupledOperator depletion with prev_results
          pointing at the accumulated depletion_results.h5, so all timestep
          results are chained into one file.
      6.  Save a per-step JSON log: critical insertion, k_eff, burnup time.

    The function honours the same reduced-chain and power-scaling logic used
    by run_depletion_simulation() so it is a drop-in replacement for the
    "CSDepletion" study_execution_mode.

    Parameters
    ----------
    params : dict
        Simulation parameter dictionary (from config.py).  Relevant keys:

        depletion_timesteps_days : list[float]
            Timestep durations (days).  One criticality search per timestep.

        thermal_power_MW : float
        use_1/6_geometry : bool
        depletion_chain_file : str
        use_reduced_chain_file : bool
        tracked_nuclides : list[str]
        depletion_integrator : str
            Name of the OpenMC integrator class to use.

        critical_search_k_tol : float   (default 0.003)
        critical_search_max_iter : int  (default 20)

        Critical-search particle settings are taken from the dedicated
        sub-keys if present, otherwise fall back to sensible defaults:
        critical_search_particles    (default: params["particles"] // 2)
        critical_search_batches      (default: 100)
        critical_search_inactive     (default: 40)

    run_dir : str
        Root directory for this simulation run.  Sub-directories are created
        for each timestep's criticality search.

    Returns
    -------
    n_trisos : int
        Number of TRISO particles per axial zone (0 for homogenised fuel).
    """
    import copy
    import glob

    # ------------------------------------------------------------------
    # Import the criticality-search helper from mol_eol_analysis.py.
    # Both files live in the same SCRIPT_DIR so a direct import works.
    # ------------------------------------------------------------------
    try:
        from mol_eol_analysis import (
            find_critical_rod_insertion,
            reconstruct_depleted_materials,
            _inject_depleted_materials,
        )
    except ImportError as exc:
        raise ImportError(
            "run_critical_search_depletion_simulation requires mol_eol_analysis.py "
            "to be importable from SCRIPT_DIR.  Check your path setup."
        ) from exc

    print(f"\n{'=' * 80}")
    print("CRITICAL SEARCH DEPLETION SIMULATION")
    print(f"{'=' * 80}")

    os.makedirs(run_dir, exist_ok=True)

    timesteps_days  = params["depletion_timesteps_days"]
    n_steps         = len(timesteps_days)
    thermal_power_W = params["thermal_power_MW"] * 1e6

    if params["use_1/6_geometry"]:
        operator_power_W = thermal_power_W / 6.0
        print(f"1/6 geometry: scaling power {thermal_power_W/1e6:.1f} MW → "
              f"{operator_power_W/1e6:.4f} MW (simulated geometry)")
    else:
        operator_power_W = thermal_power_W

    # ------------------------------------------------------------------
    # Build / load the reduced depletion chain once (reused every step)
    # ------------------------------------------------------------------
    full_chain_file = params["depletion_chain_file"]
    if full_chain_file is None or not os.path.exists(full_chain_file):
        raise FileNotFoundError(f"Depletion chain file not found: {full_chain_file}")

    reduced_chain_file = os.path.join(run_dir, "chain_reduced.xml")

    if params["use_reduced_chain_file"] and len(params.get("tracked_nuclides", [])) > 0:
        chain_file = build_reduced_chain(
            full_chain_file    = full_chain_file,
            reduced_chain_file = reduced_chain_file,
            tracked_nuclides   = params["tracked_nuclides"],
        )
    else:
        print("\nUsing full depletion chain file.")
        chain_file = full_chain_file

    # ------------------------------------------------------------------
    # Integrator selection
    # ------------------------------------------------------------------
    integrator_map = {
        "PredictorIntegrator":  openmc.deplete.PredictorIntegrator,
        "CECMIntegrator":       openmc.deplete.CECMIntegrator,
        "CF4Integrator":        openmc.deplete.CF4Integrator,
        "EPCRK4Integrator":     openmc.deplete.EPCRK4Integrator,
        "LEQIIntegrator":       openmc.deplete.LEQIIntegrator,
        "SICELIIntegrator":     openmc.deplete.SICELIIntegrator,
        "SILEQIIntegrator":     openmc.deplete.SILEQIIntegrator,
    }
    IntegratorClass = integrator_map.get(
        params["depletion_integrator"],
        openmc.deplete.PredictorIntegrator,
    )
    print(f"Integrator: {IntegratorClass.__name__}")

    # ------------------------------------------------------------------
    # Criticality-search particle settings
    # ------------------------------------------------------------------
    k_tol       = params.get("critical_search_k_tol",     0.003)
    max_iter    = params.get("critical_search_max_iter",   20)
    cs_particles = params.get("critical_search_particles",
                              max(50_000, params.get("particles", 100_000) // 2))
    cs_batches  = params.get("critical_search_batches",   50)
    cs_inactive = params.get("critical_search_inactive",  25)

    print(f"\nCriticality search settings:")
    print(f"  k tolerance  : {k_tol}")
    print(f"  Max iters    : {max_iter}")
    print(f"  Particles    : {cs_particles:,}")
    print(f"  Batches      : {cs_batches} ({cs_inactive} inactive)")

    # ------------------------------------------------------------------
    # Per-step log — accumulated and saved after every step
    # ------------------------------------------------------------------
    step_log_path = os.path.join(run_dir, "critical_search_depletion_log.json")
    step_log: list[dict] = []

    # Accumulated depletion results HDF5 — built up step by step
    depletion_h5 = os.path.join(run_dir, "depletion_results.h5")

    # We need n_trisos from the first build_model call for the return value
    n_trisos_global = 0

    # Cumulative time (days) for logging
    cumulative_days = 0.0

    # Critical search result from the previous timestep (used for warm-start)
    prev_crit_result = None

    # ================================================================
    # STEP 0: Build the initial model at fresh (BOL) conditions and
    #         record material volumes / IDs in run_params.json.
    #         This also writes the initial XML files to run_dir.
    # ================================================================
    print(f"\n{'─' * 70}")
    print(f"  BOL model build (fresh fuel, step 0 / {n_steps})")
    print(f"{'─' * 70}")

    # Use rods fully withdrawn for the first build so we know the base
    # geometry; the critical search will update the insertion each step.
    bol_params = copy.deepcopy(params)
    bol_params["bank_1_insertion"] = 0.0
    bol_params["bank_2_insertion"] = 0.0
    bol_params["bank_3_insertion"] = 0.0
    bol_params["make_geometry_plots"] = params.get("make_geometry_plots", False)

    model_bol, n_trisos_global, m_colors, fuel_clones_bol, poison_clones_bol = build_model(
        bol_params, run_dir
    )

    # Compute and save volumes / masses (mirrors run_depletion_simulation logic)
    geometry_factor = 6 if params["use_1/6_geometry"] else 1

    seen_ids = set()
    fuel_volume_simulated = 0.0
    for ri in range(len(fuel_clones_bol)):
        for ai in range(len(fuel_clones_bol[ri])):
            mat = fuel_clones_bol[ri][ai]
            if mat.id not in seen_ids:
                seen_ids.add(mat.id)
                fuel_volume_simulated += mat.volume
    total_fuel_volume = fuel_volume_simulated * geometry_factor

    seen_pids = set()
    poison_volume_simulated = 0.0
    for ri in range(len(poison_clones_bol)):
        for ai in range(len(poison_clones_bol[ri])):
            mat = poison_clones_bol[ri][ai]
            if mat.id not in seen_pids:
                seen_pids.add(mat.id)
                poison_volume_simulated += mat.volume
    total_poison_volume = poison_volume_simulated * geometry_factor

    uco_density_g_cm3 = params["kernel_density"] / 1000.0
    u_mass_fraction   = 238.0 / 268.0
    total_HM_mass_kg  = total_fuel_volume * uco_density_g_cm3 * u_mass_fraction / 1000.0

    b4c_density_g_cm3 = params["B4C_density_poison"] / 1000.0
    b10_enrichment    = params["B10_enrichment_poison"]
    mass_10           = openmc.data.atomic_mass('B10')
    mass_11           = openmc.data.atomic_mass('B11')
    b10_mass_fraction = (b10_enrichment * mass_10) / (
        b10_enrichment * mass_10 + (1.0 - b10_enrichment) * mass_11
    )
    total_B10_mass_kg = total_poison_volume * b4c_density_g_cm3 * b10_mass_fraction / 1000.0

    # Export XML now so materials.xml / geometry.xml / settings.xml exist on
    # disk before calculate_volumes (and geometry plots) need them.
    model_bol.export_to_xml()

    # --- Stochastic volume calculation for graphite depletion ---
    # Only runs once: if graphite_volume_simulated_cm3 is already in
    # run_params.json (written by a prior run), load it and skip the
    # expensive stochastic calculation.
    graphite_vol_simulated = None
    graphite_vol_full      = None
    if params.get("deplete_graphite", False):
        _saved_g_vol = None
        _rp_path_check = os.path.join(run_dir, "run_params.json")
        if os.path.exists(_rp_path_check):
            _rp_check = json.load(open(_rp_path_check))
            _saved_g_vol = _rp_check.get("graphite_volume_simulated_cm3")

        if _saved_g_vol is not None:
            graphite_vol_simulated = float(_saved_g_vol)
            graphite_vol_full      = graphite_vol_simulated * geometry_factor
            mats.graphite.volume   = graphite_vol_simulated
            print(f"\nGraphite volume loaded from run_params.json (skipping stochastic calc): "
                  f"{graphite_vol_simulated:.4f} cm³")
        else:
            n_vol_particles = params.get("graphite_volume_particles", 1_000_000)
            core_r  = params["core_radius"]
            refl_t  = params["reflector_thickness"]
            core_h  = params["core_height"]
            lower_left  = [-core_r, -core_r, -refl_t]
            upper_right = [ core_r,  core_r,  core_h + refl_t]

            print(f"\nRunning stochastic volume calculation for graphite "
                  f"({n_vol_particles:,} particles)...")
            vol_calc = openmc.VolumeCalculation(
                [mats.graphite], n_vol_particles, lower_left, upper_right
            )
            model_bol.settings.volume_calculations = [vol_calc]
            model_bol.settings.export_to_xml()
            openmc.calculate_volumes(output=True, cwd=run_dir)

            vol_h5_path = os.path.join(run_dir, "volume_1.h5")
            with h5py.File(vol_h5_path, 'r') as vf:
                mat_id_str = str(mats.graphite.id)
                mat_key = next(
                    (k for k in vf.keys() if mat_id_str in k), None
                )
                if mat_key is None:
                    raise KeyError(
                        f"Graphite material (id={mats.graphite.id}) not found in {vol_h5_path}. "
                        f"Available keys: {list(vf.keys())}"
                    )
                graphite_vol_simulated = float(vf[mat_key]['volume'][0])
            graphite_vol_full = graphite_vol_simulated * geometry_factor
            mats.graphite.volume = graphite_vol_simulated
            model_bol.materials.export_to_xml()

            # Clear volume_calculations so they don't re-run during depletion
            model_bol.settings.volume_calculations = []
            model_bol.settings.export_to_xml()

            print(f"Graphite volume (simulated geometry): {graphite_vol_simulated:.4f} cm³")
            print(f"Graphite volume (full core):          {graphite_vol_full:.4f} cm³")

    print(f"\nFuel volume   (simulated): {fuel_volume_simulated:.4f} cm³")
    print(f"Fuel volume   (full core): {total_fuel_volume:.4f} cm³")
    print(f"Uranium mass  (full core): {total_HM_mass_kg:.2f} kg")
    print(f"B4C poison    (simulated): {poison_volume_simulated:.4f} cm³")
    print(f"B4C poison    (full core): {total_poison_volume:.4f} cm³")
    print(f"B-10 mass     (full core): {total_B10_mass_kg:.4f} kg")

    # Persist metadata to run_params.json
    params_path  = os.path.join(run_dir, "run_params.json")
    saved_params = json.load(open(params_path)) if os.path.exists(params_path) else {}
    saved_params.update({
        "n_trisos":                    n_trisos_global,
        "use_homogenized_fuel":        params.get("use_homogenized_fuel", False),
        "use_spatial_burnup":          params.get("use_spatial_burnup", True),
        "fuel_volume_simulated_cm3":   fuel_volume_simulated,
        "fuel_volume_full_core_cm3":   total_fuel_volume,
        "total_HM_mass_kg":            total_HM_mass_kg,
        "poison_volume_simulated_cm3": poison_volume_simulated,
        "poison_volume_full_core_cm3": total_poison_volume,
        "total_B10_mass_kg":           total_B10_mass_kg,
        "fuel_mat_volumes": {
            str(fuel_clones_bol[ri][ai].id): fuel_clones_bol[ri][ai].volume
            for ri in range(len(fuel_clones_bol))
            for ai in range(len(fuel_clones_bol[ri]))
        },
        "fuel_mat_ids": [
            [fuel_clones_bol[ri][ai].id for ai in range(len(fuel_clones_bol[ri]))]
            for ri in range(len(fuel_clones_bol))
        ],
        "poison_mat_volumes": {
            str(poison_clones_bol[ri][ai].id): poison_clones_bol[ri][ai].volume
            for ri in range(len(poison_clones_bol))
            for ai in range(len(poison_clones_bol[ri]))
        },
        "poison_mat_ids": [
            [poison_clones_bol[ri][ai].id for ai in range(len(poison_clones_bol[ri]))]
            for ri in range(len(poison_clones_bol))
        ],
    })
    if params.get("deplete_graphite", False) and graphite_vol_simulated is not None:
        saved_params["graphite_material_id"]          = mats.graphite.id
        saved_params["graphite_volume_simulated_cm3"] = graphite_vol_simulated
        saved_params["graphite_volume_full_core_cm3"] = graphite_vol_full
    with open(params_path, "w") as f:
        json.dump(saved_params, f, indent=2)

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

    # ================================================================
    # MAIN LOOP — one iteration per depletion timestep
    # ================================================================
    for step_idx, dt_days in enumerate(timesteps_days):

        step_start_day = cumulative_days
        step_end_day   = cumulative_days + dt_days

        print(f"\n{'=' * 80}")
        print(f"  TIMESTEP {step_idx + 1} / {n_steps}  "
              f"[{step_start_day:.1f} d → {step_end_day:.1f} d  "
              f"(Δt = {dt_days:.1f} d)]")
        print(f"{'=' * 80}")

        # ------------------------------------------------------------
        # 1. Determine material state for this step
        #    Step 0 → fresh fuel (BOL model already built above)
        #    Step n → load depleted materials from depletion_results.h5
        # ------------------------------------------------------------
        if step_idx == 0:
            # Fresh fuel — use the BOL fuel clones directly.
            # We still need to run a critical search on fresh fuel.
            depleted_for_search = None   # signals "use fresh fuel"
            prev_results        = None
        else:
            # Reconstruct depleted compositions from the accumulated chained h5.
            # Results index convention: atoms[0] = initial state (t=0),
            # atoms[i] = end of step i.  Pass step_idx to get the state
            # at the end of the most recently completed depletion step.
            print(f"\n  Loading depleted materials from step {step_idx} "
                  f"(t = {step_start_day:.1f} d)...")
            depleted_for_search, _, _, _ = reconstruct_depleted_materials(
                run_dir, step_idx   # Results index: 0=initial, i=end of step i
            )
            prev_results = openmc.deplete.Results(depletion_h5)

        # ------------------------------------------------------------
        # 2. Critical rod search at the current burnup state
        # ------------------------------------------------------------
        print(f"\n{'─' * 70}")
        print(f"  CRITICAL ROD SEARCH — step {step_idx + 1} / {n_steps}  "
              f"({step_start_day:.1f} d)")
        print(f"{'─' * 70}")

        # Build a lightweight params dict for the search iterations
        search_params = copy.deepcopy(params)
        search_params["total_batches"]       = cs_batches
        search_params["inactive_batches"]    = cs_inactive
        search_params["particles"]           = cs_particles
        search_params["make_geometry_plots"] = False
        search_params["use_mesh_tallies"]    = False
        search_params["use_BeO_tallies"]     = False
        search_params["use_leakage_tallies"] = False
        search_params["use_global_tallies"]  = False

        search_dir = os.path.join(
            run_dir, f"critical_search_step{step_idx + 1:03d}_t{step_start_day:.0f}d"
        )
        os.makedirs(search_dir, exist_ok=True)

        if depleted_for_search is None:
            # BOL (step 0): fresh fuel compositions — no injection needed.
            # An empty dict is passed so _inject_depleted_materials inside
            # find_critical_rod_insertion is a no-op.
            depleted_arg = {}
        else:
            # Steps > 0: depleted_for_search contains atom densities from the
            # end of the previous timestep.  find_critical_rod_insertion passes
            # this dict to _run_eigenvalue_with_depleted → _inject_depleted_materials
            # BEFORE each trial eigenvalue solve, so every search iteration
            # sees the correct burnup-state compositions.  The critical insertion
            # found therefore reflects the actual reduced reactivity of the fuel
            # at this point in the cycle, not BOL reactivity.
            depleted_arg = depleted_for_search

        crit_result = find_critical_rod_insertion(
            params        = search_params,
            depleted      = depleted_arg,
            output_dir    = search_dir,
            k_target      = 1.0,
            k_tol         = k_tol,
            max_iter      = max_iter,
            prev_result   = prev_crit_result if step_idx > 0 else None,
        )
        prev_crit_result = crit_result

        critical_b1  = crit_result["critical_bank_1"]
        critical_b2  = crit_result["critical_bank_2"]
        critical_k   = crit_result["critical_keff"]
        critical_std = crit_result["critical_keff_std"]
        converged    = crit_result["converged"]
        search_csv   = crit_result["csv_path"]   # inside search_dir

        print(f"\n  Critical insertion found:")
        print(f"    Bank 1 = {critical_b1:.4f}, Bank 2 = {critical_b2:.4f}")
        print(f"    k_eff  = {critical_k:.5f} ± {critical_std:.5f}  "
              f"({'converged' if converged else 'NOT CONVERGED'})")

        # ------------------------------------------------------------
        # Copy the search CSV to run_dir, then delete the entire
        # search directory (all trial subdirs + their OpenMC output).
        # The CSV is the only artifact we keep from the critical search.
        # ------------------------------------------------------------
        csv_dest = os.path.join(
            run_dir,
            f"critical_search_step{step_idx + 1:03d}_t{step_start_day:.0f}d.csv"
        )
        try:
            shutil.copy2(search_csv, csv_dest)
            print(f"    Search CSV saved -> {csv_dest}")
        except Exception as e:
            print(f"    WARNING: Could not copy search CSV: {e}")

        try:
            shutil.rmtree(search_dir)
            print(f"    Search dir deleted: {search_dir}")
        except Exception as e:
            print(f"    WARNING: Could not delete search dir: {e}")

        # ------------------------------------------------------------
        # 3. Build the depletion model at the critical rod position.
        #    Read run_params.json NOW — before build_model() overwrites it
        #    — to capture the previous step's material IDs for the remap.
        # ------------------------------------------------------------
        _params_path_remap = os.path.join(run_dir, "run_params.json")
        if step_idx > 0 and os.path.exists(_params_path_remap):
            _prev_saved       = json.load(open(_params_path_remap))
            _prev_fuel_ids    = _prev_saved.get("fuel_mat_ids", [])
            _prev_poison_ids  = _prev_saved.get("poison_mat_ids", [])
            _prev_graphite_id = _prev_saved.get("graphite_material_id")
        else:
            _prev_fuel_ids    = []
            _prev_poison_ids  = []
            _prev_graphite_id = None

        depletion_params = copy.deepcopy(params)
        depletion_params["bank_1_insertion"] = critical_b1
        depletion_params["bank_2_insertion"] = critical_b2
        depletion_params["bank_3_insertion"] = 0.0
        depletion_params["make_geometry_plots"] = False

        print(f"\n  Building depletion model at critical rod position...")
        model_step, _, _, fuel_clones_step, poison_clones_step = build_model(depletion_params, run_dir)

        # Remap material IDs to match the previous step's IDs so that
        # CoupledOperator(prev_results=...) / transfer_volumes can find them
        # in the chained depletion_results.h5.  Each build_model() call
        # auto-increments material IDs, so without this remap the IDs would
        # differ from the previous step and transfer_volumes would KeyError.
        # We set _id directly to avoid triggering OpenMC's used_ids warning
        # (the old material objects still exist but are no longer used).
        if step_idx > 0 and _prev_fuel_ids:
            _remapped = set()
            for _ri, _row in enumerate(fuel_clones_step):
                for _ai, _mat in enumerate(_row):
                    _tid = int(_prev_fuel_ids[_ri][_ai])
                    if _tid not in _remapped:
                        _mat._id = _tid
                        _remapped.add(_tid)

            if _prev_poison_ids:
                _remapped_p = set()
                for _ri, _prow in enumerate(_prev_poison_ids):
                    for _ai, _tpid in enumerate(_prow):
                        _tpid = int(_tpid)
                        if _tpid not in _remapped_p and _ri < len(poison_clones_step) and _ai < len(poison_clones_step[_ri]):
                            poison_clones_step[_ri][_ai]._id = _tpid
                            _remapped_p.add(_tpid)

            if _prev_graphite_id is not None and params.get("deplete_graphite", False):
                _tgid = int(_prev_graphite_id)
                for _mat in model_step.materials:
                    if _mat.name == "Graphite":
                        _mat._id = _tgid
                        break

        # Rebuild the zone_heating_local MaterialFilter after the ID remap.
        # MaterialFilter stores IDs as integers at construction time (not live
        # object references), so changing _mat._id above does NOT update the
        # stored bins.  We must reconstruct the filter with the now-remapped IDs.
        if step_idx > 0 and params.get("use_spatial_burnup", True):
            try:
                _zheat_tally = next(
                    t for t in model_step.tallies if t.name == "zone_heating_local"
                )
                _remapped_zone_mats = [
                    fc for row in fuel_clones_step for fc in row if fc.volume
                ]
                _zheat_tally.filters = [openmc.MaterialFilter(_remapped_zone_mats)]
            except StopIteration:
                pass

        # Inject depleted materials from the previous step (if not BOL)
        if step_idx > 0 and depleted_for_search:
            print(f"  Injecting depleted compositions into fuel and poison clones...")
            _inject_depleted_materials(
                fuel_clones_step, depleted_for_search,
                model=model_step, poison_clones=poison_clones_step
            )

        model_step.export_to_xml()

        # ------------------------------------------------------------
        # 4. Run single-timestep depletion with prev_results chaining.
        #    The ID remap above ensures transfer_volumes can match every
        #    depletable material in model_step to the previous h5 entry.
        # ------------------------------------------------------------
        print(f"\n  Running depletion for Δt = {dt_days:.1f} d "
              f"at bank_1 = {critical_b1:.4f}, bank_2 = {critical_b2:.4f}...")

        operator = openmc.deplete.CoupledOperator(
            model_step,
            chain_file         = chain_file,
            normalization_mode = "fission-q",
            prev_results       = prev_results,
        )

        integrator = IntegratorClass(
            operator,
            [dt_days],
            power          = operator_power_W,
            timestep_units = "d",
        )

        # write_rates=True is REQUIRED for multi-step depletion with prev_results.
        # When the next step calls _get_bos_data_from_restart(), it reads
        # prev_res[-1].rates from the h5.  Without write_rates=True the
        # "reaction rates" dataset is never stored, so prev_res[-1].rates is
        # all zeros → pure radioactive decay (zero-power Bateman integration).
        integrator.integrate(write_rates=True)

        # Record zone heating for this step immediately after integrate()
        # while the statepoint is still the one for this step.
        if params.get("use_spatial_burnup", True):
            _cs_fuel_ids_2d = [
                [fuel_clones_step[ri][ai].id for ai in range(len(fuel_clones_step[ri]))]
                for ri in range(len(fuel_clones_step))
            ]
            _append_zone_heating_step(
                run_dir         = run_dir,
                fuel_mat_ids_2d = _cs_fuel_ids_2d,
                step_idx        = step_idx,
                step_start_days = step_start_day,
                step_end_days   = step_end_day,
            )

        cumulative_days = step_end_day

        # ------------------------------------------------------------
        # 4b. Re-save fuel metadata to run_params.json
        #
        # build_model() calls os.chdir() internally, and each critical
        # search trial also calls build_model() with a different cwd.
        # This means save_params() inside build_model() may have
        # overwritten run_dir/run_params.json with a bare params dict
        # (no fuel_mat_ids / fuel_mat_volumes) by the time we get here.
        # Re-writing it now guarantees reconstruct_depleted_materials()
        # always finds the correct material IDs at the start of the
        # next timestep's critical search.
        # ------------------------------------------------------------
        params_path = os.path.join(run_dir, "run_params.json")
        saved_params = json.load(open(params_path)) if os.path.exists(params_path) else {}
        _step_graphite = next(
            (m for m in model_step.materials if m.name == "Graphite"),
            mats.graphite,
        )
        # Compute poison volume for this step from the step's poison clones
        _seen_sp = set()
        _step_poison_vol = 0.0
        for _ri in range(len(poison_clones_step)):
            for _ai in range(len(poison_clones_step[_ri])):
                _pm = poison_clones_step[_ri][_ai]
                if _pm.id not in _seen_sp:
                    _seen_sp.add(_pm.id)
                    _step_poison_vol += (_pm.volume or 0.0)
        saved_params.update({
            "n_trisos":                    n_trisos_global,
            "use_homogenized_fuel":        params.get("use_homogenized_fuel", False),
            "use_spatial_burnup":          params.get("use_spatial_burnup", True),
            "fuel_volume_simulated_cm3":   fuel_volume_simulated,
            "fuel_volume_full_core_cm3":   total_fuel_volume,
            "total_HM_mass_kg":            total_HM_mass_kg,
            "poison_volume_simulated_cm3": _step_poison_vol,
            "poison_volume_full_core_cm3": _step_poison_vol * geometry_factor,
            "total_B10_mass_kg":           total_B10_mass_kg,
            "fuel_mat_volumes": {
                str(fuel_clones_step[ri][ai].id): fuel_clones_step[ri][ai].volume
                for ri in range(len(fuel_clones_step))
                for ai in range(len(fuel_clones_step[ri]))
            },
            "fuel_mat_ids": [
                [fuel_clones_step[ri][ai].id for ai in range(len(fuel_clones_step[ri]))]
                for ri in range(len(fuel_clones_step))
            ],
            "poison_mat_volumes": {
                str(poison_clones_step[ri][ai].id): poison_clones_step[ri][ai].volume
                for ri in range(len(poison_clones_step))
                for ai in range(len(poison_clones_step[ri]))
            },
            "poison_mat_ids": [
                [poison_clones_step[ri][ai].id for ai in range(len(poison_clones_step[ri]))]
                for ri in range(len(poison_clones_step))
            ],
        })
        if params.get("deplete_graphite", False):
            saved_params["graphite_material_id"] = _step_graphite.id
        os.chdir(run_dir)   # restore cwd after trial runs may have changed it
        with open(params_path, "w") as f:
            json.dump(saved_params, f, indent=2)

        # ------------------------------------------------------------
        # 5. Log this step's result
        # ------------------------------------------------------------
        # operational = 0 only at end-of-cycle: both banks fully withdrawn (0)
        # AND the core is still subcritical — rods can go no further out.
        # Any other condition (rods still inserted, or keff≈1 within stats)
        # is considered operational.
        step_operational = 0 if (critical_b1 == 0.0 and critical_b2 == 0.0
                                  and critical_k < 1.0) else 1

        step_entry = {
            "step":              step_idx + 1,
            "step_start_days":   step_start_day,
            "step_end_days":     step_end_day,
            "dt_days":           dt_days,
            "bank_1_insertion":  critical_b1,
            "bank_2_insertion":  critical_b2,
            "critical_keff":     critical_k,
            "critical_keff_std": critical_std,
            "converged":         converged,
            "operational":       step_operational,
            "search_csv":        csv_dest,   # search_dir has been deleted
        }
        step_log.append(step_entry)

        with open(step_log_path, "w") as f:
            json.dump(step_log, f, indent=2)

        print(f"\n  ✓ Step {step_idx + 1} complete  "
              f"[cumulative time: {cumulative_days:.1f} d]")
        print(f"    Step log saved → {step_log_path}")

    # ================================================================
    # ALL STEPS COMPLETE
    # ================================================================
    print(f"\n{'=' * 80}")
    print("CRITICAL SEARCH DEPLETION SIMULATION COMPLETE")
    print(f"  Total steps:    {n_steps}")
    print(f"  Total burnup:   {cumulative_days:.1f} days  "
          f"({cumulative_days / 365.25:.2f} years)")
    print(f"  Results HDF5:   {depletion_h5}")
    print(f"  Step log:       {step_log_path}")
    print(f"{'=' * 80}\n")

    # Print rod insertion history summary
    print(f"{'─' * 70}")
    print(f"  {'Step':>5}  {'t_start (d)':>12}  {'t_end (d)':>10}  "
          f"{'Bank 1':>8}  {'Bank 2':>8}  {'k_eff':>10}  {'Conv':>6}")
    print(f"{'─' * 70}")
    for entry in step_log:
        print(f"  {entry['step']:>5}  {entry['step_start_days']:>12.1f}  "
              f"{entry['step_end_days']:>10.1f}  "
              f"{entry['bank_1_insertion']:>8.4f}  "
              f"{entry['bank_2_insertion']:>8.4f}  "
              f"{entry['critical_keff']:>10.5f}  "
              f"{'Yes' if entry['converged'] else 'NO':>6}")
    print(f"{'─' * 70}\n")

    return n_trisos_global

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

    # ----- Run Parametric Study -----

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
    
    # ----- Run RPT Calibration -----

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
    
    # ----- Run Depletion Study -----

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

    # ----- Run Critical Search Depletion Study -----

    elif cfg.params["study_execution_mode"] == "CSDepletionStudy":
        BASE_DIR = os.path.join(OUTPUT_BASE, run_name + "_CSDepletion")
        
        print(f"\n{'='*80}")
        print("CRITICAL SEARCH DEPLETION RUN MODE")
        print(f"Run directory: {BASE_DIR}")
        print(f"{'='*80}")

        n_trisos = run_critical_search_depletion_simulation(cfg.params, BASE_DIR)

        # Run depletion-specific post-processing
        run_depletion_post_processing(BASE_DIR, cfg.params)

        print(f"\n{'='*80}")
        print("CRITICAL SEARCH DEPLETION RUN COMPLETE")
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
    
    # ----- Run Critical Rod Search -----

    elif cfg.params["study_execution_mode"] == "CriticalSearch":
        try:
            from mol_eol_analysis import find_critical_rod_insertion
        except ImportError as exc:
            raise ImportError(
                "CriticalSearch mode requires mol_eol_analysis.py to be "
                "importable from SCRIPT_DIR."
            ) from exc

        BASE_DIR_CS = os.path.join(OUTPUT_BASE, run_name + "_CriticalSearch")
        os.makedirs(BASE_DIR_CS, exist_ok=True)

        k_tol    = cfg.params.get("critical_search_k_tol",     0.003)
        max_iter = cfg.params.get("critical_search_max_iter",   20)

        print(f"\n{'='*80}")
        print(f"CRITICAL SEARCH — BOL")
        print(f"  Target: k_eff = 1.0  ±  {k_tol}")
        print(f"  Output: {BASE_DIR_CS}")
        print(f"{'='*80}")

        cs_result = find_critical_rod_insertion(
            params     = cfg.params,
            depleted   = {},
            output_dir = BASE_DIR_CS,
            k_target   = 1.0,
            k_tol      = k_tol,
            max_iter   = max_iter,
        )

        print(f"\n{'='*80}")
        print(f"  CRITICAL SEARCH RESULT — BOL")
        print(f"  Bank 1 insertion : {cs_result['critical_bank_1']:.4f}")
        print(f"  Bank 2 insertion : {cs_result['critical_bank_2']:.4f}")
        print(f"  Search stage     : {cs_result['search_stage']}")
        print(f"  k_eff            : {cs_result['critical_keff']:.5f} "
              f"± {cs_result['critical_keff_std']:.5f}")
        print(f"  Converged        : {cs_result['converged']}  "
              f"({cs_result['n_iterations']} iterations)")
        print(f"{'='*80}\n")

        result_path = os.path.join(BASE_DIR_CS, "critical_search_result.json")
        with open(result_path, "w") as f:
            json.dump(cs_result, f, indent=2)
        print(f"  Result saved to: {result_path}")

    else:
        print(f"\nERROR: Unknown study_execution_mode: '{cfg.params['study_execution_mode']}'")
        print("Valid modes: SingleStudy, ParametricStudy, ReactivityStudy, DepletionStudy, RPTCalibration, CriticalSearch")
        sys.exit(1)