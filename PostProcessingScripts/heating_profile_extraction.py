"""
Heating Profile Extraction Post-Processing Script

Extracts axial heating profiles for the hottest fuel channel and the
core-average fuel channel from the mesh_heating_full tally in an OpenMC
statepoint file.

APPROACH:
    Instead of DistribcellFilter tallies (which create enormous output
    because OpenMC enumerates every geometry path to each fuel cell),
    this script uses the existing RegularMesh heating tally.

    Fuel compact centroids are determined analytically from the reactor
    geometry (core_rings, fuel_to_coolant_distance, bundle_pitch) rather
    than by image-based peak detection.  This is mesh-resolution-
    independent and requires no threshold tuning.

    The script:
      1. Reads the mesh_heating_full tally via openmc.StatePoint
      2. Reshapes to 3-D (nx, ny, nz_full), slices to active core
      3. Analytically computes fuel compact centroid positions
      4. For each centroid, sums mesh voxels within compact_radius
      5. Identifies the hottest and average profiles, computes peaking
         factors, and generates plots + CSV outputs.

Usage:
    from heating_profile_extraction import run_heating_profile_extraction
    run_heating_profile_extraction(run_dir, params)

    python heating_profile_extraction.py <run_directory> [batch_number]
"""

import matplotlib
matplotlib.use('Agg')          # headless — no display server needed

import openmc
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import gc


# ====================================================================================================
# NORMALIZATION
# ====================================================================================================

def get_normalization_factor(sp_path, target_power_MW):
    """
    Calculate source-rate normalization from the global 'heating' tally.
    Opens and closes the StatePoint cleanly.
    """
    sp = openmc.StatePoint(sp_path)
    heating_tally = sp.get_tally(name='heating')
    heating_rate_ev = heating_tally.mean[0, 0, 0]
    sp.close()
    del sp, heating_tally
    gc.collect()

    joule_per_ev = 1.60218e-19
    heating_rate_j = heating_rate_ev * joule_per_ev
    source_per_sec = (target_power_MW * 1e6) / heating_rate_j
    return source_per_sec


# ====================================================================================================
# ANALYTICAL FUEL COMPACT CENTROID GENERATION
# ====================================================================================================

def _hex_ring_positions(r, pitch):
    """
    Cartesian (x, y) positions of all elements in ring r of a flat-top
    hex lattice (OpenMC orientation='x') centred at the origin.

    Ring 0 → [(0, 0)].  Ring r > 0 → 6r positions starting from
    (r*pitch, 0) and traversing counter-clockwise, matching OpenMC's
    HexLattice element ordering for orientation='x'.
    """
    if r == 0:
        return np.array([[0.0, 0.0]])

    p    = pitch
    s32  = np.sqrt(3.0) / 2.0
    # Six CCW step-directions for a flat-top hex ring traversal
    step_dirs = [
        (-0.5 * p,   s32 * p),   # upper-left
        (-p,         0.0),        # left
        (-0.5 * p,  -s32 * p),   # lower-left
        ( 0.5 * p,  -s32 * p),   # lower-right
        ( p,         0.0),        # right
        ( 0.5 * p,   s32 * p),   # upper-right
    ]

    positions = []
    x, y = r * pitch, 0.0
    for dx, dy in step_dirs:
        for _ in range(r):
            positions.append([x, y])
            x += dx
            y += dy

    return np.array(positions)   # shape (6r, 2)


def compute_fuel_compact_centroids(params):
    """
    Analytically compute (x, y) centroids of every fuel compact channel
    in the simulated geometry (1/6 wedge or full core).

    Uses the hex lattice geometry from params (core_rings,
    fuel_to_coolant_distance, bundle_pitch) to reconstruct exact positions
    of every fuel compact.  No mesh resolution dependence, no thresholding.

    Assembly type → fuel pattern (from assembly.py):
      "f", "fpa"   — ring1+ring2+ring3+ring4          (42 channels)
      "fp"         — ring1+ring2+ring3+ring4p          (36 channels)
                     (ring4p: i%4==0 positions are poison, not fuel)
      "fc*", "fss" — ring2+ring3+ring4                 (36 channels)
      "fcp*"       — ring2+ring3+ring4p                (30 channels)
      "fssp"       — ring2+ring3+ring4p                (30 channels)
      "r*"         — reflector, no fuel compacts

    For 1/6 geometry, only positions in the first sextant (0° ≤ θ ≤ 60°,
    inclusive of both symmetry planes) are returned.  Assemblies that
    straddle a symmetry plane contribute only the channels that fall
    inside the wedge.

    Returns
    -------
    channel_xy     : (N, 2) float — (x, y) of every fuel compact centroid
    asm_centers    : (M, 2) float — (x, y) of every fuel assembly centre
                     (only assemblies with ≥1 channel in the domain)
    channel_asm_idx: (N,)   int  — maps each channel to its assembly index
    asm_nchannels  : (M,)   int  — number of fuel channels per assembly
    """
    p_ch   = float(params["fuel_to_coolant_distance"])   # channel lattice pitch
    bundle = 5.0 * p_ch * np.sqrt(3.0)                  # assembly c-t-c distance
    core_rings = params["core_rings"]
    is_16  = params.get("use_1/6_geometry", True)
    n_rings = len(core_rings)

    # ------------------------------------------------------------------
    # Inner-assembly ring fuel masks (True = fuel compact)
    # ring2 = ([f]+[c])*6       → even indices fuel, odd coolant
    # ring3 = ([c]+[f,f])*6     → positions 1,2 in each triplet are fuel
    # ring4 = ([f,f]+[c]+[f])*6 → i%4 ∈ {0,1,3} are fuel
    # ring4p (poison variant)   → i%4==0 replaced by poison (not fuel)
    # ------------------------------------------------------------------
    ring1_mask  = [True]  * 6
    ring2_mask  = [True,  False] * 6
    ring3_mask  = [False, True, True] * 6
    ring4_mask  = [True,  True,  False, True] * 6   # standard
    ring4p_mask = [False, True,  False, True] * 6   # outer-poison variant

    def _assembly_rings(asm_type):
        """
        Return list of (inner_ring_r, fuel_mask) for a given assembly type.
        Empty list for non-fuel (reflector) assemblies.
        """
        t = asm_type.lower()
        if t.startswith('r'):
            return []     # rr, r1-r3, ra*, rss, rssa — no fuel

        uses_poison_r4 = t.startswith('fp') or t.startswith('fcp') or t.startswith('fssp')
        has_ring1      = t in ('f', 'fp', 'fpa')   # ring1 intact (no central rod)

        r4 = ring4p_mask if uses_poison_r4 else ring4_mask
        masks = [(4, r4), (3, ring3_mask), (2, ring2_mask)]
        if has_ring1:
            masks.append((1, ring1_mask))
        # ring0 centre is always graphite or poison — never a fuel compact
        return masks

    # ------------------------------------------------------------------
    # Iterate every assembly slot in the core
    # ------------------------------------------------------------------
    all_channel_xy  = []
    all_asm_centers = []   # centres of fuel-type assemblies (global, pre-filter)
    raw_asm_idx     = []   # global asm index for each channel

    global_asm = 0

    for core_ring_idx, ring_types in enumerate(core_rings):
        # core_rings[0] = outermost ring → hex lattice ring (n_rings-1)
        lat_ring      = n_rings - 1 - core_ring_idx
        asm_pos_raw   = _hex_ring_positions(lat_ring, bundle)

        # OpenMC's core HexLattice (default orientation='y', pointy-top) places
        # ring-r element 0 at 30° from east (+x), i.e. the upper-right vertex.
        # _hex_ring_positions starts at 0° (east), so rotate +30° to match.
        if lat_ring > 0:
            _th = np.radians(30.0)
            _c, _s = np.cos(_th), np.sin(_th)
            asm_positions = np.column_stack([
                asm_pos_raw[:, 0] * _c - asm_pos_raw[:, 1] * _s,
                asm_pos_raw[:, 0] * _s + asm_pos_raw[:, 1] * _c,
            ])
        else:
            asm_positions = asm_pos_raw  # ring 0 is just [(0, 0)]

        if len(ring_types) != len(asm_positions):
            raise ValueError(
                f"core_rings[{core_ring_idx}] has {len(ring_types)} entries "
                f"but hex lattice ring {lat_ring} has {len(asm_positions)} positions"
            )

        for asm_pos, asm_type in zip(asm_positions, ring_types):
            rings = _assembly_rings(asm_type)
            if not rings:
                continue     # reflector — skip

            ax, ay = asm_pos
            all_asm_centers.append([ax, ay])

            for inner_r, mask in rings:
                for pos, is_fuel in zip(_hex_ring_positions(inner_r, p_ch), mask):
                    if is_fuel:
                        all_channel_xy.append([ax + pos[0], ay + pos[1]])
                        raw_asm_idx.append(global_asm)

            global_asm += 1

    channel_xy  = np.array(all_channel_xy,  dtype=float)
    asm_centers = np.array(all_asm_centers, dtype=float)
    raw_asm_idx = np.array(raw_asm_idx,     dtype=int)

    if is_16:
        # Keep only channels inside the first sextant (0° ≤ θ ≤ 60°).
        # eps ensures positions exactly on the symmetry planes are included.
        eps = 1e-6
        in_wedge = (
            (channel_xy[:, 0] >= -eps) &
            (channel_xy[:, 1] >= -eps) &
            (channel_xy[:, 1] <= channel_xy[:, 0] * np.sqrt(3.0) + eps)
        )
        channel_xy  = channel_xy[in_wedge]
        raw_asm_idx = raw_asm_idx[in_wedge]

        # Retain only assemblies that still have ≥1 channel in the wedge
        unique_global = np.unique(raw_asm_idx)
        global_to_local = {g: l for l, g in enumerate(unique_global)}
        asm_centers = asm_centers[unique_global]
        channel_asm_idx = np.array([global_to_local[i] for i in raw_asm_idx],
                                   dtype=int)
    else:
        channel_asm_idx = raw_asm_idx

    n_asm = len(asm_centers)
    asm_nchannels = np.bincount(channel_asm_idx, minlength=n_asm).astype(int)

    return channel_xy, asm_centers, channel_asm_idx, asm_nchannels


def _point_in_hex(px, py, cx, cy, apothem):
    """Test if point (px, py) is inside a flat-topped hex centred at (cx, cy)."""
    dx = abs(px - cx)
    dy = abs(py - cy)
    # Flat-topped hex with apothem = inradius (half flat-to-flat distance)
    if dx > apothem or dy > apothem * 2 / np.sqrt(3):
        return False
    return apothem * 2 / np.sqrt(3) - dy >= dx / np.sqrt(3)


def extract_assembly_heating(heating_3d, mesh_ll, mesh_ur, asm_centers, bundle_pitch):
    """
    Compute q(z) for each assembly by summing all mesh voxels within
    its hex footprint at each axial level.

    Parameters
    ----------
    heating_3d   : (nx, ny, nz) array in watts per voxel
    mesh_ll, mesh_ur : mesh bounding box
    asm_centers  : (n_assemblies, 2) array — analytically computed (x, y)
    bundle_pitch : assembly centre-to-centre distance (hex inradius = bundle_pitch/2)

    Returns
    -------
    asm_q : (n_assemblies, nz) array — total heating [W] per axial level
    """
    nx, ny, nz = heating_3d.shape
    dx = (mesh_ur[0] - mesh_ll[0]) / nx
    dy = (mesh_ur[1] - mesh_ll[1]) / ny

    x_centers = np.linspace(mesh_ll[0] + dx/2, mesh_ur[0] - dx/2, nx)
    y_centers = np.linspace(mesh_ll[1] + dy/2, mesh_ur[1] - dy/2, ny)

    n_asm   = len(asm_centers)
    apothem = bundle_pitch / 2.0   # hex inradius (half flat-to-flat)

    asm_q = np.zeros((n_asm, nz))
    for a in range(n_asm):
        cx, cy = asm_centers[a]
        ix_lo = max(0, int((cx - apothem - mesh_ll[0]) / dx) - 1)
        ix_hi = min(nx, int((cx + apothem - mesh_ll[0]) / dx) + 2)
        iy_lo = max(0, int((cy - apothem - mesh_ll[1]) / dy) - 1)
        iy_hi = min(ny, int((cy + apothem - mesh_ll[1]) / dy) + 2)

        for ix in range(ix_lo, ix_hi):
            for iy in range(iy_lo, iy_hi):
                if _point_in_hex(x_centers[ix], y_centers[iy], cx, cy, apothem):
                    asm_q[a, :] += heating_3d[ix, iy, :]

    return asm_q


def extract_mesh_heating(sp_path, source_per_sec, params, symmetry_factor,
                         tally_name='mesh_heating_full'):
    """
    Extract per-fuel-channel axial heating profiles from a mesh heating tally.

    tally_name controls which tally is read:
      'mesh_heating_full' (default) — full-core mesh including reflector bins.
          The active-core z-slice is extracted using n_ax_zones from params.
          Used by post-processing scripts.
      'mesh_heating' — active-core-only mesh (reactor_bottom → reactor_top).
          No slicing needed; used by the TH coupler during iteration runs
          where only the active-core tally is written to save overhead.

    Steps:
      1. Read the named tally and mesh geometry
      2. Reshape to 3-D (nx, ny, nz_raw)
      3. If full-core tally: slice to active-core z-bins
      4. Sum over z -> 2-D integrated heating map
      5. Analytically compute fuel compact centroids
      6. Sum voxels within compact_radius of each centroid -> per-channel profiles

    Returns
    -------
    dict with keys: 'z_centers', 'heating_2d', 'n_channels',
                    'peak_coords', 'mesh_lower_left', 'mesh_upper_right',
                    'heating_3d', 'integrated_2d'
    """
    joule_per_ev = 1.60218e-19

    # ------------------------------------------------------------------
    # Read the requested tally via StatePoint
    # ------------------------------------------------------------------
    sp = openmc.StatePoint(sp_path)
    tally = sp.get_tally(name=tally_name)
    mesh = tally.find_filter(openmc.MeshFilter).mesh

    nx, ny, nz_raw = mesh.dimension
    ll_raw = mesh.lower_left.copy()
    ur_raw = mesh.upper_right.copy()

    # tally.mean has shape (nx*ny*nz_raw, 1, 1) for single score/nuclide
    mean_flat = tally.mean[:, 0, 0].copy()

    sp.close()
    del sp, tally, mesh
    gc.collect()

    # ------------------------------------------------------------------
    # Reshape and convert to physical heating (watts per voxel)
    # ------------------------------------------------------------------
    # OpenMC stores mesh tally in column-major order: x varies fastest
    # Reshape to (nz_raw, ny, nx) then transpose to (nx, ny, nz_raw)
    heating_3d_raw = mean_flat.reshape((nz_raw, ny, nx)).transpose((2, 1, 0))

    conv = source_per_sec * joule_per_ev / symmetry_factor
    heating_3d_raw *= conv  # now in watts per voxel

    del mean_flat
    gc.collect()

    # ------------------------------------------------------------------
    # Slice to active-core z-bins (only needed for full-core tally)
    # ------------------------------------------------------------------
    n_ax_zones = int(params.get("n_ax_zones", nz_raw))

    if tally_name == 'mesh_heating_full':
        # Full-core mesh has reflector bins above and below the active core.
        n_refl = (nz_raw - n_ax_zones) // 2
        z0, z1 = n_refl, n_refl + n_ax_zones

        heating_3d = heating_3d_raw[:, :, z0:z1]
        del heating_3d_raw
        gc.collect()

        dz = (ur_raw[2] - ll_raw[2]) / nz_raw
        ll = ll_raw.copy(); ll[2] = ll_raw[2] + n_refl * dz
        ur = ur_raw.copy(); ur[2] = ll[2] + n_ax_zones * dz
        nz = n_ax_zones
    else:
        # Active-core tally — bounds already cover only the active region.
        heating_3d = heating_3d_raw
        ll = ll_raw
        ur = ur_raw
        nz = nz_raw

    print(f"  Mesh dimensions (active core): {nx} x {ny} x {nz}")
    print(f"  Mesh bounds: ({ll[0]:.1f},{ll[1]:.1f},{ll[2]:.1f}) -> "
          f"({ur[0]:.1f},{ur[1]:.1f},{ur[2]:.1f}) cm")
    print(f"  Total mesh heating: {heating_3d.sum():.3e} W")
    print(f"  Peak voxel heating: {heating_3d.max():.3e} W")

    # Axial coordinates from sliced bounds
    z_edges = np.linspace(ll[2], ur[2], nz + 1)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0

    # ------------------------------------------------------------------
    # 2-D integrated heating map  (sum over z)
    # ------------------------------------------------------------------
    integrated_2d = heating_3d.sum(axis=2)  # shape (nx, ny)

    # ------------------------------------------------------------------
    # Analytically compute fuel compact centroids from geometry
    # ------------------------------------------------------------------
    channel_centroids, asm_centers, channel_asm_idx, asm_nchannels = \
        compute_fuel_compact_centroids(params)

    n_channels = len(channel_centroids)
    n_asm      = len(asm_centers)
    bundle_pitch = 5.0 * float(params.get("fuel_to_coolant_distance", 2.5)) * np.sqrt(3.0)

    print(f"  Fuel compact centroids (analytical): {n_channels}")
    print(f"  Fuel assemblies:                     {n_asm}")

    # ------------------------------------------------------------------
    # Sum voxels within compact_radius of each analytical centroid
    # ------------------------------------------------------------------
    dx = (ur[0] - ll[0]) / nx
    dy = (ur[1] - ll[1]) / ny
    x_vox = np.linspace(ll[0] + dx / 2, ur[0] - dx / 2, nx)
    y_vox = np.linspace(ll[1] + dy / 2, ur[1] - dy / 2, ny)

    compact_r  = float(params.get("compact_radius", 0.635))
    compact_r2 = compact_r ** 2
    search_px  = int(np.ceil(compact_r / dx)) + 1
    search_py  = int(np.ceil(compact_r / dy)) + 1

    print(f"  Fuel compact radius: {compact_r:.3f} cm  "
          f"(search window ±{search_px}×{search_py} voxels)")

    heating_profiles  = np.zeros((n_channels, nz))
    voxels_per_channel = []

    for k, (cx, cy) in enumerate(channel_centroids):
        # Nearest voxel index as starting point
        ix0 = int((cx - ll[0]) / dx)
        iy0 = int((cy - ll[1]) / dy)
        n_vox = 0
        for dix in range(-search_px, search_px + 1):
            ix = ix0 + dix
            if ix < 0 or ix >= nx:
                continue
            for diy in range(-search_py, search_py + 1):
                iy = iy0 + diy
                if iy < 0 or iy >= ny:
                    continue
                # Circle-vs-AABB: closest point on voxel cell to centroid
                near_x = max(x_vox[ix] - dx / 2, min(cx, x_vox[ix] + dx / 2))
                near_y = max(y_vox[iy] - dy / 2, min(cy, y_vox[iy] + dy / 2))
                dist2 = (near_x - cx) ** 2 + (near_y - cy) ** 2
                if dist2 <= compact_r2:
                    heating_profiles[k, :] += heating_3d[ix, iy, :]
                    n_vox += 1
        voxels_per_channel.append(n_vox)

    avg_vox = np.mean(voxels_per_channel) if voxels_per_channel else 0
    print(f"  Voxels summed per channel: avg {avg_vox:.1f}, "
          f"range [{min(voxels_per_channel)}, {max(voxels_per_channel)}]")

    # ------------------------------------------------------------------
    # Assembly-level q(z) from hex footprint summation
    # ------------------------------------------------------------------
    asm_q = extract_assembly_heating(heating_3d, ll, ur, asm_centers, bundle_pitch)

    for a in range(n_asm):
        cx, cy = asm_centers[a]
        print(f"    Asm {a}: ({cx:6.1f}, {cy:6.1f}) cm — "
              f"{asm_nchannels[a]} fuel channels")

    return {
        'z_centers':        z_centers,
        'heating_2d':       heating_profiles,    # (n_channels, nz) per-channel [W]
        'n_channels':       n_channels,
        'peak_coords':      channel_centroids,   # (n_channels, 2) analytical (x, y)
        'mesh_lower_left':  ll,
        'mesh_upper_right': ur,
        'heating_3d':       heating_3d,          # (nx, ny, nz) [W/voxel]
        'integrated_2d':    integrated_2d,       # (nx, ny)
        'asm_q':            asm_q,               # (n_asm, nz) [W per axial level]
        'asm_centers':      asm_centers,         # (n_asm, 2) analytical positions
        'asm_nchannels':    asm_nchannels,       # (n_asm,) channels per assembly
        'channel_asm_idx':  channel_asm_idx,     # (n_channels,) assembly membership
        'bundle_pitch':     bundle_pitch,
    }


# ====================================================================================================
# WEDGE POLYGON CLIPPING
# ====================================================================================================

def _clip_polygon_to_wedge(poly):
    """
    Clip a convex polygon (list of (x, y) tuples) to the 1/6-geometry wedge:
      y >= 0   AND   y <= x * sqrt(3)
    Uses the Sutherland-Hodgman algorithm.
    Returns a list of (x, y) tuples (empty if fully clipped away).
    """
    sqrt3 = np.sqrt(3.0)

    def _sh_clip(points, inside_fn, intersect_fn):
        if not points:
            return []
        out = []
        n = len(points)
        for i in range(n):
            cur = points[i]
            prv = points[i - 1]
            if inside_fn(cur):
                if not inside_fn(prv):
                    out.append(intersect_fn(prv, cur))
                out.append(cur)
            elif inside_fn(prv):
                out.append(intersect_fn(prv, cur))
        return out

    # Half-plane 1: y >= 0
    def inside_y0(p):
        return p[1] >= 0.0

    def isect_y0(a, b):
        t = a[1] / (a[1] - b[1])
        return (a[0] + t * (b[0] - a[0]), 0.0)

    # Half-plane 2: y <= x * sqrt3
    def inside_60(p):
        return p[1] <= p[0] * sqrt3

    def isect_60(a, b):
        fa = a[0] * sqrt3 - a[1]
        fb = b[0] * sqrt3 - b[1]
        t  = fa / (fa - fb)
        return (a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]))

    poly = _sh_clip(list(poly), inside_y0, isect_y0)
    poly = _sh_clip(poly,       inside_60, isect_60)
    return poly


# ====================================================================================================
# ANALYSIS AND PLOTTING
# ====================================================================================================

def analyze_and_plot(data, out_dir, batch, params, target_power_MW):
    """
    Analyze per-channel heating profiles and generate plots.
    """
    show_titles = params.get("show_titles", True)
    is_wedge    = params.get("use_1/6_geometry", False)
    run_dir = out_dir   # output directory alias used throughout function
    z_centers      = data['z_centers']
    heating_2d     = data['heating_2d']       # (n_channels, nz)
    n_channels     = data['n_channels']
    peak_coords    = data['peak_coords']
    integrated_map = data['integrated_2d']
    ll             = data['mesh_lower_left']
    ur             = data['mesh_upper_right']
    asm_q          = data['asm_q']            # (n_asm, nz) assembly q(z)
    asm_centers    = data['asm_centers']
    asm_nchannels  = data['asm_nchannels']
    n_ax           = len(z_centers)

    if n_channels == 0:
        print("  ERROR: No fuel channels found.")
        return

    print(f"\n  Fuel channels (analytical): {n_channels}")
    print(f"  Axial zones:               {n_ax}")

    # ==================================================================
    # METRICS
    # ==================================================================
    integrated   = heating_2d.sum(axis=1)
    peak         = heating_2d.max(axis=1)
    nonzero_mask = integrated > 0
    n_fuel       = int(np.sum(nonzero_mask))

    if n_fuel == 0:
        print("  ERROR: All detected channels have zero heating.")
        return

    hottest_idx     = int(np.argmax(integrated))
    hottest_profile = heating_2d[hottest_idx, :]

    nz_idx      = np.where(nonzero_mask)[0]
    coldest_idx = int(nz_idx[np.argmin(integrated[nz_idx])])
    coldest_profile = heating_2d[coldest_idx, :]

    median_integ   = np.median(integrated[nz_idx])
    median_idx     = int(nz_idx[np.argmin(np.abs(integrated[nz_idx] - median_integ))])
    median_profile = heating_2d[median_idx, :]

    avg_profile = heating_2d[nonzero_mask, :].mean(axis=0)
    std_profile = heating_2d[nonzero_mask, :].std(axis=0)

    avg_axial = np.mean(avg_profile) if np.mean(avg_profile) > 0 else 1.0
    avg_integ = np.mean(integrated[nonzero_mask])

    Fz_hot = (np.max(hottest_profile) / np.mean(hottest_profile)
              if np.mean(hottest_profile) > 0 else 0.0)
    Fz_avg = (np.max(avg_profile) / avg_axial
              if avg_axial > 0 else 0.0)
    Fxy    = integrated[hottest_idx] / avg_integ if avg_integ > 0 else 0.0
    Fq     = peak[hottest_idx] / avg_axial       if avg_axial > 0 else 0.0

    hot_x, hot_y = peak_coords[hottest_idx]

    # ==================================================================
    # PLOT 1 — Hottest vs Average axial profiles
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.plot(z_centers, hottest_profile, 'r-', lw=2.5,
            label=f'Hottest (x={hot_x:.1f}, y={hot_y:.1f} cm)')
    ax.plot(z_centers, avg_profile, 'b-', lw=2.5,
            label=f'Core Average ({n_fuel} ch.)')
    ax.fill_between(z_centers,
                    avg_profile - std_profile,
                    avg_profile + std_profile,
                    alpha=0.2, color='blue', label='\u00b11\u03c3')
    ax.set_xlabel('Axial Position [cm]')
    ax.set_ylabel('Channel Heating q(z) [W]')
    if show_titles:
        ax.set_title(f'Axial Heating \u2014 Hottest vs Average\n'
                     f'{target_power_MW} MW | Fxy={Fxy:.3f} | Fz(hot)={Fz_hot:.3f}')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    p = os.path.join(run_dir, f'batch{batch}_axial_heating_hottest_vs_average.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(fig); del fig
    print(f"  Saved: {p}"); gc.collect()

    # ==================================================================
    # PLOT 2 — Normalized shape comparison
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    mx_h = np.max(hottest_profile) if np.max(hottest_profile) > 0 else 1.0
    mx_a = np.max(avg_profile) if np.max(avg_profile) > 0 else 1.0
    ax.plot(z_centers, hottest_profile / mx_h, 'r-', lw=2.5, label='Hottest (norm.)')
    ax.plot(z_centers, avg_profile / mx_a, 'b--', lw=2.5, label='Average (norm.)')
    zn = (z_centers - z_centers[0]) / (z_centers[-1] - z_centers[0])
    cos_ref = np.cos(np.pi * (zn - 0.5)); cos_ref /= np.max(cos_ref)
    ax.plot(z_centers, cos_ref, 'k:', lw=1.5, alpha=0.5, label='Ref. cosine')
    ax.set_xlabel('Axial Position [cm]'); ax.set_ylabel('Normalized Heating')
    if show_titles:
        ax.set_title(f'Normalized Axial Shape \u2014 {target_power_MW} MW')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.15)
    p = os.path.join(run_dir, f'batch{batch}_axial_heating_shape_comparison.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(fig); del fig
    print(f"  Saved: {p}"); gc.collect()

    # ==================================================================
    # PLOT 3 — Spaghetti (all channels)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    for i in nz_idx:
        if i in (hottest_idx, coldest_idx, median_idx):
            continue
        ax.plot(z_centers, heating_2d[i, :], color='gray', alpha=0.15, lw=0.8)
    ax.plot(z_centers, hottest_profile, 'r-', lw=2.5,
            label=f'Hottest (ch {hottest_idx})')
    ax.plot(z_centers, avg_profile, 'b-', lw=2.5, label='Average')
    med_x, med_y = peak_coords[median_idx]
    ax.plot(z_centers, median_profile, color='orange', lw=2,
            label=f'Median (ch {median_idx}, x={med_x:.1f}, y={med_y:.1f} cm)')
    ax.plot(z_centers, coldest_profile, 'g-', lw=2,
            label=f'Coldest (ch {coldest_idx})')
    ax.set_xlabel('Axial Position [cm]')
    ax.set_ylabel('Channel Heating q(z) [W]')
    if show_titles:
        ax.set_title(f'All Fuel Channel Axial Profiles \u2014 {target_power_MW} MW')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    p = os.path.join(run_dir, f'batch{batch}_axial_heating_all_channels.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(fig); del fig
    print(f"  Saved: {p}"); gc.collect()

    # ==================================================================
    # PLOT 4 — Radial heating distribution (bar chart)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    s_idx  = np.argsort(integrated)[::-1]
    s_int  = integrated[s_idx]
    nz_s   = s_int[s_int > 0]
    n_plot = len(nz_s)
    colors = ['red' if s_idx[i] == hottest_idx
              else ('green' if s_idx[i] == coldest_idx else 'steelblue')
              for i in range(n_plot)]
    ax.bar(range(n_plot), nz_s, color=colors, edgecolor='none')
    ax.axhline(y=avg_integ, color='orange', ls='--', lw=2,
               label=f'Avg = {avg_integ:.2e} W')
    ax.set_xlabel('Fuel Channel (ranked)')
    ax.set_ylabel('Integrated Heating [W]')
    if show_titles:
        ax.set_title(f'Radial Heating Distribution \u2014 '
                     f'{target_power_MW} MW | Fxy = {Fxy:.3f}')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    p = os.path.join(run_dir, f'batch{batch}_radial_heating_distribution.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(fig); del fig
    print(f"  Saved: {p}"); gc.collect()

    # ==================================================================
    # PLOT 5 — 2-D radial heating map with detected channel positions
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    extent = [ll[0], ur[0], ll[1], ur[1]]
    im = ax.imshow(integrated_map.T, origin='lower', extent=extent,
                   cmap='hot', aspect='equal')
    ax.scatter(peak_coords[:, 0], peak_coords[:, 1],
               c='cyan', s=8, alpha=0.6, linewidths=0, label='Fuel compact centroids')
    ax.scatter([hot_x], [hot_y], c='lime', s=50, marker='*',
               edgecolors='black', linewidths=0.5, zorder=5,
               label='Hottest channel')
    ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]')
    if show_titles:
        ax.set_title(f'Integrated Heating Map \u2014 {target_power_MW} MW\n'
                     f'{n_channels} fuel compact centroids (analytical)')
    ax.legend(fontsize=9, loc='upper left')
    plt.colorbar(im, ax=ax, label='Integrated Heating [W]', shrink=0.8)
    p = os.path.join(run_dir, f'batch{batch}_radial_heating_map_channels.png')
    plt.savefig(p, bbox_inches='tight'); plt.close(fig); del fig
    print(f"  Saved: {p}"); gc.collect()

    # ==================================================================
    # ASSEMBLY-LEVEL q(z) ANALYSIS
    # ==================================================================
    n_asm = len(asm_q)
    asm_integrated = asm_q.sum(axis=1)           # total W per assembly
    asm_nonzero = asm_integrated > 0
    n_asm_active = int(np.sum(asm_nonzero))

    if n_asm_active > 0:
        # ------------------------------------------------------------------
        # Symmetry multipliers — for 1/6 geometry, assemblies that sit on a
        # reflection plane are only partially inside the simulated wedge:
        #   center (0,0)       → ×6  (shared by all 6 sectors)
        #   on y = 0 boundary  → ×2  (half outside wedge below x-axis)
        #   on y = x√3 boundary→ ×2  (half outside wedge above 60° line)
        #   interior           → ×1
        # ------------------------------------------------------------------
        bundle_pitch = data['bundle_pitch']
        asm_sym_mult = np.ones(n_asm)
        if is_wedge:
            eps_sym = bundle_pitch * 0.05
            sqrt3   = np.sqrt(3.0)
            for a, (ax_c, ay_c) in enumerate(asm_centers):
                if abs(ax_c) < eps_sym and abs(ay_c) < eps_sym:
                    asm_sym_mult[a] = 6.0
                elif abs(ay_c) < eps_sym:
                    asm_sym_mult[a] = 2.0
                elif abs(ay_c - ax_c * sqrt3) < eps_sym:
                    asm_sym_mult[a] = 2.0

        asm_integrated_corrected = asm_integrated * asm_sym_mult

        hot_asm_idx = int(np.argmax(asm_integrated_corrected))
        hot_asm_q   = asm_q[hot_asm_idx, :]
        avg_asm_q   = asm_q[asm_nonzero, :].mean(axis=0)

        hot_asm_x, hot_asm_y = asm_centers[hot_asm_idx]
        hot_asm_nch = int(asm_nchannels[hot_asm_idx])
        avg_asm_total_corrected = np.mean(asm_integrated_corrected[asm_nonzero])
        asm_radial_pf = (asm_integrated_corrected[hot_asm_idx] / avg_asm_total_corrected
                         if avg_asm_total_corrected > 0 else 0.0)

        print(f"\n  --- Assembly-level results ---")
        print(f"  Active assemblies:    {n_asm_active}")
        print(f"  Hottest assembly:     {hot_asm_idx} at "
              f"({hot_asm_x:.1f}, {hot_asm_y:.1f}) cm, "
              f"{hot_asm_nch} fuel channels, sym×{asm_sym_mult[hot_asm_idx]:.0f}")
        print(f"  Hottest asm total q:  {asm_integrated[hot_asm_idx]:.4e} W (raw) "
              f"→ {asm_integrated_corrected[hot_asm_idx]:.4e} W (×sym)")
        print(f"  Average asm total q:  {np.mean(asm_integrated[asm_nonzero]):.4e} W (raw) "
              f"→ {avg_asm_total_corrected:.4e} W (×sym)")
        print(f"  Assembly radial PF:   {asm_radial_pf:.4f}")

        # ==============================================================
        # PLOT 6 — Assembly q(z) profiles
        # ==============================================================
        fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
        ax.plot(z_centers, hot_asm_q, 'r-', lw=2.5,
                label=f'Hottest asm ({hot_asm_x:.0f},{hot_asm_y:.0f}) cm')
        ax.plot(z_centers, avg_asm_q, 'b-', lw=2.5,
                label=f'Average ({n_asm_active} assemblies)')
        # Spaghetti for all assemblies
        for a in range(n_asm):
            if not asm_nonzero[a] or a == hot_asm_idx:
                continue
            ax.plot(z_centers, asm_q[a, :], color='gray', alpha=0.3, lw=0.8)
        ax.set_xlabel('Axial Position [cm]')
        ax.set_ylabel('Assembly Heating Rate q(z) [W / axial level]')
        if show_titles:
            ax.set_title(f'Assembly Axial Heating q(z) \u2014 {target_power_MW} MW')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        p = os.path.join(run_dir,
                         f'batch{batch}_assembly_axial_heating_qz.png')
        plt.savefig(p, bbox_inches='tight'); plt.close(fig); del fig
        print(f"  Saved: {p}"); gc.collect()

        # ==============================================================
        # CSV — assembly q(z) profiles (raw values)
        # ==============================================================
        asm_csv = os.path.join(run_dir,
                               f'batch{batch}_assembly_qz_profiles.csv')
        hdr_asm = 'z_center_cm,hottest_asm_q_W,average_asm_q_W'
        np.savetxt(asm_csv,
                   np.column_stack([z_centers, hot_asm_q, avg_asm_q]),
                   delimiter=',', header=hdr_asm, comments='')
        print(f"  Saved: {asm_csv}")

        # CSV — all assemblies q(z)
        all_asm_csv = os.path.join(run_dir,
                                   f'batch{batch}_all_assemblies_qz.csv')
        with open(all_asm_csv, 'w') as fout:
            # Header: z, asm_0, asm_1, ...
            cols = ['z_center_cm'] + [f'asm_{a}_q_W' for a in range(n_asm)]
            fout.write(','.join(cols) + '\n')
            for iz in range(n_ax):
                row = [f'{z_centers[iz]:.4f}']
                for a in range(n_asm):
                    row.append(f'{asm_q[a, iz]:.6e}')
                fout.write(','.join(row) + '\n')
        print(f"  Saved: {all_asm_csv}")

        # ==============================================================
        # PLOT 7 — Assembly radial heating map with hex bounding boxes
        # ==============================================================
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        extent = [ll[0], ur[0], ll[1], ur[1]]
        im = ax.imshow(integrated_map.T, origin='lower', extent=extent,
                       cmap='hot', aspect='equal')
        plt.colorbar(im, ax=ax, label='Integrated Heating [W]', shrink=0.8)

        # Flat-top hex: vertices at 0°+k*60° (first vertex due east)
        R_hex = bundle_pitch / np.sqrt(3.0)
        hex_angles = np.radians([k * 60 for k in range(6)])

        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection

        patches = []
        for a, (cx, cy) in enumerate(asm_centers):
            verts_full = [(cx + R_hex * np.cos(ang),
                           cy + R_hex * np.sin(ang)) for ang in hex_angles[:6]]
            if is_wedge:
                verts = _clip_polygon_to_wedge(verts_full)
            else:
                verts = verts_full
            if len(verts) >= 3:
                patches.append(MplPolygon(verts, closed=True))

        col = PatchCollection(patches, facecolor='none', edgecolor='white',
                              linewidth=0.8, alpha=0.8)
        ax.add_collection(col)

        # Mark all assembly centroids
        asm_arr = np.array(asm_centers)
        ax.scatter(asm_arr[:, 0], asm_arr[:, 1],
                   c='deepskyblue', s=12, alpha=0.7, linewidths=0,
                   zorder=3, label='Assembly centroids')

        # Highlight hottest assembly
        ax.scatter([hot_asm_x], [hot_asm_y], c='lime', s=80, marker='*',
                   edgecolors='black', linewidths=0.5, zorder=5,
                   label=f'Hottest asm ({hot_asm_x:.0f},{hot_asm_y:.0f}) cm')

        ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]')
        if show_titles:
            ax.set_title(f'Assembly Heating Map \u2014 {target_power_MW} MW\n'
                         f'Asm radial PF = {asm_radial_pf:.3f} '
                         f'(sym-corrected, {n_asm_active} assemblies)')
        ax.legend(fontsize=9, loc='upper left')

        # For 1/6 geometry clip axes to wedge extent
        if is_wedge:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        p = os.path.join(run_dir, f'batch{batch}_radial_heating_map_assemblies.png')
        plt.savefig(p, bbox_inches='tight'); plt.close(fig); del fig
        print(f"  Saved: {p}"); gc.collect()

    else:
        hot_asm_q = None
        avg_asm_q = None
        hot_asm_idx = -1
        asm_integrated_corrected = None
        asm_sym_mult = None
        asm_radial_pf = 0.0

    # ==================================================================
    # POWER BALANCE — sum all fuel channels at each axial level
    # ==================================================================
    sym            = 6 if is_wedge else 1
    total_qz       = heating_2d.sum(axis=0)          # (nz,) W per axial level
    total_ch_power = total_qz.sum()                  # W, channel voxels in sim domain
    # heating_3d is normalized to the sim domain (P_target / sym for wedge),
    # so compare against the same domain power for a meaningful fraction.
    domain_power_W = target_power_MW * 1e6 / sym
    power_fraction = total_ch_power / domain_power_W if domain_power_W > 0 else 0.0

    print(f"\n  --- Power balance (fuel channels) ---")
    print(f"  Geometry:                     {'1/6 Wedge' if is_wedge else 'Full Core'}")
    print(f"  Full-core target power:       {target_power_MW * 1e6:.4e} W")
    print(f"  Sim-domain target power:      {domain_power_W:.4e} W  (÷{sym})")
    print(f"  Sum of all channel heating:   {total_ch_power:.4e} W")
    print(f"  Fraction of domain target:    {power_fraction:.4f}  "
          f"({power_fraction*100:.2f}%)")
    print(f"  (Note: channel sum uses compact-radius voxels; "
          f"moderator/reflector heating excluded)")

    # CSV — total q(z) summed over all channels
    total_qz_csv = os.path.join(run_dir, f'batch{batch}_total_channel_qz.csv')
    hdr_tot = 'z_center_cm,total_all_channels_q_W'
    np.savetxt(total_qz_csv,
               np.column_stack([z_centers, total_qz]),
               delimiter=',', header=hdr_tot, comments='')
    print(f"  Saved: {total_qz_csv}")

    # ==================================================================
    # CSV — axial profiles
    # ==================================================================
    csv_path = os.path.join(run_dir, f'batch{batch}_axial_heating_profiles.csv')
    hdr = 'z_center_cm,hottest_channel_q_W,average_channel_q_W,std_channel_q_W,median_channel_q_W,coldest_channel_q_W'
    np.savetxt(csv_path,
               np.column_stack([z_centers, hottest_profile, avg_profile,
                                std_profile, median_profile, coldest_profile]),
               delimiter=',', header=hdr, comments='')
    print(f"  Saved: {csv_path}")

    # CSV — channel positions
    ch_csv = os.path.join(run_dir, f'batch{batch}_fuel_channel_positions.csv')
    hdr2 = 'channel_id,x_cm,y_cm,integrated_heating_W'
    with open(ch_csv, 'w') as fout:
        fout.write(hdr2 + '\n')
        for k in range(n_channels):
            fout.write(f'{k},{peak_coords[k,0]:.4f},{peak_coords[k,1]:.4f},'
                       f'{integrated[k]:.6e}\n')
    print(f"  Saved: {ch_csv}")

    # ==================================================================
    # Summary text
    # ==================================================================
    txt = os.path.join(run_dir, f'batch{batch}_heating_profile_summary.txt')
    with open(txt, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write(f'HEATING PROFILE ANALYSIS SUMMARY (Batch {batch})\n')
        f.write('=' * 80 + '\n\n')
        f.write(f'Target Power:                    {target_power_MW} MW\n')
        f.write(f'Extraction method:               Mesh tally + analytical centroids\n')
        f.write(f'Number of fuel channels:         {n_fuel}\n')
        f.write(f'Number of axial zones:           {n_ax}\n\n')

        f.write('-' * 60 + '\n')
        f.write('HOTTEST FUEL CHANNEL\n')
        f.write('-' * 60 + '\n')
        f.write(f'  Channel index:         {hottest_idx}\n')
        f.write(f'  Position:              ({hot_x:.2f}, {hot_y:.2f}) cm\n')
        f.write(f'  Integrated heating:    {integrated[hottest_idx]:.4e} W\n')
        f.write(f'  Peak axial heating:    {peak[hottest_idx]:.4e} W\n\n')

        cold_x, cold_y = peak_coords[coldest_idx]
        f.write('-' * 60 + '\n')
        f.write('COLDEST FUEL CHANNEL\n')
        f.write('-' * 60 + '\n')
        f.write(f'  Channel index:         {coldest_idx}\n')
        f.write(f'  Position:              ({cold_x:.2f}, {cold_y:.2f}) cm\n')
        f.write(f'  Integrated heating:    {integrated[coldest_idx]:.4e} W\n')
        f.write(f'  Peak axial heating:    {peak[coldest_idx]:.4e} W\n\n')

        f.write('-' * 60 + '\n')
        f.write('MEDIAN FUEL CHANNEL\n')
        f.write('-' * 60 + '\n')
        f.write(f'  Channel index:         {median_idx}\n')
        f.write(f'  Position:              ({med_x:.2f}, {med_y:.2f}) cm\n')
        f.write(f'  Integrated heating:    {integrated[median_idx]:.4e} W\n')
        f.write(f'  Peak axial heating:    {peak[median_idx]:.4e} W\n')
        f.write(f'  Median integrated q:   {median_integ:.4e} W  '
                f'(channel nearest to median)\n\n')

        f.write('-' * 60 + '\n')
        f.write('PEAKING FACTORS\n')
        f.write('-' * 60 + '\n')
        f.write(f'  Radial  (Fxy):        {Fxy:.4f}\n')
        f.write(f'  Axial   (Fz, hot):    {Fz_hot:.4f}\n')
        f.write(f'  Axial   (Fz, avg):    {Fz_avg:.4f}\n')
        f.write(f'  Total   (Fq):         {Fq:.4f}\n\n')

        f.write('-' * 60 + '\n')
        f.write('POWER BALANCE — ALL FUEL CHANNELS\n')
        f.write('-' * 60 + '\n')
        f.write(f'  Geometry:                     {"1/6 Wedge" if is_wedge else "Full Core"}\n')
        f.write(f'  Full-core target power:       {target_power_MW * 1e6:.4e} W\n')
        f.write(f'  Sim-domain target power:      {domain_power_W:.4e} W  (÷{sym})\n')
        f.write(f'  Sum of all channel heating:   {total_ch_power:.4e} W\n')
        f.write(f'  Fraction of domain target:    {power_fraction:.4f} '
                f'({power_fraction*100:.2f}%)\n')
        f.write(f'  Note: channel sum uses compact-radius voxels only;\n')
        f.write(f'        moderator/reflector heating is not included.\n\n')
        f.write(f'  Total channel q(z) [W per axial level, all channels summed]:\n')
        for iz in range(n_ax):
            f.write(f'    z={z_centers[iz]:7.2f} cm:  {total_qz[iz]:.4e} W\n')
        f.write('\n')

        if hot_asm_q is not None:
            f.write('-' * 60 + '\n')
            f.write('ASSEMBLY-LEVEL HEATING\n')
            f.write('-' * 60 + '\n')
            f.write(f'  Active assemblies:     {n_asm_active}\n')
            f.write(f'  Hottest assembly:      {hot_asm_idx} at '
                    f'({asm_centers[hot_asm_idx,0]:.1f}, '
                    f'{asm_centers[hot_asm_idx,1]:.1f}) cm, '
                    f'sym×{asm_sym_mult[hot_asm_idx]:.0f}\n')
            f.write(f'  Hottest asm total q:   {asm_integrated[hot_asm_idx]:.4e} W (raw)\n')
            f.write(f'  Hottest asm total q:   {asm_integrated_corrected[hot_asm_idx]:.4e} W (sym-corrected)\n')
            f.write(f'  Average asm total q:   {np.mean(asm_integrated[asm_nonzero]):.4e} W (raw)\n')
            f.write(f'  Average asm total q:   {avg_asm_total_corrected:.4e} W (sym-corrected)\n')
            f.write(f'  Assembly radial PF:    {asm_radial_pf:.4f}  (sym-corrected)\n\n')

            f.write('  Hottest Assembly q(z) [W per axial level]:\n')
            for iz in range(n_ax):
                f.write(f'    z={z_centers[iz]:7.2f} cm:  '
                        f'{hot_asm_q[iz]:.4e} W\n')
            f.write('\n')
            f.write('  Average Assembly q(z) [W per axial level]:\n')
            for iz in range(n_ax):
                f.write(f'    z={z_centers[iz]:7.2f} cm:  '
                        f'{avg_asm_q[iz]:.4e} W\n')
            f.write('\n')

        f.write('-' * 60 + '\n')
        f.write('ALL FUEL CHANNELS \u2014 INTEGRATED HEATING (RANKED)\n')
        f.write('-' * 60 + '\n')
        rank = 0
        for idx in s_idx:
            if integrated[idx] <= 0:
                continue
            rank += 1
            rpf = integrated[idx] / avg_integ if avg_integ > 0 else 0
            cx, cy = peak_coords[idx]
            f.write(f'  {rank:4d}. Ch {idx:4d} ({cx:6.1f},{cy:6.1f}): '
                    f'{integrated[idx]:.4e} W  (PF = {rpf:.3f})\n')
        f.write('\n' + '=' * 80 + '\n')
    print(f"  Saved: {txt}")

    # Console
    print(f"\n  {'='*60}")
    print(f"  HEATING PROFILE RESULTS")
    print(f"  {'='*60}")
    print(f"  Fuel channels analysed:       {n_fuel} (of {n_channels} analytical)")
    print(f"  Hottest channel:              {hottest_idx} at ({hot_x:.1f}, {hot_y:.1f}) cm")
    print(f"  Radial peaking factor (Fxy):  {Fxy:.4f}")
    print(f"  Axial peaking factor  (Fz):   {Fz_hot:.4f}")
    print(f"  Total peaking factor  (Fq):   {Fq:.4f}")
    if hot_asm_q is not None:
        avg_asm_total = np.mean(asm_integrated[asm_nonzero])
        print(f"  ---")
        print(f"  Assemblies detected:          {n_asm_active}")
        print(f"  Hottest assembly:             {hot_asm_idx} "
              f"({asm_centers[hot_asm_idx,0]:.1f}, "
              f"{asm_centers[hot_asm_idx,1]:.1f}) cm")
        print(f"  Hottest asm total q:          {asm_integrated[hot_asm_idx]:.4e} W")
        print(f"  Average asm total q:          {avg_asm_total:.4e} W")
        print(f"  Assembly radial PF:           "
              f"{asm_integrated[hot_asm_idx]/avg_asm_total:.4f}")
    print(f"  {'='*60}")


# ====================================================================================================
# MAIN ENTRY POINT
# ====================================================================================================

def run_heating_profile_extraction(run_dir, params, batch=None):
    """Run the full heating profile extraction and analysis."""

    print(f"\n{'='*80}")
    print("HEATING PROFILE EXTRACTION (Mesh Tally + Analytical Centroids)")
    print(f"{'='*80}")
    print(f"Run directory: {run_dir}")

    # --- Find batch ---
    if batch is None:
        for f in os.listdir(run_dir):
            if f.startswith('statepoint') and f.endswith('.h5'):
                batch = int(f.split('.')[1])
                break
    if batch is None:
        print("ERROR: No statepoint file found!"); return

    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    if not os.path.exists(sp_path):
        print(f"ERROR: Not found: {sp_path}"); return

    print(f"Batch number: {batch}")

    is_wedge     = params.get("use_1/6_geometry", False)
    target_power = params.get("thermal_power_MW",
                              params.get("thermal_power", 15.0))
    sym          = 6 if is_wedge else 1

    print(f"Geometry: {'1/6 Wedge' if is_wedge else 'Full Core'}")
    print(f"Target power: {target_power} MW")

    # --- Normalization ---
    source_per_sec = get_normalization_factor(sp_path, target_power)
    print(f"Source rate: {source_per_sec:.3e} source/s")

    # --- Extract from mesh tally ---
    try:
        print("\nExtracting heating profiles from mesh_heating tally...")
        data = extract_mesh_heating(sp_path, source_per_sec, params, sym)
    except KeyError as e:
        print(f"  ERROR: {e}"); return
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc(); return

    # --- Analyze ---
    out_dir = os.path.join(run_dir, "heating_profile_results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    analyze_and_plot(data, out_dir, batch, params, target_power)

    del data; gc.collect()

    print(f"\n{'='*80}")
    print("HEATING PROFILE EXTRACTION COMPLETE")
    print(f"{'='*80}\n")


# ====================================================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python heating_profile_extraction.py "
              "<run_directory> [batch_number]")
        sys.exit(1)

    run_dir = sys.argv[1]
    batch   = int(sys.argv[2]) if len(sys.argv) > 2 else None

    params_path = os.path.join(run_dir, 'run_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as fp:
            params = json.load(fp)
    else:
        print("ERROR: run_params.json not found."); sys.exit(1)

    run_heating_profile_extraction(run_dir, params, batch)