"""
Heating Profile Extraction Post-Processing Script

Extracts axial heating profiles for the hottest fuel channel and the
core-average fuel channel from the mesh_heating tally in an OpenMC
statepoint file.

APPROACH:
    Instead of DistribcellFilter tallies (which create enormous output
    because OpenMC enumerates every geometry path to each fuel cell),
    this script uses the existing RegularMesh heating tally.

    The 250x217x50 mesh has 0.36 cm cells — much finer than the 2.5 cm
    fuel channel pitch.  Each fuel compact (1.27 cm diameter) covers
    ~3.5 mesh cells across, so individual channels are clearly resolved.

    The script:
      1. Reads the mesh_heating tally via openmc.StatePoint
      2. Reshapes to 3-D  (nx, ny, nz)
      3. Sums over z to form a 2-D radial heating map
      4. Detects local maxima -> one peak per fuel channel centre
      5. Extracts the axial z-profile at each peak
      6. Identifies the hottest and average profiles, computes peaking
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
# MESH-BASED FUEL CHANNEL EXTRACTION
# ====================================================================================================

def _find_local_maxima_2d(arr, threshold_frac=0.05, min_separation=3):
    """
    Find local maxima in a 2-D array.

    A pixel is a local maximum if it exceeds all 8 neighbours and
    is above threshold_frac * global_max.  Peaks closer than
    min_separation pixels are merged (keep the brighter one).

    Returns array of (iy, ix) peak positions, shape (n_peaks, 2).
    """
    threshold = threshold_frac * np.max(arr)
    ny, nx = arr.shape

    # Pad array to simplify boundary handling
    padded = np.pad(arr, 1, mode='constant', constant_values=0)

    # Check each pixel against its 8 neighbours
    is_max = np.ones((ny, nx), dtype=bool)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            neighbour = padded[1+di:ny+1+di, 1+dj:nx+1+dj]
            is_max &= (arr >= neighbour)

    is_max &= (arr > threshold)

    # Get peak positions sorted by brightness (descending)
    peak_ys, peak_xs = np.where(is_max)
    peak_vals = arr[peak_ys, peak_xs]
    order = np.argsort(-peak_vals)
    peak_ys = peak_ys[order]
    peak_xs = peak_xs[order]
    peak_vals = peak_vals[order]

    # Suppress peaks too close to a brighter peak
    keep_idx = []
    keep_pos = []
    for k in range(len(peak_ys)):
        y, x = peak_ys[k], peak_xs[k]
        too_close = False
        for (ky, kx) in keep_pos:
            if abs(y - ky) < min_separation and abs(x - kx) < min_separation:
                too_close = True
                break
        if not too_close:
            keep_idx.append(k)
            keep_pos.append((y, x))

    if not keep_pos:
        return np.empty((0, 2), dtype=int)

    # ---- Median-based filter to reject non-fuel peaks ----
    # Fuel channels (TRISO fission heating) are far brighter than
    # control rods / poison rods (gamma absorption only).  Reject any
    # peak whose value is below 30% of the median of the kept peaks.
    kept_vals = peak_vals[keep_idx]
    median_val = np.median(kept_vals)
    fuel_cutoff = 0.30 * median_val

    filtered = []
    n_rejected = 0
    for i, (y, x) in enumerate(keep_pos):
        if kept_vals[i] >= fuel_cutoff:
            filtered.append((y, x))
        else:
            n_rejected += 1
    if n_rejected > 0:
        print(f"  Peak filter: kept {len(filtered)}, "
              f"rejected {n_rejected} sub-threshold peaks "
              f"(cutoff = 30% of median = {fuel_cutoff:.2e})")

    return np.array(filtered, dtype=int) if filtered else np.empty((0, 2), dtype=int)


def _cluster_into_assemblies(peak_coords, bundle_pitch):
    """
    Cluster fuel channel peaks into assemblies using spatial proximity.

    Two channels belong to the same assembly if they are within
    bundle_pitch * 0.55 of each other (transitive closure).
    Returns list of arrays, each containing indices of channels in
    that assembly.
    """
    n = len(peak_coords)
    link_dist = bundle_pitch * 0.55  # well above max intra-assembly (~10 cm)

    # Union-find
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((peak_coords[i, 0] - peak_coords[j, 0])**2 +
                           (peak_coords[i, 1] - peak_coords[j, 1])**2)
            if dist < link_dist:
                union(i, j)

    # Group by root
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    return list(groups.values())


def _point_in_hex(px, py, cx, cy, apothem):
    """Test if point (px, py) is inside a flat-topped hex centred at (cx, cy)."""
    dx = abs(px - cx)
    dy = abs(py - cy)
    # Flat-topped hex with apothem = inradius (half flat-to-flat distance)
    if dx > apothem or dy > apothem * 2 / np.sqrt(3):
        return False
    return apothem * 2 / np.sqrt(3) - dy >= dx / np.sqrt(3)


def extract_assembly_heating(heating_3d, mesh_ll, mesh_ur, peak_coords,
                             assembly_groups, bundle_pitch):
    """
    Compute q(z) for each assembly by summing all mesh voxels within
    its hex footprint at each axial level.

    Parameters
    ----------
    heating_3d : (nx, ny, nz) array in watts per voxel
    mesh_ll, mesh_ur : mesh bounding box
    peak_coords : (n_channels, 2) fuel channel positions
    assembly_groups : list of index-lists from clustering
    bundle_pitch : assembly flat-to-flat distance

    Returns
    -------
    asm_q : (n_assemblies, nz) array — total heating [W] per axial level
    asm_centers : (n_assemblies, 2) array — (x, y) of assembly centres
    """
    nx, ny, nz = heating_3d.shape
    dx = (mesh_ur[0] - mesh_ll[0]) / nx
    dy = (mesh_ur[1] - mesh_ll[1]) / ny

    x_centers = np.linspace(mesh_ll[0] + dx/2, mesh_ur[0] - dx/2, nx)
    y_centers = np.linspace(mesh_ll[1] + dy/2, mesh_ur[1] - dy/2, ny)

    n_asm = len(assembly_groups)
    apothem = bundle_pitch / 2.0  # hex inradius

    # Compute assembly centres as centroid of their fuel channels
    asm_centers = np.zeros((n_asm, 2))
    for a, group in enumerate(assembly_groups):
        asm_centers[a, 0] = np.mean(peak_coords[group, 0])
        asm_centers[a, 1] = np.mean(peak_coords[group, 1])

    # For each assembly, build a mask of (ix, iy) voxels inside its hex
    asm_q = np.zeros((n_asm, nz))
    for a in range(n_asm):
        cx, cy = asm_centers[a]
        # Bounding box in pixel coords to limit search
        ix_lo = max(0, int((cx - apothem - mesh_ll[0]) / dx) - 1)
        ix_hi = min(nx, int((cx + apothem - mesh_ll[0]) / dx) + 2)
        iy_lo = max(0, int((cy - apothem - mesh_ll[1]) / dy) - 1)
        iy_hi = min(ny, int((cy + apothem - mesh_ll[1]) / dy) + 2)

        for ix in range(ix_lo, ix_hi):
            for iy in range(iy_lo, iy_hi):
                if _point_in_hex(x_centers[ix], y_centers[iy], cx, cy, apothem):
                    asm_q[a, :] += heating_3d[ix, iy, :]

    return asm_q, asm_centers


def extract_mesh_heating(sp_path, source_per_sec, params, symmetry_factor):
    """
    Extract per-fuel-channel axial heating profiles from the mesh_heating
    tally using openmc.StatePoint.

    Steps:
      1. Read mesh_heating tally and mesh geometry
      2. Reshape to 3-D (nx, ny, nz)
      3. Sum over z -> 2-D integrated heating map
      4. Find local maxima -> fuel channel centres
      5. Extract z-profile at each peak -> per-channel profiles

    Returns
    -------
    dict with keys: 'z_centers', 'heating_2d', 'n_channels',
                    'peak_coords', 'mesh_lower_left', 'mesh_upper_right',
                    'heating_3d', 'integrated_2d'
    """
    joule_per_ev = 1.60218e-19

    # ------------------------------------------------------------------
    # Read mesh_heating tally via StatePoint
    # ------------------------------------------------------------------
    sp = openmc.StatePoint(sp_path)
    tally = sp.get_tally(name='mesh_heating')
    mesh = tally.find_filter(openmc.MeshFilter).mesh

    nx, ny, nz = mesh.dimension
    ll = mesh.lower_left.copy()
    ur = mesh.upper_right.copy()

    # tally.mean has shape (nx*ny*nz, 1, 1) for single score/nuclide
    mean_flat = tally.mean[:, 0, 0].copy()

    sp.close()
    del sp, tally, mesh
    gc.collect()

    print(f"  Mesh dimensions: {nx} x {ny} x {nz}")
    print(f"  Mesh bounds: ({ll[0]:.1f},{ll[1]:.1f},{ll[2]:.1f}) -> "
          f"({ur[0]:.1f},{ur[1]:.1f},{ur[2]:.1f}) cm")

    # Axial coordinates from mesh bounds
    z_edges = np.linspace(ll[2], ur[2], nz + 1)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0

    # ------------------------------------------------------------------
    # Reshape and convert to physical heating (watts per voxel)
    # ------------------------------------------------------------------
    # OpenMC stores mesh tally in column-major order: x varies fastest
    # Reshape to (nz, ny, nx) then transpose to (nx, ny, nz)
    heating_3d = mean_flat.reshape((nz, ny, nx)).transpose((2, 1, 0))

    conv = source_per_sec * joule_per_ev / symmetry_factor
    heating_3d *= conv  # now in watts per voxel

    print(f"  Total mesh heating: {heating_3d.sum():.3e} W")
    print(f"  Peak voxel heating: {heating_3d.max():.3e} W")

    del mean_flat
    gc.collect()

    # ------------------------------------------------------------------
    # 2-D integrated heating map  (sum over z)
    # ------------------------------------------------------------------
    integrated_2d = heating_3d.sum(axis=2)  # shape (nx, ny)

    # ------------------------------------------------------------------
    # Detect fuel channel centres as local maxima
    # ------------------------------------------------------------------
    map_yx = integrated_2d.T  # shape (ny, nx) for image-like indexing

    dx = (ur[0] - ll[0]) / nx
    dy = (ur[1] - ll[1]) / ny
    channel_pitch = params.get("fuel_to_coolant_distance", 2.5)
    min_sep = max(2, int(channel_pitch / max(dx, dy) * 0.6))

    peaks = _find_local_maxima_2d(map_yx, threshold_frac=0.05, min_separation=min_sep)

    if len(peaks) == 0:
        raise RuntimeError("No fuel channel peaks found in mesh heating map. "
                           "Check that mesh_heating tally is correctly configured.")

    n_channels = len(peaks)
    print(f"  Detected {n_channels} fuel channel centres (local maxima)")

    # ------------------------------------------------------------------
    # Extract z-profile at each peak by summing voxels within fuel
    # channel radius (× 1.2 to capture all intersecting voxels)
    # ------------------------------------------------------------------
    # peaks[:, 0] = iy, peaks[:, 1] = ix  (image convention)
    x_centers = np.linspace(ll[0] + dx/2, ur[0] - dx/2, nx)
    y_centers = np.linspace(ll[1] + dy/2, ur[1] - dy/2, ny)
    peak_coords = np.array([(x_centers[peaks[k, 1]], y_centers[peaks[k, 0]])
                            for k in range(n_channels)])

    compact_r = params.get("compact_radius", 0.635)
    capture_r = compact_r * 1.2
    # Max pixel offset to search (bounding box)
    search_px = int(np.ceil(capture_r / dx)) + 1
    search_py = int(np.ceil(capture_r / dy)) + 1
    capture_r2 = capture_r ** 2

    print(f"  Fuel compact radius: {compact_r:.3f} cm, "
          f"capture radius (×1.2): {capture_r:.3f} cm")

    heating_profiles = np.zeros((n_channels, nz))
    voxels_per_channel = []
    for k in range(n_channels):
        iy0, ix0 = peaks[k]
        cx = x_centers[ix0]
        cy = y_centers[iy0]
        n_vox = 0
        for dix in range(-search_px, search_px + 1):
            ix = ix0 + dix
            if ix < 0 or ix >= nx:
                continue
            for diy in range(-search_py, search_py + 1):
                iy = iy0 + diy
                if iy < 0 or iy >= ny:
                    continue
                dist2 = (x_centers[ix] - cx)**2 + (y_centers[iy] - cy)**2
                if dist2 <= capture_r2:
                    heating_profiles[k, :] += heating_3d[ix, iy, :]
                    n_vox += 1
        voxels_per_channel.append(n_vox)

    avg_vox = np.mean(voxels_per_channel)
    print(f"  Voxels summed per channel: avg {avg_vox:.1f}, "
          f"range [{min(voxels_per_channel)}, {max(voxels_per_channel)}]")

    # ------------------------------------------------------------------
    # Cluster fuel channels into assemblies and extract assembly q(z)
    # ------------------------------------------------------------------
    bundle_pitch = 5 * channel_pitch * np.sqrt(3.0)
    assembly_groups = _cluster_into_assemblies(peak_coords, bundle_pitch)
    n_asm = len(assembly_groups)
    print(f"  Clustered into {n_asm} fuel assemblies")

    asm_q, asm_centers = extract_assembly_heating(
        heating_3d, ll, ur, peak_coords, assembly_groups, bundle_pitch)

    # Assembly channel counts for diagnostics
    asm_ch_counts = [len(g) for g in assembly_groups]
    for a in range(n_asm):
        cx, cy = asm_centers[a]
        print(f"    Asm {a}: ({cx:6.1f}, {cy:6.1f}) cm — "
              f"{asm_ch_counts[a]} fuel channels")

    return {
        'z_centers':        z_centers,
        'heating_2d':       heating_profiles,   # (n_channels, nz) per-channel
        'n_channels':       n_channels,
        'peak_coords':      peak_coords,        # (n_channels, 2) as (x, y)
        'mesh_lower_left':  ll,
        'mesh_upper_right': ur,
        'heating_3d':       heating_3d,         # (nx, ny, nz)
        'integrated_2d':    integrated_2d,      # (nx, ny)
        'asm_q':            asm_q,              # (n_asm, nz) watts per axial level
        'asm_centers':      asm_centers,        # (n_asm, 2) assembly positions
        'asm_groups':       assembly_groups,    # list of channel-index lists
        'bundle_pitch':     bundle_pitch,
    }


# ====================================================================================================
# ANALYSIS AND PLOTTING
# ====================================================================================================

def analyze_and_plot(data, run_dir, batch, params, target_power_MW):
    """
    Analyze per-channel heating profiles and generate plots.
    """
    z_centers      = data['z_centers']
    heating_2d     = data['heating_2d']       # (n_channels, nz)
    n_channels     = data['n_channels']
    peak_coords    = data['peak_coords']
    integrated_map = data['integrated_2d']
    ll             = data['mesh_lower_left']
    ur             = data['mesh_upper_right']
    asm_q          = data['asm_q']            # (n_asm, nz) assembly q(z)
    asm_centers    = data['asm_centers']
    asm_groups     = data['asm_groups']
    n_ax           = len(z_centers)

    if n_channels == 0:
        print("  ERROR: No fuel channels found.")
        return

    print(f"\n  Fuel channels detected: {n_channels}")
    print(f"  Axial zones:           {n_ax}")

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
        if i == hottest_idx or i == coldest_idx:
            continue
        ax.plot(z_centers, heating_2d[i, :], color='gray', alpha=0.15, lw=0.8)
    ax.plot(z_centers, hottest_profile, 'r-', lw=2.5,
            label=f'Hottest (ch {hottest_idx})')
    ax.plot(z_centers, avg_profile, 'b-', lw=2.5, label='Average')
    ax.plot(z_centers, coldest_profile, 'g-', lw=2,
            label=f'Coldest (ch {coldest_idx})')
    ax.set_xlabel('Axial Position [cm]')
    ax.set_ylabel('Channel Heating q(z) [W]')
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
               c='cyan', s=8, alpha=0.6, linewidths=0, label='Detected channels')
    ax.scatter([hot_x], [hot_y], c='lime', s=50, marker='*',
               edgecolors='black', linewidths=0.5, zorder=5,
               label='Hottest channel')
    ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]')
    ax.set_title(f'Integrated Heating Map \u2014 {target_power_MW} MW\n'
                 f'{n_channels} channels detected')
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
        hot_asm_idx = int(np.argmax(asm_integrated))
        hot_asm_q   = asm_q[hot_asm_idx, :]
        avg_asm_q   = asm_q[asm_nonzero, :].mean(axis=0)

        hot_asm_x, hot_asm_y = asm_centers[hot_asm_idx]
        hot_asm_nch = len(asm_groups[hot_asm_idx])

        print(f"\n  --- Assembly-level results ---")
        print(f"  Active assemblies:    {n_asm_active}")
        print(f"  Hottest assembly:     {hot_asm_idx} at "
              f"({hot_asm_x:.1f}, {hot_asm_y:.1f}) cm, "
              f"{hot_asm_nch} fuel channels")
        print(f"  Hottest asm total q:  {asm_integrated[hot_asm_idx]:.4e} W")
        print(f"  Average asm total q:  {np.mean(asm_integrated[asm_nonzero]):.4e} W")

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
    else:
        hot_asm_q = None
        avg_asm_q = None
        hot_asm_idx = -1

    # ==================================================================
    # CSV — axial profiles
    # ==================================================================
    csv_path = os.path.join(run_dir, f'batch{batch}_axial_heating_profiles.csv')
    hdr = 'z_center_cm,hottest_channel_q_W,average_channel_q_W,std_channel_q_W,coldest_channel_q_W'
    np.savetxt(csv_path,
               np.column_stack([z_centers, hottest_profile, avg_profile,
                                std_profile, coldest_profile]),
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
        f.write(f'Extraction method:               Mesh tally + peak detection\n')
        f.write(f'Number of fuel channels detected: {n_fuel}\n')
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
        f.write('PEAKING FACTORS\n')
        f.write('-' * 60 + '\n')
        f.write(f'  Radial  (Fxy):        {Fxy:.4f}\n')
        f.write(f'  Axial   (Fz, hot):    {Fz_hot:.4f}\n')
        f.write(f'  Axial   (Fz, avg):    {Fz_avg:.4f}\n')
        f.write(f'  Total   (Fq):         {Fq:.4f}\n\n')

        if hot_asm_q is not None:
            f.write('-' * 60 + '\n')
            f.write('ASSEMBLY-LEVEL HEATING\n')
            f.write('-' * 60 + '\n')
            f.write(f'  Active assemblies:     {n_asm_active}\n')
            f.write(f'  Hottest assembly:      {hot_asm_idx} at '
                    f'({asm_centers[hot_asm_idx,0]:.1f}, '
                    f'{asm_centers[hot_asm_idx,1]:.1f}) cm\n')
            f.write(f'  Hottest asm total q:   {asm_integrated[hot_asm_idx]:.4e} W\n')
            avg_asm_total = np.mean(asm_integrated[asm_nonzero])
            f.write(f'  Average asm total q:   {avg_asm_total:.4e} W\n')
            f.write(f'  Assembly radial PF:    '
                    f'{asm_integrated[hot_asm_idx]/avg_asm_total:.4f}\n\n')

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
    print(f"  Fuel channels analysed:       {n_fuel}")
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
    print("HEATING PROFILE EXTRACTION (Mesh Tally + Peak Detection)")
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
    analyze_and_plot(data, run_dir, batch, params, target_power)

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