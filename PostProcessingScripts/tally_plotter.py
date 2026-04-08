"""
Tally Plotter Post-Processing Script

Generates flux, fission rate, and heating plots from OpenMC simulation results.

Memory management: all StatePoint files are opened in `with` blocks so the HDF5
handle is released immediately after use.  Large 3-D arrays are explicitly deleted
and gc.collect() is called between heavy plot functions to keep peak RSS low on
memory-constrained remote desktops.

Usage:
    # As a module:
    from tally_plotter import run_tally_plots
    run_tally_plots(run_dir, params, batch=None)

    # Standalone:
    python tally_plotter.py <reactivity_study_directory> <batch=None>
"""

import openmc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator
import sys
import json
import gc

# ====================================================================================================
# NORMALIZATION FACTOR FUNCTION
# ====================================================================================================

def get_normalization_factor(sp_path, target_power_MW):
    """
    Calculate manual power normalization factor.

    Opens and closes the statepoint within a context manager so the HDF5
    file handle is released immediately.
    """
    with openmc.StatePoint(sp_path) as sp:
        heating_tally = sp.get_tally(name='heating')
        heating_rate_ev = float(heating_tally.mean[0, 0, 0])

    joule_per_ev = 1.60218e-19
    heating_rate_j = heating_rate_ev * joule_per_ev
    power_watts = target_power_MW * 1e6
    source_per_sec = power_watts / heating_rate_j
    return source_per_sec

# ====================================================================================================
# RECONSTRUCT CORE FROM 1/6 WEDGE
# ====================================================================================================

def reconstruct_full_core_vectorized(data_wedge, mesh, core_radius):
    """
    Efficiently reconstruct full core data from 1/6 wedge by rotating 6 times.
    The wedge spans from 0deg to 60deg in the first quadrant.
    """
    nx, ny, nz = data_wedge.shape

    x_edges = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx + 1)
    y_edges = np.linspace(mesh.lower_left[1], mesh.upper_right[1], ny + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    n_full = max(nx, ny) * 2
    full_extent = core_radius * 1.05
    x_full = np.linspace(-full_extent, full_extent, n_full)
    y_full = np.linspace(-full_extent, full_extent, n_full)

    data_full = np.zeros((n_full, n_full, nz))

    for z_idx in range(nz):
        data_slice = data_wedge[:, :, z_idx]
        data_clean = np.nan_to_num(data_slice, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            interp = RegularGridInterpolator(
                (x_centers, y_centers),
                data_clean,
                method='nearest',
                bounds_error=False,
                fill_value=0.0
            )
        except ValueError:
            continue

        for i, x in enumerate(x_full):
            for j, y in enumerate(y_full):
                r = np.sqrt(x**2 + y**2)
                if r > core_radius or r < 1e-6:
                    continue
                theta = np.arctan2(y, x)
                if theta < 0:
                    theta += 2 * np.pi
                sector = int(theta / (np.pi / 3)) % 6
                theta_in_wedge = theta - sector * (np.pi / 3)
                x_wedge = r * np.cos(theta_in_wedge)
                y_wedge = r * np.sin(theta_in_wedge)
                if (mesh.lower_left[0] <= x_wedge <= mesh.upper_right[0] and
                    mesh.lower_left[1] <= y_wedge <= mesh.upper_right[1]):
                    value = interp([[x_wedge, y_wedge]])[0]
                    data_full[i, j, z_idx] = value

    return data_full, x_full, y_full

# ====================================================================================================
# HELPER: LOAD MESH TALLY DATA (opens/closes statepoint)
# ====================================================================================================

def _load_mesh_tally(sp_path, tally_name, scores_needed, source_per_sec, symmetry_factor):
    """
    Load a mesh tally's mean data, normalize, reshape to 3-D, and return it
    together with the mesh metadata.  The statepoint is closed before return.

    Returns
    -------
    dict  with keys 'data' (dict score->3d-array), 'mesh_dim', 'mesh_ll', 'mesh_ur'
    """
    with openmc.StatePoint(sp_path) as sp:
        tally = sp.get_tally(name=tally_name)
        mesh = tally.find_filter(openmc.MeshFilter).mesh
        nx, ny, nz = mesh.dimension
        scores = tally.scores
        mean = tally.mean[:, 0, :].copy()        # copy out of HDF5
        mesh_ll = mesh.lower_left.copy()
        mesh_ur = mesh.upper_right.copy()

    result = {}
    for score in scores_needed:
        idx = scores.index(score)
        arr = mean[:, idx] * source_per_sec / symmetry_factor
        result[score] = arr.reshape((nx, ny, nz), order='F')

    return {
        'data': result,
        'mesh_dim': (nx, ny, nz),
        'mesh_ll': mesh_ll,
        'mesh_ur': mesh_ur,
    }

def _load_heating_mesh_tally(sp_path, tally_name, source_per_sec, symmetry_factor):
    """
    Load a heating-local mesh tally, normalize to watts, reshape to 3-D.
    """
    joule_per_ev = 1.60218e-19
    with openmc.StatePoint(sp_path) as sp:
        tally = sp.get_tally(name=tally_name)
        mesh = tally.find_filter(openmc.MeshFilter).mesh
        nx, ny, nz = mesh.dimension
        mean = tally.mean[:, 0, 0].copy()
        mesh_ll = mesh.lower_left.copy()
        mesh_ur = mesh.upper_right.copy()

    heating_watts = mean * source_per_sec * joule_per_ev / symmetry_factor
    heating_3d = heating_watts.reshape((nx, ny, nz), order='F')

    return {
        'data': heating_3d,
        'mesh_dim': (nx, ny, nz),
        'mesh_ll': mesh_ll,
        'mesh_ur': mesh_ur,
    }


def _active_core_slice(info, n_ax_zones):
    """
    Slice a full-core mesh info dict down to just the active-core z-bins.

    The full-core mesh has reflector zones below and above the active core.
    The number of reflector bins on each side is (nz_full - n_ax_zones) // 2.
    Returns a new info dict with 'data', 'mesh_dim', 'mesh_ll', 'mesh_ur'
    adjusted to cover only the active fuel region.

    Works for both _load_mesh_tally output (data is dict of score->3d arrays)
    and _load_heating_mesh_tally output (data is a single 3d array).
    """
    nx, ny, nz_full = info['mesh_dim']
    n_refl = (nz_full - n_ax_zones) // 2
    z0 = n_refl
    z1 = n_refl + n_ax_zones

    ll = info['mesh_ll'].copy()
    ur = info['mesh_ur'].copy()
    dz = (ur[2] - ll[2]) / nz_full
    ll[2] = ll[2] + n_refl * dz
    ur[2] = ll[2] + n_ax_zones * dz

    data = info['data']
    if isinstance(data, dict):
        sliced = {k: v[:, :, z0:z1] for k, v in data.items()}
    else:
        sliced = data[:, :, z0:z1]

    return {
        'data':     sliced,
        'mesh_dim': (nx, ny, n_ax_zones),
        'mesh_ll':  ll,
        'mesh_ur':  ur,
    }

# ====================================================================================================
# PLOTTING FUNCTION FOR CORE XY CROSS-SECTIONS
# ====================================================================================================

def plot_xy_slice(run_dir, batch, z_index, is_wedge, reconstruct, target_power_MW, core_radius, n_ax_zones, save_dir=None, show_titles=True):
    """Plot XY slices of flux, fission, and heating at a given axial index."""

    save_dir = save_dir or run_dir
    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    if not os.path.exists(sp_path):
        print(f"Warning: Statepoint file not found: {sp_path}")
        return

    source_per_sec = get_normalization_factor(sp_path, target_power_MW)
    symmetry_factor = 6 if is_wedge else 1

    # --- Load flux + fission from full-core mesh, then slice to active core ---
    info = _load_mesh_tally(sp_path, 'mesh_rates_full', ['flux', 'fission'],
                            source_per_sec, symmetry_factor)
    info = _active_core_slice(info, n_ax_zones)
    nx, ny, nz = info['mesh_dim']

    for score, cmap, label, unit in [
        ('flux',    'hot', 'Neutron Flux',  'Flux [n/(cm\u00b2 \u00b7 s)]'),
        ('fission', 'hot', 'Fission Rate',  'Fission Rate [fissions/s]'),
    ]:
        data_3d = info['data'][score]

        if is_wedge and reconstruct:
            # Build a minimal mock mesh for the reconstructor
            class _M:
                pass
            m = _M()
            m.lower_left = info['mesh_ll']
            m.upper_right = info['mesh_ur']
            data_full, x_full, y_full = reconstruct_full_core_vectorized(data_3d, m, core_radius)
            data_xy = data_full[:, :, z_index].T
            dx = x_full[1] - x_full[0]; dy = y_full[1] - y_full[0]
            x_edges = np.concatenate([x_full - dx/2, [x_full[-1] + dx/2]])
            y_edges = np.concatenate([y_full - dy/2, [y_full[-1] + dy/2]])
            geometry_label = "Full Core (Reconstructed)"
            del data_full, x_full, y_full
        else:
            data_xy = data_3d[:, :, z_index].T
            x_edges = np.linspace(info['mesh_ll'][0], info['mesh_ur'][0], nx + 1)
            y_edges = np.linspace(info['mesh_ll'][1], info['mesh_ur'][1], ny + 1)
            geometry_label = "1/6 Wedge" if is_wedge else "Full Core"

        z_edges = np.linspace(info['mesh_ll'][2], info['mesh_ur'][2], nz + 1)
        z_center = (z_edges[z_index] + z_edges[z_index + 1]) / 2
        vmax = data_xy.max() if data_xy.max() > 0 else 1

        suffix = '_reconstructed' if (is_wedge and reconstruct) else ('_wedge' if is_wedge else '')

        fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
        pcm = ax.pcolormesh(x_edges, y_edges, data_xy, shading='auto', cmap=cmap, vmin=0, vmax=vmax)
        cbar = plt.colorbar(pcm, ax=ax, label=unit)
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        ax.set_xlabel('X [cm]'); ax.set_ylabel('Y [cm]')
        if show_titles:
            ax.set_title(f'{label} - {geometry_label}\nZ = {z_center:.1f} cm (index {z_index}), {target_power_MW} MW')
        ax.set_aspect('equal')
        save_path = os.path.join(save_dir, f'batch{batch}_{score}_xy_z{z_index}{suffix}.png')
        plt.savefig(save_path, bbox_inches='tight'); plt.close()
        print(f"Saved: {save_path}")

        del data_xy

    del info
    gc.collect()

    # --- Heating XY slice (separate load to avoid holding both tallies) ---
    try:
        h_info = _load_heating_mesh_tally(sp_path, 'mesh_heating_full', source_per_sec, symmetry_factor)
        h_info = _active_core_slice(h_info, n_ax_zones)
        h_nx, h_ny, h_nz = h_info['mesh_dim']
        heating_3d = h_info['data']

        if is_wedge and reconstruct:
            class _M2:
                pass
            m2 = _M2()
            m2.lower_left = h_info['mesh_ll']
            m2.upper_right = h_info['mesh_ur']
            heating_full, hx_full, hy_full = reconstruct_full_core_vectorized(heating_3d, m2, core_radius)
            heating_xy = heating_full[:, :, z_index].T
            hdx = hx_full[1] - hx_full[0]; hdy = hy_full[1] - hy_full[0]
            hx_edges = np.concatenate([hx_full - hdx/2, [hx_full[-1] + hdx/2]])
            hy_edges = np.concatenate([hy_full - hdy/2, [hy_full[-1] + hdy/2]])
            del heating_full, hx_full, hy_full
        else:
            heating_xy = heating_3d[:, :, z_index].T
            hx_edges = np.linspace(h_info['mesh_ll'][0], h_info['mesh_ur'][0], h_nx + 1)
            hy_edges = np.linspace(h_info['mesh_ll'][1], h_info['mesh_ur'][1], h_ny + 1)

        z_edges_h = np.linspace(h_info['mesh_ll'][2], h_info['mesh_ur'][2], h_nz + 1)
        z_center_h = (z_edges_h[z_index] + z_edges_h[z_index + 1]) / 2
        hmax = heating_xy.max() if heating_xy.max() > 0 else 1
        suffix = '_reconstructed' if (is_wedge and reconstruct) else ('_wedge' if is_wedge else '')
        geom_label = "Full Core (Reconstructed)" if (is_wedge and reconstruct) else ("1/6 Wedge" if is_wedge else "Full Core")

        fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
        pcm = ax.pcolormesh(hx_edges, hy_edges, heating_xy, shading='auto', cmap='inferno', vmin=0, vmax=hmax)
        cbar = plt.colorbar(pcm, ax=ax, label='Heating Rate [W/mesh element]')
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        ax.set_xlabel('X [cm]'); ax.set_ylabel('Y [cm]')
        if show_titles:
            ax.set_title(f'Local Heating Rate - {geom_label}\nZ = {z_center_h:.1f} cm (index {z_index}), {target_power_MW} MW')
        ax.set_aspect('equal')
        save_path = os.path.join(save_dir, f'batch{batch}_heating_xy_z{z_index}{suffix}.png')
        plt.savefig(save_path, bbox_inches='tight'); plt.close()
        print(f"Saved: {save_path}")

        del heating_3d, heating_xy, h_info
    except Exception as e:
        print(f"  Note: Could not plot heating XY slice: {e}")

    gc.collect()

# ====================================================================================================
# PLOTTING FUNCTION FOR AXIAL FLUX, FISSION, AND HEATING PROFILES
# ====================================================================================================

def plot_axial_profile(run_dir, batch, is_wedge, target_power_MW, n_ax_zones, save_dir=None, show_titles=True):
    """Plot axial profiles of flux, fission rate, and heating."""

    save_dir = save_dir or run_dir
    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    if not os.path.exists(sp_path):
        print(f"Warning: Statepoint file not found: {sp_path}")
        return

    source_per_sec = get_normalization_factor(sp_path, target_power_MW)
    symmetry_factor = 6 if is_wedge else 1
    geometry_label = "1/6 Wedge" if is_wedge else "Full Core"

    # --- Flux + Fission axial profiles — load full-core, slice to active core ---
    info = _load_mesh_tally(sp_path, 'mesh_rates_full', ['flux', 'fission'],
                            source_per_sec, symmetry_factor)
    info = _active_core_slice(info, n_ax_zones)
    nx, ny, nz = info['mesh_dim']

    flux_axial = info['data']['flux'].mean(axis=(0, 1))
    fission_axial = info['data']['fission'].mean(axis=(0, 1))
    z_coords = np.linspace(info['mesh_ll'][2], info['mesh_ur'][2], nz + 1)
    z_centers = (z_coords[:-1] + z_coords[1:]) / 2

    del info; gc.collect()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    ax1.plot(z_centers, flux_axial, 'b-', linewidth=2)
    ax1.set_xlabel('Axial Position [cm]'); ax1.set_ylabel('Average Flux [n/(cm\u00b2 \u00b7 s)]')
    if show_titles:
        ax1.set_title(f'Axial Flux Profile - {geometry_label}\n{target_power_MW} MW')
    ax1.grid(True, alpha=0.3); ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    ax2.plot(z_centers, fission_axial, 'r-', linewidth=2)
    ax2.set_xlabel('Axial Position [cm]'); ax2.set_ylabel('Average Fission Rate [fissions/s]')
    if show_titles:
        ax2.set_title(f'Axial Fission Profile - {geometry_label}\n{target_power_MW} MW')
    ax2.grid(True, alpha=0.3); ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'batch{batch}_axial_profiles.png')
    plt.savefig(save_path, bbox_inches='tight'); plt.close()
    print(f"Saved: {save_path}")

    del flux_axial, fission_axial; gc.collect()

    # --- Heating axial profile ---
    try:
        h_info = _load_heating_mesh_tally(sp_path, 'mesh_heating_full', source_per_sec, symmetry_factor)
        h_info = _active_core_slice(h_info, n_ax_zones)
        h_nz = h_info['mesh_dim'][2]
        heating_axial = h_info['data'].sum(axis=(0, 1))
        hz_coords = np.linspace(h_info['mesh_ll'][2], h_info['mesh_ur'][2], h_nz + 1)
        hz_centers = (hz_coords[:-1] + hz_coords[1:]) / 2

        del h_info; gc.collect()

        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        ax.plot(hz_centers, heating_axial, 'darkorange', linewidth=2)
        ax.set_xlabel('Axial Position [cm]'); ax.set_ylabel('Integrated Heating Rate [W]')
        if show_titles:
            ax.set_title(f'Axial Heating Profile - {geometry_label}\n{target_power_MW} MW')
        ax.grid(True, alpha=0.3); ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        save_path = os.path.join(save_dir, f'batch{batch}_axial_heating_profile.png')
        plt.savefig(save_path, bbox_inches='tight'); plt.close()
        print(f"Saved: {save_path}")
        del heating_axial
    except Exception as e:
        print(f"  Note: Could not plot heating axial profile: {e}")

    gc.collect()

# ====================================================================================================
# PLOTTING FUNCTION FOR CORE RZ CROSS-SECTIONS
# ====================================================================================================

def plot_rz_crosssection(run_dir, batch, angle_deg, is_wedge, target_power_MW, include_reflector, n_ax_zones, save_dir=None, show_titles=True):
    """Plot RZ cross-section at specified angle."""

    save_dir = save_dir or run_dir
    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    if not os.path.exists(sp_path):
        print(f"Warning: Statepoint file not found: {sp_path}")
        return

    source_per_sec = get_normalization_factor(sp_path, target_power_MW)
    symmetry_factor = 6 if is_wedge else 1
    geometry_label = "1/6 Wedge" if is_wedge else "Full Core"
    angle_rad = np.radians(angle_deg)

    # --- Flux + Fission RZ ---
    # Always load from full-core mesh; active-core-only view uses a z-slice below.
    info = _load_mesh_tally(sp_path, 'mesh_rates_full', ['flux', 'fission'],
                            source_per_sec, symmetry_factor)
    if not include_reflector:
        info = _active_core_slice(info, n_ax_zones)

    nx, ny, nz = info['mesh_dim']
    x_centers = np.linspace(info['mesh_ll'][0], info['mesh_ur'][0], nx)
    y_centers = np.linspace(info['mesh_ll'][1], info['mesh_ur'][1], ny)
    z_centers = np.linspace(info['mesh_ll'][2], info['mesh_ur'][2], nz)

    r_max = max(abs(info['mesh_ll'][0]), abs(info['mesh_ur'][0]),
                abs(info['mesh_ll'][1]), abs(info['mesh_ur'][1]))
    r_vals = np.linspace(0, r_max, nx)

    for score, cmap, label, unit in [
        ('flux',    'hot', 'Flux',          'Flux [n/(cm\u00b2 \u00b7 s)]'),
        ('fission', 'hot', 'Fission Rate',  'Fission Rate [fissions/s]'),
    ]:
        data_3d = info['data'][score]
        rz = np.zeros((len(r_vals), nz))
        for i, r in enumerate(r_vals):
            x_t = r * np.cos(angle_rad); y_t = r * np.sin(angle_rad)
            if (info['mesh_ll'][0] <= x_t <= info['mesh_ur'][0] and
                info['mesh_ll'][1] <= y_t <= info['mesh_ur'][1]):
                ix = np.argmin(np.abs(x_centers - x_t))
                iy = np.argmin(np.abs(y_centers - y_t))
                rz[i, :] = data_3d[ix, iy, :]

        R, Z = np.meshgrid(r_vals, z_centers, indexing='ij')
        r_range = r_vals[-1] - r_vals[0]; z_range = z_centers[-1] - z_centers[0]
        fig_w = 6; fig_h = max(4, min(12, fig_w * (z_range / r_range) * 0.7))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        pcm = ax.pcolormesh(R, Z, rz, shading='auto', cmap=cmap)
        cbar = plt.colorbar(pcm, ax=ax, label=unit)
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        ax.set_xlabel('Radial Position [cm]'); ax.set_ylabel('Axial Position [cm]')
        if show_titles:
            ax.set_title(f'{label} - RZ Cross-Section at {angle_deg}\u00b0 - {geometry_label}\n{target_power_MW} MW')
        save_path = os.path.join(save_dir, f'batch{batch}_{score}_rz_angle{angle_deg}.png')
        plt.savefig(save_path, bbox_inches='tight'); plt.close()
        print(f"Saved: {save_path}")
        del rz

    del info; gc.collect()

    # --- Heating RZ ---
    try:
        h_info = _load_heating_mesh_tally(sp_path, 'mesh_heating_full', source_per_sec, symmetry_factor)
        if not include_reflector:
            h_info = _active_core_slice(h_info, n_ax_zones)

        h_nx, h_ny, h_nz = h_info['mesh_dim']
        hx_c = np.linspace(h_info['mesh_ll'][0], h_info['mesh_ur'][0], h_nx)
        hy_c = np.linspace(h_info['mesh_ll'][1], h_info['mesh_ur'][1], h_ny)
        hz_c = np.linspace(h_info['mesh_ll'][2], h_info['mesh_ur'][2], h_nz)
        h_r_max = max(abs(h_info['mesh_ll'][0]), abs(h_info['mesh_ur'][0]),
                      abs(h_info['mesh_ll'][1]), abs(h_info['mesh_ur'][1]))
        h_r_vals = np.linspace(0, h_r_max, h_nx)

        heating_rz = np.zeros((len(h_r_vals), h_nz))
        h3d = h_info['data']
        for i, r in enumerate(h_r_vals):
            x_t = r * np.cos(angle_rad); y_t = r * np.sin(angle_rad)
            if (h_info['mesh_ll'][0] <= x_t <= h_info['mesh_ur'][0] and
                h_info['mesh_ll'][1] <= y_t <= h_info['mesh_ur'][1]):
                ix = np.argmin(np.abs(hx_c - x_t))
                iy = np.argmin(np.abs(hy_c - y_t))
                heating_rz[i, :] = h3d[ix, iy, :]

        del h3d, h_info; gc.collect()

        H_R, H_Z = np.meshgrid(h_r_vals, hz_c, indexing='ij')
        r_range = h_r_vals[-1] - h_r_vals[0]; z_range = hz_c[-1] - hz_c[0]
        fig_w = 6; fig_h = max(4, min(12, fig_w * (z_range / r_range) * 0.7))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        pcm = ax.pcolormesh(H_R, H_Z, heating_rz, shading='auto', cmap='inferno')
        cbar = plt.colorbar(pcm, ax=ax, label='Heating Rate [W/mesh element]')
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        ax.set_xlabel('Radial Position [cm]'); ax.set_ylabel('Axial Position [cm]')
        if show_titles:
            ax.set_title(f'Heating - RZ Cross-Section at {angle_deg}\u00b0 - {geometry_label}\n{target_power_MW} MW')
        save_path = os.path.join(save_dir, f'batch{batch}_heating_rz_angle{angle_deg}.png')
        plt.savefig(save_path, bbox_inches='tight'); plt.close()
        print(f"Saved: {save_path}")
        del heating_rz
    except Exception as e:
        print(f"  Note: Could not plot heating RZ cross-section: {e}")

    gc.collect()

# ====================================================================================================
# EXTRACT AND SAVE GLOBAL RATES
# ====================================================================================================

def print_global_rates(run_dir, batch, is_wedge, target_power_MW, save_dir=None):
    """Print and save global reaction rates."""

    save_dir = save_dir or run_dir
    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    if not os.path.exists(sp_path):
        print(f"Warning: Statepoint file not found: {sp_path}")
        return

    with openmc.StatePoint(sp_path) as sp:
        heating_tally = sp.get_tally(name='heating')
        heating_rate_ev = float(heating_tally.mean[0, 0, 0])

        tally = sp.get_tally(name='global_rates')
        scores = tally.scores
        flux_idx = scores.index('flux')
        fission_idx = scores.index('fission')
        nu_fission_idx = scores.index('nu-fission')
        mean = tally.mean[0, 0, :].copy()

        keff_nom = float(sp.keff.nominal_value)
        keff_std = float(sp.keff.std_dev)

    joule_per_ev = 1.60218e-19
    heating_rate_j = heating_rate_ev * joule_per_ev
    power_watts = target_power_MW * 1e6
    source_per_sec = power_watts / heating_rate_j

    total_flux = mean[flux_idx] * source_per_sec
    total_fission = mean[fission_idx] * source_per_sec
    total_nu_fission = mean[nu_fission_idx] * source_per_sec
    energy_per_fission = 200e6 * joule_per_ev
    power_mw = (total_fission * energy_per_fission) / 1e6

    geometry_label = "1/6 WEDGE GEOMETRY" if is_wedge else "FULL CORE GEOMETRY"

    output = []
    output.append('='*80)
    output.append(f"GLOBAL REACTION RATES (Batch {batch}) - {geometry_label}")
    output.append('='*80)
    output.append(f"k-effective: {keff_nom:.5f} +/- {keff_std:.5f}")
    output.append(f"\nNormalization:")
    output.append(f"   Heating rate: {heating_rate_ev:.3e} eV/source")
    output.append(f"   Source rate: {source_per_sec:.3e} source/s")
    output.append(f"\nReaction Rates:")
    output.append(f"   Total Flux: {total_flux:.3e} n/(cm^2*s)")
    output.append(f"   Total Fission Rate: {total_fission:.3e} fissions/s")
    output.append(f"   Total Nu-Fission Rate: {total_nu_fission:.3e} neutrons/s")
    output.append(f"\nPower:")
    output.append(f"   Target Power: {target_power_MW:.3f} MW")
    output.append(f"   Calculated Power (from fission): {power_mw:.3f} MW")
    output.append('='*80 + "\n")

    for line in output:
        print(line)

    save_path = os.path.join(save_dir, f'batch{batch}_global_rates.txt')
    with open(save_path, 'w') as f:
        f.write('\n'.join(output))
    print(f"Saved: {save_path}")

# ====================================================================================================
# MAIN TALLY PLOTTING FUNCTION
# ====================================================================================================

def run_tally_plots(run_dir, params, batch=None):
    """Run all tally plotting for a simulation."""

    print(f"\n{'='*80}")
    print("TALLY PLOTTING")
    print(f"{'='*80}")
    print(f"Run directory: {run_dir}")

    results_dir = os.path.join(run_dir, 'tally_plotter_results')
    os.makedirs(results_dir, exist_ok=True)

    if batch is None:
        for f in os.listdir(run_dir):
            if f.startswith('statepoint') and f.endswith('.h5'):
                batch = int(f.split('.')[1])
                break

    if batch is None:
        print("ERROR: No statepoint file found!")
        return

    print(f"Batch number: {batch}")

    is_wedge = params["use_1/6_geometry"]
    target_power = params["thermal_power_MW"]
    core_radius = params["core_radius"]
    n_ax_zones = params["n_ax_zones"]
    show_titles = params.get("show_titles", True)

    print(f"Geometry: {'1/6 Wedge' if is_wedge else 'Full Core'}")
    print(f"Target power: {target_power} MW")
    print(f"{'='*80}")

    # Global rates (lightweight — no big arrays)
    print("\nGenerating global rates...\n")
    print_global_rates(run_dir, batch, is_wedge, target_power, save_dir=results_dir)

    # Axial profiles
    print("\nGenerating axial profiles...")
    plot_axial_profile(run_dir, batch, is_wedge, target_power, n_ax_zones, save_dir=results_dir,
                       show_titles=show_titles)
    gc.collect()

    # RZ cross-sections
    print("\nGenerating RZ cross-sections...")
    plot_rz_crosssection(run_dir, batch, 0, is_wedge, target_power, True, n_ax_zones, save_dir=results_dir,
                          show_titles=show_titles)
    gc.collect()

    # XY slices — one at a time with gc between each
    print("\nGenerating XY slices...")
    z_indices = [0, n_ax_zones//4, n_ax_zones//2, 3*n_ax_zones//4, n_ax_zones-1]

    for z_idx in z_indices:
        plot_xy_slice(run_dir, batch, z_idx, is_wedge, reconstruct=False,
                      target_power_MW=target_power, core_radius=core_radius,
                      n_ax_zones=n_ax_zones, save_dir=results_dir,
                      show_titles=show_titles)
        gc.collect()

    print(f"\n{'='*80}")
    print("TALLY PLOTTING COMPLETE")
    print(f"{'='*80}\n")

# ====================================================================================================
# LOAD PARAMETERS FROM JSON FILE
# ====================================================================================================

def load_params_from_run_dir(run_dir):
    """Load parameters from run_params.json in the run directory."""
    params_path = os.path.join(run_dir, 'run_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        params.pop('n_trisos', None)
        return params
    return None

# ====================================================================================================
# CORE GEOMETRY DETECTION FUNCTION
# ====================================================================================================

def detect_geometry_from_statepoint(run_dir, batch=None):
    """Auto-detect geometry parameters from statepoint file."""
    if batch is None:
        for f in os.listdir(run_dir):
            if f.startswith('statepoint') and f.endswith('.h5'):
                batch = int(f.split('.')[1])
                break
    if batch is None:
        return None

    sp_path = os.path.join(run_dir, f'statepoint.{batch}.h5')
    with openmc.StatePoint(sp_path) as sp:
        tally = sp.get_tally(name='mesh_rates_full')
        mesh = tally.find_filter(openmc.MeshFilter).mesh
        is_wedge = mesh.lower_left[0] >= -0.1 and mesh.lower_left[1] >= -0.1
        core_radius = max(mesh.upper_right[0], mesh.upper_right[1])
        # Full-core mesh includes reflector zones above and below; n_ax_zones is
        # the active-core bin count, which is stored in run_params.json. For
        # auto-detection fall back to reading it from the mesh dimension directly
        # by assuming symmetric reflector zones matching the stored reflector_thickness.
        n_ax_zones = mesh.dimension[2]  # overestimate; run_params.json preferred

    return {
        "use_1/6_geometry": is_wedge,
        "thermal_power": 15.0,
        "core_radius": core_radius,
        "n_ax_zones": n_ax_zones
    }

# ====================================================================================================
# STANDALONE ENTRY POINT
# ====================================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tally_plotter.py <run_directory> [batch_number]")
        print("\nThe script will load parameters from run_params.json in the run directory.")
        print("If not found, it will auto-detect geometry from the statepoint file.")
        sys.exit(1)

    run_dir = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"\nProcessing: {run_dir}")

    params = load_params_from_run_dir(run_dir)

    if params is not None:
        print("\nLoaded parameters from run_params.json")
        print(f"  Geometry: {'1/6 Wedge' if params.get('use_1/6_geometry', False) else 'Full Core'}")
        print(f"  Core radius: {params.get('core_radius', 'unknown')} cm")
        print(f"  Axial zones: {params.get('n_ax_zones', 'unknown')}")
    else:
        print("run_params.json not found, detecting from statepoint...")
        params = detect_geometry_from_statepoint(run_dir, batch)
        if params is None:
            print("ERROR: Could not load or detect parameters.")
            sys.exit(1)
        print(f"  Detected geometry: {'1/6 Wedge' if params['use_1/6_geometry'] else 'Full Core'}")
        print(f"  Core radius: {params['core_radius']:.1f} cm")
        print(f"  Axial zones: {params['n_ax_zones']}")

    run_tally_plots(run_dir, params, batch)