"""
nc_htgr — Single-Channel Thermal-Hydraulics and Recuperated Brayton Cycle Solver
================================================================================

Steady-state single-channel analysis of a prismatic HTGR coolant channel coupled
to a direct recuperated Brayton cycle. Solves the axial coolant enthalpy rise,
convective film drop, graphite conduction and compact/kernel temperatures node by
node, then closes the power cycle to report net electrical output.

Used by this repository as the thermal-hydraulics half of the coupled
neutronics/TH iteration in ``mol_eol_analysis.th_coupler`` and for the per-step
MWe calculation in ``PostProcessingScripts/depletion_postprocessing.py``.

--------------------------------------------------------------------------------
ATTRIBUTION
--------------------------------------------------------------------------------
This file originates from the HTGR-SCAPC project by **Bryan Huynh**:

    https://github.com/bryanhhuynh/HTGR-SCAPC

It was written by Bryan Huynh as the thermal-hydraulics and power-cycle
contribution to the CHUDR senior design project (University of Florida,
ENU 4192). It is included here with his permission so that the coupled
neutronics/TH workflow lives in a single repository.

Modifications made relative to the upstream original:
  * Corrected the axial heating profile for downward flow — the z lookup in
    _get_qprime is mirrored so the inlet node reads the power at the physical
    top of the core. Without this the profile is applied backwards.
  * Recuperator effectiveness raised from 0.90 to 0.94.
  * TRISO packing fraction raised from 0.30 to 0.33.
  * neutronics_file repointed to the neutronics.csv that ships here, and a
    relative neutronics_file is resolved against the deck's own directory.

The power cycle was already a direct recuperated Brayton cycle upstream; the
upstream repository description calling it indirect is stale metadata.

See NOTICE.md in the repository root for the full provenance statement.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
Standalone::

    python nc_htgr.py --deck nc_input.csv

All inputs are read from ``nc_input.csv``. When ``axial_shape`` is set to
``neutronics_table`` the axial heating profile is loaded from the CSV named by
the ``neutronics_file`` key (``neutronics.csv`` by default), which the neutronics
side of this repository writes out from OpenMC mesh tallies.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from CoolProp import AbstractState as _AS_factory
import CoolProp

# AbstractState handle !!!!
# no longer calling prop lookups 250,000 times = PC saved
_He: CoolProp.AbstractState = _AS_factory("HEOS", "Helium")



# Helium properties


def helium_cp_gamma_R() -> Tuple[float, float, float]:
    # Ideal-gas constants for helium. Used by the Brayton cycle only and for time
    R     = 2077.0
    gamma = 5.0 / 3.0
    cp    = 2.5 * R
    return cp, gamma, R


def helium_props(T_K: float, P_Pa: float) -> Dict[str, float]:
    # Return helium properties at (T_K, P_Pa) using AbstractState.
    T = max(1.0, float(T_K))
    P = max(1.0, float(P_Pa))
    _He.update(CoolProp.PT_INPUTS, P, T)
    _, gamma, R = helium_cp_gamma_R()
    return {
        "cp":    _He.cpmass(),
        "gamma": gamma,
        "R":     R,
        "rho":   _He.rhomass(),
        "mu":    _He.viscosity(),
        "k":     _He.conductivity(),
        "Pr":    _He.Prandtl(),
    }



# Internal flow correlations


def friction_factor_churchill(Re: float, rel_rough: float) -> float:
    Re = max(1e-12, float(Re))
    rr = max(0.0, float(rel_rough))
    A = (2.457 * math.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * rr))) ** 16.0
    B = (37530.0 / Re) ** 16.0
    return float(8.0 * ((8.0 / Re) ** 12.0 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0))


def nusselt_internal(Re: float, Pr: float, f: float) -> float:
    Re = float(Re)
    if Re < 2300.0:
        return 4.36
    f = max(1e-12, float(f))

    def gnielinski(Re_g: float) -> float:
        Nu = (f / 8.0) * (Re_g - 1000.0) * Pr / (
            1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0))
        return max(1.0, Nu)

    if Re > 4000.0:
        return float(gnielinski(Re))
    w = (Re - 2300.0) / (4000.0 - 2300.0)
    return float((1.0 - w) * 4.36 + w * gnielinski(Re))



# Graphite conduction


def graphite_k(T_K: float, model: str = "pcea_table") -> float:
    T_pts = np.array([300.0, 500.0, 700.0, 900.0, 1100.0, 1500.0])
    k_pts = np.array([65.0,  56.0,  45.0,  38.0,  33.0,   27.0])
    T = float(T_K)
    if T <= T_pts[0]:
        return float(k_pts[0])
    if T >= T_pts[-1]:
        return float(k_pts[-1])
    return float(np.interp(T, T_pts, k_pts))


def shape_factor_parallel_cylinders(R1: float, R2: float, L_center: float) -> float:
    arg = (L_center ** 2 - R1 ** 2 - R2 ** 2) / (2.0 * R1 * R2)
    if arg <= 1.0:
        arg = 1.0000001
    return float(2.0 * math.pi / math.acosh(arg))



# Gap HTC (conduction + radiation)


def gap_htc_concentric_cylinders(
    T_inner_K: float, T_outer_K: float,
    r_inner: float, r_outer: float,
    emiss_inner: float, emiss_outer: float,
    k_He: float,                              # pre-computed helium conductivity [W/m·K]
) -> float:

    gap   = max(1e-9, float(r_outer - r_inner))
    h_cond = k_He / gap

    sigma = 5.670374419e-8
    e1    = max(1e-6, min(1.0, float(emiss_inner)))
    e2    = max(1e-6, min(1.0, float(emiss_outer)))
    denom = (1.0 / e1) + (r_inner / r_outer) * (1.0 / e2 - 1.0)
    Tm    = 0.5 * (float(T_inner_K) + float(T_outer_K))
    if abs(T_inner_K - T_outer_K) < 1e-9:
        dT4_dT = 4.0 * Tm ** 3
    else:
        dT4_dT = (T_inner_K ** 4 - T_outer_K ** 4) / (T_inner_K - T_outer_K)
    h_rad = sigma * dT4_dT / denom
    return float(h_cond + h_rad)


def solve_compact_surface_temp(
    T_fuel_hole_wall_K: float, qprime_fuel_W_m: float,
    D_compact: float, D_fuel_hole: float, P_Pa: float,
    emiss_compact: float, emiss_fuel_hole: float,
    max_iter: int = 10, tol_K: float = 1e-4
) -> Tuple[float, float]:
    r_inner = 0.5 * float(D_compact)
    r_outer = 0.5 * float(D_fuel_hole)
    A_surf_per_m = math.pi * float(D_compact)
    qpp = float(qprime_fuel_W_m) / max(1e-12, A_surf_per_m)

    # Evaluate helium k once at the mean gap temperature (outer wall + 50 K guess).
    T_gap_est = float(T_fuel_hole_wall_K) + 50.0
    k_He = helium_props(T_gap_est, max(1.0, float(P_Pa)))["k"]

    T_surf = T_gap_est
    h_eff  = 0.0
    for _ in range(max_iter):
        h_eff = gap_htc_concentric_cylinders(
            T_inner_K=T_surf, T_outer_K=T_fuel_hole_wall_K,
            r_inner=r_inner, r_outer=r_outer,
            emiss_inner=emiss_compact, emiss_outer=emiss_fuel_hole,
            k_He=k_He)
        T_new = float(T_fuel_hole_wall_K) + qpp / max(1e-12, h_eff)
        if abs(T_new - T_surf) < tol_K:
            T_surf = T_new
            break
        T_surf = 0.6 * T_surf + 0.4 * T_new
    return float(T_surf), float(h_eff)



# TRISO kernel temperature


def triso_layer_k(layer: str, T_K: float) -> float:
    T_K = max(300.0, float(T_K))
    layer = layer.lower().strip()
    if layer == "sic":
        return float(max(60.0, 166.0 * (300.0 / T_K) ** 0.85))
    elif layer in ("ipyc", "opyc"):
        k_RT = 11.1 if layer == "ipyc" else 8.6
        return float(k_RT * (T_K / 300.0) ** 0.25)
    elif layer == "buffer":
        return float(min(1.0 * (T_K / 300.0) ** 0.20, 2.0))
    raise ValueError("Unknown TRISO layer: {}".format(layer))


def triso_kernel_center_temp(
    T_compact_center_K: float, qprime_fuel_W_m: float,
    packing_fraction: float, D_compact: float
) -> float:
    r_kernel   = 214.85e-6
    r_buf_out  = 314.85e-6
    r_ipyc_out = 354.85e-6
    r_sic_out  = 389.85e-6
    r_opyc_out = 429.85e-6

    Vp = (4.0 / 3.0) * math.pi * r_opyc_out ** 3
    n_per_m3 = float(packing_fraction) / max(1e-18, Vp)
    A_compact = math.pi * (0.5 * float(D_compact)) ** 2
    Np_per_m = n_per_m3 * A_compact
    q_particle = float(qprime_fuel_W_m) / max(1.0, Np_per_m)

    def shell_dT(r_in: float, r_out: float, k_val: float) -> float:
        return q_particle * (1.0 / r_in - 1.0 / r_out) / (4.0 * math.pi * k_val)

    T_opyc_out = float(T_compact_center_K)
    k_opyc = triso_layer_k("opyc", T_opyc_out + 5.0)
    dT_opyc = shell_dT(r_ipyc_out, r_opyc_out, k_opyc)
    T_opyc_in = T_opyc_out + dT_opyc

    dT_sic = q_particle * (1.0 / r_ipyc_out - 1.0 / r_sic_out) / (
        4.0 * math.pi * triso_layer_k("sic", T_opyc_in + 10.0))
    T_sic_in = T_opyc_in + dT_sic

    k_ipyc = triso_layer_k("ipyc", T_sic_in + 5.0)
    dT_ipyc = q_particle * (1.0 / r_buf_out - 1.0 / r_ipyc_out) / (4.0 * math.pi * k_ipyc)
    T_ipyc_in = T_sic_in + dT_ipyc

    k_buf = triso_layer_k("buffer", T_ipyc_in + 20.0)
    dT_buf = q_particle * (1.0 / r_kernel - 1.0 / r_buf_out) / (4.0 * math.pi * k_buf)
    T_buf_in = T_ipyc_in + dT_buf

    dT_kernel = q_particle / (4.0 * math.pi * 3.5 * r_kernel)
    return float(T_buf_in + dT_kernel)



# Neutronics table loader


class NeutronicsTable:

    COLUMN_MAP = {
        "average": "average_channel_q_W",
        "hot":     "hottest_channel_q_W",
        "cold":    "coldest_channel_q_W",
    }

    def __init__(self, path: str, N_fuel_channels: int) -> None:
        df = pd.read_csv(path)
        self.z_nodes_m: np.ndarray = df["z_center_cm"].values / 100.0
        dz = float(self.z_nodes_m[1] - self.z_nodes_m[0])
        self._dz = dz
        self.L_heated = float(self.z_nodes_m[-1]) + 0.5 * dz

        # Store raw W/node arrays and derive per-channel totals
        self._q_nodes: Dict[str, np.ndarray] = {}
        self.Q_per_channel: Dict[str, float] = {}
        for case, col in self.COLUMN_MAP.items():
            q_node = df[col].values.astype(float)
            self._q_nodes[case] = q_node
            self.Q_per_channel[case] = float(q_node.sum())

        # Reactor thermal power = average fuel channel power times number of fuel channels.
        self.Q_reactor_MWth = (
            self.Q_per_channel["average"] * int(N_fuel_channels) / 1.0e6
        )

    def qprime(self, z_bottom_m: float, case: str) -> float:
        case = case.lower().strip()
        if case not in self._q_nodes:
            raise ValueError("channel_case must be one of: {}".format(
                list(self.COLUMN_MAP)))
        z = float(z_bottom_m)
        if z < 0.0 or z > self.L_heated:
            return 0.0
        return float(np.interp(z, self.z_nodes_m,
                               self._q_nodes[case] / self._dz,
                               left=0.0, right=0.0))



# Dataclasses
# Input needs to be "python nc_htgr.py --deck file_name.csv" 

@dataclass
class ChannelInputs:
    # geometry
    L: float                          # total channel length [m] (includes unheated ends)
    L_heated: float                   # heated length [m]
    N: int                            # axial control volumes
    D_cool: float
    D_fuel_hole: float
    D_compact: float
    pitch: float
    roughness: float

    # flow BCs
    m_dot: float                      # per-channel mass flow [kg/s]
    P_in: float
    T_in_C: float
    flow_upward: bool

    # analytic power profile (used when axial_shape != 'neutronics_table')
    qprime_max: float
    axial_shape: str
    peaking_factor: float

    # lattice sharing
    n_fuel_adjacent_to_coolant: int
    n_coolant_adjacent_to_fuel: int

    # gap emissivities
    emiss_compact: float
    emiss_fuel_hole: float

    # solid thermal
    graphite_k_model: str
    k_compact_eff: float
    packing_fraction: float

    # reactor-scale
    N_fuel_channels: int              # total fuel (compact) channels in core
    N_cool_channels: int              # total coolant channels in core
    Q_total_MWth: Optional[float]

    # neutronics coupling
    neutronics_file: Optional[str]    # path to neutronics CSV
    channel_case: str                 # average | hot | cold

    # internal: set at runtime by main()
    _neutronics_table: Optional[NeutronicsTable] = field(default=None, repr=False)
    _Q_per_channel_W: float = field(default=0.0, repr=False)


@dataclass
class BraytonInputs:
    P1: float
    T1_C: float
    pressure_ratio: float
    eta_c: float
    eta_t: float
    eps_recup: float
    dP_recup_cold: float
    dP_recup_hot: float
    dP_precooler: float
    dP_core_extra: float
    max_iter: int
    tol_K: float



# Power profile helpers


def qprime_analytic(z_mid: float, L_heated: float, qprime_max: float, shape: str) -> float:
    # Analytic axial profile. z_mid is measured from channel midplane.
    if abs(z_mid) > 0.5 * float(L_heated):
        return 0.0
    s = str(shape).lower().strip()
    if s == "cosine":
        return float(qprime_max * math.cos(math.pi * z_mid / float(L_heated)))
    if s == "uniform":
        return float(qprime_max)
    raise ValueError("Unknown axial_shape: {}".format(shape))


def qprime_from_total_power(
    Q_total_W: float, N_channels: int, L_heated: float,
    shape: str, peaking_factor: float
) -> float:
    N_channels = max(1, int(N_channels))
    P_channel = float(Q_total_W) / float(N_channels)
    Lh = max(1e-12, float(L_heated))
    pk = max(1e-12, float(peaking_factor))
    s = str(shape).lower().strip()
    if s == "cosine":
        qmax_eff = P_channel * (math.pi / (2.0 * Lh))
    elif s == "uniform":
        qmax_eff = P_channel / Lh
    else:
        raise ValueError("Unknown axial_shape: {}".format(shape))
    return float(qmax_eff / pk)



# Single-channel solver


def _channel_geometry(inputs: ChannelInputs):
    #Pre-compute geometry constants shared by both solvers
    D         = float(inputs.D_cool)
    A_flow    = math.pi * D ** 2 / 4.0
    rel_rough = float(inputs.roughness) / max(1e-12, D)
    R_cool    = 0.5 * D
    R_fuel    = 0.5 * float(inputs.D_fuel_hole)
    S_pair    = shape_factor_parallel_cylinders(R_cool, R_fuel, float(inputs.pitch))
    n_fuel_adj = max(1, int(inputs.n_fuel_adjacent_to_coolant))
    n_cool_adj = max(1, int(inputs.n_coolant_adjacent_to_fuel))
    N  = int(inputs.N)
    L  = float(inputs.L)
    dz = L / float(N)
    z_bottom = (np.arange(N) + 0.5) * dz
    z_mid    = z_bottom - 0.5 * L
    L_unheated_entrance = 0.5 * (L - float(inputs.L_heated))
    use_neutronics = (
        str(inputs.axial_shape).lower().strip() == "neutronics_table"
        and inputs._neutronics_table is not None
    )
    return dict(
        D=D, A_flow=A_flow, Dh=D, rel_rough=rel_rough,
        S_pair=S_pair, n_fuel_adj=n_fuel_adj, n_cool_adj=n_cool_adj,
        N=N, L=L, dz=dz, z_bottom=z_bottom, z_mid=z_mid,
        L_unheated_entrance=L_unheated_entrance,
        use_neutronics=use_neutronics,
    )


def _get_qprime(i: int, geo: dict, inputs: ChannelInputs) -> float:
    if geo["use_neutronics"]:
        z_in_heated = float(geo["z_bottom"][i]) - geo["L_unheated_entrance"]
        if not bool(inputs.flow_upward):
            # Downward flow: solver iterates from physical bottom (i=0) to top,
            # but the inlet is at the physical top.  Mirror the z lookup so that
            # the inlet node (i=0) uses the power at the physical top of the core
            # and the outlet node (i=N-1) uses the power at the physical bottom.
            z_in_heated = float(inputs.L_heated) - z_in_heated
        return inputs._neutronics_table.qprime(
            z_bottom_m=z_in_heated, case=str(inputs.channel_case))
    return qprime_analytic(
        z_mid=float(geo["z_mid"][i]),
        L_heated=float(inputs.L_heated),
        qprime_max=float(inputs.qprime_max) * float(inputs.peaking_factor),
        shape=str(inputs.axial_shape))


def _bulk_node(i: int, T_prev: float, P_prev: float, geo: dict,
               inputs: ChannelInputs, qprime: float):
    qprime_coolant = qprime * float(geo["n_fuel_adj"]) / float(geo["n_cool_adj"])
    dz = geo["dz"]; Dh = geo["Dh"]; A_flow = geo["A_flow"]

    props_prev = helium_props(T_prev, P_prev)
    cp  = props_prev["cp"]
    T_new = T_prev + (qprime_coolant * dz) / max(1e-12, float(inputs.m_dot) * cp)

    T_m   = 0.5 * (T_prev + T_new)
    props = helium_props(T_m, P_prev)
    rho, mu, k_He, Pr, gamma, R_He = (
        props["rho"], props["mu"], props["k"],
        props["Pr"], props["gamma"], props["R"])

    vel    = float(inputs.m_dot) / max(1e-12, rho * A_flow)
    Re_i   = rho * vel * Dh / max(1e-12, mu)
    f_i    = friction_factor_churchill(Re_i, geo["rel_rough"])
    Nu_i   = nusselt_internal(Re_i, Pr, f_i)
    h_i    = Nu_i * k_He / max(1e-12, Dh)
    a      = math.sqrt(gamma * R_He * T_m)
    Mach_i = vel / max(1e-12, a)

    rho_new = helium_props(T_new, P_prev)["rho"]
    dp_fric = f_i * (dz / Dh) * (rho * vel ** 2 / 2.0)
    dp_acc  = (float(inputs.m_dot) ** 2) * (
        1.0 / max(1e-12, rho_new) - 1.0 / max(1e-12, rho)
    ) / max(1e-12, A_flow ** 2)
    g       = 9.80665
    dp_grav = rho * g * dz * (1.0 if bool(inputs.flow_upward) else -1.0)
    dp      = dp_fric + dp_acc + dp_grav
    P_new   = P_prev - dp

    return T_new, P_new, dict(
        qprime_coolant=qprime_coolant, T_m=T_m, k_He=k_He,
        vel=vel, Re=Re_i, f=f_i, Nu=Nu_i, h=h_i, Mach=Mach_i, dp=dp,
    )


def _temperature_chain(T_m: float, k_He: float, h_i: float, P_prev: float,
                       qprime: float, qprime_coolant: float, geo: dict,
                       inputs: ChannelInputs):
    D = geo["D"]
    n_cool_adj = geo["n_cool_adj"]
    S_pair     = geo["S_pair"]

    if qprime <= 0.0:
        return T_m, T_m, T_m, T_m, T_m

    qpp_cool   = qprime_coolant / max(1e-12, math.pi * D)
    T_cw       = T_m + qpp_cool / max(1e-12, h_i)

    qprime_pair = qprime / float(n_cool_adj)
    k_gra       = graphite_k(T_m, model=str(inputs.graphite_k_model))
    T_fh        = T_cw + qprime_pair / max(1e-12, k_gra * S_pair)

    T_comp_surf_K, _ = solve_compact_surface_temp(
        T_fuel_hole_wall_K=T_fh,
        qprime_fuel_W_m=qprime_pair,
        D_compact=float(inputs.D_compact),
        D_fuel_hole=float(inputs.D_fuel_hole),
        P_Pa=max(1.0, P_prev),
        emiss_compact=float(inputs.emiss_compact),
        emiss_fuel_hole=float(inputs.emiss_fuel_hole),
    )

    dT_comp    = qprime_pair / max(1e-12, 4.0 * math.pi * float(inputs.k_compact_eff))
    T_comp_ctr = T_comp_surf_K + dT_comp

    T_kernel = triso_kernel_center_temp(
        T_compact_center_K=T_comp_ctr,
        qprime_fuel_W_m=qprime_pair,
        packing_fraction=float(inputs.packing_fraction),
        D_compact=float(inputs.D_compact),
    )
    return T_cw, T_fh, T_comp_surf_K, T_comp_ctr, T_kernel


def channel_bulk_only(inputs: ChannelInputs) -> Tuple[float, float, float]:
    geo = _channel_geometry(inputs)
    N   = geo["N"]
    T_prev = float(inputs.T_in_C) + 273.15
    P_prev = float(inputs.P_in)
    dP_total = 0.0
    for i in range(N):
        qprime = _get_qprime(i, geo, inputs)
        T_prev, P_prev, _ = _bulk_node(i, T_prev, P_prev, geo, inputs, qprime)
        dP_total += float(geo["dz"])   # placeholder — dp accumulated below
    # redo accumulating dp
    T_prev = float(inputs.T_in_C) + 273.15
    P_prev = float(inputs.P_in)
    dP_total = 0.0
    for i in range(N):
        qprime = _get_qprime(i, geo, inputs)
        T_prev, P_prev, nd = _bulk_node(i, T_prev, P_prev, geo, inputs, qprime)
        dP_total += nd["dp"]
    return T_prev, dP_total, P_prev


def solve_htgr_single_channel(inputs: ChannelInputs) -> pd.DataFrame:
    #Full channel solve including temperature chain.  Called once after
    #Brayton convergence, not on every iteration.

    geo = _channel_geometry(inputs)
    N   = geo["N"]
    z_bottom = geo["z_bottom"]
    z_mid    = geo["z_mid"]

    # Storage
    T_bulk           = np.zeros(N)
    P_arr            = np.zeros(N)
    v_arr            = np.zeros(N)
    Re_arr           = np.zeros(N)
    f_arr            = np.zeros(N)
    Nu_arr           = np.zeros(N)
    h_arr            = np.zeros(N)
    Mach_arr         = np.zeros(N)
    dP_cell          = np.zeros(N)
    qprime_fuel_W_m  = np.zeros(N)
    qprime_cool_W_m  = np.zeros(N)
    T_cool_wall      = np.zeros(N)
    T_fuel_hole_wall = np.zeros(N)
    T_compact_surf   = np.zeros(N)
    T_compact_center = np.zeros(N)
    T_triso_kernel   = np.zeros(N)

    T_prev = float(inputs.T_in_C) + 273.15
    P_prev = float(inputs.P_in)

    for i in range(N):
        qprime = _get_qprime(i, geo, inputs)
        T_new, P_new, nd = _bulk_node(i, T_prev, P_prev, geo, inputs, qprime)

        qprime_fuel_W_m[i] = qprime
        qprime_cool_W_m[i] = nd["qprime_coolant"]
        T_bulk[i]   = T_new
        P_arr[i]    = P_new
        v_arr[i]    = nd["vel"]
        Re_arr[i]   = nd["Re"]
        f_arr[i]    = nd["f"]
        Nu_arr[i]   = nd["Nu"]
        h_arr[i]    = nd["h"]
        Mach_arr[i] = nd["Mach"]
        dP_cell[i]  = nd["dp"]

        T_cw, T_fh, T_cs, T_cc, T_kn = _temperature_chain(
            T_m=nd["T_m"], k_He=nd["k_He"], h_i=nd["h"],
            P_prev=P_prev, qprime=qprime,
            qprime_coolant=nd["qprime_coolant"],
            geo=geo, inputs=inputs)

        T_cool_wall[i]      = T_cw
        T_fuel_hole_wall[i] = T_fh
        T_compact_surf[i]   = T_cs
        T_compact_center[i] = T_cc
        T_triso_kernel[i]   = T_kn

        T_prev = T_new
        P_prev = P_new

    df = pd.DataFrame({
        "z_m":               z_bottom,
        "z_mid_m":           z_mid,
        "qprime_fuel_W_m":   qprime_fuel_W_m,
        "qprime_coolant_W_m":qprime_cool_W_m,
        "T_bulk_C":          T_bulk - 273.15,
        "P_MPa":             P_arr * 1e-6,
        "v_m_s":             v_arr,
        "Re":                Re_arr,
        "f":                 f_arr,
        "Nu":                Nu_arr,
        "h_W_m2K":           h_arr,
        "Mach":              Mach_arr,
        "T_coolant_wall_C":  T_cool_wall  - 273.15,
        "T_fuel_hole_wall_C":T_fuel_hole_wall - 273.15,
        "T_compact_surf_C":  T_compact_surf  - 273.15,
        "T_compact_center_C":T_compact_center - 273.15,
        "T_TRISO_kernel_C":  T_triso_kernel - 273.15,
        "dP_cell_Pa":        dP_cell,
    })
    return df



# Direct recuperated Brayton cycle


def compressor_T2(T1: float, PR: float, gamma: float, eta_c: float) -> float:
    T2s = float(T1) * float(PR) ** ((gamma - 1.0) / gamma)
    return float(T1 + (T2s - T1) / max(1e-6, eta_c))


def turbine_T5(T4: float, PR: float, gamma: float, eta_t: float) -> float:
    T5s = float(T4) * float(PR) ** (-(gamma - 1.0) / gamma)
    return float(T4 - eta_t * (T4 - T5s))


def integrated_cycle_with_channel(
    core_inputs: ChannelInputs,
    br_inputs: BraytonInputs
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:

    cp, gamma, R = helium_cp_gamma_R()
    P1 = float(br_inputs.P1)
    T1 = float(br_inputs.T1_C) + 273.15
    PR = float(br_inputs.pressure_ratio)
    P2 = P1 * PR
    T2 = compressor_T2(T1, PR, gamma, br_inputs.eta_c)

    T3 = T2   # initial guess w no recuperation
    P4 = T4 = T5 = dP_core = float("nan")

    for _ in range(int(br_inputs.max_iter)):
        core_inputs.P_in   = P2 - float(br_inputs.dP_recup_cold)
        core_inputs.T_in_C = T3 - 273.15

        # Fast pass - some stuff but no temp chain
        T4_K, dP_ch, P4 = channel_bulk_only(core_inputs)
        T4      = T4_K
        dP_core = dP_ch + float(br_inputs.dP_core_extra)

        P5      = P1 + float(br_inputs.dP_precooler)
        PR_turb = max(1e-6, P4 / max(1e-6, P5))
        T5      = turbine_T5(T4, PR_turb, gamma, br_inputs.eta_t)

        eps    = max(0.0, min(0.9999, float(br_inputs.eps_recup)))
        T3_new = T2 + eps * (T5 - T2)

        if abs(T3_new - T3) < float(br_inputs.tol_K):
            T3 = T3_new
            break
        T3 = 0.7 * T3 + 0.3 * T3_new

    # One full solve with temperature chain at the converged operating point
    core_inputs.P_in   = P2 - float(br_inputs.dP_recup_cold)
    core_inputs.T_in_C = T3 - 273.15
    channel_df = solve_htgr_single_channel(core_inputs)

    # Recompute T4 from the full solve
    T4      = float(channel_df["T_bulk_C"].iloc[-1]) + 273.15
    dP_core = float(channel_df["dP_cell_Pa"].sum()) + float(br_inputs.dP_core_extra)
    P5      = P1 + float(br_inputs.dP_precooler)
    PR_turb = max(1e-6, P4 / max(1e-6, P5))
    T5      = turbine_T5(T4, PR_turb, gamma, br_inputs.eta_t)

    Wc   = cp * (T2 - T1)
    Wt   = cp * (T4 - T5)
    Wnet = Wt - Wc
    Qin  = cp * (T4 - T3)
    eta  = Wnet / max(1e-12, Qin)

    # Energy balance on recuperator:
    # cold side gain:  cp*(T3 - T2) = eps*cp*(T5 - T2)
    # hot side loss:   cp*(T5 - T5p) = cp*(T3 - T2)
    # tf T5p = T5 - eps*(T5 - T2) = T2 + (1-eps)*(T5 - T2)
    eps    = max(0.0, min(0.9999, float(br_inputs.eps_recup)))
    T5p    = T5 - eps * (T5 - T2)          # T5' recuperator hot-side outlet / precooler inlet
    Qrej   = cp * (T5p - T1)               # heat rejected per kg in precooler

    # First-law residual for sanity (i think should be zero perchance)
    first_law_residual = Qin - Wnet - Qrej 

    states = pd.DataFrame([
        {"state": 1,  "description": "Compressor inlet",          "P_Pa": P1, "T_K": T1,  "T_C": T1  - 273.15},
        {"state": 2,  "description": "Compressor outlet",         "P_Pa": P2, "T_K": T2,  "T_C": T2  - 273.15},
        {"state": 3,  "description": "Recup cold out / Core in",  "P_Pa": float(core_inputs.P_in), "T_K": T3, "T_C": T3 - 273.15},
        {"state": 4,  "description": "Core out / Turbine in",     "P_Pa": P4, "T_K": T4,  "T_C": T4  - 273.15},
        {"state": 5,  "description": "Turbine out / Recup hot in","P_Pa": P5, "T_K": T5,  "T_C": T5  - 273.15},
        {"state": "5p","description": "Recup hot out / Precooler in","P_Pa": P5, "T_K": T5p, "T_C": T5p - 273.15},
    ])

    summary = {
        "Wc_J_per_kg":          float(Wc),
        "Wt_J_per_kg":          float(Wt),
        "Wnet_J_per_kg":        float(Wnet),
        "Qin_J_per_kg":         float(Qin),
        "Qrej_J_per_kg":        float(Qrej),
        "first_law_residual_J": float(first_law_residual),
        "eta_th":               float(eta),
        "P4_Pa":                float(P4),
        "T1_K":                 float(T1),
        "T2_K":                 float(T2),
        "T3_K":                 float(T3),
        "T4_K":                 float(T4),
        "T5_K":                 float(T5),
        "T5p_K":                float(T5p),
        "dP_core_Pa":           float(dP_core),
    }
    return channel_df, states, summary



# Deck parser


def read_key_value_csv(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or len(row) < 2:
                continue
            key = str(row[0]).strip()
            val = str(row[1]).strip()
            if key.startswith("#") or key == "":
                continue
            data[key] = val
    return data


def _g(d, k, dflt): return float(d[k]) if k in d and d[k].strip() else float(dflt)
def _i(d, k, dflt): return int(float(d[k])) if k in d and d[k].strip() else int(dflt)
def _b(d, k, dflt): return str(d[k]).strip().lower() in ("true","1","yes","y") if k in d else bool(dflt)
def _s(d, k, dflt): return str(d[k]).strip() if k in d and d[k].strip() else str(dflt)


def parse_inputs_from_deck(deck: Dict[str, str]) -> Tuple[ChannelInputs, BraytonInputs]:
    ch = ChannelInputs(
        L=_g(deck, "L_m", 3.965),
        L_heated=_g(deck, "L_heated_m", 2.379),
        N=_i(deck, "N", 200),
        D_cool=_g(deck, "D_cool_m", 0.016),
        D_fuel_hole=_g(deck, "D_fuel_hole_m", 0.0127),
        D_compact=_g(deck, "D_compact_m", 0.01245),
        pitch=_g(deck, "pitch_m", 0.025),
        roughness=_g(deck, "roughness_m", 1e-5),
        m_dot=_g(deck, "m_dot_kg_s", 0.0093),
        P_in=_g(deck, "P_in_Pa", 7.0e6),
        T_in_C=_g(deck, "T_in_C", 300.0),
        flow_upward=_b(deck, "flow_upward", False),
        qprime_max=_g(deck, "qprime_max_W_m", 1600.0),
        axial_shape=_s(deck, "axial_shape", "neutronics_table"),
        peaking_factor=_g(deck, "peaking_factor", 1.2),
        n_fuel_adjacent_to_coolant=_i(deck, "n_fuel_adjacent_to_coolant", 6),
        n_coolant_adjacent_to_fuel=_i(deck, "n_coolant_adjacent_to_fuel", 3),
        emiss_compact=_g(deck, "emiss_compact", 0.85),
        emiss_fuel_hole=_g(deck, "emiss_fuel_hole", 0.85),
        graphite_k_model=_s(deck, "graphite_k_model", "pcea_table"),
        k_compact_eff=_g(deck, "k_compact_eff_W_mK", 6.0),
        packing_fraction=_g(deck, "packing_fraction", 0.30),
        N_fuel_channels=_i(deck, "N_fuel_channels", 1254),
        N_cool_channels=_i(deck, "N_cool_channels", 558),
        Q_total_MWth=(float(deck["Q_total_MWth"])
                      if "Q_total_MWth" in deck and deck["Q_total_MWth"].strip() else None),
        neutronics_file=_s(deck, "neutronics_file", "") or None,
        channel_case=_s(deck, "channel_case", "average"),
    )

    br = BraytonInputs(
        P1=_g(deck, "P1_Pa", 2.8e6),
        T1_C=_g(deck, "T1_C", 30.0),
        pressure_ratio=_g(deck, "pressure_ratio", 2.5),
        eta_c=_g(deck, "eta_c", 0.84),
        eta_t=_g(deck, "eta_t", 0.86),
        eps_recup=_g(deck, "eps_recup", 0.93),
        dP_recup_cold=_g(deck, "dP_recup_cold_Pa", 0.0),
        dP_recup_hot=_g(deck, "dP_recup_hot_Pa", 0.0),
        dP_precooler=_g(deck, "dP_precooler_Pa", 0.0),
        dP_core_extra=_g(deck, "dP_core_extra_Pa", 0.0),
        max_iter=_i(deck, "max_iter", 100),
        tol_K=_g(deck, "tol_K", 1e-3),
    )
    return ch, br



# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="HTGR SCA + Direct Brayton cycle")
    parser.add_argument("--deck",       type=str, default=None,
                        help="Key-value CSV input deck.")
    parser.add_argument("--out_prefix", type=str, default="htgr_out",
                        help="Output filename prefix.")
    args = parser.parse_args()

    # Parse inputs
    if not args.deck:
        parser.error("--deck is required. Please provide a key-value CSV input deck.")
    deck = read_key_value_csv(args.deck)
    ch_in, br_in = parse_inputs_from_deck(deck)

    # A relative neutronics_file is interpreted relative to the deck, not the
    # working directory, so the shipped deck runs from anywhere.
    if ch_in.neutronics_file and not os.path.isabs(ch_in.neutronics_file):
        if not os.path.exists(ch_in.neutronics_file):
            _beside_deck = os.path.join(os.path.dirname(os.path.abspath(args.deck)),
                                        ch_in.neutronics_file)
            if os.path.exists(_beside_deck):
                ch_in.neutronics_file = _beside_deck

    # Load neutronics table if requested
    ntable: Optional[NeutronicsTable] = None
    if (str(ch_in.axial_shape).lower().strip() == "neutronics_table"
            and ch_in.neutronics_file):
        print("Loading neutronics table from: {}".format(ch_in.neutronics_file))
        ntable = NeutronicsTable(ch_in.neutronics_file,
                                   N_fuel_channels=int(ch_in.N_fuel_channels))
        # L_heated and power come from the table by cade
        ch_in.L_heated = ntable.L_heated
        ch_in._neutronics_table = ntable
        print("  Neutronics L_heated  = {:.4f} m".format(ntable.L_heated))
        print("  Nodes                = {}".format(len(ntable.z_nodes_m)))
        print("  Q_avg_channel        = {:.1f} W".format(ntable.Q_per_channel["average"]))
        print("  Q_hot_channel        = {:.1f} W".format(ntable.Q_per_channel["hot"]))
        print("  Q_cold_channel       = {:.1f} W".format(ntable.Q_per_channel["cold"]))
        print("  Q_reactor (avg×N)    = {:.4f} MWth".format(ntable.Q_reactor_MWth))

    # analytic profile if no neutronics, but we got neutronics now
    if ntable is None and ch_in.Q_total_MWth is not None:
        N_ch = max(1, int(ch_in.N_fuel_channels))
        Q_total_W = float(ch_in.Q_total_MWth) * 1.0e6
        ch_in.qprime_max = qprime_from_total_power(
            Q_total_W=Q_total_W,
            N_channels=N_ch,
            L_heated=float(ch_in.L_heated),
            shape=str(ch_in.axial_shape),
            peaking_factor=float(ch_in.peaking_factor),
        )
        print("Analytic profile: qprime_max = {:.4g} W/m".format(ch_in.qprime_max))

    ch_avg = ch_in  # may already be "average"- kept as-is
    ch_avg.channel_case = "average"

    print("\nSolving average channel:")
    avg_channel_df, cycle_states, cycle_summary = integrated_cycle_with_channel(ch_avg, br_in)

    # Converged inlet conditions
    T3_K  = float(cycle_summary["T3_K"])   # core inlet temperature [K]
    P_in  = float(ch_avg.P_in)             # core inlet pressure [Pa]

    N_fuel = max(1, int(ch_in.N_fuel_channels))
    N_cool = max(1, int(ch_in.N_cool_channels))
    dz     = float(ch_in.L) / float(int(ch_in.N))

    # Average channel thermal power and reactor electric output
    P_avg_channel_W = float((avg_channel_df["qprime_coolant_W_m"] * dz).sum())
    if ntable is not None:
        P_reactor_fuel_W = ntable.Q_per_channel["average"] * N_fuel  # reference only
        P_reactor_W      = P_avg_channel_W * N_cool                   # cycle-consistent
    else:
        P_reactor_fuel_W = P_avg_channel_W * N_fuel
        P_reactor_W      = P_avg_channel_W * N_cool

    Wnet_J_per_kg = float(cycle_summary.get("Wnet_J_per_kg", float("nan")))
    We_channel_W  = float(ch_in.m_dot) * Wnet_J_per_kg
    We_reactor_W  = We_channel_W * N_cool

    print("Solving hot channel:")
    ch_hot = ch_in
    ch_hot.channel_case = "hot"
    ch_hot.T_in_C = T3_K - 273.15   # same inlet as average case
    ch_hot.P_in   = P_in             # same inlet pressure

    hot_channel_df = solve_htgr_single_channel(ch_hot)

    # Print summaries
    T4_avg = float(avg_channel_df["T_bulk_C"].iloc[-1])
    T4_hot = float(hot_channel_df["T_bulk_C"].iloc[-1])
    dT_avg = T4_avg - (T3_K - 273.15)
    dT_hot = T4_hot - (T3_K - 273.15)

    print("\nAverage Channel")
    print("Inlet T [C]:              {:.1f}".format(float(avg_channel_df["T_bulk_C"].iloc[0])))
    print("Outlet bulk T [C]:        {:.2f}".format(T4_avg))
    print("Channel dT [K]:           {:.2f}".format(dT_avg))
    print("Total channel dP [Pa]:    {:.1f}".format(float(avg_channel_df["dP_cell_Pa"].sum())))
    print("Peak TRISO kernel T [C]:  {:.2f}".format(float(avg_channel_df["T_TRISO_kernel_C"].max())))
    print("Peak compact center [C]:  {:.2f}".format(float(avg_channel_df["T_compact_center_C"].max())))
    print("Max Re:                   {:.0f}".format(float(avg_channel_df["Re"].max())))
    print("Max Mach:                 {:.5f}".format(float(avg_channel_df["Mach"].max())))

    print("\nHot Channel")
    print("Inlet T [C]:              {:.1f}".format(
        float(hot_channel_df["T_bulk_C"].iloc[0])))
    print("Outlet bulk T [C]:        {:.2f}".format(T4_hot))
    print("Channel dT [K]:           {:.2f}".format(dT_hot))
    print("dT ratio hot/avg:         {:.4f}  (expected Q ratio = {:.4f})".format(
        dT_hot / max(1e-12, dT_avg),
        (ntable.Q_per_channel["hot"] / ntable.Q_per_channel["average"])
        if ntable is not None else float("nan")))
    print("Total channel dP [Pa]:    {:.1f}".format(float(hot_channel_df["dP_cell_Pa"].sum())))
    print("Peak TRISO kernel T [C]:  {:.2f}".format(float(hot_channel_df["T_TRISO_kernel_C"].max())))
    print("Peak compact center [C]:  {:.2f}".format(float(hot_channel_df["T_compact_center_C"].max())))
    print("Max Re:                   {:.0f}".format(float(hot_channel_df["Re"].max())))
    print("Max Mach:                 {:.5f}".format(float(hot_channel_df["Mach"].max())))

    print("\nBrayton Cycle Summary (average case)")
    print("eta_th [%]:               {:.2f}".format(100.0 * float(cycle_summary["eta_th"])))
    print("T1 precooler outlet [C]:  {:.2f}".format(float(cycle_summary["T1_K"]) - 273.15))
    print("T2 compressor outlet [C]: {:.2f}".format(float(cycle_summary["T2_K"]) - 273.15))
    print("T3 core inlet [C]:        {:.2f}".format(T3_K - 273.15))
    print("T4 turbine inlet [C]:     {:.2f}".format(
        float(cycle_summary["T4_K"] - 273.15)))
    print("T5 turbine outlet [C]:    {:.2f}".format(float(cycle_summary["T5_K"] - 273.15)))
    print("T5'recup hot out (precooler inlet) [C]:     {:.2f}".format(
        float(cycle_summary["T5p_K"]) - 273.15))
    print("Core dP [Pa]:             {:.1f}".format(float(cycle_summary["dP_core_Pa"])))

    # Reactor-scale heat rejection
    Qrej_J_per_kg  = float(cycle_summary["Qrej_J_per_kg"])
    Qrej_channel_W = float(ch_in.m_dot) * Qrej_J_per_kg
    Qrej_reactor_W = Qrej_channel_W * N_cool
    first_law_res  = float(cycle_summary["first_law_residual_J"])

    print("\nReactor-Scale Summary")
    print("N_fuel_channels:          {}".format(N_fuel))
    print("N_cool_channels:          {}".format(N_cool))
    if ntable is not None:
        print("Thermal power - fuel side [MWth]:    {:.3f}".format(
            P_reactor_fuel_W / 1.0e6))
    print("Thermal power - coolant side [MWth]: {:.3f}".format(
        P_reactor_W / 1.0e6))
    print("Net electric [MWe]:       {:.3f}".format(
        We_reactor_W / 1.0e6))
    print("Heat rejected (precooler) [MWth]:     {:.3f}".format(
        Qrej_reactor_W / 1.0e6))
    print("Efficiency (eta_th) [%]:  {:.2f}".format(100.0 * float(cycle_summary["eta_th"])))
    print("First-law residual [J/kg]:{:.4f}".format(first_law_res))

    # Write outputs
    pfx = args.out_prefix
    avg_channel_df.to_csv("{}_avg_channel.csv".format(pfx), index=False, float_format="%.6e")
    hot_channel_df.to_csv("{}_hot_channel.csv".format(pfx), index=False, float_format="%.6e")
    cycle_states.to_csv("{}_cycle_states.csv".format(pfx), index=False, float_format="%.6e")

    reactor_summary = pd.DataFrame([{
        # Cycle / reactor performance — all from average case
        "N_fuel_channels":                N_fuel,
        "N_cool_channels":                N_cool,
        "Q_total_input_MWth":             (float(ch_in.Q_total_MWth)
                                           if ch_in.Q_total_MWth is not None else float("nan")),
        "P_coolant_avg_channel_kW":       P_avg_channel_W / 1e3,
        "P_reactor_coolant_side_MWth":    P_reactor_W / 1.0e6,
        "P_reactor_fuel_side_MWth":       (P_reactor_fuel_W / 1.0e6
                                           if ntable is not None else float("nan")),
        "We_reactor_MWe":                 We_reactor_W / 1.0e6,
        "Qrej_reactor_MWth":              Qrej_reactor_W / 1.0e6,
        "eta_th":                         float(cycle_summary.get("eta_th", float("nan"))),
        "first_law_residual_J_per_kg":    first_law_res,
        "T1_C":                           float(cycle_summary["T1_K"]) - 273.15,
        "T2_C":                           float(cycle_summary["T2_K"]) - 273.15,
        "T3_core_inlet_C":                T3_K - 273.15,
        "T4_avg_outlet_C":                T4_avg,
        "T5_C":                           float(cycle_summary["T5_K"]) - 273.15,
        "T5p_precooler_inlet_C":          float(cycle_summary["T5p_K"]) - 273.15,
        "core_dP_avg_Pa":                 float(avg_channel_df["dP_cell_Pa"].sum()),
        # Average channel temperatures
        "avg_T_TRISO_peak_C":             float(avg_channel_df["T_TRISO_kernel_C"].max()),
        "avg_T_compact_center_peak_C":    float(avg_channel_df["T_compact_center_C"].max()),
        # Hot channel
        "hot_T_outlet_C":                 T4_hot,
        "hot_dT_K":                       dT_hot,
        "hot_dT_ratio":                   dT_hot / max(1e-12, dT_avg),
        "core_dP_hot_Pa":                 float(hot_channel_df["dP_cell_Pa"].sum()),
        "hot_T_TRISO_peak_C":             float(hot_channel_df["T_TRISO_kernel_C"].max()),
        "hot_T_compact_center_peak_C":    float(hot_channel_df["T_compact_center_C"].max()),
    }])
    reactor_summary.to_csv("{}_reactor_summary.csv".format(pfx), index=False, float_format="%.6e")

    print("\nOutputs written:")
    print("  {}_avg_channel.csv".format(pfx))
    print("  {}_hot_channel.csv".format(pfx))
    print("  {}_cycle_states.csv".format(pfx))
    print("  {}_reactor_summary.csv".format(pfx))


if __name__ == "__main__":
    main()