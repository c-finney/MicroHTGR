import openmc
import config as cfg

# ====================================================================================================
# OPENMC MATERIALS DEFINITIONS
# ====================================================================================================

params = cfg.params

materials = openmc.Materials()

# ----- Fuel Kernel -----
fuel = openmc.Material(name="Fuel")
fuel.add_nuclide("U235", params["enrichment"])
fuel.add_nuclide("U238", 1.0 - params["enrichment"])
fuel.add_element("C", 1.0)
fuel.add_element("O", 0.50)
fuel.set_density("kg/m3", params["kernel_density"])
fuel.depletable = True

# ----- TRISO Layers -----
buffer = openmc.Material(name="Buffer") # Carbon buffer layer
buffer.add_element("C", 1.0)
buffer.set_density("kg/m3", params["buffer_density"])

pyc = openmc.Material(name="PyC") # Pyrolytic carbon layer (inner and outer)
pyc.add_element("C", 1.0)
pyc.set_density("kg/m3", params["pyc_density"])

sic = openmc.Material(name="SiC") # Silicon carbide layer
sic.add_element("Si", 1.0)
sic.add_element("C", 1.0)
sic.set_density("kg/m3", params["sic_density"])

# ----- Moderator/Reflector -----
graphite = openmc.Material(name="Graphite")
boron_mass_fraction = params["boron_ppm"] / 1e6
A_carbon = 12.011
A_boron = 10.811
boron_atom_fraction = boron_mass_fraction * A_carbon / A_boron
graphite.add_element("C", 1.0 - boron_atom_fraction)
graphite.add_element("B", boron_atom_fraction)
graphite.set_density("kg/m3", params["matrix_density"])
if params.get("deplete_graphite", False):
    graphite.depletable = True

# ----- Helium Coolant -----
helium = openmc.Material(name="Helium")
helium.add_nuclide("He4", 1.0)
helium.set_density("kg/m3", params["coolant_density"])

# ----- Boron Carbide Burnable Poison -----
b4c_poison = openmc.Material(name="B4C_Poison")
enrichment_10_poison = params["B10_enrichment_poison"]
mass_10 = openmc.data.atomic_mass("B10")
mass_11 = openmc.data.atomic_mass("B11")

# Number of atoms in one gram of boron mixture
n_10_poison = enrichment_10_poison / mass_10
n_11_poison = (1.0 - enrichment_10_poison) / mass_11
total_n_poison = n_10_poison + n_11_poison
grams_10_poison = n_10_poison / total_n_poison
grams_11_poison = n_11_poison / total_n_poison

# Now, figure out how much carbon needs to be in the poison to get an overall specified B10 weight percent
total_b10_weight_percent_poison = params["B10_wt_percent_poison"]
total_mass_poison = grams_10_poison / total_b10_weight_percent_poison
carbon_mass_poison = total_mass_poison - grams_10_poison - grams_11_poison

b4c_poison.add_nuclide("B10", grams_10_poison / total_mass_poison, 'wo')
b4c_poison.add_nuclide("B11", grams_11_poison / total_mass_poison, 'wo')
b4c_poison.add_element("C", carbon_mass_poison / total_mass_poison, 'wo')
b4c_poison.set_density("kg/m3", params["B4C_density_poison"])
b4c_poison.depletable = True

# ----- Boron Carbide Control Rod -----
b4c_control = openmc.Material(name="B4C_Control")
enrichment_10_control = params["B10_enrichment_control"]

# Number of atoms in one gram of boron mixture
n_10_control = enrichment_10_control / mass_10
n_11_control = (1.0 - enrichment_10_control) / mass_11
total_n_control = n_10_control + n_11_control
grams_10_control = n_10_control / total_n_control
grams_11_control = n_11_control / total_n_control

# Now, figure out how much carbon needs to be in the control rod to get an overall specified B10 weight percent
total_b10_weight_percent_control = params["B10_wt_percent_control"]
total_mass_control = grams_10_control / total_b10_weight_percent_control
carbon_mass_control = total_mass_control - grams_10_control - grams_11_control

b4c_control.add_nuclide("B10", grams_10_control / total_mass_control, 'wo')
b4c_control.add_nuclide("B11", grams_11_control / total_mass_control, 'wo')
b4c_control.add_element("C", carbon_mass_control / total_mass_control, 'wo')
b4c_control.set_density("kg/m3", params["B4C_density_control"])

# ----- Secondary Shutdown Rod Material -----
b4c_ss = openmc.Material.mix_materials(
    [b4c_control, helium],
    [0.55, 0.45], # 55% b4c_control + 45% helium by volume
    'vo',
    name="B4C_SS"
)

# ----- Incoloy 800H -----
incoloy800H = openmc.Material(name='Incoloy 800H')
incoloy800H.set_density('kg/m3', params["Incoloy800H_density"])
incoloy800H.add_element('Ni', 32.5, 'wo')
incoloy800H.add_element('Cr', 21.0, 'wo')
incoloy800H.add_element('Al', 0.40, 'wo')
incoloy800H.add_element('Ti', 0.40, 'wo')
incoloy800H.add_element('C', 0.08, 'wo')
incoloy800H.add_element('Mn', 1.0, 'wo')
incoloy800H.add_element('Si', 0.50, 'wo')
incoloy800H.add_element('S', 0.015, 'wo')
incoloy800H.add_element('Cu', 0.50, 'wo')
incoloy800H.add_element('Fe', 43.605, 'wo')

# ----- BeO Reflector -----
beo = openmc.Material(name='BeO')
beo.add_element('Be', 1.0)
beo.add_element('O', 1.0)
beo.set_density('kg/m3', params["BeO_density"])

materials += [fuel, buffer, pyc, sic, graphite, helium, b4c_poison, b4c_control, b4c_ss, incoloy800H, beo]

# # ====================================================================================================
# # HOMOGENIZED FUEL COMPACT
# # ====================================================================================================

# def make_homogenized_fuel_compact(
#     params: dict,
#     name: str = "HomogFuel"
# ) -> openmc.Material:
#     """
#     Create a single homogenized material representing a TRISO fuel compact.

#     Volume-averages the fuel kernel, all five TRISO coating layers, and the
#     surrounding graphite matrix based on the TRISO packing fraction.  The
#     result is a single openmc.Material that can fill a plain cylindrical cell,
#     completely replacing the explicit TRISO lattice geometry.

#     Physics notes
#     -------------
#     - Valid when the neutron mean free path >> TRISO diameter (~0.04 cm).
#       This holds for all HTGR thermal spectra.
#     - Resonance self-shielding within the kernel is slightly underestimated
#       (no explicit Dancoff factor).  Typical Δk vs. explicit TRISO:
#       50–200 pcm — acceptable for burnup / depletion studies.
#     - This is the standard approach used by VSOP, PEBBED, and Griffin for
#       production HTGR depletion calculations.

#     Volume fraction breakdown
#     -------------------------
#     Each TRISO sphere occupies a volume fraction `pf` of the compact.
#     Within that sphere the layer volume fractions are:

#         vf_kernel = pf * (r_kernel / r_opyc)^3
#         vf_layer  = pf * (r_outer^3 - r_inner^3) / r_opyc^3   (for each shell)
#         vf_matrix = 1 - pf                                      (graphite outside spheres)

#     Args:
#         params (dict): Simulation parameters dictionary (same dict used everywhere)
#         name (str): Base name for the returned material

#     Returns:
#         openmc.Material: Homogenized, depletable fuel compact material.
#     """

#     pf = params["triso_pf"]

#     r_k = params["kernel_radius"]
#     r_b = r_k + params["buffer_thickness"]
#     r_i = r_b + params["ipyc_thickness"]
#     r_s = r_i + params["sic_thickness"]
#     r_o = r_s + params["opyc_thickness"]

#     # Volume fractions of each region (relative to total compact volume)
#     # Computes the volume fraction of a spherical shell between r_in and r_out, scaled by packing fraction.
#     def shell_vf(r_out, r_in):
#         return pf * (r_out**3 - r_in**3) / r_o**3

#     vf_kernel = pf * (r_k / r_o)**3
#     vf_buffer = shell_vf(r_b, r_k)
#     vf_ipyc   = shell_vf(r_i, r_b)
#     vf_sic    = shell_vf(r_s, r_i)
#     vf_opyc   = shell_vf(r_o, r_s)
#     vf_matrix = 1.0 - pf          # graphite matrix surrounding the spheres

#     # Sanity check — all fractions must sum to 1
#     vf_total = vf_kernel + vf_buffer + vf_ipyc + vf_sic + vf_opyc + vf_matrix
#     assert abs(vf_total - 1.0) < 1e-9, (
#         f"Homogenized compact volume fractions sum to {vf_total:.10f}, expected 1.0"
#     )

#     # Build temporary constituent materials for mix_materials()
#     kernel_tmp = openmc.Material(name=f"{name}_kernel_tmp")
#     kernel_tmp.add_nuclide("U235", params["enrichment"])
#     kernel_tmp.add_nuclide("U238", 1.0 - params["enrichment"])
#     kernel_tmp.add_element("C", 1.0)
#     kernel_tmp.add_element("O", 0.50)
#     kernel_tmp.set_density("kg/m3", params["kernel_density"])

#     buffer_tmp = openmc.Material(name=f"{name}_buffer_tmp")
#     buffer_tmp.add_element("C", 1.0)
#     buffer_tmp.set_density("kg/m3", params["buffer_density"])

#     pyc_tmp = openmc.Material(name=f"{name}_pyc_tmp")
#     pyc_tmp.add_element("C", 1.0)
#     pyc_tmp.set_density("kg/m3", params["pyc_density"])

#     sic_tmp = openmc.Material(name=f"{name}_sic_tmp")
#     sic_tmp.add_element("Si", 1.0)
#     sic_tmp.add_element("C", 1.0)
#     sic_tmp.set_density("kg/m3", params["sic_density"])

#     # Graphite matrix — same composition as the module-level graphite material
#     boron_mass_fraction = params["boron_ppm"] / 1e6
#     A_carbon = 12.011
#     A_boron  = 10.811
#     boron_atom_fraction = boron_mass_fraction * A_carbon / A_boron
#     matrix_tmp = openmc.Material(name=f"{name}_matrix_tmp")
#     matrix_tmp.add_element("C", 1.0 - boron_atom_fraction)
#     matrix_tmp.add_element("B", boron_atom_fraction)
#     matrix_tmp.set_density("kg/m3", params["matrix_density"])

#     # IPyC and OPyC use the same pyc material at the same density
#     homog = openmc.Material.mix_materials(
#         [kernel_tmp, buffer_tmp, pyc_tmp,   sic_tmp, pyc_tmp,   matrix_tmp],
#         [vf_kernel,  vf_buffer,  vf_ipyc,   vf_sic,  vf_opyc,   vf_matrix],
#         'vo',
#         name=name,
#     )
#     homog.depletable = True
#     return homog

def make_rpt_inner_material(
    params: dict,
    r_rpt: float,
    name: str = "RPTInner"
) -> openmc.Material:
    """
    Create the homogenized inner-cylinder material for the RPT method.

    In the Reactivity Equivalent Physical Transform (RPT, Kim & Cho 2011),
    the N TRISO particles in a compact are conceptually "compressed" into a
    smaller inner cylinder of radius r_rpt < r_compact.  The inner cylinder
    is then homogenized in a volume-weighted sense:

        pf_inner = pf * (r_compact / r_rpt)^2

    The inner cylinder contains TRISO sphere material (all layers) at effective
    packing fraction pf_inner, plus graphite matrix filling the remainder
    (1 - pf_inner).  The outer annulus (r_rpt < r < r_compact) is pure graphite.

    Key properties
    --------------
    - r_rpt = r_compact * sqrt(pf): minimum — pf_inner = 1, no graphite in inner
      cylinder, maximum self-shielding (same as our previous wrong implementation)
    - r_rpt = r_compact: degenerate — pf_inner = pf, equals flat homogenization
    - The actual r_rpt is calibrated so that k_eff(RPT) = k_eff(explicit TRISO)
    - Fissile inventory is conserved regardless of r_rpt because:
        V_inner * vf_kernel_inner = π*r_rpt^2*h * pf_inner*(r_k/r_o)^3
                                  = π*r_compact^2*h * pf*(r_k/r_o)^3  (constant)
    - material.volume for depletion = π * r_rpt^2 * h * n_compacts

    Volume fractions within the inner cylinder
    ------------------------------------------
        vf_kernel = pf_inner * (r_k / r_o)^3
        vf_layer  = pf_inner * (r_outer^3 - r_inner^3) / r_o^3   (each shell)
        vf_matrix = 1 - pf_inner

    Constraint: r_rpt >= r_compact * sqrt(pf)  (so that pf_inner <= 1)

    Args:
        params (dict): Simulation parameters dictionary
        r_rpt (float): Inner cylinder radius in cm; must satisfy pf*(r_compact/r_rpt)^2 <= 1
        name (str): Base name for the returned material

    Returns:
        openmc.Material: Depletable homogenized RPT inner-cylinder material.
    """

    r_compact = params["compact_radius"]
    pf        = params["triso_pf"]
    pf_inner  = pf * (r_compact / r_rpt)**2

    r_min_rpt = r_compact * pf**0.5
    assert r_rpt >= r_min_rpt - 1e-9, (
        f"r_rpt = {r_rpt:.4f} cm is below the minimum {r_min_rpt:.4f} cm "
        f"(r_compact*sqrt(pf)); pf_inner would exceed 1."
    )
    pf_inner = min(pf_inner, 1.0)   # clamp floating-point edge case at r_rpt == r_min

    r_k = params["kernel_radius"]
    r_b = r_k + params["buffer_thickness"]
    r_i = r_b + params["ipyc_thickness"]
    r_s = r_i + params["sic_thickness"]
    r_o = r_s + params["opyc_thickness"]

    # Computes the volume fraction of a spherical shell between r_in and r_out, scaled by effective packing fraction.
    def shell_vf(r_out, r_in):
        return pf_inner * (r_out**3 - r_in**3) / r_o**3

    vf_kernel = pf_inner * (r_k / r_o)**3
    vf_buffer = shell_vf(r_b, r_k)
    vf_ipyc   = shell_vf(r_i, r_b)
    vf_sic    = shell_vf(r_s, r_i)
    vf_opyc   = shell_vf(r_o, r_s)
    vf_matrix = 1.0 - pf_inner

    vf_total = vf_kernel + vf_buffer + vf_ipyc + vf_sic + vf_opyc + vf_matrix
    assert abs(vf_total - 1.0) < 1e-9, (
        f"RPT inner volume fractions sum to {vf_total:.10f}, expected 1.0"
    )

    kernel_tmp = openmc.Material(name=f"{name}_kernel_tmp")
    kernel_tmp.add_nuclide("U235", params["enrichment"])
    kernel_tmp.add_nuclide("U238", 1.0 - params["enrichment"])
    kernel_tmp.add_element("C", 1.0)
    kernel_tmp.add_element("O", 0.50)
    kernel_tmp.set_density("kg/m3", params["kernel_density"])

    buffer_tmp = openmc.Material(name=f"{name}_buffer_tmp")
    buffer_tmp.add_element("C", 1.0)
    buffer_tmp.set_density("kg/m3", params["buffer_density"])

    pyc_tmp = openmc.Material(name=f"{name}_pyc_tmp")
    pyc_tmp.add_element("C", 1.0)
    pyc_tmp.set_density("kg/m3", params["pyc_density"])

    sic_tmp = openmc.Material(name=f"{name}_sic_tmp")
    sic_tmp.add_element("Si", 1.0)
    sic_tmp.add_element("C", 1.0)
    sic_tmp.set_density("kg/m3", params["sic_density"])

    boron_mass_fraction = params["boron_ppm"] / 1e6
    A_carbon = 12.011
    A_boron  = 10.811
    boron_atom_fraction = boron_mass_fraction * A_carbon / A_boron
    matrix_tmp = openmc.Material(name=f"{name}_matrix_tmp")
    matrix_tmp.add_element("C", 1.0 - boron_atom_fraction)
    matrix_tmp.add_element("B", boron_atom_fraction)
    matrix_tmp.set_density("kg/m3", params["matrix_density"])

    rpt_mat = openmc.Material.mix_materials(
        [kernel_tmp, buffer_tmp, pyc_tmp,  sic_tmp, pyc_tmp,  matrix_tmp],
        [vf_kernel,  vf_buffer,  vf_ipyc,  vf_sic,  vf_opyc,  vf_matrix],
        'vo',
        name=name,
    )
    rpt_mat.depletable = True
    return rpt_mat