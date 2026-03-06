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

# ----- Helium Coolant -----
helium = openmc.Material(name="Helium")
helium.add_nuclide("He4", 1.0)
helium.set_density("kg/m3", params["coolant_density"])

# ----- Boron Carbide Burnable Poison -----
b4c_poison = openmc.Material(name="B4C_Poison")
enrichment_10_poison = params["B10_enrichment_poison"]
mass_10 = openmc.data.atomic_mass("B10")
mass_11 = openmc.data.atomic_mass("B11")

# number of atoms in one gram of boron mixture
n_10_poison = enrichment_10_poison / mass_10
n_11_poison = (1.0 - enrichment_10_poison) / mass_11
total_n_poison = n_10_poison + n_11_poison
grams_10_poison = n_10_poison / total_n_poison
grams_11_poison = n_11_poison / total_n_poison

# now, figure out how much carbon needs to be in the poison to get
# an overall specified B10 weight percent
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

# number of atoms in one gram of boron mixture
n_10_control = enrichment_10_control / mass_10
n_11_control = (1.0 - enrichment_10_control) / mass_11
total_n_control = n_10_control + n_11_control
grams_10_control = n_10_control / total_n_control
grams_11_control = n_11_control / total_n_control

# now, figure out how much carbon needs to be in the control rod to get
# an overall specified B10 weight percent
total_b10_weight_percent_control = params["B10_wt_percent_control"]
total_mass_control = grams_10_control / total_b10_weight_percent_control
carbon_mass_control = total_mass_control - grams_10_control - grams_11_control

b4c_control.add_nuclide("B10", grams_10_control / total_mass_control, 'wo')
b4c_control.add_nuclide("B11", grams_11_control / total_mass_control, 'wo')
b4c_control.add_element("C", carbon_mass_control / total_mass_control, 'wo')
b4c_control.set_density("kg/m3", params["B4C_density_control"])

# ----- Secondary Shutdown Rod Material (55% B4C control + 45% Helium by volume) -----
b4c_ss = openmc.Material.mix_materials(
    [b4c_control, helium],
    [0.55, 0.45],
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