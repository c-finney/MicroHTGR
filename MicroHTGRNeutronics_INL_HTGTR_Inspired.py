
import os
import math
import shutil
import openmc
import numpy as np
import openmc.deplete
from datetime import datetime

cross_sections_path = '/home/cade/Desktop/OpenMC/CrossSections/cross_sections.xml'
os.environ['OPENMC_CROSS_SECTIONS'] = cross_sections_path

# ====================================================================================================
# 0. CREATE DATE-STAMPED RUN DIRECTORY AND INITIALIZE MODEL
# ====================================================================================================

now = datetime.now()
run_name = f"htgr_run_{now.strftime('%H.%M.%S_%m.%d.%Y')}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

OUTPUT_BASE = os.path.join(PARENT_DIR, "MicroHTGR_Output")
os.makedirs(OUTPUT_BASE, exist_ok=True)

BASE_DIR = os.path.join(OUTPUT_BASE, run_name)
os.makedirs(BASE_DIR, exist_ok=True)

os.chdir(BASE_DIR)

# Saves cross_sections.xml file to run directory, useful to ensure used cross sections are correct
shutil.copy2(cross_sections_path, os.path.join(BASE_DIR, 'cross_sections.xml'))

model = openmc.model.Model()

# ====================================================================================================
# 1. GLOBAL PARAMETERS (Most Major Design Variables Live Here)
# ====================================================================================================

# Unless otherwise stated all length dimensions are in cm

params = {
    # ----- Fuel Kernel -----
    "fuel_type": "UCO",
    "enrichment": 0.2, # U-235 atom fraction
    "kernel_density": 10.5, # g/cm3

    # ----- TRISO Layers -----
    "kernel_radius": 0.021485,
    "buffer_thickness": 0.01,
    "ipyc_thickness": 0.004,
    "sic_thickness": 0.0035,
    "opyc_thickness": 0.004,

    # ----- Fuel Compact  -----
    "compact_radius": 0.635,
    "compact_height": 4.93,
    "triso_pf": 0.15, # Packing fraction of triso particles in fuel compact

    # ----- Coolant Channel -----
    "n_coolant_channels_per_block": 18, # number of coolant channels per assembly
    "coolant_radius": 0.8, 

    # ----- Hexagonal Lattice -----
    "fuel_to_coolant_distance": 1.88,
    "bundle_pitch": 16,

    # ----- Core Dimensions -----
    "core_radius": 90.0,
    "core_height": 237.9,
    "reflector_thickness": 79.3,
    "n_ax_zones": 50,

    # ----- Control Rods -----
    "control_material": "B4C", # Currently only pure B4C or Hf implemented
    "control_radius": 0.50,
    "control_insertion": 0.50, # fractional insertion (0-1.0)

    # ----- Graphite Moderator -----
    "boron_wt_percent": 0.001,

    # ----- Steady-State Operation -----
    "power_MW": 5.0, # desired reactor power, used as initial condition
    "temperature_K": 900 # desired reactor temperature, used as intitial condition
}

# ====================================================================================================
# 2. MATERIAL DEFINITIONS (Material Constants And Compositions)
# ====================================================================================================

materials = openmc.Materials()

# ----- Fuel Kernel -----
fuel = openmc.Material(name="Fuel")
fuel.add_nuclide("U235", params["enrichment"])
fuel.add_nuclide("U238", 1.0 - params["enrichment"])
fuel.add_element("C", 1.0)
fuel.set_density("g/cm3", params["kernel_density"])
fuel.depletable = True

# ----- TRISO Layers -----
buffer = openmc.Material(name="Buffer") # Carbon buffer layer
buffer.add_element("C", 1.0)
buffer.set_density("g/cm3", 1.0)

pyc = openmc.Material(name="PyC") # Pyrolytic carbon layer (inner and outer)
pyc.add_element("C", 1.0)
pyc.set_density("g/cm3", 1.9)

sic = openmc.Material(name="SiC") # Silicon carbide layer
sic.add_element("Si", 1.0)
sic.add_element("C", 1.0)
sic.set_density("g/cm3", 3.2)

# ----- Moderator / Reflector -----
graphite = openmc.Material(name="Graphite")
graphite.add_element("C", 1.0)
graphite.set_density("g/cm3", 1.7)

# ----- Helium Coolant -----
helium = openmc.Material(name="Helium")
helium.add_nuclide("He4", 1.0)
helium.set_density("g/cm3", 0.00018)

# ----- Control Rods -----
control = openmc.Material(name="Control Rod")

if params["control_material"] == "B4C":
    control.add_nuclide("B10", 0.2)
    control.add_nuclide("B11", 0.8)
    control.add_element("C", 1.0)
    control.set_density("g/cm3", 2.52)

elif params["control_material"] == "Hf":
    control.add_element("Hf", 1.0)
    control.set_density("g/cm3", 13.3)

materials += [fuel, buffer, pyc, sic, graphite, helium, control]
materials.export_to_xml()

# ====================================================================================================
# 3. TRISO PARTICLE CREATION
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

c_triso_fuel   = openmc.Cell(name = 'c_triso_fuel'     , fill = fuel,     region = -s_fuel)
c_triso_buffer = openmc.Cell(name = 'c_triso_c_buffer' , fill = buffer,   region = +s_fuel & -s_buffer)
c_triso_ipyc   = openmc.Cell(name = 'c_triso_pyc_inner', fill = pyc,      region = +s_buffer & -s_ipyc)
c_triso_sic    = openmc.Cell(name = 'c_triso_sic'      , fill = sic,      region = +s_ipyc & -s_sic)
c_triso_opyc   = openmc.Cell(name = 'c_triso_pyc_outer', fill = pyc,      region = +s_sic & -s_opyc)
c_triso_matrix = openmc.Cell(name = 'c_triso_matrix'   , fill = graphite, region = +s_opyc)

triso_universe = openmc.Universe(cells=[c_triso_fuel, c_triso_buffer, c_triso_ipyc, c_triso_sic, c_triso_opyc, c_triso_matrix])

# ====================================================================================================
# 4. FUEL COMPACT AND COOLANT CHANNEL LATTICE CREATION
# ====================================================================================================

reactor_bottom = 0.0
reactor_top = reactor_bottom + params["core_height"]

axial_section_height = params["core_height"] / params["n_ax_zones"]

# superimposed search lattice
triso_lattice_shape = (4, 4, int(axial_section_height / 0.5))

fuel_cyl = openmc.ZCylinder(r = params["compact_radius"])
coolant_cyl = openmc.ZCylinder(r = params["coolant_radius"])

# create a TRISO lattice for one axial section (to be used in the rest of the axial zones)
# center the TRISO region on the origin so it fills lattice cells appropriately
zmin_local = -0.5 * axial_section_height
zmax_local =  0.5 * axial_section_height
min_z = openmc.ZPlane(z0 = zmin_local)
max_z = openmc.ZPlane(z0 = zmax_local)

# region in which TRISOs are generated
triso_region = -fuel_cyl & +min_z & -max_z

rand_spheres = openmc.model.pack_spheres(radius=r_opyc, region=triso_region, pf=params["triso_pf"])

print(f"Number of TRISOs created per axial zone: {len(rand_spheres)}")

# ---- hard boundary filter (critical at high PF)
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

# ---- report achieved PF
V_triso = (4/3) * np.pi * r_opyc**3
V_compact = np.pi * params["compact_radius"]**2 * axial_section_height
actual_pf = len(safe_trisos) * V_triso / V_compact

print(f"Number of safe TRISOs per axial zone: {len(safe_trisos)}")
print(f"Requested TRISO PF: {params['triso_pf']:.3f}")
print(f"Achieved TRISO PF: {actual_pf:.3f}")

random_trisos = [openmc.model.TRISO(r_opyc, triso_universe, i) for i in safe_trisos]

pitch = (urc - llc) / triso_lattice_shape

# insert TRISOs into a lattice to accelerate point location queries
triso_lattice = openmc.model.create_triso_lattice(random_trisos, llc, pitch, triso_lattice_shape, graphite)

axial_coords = np.linspace(reactor_bottom, reactor_top, params["n_ax_zones"] + 1)
fuel_lattice_univs = []

m_colors = {}

for z_min, z_max in zip(axial_coords[0:-1], axial_coords[1:]):
    # use the middle of the axial section to compute the temperature and density
    ax_pos = 0.5 * (z_min + z_max)

    min_z_plane = openmc.ZPlane(z0=z_min)
    max_z_plane = openmc.ZPlane(z0=z_max)

    # create solid cells, which don't require us to clone materials in order to set temperatures
    fuel_ch_cell = openmc.Cell(region=-fuel_cyl, fill=triso_lattice)

    fuel_ch_matrix_cell = openmc.Cell(region=+fuel_cyl, fill=graphite)

    graphite_cell = openmc.Cell(fill=graphite)

    coolant_matrix_cell = openmc.Cell(region=+coolant_cyl, fill=graphite)

    # create fluid cells, which require us to clone the material in order to be able to
    # set unique densities
    coolant_cell = openmc.Cell(region=-coolant_cyl, fill=helium)
    
    coolant_cell.fill = [helium.clone() for i in range(55 * params["n_coolant_channels_per_block"])]

    for mat in range(len(coolant_cell.fill)):
        m_colors[coolant_cell.fill[mat]] = 'red'

    # Define a universe for each type of solid-only pin (fuel, poison, and graphite)
    f = openmc.Universe(cells=[fuel_ch_cell, fuel_ch_matrix_cell])
    c = openmc.Universe(cells=[coolant_cell, coolant_matrix_cell])
    g = openmc.Universe(cells=[graphite_cell])

    d = [f] * 2

    ring2 = ([f] + [c]) * 6
    ring3 = ([c] + d) * 6
    ring4 = (d + [c] + [f]) * 6

    # inner two rings where there aren't any fuel/compact/poison pins
    ring1 = [f] * 6
    ring0 = [g]

    fuel_lattice_univs.append([ring4, ring3, ring2, ring1, ring0])

# ====================================================================================================
# 5. FUEL ASSEMBLY CREATION (Hexagonal Lattice of Fuel Pins)
# ====================================================================================================

# This creates ONE assembly (hexagonal arrangement of fuel pins)
fuel_assembly_lat = openmc.HexLattice(name = "Fuel Lattice")
fuel_assembly_lat.orientation = 'x'
fuel_assembly_lat.center = (0.0, 0.0, 0.5 * (reactor_bottom + reactor_top))
fuel_assembly_lat.pitch = (params["fuel_to_coolant_distance"], axial_section_height)
fuel_assembly_lat.universes = fuel_lattice_univs

graphite_outer_cell = openmc.Cell(fill=graphite)
inf_graphite_universe = openmc.Universe(cells=[graphite_outer_cell])
fuel_assembly_lat.outer = inf_graphite_universe

hex_prism_fuel = openmc.model.hexagonal_prism(params["bundle_pitch"] / math.sqrt(3.0), 'x')
min_z = openmc.ZPlane(z0 = reactor_bottom)
max_z = openmc.ZPlane(z0 = reactor_top)

fuel_assembly_cell = openmc.Cell(fill=fuel_assembly_lat, region=hex_prism_fuel & +min_z & -max_z)
fuel_assembly_univ = openmc.Universe(cells=[fuel_assembly_cell])

# ====================================================================================================
# 6. REFLECTOR ASSEMBLY CREATION
# ====================================================================================================

hex_prism_refl = openmc.model.hexagonal_prism(params["bundle_pitch"] / math.sqrt(3.0), 'x')

graphite_refl_cell = openmc.Cell(fill = graphite)

reflector_assembly_cell = openmc.Cell(fill = graphite, region = hex_prism_refl & +min_z & -max_z)
reflector_assembly_univ = openmc.Universe(cells = [reflector_assembly_cell])

# ====================================================================================================
# 7. CORE LATTICE CREATION
# ====================================================================================================

f = fuel_assembly_univ
r = reflector_assembly_univ

ring0 = [f]
ring1 = [f] * 6
ring2 = [f] * 12
ring3 = [f] * 18
ring4 = ([r] + [f] + [f] + [f]) * 6

core_lattice_univs = [ring4, ring3, ring2, ring1, ring0]

core_lattice = openmc.HexLattice(name="Core Lattice")
core_lattice.center = (0.0, 0.0)
core_lattice.pitch = (params["bundle_pitch"],)
core_lattice.universes = core_lattice_univs

# ====================================================================================================
# 7. FULL CORE AND REFLECTOR CREATION
# ====================================================================================================

core_outer_cell = openmc.Cell(fill = graphite)
core_outer_univ = openmc.Universe(cells = [core_outer_cell])
core_lattice.outer = core_outer_univ

core_cyl = openmc.ZCylinder(r = params["core_radius"], boundary_type = 'vacuum')
core_cell = openmc.Cell(fill=core_lattice, region=-core_cyl & +min_z & -max_z)

top_refl_z = reactor_top + params["reflector_thickness"]
bottom_refl_z = reactor_bottom - params["reflector_thickness"]
top_refl = openmc.ZPlane(z0 = top_refl_z, boundary_type = 'vacuum')
bottom_refl = openmc.ZPlane(z0 = bottom_refl_z, boundary_type = 'vacuum')

top_refl_cell = openmc.Cell(fill=graphite, region=-core_cyl & +max_z & -top_refl)

bottom_refl_cell = openmc.Cell(fill=graphite, region=-core_cyl & +bottom_refl & -min_z)

geometry = openmc.Geometry([core_cell, top_refl_cell, bottom_refl_cell])

model.geometry = geometry

# ====================================================================================================
# 8. GEOMETRY PLOT GENERATION
# ====================================================================================================

m_colors[fuel] = 'palegreen'
m_colors[buffer] = 'sandybrown'
m_colors[pyc] = 'orange'
m_colors[sic] = 'yellow'
m_colors[graphite] = 'darkblue'

plot1 = openmc.Plot()
plot1.filename = 'plot1'
plot1.width = (2 * params["core_radius"], 2 * params["core_height"])
plot1.basis = 'xz'
plot1.origin = (0.0, 0.0, params["core_height"] / 2.0)
plot1.pixels = (800, 1200)
plot1.color_by = 'material'
plot1.colors = m_colors

plot2 = openmc.Plot()
plot2.filename = 'plot2'
plot2.width = (params["bundle_pitch"], params["bundle_pitch"])
plot2.basis = 'xy'
plot2.origin = (0.0, 0.0, axial_section_height / 4.0)
plot2.pixels = (2000, 2000)
plot2.color_by = 'material'
plot2.colors = m_colors

plot3 = openmc.Plot()
plot3.filename = 'plot3'
plot3.width = plot2.width
plot3.basis = plot2.basis
plot3.origin = plot2.origin
plot3.pixels = plot2.pixels
plot3.color_by = 'cell'

plot4 = openmc.Plot()
plot4.filename = 'plot4'
plot4.width = (2 * params["core_radius"], 2 * params["core_radius"])
plot4.basis = 'xy'
plot4.origin = (0.0, 0.0, axial_section_height / 4.0)
plot4.pixels = (1000, 1000)
plot4.color_by = 'material'
plot4.colors = m_colors

plot5 = openmc.Plot()
plot5.filename = 'plot5'
plot5.width = plot4.width
plot5.basis = plot4.basis
plot5.origin = plot4.origin
plot5.pixels = plot4.pixels
plot5.color_by = 'cell'

model.plots = openmc.Plots([plot1, plot2, plot3, plot4, plot5])

# ====================================================================================================
# 9. TALLY CREATION
# ====================================================================================================

tallies = openmc.Tallies()

fuel_filter = openmc.MaterialFilter(fuel)

energy_bins = np.logspace(-9, 7, 200) # thermal --> fast
energy_filter = openmc.EnergyFilter(energy_bins)

flux_tally = openmc.Tally(name="flux")
flux_tally.scores = ["flux"]
flux_tally.filters = [energy_filter]

power_tally = openmc.Tally(name="power")
power_tally.scores = ["fission"]
power_tally.filters = [fuel_filter]

tallies += [flux_tally, power_tally]

model.tallies = tallies

# ====================================================================================================
# 10. MONTE CARLO SETTINGS
# ====================================================================================================

settings = openmc.Settings()
settings.run_mode = "eigenvalue"
settings.batches = 2500
settings.inactive = 500
settings.particles = 50_000
settings.power = params["power_MW"] * 1e6
settings.temperature = {
    'method': 'interpolation',
    'range': (293.0, 1800.0),
    'tolerance': 100.0
}
settings.trace = (1, 1, 16673)

r_dist = openmc.stats.Uniform(a = 0.0, b = params["core_radius"])
phi_dist = openmc.stats.Uniform(a = 0.0, b = 2*np.pi)
z_dist = openmc.stats.Uniform(a = reactor_bottom, b = reactor_top)
source = openmc.IndependentSource()
source.space = openmc.stats.CylindricalIndependent(
    r = r_dist,
    phi = phi_dist,
    z = z_dist,
    origin = (0.0, 0.0, 0.0)  # center of the cylinder
)
settings.source = source
model.settings = settings

# ============================================================
# 11. RUN OPENMC
# ============================================================

model.export_to_xml()

openmc.plot_geometry(output=False, cwd=BASE_DIR)

openmc.run(
    cwd=BASE_DIR,
    threads=24,
    output=True
)

# # ====================================================================================================
# # CONTROL ROD UNIVERSE (Axially Inserted)
# # ====================================================================================================

# # Creates model of control rods that are axially inserted from above

# control_cyl = openmc.ZCylinder(r = params["control_radius"])

# control_top = openmc.ZPlane(z0 = params["core_height"] * 0.5)  # Top of core
# control_bottom = openmc.ZPlane(z0 = params["core_height"] * (0.5 - params["control_insertion"]))  # Insertion depth

# control_region = -control_cyl & -control_top & +control_bottom

# # Helium region is the coolant channel where control rod is NOT present
# helium_region = -control_cyl & ~control_region & -top & +bottom

# control_cell = openmc.Cell(fill=control, region=control_region)
# helium_cell = openmc.Cell(fill=helium, region=helium_region)

# control_rod = openmc.Universe(cells=[control_cell, helium_cell])

# # ====================================================================================================
# # DEPLETION
# # ====================================================================================================

# operator = openmc.deplete.Operator(
#     geometry=geometry,
#     settings=settings,
#     chain_file="chain_casl.xml"
# )

# timesteps = [30.0] * 12

# integrator = openmc.deplete.PredictorIntegrator(
#     operator,
#     timesteps,
#     power=settings.power
# )