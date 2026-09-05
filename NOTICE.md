# NOTICE: Third-Party Attribution

This repository is released under the MIT License (see [`LICENSE`](LICENSE)), but parts of
it derive from or incorporate work by others, and this file records that provenance so that
the attribution conditions those works carry are satisfied. **Attribution below is required
when redistributing this repository or any work derived from it.**

---

## 1. NRIC Virtual Test Bed, prismatic HTGR assembly model

**Licensed under CC BY 4.0. Attribution required.**

| | |
|---|---|
| **Work** | Virtual Test Bed (VTB), `htgr/assembly`, the prismatic HTGR assembly model |
| **Creators** | Idaho National Laboratory, on behalf of the National Reactor Innovation Center (NRIC) and the U.S. Department of Energy NEAMS program |
| **Source repository** | <https://github.com/idaholab/virtual_test_bed/tree/main/htgr/assembly> |
| **Documentation** | <https://virtualtestbed.inl.gov/htgr/assembly/index.html> |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **License text** | <https://creativecommons.org/licenses/by/4.0/>, with the full text also included at [`licenses/CC-BY-4.0.txt`](licenses/CC-BY-4.0.txt) |
| **Modified?** | **Yes.** This repository contains a modified derivative of the VTB material. Changes are described below. |

### What was derived

| File in this repository | Relationship to the VTB material |
|---|---|
| `materials.py` | **Directly derived** from `htgr/assembly/materials.py`. The TRISO kernel/buffer/PyC/SiC material definitions, their reference densities (10820, 1050, 1900, 3203 kg/m³), the graphite matrix and reflector definitions, and the `c_Graphite` S(α,β) thermal scattering treatment follow the VTB model. |
| `config.py` | **Derived in part.** The prismatic-assembly parameter set (TRISO layer thicknesses, compact and coolant channel dimensions, lattice pitch and axial zoning scheme) follows the parameterisation of `htgr/assembly/common_input.py`. |
| `assembly.py` | **Approach adapted.** The hexagonal prismatic assembly construction (hex prism regions, axially zoned universes, and TRISO particles packed into an OpenMC repeated-structures lattice) follows the modelling approach of `htgr/assembly/assembly.py`. The implementation here is substantially rewritten and expanded. |
| `main_simulation.py` | **Approach adapted.** The overall OpenMC model-build and axial temperature-profile structure follows the VTB assembly driver. |

### Changes made relative to the VTB material

This repository is **not** the VTB model and should not be represented as such, and the
changes listed below are what separate the two:

- **Scope.** The VTB material models a *single* prismatic assembly. This
  repository models a *full core*, meaning a configurable hexagonal lattice of
  assemblies with per-position assembly types, full or 1/6-symmetric geometry,
  and radial reflector regions.
- **Fuel.** Enrichment raised from 15.5% to 19.75% HALEU; TRISO packing
  fraction changed to 0.33; U-234 treatment simplified.
- **Reactivity control.** Added multi-bank B4C control rods with continuous
  (non-zone-snapped) insertion depths, burnable poison rods, and a reserve
  shutdown system modelled as an explicitly packed B4C sphere lattice
  (`b4c_spheres.py`). None of this exists in the VTB material.
- **Reflector.** Added an optional BeO radial reflector with configurable inner
  radius, thickness and density.
- **Depletion.** Added spatially resolved depletion (radial × axial burnup
  zones), reduced-chain generation, graphite depletion, restart support, and a
  criticality-search-driven depletion mode that repositions control rods at every
  timestep. The VTB material performs no depletion.
- **Homogenisation.** Added a Reactivity-equivalent Physical Transform (RPT)
  double-heterogeneity treatment with an automated calibration mode.
- **Thermal-hydraulic coupling.** Replaced the VTB fixed cosine coolant
  temperature profile with a converged neutronics ↔ single-channel TH iteration
  (see section 2), plus isothermal and CSV-driven temperature profile sources.
- **Post-processing.** Added the entire `PostProcessingScripts/` suite
  (burnup, isotopics, BeO fluence, leakage spectra, reactivity coefficients,
  tally plotting, parametric aggregation), which has no VTB counterpart.

### Disclaimer carried from the VTB repository

> This information was prepared as an account of work sponsored by an agency of
> the U.S. Government. Neither the U.S. Government nor any agency thereof, nor
> any of their employees, makes any warranty, expressed or implied, or assumes
> any legal liability or responsibility for the accuracy, completeness, or
> usefulness, of any information, apparatus, product, or process disclosed, or
> represents that its use would not infringe privately owned rights. References
> herein to any specific commercial product, process, or service by trade name,
> trade mark, manufacturer, or otherwise, do not necessarily constitute or imply
> its endorsement, recommendation, or favoring by the U.S. Government or any
> agency thereof. The views and opinions of authors expressed herein do not
> necessarily state or reflect those of the U.S. Government or any agency
> thereof.

Neither Idaho National Laboratory, NRIC, nor the U.S. Department of Energy
endorses this repository or has reviewed it.

---

## 2. HTGR-SCAPC, the single-channel thermal-hydraulics and Brayton cycle solver

| | |
|---|---|
| **Work** | HTGR-SCAPC, `nc_htgr.py` |
| **Creator** | **Bryan Huynh** |
| **Source repository** | <https://github.com/bryanhhuynh/HTGR-SCAPC> |
| **Status** | Included in this repository with the author's permission |
| **Location here** | [`ThermalHydraulics/nc_htgr.py`](ThermalHydraulics/nc_htgr.py) |

`nc_htgr.py` was written by Bryan Huynh as the thermal-hydraulics and power-cycle
contribution to the CHUDR senior design project in the ENU 4192 course at the University of
Florida. It performs the steady-state single-channel solve, covering axial coolant
enthalpy rise, convective film drop, graphite conduction and compact/kernel
temperatures, then closes a recuperated Brayton cycle to report net electrical
output.

It is thus vendored here rather than referenced as an external dependency so that the coupled
neutronics/thermal-hydraulics workflow can be run from a single repository, though the
upstream project at the link above remains the canonical source.

**Modifications made in this repository** (originally made in the fork at
`c-finney/HTGR-SCAPC`, now integrated here):

- Corrected the axial heating profile for downward flow. The solver iterates
  from the physical bottom of the core upward, but with `flow_upward` false the
  inlet is at the physical top; the z lookup in `_get_qprime` is now mirrored so
  the inlet node reads the power at the top of the core and the outlet node the
  power at the bottom. Without this the profile is applied backwards.
- Recuperator effectiveness raised from 0.90 to 0.94.
- TRISO packing fraction raised from 0.30 to 0.33, matching the calibrated
  neutronics model.
- Module docstring and input-deck header added for provenance and usage.
- `neutronics_file` in the deck repointed from `BOLCriticalHeating.csv`, which
  was never part of the project, to the `neutronics.csv` that ships here, and a
  relative `neutronics_file` is now resolved against the deck's own directory.

The solver's power cycle was already a direct recuperated Brayton cycle in the
upstream source (`# Direct recuperated Brayton cycle`); the upstream repository
description calling it indirect is stale metadata, not a difference in code.

`ThermalHydraulics/neutronics.csv` is an example axial heating profile produced
by the neutronics side of this repository and consumed by `nc_htgr.py`; it is not
part of the upstream project.

---

## 3. OpenMC

This framework is built on **OpenMC**, an MIT-licensed Monte Carlo particle
transport code developed by the Massachusetts Institute of Technology and
contributors.

- Repository: <https://github.com/openmc-dev/openmc>
- Documentation: <https://docs.openmc.org/>

> P. K. Romano, N. E. Horelik, B. R. Herman, A. G. Nelson, B. Forget, and
> K. Smith, "OpenMC: A State-of-the-Art Monte Carlo Code for Research and
> Development," *Annals of Nuclear Energy*, **82**, 90–97 (2015).

OpenMC is a dependency rather than a derivative, and no OpenMC source is included here.

---

## 4. CHUDR design work

The CHUDR reactor design that this framework was built to analyse was produced by
a six-person senior design team at the University of Florida (ENU 4192):
Cade Finney, Evan Alder, Bryan Huynh, Colin Frazier, William St. Peter, and
Daniel Fernandez, under the guidance of Dr. DuWayne Schubring.

The **code in this repository** was written by Cade Finney, with the exception of
`ThermalHydraulics/nc_htgr.py` as noted in section 2, while the design parameters appearing
in `config.py` and `examples/` reflect the team's collective work.
