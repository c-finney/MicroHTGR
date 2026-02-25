from leakage_spectrum import run_leakage_analysis
import openmc
import json, glob, os

run_dir = "/path/to/your/depletion/run"

# Find the last statepoint (EOL)
statepoints = sorted(glob.glob(os.path.join(run_dir, "openmc_simulation_n*.h5")))
print(f"Found {len(statepoints)} statepoints")
print(f"EOL file: {statepoints[-1]}")