from pathlib import Path

import numpy as np

import pycanha as pc
from pycanha import gmm

INPUT_FILE = Path("path/to/model.erg")  # .erg, .gmm or .etms (ESATAN-TMS), or .stp (STEP-TAS)
OUTPUT_FILE = Path("exported.stp")  # .stp or .erg

tm = pc.ThermalModel("model")
if INPUT_FILE.suffix.lower() == ".stp":
    tm.gmm.io.read_steptas(INPUT_FILE)
else:
    tm.gmm.io.read_esatan_erg(INPUT_FILE)

# A 1 m x 1 m plate at the origin with a cylinder standing on it. Their node
# numbers start at 900001 and 900101: change them if the file already uses those.
paint = gmm.OpticalMaterial("white_paint", emissivity_ir=0.85, absorptivity_solar=0.2)
aluminium = gmm.BulkMaterial("aluminium", density=2700.0, conductivity=167.0, specific_heat=896.0)

plate_mesh = gmm.ThermalMesh(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
plate_mesh.side1_optical = plate_mesh.side2_optical = paint
plate_mesh.side1_material = aluminium
plate_mesh.side1_thick = 2.0e-3
plate_mesh.node1_start = plate_mesh.node2_start = 900_001
plate_mesh.node1_step = plate_mesh.node2_step = 1
plate = gmm.Rectangle((0, 0, 0), (1, 0, 0), (0, 1, 0))  # corner, x end, y end
tm.gmm.add(gmm.GeometryItem("added_plate", plate, plate_mesh))

cylinder_mesh = gmm.ThermalMesh(np.linspace(0, 1, 9), np.linspace(0, 1, 4))
cylinder_mesh.side1_optical = cylinder_mesh.side2_optical = paint
cylinder_mesh.node1_start = cylinder_mesh.node2_start = 900_101
cylinder_mesh.node1_step = cylinder_mesh.node2_step = 1
# Base centre, top centre, a point that sets where the angles start; radius 0.25 m.
cylinder = gmm.Cylinder((0.5, 0.5, 0), (0.5, 0.5, 1), (1, 0.5, 0), 0.25, 0, 2 * np.pi)
tm.gmm.add(gmm.GeometryItem("added_cylinder", cylinder, cylinder_mesh))

if OUTPUT_FILE.suffix.lower() == ".stp":
    tm.gmm.io.write_steptas(OUTPUT_FILE)
else:
    tm.gmm.io.write_esatan_erg(OUTPUT_FILE)
print(f"wrote {OUTPUT_FILE}")

# Read the exported file back into a new model and show it.
exported = pc.ThermalModel("exported")
if OUTPUT_FILE.suffix.lower() == ".stp":
    exported.gmm.io.read_steptas(OUTPUT_FILE)
else:
    exported.gmm.io.read_esatan_erg(OUTPUT_FILE)
exported.explore()
