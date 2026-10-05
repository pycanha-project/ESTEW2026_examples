import math
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from pycanha.tmm import NodeType

import pycanha as pc
from pycanha import gmm

N_NODES = 100_000  # plate nodes, on a grid of square nodes
OPEN_VIEWER = True  # open the results in the viewer (a large model can take long to load)

# Plate
LENGTH = 1.0  # m, along x
WIDTH = 0.5  # m, along y
THICKNESS = 2.0e-3  # m
DENSITY = 2700.0  # kg/m^3, aluminium 6061
CONDUCTIVITY = 167.0  # W/(m K)
SPECIFIC_HEAT = 896.0  # J/(kg K)
EDGE_TEMPERATURE = 293.15  # K, held on the four edges, and the initial temperature

# Heater trace
HEATER_POWER = 100.0  # W, while on
HEATER_WIDTH = 6.0e-3  # m

# Transient
SIMULATION_TIME = 92.0  # s
TIME_STEP = 0.1  # s
OUTPUT_TIME = 1.0  # s between stored results (one GIF frame each)

# --------------------------------------------------------------------------- #
# Geometry: a rectangle split into nx x ny nodes
# --------------------------------------------------------------------------- #
start = time.perf_counter()
ny = round(math.sqrt(N_NODES * WIDTH / LENGTH))
nx = round(N_NODES / ny)
n = nx * ny
dx = LENGTH / nx  # node size along x
dy = WIDTH / ny  # node size along y

tm = pc.ThermalModel("plate")
mesh = gmm.ThermalMesh(np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1))
mesh.side1_material = gmm.BulkMaterial(
    "plate", density=DENSITY, conductivity=CONDUCTIVITY, specific_heat=SPECIFIC_HEAT
)
mesh.side1_thick = THICKNESS
mesh.conductive_active_side = gmm.ActiveSide.SIDE1  # one layer of nodes
mesh.node1_start = 1
mesh.node1_step = 1
tm.gmm.add(gmm.GeometryItem("plate", gmm.Rectangle((0, 0, 0), (LENGTH, 0, 0), (0, WIDTH, 0)), mesh))

# --------------------------------------------------------------------------- #
# Thermal model: capacities and conductive couplings from the geometry
# --------------------------------------------------------------------------- #
tm.build_tmm_from_gmm(pc.conduction.TmmBuildOptions(initial_temperature=EDGE_TEMPERATURE))

# Node numbers run along x first: the node in row j and column i is j * nx + i + 1.
edges = []
for j in range(ny):
    for i in range(nx):
        if j in (0, ny - 1) or i in (0, nx - 1):
            edges.append(j * nx + i + 1)

# The edge nodes become boundary nodes. One call for all of them is the same as
#     for node in edges:
#         tm.tmm.nodes.set_type(node, NodeType.BOUNDARY)
# but each set_type reorders the nodes, which takes seconds on a large model.
tm.tmm.nodes.set_types(edges, NodeType.BOUNDARY)


def heater_centre(x: float) -> float:
    """y [m] of the heater trace centreline at x [m]: a distorted sine."""
    phase = 6 * math.pi * x / LENGTH + 0.8 * math.sin(2 * math.pi * x / LENGTH)
    return WIDTH * (0.5 + 0.34 * math.sin(phase))


# Heater nodes: closer to the trace centreline than half its width (at least
# 0.75 nodes, so a coarse grid still gets a connected trace), measured across the trace.
half_width = max(HEATER_WIDTH / 2, 0.75 * dx)
heater = []
for j in range(ny):
    for i in range(nx):
        x, y = (i + 0.5) * dx, (j + 0.5) * dy  # node centre
        slope = (heater_centre(x + dx) - heater_centre(x - dx)) / (2 * dx)
        distance = abs(y - heater_centre(x)) / math.sqrt(1 + slope**2)
        if 0.08 * LENGTH < x < 0.92 * LENGTH and distance < half_width:
            heater.append(j * nx + i + 1)
print(f"{nx} x {ny} = {n:,} nodes, {len(heater):,} on the heater")
print(f"model built in {time.perf_counter() - start:.2f} s")

# --------------------------------------------------------------------------- #
# Steady state, heater on
# --------------------------------------------------------------------------- #
for node in heater:
    tm.tmm.nodes.set_qi(node, HEATER_POWER / len(heater))

start = time.perf_counter()
steady = tm.solvers.sslu
steady.initialize()
steady.solve()
print(f"steady state solved in {time.perf_counter() - start:.2f} s")

temperature = np.zeros((ny, nx))
for j in range(ny):
    for i in range(nx):
        temperature[j, i] = tm.tmm.nodes.get_T(j * nx + i + 1) - 273.15

fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
image = ax.imshow(temperature, origin="lower", extent=(0, LENGTH, 0, WIDTH), cmap="inferno")
fig.colorbar(image, ax=ax, label="Temperature [°C]")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"Steady state, {n:,} nodes")
fig.savefig("plate_steady.png", dpi=150)

# --------------------------------------------------------------------------- #
# Transient: back to the edge temperature everywhere, heater switched on and off
# --------------------------------------------------------------------------- #
for node in range(1, n + 1):
    tm.tmm.nodes.set_T(node, EDGE_TEMPERATURE)
for node in heater:
    tm.tmm.nodes.set_qi(node, 0.0)


def switch_heater(context: pc.solvers.CallbackContext) -> None:
    """Called by the solver at every new time: every 30 s, off for 2 s, then on for 10 s."""
    on = np.interp(context.time % 30.0, [0, 2, 2.5, 12, 12.5, 30], [0, 0, 1, 1, 0, 0])
    for node in heater:
        context.tmm.nodes.set_qi(node, on * HEATER_POWER / len(heater))


tm.callbacks.time_change = switch_heater

start = time.perf_counter()
solver = tm.solvers.tscnrlds
solver.abstol_temp = math.inf  # conduction only is linear: one iteration per step is exact
# Speed-up for this model, about 3 times faster per step. The solver factorises
# its matrix with PARDISO (Intel MKL), by default again at every step. With
# pardiso_iparm_3 = 61, PARDISO first solves iteratively with the factors of a
# previous step, down to a relative residual of 1e-6, and factorises again only
# if that fails. Here the matrix is the same at every step (only the heater load
# changes), so the factors of the first step serve the whole run. It needs the
# ONE_LEVEL factorisation. These solver settings will probably change in future versions.
solver.solver_type = pc.solvers.DirectSolverType.ONE_LEVEL
solver.pardiso_iparm_3 = 61
solver.set_simulation_time(0.0, SIMULATION_TIME, TIME_STEP, OUTPUT_TIME)
solver.initialize()
solver.solve()
tm.callbacks.active = False
seconds = time.perf_counter() - start
steps = round(SIMULATION_TIME / TIME_STEP)
print(f"transient solved in {seconds:.2f} s, {seconds / steps * 1e3:.1f} ms per step")

# Stored results: one row per output time and one column per node.
times = solver.output_model.T.times
values = solver.output_model.T.values
temperature = np.zeros((len(times), ny, nx))
for column, node in enumerate(solver.output_model.node_numbers):
    j, i = divmod(node - 1, nx)  # row and column of the node in the plate
    temperature[:, j, i] = values[:, column] - 273.15

fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
image = ax.imshow(
    temperature[0],
    origin="lower",
    extent=(0, LENGTH, 0, WIDTH),
    cmap="inferno",
    vmin=temperature.min(),
    vmax=temperature.max(),
)
fig.colorbar(image, ax=ax, label="Temperature [°C]")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
gif = PillowWriter(fps=10)
with gif.saving(fig, "plate_transient.gif", dpi=80):
    for t, field in zip(times, temperature, strict=True):
        image.set_data(field)
        ax.set_title(f"t = {t:5.1f} s, {n:,} nodes")
        gif.grab_frame()
print("wrote plate_steady.png and plate_transient.gif")
plt.show()

if OPEN_VIEWER:
    print("opening the viewer: a large model can take a long time to load")
    tm.explore()
