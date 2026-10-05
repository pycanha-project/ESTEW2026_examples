import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyvista as pv
from PIL import Image

import pycanha as pc
from pycanha import gmm
from pycanha import radiative as rad
from pycanha.tmm import DataModelAttribute, NodeType

PLATE_NODES = 50  # nodes along each side of the plate
SOLAR_RAYS_PER_FACE = 2_500  # per time step: about as long on the GPU as one step on the CPU
REF_RAYS_PER_FACE = 20_000
REF_CUTOFF = 5e-4  # drop the REF of two faces that each get <= 0.05 % of what the other emits
OPEN_VIEWER = True  # open the temperatures and solar loads in the viewer

TIME_STEP = 10.0  # s, a new sun position every step
OUTPUT_TIME = 60.0  # s between stored results (one GIF frame each)
ORBIT_PERIOD = 5400.0  # s, one turn of the sun
SUN_ELEVATION = 60.0  # deg
SOLAR_CONSTANT = 1361.0  # W/m^2
EPS = 0.5  # IR emissivity, every surface
ALPHA = 0.5  # solar absorptivity, every surface (the rest is reflected)
INITIAL_TEMPERATURE = 293.15  # K
SPACE_NODE = 1_999_999_999
SPACE_TEMPERATURE = 3.0  # K

PLATE_WIDTH = 2.0  # m, square plate in z = 0
PLATE_GAP = 0.5  # m, plate to the bottom of the honeycomb
HONEYCOMB_WIDTH = 1.0  # m
CELLS_ACROSS = 10  # hexagonal cells across the honeycomb
WALL_DEPTH = 1.4  # cell depth, in hexagon sides
RING_GAP = 0.3  # m, top of the honeycomb to the ring
RING_RADII = (0.125, 0.25)  # m, inner and outer

tm = pc.ThermalModel("honeycomb")
paint = gmm.OpticalMaterial("paint", emissivity_ir=EPS, absorptivity_solar=ALPHA)
# The plate is a 0.5 mm stainless steel sheet. The walls and the ring have a
# tiny capacity and no conduction: their temperature comes from radiation only.
steel = gmm.BulkMaterial("steel", density=8000.0, conductivity=16.0, specific_heat=500.0)
shell = gmm.BulkMaterial("shell", density=1.0, conductivity=1.0, specific_heat=1000.0)

# Plate: one node per face, numbered from 100001.
plate_mesh = gmm.ThermalMesh(np.linspace(0, 1, PLATE_NODES + 1), np.linspace(0, 1, PLATE_NODES + 1))
plate_mesh.side1_optical = plate_mesh.side2_optical = paint
plate_mesh.side1_material = plate_mesh.side2_material = steel
plate_mesh.side1_thick = plate_mesh.side2_thick = 0.25e-3  # both sides share the node: half each
plate_mesh.node1_start = plate_mesh.node2_start = 100_001
plate_mesh.node1_step = plate_mesh.node2_step = 1
h = PLATE_WIDTH / 2
tm.gmm.add(
    gmm.GeometryItem("plate", gmm.Rectangle((-h, -h, 0), (h, -h, 0), (-h, h, 0)), plate_mesh)
)

# Open honeycomb: the walls of hexagonal cells, no floor. A wall shared by two
# cells is built once. Each wall is one node, numbered from 1001.
side = HONEYCOMB_WIDTH / (CELLS_ACROSS * math.sqrt(3))  # hexagon side
depth = WALL_DEPTH * side
rows = round(HONEYCOMB_WIDTH / (1.5 * side))
angles = np.radians(np.arange(30, 390, 60))  # hexagon corners
walls = set()
for row in range(rows):
    for col in range(CELLS_ACROSS - row % 2):
        cx = math.sqrt(3) * side * (col + 0.5 + 0.5 * (row % 2)) - HONEYCOMB_WIDTH / 2
        cy = 1.5 * side * (row - (rows - 1) / 2)
        # Rounded, so the two cells sharing a wall give it the same end points.
        corners = [
            (round(cx + side * math.cos(a), 9), round(cy + side * math.sin(a), 9)) for a in angles
        ]
        walls.update(
            tuple(sorted(edge)) for edge in zip(corners, corners[1:] + corners[:1], strict=True)
        )

for node, ((x1, y1), (x2, y2)) in enumerate(sorted(walls), start=1001):
    wall_mesh = gmm.ThermalMesh()
    wall_mesh.side1_optical = wall_mesh.side2_optical = paint
    wall_mesh.side1_material = wall_mesh.side2_material = shell
    wall_mesh.side1_thick = wall_mesh.side2_thick = 0.05e-3
    wall_mesh.conductive_active_side = gmm.ActiveSide.NONE
    wall_mesh.node1_start = wall_mesh.node2_start = node
    wall = gmm.Rectangle((x1, y1, PLATE_GAP), (x2, y2, PLATE_GAP), (x1, y1, PLATE_GAP + depth))
    tm.gmm.add(gmm.GeometryItem(f"wall_{node}", wall, wall_mesh))

# Ring: 24 sectors around and 2 across, one node each, numbered from 1.
z = PLATE_GAP + depth + RING_GAP
ring_mesh = gmm.ThermalMesh(np.linspace(0, 1, 25), np.linspace(0, 1, 3))
ring_mesh.side1_optical = ring_mesh.side2_optical = paint
ring_mesh.side1_material = ring_mesh.side2_material = shell
ring_mesh.side1_thick = ring_mesh.side2_thick = 0.05e-3
ring_mesh.conductive_active_side = gmm.ActiveSide.NONE
ring_mesh.node1_start = ring_mesh.node2_start = 1
ring_mesh.node1_step = ring_mesh.node2_step = 1
ring = gmm.Disc((0, 0, z), (0, 0, z + 1), (1, 0, z), *RING_RADII, 0, 2 * math.pi)
tm.gmm.add(gmm.GeometryItem("ring", ring, ring_mesh))

# --------------------------------------------------------------------------- #
# Thermal model: capacities and plate conduction from the geometry, deep space
# --------------------------------------------------------------------------- #
tm.build_tmm_from_gmm(pc.conduction.TmmBuildOptions(initial_temperature=INITIAL_TEMPERATURE))
tm.tmm.add_node(SPACE_NODE)
tm.tmm.nodes.set_type(SPACE_NODE, NodeType.BOUNDARY)
tm.tmm.nodes.set_T(SPACE_NODE, SPACE_TEMPERATURE)

# --------------------------------------------------------------------------- #
# Radiation on the GPU
# --------------------------------------------------------------------------- #
# How the radiative calculation works (this API will probably change):
# - The GPU traces rays against triangles. create_mesh() triangulates every
#   face, and each triangle keeps the number of the face it belongs to.
# - A RadiativeScene uploads the triangles, and the optical properties and active
#   sides of every face, to the GPU. It also builds the structures the GPU uses to
#   find the first triangle a ray hits. It is built once, and used here for the
#   REFs and for the solar loads of every time step.
# - An accumulator holds counters on the GPU. scene.accumulate_exchange() traces
#   rays_per_face IR rays from every active face. Each ray starts with energy 1,
#   leaves on every face it hits the part that face absorbs, and goes on with the
#   reflected part. What leaves the scene is counted as going to space.
#   scene.accumulate_solar() does the same with the sunlight of one sun direction.
# - accumulator.result() copies the counters to the CPU. The REFs come as a
#   sparse matrix with one row and one column per face. For each two faces i and j
#   it stores A_i eps_i B_ij once, in the upper triangle. Three extra columns at
#   the end hold what each face sends to space, to inactive faces, and what is
#   lost. The solar loads come as W per face.
# - rad.aggregate_matrix() and rad.aggregate_flux() sum the face values per node.
tm.gmm.create_mesh()  # triangulation: every face covered by triangles
device = rad.Device.create()  # the GPU with ray tracing support
# mesh_parts(): the triangles. material_table(): optical properties and active sides.
scene = rad.RadiativeScene(device, tm.gmm.mesh_parts(), tm.gmm.material_table())
areas = scene.face_areas()  # m^2, per face
face_nodes = tm.gmm.mesh.node_numbers  # node of every face
nodes = rad.aggregate_nodes(face_nodes)  # the node numbers, sorted
print(f"GPU: {device.info.name}")
print(f"{len(nodes):,} nodes, {scene.num_faces():,} faces")

# REFs, condensed into radiative couplings between nodes and to deep space.
start = time.perf_counter()
# The face x face counters are kept on the GPU in blocks of 256 rows (Tiled), so a
# large model fits in GPU memory. Pairs of faces below the cut-off are dropped.
config = rad.AccumConfig(layout=rad.AccumLayout.Tiled, tile_rows=256, sparse_threshold=REF_CUTOFF)
accumulator = rad.ExchangeAccumulator(scene, rad.Band.IR, config)  # IR counters, on the GPU
settings = rad.TraceSettings(rays_per_face=REF_RAYS_PER_FACE, seed=0)
scene.accumulate_exchange(accumulator, settings)  # trace the rays, count where energy is absorbed
factors = accumulator.result().factors  # counters back to the CPU, A_i eps_i B_ij [m^2]

# Sum the face values per pair of nodes. Rows and columns follow the order of nodes.
node_matrix = rad.aggregate_matrix(factors, face_nodes).matrix.tocoo()  # (row, col, value) form
couplings = 0
for row, col, value in zip(node_matrix.row, node_matrix.col, node_matrix.data, strict=True):
    if row != col:  # what a node exchanges with itself carries no heat
        tm.tmm.add_radiative_coupling(nodes[row], nodes[col], value)
        couplings += 1

# The space column holds what every face sends to space. Sum it per node.
to_space = factors[:, scene.num_faces() + rad.space_column_offset].toarray()
space_couplings = {}
for face, node in enumerate(face_nodes):
    space_couplings[node] = space_couplings.get(node, 0.0) + to_space[face, 0]
for node, value in space_couplings.items():
    tm.tmm.add_radiative_coupling(node, SPACE_NODE, value)
print(f"REFs: {couplings:,} radiative couplings in {time.perf_counter() - start:.1f} s")

# Direction the sunlight travels, at every time step.
steps = round(ORBIT_PERIOD / TIME_STEP)
e = math.radians(SUN_ELEVATION)
suns = []
for step in range(steps + 1):
    azimuth = 2 * math.pi * step / steps
    suns.append(
        -np.array([math.cos(e) * math.cos(azimuth), math.cos(e) * math.sin(azimuth), math.sin(e)])
    )


def trace(step: int, tracing: threading.Event) -> np.ndarray:
    """Absorbed solar power [W] per node (same order as nodes) for the sun of a time step."""
    accumulator = rad.SolarAccumulator(scene)  # absorbed power per face, on the GPU
    state = rad.SolarState(suns[step], SOLAR_CONSTANT)
    settings = rad.TraceSettings(rays_per_face=SOLAR_RAYS_PER_FACE, seed=step)
    tracing.set()  # from here on the GPU works without Python, so the solver can resume
    scene.accumulate_solar(state, accumulator, settings)
    watts = accumulator.result().total  # W per face, back on the CPU
    return rad.aggregate_flux(watts / areas, face_nodes, areas)  # W per node


# The time of one sun position on the GPU, measured alone.
trace(0, threading.Event())  # the first trace also sets the GPU up
start = time.perf_counter()
for step in range(10):
    trace(step, threading.Event())
trace_seconds = (time.perf_counter() - start) / 10

# --------------------------------------------------------------------------- #
# Steady state for the first sun position, then one turn of the sun
# --------------------------------------------------------------------------- #
# Nodes with a tiny capacity oscillate when the transient starts away from balance.
for node, load in zip(nodes, trace(0, threading.Event()), strict=True):
    tm.tmm.nodes.set_qs(node, load)
steady = tm.solvers.sslu
steady.max_iters = 50
steady.initialize()
steady.solve()

# While the CPU solves a step, the GPU traces the solar loads of the next one.
gpu = ThreadPoolExecutor(max_workers=1)
pending = {}  # step -> its solar loads, being traced
waits = []  # time the solver spent waiting for the GPU [s]


def next_sun_position(context: pc.solvers.CallbackContext) -> None:
    """Called by the solver at every new time: set this step's loads, start tracing the next."""
    step = round(context.time / TIME_STEP)
    begin = time.perf_counter()
    loads = pending.pop(step).result() if step in pending else trace(step, threading.Event())
    waits.append(time.perf_counter() - begin)
    if step < steps:  # start the next trace first, so the GPU works while the loads are set
        tracing = threading.Event()
        pending[step + 1] = gpu.submit(trace, step + 1, tracing)
        tracing.wait()
    for node, load in zip(nodes, loads, strict=True):
        context.tmm.nodes.set_qs(node, load)


tm.callbacks.time_change = next_sun_position
solver = tm.solvers.tscnrlds
# Speed-up for this model, about 1.8 times faster per step. The solver factorises
# its matrix with PARDISO (Intel MKL), by default again at every iteration. With
# pardiso_iparm_3 = 61, PARDISO first solves iteratively with the factors of a
# previous iteration, down to a relative residual of 1e-6, and factorises again
# only if that fails. The radiation is linearised at every iteration, so the matrix
# changes, but little from one step to the next. It needs the ONE_LEVEL
# factorisation. Here one step of the turn logs that it did not converge, and the
# next steps recover. These solver settings will probably change in future versions.
solver.solver_type = pc.solvers.DirectSolverType.ONE_LEVEL
solver.pardiso_iparm_3 = 61
solver.output_config.add(DataModelAttribute.QS)  # store the solar loads with the temperatures
solver.set_simulation_time(0.0, steps * TIME_STEP, TIME_STEP, OUTPUT_TIME)
solver.initialize()
start = time.perf_counter()
solver.solve()
seconds = time.perf_counter() - start
tm.callbacks.active = False
gpu.shutdown()

waited = sum(waits)
gpu_seconds = len(waits) * trace_seconds
print(f"transient: {steps} steps in {seconds:.1f} s ({seconds / steps * 1e3:.0f} ms per step)")
print(f"CPU busy {(seconds - waited) / seconds:.0%} of the time, waited {waited:.1f} s for the GPU")
print(f"GPU busy {gpu_seconds / seconds:.0%} of the time, {trace_seconds * 1e3:.0f} ms per trace")

# --------------------------------------------------------------------------- #
# GIF of the temperatures, with an arrow along the sunlight
# --------------------------------------------------------------------------- #
# Stored results: one row per output time and one column per node.
times = solver.output_model.T.times
values = solver.output_model.T.values
poly = tm.gmm.to_polydata(both_sides=True)  # the triangles of both sides of every face
node_of_triangle = poly.cell_data["node_number"]
frame_temperatures = []  # degC of every triangle, one list per output time
for row in values:
    temperature_of = dict(zip(solver.output_model.node_numbers, row, strict=True))  # node -> K
    frame_temperatures.append([temperature_of[node] - 273.15 for node in node_of_triangle])

poly.cell_data["T"] = frame_temperatures[0]
plotter = pv.Plotter(off_screen=True)
actor = plotter.add_mesh(
    poly,
    scalars="T",
    cmap="turbo",
    clim=(min(map(min, frame_temperatures)), max(map(max, frame_temperatures))),
    backface_culling=True,  # each side shows its own face
    scalar_bar_args={"title": "Temperature [degC]"},
)
frames = []
for t, triangle_temperatures in zip(times, frame_temperatures, strict=True):
    sun = suns[round(t / TIME_STEP)]
    actor.mapper.dataset.cell_data["T"] = triangle_temperatures
    arrow = pv.Arrow(start=(0, 0, z) - 0.5 * sun, direction=sun, scale=0.5)
    plotter.add_mesh(arrow, color="yellow", name="sun")
    frames.append(Image.fromarray(plotter.screenshot(return_img=True)))
plotter.close()
frames[0].save("temperatures.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
print("wrote temperatures.gif")

if OPEN_VIEWER:
    tm.explore()
