import math
import time

import numpy as np
import pyvista as pv
from PIL import Image

import pycanha as pc
from pycanha import gmm
from pycanha import radiative as rad
from pycanha.tmm import DataModel

PLATE_NODES = 100  # nodes along each side of the plate
SOLAR_RAYS_PER_FACE = 1_000
SUN_POSITIONS = 72  # over one turn of the sun, one GIF frame each
OPEN_VIEWER = True  # open the solar loads in the viewer

SUN_ELEVATION = 60.0  # deg
SOLAR_CONSTANT = 1361.0  # W/m^2
ORBIT_PERIOD = 5400.0  # s, one turn of the sun (the time axis in the viewer)
EPS = 0.5  # IR emissivity, every surface
ALPHA = 0.5  # solar absorptivity, every surface (the rest is reflected)

PLATE_WIDTH = 2.0  # m, square plate in z = 0
PLATE_GAP = 0.5  # m, plate to the bottom of the honeycomb
HONEYCOMB_WIDTH = 1.0  # m
CELLS_ACROSS = 10  # hexagonal cells across the honeycomb
WALL_DEPTH = 1.4  # cell depth, in hexagon sides
RING_GAP = 0.3  # m, top of the honeycomb to the ring
RING_RADII = (0.125, 0.25)  # m, inner and outer

tm = pc.ThermalModel("honeycomb")
paint = gmm.OpticalMaterial("paint", emissivity_ir=EPS, absorptivity_solar=ALPHA)

# Plate: one node per face, numbered from 100001.
plate_mesh = gmm.ThermalMesh(np.linspace(0, 1, PLATE_NODES + 1), np.linspace(0, 1, PLATE_NODES + 1))
plate_mesh.side1_optical = plate_mesh.side2_optical = paint
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
    wall_mesh.node1_start = wall_mesh.node2_start = node
    wall = gmm.Rectangle((x1, y1, PLATE_GAP), (x2, y2, PLATE_GAP), (x1, y1, PLATE_GAP + depth))
    tm.gmm.add(gmm.GeometryItem(f"wall_{node}", wall, wall_mesh))

# Ring: 24 sectors around and 2 across, one node each, numbered from 1.
z = PLATE_GAP + depth + RING_GAP
ring_mesh = gmm.ThermalMesh(np.linspace(0, 1, 25), np.linspace(0, 1, 3))
ring_mesh.side1_optical = ring_mesh.side2_optical = paint
ring_mesh.node1_start = ring_mesh.node2_start = 1
ring_mesh.node1_step = ring_mesh.node2_step = 1
ring = gmm.Disc((0, 0, z), (0, 0, z + 1), (1, 0, z), *RING_RADII, 0, 2 * math.pi)
tm.gmm.add(gmm.GeometryItem("ring", ring, ring_mesh))

# --------------------------------------------------------------------------- #
# Solar loads on the GPU, multiple reflections included
# --------------------------------------------------------------------------- #
# How the radiative calculation works (this API will probably change):
# - The GPU traces rays against triangles. create_mesh() triangulates every
#   face, and each triangle keeps the number of the face it belongs to.
# - A RadiativeScene uploads the triangles, and the optical properties and active
#   sides of every face, to the GPU. It also builds the structures the GPU uses to
#   find the first triangle a ray hits. It is built once and reused.
# - A SolarAccumulator holds the solar power absorbed by every face, on the GPU.
#   scene.accumulate_solar() traces rays_per_face rays from random points of
#   every face, for one sun direction. A ray that sees the sun deposits the
#   absorbed part on its face, and the reflected part is traced on and absorbed
#   by the faces it reaches. A new sun direction needs a new accumulator.
# - accumulator.result() copies the absorbed power to the CPU, in W per face:
#   .direct (sunlight only) and .total (reflections included).
tm.gmm.create_mesh()  # triangulation: every face covered by triangles
device = rad.Device.create()  # the GPU with ray tracing support
# mesh_parts(): the triangles. material_table(): optical properties and active sides.
scene = rad.RadiativeScene(device, tm.gmm.mesh_parts(), tm.gmm.material_table())
areas = scene.face_areas()  # m^2, per face

# Direction the sunlight travels, for each sun position.
e = math.radians(SUN_ELEVATION)
suns = []
for i in range(SUN_POSITIONS):
    azimuth = 2 * math.pi * i / SUN_POSITIONS
    suns.append(
        -np.array([math.cos(e) * math.cos(azimuth), math.cos(e) * math.sin(azimuth), math.sin(e)])
    )

fluxes = []  # absorbed flux of every face [W/m^2], one array per sun position
seconds = []
for i, sun in enumerate(suns):
    start = time.perf_counter()
    accumulator = rad.SolarAccumulator(scene)  # absorbed power per face, on the GPU
    settings = rad.TraceSettings(rays_per_face=SOLAR_RAYS_PER_FACE, seed=i)
    scene.accumulate_solar(rad.SolarState(sun, SOLAR_CONSTANT), accumulator, settings)
    watts = accumulator.result().total  # W per face, back on the CPU
    seconds.append(time.perf_counter() - start)
    fluxes.append(watts / areas)

rays = len(areas) * SOLAR_RAYS_PER_FACE
print(f"GPU: {device.info.name}")
print(f"{len(areas):,} faces, {rays / 1e6:.0f} million rays per sun position")
print(f"{np.median(seconds) * 1e3:.1f} ms per sun position on the GPU")

# --------------------------------------------------------------------------- #
# GIF of the absorbed flux, with an arrow along the sunlight
# --------------------------------------------------------------------------- #
poly = tm.gmm.to_polydata(both_sides=True)  # the triangles of both sides of every face
face_of_triangle = poly.cell_data["face_id"]
poly.cell_data["flux"] = [fluxes[0][face] for face in face_of_triangle]
plotter = pv.Plotter(off_screen=True)
actor = plotter.add_mesh(
    poly,
    scalars="flux",
    cmap="inferno",
    clim=(0, max(flux.max() for flux in fluxes)),
    backface_culling=True,  # each side shows its own face
    scalar_bar_args={"title": "Absorbed solar flux [W/m^2]"},
)
frames = []
for flux, sun in zip(fluxes, suns, strict=True):
    actor.mapper.dataset.cell_data["flux"] = [flux[face] for face in face_of_triangle]
    arrow = pv.Arrow(start=(0, 0, z) - 0.5 * sun, direction=sun, scale=0.5)
    plotter.add_mesh(arrow, color="yellow", name="sun")
    frames.append(Image.fromarray(plotter.screenshot(return_img=True)))
plotter.close()
frames[0].save("solar_fluxes.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
print("wrote solar_fluxes.gif")

# --------------------------------------------------------------------------- #
# Viewer: the solar loads per node, stored as a transient result
# --------------------------------------------------------------------------- #
if OPEN_VIEWER:
    face_nodes = tm.gmm.mesh.node_numbers  # node of every face
    nodes = rad.aggregate_nodes(face_nodes)  # the node numbers, sorted
    results = DataModel(nodes)
    results.QS.resize(SUN_POSITIONS, len(nodes))
    for i, flux in enumerate(fluxes):
        node_loads = rad.aggregate_flux(flux, face_nodes, areas)  # W per node, same order as nodes
        results.QS.set_row(i, i * ORBIT_PERIOD / SUN_POSITIONS, node_loads)
    tm.tmm.thermal_data.models.add_model("solar loads", results)
    tm.explore()
