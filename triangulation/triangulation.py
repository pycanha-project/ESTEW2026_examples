import math
import time

import pycanha as pc
from pycanha import gmm
from pycanha import radiative as rad

DEVIATION_TOLERANCE = 1.0e-3  # m, largest gap between the triangles and the true surfaces
RAYS_PER_FACE = 1_000_000
SEED = 0

DIAMETER = 1.0  # m, sphere
HEIGHT = 1.5  # m, from the plate to the sphere centre
SIDE = 1.0  # m, square plate

# Analytical sphere-to-plate view factor: the plate is four rectangles with
# one corner below the sphere centre, each seeing atan(1 / sqrt(D1^2 + D2^2 + D1^2 D2^2)) / (4 pi).
d = HEIGHT / (SIDE / 2)
F_ANALYTICAL = math.atan(1 / math.sqrt(2 * d**2 + d**4)) / math.pi

tm = pc.ThermalModel("sphere_over_plate")
black = gmm.OpticalMaterial("black", emissivity_ir=1.0, absorptivity_solar=1.0)

# Only the outer side of the sphere (side 1, face 0) and the upper side of the
# plate (side 1, face 2) take part in the radiative exchange.
sphere_mesh = gmm.ThermalMesh()
sphere_mesh.radiative_active_side = gmm.ActiveSide.SIDE1
sphere_mesh.side1_optical = sphere_mesh.side2_optical = black
sphere_mesh.node1_start = 1
sphere = gmm.Sphere(
    (0, 0, HEIGHT),  # centre
    (0, 0, HEIGHT + 1),  # a point on the axis
    (1, 0, HEIGHT),  # where the angles start
    radius=DIAMETER / 2,
    base_truncation=-DIAMETER / 2,  # no truncation: the full sphere
    apex_truncation=DIAMETER / 2,
    start_angle=0,
    end_angle=2 * math.pi,
)
tm.gmm.add(gmm.GeometryItem("sphere", sphere, sphere_mesh))

plate_mesh = gmm.ThermalMesh()
plate_mesh.radiative_active_side = gmm.ActiveSide.SIDE1
plate_mesh.side1_optical = plate_mesh.side2_optical = black
plate_mesh.node1_start = 2
plate = gmm.Rectangle((-SIDE / 2, -SIDE / 2, 0), (SIDE / 2, -SIDE / 2, 0), (-SIDE / 2, SIDE / 2, 0))
tm.gmm.add(gmm.GeometryItem("plate", plate, plate_mesh))

tm.gmm.default_mesh_options = gmm.MeshOptions(DEVIATION_TOLERANCE)
tm.gmm.create_mesh()  # triangulation: every face covered by triangles

# --------------------------------------------------------------------------- #
# View factor by Monte Carlo ray tracing on the GPU
# --------------------------------------------------------------------------- #
# How the radiative calculation works (this API will probably change):
# - The GPU traces rays against triangles. create_mesh() above triangulates
#   every face, and each triangle keeps the number of the face it belongs to.
# - A RadiativeScene uploads the triangles, and the optical properties and active
#   sides of every face, to the GPU. It also builds the structures the GPU uses to
#   find the first triangle a ray hits. It is built once and reused.
# - An accumulator is a set of counters that stays on the GPU. Each
#   scene.accumulate_vf() call traces rays_per_face rays from every active face
#   and adds the face each ray hits first to the counters. Another call, with
#   another seed, adds more rays to the same counters.
# - accumulator.result() copies the counters to the CPU and turns them into view
#   factors: a sparse matrix with one row and one column per face. For each two
#   faces i and j it stores A_i F_ij (= A_j F_ji) once, in the upper triangle,
#   combining the rays sent both ways.
device = rad.Device.create()  # the GPU with ray tracing support
# mesh_parts(): the triangles. material_table(): optical properties and active sides.
scene = rad.RadiativeScene(device, tm.gmm.mesh_parts(), tm.gmm.material_table())

start = time.perf_counter()
accumulator = rad.VfAccumulator(scene)  # view factor counters, on the GPU
settings = rad.TraceSettings(rays_per_face=RAYS_PER_FACE, seed=SEED)
scene.accumulate_vf(accumulator, settings)  # trace the rays, count the faces they hit
result = accumulator.result()  # counters back to the CPU, as view factors
seconds = time.perf_counter() - start

# Faces are numbered as in the geometry: 0 is the outer side of the sphere, 2
# the upper side of the plate. Divide A_i F_ij by the area of the sphere triangles.
view_factor = result.vf[0, 2] / scene.face_areas()[0]
error = view_factor / F_ANALYTICAL - 1

print(f"GPU:         {device.info.name}")
print(f"triangles:   {tm.gmm.mesh.nt():,}")
print(f"rays:        {RAYS_PER_FACE:,} per face, {result.stats.total_rays:,} in total")
print(f"time:        {seconds:.3f} s")
print(f"view factor: {view_factor:.6f} (analytical {F_ANALYTICAL:.6f}, error {100 * error:+.3f} %)")

tm.explore()
