import numpy as np

from pycanha import gmm

TAU = 2 * np.pi

# A 2 m x 1 m plate in z = 0 and a sphere 1 m across, 0.5 m above it.
plate = gmm.GeometryItem(
    "plate",
    gmm.Rectangle((-1, -0.5, 0), (1, -0.5, 0), (-1, 0.5, 0)),
    gmm.ThermalMesh(np.linspace(0, 1, 11), np.linspace(0, 1, 6)),
)
sphere = gmm.GeometryItem(
    "sphere",
    gmm.Sphere((0, 0, 1), (0, 0, 2), (1, 0, 1), 0.5, -0.5, 0.5, 0, TAU),
    gmm.ThermalMesh(np.linspace(0, 1, 13), np.linspace(0, 1, 9)),
)

# The cutters. A cutter is a closed solid: cylinders and cones cut with their end caps.
tunnel = gmm.GeometryItem(  # cylinder along y through the sphere centre
    "tunnel", gmm.Cylinder((0, -0.8, 1), (0, 0.8, 1), (1, -0.8, 1), 0.2, 0, TAU), gmm.ThermalMesh()
)
funnel = gmm.GeometryItem(  # cone from the sphere centre, opening upwards
    "funnel", gmm.Cone((0, 0, 1), (0, 0, 1.8), (1, 0, 1), 0.0, 0.45, 0, TAU), gmm.ThermalMesh()
)
apex = np.array([-0.17, 0, -0.26])
axis = np.array([-np.sin(np.radians(50)), 0, np.cos(np.radians(50))])
cone = gmm.GeometryItem(  # cone tilted 50 degrees: an elliptical hole in the plate
    "cone",
    gmm.Cone(apex, apex + 0.91 * axis, (-0.17, 1, -0.26), 0.0, 0.33, 0, TAU),
    gmm.ThermalMesh(),
)
ball = gmm.GeometryItem(  # sphere centred on the plate: a round hole
    "ball",
    gmm.Sphere((0.6, 0, 0), (0.6, 0, 1), (1, 0, 0), 0.28, -0.28, 0.28, 0, TAU),
    gmm.ThermalMesh(),
)

# target - cutter removes the volume of the cutter from the target.
model = gmm.GeometryModel("boolean")
model.add((sphere - tunnel - funnel) + (plate - cone - ball))

model.explore()
