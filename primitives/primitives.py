import numpy as np

from pycanha import gmm

SPACING = 3.0  # m, distance between primitives along x
SWEEP = 1.5 * np.pi  # rad, angular extent of the revolved primitives (a quarter left open)

# The flat primitives stand in the x-z plane. The revolved ones turn about the
# z axis: p1 is the base centre, p2 a point on the axis, p3 where the angles start.
primitives = [
    (gmm.Triangle((-1, 0, 0), (1, 0, 0), (-1, 0, 2)), 4, 4),
    (gmm.Rectangle((-1, 0, 0), (1, 0, 0), (-1, 0, 2)), 6, 6),
    (gmm.Quadrilateral((-1, 0, 0), (1, 0, 0.3), (0.6, 0, 2), (-0.8, 0, 1.6)), 6, 6),
    (gmm.Disc((0, 0, 1), (0, 0, 2), (1, 0, 1), 0.4, 1.0, 0, SWEEP), 12, 4),
    (gmm.Cylinder((0, 0, 0), (0, 0, 2), (1, 0, 0), 0.8, 0, SWEEP), 12, 6),
    (gmm.Cone((0, 0, 0), (0, 0, 2), (1, 0, 0), 1.0, 0.3, 0, SWEEP), 12, 6),
    (gmm.Sphere((0, 0, 1), (0, 0, 2), (1, 0, 1), 1.0, -0.6, 0.8, 0, SWEEP), 12, 8),
    (gmm.Paraboloid((0, 0, 0), (0, 0, 2), (1, 0, 0), 1.0, 0, SWEEP), 12, 6),
]

model = gmm.GeometryModel("primitives")
for i, (primitive, n1, n2) in enumerate(primitives):
    # Thermal mesh: n1 x n2 equal divisions of the primitive's two directions.
    mesh = gmm.ThermalMesh(np.linspace(0, 1, n1 + 1), np.linspace(0, 1, n2 + 1))
    position = gmm.CoordinateTransformation.from_translation((SPACING * i, 0, 0))
    model.add(gmm.GeometryItem(type(primitive).__name__, primitive, mesh, position))

model.explore()
