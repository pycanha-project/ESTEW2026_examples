# Triangulation and view factor

A sphere 1 m in diameter with its centre 1.5 m above a 1 m x 1 m plate, all
black. The ray tracer works on a triangulation of the geometry, and the
deviation tolerance sets how close the triangles stay to the true sphere. The
script traces rays from the outer side of the sphere and the upper side of the
plate on the GPU and computes the sphere-to-plate view factor. It prints the
number of triangles, the number of rays, the time of the calculation and the
error against the analytical view factor. The error has two parts: the
triangles are not the sphere, and the number of rays is finite.

The viewer opens after the calculation. The "Mesh" button of its toolbar draws
the triangles.

Settings:

- `DEVIATION_TOLERANCE` [m]: largest gap between the triangles and the true
  surface. A smaller value gives more triangles.
- `RAYS_PER_FACE`: rays traced from each face (the sphere and the plate).
- `SEED`: seed of the random numbers. Change it to repeat the calculation with
  other rays.
- `DIAMETER`, `HEIGHT`, `SIDE` [m]: the geometry. The analytical value follows
  them.

```bash
python triangulation.py
```
