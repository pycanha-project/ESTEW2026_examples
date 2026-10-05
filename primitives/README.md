# Primitives

The eight primitives of pycanha (triangle, rectangle, quadrilateral, disc,
cylinder, cone, sphere and paraboloid) placed in a row along x, each divided
by its thermal mesh into faces. The flat primitives stand in the x-z plane and
the revolved ones turn about the z axis, leaving a quarter open so that both
sides show. The script then opens the interactive viewer on the geometry.

Settings:

- `SPACING`: distance between the primitives along x [m].
- `SWEEP`: angular extent of the revolved primitives [rad].
- The two numbers after each primitive in the `primitives` list: thermal mesh
  divisions in its first and second direction.

```bash
python primitives.py
```
