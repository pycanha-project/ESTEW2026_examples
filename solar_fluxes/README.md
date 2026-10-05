# Solar fluxes

A 2 m x 2 m plate under an open honeycomb (the walls of hexagonal cells, no
floor) and a ring above it. The sun stays at 60° of elevation and makes one
turn around the scene. For each sun position the GPU traces the absorbed solar
flux of every face, including the sunlight reflected between the surfaces
(every surface has a solar absorptivity of 0.5). The script prints the number
of rays per sun position and the GPU time, and saves `solar_fluxes.gif` with
an arrow along the sunlight.

With `OPEN_VIEWER = True` the solar loads of every node are stored as a
transient result, one sun position per time, and the model opens in the
viewer.

Settings:

- `PLATE_NODES`: nodes along each side of the plate (one node per face).
- `SOLAR_RAYS_PER_FACE`: rays traced from each face for each sun position.
- `SUN_POSITIONS`: sun positions over one turn, one GIF frame each.
- `OPEN_VIEWER`: open the solar loads in the viewer.
- `SUN_ELEVATION` [deg], `SOLAR_CONSTANT` [W/m^2], `ORBIT_PERIOD` [s]: the sun.
  The period only sets the time axis in the viewer.
- `EPS`, `ALPHA`: IR emissivity and solar absorptivity of every surface.
- `PLATE_WIDTH`, `PLATE_GAP`, `HONEYCOMB_WIDTH`, `CELLS_ACROSS`, `WALL_DEPTH`,
  `RING_GAP`, `RING_RADII`: the geometry.

```bash
python solar_fluxes.py
```
