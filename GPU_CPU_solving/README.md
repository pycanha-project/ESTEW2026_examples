# GPU and CPU solving together

The scene of `solar_fluxes` as a thermal model. The plate is a 0.5 mm
stainless steel sheet that conducts on its grid. Every honeycomb wall and ring
sector is one node with a tiny capacity and no conduction. The radiative
couplings come from Monte Carlo Radiative Exchange Factors (REFs) traced on
the GPU, with a cut-off, and every surface radiates to deep space. The
transient covers one turn of the sun with a new sun position at every time
step, starting from the steady state of the first one.

While the CPU solves a time step, the GPU traces the solar loads of the next
one. A callback hands the solver the loads of the current step and starts the
next trace in a second thread. After the transient the script prints the
fraction of the time the CPU solver and the GPU were busy, and the time the
solver waited for the GPU. It saves `temperatures.gif` with an arrow along the
sunlight. With `OPEN_VIEWER = True` the temperatures and the solar loads open
in the viewer.

Settings:

- `PLATE_NODES`: nodes along each side of the plate.
- `SOLAR_RAYS_PER_FACE`: solar rays per face and time step. More rays keep the
  GPU busy for longer: about as long as one solver step balances the two.
- `REF_RAYS_PER_FACE`, `REF_CUTOFF`: REFs. The REF of two faces is dropped when
  each face receives at most `REF_CUTOFF` of what the other emits.
- `OPEN_VIEWER`: open the results in the viewer.
- `TIME_STEP`, `OUTPUT_TIME`, `ORBIT_PERIOD` [s]: the transient. Each stored
  result is one GIF frame.
- `SUN_ELEVATION` [deg], `SOLAR_CONSTANT` [W/m^2], `EPS`, `ALPHA`,
  `INITIAL_TEMPERATURE` [K], `SPACE_TEMPERATURE` [K]: environment and surfaces.
- `PLATE_WIDTH`, `PLATE_GAP`, `HONEYCOMB_WIDTH`, `CELLS_ACROSS`, `WALL_DEPTH`,
  `RING_GAP`, `RING_RADII`: the geometry.

```bash
python gpu_cpu_solving.py
```
