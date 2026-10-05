# Conduction

A 1 m x 0.5 m aluminium plate, 2 mm thick, with the nodes along its four
edges held at 20 °C as boundary nodes. A heater trace 6 mm wide meanders across
the plate. pycanha builds the Thermal Mathematical Model (TMM) from the
geometry: the node capacities and the conductive couplings. The script solves
the steady state with the heater on (SSLU steady-state solver) and saves
`plate_steady.png`. It then sets every node back to the edge temperature and
integrates a transient with the TSCNRLDS solver, the heater switched by a
callback: every 30 s, off for 2 s and on for 10 s. The temperature field is
saved as `plate_transient.gif`.

The transient uses PARDISO's one-level factorisation with
`pardiso_iparm_3 = 61`: the matrix is the same at every step, so the factors of
the first step are reused instead of factorising again.

With `OPEN_VIEWER = True` the model opens in the viewer after the matplotlib
figures are closed. The results are chosen in its results panel. Loading a
large model in the viewer can take a long time.

Settings:

- `N_NODES`: number of plate nodes, on a grid of square nodes.
- `OPEN_VIEWER`: open the results in the viewer.
- `LENGTH`, `WIDTH` [m], `THICKNESS` [m]: plate size.
- `DENSITY` [kg/m^3], `CONDUCTIVITY` [W/(m K)], `SPECIFIC_HEAT` [J/(kg K)]: plate material.
- `EDGE_TEMPERATURE` [K]: temperature of the edges, and initial temperature of the transient.
- `HEATER_POWER` [W], `HEATER_WIDTH` [m]: heater trace.
- `SIMULATION_TIME`, `TIME_STEP`, `OUTPUT_TIME` [s]: transient. Each stored result is one GIF frame.

```bash
python conduction.py
```
