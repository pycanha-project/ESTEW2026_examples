# pycanha examples, ESTEW 2026

Short scripts that show the main features of pycanha. Each folder holds one
script and a README that describes it and lists its settings.

| Folder | Script | Content | GPU |
|---|---|---|---|
| [primitives](primitives/) | `primitives.py` | The eight primitives with their thermal meshes, in the viewer. | |
| [boolean](boolean/) | `boolean.py` | A plate and a sphere cut by solid cutters, in the viewer. | |
| [conduction](conduction/) | `conduction.py` | Heated plate: Thermal Mathematical Model (TMM) built from the geometry, steady state and transient. | |
| [triangulation](triangulation/) | `triangulation.py` | Sphere-to-plate view factor by Monte Carlo ray tracing, compared with the analytical value. | yes |
| [viewer](viewer/) | `viewer.py` | An ESATAN-TMS geometry file, and optionally its results, in the viewer. | |
| [import_export](import_export/) | `import_export.py` | Geometry read from ESATAN-TMS or STEP-TAS, extended in code and written to the other format. | |
| [solar_fluxes](solar_fluxes/) | `solar_fluxes.py` | Solar loads on a plate under an open honeycomb, over one turn of the sun. | yes |
| [GPU_CPU_solving](GPU_CPU_solving/) | `gpu_cpu_solving.py` | Transient of the same scene: the GPU traces the next solar loads while the CPU solves the current step. | yes |

## Requirements

- Python 3.13 or 3.14.
- pycanha: `pip install pycanha`. The scripts were tested with pycanha 0.23.0.
- For the examples marked GPU: a GPU with Vulkan ray tracing support.

## Running

The settings are constants at the top of each script. Edit them and run the
script from its folder:

```bash
cd conduction
python conduction.py
```

The outputs (GIF and PNG files, exported geometry) are written to the folder
the script runs from. Scripts that open the viewer end when its window is
closed.
