# Viewer

Reads an ESATAN-TMS geometry file and opens it in the interactive viewer.
Uncomment the `read_tmd_transient` line to also read the results of an
ESATAN-TMS `.TMD` file. The viewer then shows them on the geometry, through
the node numbers.

Settings:

- `ERG_FILE`: path of the geometry file (`.erg`, `.gmm` or `.etms`).
- `TMD_FILE`: path of the results file, used by the commented line.

```bash
python viewer.py
```
