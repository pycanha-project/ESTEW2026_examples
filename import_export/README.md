# Import / export

Reads a geometry file, adds a meshed plate and a cylinder in code, and writes
the model in the other format: ESATAN-TMS to STEP-TAS, or STEP-TAS to
ESATAN-TMS. The plate is 1 m x 1 m at the origin, with the cylinder standing on
it. Their nodes are numbered from 900001 and 900101: change these numbers in
the script if the file already uses them. The exported file is then read back
into a new model and opened in the viewer.

The readers and writers report what they could not represent. The report goes
to the console and to a diagnostics file in `logs/`.

Settings:

- `INPUT_FILE`: geometry to read. `.stp` is read as STEP-TAS, any other
  extension (`.erg`, `.gmm`, `.etms`) as ESATAN-TMS.
- `OUTPUT_FILE`: file to write. `.stp` writes STEP-TAS, `.erg` writes
  ESATAN-TMS.

```bash
python import_export.py
```
