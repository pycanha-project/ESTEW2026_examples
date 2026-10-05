import pycanha as pc

ERG_FILE = "path/to/model.erg"  # ESATAN-TMS geometry: .erg, .gmm or .etms
TMD_FILE = "path/to/results.TMD"  # ESATAN-TMS results

tm = pc.ThermalModel("model")
tm.gmm.io.read_esatan_erg(ERG_FILE)

# tm.tmm.read_tmd_transient(TMD_FILE, "results")  # uncomment to show the results too

tm.explore()
