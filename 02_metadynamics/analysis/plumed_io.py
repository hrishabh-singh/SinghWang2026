"""Reading PLUMED output files (COLVARS) from a multiple-walker metadynamics run.

The runs were restarted from checkpoints several times. On each restart PLUMED appends a new header to COLVARS and
writes again from the checkpoint time, so the files contain overlapping time ranges. ``drop_restart_overlaps`` keeps
only the rows written by the later segment, which is what the trajectory actually continued from.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Columns written by the PRINT ARG=* line of plumed.dat that the analysis needs.
COLVAR_COLUMNS = ["time", "cn", "ca_rmsd", "uwall.bias", "lwall.bias", "meta.bias"]


def read_plumed(path, usecols):
    """Read a PLUMED text file into a DataFrame, using the field names of its first '#! FIELDS' header."""
    with open(path) as fh:
        fields = fh.readline().split()[2:]
    return pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=fields, usecols=usecols,
                       engine="c", dtype=np.float64)


def drop_restart_overlaps(t):
    """Boolean mask that keeps, for every restart (time going backwards), only the rows of the later segment."""
    t = np.asarray(t)
    keep = np.ones(t.size, bool)
    for b in (np.where(np.diff(t) < 0)[0] + 1)[::-1]:     # first row of each new segment
        keep[:b][t[:b] >= t[b]] = False
    return keep


def restart_times(colvars_path):
    """Times (ps) at which a job restart wrote a new header into COLVARS (the first header is not a restart)."""
    times, grab = [], False
    with open(colvars_path) as fh:
        for line in fh:
            if line.startswith("#! FIELDS"):
                grab = True
            elif grab and not line.startswith("#"):
                times.append(float(line.split()[0]))
                grab = False
    return np.unique(np.array(times[1:]))


def load_walker(colvars_path, stride_ps=None):
    """COLVARS of one walker as an array with columns COLVAR_COLUMNS (restart overlaps removed).

    stride_ps: keep only frames whose time is a multiple of this value (None keeps every frame)."""
    df = read_plumed(colvars_path, COLVAR_COLUMNS)
    df = df[drop_restart_overlaps(df["time"].values)]
    if stride_ps:
        df = df[(np.round(df["time"].values).astype(np.int64) % stride_ps) == 0]
    return df[COLVAR_COLUMNS].to_numpy(np.float64)


def colvars_files(run_dir, n_walkers):
    """Paths COLVARS.0 ... COLVARS.<n-1> (one file per walker, as written by WALKERS_MPI)."""
    files = [Path(run_dir) / f"COLVARS.{w}" for w in range(n_walkers)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"missing COLVARS files: {missing}")
    return files
