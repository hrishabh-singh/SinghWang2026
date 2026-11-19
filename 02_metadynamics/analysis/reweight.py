"""Free-energy surfaces and coordination-state populations from the multiple-walker metadynamics runs.

Usage
    python reweight.py --system Tb --run-dir <dir with COLVARS.0..3> --n-walkers 4 --out ../results

Method (Methods, "Reweighting and populations")
    1. The bias each walker felt is rebuilt on a grid. Every walker deposits one Gaussian per PACE (500 steps = 1 ps)
       at its current (CN, Ca RMSD) with the well-tempered height h = W exp[-V(s,t) / ((gamma-1) kT)], where V is the
       bias printed in COLVARS (meta.bias). This reconstruction reproduces the HILLS file exactly where it was kept.
    2. The runs were restarted several times. With WALKERS_MPI and WALKERS_DIR=./ only the first walker re-read the
       accumulated HILLS at each restart; the other walkers restarted from zero bias. Within each restart segment those
       walkers therefore share one consistent well-tempered bias built from zero. The data are split into these
       segments, and the first 20 ns after each restart are discarded. The first walker contributes only the segment
       before its first restart, when all walkers still shared the bias.
    3. Frames are reweighted with w = exp[beta (V_w(s,t) + U_walls(s) - c(t))], where c(t) is obtained self-consistently
       with the iterative trajectory reweighting scheme (ITRE; Giberti et al., J. Chem. Theory Comput. 16, 100, 2020),
       which starts from the time-independent estimator of Tiwary and Parrinello. Surfaces use Gaussian kernels of width
       0.15 (CN) and 0.025 nm (Ca RMSD).
    4. The population of state CN = n is the reweighted probability of n - 1/2 <= CN < n + 1/2 (Eq. 7). Every restart
       segment with at least 20 ns of data gives its own estimate; standard errors are the standard deviation of the
       segment estimates divided by the square root of their number.

Outputs (in --out)
    state_populations_<system>.csv   population and standard error of CN = 5..10
    reweighting_log_<system>.csv     frames, segments, ITRE iterations, effective sample size
    fes_<system>.npz                 reweighted surface F[rmsd, cn] (kJ/mol) and per-segment surfaces
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from plumed_io import colvars_files, load_walker, restart_times

# ---- simulation constants (plumed.dat and the .mdp file) ------------------------------------------------------------
T = 298.0                       # K
KB = 0.0083144626               # kJ/(mol K)
KT = KB * T
BETA = 1.0 / KT
GAMMA = 3.0                     # BIASFACTOR
HEIGHT = 1.0                    # HEIGHT (kJ/mol)
SIGMA_CN, SIGMA_RMSD = 0.3, 0.05
WALL_KAPPA = 300.0              # kJ/mol per unit CV^2 (EXP=2, EPS=1)
RMSD_UPPER_WALL, CN_LOWER_WALL = 0.6, 5.0

# ---- analysis settings ----------------------------------------------------------------------------------------------
FRAME_STRIDE_PS = 10            # frames used for reweighting (every 10 ps)
BIAS_GRID_EVERY_PS = 2_000      # bias grids stored every 2 ns
BURN_IN_PS = 20_000             # discarded after every restart (bias re-equilibration)
KERNEL_CN, KERNEL_RMSD = 0.15, 0.025
MIN_SEGMENT_FRAMES = 2_000      # 20 ns at 10 ps per frame
STATES = range(5, 11)

# Bias lattice (the Gaussian widths 0.3 / 0.05 are resolved by 6 / 6.25 cells) and analysis grid
CN_EDGES = np.linspace(2.0, 11.0, 181)
RMSD_EDGES = np.linspace(0.0, 1.1, 141)
CN_C = 0.5 * (CN_EDGES[1:] + CN_EDGES[:-1])
RMSD_C = 0.5 * (RMSD_EDGES[1:] + RMSD_EDGES[:-1])
D_CN, D_RMSD = CN_EDGES[1] - CN_EDGES[0], RMSD_EDGES[1] - RMSD_EDGES[0]
GAUSS_NORM = 2 * np.pi * (SIGMA_CN / D_CN) * (SIGMA_RMSD / D_RMSD)
_RR, _CC = np.meshgrid(RMSD_C, CN_C, indexing="ij")
U_WALLS = (WALL_KAPPA * np.clip(_RR - RMSD_UPPER_WALL, 0, None) ** 2
           + WALL_KAPPA * np.clip(CN_LOWER_WALL - _CC, 0, None) ** 2)


# ======================================================================================================================
# 1-2. Hills and per-walker bias history
# ======================================================================================================================
def reconstruct_hills(files):
    """All deposited Gaussians as rows (time, cn, rmsd, height in HILLS convention, walker), time-ordered."""
    parts = []
    for w, f in enumerate(files):
        a = load_walker(f)
        a = a[a[:, 0] > 0.5]                                  # the first hill is deposited at t = 1 ps
        h = HEIGHT * np.exp(-a[:, 5] / ((GAMMA - 1.0) * KT)) * GAMMA / (GAMMA - 1.0)
        parts.append(np.column_stack([a[:, 0], a[:, 1], a[:, 2], h, np.full(len(a), w)]))
    hills = np.concatenate(parts)
    return hills[np.lexsort((hills[:, 4], hills[:, 0]))]


def _gaussian_sum(hist):
    return gaussian_filter(hist, sigma=(SIGMA_RMSD / D_RMSD, SIGMA_CN / D_CN), mode="constant", truncate=5.0) * GAUSS_NORM


def bias_grids(hills, restarts):
    """Bias on the lattice every BIAS_GRID_EVERY_PS for the two bias histories:
       V_all   all hills since t = 0 (felt by the first walker, which re-read HILLS at each restart)
       V_reset hills since the most recent restart (felt by the walkers that restarted from zero bias)."""
    fac = (GAMMA - 1.0) / GAMMA
    times = np.arange(BIAS_GRID_EVERY_PS, hills[-1, 0] + BIAS_GRID_EVERY_PS, BIAS_GRID_EVERY_PS)
    h_all = np.zeros((RMSD_C.size, CN_C.size))
    h_seg = np.zeros_like(h_all)
    v_all, v_reset = [], []
    ends = np.searchsorted(hills[:, 0], times, side="right")
    lo = 0
    for k, t in enumerate(times):
        chunk = hills[lo:ends[k]]
        last_restart = restarts[restarts <= t].max() if np.any(restarts <= t) else -1
        h, _, _ = np.histogram2d(chunk[:, 2], chunk[:, 1], bins=[RMSD_EDGES, CN_EDGES], weights=chunk[:, 3])
        h_all += h
        if last_restart > t - BIAS_GRID_EVERY_PS:             # a restart happened inside this interval
            post = chunk[chunk[:, 0] > last_restart]
            h_seg[:] = 0.0
            h2, _, _ = np.histogram2d(post[:, 2], post[:, 1], bins=[RMSD_EDGES, CN_EDGES], weights=post[:, 3])
            h_seg += h2
        else:
            h_seg += h
        v_all.append(_gaussian_sum(h_all) * fac)
        v_reset.append(_gaussian_sum(h_seg) * fac)
        lo = ends[k]
    return times, np.array(v_all, np.float32), np.array(v_reset, np.float32)


# ======================================================================================================================
# 3. Iterative trajectory reweighting on the consistent segments
# ======================================================================================================================
def kde(cn, rmsd, w, s_cn=KERNEL_CN, s_rmsd=KERNEL_RMSD):
    hist, _, _ = np.histogram2d(rmsd, cn, bins=[RMSD_EDGES, CN_EDGES], weights=w)
    p = gaussian_filter(hist, sigma=(s_rmsd / D_RMSD, s_cn / D_CN), mode="constant", truncate=4.0)
    return p / (p.sum() * D_CN * D_RMSD)


def free_energy(p, floor=1e-7):
    with np.errstate(divide="ignore"):
        f = -KT * np.log(np.where(p > floor * p.max(), p, np.nan))
    return f - np.nanmin(f)


def _logsumexp(a, axis):
    m = np.nanmax(a, axis=axis, keepdims=True)
    return (m + np.log(np.nansum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def select_frames(files, restarts, grid_times):
    """Frames of the segment estimator: columns t, cn, rmsd, total bias felt, bias-grid index, walker, segment."""
    seg_starts = np.r_[0.0, restarts]
    frames = []
    for w, f in enumerate(files):
        a = load_walker(f, stride_ps=FRAME_STRIDE_PS)
        seg = np.clip(np.searchsorted(seg_starts, a[:, 0], side="left") - 1, 0, None)
        keep = (a[:, 0] - seg_starts[seg]) > BURN_IN_PS
        if w == 0:
            keep &= seg == 0                                  # first walker: only before its first restart
        a, seg = a[keep], seg[keep]
        k = np.clip(np.searchsorted(grid_times, a[:, 0], side="left"), 0, len(grid_times) - 1)
        frames.append(np.column_stack([a[:, 0], a[:, 1], a[:, 2], a[:, 3] + a[:, 4] + a[:, 5], k,
                                       np.full(len(a), w), seg]))
    return np.concatenate(frames)


def itre(frames, grid_times, v_reset, n_iter=30, tol=0.01):
    """Self-consistent c(t) and frame weights. Returns (F, weights, iterations, final change in kJ/mol)."""
    f = free_energy(kde(frames[:, 1], frames[:, 2], np.exp(BETA * (frames[:, 3] - frames[:, 3].max()))))
    v = v_reset.astype(np.float64) + U_WALLS[None]
    delta, it = np.inf, 0
    for it in range(1, n_iter + 1):
        fz = np.where(np.isfinite(f), f, np.inf)
        log_z0 = _logsumexp((-BETA * fz).ravel()[None, :], 1)[0]
        c = KT * (_logsumexp((-BETA * (fz[None] - v)).reshape(len(grid_times), -1), 1) - log_z0)
        logw = BETA * (frames[:, 3] - c[frames[:, 4].astype(int)])
        w = np.exp(logw - logw.max())
        f_new = free_energy(kde(frames[:, 1], frames[:, 2], w))
        region = np.isfinite(f) & np.isfinite(f_new) & (f_new < 20)
        delta = float(np.nanmax(np.abs((f_new - f)[region]))) if region.any() else np.inf
        f = f_new
        if delta < tol:
            break
    return f, w, it, delta


# ======================================================================================================================
# 4. Coordination-state populations
# ======================================================================================================================
def state_probabilities(cn, w):
    return {n: w[(cn >= n - 0.5) & (cn < n + 0.5)].sum() / w.sum() for n in STATES}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, help="label used in output names, e.g. Tb")
    ap.add_argument("--run-dir", required=True, type=Path, help="directory holding COLVARS.0 ... COLVARS.<n-1>")
    ap.add_argument("--n-walkers", required=True, type=int)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = colvars_files(args.run_dir, args.n_walkers)
    restarts = restart_times(files[0])
    hills = reconstruct_hills(files)
    grid_times, _, v_reset = bias_grids(hills, restarts)
    frames = select_frames(files, restarts, grid_times)
    f, w, iterations, delta = itre(frames, grid_times, v_reset)

    total = state_probabilities(frames[:, 1], w)
    segments = [s for s in np.unique(frames[:, 6]).astype(int) if (frames[:, 6] == s).sum() >= MIN_SEGMENT_FRAMES]
    per_seg = pd.DataFrame([state_probabilities(frames[frames[:, 6] == s, 1], w[frames[:, 6] == s]) for s in segments])
    pops = pd.DataFrame({"system": args.system, "CN": list(STATES),
                         "population": [total[n] for n in STATES],
                         "standard_error": [per_seg[n].std(ddof=1) / np.sqrt(len(per_seg)) for n in STATES],
                         "n_segments": len(per_seg)})
    pops.to_csv(args.out / f"state_populations_{args.system}.csv", index=False)

    seg_fes = {f"segment_{s}": free_energy(kde(frames[frames[:, 6] == s, 1], frames[frames[:, 6] == s, 2],
                                              w[frames[:, 6] == s])) for s in segments}
    np.savez_compressed(args.out / f"fes_{args.system}.npz", cn=CN_C, rmsd=RMSD_C, F=f, **seg_fes)
    ess = w.sum() ** 2 / (w ** 2).sum()
    pd.DataFrame([dict(system=args.system, walkers=args.n_walkers, restarts=len(restarts), frames=len(frames),
                       segments=len(segments), itre_iterations=iterations, final_change_kJ_mol=delta,
                       effective_sample_size=ess)]).to_csv(args.out / f"reweighting_log_{args.system}.csv", index=False)
    print(pops.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
