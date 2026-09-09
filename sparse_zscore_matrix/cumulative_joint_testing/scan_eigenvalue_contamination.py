import os

def _require_positive_int(name, value):
    if value is None:
        raise SystemExit(f"Set {name} in the configuration block above.")
    if not isinstance(value, int) or isinstance(value, bool):
        raise SystemExit(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if value <= 0:
        raise SystemExit(f"{name} must be a positive integer, got {value}")

N_WORKERS = None   # concurrent worker processes; User to define

_require_positive_int("N_WORKERS", N_WORKERS)

os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "POLARS_MAX_THREADS": str(N_WORKERS)})

import sys
import time
import numpy as np
import pandas as pd
import polars as pl
from multiprocessing import Pool

base_out     = "/path/to.output/directory"
score_path   = "/path/to/score_matrix.dat"
z_path       = "/path/to/Z_matrix.npy"
save_path    = "/path/to/df_allSNPs_allphenos.parquet"
ldsc_path    = "/path/to/LDSC_intercept_matrix.csv"
corrected    = f"{base_out}/cumulative_joint_test_real_corrected.parquet"
null_parquet = f"{base_out}/cumulative_joint_test_null.parquet"
out_path     = f"{base_out}/eigenvalue_contamination_scan.parquet"

RIDGE_MARGIN = 1e-4    # floor for the data-driven ridge; see COR loading below
SMALL_EV     = 1e-2
CONTAM_FRAC  = 0.5
log_path     = f"{base_out}/scan_eigenvalue_contamination_output.txt"


class _Tee:
    """Writes to multiple streams at once, so console output is also saved to a file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


pheno_cols = pl.read_parquet(save_path, n_rows=0).columns[1:]
Np         = len(pheno_cols)
N_snps     = np.load(z_path, mmap_mode='r').shape[0]

_Sigma    = None
_RIDGE    = None
_score_mm = None
_Z_mm     = None


def _init_worker(sigma, ridge, score_path_w, z_path_w, n_snps_w, np_w):
    global _Sigma, _RIDGE, _score_mm, _Z_mm
    _Sigma    = sigma
    _RIDGE    = ridge
    _score_mm = np.memmap(score_path_w, dtype='float32', mode='r', shape=(n_snps_w, np_w))
    _Z_mm     = np.load(z_path_w, mmap_mode='r')


def _process_one(args):
    row_idx, k = args
    scores_full = np.asarray(_score_mm[row_idx, :], dtype=np.float64)
    z_full      = np.asarray(_Z_mm[row_idx, :], dtype=np.float64)

    # Restrict to observed phenotypes BEFORE ranking -- see docstring point 1.
    observed    = ~np.isnan(scores_full) & ~np.isnan(z_full)
    obs_idx     = np.where(observed)[0]
    order_local = np.argsort(scores_full[obs_idx])[::-1][:k]
    order       = obs_idx[order_local]
    z_k         = z_full[order]

    Sig_k = _Sigma[np.ix_(order, order)] + _RIDGE * np.eye(k)
    evals, evecs = np.linalg.eigh(Sig_k)
    proj    = evecs.T @ z_k
    contrib = proj**2 / evals
    S       = float(contrib.sum())

    small_mask = evals < SMALL_EV
    frac_small = float(contrib[small_mask].sum() / S)
    top1_frac  = float(contrib.max() / S)
    n_small    = int(small_mask.sum())
    min_eval   = float(evals.min())
    sum_z2     = float(z_k @ z_k)

    return (row_idx, k, S, S / k, sum_z2, frac_small, top1_frac, n_small, min_eval)


if __name__ == '__main__':
    _stdout, _stderr = sys.stdout, sys.stderr
    log_file = open(log_path, "w")
    sys.stdout = _Tee(_stdout, log_file)
    sys.stderr = _Tee(_stderr, log_file)

    t0 = time.time()
    print("Loading LDSC matrix...", flush=True)
    COR_df         = pd.read_csv(ldsc_path, sep='\t', index_col=0)
    COR_df.index   = COR_df.index.str.removeprefix('z_')
    COR_df.columns = COR_df.columns.str.removeprefix('z_')
    Sigma          = COR_df.loc[pheno_cols, pheno_cols].values.astype(np.float64)
    assert Sigma.shape == (Np, Np), f"unexpected Sigma shape {Sigma.shape}"

    lambda_min = np.linalg.eigvalsh(Sigma).min()
    RIDGE      = max(1e-4, RIDGE_MARGIN - lambda_min)
    print(f"  COR min eigenvalue = {lambda_min:.6f} -> using RIDGE = {RIDGE:.6f}", flush=True)

    print("Loading null parquet to recover the empirical significance threshold...", flush=True)
    null_df  = pl.read_parquet(null_parquet)
    null_max = float(null_df["peak_log10p"].max())
    print(f"  null max peak_log10p = {null_max:.3f}")

    print("Loading corrected parquet and filtering to hits...", flush=True)
    df = pl.read_parquet(corrected).with_row_index("row_idx")
    hits = df.filter(pl.col("peak_log10p").is_not_nan() & (pl.col("peak_log10p") > null_max))
    print(f"  {hits.height:,} hits", flush=True)

    tasks = list(zip(hits["row_idx"].to_list(), hits["peak_k"].to_list()))

    print(f"Scanning {len(tasks):,} hits across {N_WORKERS} workers...", flush=True)
    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(Sigma, RIDGE, score_path, z_path, N_snps, Np)) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=64)):
            results.append(res)
            if (i + 1) % 2_000 == 0:
                print(f"  {i+1:,} / {len(tasks):,}  ({time.time()-t0:.0f}s)", flush=True)

    cols = ["row_idx", "peak_k", "S", "inflation", "sum_z2",
            "frac_small", "top1_frac", "n_small", "min_eval"]
    res_df = pl.DataFrame(results, schema=cols, orient="row")

    id_map = df.select(["row_idx", "SNP_ID", "peak_log10p", "p_emp"])
    res_df = res_df.join(id_map, on="row_idx", how="left")
    res_df.write_parquet(out_path)
    print(f"\nSaved -> {out_path}", flush=True)

    contaminated = res_df.filter(pl.col("frac_small") > CONTAM_FRAC)
    n_contam = contaminated.height
    n_total  = res_df.height
    print(f"\n=== SUMMARY ===")
    print(f"Contaminated (frac_small > {CONTAM_FRAC}): {n_contam:,} / {n_total:,} "
          f"({100*n_contam/n_total:.2f}%)")

    fs = res_df["frac_small"].to_numpy()
    for q in [50, 75, 90, 95, 99, 100]:
        print(f"  frac_small p{q}: {np.percentile(fs, q):.3f}")

    k = res_df["peak_k"].to_numpy()
    print("\nContamination rate by peak_k band:")
    for lo, hi in [(0, 10), (10, 30), (30, 50), (50, 65), (65, 75), (75, Np + 1)]:
        band = (k >= lo) & (k < hi)
        n_band = band.sum()
        if n_band == 0:
            continue
        n_band_contam = (band & (fs > CONTAM_FRAC)).sum()
        print(f"  k in [{lo:>3},{hi:>3}): {n_band_contam:>6,} / {n_band:>6,} "
              f"contaminated ({100*n_band_contam/n_band:.1f}%)")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} minutes")
    print(f"\nLog saved -> {log_path}")
    sys.stdout, sys.stderr = _stdout, _stderr
    log_file.close()