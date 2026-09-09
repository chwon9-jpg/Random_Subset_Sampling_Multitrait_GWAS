##########################################################################################
#                           User Configuration required
# These are None on purpose: set them for the machine this runs on rather than inheriting
# the values from the machine it last ran on.
#
#   N_WORKERS    : worker processes (loky backend).
#   BLAS_THREADS : threads per worker for the triangular solve. Total threads in use is
#                  roughly N_WORKERS * BLAS_THREADS — keep that at or below the core count.
#   SNP_BATCH    : SNPs per batch. Peak RAM is dominated by stat_batch (Ns * SNP_BATCH * 4
#                  bytes) plus a transient boolean validity mask of the same shape (1 byte
#                  each); roughly N_WORKERS * Ns * SNP_BATCH * 5 bytes; Z_batch and its
#                  derived arrays are comparatively tiny since Np is small.
###########################################################################################

# Configuration used to generate the toy example expected outputs. Adjust for your machine.
N_WORKERS, BLAS_THREADS = 8, 1
SNP_BATCH = 5_000

out_dir = "/path/to/output/directory"

# Z-score matrix: memory-mapped .npy array, shape (N_snps, Np).
Z_PATH = "/path/to/Z_matrix.npy"

stat_path = "/path/to/stat_matrix.dat"
cor_path  = "/path/to/LDSC_intercept_matrix.csv"

# Parquet with an 'ID' column giving SNP identifiers, in the same row order as
# the z-score matrix. If None, SNP ids in the output default to plain row indices.
save_path = "/path/to/df_toy_allSNPs_allphenos.parquet"

if None in (N_WORKERS, BLAS_THREADS, SNP_BATCH):
    raise SystemExit("Set N_WORKERS, BLAS_THREADS and SNP_BATCH in the configuration block above.")
if Z_PATH is None:
    raise SystemExit("Set Z_PATH in the configuration block above.")

import os
for env in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]: os.environ[env] = str(BLAS_THREADS)

import time, math, numpy as np, polars as pl, pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.special import chdtrc
from joblib import Parallel, delayed

os.makedirs(out_dir, exist_ok=True)

col1_path, col2_path, col2n_path, col3_path, col3n_path = (
    f"{out_dir}/col1.dat", f"{out_dir}/col2.dat", f"{out_dir}/col2_n.dat",
    f"{out_dir}/col3.dat", f"{out_dir}/col3_n.dat",
)

# Shapes taken from the inputs. stat_matrix is a raw .dat with no header, so its subset
# count is recovered from the file size; the check catches a stale or truncated file.
N_snps, Np = np.load(Z_PATH, mmap_mode='r').shape
_nbytes    = os.path.getsize(stat_path)
if _nbytes % (N_snps * 4):
    raise SystemExit(f"{stat_path}: {_nbytes:,} B is not a whole number of rows for N_snps={N_snps:,} float32.")
Ns = _nbytes // (N_snps * 4)

pheno_cols = pl.read_parquet(save_path, n_rows=0).columns[1:]
if len(pheno_cols) != Np:
    raise SystemExit(f"{len(pheno_cols)} phenotype columns vs {Np} columns in the z-score matrix.")


def process_snp_range(snp_start, snp_end, COR, L_full):
    Z    = np.load(Z_PATH, mmap_mode='r')
    stat = np.memmap(stat_path, dtype='float32', mode='r',  shape=(Ns, N_snps))
    col1  = np.memmap(col1_path,  dtype='float64', mode='r+', shape=(N_snps,))
    col2  = np.memmap(col2_path,  dtype='float64', mode='r+', shape=(N_snps,))
    col2n = np.memmap(col2n_path, dtype='int32',   mode='r+', shape=(N_snps,))
    col3  = np.memmap(col3_path,  dtype='float64', mode='r+', shape=(N_snps,))
    col3n = np.memmap(col3n_path, dtype='int32',   mode='r+', shape=(N_snps,))

    n_batches, t_start = math.ceil((snp_end - snp_start) / SNP_BATCH), time.time()

    for batch_i, b_start in enumerate(range(snp_start, snp_end, SNP_BATCH), 1):
        b_end = min(b_start + SNP_BATCH, snp_end)

        Z_batch  = np.array(Z[b_start:b_end, :], dtype=np.float32)
        observed = ~np.isnan(Z_batch)
        n_obs    = observed.sum(axis=1).astype(np.int32)

        with np.errstate(invalid='ignore'):
            col1[b_start:b_end] = np.nanmin(chdtrc(1, Z_batch.astype(np.float64) ** 2), axis=1)

        # col2: group the SNPs in this batch by their exact observed-phenotype pattern --
        # one Cholesky and one batched solve per unique pattern, not per SNP.
        packed = np.packbits(observed, axis=1)
        _, group_id, group_counts = np.unique(packed, axis=0, return_inverse=True, return_counts=True)
        group_id = group_id.ravel()

        col2_batch = np.full(b_end - b_start, np.nan, dtype=np.float64)
        for g in range(len(group_counts)):
            rows = np.where(group_id == g)[0]
            obs_idx = np.where(observed[rows[0]])[0]
            m = len(obs_idx)
            if m == 0:
                continue  # no phenotype observed at all for these SNPs; col2 stays NaN
            L_sub = L_full if m == Np else cholesky(COR[np.ix_(obs_idx, obs_idx)], lower=True, check_finite=False)
            Z_sub = Z_batch[np.ix_(rows, obs_idx)].T.astype(np.float64)
            Y_sub = solve_triangular(L_sub, Z_sub, lower=True, check_finite=False)
            col2_batch[rows] = chdtrc(m, (Y_sub ** 2).sum(axis=0))

        col2[b_start:b_end]  = col2_batch
        col2n[b_start:b_end] = n_obs

        # col3: nanmax instead of max, and a per-SNP Bonferroni count instead of a flat Ns.
        stat_batch      = np.array(stat[:, b_start:b_end], dtype=np.float32)
        n_valid_subsets = (~np.isnan(stat_batch)).sum(axis=0).astype(np.int32)
        with np.errstate(invalid='ignore'):
            max_stat     = np.nanmax(stat_batch, axis=0)
            p_min_subset = chdtrc(1, max_stat.astype(np.float64))
            col3[b_start:b_end] = np.minimum(p_min_subset * n_valid_subsets, 1.0)
        col3n[b_start:b_end] = n_valid_subsets

        print(f"[worker {snp_start:,}-{snp_end:,}] batch {batch_i}/{n_batches} | SNPs {b_start:,}-{b_end:,} | {time.time() - t_start:.1f}s", flush=True)

    for c in [col1, col2, col2n, col3, col3n]: c.flush()


if __name__ == '__main__':
    print("Loading COR matrix and computing Cholesky decomposition...")
    COR_df         = pd.read_csv(cor_path, sep='\t', index_col=0)
    COR_df.index   = COR_df.index.str.removeprefix('z_')
    COR_df.columns = COR_df.columns.str.removeprefix('z_')
    COR            = COR_df.loc[pheno_cols, pheno_cols].values.astype(np.float64)

    # Ridge for PD-ness -- same fix as Fullscale_sampling_loop_dynamic_sparsity_check.py.
    # By Cauchy interlacing, fixing the full matrix's min eigenvalue also fixes every
    # principal submatrix col2 can construct from it (see module docstring).
    RIDGE_MARGIN = 1e-4
    lambda_min = np.linalg.eigvalsh(COR).min()
    ridge_eps  = max(0.0, RIDGE_MARGIN - lambda_min)
    if ridge_eps > 0:
        print(f"COR min eigenvalue = {lambda_min:.6f} -> adding ridge eps = {ridge_eps:.6f} to diagonal")
        COR += ridge_eps * np.eye(Np)
    L_full = cholesky(COR, lower=True, check_finite=False)

    for p in [col1_path, col2_path, col3_path]: np.memmap(p, dtype='float64', mode='w+', shape=(N_snps,))
    for p in [col2n_path, col3n_path]:          np.memmap(p, dtype='int32',   mode='w+', shape=(N_snps,))

    chunk  = math.ceil(N_snps / N_WORKERS)
    ranges = [(i * chunk, min((i + 1) * chunk, N_snps)) for i in range(N_WORKERS)]
    print(f"Launching {N_WORKERS} workers | SNP_BATCH={SNP_BATCH:,} | BLAS={BLAS_THREADS}")

    Parallel(n_jobs=N_WORKERS, backend='loky', verbose=5)(
        delayed(process_snp_range)(s, e, COR, L_full) for s, e in ranges
    )

    print("Combining outputs and saving...")
    c1  = np.memmap(col1_path,  dtype='float64', mode='r', shape=(N_snps,))
    c2  = np.memmap(col2_path,  dtype='float64', mode='r', shape=(N_snps,))
    c2n = np.memmap(col2n_path, dtype='int32',   mode='r', shape=(N_snps,))
    c3  = np.memmap(col3_path,  dtype='float64', mode='r', shape=(N_snps,))
    c3n = np.memmap(col3n_path, dtype='int32',   mode='r', shape=(N_snps,))

    if save_path is not None:
        snp_ids = pl.scan_parquet(save_path).select("ID").collect()["ID"].to_list()
    else:
        snp_ids = list(range(N_snps))

    df = pd.DataFrame({
        'snp_id': snp_ids,
        'min_univariate_pval': c1,
        'joint_pval': c2,
        'n_pheno_observed': c2n,
        'min_subset_pval_bonf': c3,
        'n_subsets_used': c3n,
    })
    df.to_csv(f"{out_dir}/pvalue_matrix.csv", index=False)
    np.save(f"{out_dir}/pvalue_matrix.npy", np.column_stack([c1, c2, c2n, c3, c3n]))

    print(f"Saved outputs. Preview:\n{df.head(10).to_string(index=False)}")