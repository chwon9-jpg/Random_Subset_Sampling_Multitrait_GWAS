#########################################################################################
#                          User Configuration required
# These are None on purpose: set them for the machine this runs on rather than inheriting
# the values from the machine it last ran on.
#
#   n_workers : concurrent worker threads.
#   n_chunks  : how the Ns subsets are divided into units of work. NOT a free tuning knob —
#               one seed is spawned per chunk, so changing n_chunks changes which subsets
#               get drawn. Hold it fixed to reproduce a run; vary n_workers freely.
#   SNP_BATCH : SNPs held per worker at a time. Peak RAM is roughly
#               n_workers * k_max * SNP_BATCH * 4 bytes on top of the z-score matrix.
#               Lower this first if memory is tight.
#########################################################################################

def _require_positive_int(name, value):
    if value is None:
        raise SystemExit(f"Set {name} in the configuration block above.")
    if not isinstance(value, int) or isinstance(value, bool):
        raise SystemExit(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if value <= 0:
        raise SystemExit(f"{name} must be a positive integer, got {value}")

n_workers, n_chunks = None, None
SNP_BATCH  = None

base_dir  = "/path/to/directory"
save_path = "/path/to/df_allSNPs_allphenos.parquet"
cor_path  = "/path/to/LDSC_intercept_matrix.csv"

_require_positive_int("n_workers", n_workers)
_require_positive_int("n_chunks", n_chunks)
_require_positive_int("SNP_BATCH", SNP_BATCH)

import os, math
# BLAS pinned to 1 on purpose: each of the n_workers threads runs its own triangular
# solve, so multi-threaded BLAS here would oversubscribe the cores.
os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "POLARS_MAX_THREADS": str(n_workers)})

import numpy as np, pandas as pd, polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.linalg import cholesky, solve_triangular
from scipy.special import chdtrc, chdtri
pl.Config.set_tbl_cols(-1)

# File Paths
z_path, stat_path, sel_path = f"{base_dir}/Z_matrix.npy", f"{base_dir}/stat_matrix.dat", f"{base_dir}/sel_matrix.npy"

# Load Data
pheno_cols = pl.read_parquet(save_path, n_rows=0).columns[1:]
COR_df     = pd.read_csv(cor_path, index_col=0)
COR        = COR_df.loc[pheno_cols, pheno_cols].values.astype(np.float32)

# Load Z, transpose & free memory immediately
Z_raw      = np.load(z_path)
Ns, N_snps, Np = 10_000, Z_raw.shape[0], len(pheno_cols)
Z_T        = np.ascontiguousarray(Z_raw.T)
del Z_raw

chunk_size = math.ceil(Ns / n_chunks)
seeds      = np.random.SeedSequence(7).spawn(n_chunks)

def run_chunk(c_start, c_end, seed):
    rng       = np.random.default_rng(seed)
    c_size    = c_end - c_start
    stat_mat  = np.memmap(stat_path, dtype='float32', mode='r+', shape=(Ns, N_snps))
    sel_chunk = np.zeros((c_size, Np), dtype=np.uint8)
    stat_all, p = np.empty(N_snps, dtype=np.float32), np.empty(N_snps, dtype=np.float64)

    for i in range(c_size):
        k = int(rng.integers(5, 91))
        isSel = np.sort(rng.choice(Np, k, replace=False))
        L_sub = cholesky(COR[np.ix_(isSel, isSel)], lower=True, check_finite=False)
        stat_all[:] = 0.0  
        
        for b in range(0, N_snps, SNP_BATCH):
            b_end   = min(b + SNP_BATCH, N_snps)
            Y_batch = solve_triangular(L_sub, Z_T[isSel, b:b_end], lower=True, check_finite=False, overwrite_b=True)
            np.einsum('ij,ij->j', Y_batch, Y_batch, out=stat_all[b:b_end])

        chdtrc(k, stat_all.astype(np.float64), out=p)
        chdtri(1, np.clip(p, 1e-300, 1.0, out=p), out=p)
        stat_mat[c_start + i, :] = p
        sel_chunk[i, isSel] = 1

    stat_mat.flush()
    return c_start, sel_chunk

# Initialize empty memmap file on disk
np.memmap(stat_path, dtype='float32', mode='w+', shape=(Ns, N_snps))
sel_matrix = np.zeros((Ns, Np), dtype=np.uint8)

print(f"Running {Ns} subsets | {n_chunks} chunks | {n_workers} workers\nSNP_BATCH={SNP_BATCH:,} | N_snps={N_snps:,} | Np={Np}")

results = list(tqdm(
    Parallel(n_jobs=n_workers, prefer='threads', return_as="generator")(
        delayed(run_chunk)(c * chunk_size, min((c + 1) * chunk_size, Ns), seeds[c]) for c in range(n_chunks)
    ), total=n_chunks, desc="Chunks completed"
))

for c_start, sel_chunk in results:
    sel_matrix[c_start : c_start + len(sel_chunk), :] = sel_chunk

np.save(sel_path, sel_matrix)
stat_matrix = np.memmap(stat_path, dtype='float32', mode='r', shape=(Ns, N_snps))
print(f"stat_matrix : {stat_matrix.shape}  dtype={stat_matrix.dtype}\nsel_matrix  : {sel_matrix.shape}   dtype={sel_matrix.dtype}")