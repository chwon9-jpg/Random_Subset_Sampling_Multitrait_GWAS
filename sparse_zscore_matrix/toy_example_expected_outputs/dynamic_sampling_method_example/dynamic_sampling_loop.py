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

# Configuration used to generate output files - alter as needed for your machine
n_workers, n_chunks = 8, 16
SNP_BATCH  = 5_000

base_dir  = "/path/to/output/directory"  # Set this to the directory where Z_matrix.npy, stat_matrix.dat, and sel_matrix.npy will be stored
save_path = f"{base_dir}/df_toy_allSNPs_allphenos.parquet"
cor_path  = f"{base_dir}/LDSC_intercept_matrix.csv"

# Z_matrix.npy and save_path's Parquet are produced together by convert_zscore_to_npy.py
# if the z-score matrix currently only exists as a CSV/TSV/TXT file.

if None in (n_workers, n_chunks, SNP_BATCH):
    raise SystemExit("Set n_workers, n_chunks and SNP_BATCH in the configuration block above.")

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
n_used_path = f"{base_dir}/n_snp_used.npy"

# Load Data
pheno_cols = pl.read_parquet(save_path, n_rows=0).columns[1:]
Z_raw      = np.load(z_path)

COR_df          = pd.read_csv(cor_path, sep='\t', index_col=0)
COR_df.index    = COR_df.index.str.removeprefix('z_')
COR_df.columns  = COR_df.columns.str.removeprefix('z_')
COR             = COR_df.loc[pheno_cols, pheno_cols].values.astype(np.float64)

RIDGE_MARGIN = 1e-4
lambda_min   = np.linalg.eigvalsh(COR).min()
ridge_eps    = max(0.0, RIDGE_MARGIN - lambda_min)
if ridge_eps > 0:
    print(f"COR min eigenvalue = {lambda_min:.6f} -> adding ridge eps = {ridge_eps:.6f} to diagonal")
    COR += ridge_eps * np.eye(len(pheno_cols))
COR = COR.astype(np.float32)

# Load Z, transpose & free memory immediately
N_snps, Np = Z_raw.shape[0], len(pheno_cols)
Z_T        = np.ascontiguousarray(Z_raw.T)
del Z_raw


'''
Adaptive N_s: sparser Z matrices need more random subsets to get a stable null,
since more SNPs get dropped per subset (see per-subset filtering in run_chunk).
Sparsity = per-SNP missingness (fraction of phenotypes NaN for that SNP), averaged
genome-wide. Factor scales linearly between the two bands and is flat outside them.
'''
BASE_NS = 10_000
SPARSITY_LOW,  SPARSITY_HIGH  = 0.20, 0.80 # user can alter
FACTOR_LOW,    FACTOR_HIGH    = 1.1,  1.5 # user can alter
 
sparsity = float(np.isnan(Z_T).mean(axis=0).mean())
if sparsity <= SPARSITY_LOW:
    ns_factor = FACTOR_LOW
elif sparsity >= SPARSITY_HIGH:
    ns_factor = FACTOR_HIGH
else:
    ns_factor = FACTOR_LOW + (FACTOR_HIGH - FACTOR_LOW) * (sparsity - SPARSITY_LOW) / (SPARSITY_HIGH - SPARSITY_LOW)
Ns = int(round(BASE_NS * ns_factor))
print(f"Z-matrix sparsity (avg per-SNP missingness) = {sparsity:.2%} -> N_s factor = {ns_factor:.3f} -> Ns = {Ns:,}")

chunk_size = math.ceil(Ns / n_chunks)
seeds      = np.random.SeedSequence(7).spawn(n_chunks)

def run_chunk(c_start, c_end, seed):
    rng       = np.random.default_rng(seed)
    c_size    = c_end - c_start
    stat_mat  = np.memmap(stat_path, dtype='float32', mode='r+', shape=(Ns, N_snps))
    sel_chunk = np.zeros((c_size, Np), dtype=np.uint8)
    n_used_chunk = np.zeros(c_size, dtype=np.int32)
    stat_all, p = np.empty(N_snps, dtype=np.float32), np.empty(N_snps, dtype=np.float64)

    for i in range(c_size):
        k = int(rng.integers(5, 79))
        isSel = np.sort(rng.choice(Np, k, replace=False))
        L_sub = cholesky(COR[np.ix_(isSel, isSel)], lower=True, check_finite=False)
        stat_all[:] = np.nan
        n_valid = 0

        for b in range(0, N_snps, SNP_BATCH):
            b_end      = min(b + SNP_BATCH, N_snps)
            Z_block    = Z_T[isSel, b:b_end]
            valid_cols = ~np.any(np.isnan(Z_block), axis=0)
            n_valid   += int(valid_cols.sum())
            if not valid_cols.any():
                continue
            Y_valid = solve_triangular(L_sub, Z_block[:, valid_cols], lower=True, check_finite=False, overwrite_b=True)
            stat_all[b:b_end][valid_cols] = np.einsum('ij,ij->j', Y_valid, Y_valid)

        # SNPs excluded from this subset (missing phenotype in isSel) stay NaN in
        # stat_all and flow through as NaN in stat_chi2_1. — the k used for the chi2_1 rescale
        # is fixed by the subset size, not by how many SNPs survived filtering.
        with np.errstate(invalid='ignore'):
            chdtrc(k, stat_all.astype(np.float64), out=p)
            chdtri(1, np.clip(p, 1e-300, 1.0, out=p), out=p)
        stat_mat[c_start + i, :] = p
        sel_chunk[i, isSel] = 1
        n_used_chunk[i] = n_valid

    stat_mat.flush()
    return c_start, sel_chunk, n_used_chunk

# Initialize empty memmap file on disk
np.memmap(stat_path, dtype='float32', mode='w+', shape=(Ns, N_snps))
sel_matrix   = np.zeros((Ns, Np), dtype=np.uint8)
n_snp_used   = np.zeros(Ns, dtype=np.int32)

print(f"Running {Ns} subsets | {n_chunks} chunks | {n_workers} workers\nSNP_BATCH={SNP_BATCH:,} | N_snps={N_snps:,} | Np={Np}")

results = list(tqdm(
    Parallel(n_jobs=n_workers, prefer='threads', return_as="generator")(
        delayed(run_chunk)(c * chunk_size, min((c + 1) * chunk_size, Ns), seeds[c]) for c in range(n_chunks)
    ), total=n_chunks, desc="Chunks completed"
))

for c_start, sel_chunk, n_used_chunk in results:
    sel_matrix[c_start : c_start + len(sel_chunk), :] = sel_chunk
    n_snp_used[c_start : c_start + len(n_used_chunk)] = n_used_chunk

np.save(sel_path, sel_matrix)
np.save(n_used_path, n_snp_used)
stat_matrix = np.memmap(stat_path, dtype='float32', mode='r', shape=(Ns, N_snps))
print(f"stat_matrix : {stat_matrix.shape}  dtype={stat_matrix.dtype}\nsel_matrix  : {sel_matrix.shape}   dtype={sel_matrix.dtype}\nn_snp_used  : {n_snp_used.shape}   dtype={n_snp_used.dtype}  (SNPs per subset after missingness filter)")