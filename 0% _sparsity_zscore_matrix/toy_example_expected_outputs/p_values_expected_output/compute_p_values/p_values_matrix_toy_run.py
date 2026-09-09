##########################################################################################
#                           User Configuration required 
# These are None on purpose: set them for the machine this runs on rather than inheriting
# the values from the machine it last ran on.
#
#   N_WORKERS    : worker processes (loky backend).
#   BLAS_THREADS : threads per worker for the triangular solve. Total threads in use is
#                  roughly N_WORKERS * BLAS_THREADS — keep that at or below the core count.
#   SNP_BATCH    : SNPs per batch. Peak RAM is roughly 
#                   N_WORKERS * (SNP_BATCH * Np * 8 * 2 + Ns * SNP_BATCH * 4) bytes.
###########################################################################################

N_WORKERS, BLAS_THREADS = 8, 1
SNP_BATCH = 5_000

base_dir  = "/path/to/directory"
z_path    = "/path/to/Z_matrix.npy"  # output of the sampling script
stat_path = "/path/to/stat_matrix.dat" # output of the sampling script
cor_path  = "/path/to/LDSC_toy_matrix.csv"
save_path = "/path/to/df_toy_allSNPs_allphenos.parquet"
out_dir   = f"{base_dir}/p_values"

if None in (N_WORKERS, BLAS_THREADS, SNP_BATCH):
    raise SystemExit("Set N_WORKERS, BLAS_THREADS and SNP_BATCH in the configuration block above.")

import os
for env in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]: os.environ[env] = str(BLAS_THREADS)

import time, math, numpy as np, polars as pl, pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.special import chdtrc
from joblib import Parallel, delayed

col1_path, col2_path, col3_path = f"{out_dir}/col1.dat", f"{out_dir}/col2.dat", f"{out_dir}/col3.dat"

# Shapes taken from the inputs. stat_matrix is a raw .dat with no header, so its subset
# count is recovered from the file size; the check catches a stale or truncated file.
N_snps, Np = np.load(z_path, mmap_mode='r').shape
_nbytes    = os.path.getsize(stat_path)
if _nbytes % (N_snps * 4):
    raise SystemExit(f"{stat_path}: {_nbytes:,} B is not a whole number of rows for N_snps={N_snps:,} float32.")
Ns = _nbytes // (N_snps * 4)

def process_snp_range(snp_start, snp_end, L):
    Z    = np.load(z_path, mmap_mode='r')
    stat = np.memmap(stat_path, dtype='float32', mode='r',  shape=(Ns, N_snps))
    col1 = np.memmap(col1_path, dtype='float64', mode='r+', shape=(N_snps,))
    col2 = np.memmap(col2_path, dtype='float64', mode='r+', shape=(N_snps,))
    col3 = np.memmap(col3_path, dtype='float64', mode='r+', shape=(N_snps,))

    n_batches, t_start = math.ceil((snp_end - snp_start) / SNP_BATCH), time.time()

    for batch_i, b_start in enumerate(range(snp_start, snp_end, SNP_BATCH), 1):
        b_end = min(b_start + SNP_BATCH, snp_end)

        Z_batch  = np.array(Z[b_start:b_end, :], dtype=np.float32)
        col1[b_start:b_end] = chdtrc(1, Z_batch.astype(np.float64) ** 2).min(axis=1)

        Y_batch = solve_triangular(L, Z_batch.T.astype(np.float64), lower=True, check_finite=False)
        col2[b_start:b_end] = chdtrc(Np, (Y_batch ** 2).sum(axis=0))

        stat_batch = np.array(stat[:, b_start:b_end], dtype=np.float32)
        col3[b_start:b_end] = np.minimum(chdtrc(1, stat_batch.max(axis=0).astype(np.float64)) * Ns, 1.0)

        print(f"[worker {snp_start:,}-{snp_end:,}] batch {batch_i}/{n_batches} | SNPs {b_start:,}-{b_end:,} | {time.time() - t_start:.1f}s", flush=True)

    for c in [col1, col2, col3]: c.flush()

if __name__ == '__main__':
    pheno_cols = pl.read_parquet(save_path, n_rows=0).columns[1:]
    if len(pheno_cols) != Np:
        raise SystemExit(f"{len(pheno_cols)} phenotype columns in the parquet vs {Np} columns in the z-score matrix.")

    print("Loading COR matrix and computing Cholesky decomposition...")
    COR = pd.read_csv(cor_path, index_col=0).loc[pheno_cols, pheno_cols].values.astype(np.float64)
    L   = cholesky(COR, lower=True, check_finite=False)

    for p in [col1_path, col2_path, col3_path]: np.memmap(p, dtype='float64', mode='w+', shape=(N_snps,))

    chunk  = math.ceil(N_snps / N_WORKERS)
    ranges = [(i * chunk, min((i + 1) * chunk, N_snps)) for i in range(N_WORKERS)]
    print(f"Launching {N_WORKERS} workers | SNP_BATCH={SNP_BATCH:,} | BLAS={BLAS_THREADS}\nPeak RAM: {N_WORKERS * (SNP_BATCH * Np * 8 * 2 + Ns * SNP_BATCH * 4) / 1e9:.1f} GB")

    Parallel(n_jobs=N_WORKERS, backend='loky', verbose=5)(delayed(process_snp_range)(s, e, L) for s, e in ranges)

    print("Combining outputs and saving...")
    c1, c2, c3 = [np.memmap(p, dtype='float64', mode='r', shape=(N_snps,)) for p in [col1_path, col2_path, col3_path]]
    snp_ids = pl.scan_parquet(save_path).select("ID").collect()["ID"].to_list()

    df = pd.DataFrame({'snp_id': snp_ids, 'min_univariate_pval': c1, 'joint_pval': c2, 'min_subset_pval_bonf': c3})
    df.to_csv(f"{out_dir}/pvalue_matrix.csv", index=False)
    np.save(f"{out_dir}/pvalue_matrix.npy", np.column_stack([c1, c2, c3]))

    print(f"Saved outputs. Preview:\n{df.head(10).to_string(index=False)}")



'''
    Terminal output: 
    
    Loading COR matrix and computing Cholesky decomposition...
Launching 8 workers | SNP_BATCH=5,000 | BLAS=1
Peak RAM: 1.7 GB
[Parallel(n_jobs=8)]: Using backend LokyBackend with 8 concurrent workers.
[worker 7,500-10,000] batch 1/1 | SNPs 7,500-10,000 | 0.7s
[worker 10,000-12,500] batch 1/1 | SNPs 10,000-12,500 | 0.7s
[worker 15,000-17,500] batch 1/1 | SNPs 15,000-17,500 | 0.7s
[worker 2,500-5,000] batch 1/1 | SNPs 2,500-5,000 | 0.7s
[worker 12,500-15,000] batch 1/1 | SNPs 12,500-15,000 | 0.7s
[Parallel(n_jobs=8)]: Done   3 out of   8 | elapsed:    1.3s remaining:    2.1s
[Parallel(n_jobs=8)]: Done   5 out of   8 | elapsed:    1.3s remaining:    0.8s
[worker 5,000-7,500] batch 1/1 | SNPs 5,000-7,500 | 0.9s
[worker 0-2,500] batch 1/1 | SNPs 0-2,500 | 1.0s
[worker 17,500-20,000] batch 1/1 | SNPs 17,500-20,000 | 1.0s
[Parallel(n_jobs=8)]: Done   8 out of   8 | elapsed:    1.4s finished
Combining outputs and saving...
Saved outputs. Preview:
               snp_id  min_univariate_pval  joint_pval  min_subset_pval_bonf
 10:100057896_TTATC_T             0.004401    0.613672              1.000000
    10:115482961_CT_C             0.007863    0.335552              1.000000
    10:116006264_CA_C             0.002481    0.240644              1.000000
10:119171746_GGCTGT_G             0.029096    0.178143              1.000000
   10:120405445_CTG_C             0.013621    0.642911              1.000000
    10:120477466_CA_C             0.003961    0.841804              1.000000
    10:124510498_CT_C             0.002757    0.667971              1.000000
    10:126594360_CT_C             0.023129    0.010788              0.050559
    10:126725076_GA_G             0.031957    0.340958              1.000000
 10:129889492_ATTTG_A             0.000180    0.087707              0.748425

 '''