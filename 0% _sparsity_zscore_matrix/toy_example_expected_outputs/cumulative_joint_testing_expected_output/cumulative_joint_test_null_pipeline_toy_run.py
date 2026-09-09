import os
os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "POLARS_MAX_THREADS": "4"})

import math
import time
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats
from scipy.linalg import cholesky, solve_triangular
from scipy.special import chdtrc, chdtri
import multiprocessing as mp
from tqdm import tqdm

# Paths
base_out     = "/path/to/directory"
parquet_path = "/path/to/df_toy_allSNPs_allphenos.parquet"
ldsc_path    = "/path/to/LDSC_toy_matrix.csv"

null_z_path    = f"{base_out}/null_Z_T.npy"
null_stat_path = f"{base_out}/null_stat_matrix.dat"
null_sel_path  = f"{base_out}/null_sel_matrix.npy"
null_score_path= f"{base_out}/null_score_matrix.dat"
out_parquet    = f"{base_out}/cumulative_joint_test_null_toy.parquet"
out_png        = f"{base_out}/cumulative_joint_test_null_toy.png"

# Configuration
N_snps    = 20_000
Np        = 100
Ns        = 10_000     # subsets, matching the real toy sampling run
RIDGE     = 1e-4
SEED_NULL = 123         # null z-vectors
SEED_SUB  = 7            # subsets: matches sampling_script_toy_run.py exactly

N_CHUNKS      = 16       # matches sampling_script_toy_run.py exactly (changes which
                         # subsets get drawn if changed — do not tune independently)
N_WORKERS_SUB = 8        # stage B workers (matches sampling_script_toy_run.py)
N_WORKERS_CJT = 16       # stage D workers (matches cumulative_joint_test_pipeline_toy_run.py)
SNP_BATCH     = 5_000    # stage B inner batch (matches sampling_script_toy_run.py)

EARLY_STOP_TOL   = 0.30   # MUST match cumulative_joint_test_pipeline_toy_run.py
EARLY_STOP_MIN_K = 5      # MUST match cumulative_joint_test_pipeline_toy_run.py
KEEP_EVERY       = 10     # matches cumulative_joint_test_pipeline_toy_run.py


def _elapsed(t0):
    m, s = divmod(time.time() - t0, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


# Shared inputs

print("Loading phenotype columns and LDSC intercept matrix...", flush=True)
pheno_cols = pl.read_parquet(parquet_path, n_rows=0).columns[1:]
assert len(pheno_cols) == Np, f"{len(pheno_cols)} phenotype columns vs Np={Np}"
COR_df = pd.read_csv(ldsc_path, index_col=0)
COR    = COR_df.loc[pheno_cols, pheno_cols].values.astype(np.float32)

# The cumulative-test stage reads the LDSC csv without reordering, so the two
# orderings must agree or z and Sigma end up permuted relative to each other.
assert list(COR_df.columns) == list(pheno_cols), "LDSC column order != parquet phenotype order"


# Stage A: null z-vectors

def run_stageA_null_z():
    print("=" * 70)
    print("STAGE A: null z-vectors ~ N(0, Sigma)")
    print("=" * 70)
    t0 = time.time()

    L_full = cholesky(COR.astype(np.float64) + RIDGE * np.eye(Np),
                      lower=True, check_finite=False)
    rng    = np.random.default_rng(SEED_NULL)

    Z_T = np.lib.format.open_memmap(null_z_path, mode='w+', dtype=np.float32,
                                    shape=(Np, N_snps))
    Z_T[:, :] = (L_full @ rng.standard_normal((Np, N_snps))).astype(np.float32)
    Z_T.flush()
    del Z_T, L_full

    print(f"  done in {_elapsed(t0)} -> {null_z_path}", flush=True)


# Stage B: sampling loop (identical to sampling_script_toy_run.py) 

def run_stageB_sampling_loop():
    print("\n" + "=" * 70)
    print("STAGE B: sampling loop on the null z-data")
    print("=" * 70)
    t0 = time.time()

    Z_T = np.load(null_z_path, mmap_mode='r')
    assert Z_T.shape == (Np, N_snps), f"unexpected Z_T shape {Z_T.shape}"
    Z_T = np.ascontiguousarray(Z_T)

    chunk_size = math.ceil(Ns / N_CHUNKS)
    seeds      = np.random.SeedSequence(SEED_SUB).spawn(N_CHUNKS)

    def run_chunk(c_start, c_end, seed):
        rng       = np.random.default_rng(seed)
        c_size    = c_end - c_start
        stat_mat  = np.memmap(null_stat_path, dtype='float32', mode='r+', shape=(Ns, N_snps))
        sel_chunk = np.zeros((c_size, Np), dtype=np.uint8)
        stat_all, p = np.empty(N_snps, dtype=np.float32), np.empty(N_snps, dtype=np.float64)

        for i in range(c_size):
            k     = int(rng.integers(5, 91))
            isSel = np.sort(rng.choice(Np, k, replace=False))
            L_sub = cholesky(COR[np.ix_(isSel, isSel)], lower=True, check_finite=False)
            stat_all[:] = 0.0

            for b in range(0, N_snps, SNP_BATCH):
                b_end   = min(b + SNP_BATCH, N_snps)
                Y_batch = solve_triangular(L_sub, Z_T[isSel, b:b_end], lower=True,
                                           check_finite=False, overwrite_b=True)
                np.einsum('ij,ij->j', Y_batch, Y_batch, out=stat_all[b:b_end])

            chdtrc(k, stat_all.astype(np.float64), out=p)
            chdtri(1, np.clip(p, 1e-300, 1.0, out=p), out=p)
            stat_mat[c_start + i, :] = p
            sel_chunk[i, isSel] = 1

        stat_mat.flush()
        return c_start, sel_chunk

    np.memmap(null_stat_path, dtype='float32', mode='w+', shape=(Ns, N_snps))
    sel_matrix = np.zeros((Ns, Np), dtype=np.uint8)

    print(f"Running {Ns} subsets | {N_CHUNKS} chunks | {N_WORKERS_SUB} workers", flush=True)
    results = list(tqdm(
        Parallel(n_jobs=N_WORKERS_SUB, prefer='threads', return_as="generator")(
            delayed(run_chunk)(c * chunk_size, min((c + 1) * chunk_size, Ns), seeds[c])
            for c in range(N_CHUNKS)
        ), total=N_CHUNKS, desc="  chunks"))

    for c_start, sel_chunk in results:
        sel_matrix[c_start: c_start + len(sel_chunk), :] = sel_chunk
    np.save(null_sel_path, sel_matrix)
    del Z_T

    print(f"  done in {_elapsed(t0)}", flush=True)
    print(f"  null_stat_matrix ({Ns:,} x {N_snps:,}) -> {null_stat_path}", flush=True)
    print(f"  null_sel_matrix  ({Ns:,} x {Np}) -> {null_sel_path}", flush=True)


# Stage C: null score matrix 

def run_stageC_score_matrix():
    print("\n" + "=" * 70)
    print("STAGE C: null score matrix")
    print("=" * 70)
    t0 = time.time()

    sel = np.load(null_sel_path)
    assert sel.shape == (Ns, Np)
    counts    = sel.sum(axis=0).astype(np.float64)
    assert (counts > 0).all(), "some phenotype was never drawn"
    sel_f64   = sel.astype(np.float64)

    stat_mm  = np.memmap(null_stat_path, dtype='float32', mode='r', shape=(Ns, N_snps))
    score_mm = np.memmap(null_score_path, dtype='float32', mode='w+', shape=(N_snps, Np))
    block    = np.ascontiguousarray(stat_mm[:, :].T)          # (N_snps, Ns) f32
    score_mm[:, :] = ((block.astype(np.float64) @ sel_f64) / counts).astype(np.float32)
    score_mm.flush()

    n_empty = int((np.asarray(score_mm).sum(axis=1) == 0).sum())
    if n_empty:
        raise RuntimeError(f"{n_empty:,} rows of null_score_matrix were never written.")

    print(f"  done in {_elapsed(t0)} -> {null_score_path}", flush=True)


# Stage D: null cumulative joint test (leave as None)

Sigma_g    = None
score_mm_g = None
Z_T_g      = None


def _init_worker(sigma, score_path_w, z_path_w, n_snps_w, np_w):
    global Sigma_g, score_mm_g, Z_T_g
    Sigma_g    = sigma
    score_mm_g = np.memmap(score_path_w, dtype='float32', mode='r', shape=(n_snps_w, np_w))
    Z_T_g      = np.load(z_path_w, mmap_mode='r')


def _safe_log10p(S, df):
    lsf = stats.chi2.logsf(S, df)
    if np.isneginf(lsf):
        z = (S - df) / np.sqrt(2.0 * df)
        return -stats.norm.logsf(z) / np.log(10), True
    return -lsf / np.log(10), False


def _process_null_snp(idx):
    snp_scores = np.array(score_mm_g[idx, :], dtype=np.float64)
    z_vec      = np.array(Z_T_g[:, idx],       dtype=np.float64)

    order      = np.argsort(snp_scores)[::-1]
    z_s        = z_vec[order]
    Sig_s      = Sigma_g[np.ix_(order, order)] + RIDGE * np.eye(Np)
    score_vals = snp_scores[order]

    L            = np.zeros((Np, Np), dtype=np.float64)
    w            = np.zeros(Np,       dtype=np.float64)
    log10p_curve = np.full(Np, np.nan)

    L[0, 0]         = np.sqrt(Sig_s[0, 0])
    w[0]            = z_s[0] / L[0, 0]
    S_k             = w[0] ** 2
    log10p_curve[0], fb = _safe_log10p(S_k, 1)
    fallback_any = fb

    best_v, best_k  = log10p_curve[0], 1
    stop_k, stopped = Np, False

    for k in range(1, Np):
        b        = Sig_s[:k, k]
        l        = solve_triangular(L[:k, :k], b, lower=True)
        schur    = max(Sig_s[k, k] - float(l @ l), 1e-12)
        l_kk     = np.sqrt(schur)
        L[k, :k] = l
        L[k,  k] = l_kk
        alpha    = (z_s[k] - float(l @ w[:k])) / l_kk
        w[k]     = alpha
        S_k     += alpha ** 2
        log10p_curve[k], fb = _safe_log10p(S_k, k + 1)
        fallback_any |= fb

        v = log10p_curve[k]
        if v > best_v:
            best_v, best_k = v, k + 1
        elif (k + 1) > EARLY_STOP_MIN_K and v < (1.0 - EARLY_STOP_TOL) * best_v:
            stop_k, stopped = k + 1, True
            break

    peak_k     = best_k
    peak_score = float(score_vals[peak_k - 1])
    peak_logp  = float(best_v)

    keep = (idx % KEEP_EVERY == 0)
    return (peak_k, peak_score, peak_logp, fallback_any, stop_k, stopped,
            log10p_curve.astype(np.float32) if keep else None)


def run_stageD_null_cjt():
    print("\n" + "=" * 70)
    print("STAGE D: cumulative joint test on the null toy data")
    print("=" * 70)

    Sigma = pd.read_csv(ldsc_path, index_col=0).loc[pheno_cols, pheno_cols].values.astype(np.float64)
    assert Sigma.shape == (Np, Np)

    score_mm = np.memmap(null_score_path, dtype='float32', mode='r', shape=(N_snps, Np))
    Z_T      = np.load(null_z_path, mmap_mode='r')
    assert Z_T.shape == (Np, N_snps)

    snp_indices = np.arange(N_snps)

    print(f"Running NULL cumulative joint test on {N_snps:,} simulated SNPs "
          f"with {N_WORKERS_CJT} workers", flush=True)
    print(f"  early stopping: tol={EARLY_STOP_TOL:.0%}, min k={EARLY_STOP_MIN_K}",
          flush=True)
    t0 = time.time()

    peak_ks      = np.empty(N_snps, dtype=np.int32)
    peak_scores  = np.empty(N_snps, dtype=np.float32)
    peak_logps   = np.empty(N_snps, dtype=np.float64)
    fallback_any = np.zeros(N_snps, dtype=bool)
    stop_ks      = np.empty(N_snps, dtype=np.int32)
    stopped      = np.zeros(N_snps, dtype=bool)
    kept_curves  = []

    with mp.Pool(N_WORKERS_CJT,
                 initializer=_init_worker,
                 initargs=(Sigma, null_score_path, null_z_path, N_snps, Np)) as pool:
        for i, r in enumerate(tqdm(
                pool.imap(_process_null_snp, snp_indices, chunksize=200),
                total=N_snps, desc="  SNPs (null)")):
            peak_ks[i], peak_scores[i], peak_logps[i] = r[0], r[1], r[2]
            fallback_any[i] = r[3]
            stop_ks[i], stopped[i] = r[4], r[5]
            if r[6] is not None:
                kept_curves.append(r[6])

    curves = np.array(kept_curves) if kept_curves else np.empty((0, Np), dtype=np.float32)
    print(f"\n  computed in {_elapsed(t0)}", flush=True)
    print(f"  curves retained for plotting: {len(curves):,}", flush=True)

    print(f"\nNULL - Normal-approximation fallback fired: "
          f"{int(fallback_any.sum()):,} of {N_snps:,}", flush=True)
    print(f"\nNULL - Early stopping:", flush=True)
    print(f"  triggered : {int(stopped.sum()):,} ({100*stopped.mean():.1f}%)", flush=True)
    print(f"  stop k    : mean={stop_ks.mean():.1f}, median={np.median(stop_ks):.0f}, "
          f"min={stop_ks.min()}, max={stop_ks.max()}", flush=True)

    print(f"\nNULL - Peak k     : mean={peak_ks.mean():.1f}, "
          f"median={np.median(peak_ks):.0f}, min={peak_ks.min()}, max={peak_ks.max()}",
          flush=True)
    print(f"NULL - Peak score : mean={peak_scores.mean():.4f}, "
          f"median={np.median(peak_scores):.4f}", flush=True)

    qs = [50, 90, 95, 99, 99.9, 99.99]
    qv = np.percentile(peak_logps, qs)
    print(f"\nNULL - Peak -log10(p):", flush=True)
    for q, v in zip(qs, qv):
        print(f"  p{q:<6} = {v:8.3f}", flush=True)
    print(f"  max    = {peak_logps.max():8.3f}   <- ceiling for the real comparison",
          flush=True)
    print(f"\n  resolution floor of the empirical p-value: "
          f"{1.0/(N_snps+1):.2e}", flush=True)

    pl.DataFrame({
        "sim_index":   snp_indices.astype(np.int32),
        "peak_k":      peak_ks,
        "peak_score":  peak_scores,
        "peak_log10p": peak_logps,
        "stop_k":      stop_ks,
        "stopped":     stopped.astype(np.uint8),
    }).write_parquet(out_parquet)
    print(f"\nSummary saved -> {out_parquet}", flush=True)

    x_vals = np.arange(1, Np + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 12), facecolor='white',
                                        gridspec_kw={'height_ratios': [3, 1, 2]})

    if len(curves):
        with np.errstate(invalid='ignore'):
            mean_c = np.nanmean(curves, axis=0)
            med_c  = np.nanmedian(curves, axis=0)
            q25    = np.nanpercentile(curves, 25, axis=0)
            q75    = np.nanpercentile(curves, 75, axis=0)
        n_at_k = np.sum(~np.isnan(curves), axis=0)

        ax1.fill_between(x_vals, q25, q75, color='grey', alpha=0.25,
                         label='interquartile range')
        ax1.plot(x_vals, mean_c, color='grey', lw=1.6, label='mean')
        ax1.plot(x_vals, med_c,  color='black', lw=1.6, ls='--', label='median')
        ax1.axhline(peak_logps.max(), color='firebrick', lw=1.2,
                    label=f'null max ({peak_logps.max():.1f})')
        ax1.legend(frameon=False, fontsize=9)

        ax2.plot(x_vals, n_at_k, color='grey', lw=1.2)
        ax2.set_ylabel('SNPs reaching $k$', fontsize=10)
        ax2.set_xlabel('Number of phenotypes added (ranked by phenotype score, descending)',
                       fontsize=11)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    ax1.set_ylabel(r'$-\log_{10}(p)$  of cumulative joint test ($\chi^2_k$)', fontsize=11)
    ax1.set_title(f'NULL cumulative joint test - {N_snps:,} simulated SNPs, '
                  r'$z \sim N(0, \Sigma)$'
                  f'\n(summary of {len(curves):,} retained curves)',
                  fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax3.hist(peak_ks, bins=100, color='grey', edgecolor='white', linewidth=0.3)
    ax3.set_xlabel('Peak k  (optimal number of phenotypes)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('NULL - Distribution of peak k', fontsize=11, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Plot saved -> {out_png}", flush=True)


if __name__ == '__main__':
    pipeline_t0 = time.time()

    run_stageA_null_z()
    run_stageB_sampling_loop()
    run_stageC_score_matrix()
    run_stageD_null_cjt()

    print("\n" + "=" * 70)
    print(f"Null pipeline complete in {_elapsed(pipeline_t0)}")
    print("=" * 70)
