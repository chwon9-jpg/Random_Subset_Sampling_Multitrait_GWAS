import os
# Configuration used to generate the toy example expected outputs. Adjust for your machine.
os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "POLARS_MAX_THREADS": "32"}) 


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
base_out    = "/path/to/output/directory"
real_z_path = "/path/to/Z_matrix.npy"
save_path   = "/path/to/df_toy_allSNPs_allphenos.parquet"
ldsc_path   = "/path/to/LDSC_intercept_matrix.csv"

null_z_path    = f"{base_out}/null_Z_T.npy"
null_stat_path = f"{base_out}/null_stat_matrix.dat"
null_sel_path  = f"{base_out}/null_sel_matrix.npy"
null_score_path= f"{base_out}/null_score_matrix.dat"
out_parquet    = f"{base_out}/cumulative_joint_test_null.parquet"
out_png        = f"{base_out}/cumulative_joint_test_null.png"

# Configuration used to generate the toy example expected outputs. Adjust N_WORKERS_SUB/_CJT your machine.
N_snps    = 20_000   # matches the z-score matrix's SNP count exactly (user's call --
                     # gives the tightest possible calibration, resolution floor ~1.17e-7)
Ns        = 11_000
RIDGE_MARGIN = 1e-4    # floor for the data-driven ridge; see COR loading below
SEED_NULL    = 123     # null z-vectors
SEED_SUB     = 7       # subsets: match sampling_script.py's seed
SEED_MISSING = 99      # which real SNPs' missingness masks get sampled for the null

N_CHUNKS      = 100      # matches dynamic_sampling_loop.py
N_WORKERS_SUB = 32       # matches dynamic_sampling_loop.py
N_WORKERS_CJT = 16       # Stage D, mp.Pool (process-based, unlike B's threads)
SNP_BATCH     = 5_000  # Stage B only proven safe at this exact value/worker countin dynamic_sampling_loop.py
SCORE_SNP_BATCH = 5_000  # Stage C only (single-threaded) 

EARLY_STOP_TOL   = 0.30
EARLY_STOP_MIN_K = 5
KEEP_EVERY       = 10

if None in (N_snps, Ns, N_CHUNKS, N_WORKERS_SUB, N_WORKERS_CJT, SNP_BATCH, SCORE_SNP_BATCH):
    raise SystemExit("Set N_snps, Ns, N_CHUNKS, N_WORKERS_SUB, N_WORKERS_CJT, SNP_BATCH "
                      "and SCORE_SNP_BATCH in the configuration block above.")


def _elapsed(t0):
    m, s = divmod(time.time() - t0, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


# Shared inputs

print("Loading phenotype columns and LDSC intercept matrix...", flush=True)
pheno_cols = pl.read_parquet(save_path, n_rows=0).columns[1:]
Np = len(pheno_cols)

COR_df         = pd.read_csv(ldsc_path, sep='\t', index_col=0)
COR_df.index   = COR_df.index.str.removeprefix('z_')
COR_df.columns = COR_df.columns.str.removeprefix('z_')
COR            = COR_df.loc[pheno_cols, pheno_cols].values.astype(np.float64)

lambda_min = np.linalg.eigvalsh(COR).min()
RIDGE      = max(1e-4, RIDGE_MARGIN - lambda_min)
if RIDGE > 1e-4:
    print(f"COR min eigenvalue = {lambda_min:.6f} -> adding RIDGE = {RIDGE:.6f} to the "
          f"diagonal now (Cauchy interlacing then guarantees every submatrix built from "
          f"COR is PD too)", flush=True)
    COR += RIDGE * np.eye(Np)
COR = COR.astype(np.float32)


# Stage A: null z-vectors, with real-data-matched missingness

def run_stageA_null_z():
    print("=" * 70)
    print("STAGE A: null z-vectors ~ N(0, Sigma), with real-data-matched missingness")
    print("=" * 70)
    t0 = time.time()

    # COR already includes RIDGE (added once, upfront, above) -- not re-added here.
    L_full = cholesky(COR.astype(np.float64), lower=True, check_finite=False)
    rng    = np.random.default_rng(SEED_NULL)

    Z_T = np.lib.format.open_memmap(null_z_path, mode='w+', dtype=np.float32,
                                    shape=(Np, N_snps))
    Z_T[:, :] = (L_full @ rng.standard_normal((Np, N_snps))).astype(np.float32)

    print("  Injecting missingness sampled from the real z-score matrix's own pattern...",
          flush=True)
    Z_real = np.load(real_z_path, mmap_mode='r')
    if Z_real.shape[1] != Np:
        raise SystemExit(f"{Z_real.shape[1]} columns in {real_z_path} vs Np={Np}.")
    mrng     = np.random.default_rng(SEED_MISSING)
    real_idx = mrng.integers(0, Z_real.shape[0], size=N_snps)

    mask_batch = 2_000_000
    for b in range(0, N_snps, mask_batch):
        e = min(b + mask_batch, N_snps)
        masks = np.isnan(np.array(Z_real[real_idx[b:e], :]))   # (batch, Np)
        Z_T[:, b:e][masks.T] = np.nan

    Z_T.flush()
    n_missing_cells = int(np.isnan(np.asarray(Z_T)).sum())
    print(f"  injected {n_missing_cells:,} missing cells "
          f"({100 * n_missing_cells / (Np * N_snps):.3f}% of {Np * N_snps:,})", flush=True)
    del Z_T, L_full

    print(f"  done in {_elapsed(t0)} -> {null_z_path}", flush=True)


# Stage B: sampling loop (NaN-aware, mirrors Fullscale_sampling_loop_dynamic_sparsity_check.py)

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
            k     = int(rng.integers(5, Np + 1))
            isSel = np.sort(rng.choice(Np, k, replace=False))
            L_sub = cholesky(COR[np.ix_(isSel, isSel)], lower=True, check_finite=False)
            stat_all[:] = np.nan

            for b in range(0, N_snps, SNP_BATCH):
                b_end      = min(b + SNP_BATCH, N_snps)
                Z_block    = Z_T[isSel, b:b_end]
                valid_cols = ~np.any(np.isnan(Z_block), axis=0)
                if not valid_cols.any():
                    continue
                Y_valid = solve_triangular(L_sub, Z_block[:, valid_cols], lower=True,
                                           check_finite=False, overwrite_b=True)
                stat_all[b:b_end][valid_cols] = np.einsum('ij,ij->j', Y_valid, Y_valid)

            with np.errstate(invalid='ignore'):
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


# Stage C: null score matrix (NaN-aware, mirrors generate_score_matrix.py)

def run_stageC_score_matrix():
    print("\n" + "=" * 70)
    print("STAGE C: null score matrix")
    print("=" * 70)
    t0 = time.time()

    sel = np.load(null_sel_path)
    assert sel.shape == (Ns, Np)
    assert (sel.sum(axis=0) > 0).all(), "some phenotype was never drawn into any subset"
    sel_f32 = sel.astype(np.float32)

    stat_mm  = np.memmap(null_stat_path, dtype='float32', mode='r', shape=(Ns, N_snps))
    score_mm = np.memmap(null_score_path, dtype='float32', mode='w+', shape=(N_snps, Np))

    for b in range(0, N_snps, SCORE_SNP_BATCH):
        e = min(b + SCORE_SNP_BATCH, N_snps)
        stat_batch = np.array(stat_mm[:, b:e], dtype=np.float32)   # (Ns, batch)
        valid      = ~np.isnan(stat_batch)
        np.nan_to_num(stat_batch, copy=False, nan=0.0)
        valid_f32  = valid.astype(np.float32)

        numerator = stat_batch.T @ sel_f32
        denom     = valid_f32.T   @ sel_f32
        with np.errstate(invalid='ignore'):
            score_mm[b:e, :] = numerator / denom

    score_mm.flush()

    n_allnan = int(np.isnan(np.asarray(score_mm)).all(axis=1).sum())
    if n_allnan:
        print(f"  [!] {n_allnan:,} rows of null_score_matrix are entirely NaN "
              f"(no valid subset test for any phenotype at that simulated SNP).", flush=True)

    print(f"  done in {_elapsed(t0)} -> {null_score_path}", flush=True)


# Stage D: null cumulative joint test

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
    snp_scores_full = np.array(score_mm_g[idx, :], dtype=np.float64)
    z_vec_full      = np.array(Z_T_g[:, idx],       dtype=np.float64)

    # Restrict to jointly-observed phenotypes BEFORE ranking otherwise NaN-scored
    # phenotypes sort to the front of the descending order and corrupt the whole curve.
    observed = ~np.isnan(snp_scores_full) & ~np.isnan(z_vec_full)
    obs_idx  = np.where(observed)[0]
    m = len(obs_idx)

    log10p_curve = np.full(Np, np.nan)
    if m == 0:
        keep = (idx % KEEP_EVERY == 0)
        return (0, np.nan, np.nan, False, 0, False,
                log10p_curve.astype(np.float32) if keep else None)

    snp_scores = snp_scores_full[obs_idx]
    z_vec      = z_vec_full[obs_idx]
    Sig_obs    = Sigma_g[np.ix_(obs_idx, obs_idx)]

    order      = np.argsort(snp_scores)[::-1]
    z_s        = z_vec[order]
    # Sigma_g already includes RIDGE so it's not re-added here, unlike the real-data
    # scripts' _process_snp, which never pre-ridge Sigma and so must ridge at every call.
    Sig_s      = Sig_obs[np.ix_(order, order)]
    score_vals = snp_scores[order]

    L = np.zeros((m, m), dtype=np.float64)
    w = np.zeros(m,       dtype=np.float64)

    L[0, 0]         = np.sqrt(Sig_s[0, 0])
    w[0]            = z_s[0] / L[0, 0]
    S_k             = w[0] ** 2
    log10p_curve[0], fb = _safe_log10p(S_k, 1)
    fallback_any = fb

    best_v, best_k  = log10p_curve[0], 1
    stop_k, stopped = m, False

    for k in range(1, m):
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
                 initargs=(COR.astype(np.float64), null_score_path, null_z_path, N_snps, Np)) as pool:
        for i, r in enumerate(tqdm(
                pool.imap(_process_null_snp, snp_indices, chunksize=200),
                total=N_snps, desc="  SNPs (null)")):
            peak_ks[i], peak_scores[i], peak_logps[i] = r[0], r[1], r[2]
            fallback_any[i] = r[3]
            stop_ks[i], stopped[i] = r[4], r[5]
            if r[6] is not None:
                kept_curves.append(r[6])

    curves = np.array(kept_curves) if kept_curves else np.empty((0, Np), dtype=np.float32)
    n_untested = int(np.isnan(peak_logps).sum())
    print(f"\n  computed in {_elapsed(t0)}", flush=True)
    print(f"  curves retained for plotting: {len(curves):,}", flush=True)
    if n_untested:
        print(f"  [!] {n_untested:,} simulated SNPs had zero jointly-observed phenotypes "
              f"and were left out of the summary stats below.", flush=True)

    print(f"\nNULL - Normal-approximation fallback fired: "
          f"{int(fallback_any.sum()):,} of {N_snps:,}", flush=True)
    print(f"\nNULL - Early stopping:", flush=True)
    print(f"  triggered : {int(stopped.sum()):,} ({100*stopped.mean():.1f}%)", flush=True)
    print(f"  stop k    : mean={np.nanmean(stop_ks):.1f}, median={np.nanmedian(stop_ks):.0f}, "
          f"min={np.nanmin(stop_ks)}, max={np.nanmax(stop_ks)}", flush=True)

    print(f"\nNULL - Peak k     : mean={np.nanmean(peak_ks):.1f}, "
          f"median={np.nanmedian(peak_ks):.0f}, min={np.nanmin(peak_ks)}, max={np.nanmax(peak_ks)}",
          flush=True)
    print(f"NULL - Peak score : mean={np.nanmean(peak_scores):.4f}, "
          f"median={np.nanmedian(peak_scores):.4f}", flush=True)

    qs = [50, 90, 95, 99, 99.9, 99.99]
    qv = np.nanpercentile(peak_logps, qs)
    print(f"\nNULL - Peak -log10(p):", flush=True)
    for q, v in zip(qs, qv):
        print(f"  p{q:<6} = {v:8.3f}", flush=True)
    print(f"  max    = {np.nanmax(peak_logps):8.3f}   <- ceiling for the real comparison",
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
        ax1.axhline(np.nanmax(peak_logps), color='firebrick', lw=1.2,
                    label=f'null max ({np.nanmax(peak_logps):.1f})')
        ax1.legend(frameon=False, fontsize=9)

        ax2.plot(x_vals, n_at_k, color='grey', lw=1.2)
        ax2.set_ylabel('SNPs reaching $k$', fontsize=10)
        ax2.set_xlabel('Number of phenotypes added (ranked by phenotype score, descending)',
                       fontsize=11)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    ax1.set_ylabel(r'$-\log_{10}(p)$  of cumulative joint test ($\chi^2_k$)', fontsize=11)
    ax1.set_title(f'NULL cumulative joint test - {N_snps:,} simulated SNPs, '
                  r'$z \sim N(0, \Sigma)$, missingness matched to the real data'
                  f'\n(summary of {len(curves):,} retained curves)',
                  fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax3.hist(peak_ks[~np.isnan(peak_ks.astype(np.float64))], bins=100, color='grey',
             edgecolor='white', linewidth=0.3)
    ax3.set_xlabel('Peak k  (optimal number of phenotypes)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('NULL - Distribution of peak k', fontsize=11, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
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