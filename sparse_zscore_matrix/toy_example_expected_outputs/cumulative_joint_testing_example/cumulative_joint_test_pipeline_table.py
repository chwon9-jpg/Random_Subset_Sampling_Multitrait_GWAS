import os
# Configuration used to generate the toy example expected outputs. Adjust for your machine.
os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "POLARS_MAX_THREADS": "8"})

import time
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from scipy.linalg import cholesky, solve_triangular
import multiprocessing as mp
from tqdm import tqdm
from mpmath import mp as mp_ctx, mpf, gammainc, log10 as mp_log10

mp_ctx.dps = 80

# Paths
base_out          = "/path/to/output/directory"
score_path        = "/path/to/score_matrix.dat"
z_path            = "/path/to/Z_matrix.npy"
save_path         = "/path/to/df_toy_allSNPs_allphenos.parquet"
ldsc_path         = "/path/to/LDSC_intercept_matrix.csv"
null_parquet      = f"{base_out}/cumulative_joint_test_null.parquet"
pvalue_matrix_npy = "/path/to/pvalue_matrix.npy"

# Stage 1 outputs
out_parquet  = f"{base_out}/cumulative_joint_test_real.parquet"
out_hits_tsv = f"{base_out}/cumulative_joint_test_real_hits.tsv"

# Stage 2 outputs
out_parquet_corrected = f"{base_out}/cumulative_joint_test_real_corrected.parquet"
out_hits_corrected    = f"{base_out}/cumulative_joint_test_real_hits_corrected.tsv"

# Stage 3 output (table, in place of the figure)
out_table_parquet = f"{base_out}/final_output_table.parquet"

# Parameter configuration
N_WORKERS = 16
RIDGE_MARGIN = 1e-4    # floor for the data-driven ridge; see COR loading below

EARLY_STOP_TOL   = 0.30
EARLY_STOP_MIN_K = 5

KEEP_EVERY = 10     # unused by stage 3 here, kept for parity with the figure script
CHUNK      = 5_000   # stage 3: SNP rows per chunk when assembling the output table

MIN_CLEAN_STREAK = 100

if N_WORKERS is None or CHUNK is None:
    raise SystemExit("Set N_WORKERS and CHUNK in the configuration block above.")

Sigma       = None
score_mm    = None
Z_mm        = None
RIDGE       = 1e-4  
                
pheno_cols = pl.read_parquet(save_path, n_rows=0).columns[1:]
Np = len(pheno_cols)
N_snps = np.load(z_path, mmap_mode='r').shape[0]
N_SNPS = N_snps            # all SNPs, no sampling
pheno_names = pheno_cols


# Stage 1: real cumulative joint test

def _init_worker(sigma, ridge, score_path_w, z_path_w, n_snps_w, np_w):
    global Sigma, RIDGE, score_mm, Z_mm
    Sigma    = sigma
    RIDGE    = ridge
    score_mm = np.memmap(score_path_w, dtype='float32', mode='r', shape=(n_snps_w, np_w))
    Z_mm     = np.load(z_path_w, mmap_mode='r')


def _safe_log10p(S, df):
    """Return (-log10 p, fallback_used).

    The normal approximation is only reached when chi2.logsf underflows. It is
    unreliable in the far tail, so its use is tracked rather than silent.
    """
    lsf = stats.chi2.logsf(S, df)
    if np.isneginf(lsf):
        z = (S - df) / np.sqrt(2.0 * df)
        return -stats.norm.logsf(z) / np.log(10), True
    return -lsf / np.log(10), False


def _process_snp(idx):
    """Cumulative joint test for one real SNP, with early stopping."""
    snp_scores_full = np.array(score_mm[idx, :], dtype=np.float64)
    z_vec_full      = np.array(Z_mm[idx, :],     dtype=np.float64)

    observed = ~np.isnan(snp_scores_full) & ~np.isnan(z_vec_full)
    obs_idx  = np.where(observed)[0]
    m = len(obs_idx)

    log10p_curve = np.full(Np, np.nan)
    if m == 0:
        return (0, np.nan, np.nan, False, False, 0, False, log10p_curve.astype(np.float32))

    snp_scores = snp_scores_full[obs_idx]
    z_vec      = z_vec_full[obs_idx]
    Sig_obs    = Sigma[np.ix_(obs_idx, obs_idx)]

    order      = np.argsort(snp_scores)[::-1]
    z_s        = z_vec[order]
    Sig_s      = Sig_obs[np.ix_(order, order)] + RIDGE * np.eye(m)
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
    peak_fallback = bool(np.isneginf(stats.chi2.logsf(
        np.nansum([w[i] ** 2 for i in range(peak_k)]), peak_k)))

    keep = (idx % KEEP_EVERY == 0)
    return (peak_k, peak_score, peak_logp, fallback_any, peak_fallback,
            stop_k, stopped, log10p_curve.astype(np.float32) if keep else None)


def _elapsed(t0):
    m, s = divmod(time.time() - t0, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


def run_stage1_real_test():
    """Runs the real cumulative joint test over all toy SNPs.
    Returns (out_df, curves, p_emp, null_logps, null_max) for parity with the
    figure-producing script; stage 3 here only uses out_df plus the globals
    (Sigma, score_mm, Z_mm, pheno_names) set below."""
    global Sigma, RIDGE, score_mm, Z_mm

    print("=" * 70)
    print("STAGE 1: cumulative joint test on the real toy data")
    print("=" * 70)

    print("Loading LDSC matrix...", flush=True)
    COR_df         = pd.read_csv(ldsc_path, sep='\t', index_col=0)
    COR_df.index   = COR_df.index.str.removeprefix('z_')
    COR_df.columns = COR_df.columns.str.removeprefix('z_')
    Sigma          = COR_df.loc[pheno_cols, pheno_cols].values.astype(np.float64)
    assert Sigma.shape == (Np, Np), f"unexpected Sigma shape {Sigma.shape}"

    lambda_min = np.linalg.eigvalsh(Sigma).min()
    RIDGE      = max(1e-4, RIDGE_MARGIN - lambda_min)
    if RIDGE > 1e-4:
        print(f"  COR min eigenvalue = {lambda_min:.6f} -> using RIDGE = {RIDGE:.6f}",
              flush=True)

    print("Opening score and Z memmaps...", flush=True)
    score_mm = np.memmap(score_path, dtype='float32', mode='r', shape=(N_snps, Np))
    Z_mm     = np.load(z_path, mmap_mode='r')

    snp_indices = np.arange(N_SNPS)
    snp_ids = snp_indices   # no SNP-ID parquet for this toy z-score matrix; use row index

    print(f"\nRunning cumulative joint test on {N_SNPS:,} SNPs with {N_WORKERS} workers",
          flush=True)
    print(f"  early stopping: tol={EARLY_STOP_TOL:.0%}, min k={EARLY_STOP_MIN_K}",
          flush=True)
    t0 = time.time()

    peak_ks       = np.empty(N_SNPS, dtype=np.int32)
    peak_scores   = np.empty(N_SNPS, dtype=np.float32)
    peak_logps    = np.empty(N_SNPS, dtype=np.float64)
    fallback_any  = np.zeros(N_SNPS, dtype=bool)
    peak_fallback = np.zeros(N_SNPS, dtype=bool)
    stop_ks       = np.empty(N_SNPS, dtype=np.int32)
    stopped       = np.zeros(N_SNPS, dtype=bool)
    kept_curves   = []

    with mp.Pool(N_WORKERS,
                 initializer=_init_worker,
                 initargs=(Sigma, RIDGE, score_path, z_path, N_snps, Np)) as pool:
        for i, r in enumerate(tqdm(
                pool.imap(_process_snp, snp_indices, chunksize=200),
                total=N_SNPS, desc="  SNPs")):
            peak_ks[i], peak_scores[i], peak_logps[i] = r[0], r[1], r[2]
            fallback_any[i], peak_fallback[i]         = r[3], r[4]
            stop_ks[i], stopped[i]                    = r[5], r[6]
            if r[7] is not None:
                kept_curves.append(r[7])

    curves = np.array(kept_curves) if kept_curves else np.empty((0, Np), dtype=np.float32)
    n_untested = int(np.isnan(peak_logps).sum())
    print(f"\n  computed in {_elapsed(t0)}", flush=True)
    print(f"  curves retained for plotting: {len(curves):,}", flush=True)
    if n_untested:
        print(f"  [!] {n_untested:,} SNPs had zero jointly-observed phenotypes and were "
              f"excluded from the summary stats below.", flush=True)

    print(f"\nNormal-approximation fallback:", flush=True)
    print(f"  fired anywhere on a curve : {int(fallback_any.sum()):,} of {N_SNPS:,}",
          flush=True)
    print(f"  fired at the PEAK value   : {int(peak_fallback.sum()):,}", flush=True)
    if peak_fallback.any():
        print("  [!] affected peaks are unreliable - recompute with mpmath:", flush=True)
        print(f"      max affected peak = {np.nanmax(peak_logps[peak_fallback]):.3f}",
              flush=True)
    else:
        print("  all reported peak values come from the exact chi-squared computation.",
              flush=True)

    print(f"\nEarly stopping:", flush=True)
    print(f"  triggered : {int(stopped.sum()):,} ({100*stopped.mean():.1f}%)", flush=True)
    print(f"  stop k    : mean={np.nanmean(stop_ks):.1f}, median={np.nanmedian(stop_ks):.0f}, "
          f"min={np.nanmin(stop_ks)}, max={np.nanmax(stop_ks)}", flush=True)

    print(f"\nPeak k         : mean={np.nanmean(peak_ks):.1f}, median={np.nanmedian(peak_ks):.0f}, "
          f"min={np.nanmin(peak_ks)}, max={np.nanmax(peak_ks)}", flush=True)
    print(f"Peak score     : mean={np.nanmean(peak_scores):.4f}, "
          f"median={np.nanmedian(peak_scores):.4f}", flush=True)
    qs = [50, 90, 95, 99, 99.9, 99.99]
    qv = np.nanpercentile(peak_logps, qs)
    print(f"Peak -log10(p) :", flush=True)
    for q, v in zip(qs, qv):
        print(f"  p{q:<6} = {v:8.3f}", flush=True)
    print(f"  max    = {np.nanmax(peak_logps):8.3f}", flush=True)

    # Calibration against a full null run
    p_emp = None
    null_max, null_logps = None, None
    if os.path.exists(null_parquet):
        null_df    = pl.read_parquet(null_parquet)
        null_logps = null_df["peak_log10p"].to_numpy()
        null_logps = null_logps[~np.isnan(null_logps)]
        n_null     = len(null_logps)
        null_max   = null_logps.max()

        print(f"\n--- Calibration against {n_null:,} null SNPs ---", flush=True)
        print(f"Null peak -log10(p): median={np.median(null_logps):.3f}, "
              f"p99={np.percentile(null_logps, 99):.3f}, max={null_max:.3f}", flush=True)

        null_sorted = np.sort(null_logps)
        valid_peak       = ~np.isnan(peak_logps)
        rank             = np.full(N_SNPS, -1, dtype=np.int64)
        rank[valid_peak] = np.searchsorted(null_sorted, peak_logps[valid_peak], side='left')
        ge_count         = n_null - rank
        p_emp            = np.where(valid_peak, (ge_count + 1) / (n_null + 1), np.nan)

        n_exceed = int((peak_logps[valid_peak] > null_max).sum())
        exp_fp   = N_SNPS / (n_null + 1)
        print(f"\nReal SNPs exceeding the null maximum : {n_exceed:,} "
              f"({100*n_exceed/N_SNPS:.4f}% of {N_SNPS:,})", flush=True)
        print(f"Expected false positives at this threshold : {exp_fp:.2f}", flush=True)
        print(f"Enrichment : {n_exceed/max(exp_fp, 1e-12):.1f}x", flush=True)
        for thr in (0.05, 0.01, 0.001, 1.0/(n_null+1)):
            print(f"  p_emp <= {thr:.2e} : {int(np.nansum(p_emp <= thr)):,}", flush=True)
        print(f"Resolution floor of the empirical p-value: {1.0/(n_null+1):.2e}",
              flush=True)
    else:
        print(f"\n[!] Null file not found at {null_parquet} - skipping calibration. "
              f"Run cumulative_joint_test_null_pipeline.py first.", flush=True)

    # Output
    out_df = pl.DataFrame({
        "SNP_ID":      snp_ids,
        "peak_k":      peak_ks,
        "peak_score":  peak_scores,
        "peak_log10p": peak_logps,
        "stop_k":      stop_ks,
        "stopped":     stopped.astype(np.uint8),
    })
    if p_emp is not None:
        out_df = out_df.with_columns(pl.Series("p_emp", p_emp))
    out_df.write_parquet(out_parquet)
    print(f"\nSummary saved -> {out_parquet}", flush=True)

    if p_emp is not None:
        hits = (out_df.filter(pl.col("peak_log10p") > null_max)
                       .sort("peak_log10p", descending=True))
        hits.write_csv(out_hits_tsv, separator="\t")
        print(f"SNPs above the null ceiling ({hits.height:,}) -> {out_hits_tsv}",
              flush=True)

    print(f"\nStage 1 total: {_elapsed(t0)}", flush=True)

    return out_df, curves, p_emp, null_logps, null_max


# Stage 2: fallback correction

def exact_neg_log10p(S, k):
    """Exact -log10 P(chi2_k > S) via the regularised upper incomplete gamma,
    evaluated at high precision so it never underflows."""
    q = gammainc(mpf(k) / 2, mpf(S) / 2, mp_ctx.inf, regularized=True)
    return float(-mp_log10(q))


def run_stage2_fix_fallback(df):
    """Corrects fallback-affected peaks in df. Returns the corrected DataFrame."""
    print("\n" + "=" * 70)
    print("STAGE 2: fallback correction")
    print("=" * 70)

    n_dup = df.height - df["SNP_ID"].n_unique()
    if n_dup:
        print(f"  [!] {n_dup} duplicate SNP_ID values in the output - "
              f"name-based lookup would have been unsafe; using row index instead.")

    df_indexed = df.with_row_index("row_idx")
    candidates = (df_indexed.filter(pl.col("peak_log10p").is_not_nan())
                             .sort("peak_log10p", descending=True))
    total_candidates = candidates.height
    print(f"Checking candidates by reported peak -log10(p), widening until "
          f"{MIN_CLEAN_STREAK} consecutive candidates in a row need no correction "
          f"({total_candidates:,} candidates available)...")

    n_corrected = 0
    n_checked = 0
    corrections = {}
    correction_ranks = []
    clean_streak = 0

    for rank, row in enumerate(candidates.iter_rows(named=True)):
        idx = row["row_idx"]
        k   = int(row["peak_k"])

        snp_scores_full = np.asarray(score_mm[idx, :], dtype=np.float64)
        z_vec_full      = np.asarray(Z_mm[idx, :],      dtype=np.float64)

        observed = ~np.isnan(snp_scores_full) & ~np.isnan(z_vec_full)
        obs_idx  = np.where(observed)[0]
        snp_scores = snp_scores_full[obs_idx]
        order_local = np.argsort(snp_scores)[::-1][:k]
        order = obs_idx[order_local]

        Sig_k = Sigma[np.ix_(order, order)] + RIDGE * np.eye(k)
        z_k   = z_vec_full[order]

        L = cholesky(Sig_k, lower=True, check_finite=False)
        Y = np.linalg.solve(L, z_k)
        S = float(Y @ Y)

        exact_val = exact_neg_log10p(S, k)
        stored_val = row["peak_log10p"]
        n_checked = rank + 1

        if abs(exact_val - stored_val) > max(1e-3, 1e-6 * abs(stored_val)):
            corrections[idx] = exact_val
            correction_ranks.append(rank)
            n_corrected += 1
            clean_streak = 0
        else:
            clean_streak += 1
            if clean_streak >= MIN_CLEAN_STREAK:
                break

    print(f"\nChecked {n_checked:,} of {total_candidates:,} candidates with a "
          f"non-NaN peak_log10p")
    print(f"Corrected {n_corrected} of {n_checked:,} checked")
    if correction_ranks:
        last_rank = max(correction_ranks)
        print(f"Rank (0-indexed, by peak_log10p) of the last SNP requiring "
              f"correction: {last_rank}")
        if n_checked == total_candidates and clean_streak < MIN_CLEAN_STREAK:
            print(f"  Reached the end of all candidates before completing a "
                  f"{MIN_CLEAN_STREAK}-candidate clean streak (only {clean_streak} "
                  f"available past the last correction) -- every candidate has now "
                  f"been checked, so the boundary is still fully confirmed, just by "
                  f"exhaustion rather than by streak length.")
        else:
            print(f"  Boundary confirmed: {n_checked - 1 - last_rank} consecutive "
                  f"clean candidates checked past it "
                  f"(target streak: {MIN_CLEAN_STREAK}).")
    else:
        print(f"No corrections needed in the first {n_checked} candidates checked "
              f"(the most extreme SNPs) -- nothing in this dataset needed correction.")

    if corrections:
        print("\nLargest corrections:")
        shown = sorted(corrections.items(), key=lambda kv: -abs(
            df["peak_log10p"][kv[0]] - kv[1]))[:15]
        for idx, new_val in shown:
            sid = df["SNP_ID"][idx]
            old_val = df["peak_log10p"][idx]
            print(f"  {sid:<20} stored={old_val:12.3f}  exact={new_val:12.3f}")

        new_logp = df["peak_log10p"].to_numpy().copy()
        for idx, val in corrections.items():
            new_logp[idx] = val
        df = df.with_columns(pl.Series("peak_log10p", new_logp))
        df = df.with_columns(
            pl.Series("was_fallback_corrected",
                      np.isin(np.arange(len(df)), list(corrections.keys())).astype(np.uint8))
        )
    else:
        df = df.with_columns(pl.lit(0, dtype=pl.UInt8).alias("was_fallback_corrected"))

    new_max = df["peak_log10p"].max()
    print(f"\nNew maximum peak -log10(p): {new_max:.3f}")

    if os.path.exists(null_parquet):
        null_df    = pl.read_parquet(null_parquet)
        null_logps = null_df["peak_log10p"].to_numpy()
        null_logps = null_logps[~np.isnan(null_logps)]
        n_null     = len(null_logps)
        null_max   = null_logps.max()
        null_sorted = np.sort(null_logps)

        peak_logps = df["peak_log10p"].to_numpy()
        valid_peak = ~np.isnan(peak_logps)
        rank       = np.full(len(df), -1, dtype=np.int64)
        rank[valid_peak] = np.searchsorted(null_sorted, peak_logps[valid_peak], side='left')
        p_emp      = np.where(valid_peak, (n_null - rank + 1) / (n_null + 1), np.nan)
        df = df.with_columns(pl.Series("p_emp", p_emp))

        n_exceed = int((peak_logps[valid_peak] > null_max).sum())
        print(f"\nCorrected: SNPs exceeding the null maximum ({null_max:.3f}): "
              f"{n_exceed:,} ({100*n_exceed/N_snps:.4f}% of {N_snps:,})")

        hits = df.filter(pl.col("peak_log10p") > null_max).sort(
            "peak_log10p", descending=True)
        hits.write_csv(out_hits_corrected, separator="\t")
        print(f"Corrected hits -> {out_hits_corrected}")
    else:
        print(f"\n[!] Null file not found at {null_parquet} - skipping calibration "
              f"recompute.")

    df.write_parquet(out_parquet_corrected)
    print(f"\nCorrected summary -> {out_parquet_corrected}")

    return df


# Stage 3: final output table (replaces the figure)

def run_stage3_final_table(corrected_df):
    print("\n" + "=" * 70)
    print("STAGE 3: final output table (SNP x phenotype scores + test results)")
    print("=" * 70)
    t0 = time.time()

    snp_ids   = corrected_df["SNP_ID"].to_numpy()
    peak_k    = corrected_df["peak_k"].to_numpy().astype(np.int32)
    peak_logp = corrected_df["peak_log10p"].to_numpy().astype(np.float64)

    print("Loading joint p-values...", flush=True)
    pv = np.load(pvalue_matrix_npy)
    assert pv.shape[0] == N_snps, f"pvalue_matrix rows {pv.shape[0]} != {N_snps}"
    joint_p          = pv[:, 1].astype(np.float64)   # column 1 = 'joint_pval'
    n_pheno_observed = pv[:, 2].astype(np.int32)      # column 2 = 'n_pheno_observed'

    with np.errstate(divide='ignore'):
        joint_neglog10p = -np.log10(joint_p)

    zero_idx = np.where(joint_p == 0.0)[0]
    print(f"Rescuing {len(zero_idx)} underflowed joint p-values via mpmath...", flush=True)

    for idx in zero_idx:
        z_full  = np.asarray(Z_mm[idx, :], dtype=np.float64)
        obs_idx = np.where(~np.isnan(z_full))[0]
        m = len(obs_idx)
        if m == 0:
            continue
        assert m == n_pheno_observed[idx], (
            f"SNP {idx}: recomputed observed-phenotype count {m} != "
            f"n_pheno_observed {n_pheno_observed[idx]} stored in pvalue_matrix.npy -- "
            f"the Z matrix used here doesn't match the one p_values_matrix.py used.")
        Sig_m = Sigma[np.ix_(obs_idx, obs_idx)] + RIDGE * np.eye(m)
        L = cholesky(Sig_m, lower=True, check_finite=False)
        y = solve_triangular(L, z_full[obs_idx], lower=True, check_finite=False)
        S = float(y @ y)
        joint_neglog10p[idx] = exact_neg_log10p(S, m)

    print("\nAssembling the output table in chunks from the score matrix...", flush=True)
    out_cols = (["SNP_ID"] + list(pheno_names) +
                ["n_optimal_phenotypes", "neglog10p_optimal_set",
                 "n_pheno_observed_joint", "neglog10p_joint_observed_phenotypes"])

    parts = []
    for start in range(0, N_snps, CHUNK):
        end = min(start + CHUNK, N_snps)
        block = np.ascontiguousarray(score_mm[start:end, :])
        df = pl.from_numpy(block, schema=list(pheno_names), orient="row")
        df = df.with_columns([
            pl.Series("SNP_ID", snp_ids[start:end]),
            pl.Series("n_optimal_phenotypes", peak_k[start:end]),
            pl.Series("neglog10p_optimal_set", peak_logp[start:end]),
            pl.Series("n_pheno_observed_joint", n_pheno_observed[start:end]),
            pl.Series("neglog10p_joint_observed_phenotypes", joint_neglog10p[start:end]),
        ]).select(out_cols)
        parts.append(df)
        print(f"  chunk {start:>7,}-{end:>7,}  ({_elapsed(t0)})", flush=True)

    out_df = pl.concat(parts, rechunk=False)
    assert out_df.height == N_snps
    assert out_df.width == 1 + Np + 4

    out_df.write_parquet(out_table_parquet)
    print(f"\nWrote {out_df.height:,} rows x {out_df.width} columns -> {out_table_parquet}",
          flush=True)
    print(f"Stage 3 total: {_elapsed(t0)}", flush=True)


if __name__ == '__main__':
    pipeline_t0 = time.time()

    real_df, curves, p_emp, null_logps, null_max = run_stage1_real_test()
    corrected_df = run_stage2_fix_fallback(real_df)
    run_stage3_final_table(corrected_df)

    print("\n" + "=" * 70)
    print(f"Pipeline complete in {_elapsed(pipeline_t0)}")
    print("=" * 70)