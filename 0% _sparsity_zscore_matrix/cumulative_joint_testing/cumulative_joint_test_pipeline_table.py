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
base_out     = "/path/to/directory"
score_path   = "/path/to/score_matrix.dat"
z_path       = "/path/to/Z_matrix.npy"
parquet_path = "/path/to/df_allSNPs_allphenos.parquet"
ldsc_path    = "/path/to/LDSC_intercept_matrix.csv"
null_parquet = f"{base_out}/cumulative_joint_test_null.parquet"
pvalue_matrix_npy = "/path/to/pvalue_matrix.npy"

# Stage 1 outputs
out_parquet  = f"{base_out}/cumulative_joint_test_real.parquet"
out_hits_tsv = f"{base_out}/cumulative_joint_test_real_hits.tsv"

# Stage 2 outputs
out_parquet_corrected = f"{base_out}/cumulative_joint_test_real_corrected.parquet"
out_hits_corrected    = f"{base_out}/cumulative_joint_test_real_hits_corrected.tsv"

# Stage 3 output (table)
out_table_parquet = f"{base_out}/final_output_table.parquet"

# Configuration: remaining "User to define" values
RIDGE = 1e-4

EARLY_STOP_TOL   = 0.30
EARLY_STOP_MIN_K = 5

KEEP_EVERY = 10     # curves retained for one SNP in KEEP_EVERY (unused by stage 3 here, kept for parity)
CHUNK      = None   # stage 3: SNP rows per chunk when assembling the output table

# Stage 2 no longer checks a fixed number of top candidates for fallback correction.
# A fixed guess can silently miss real corrections just past its edge. Instead it
# walks down the ranked list until MIN_CLEAN_STREAK consecutive candidates in a row need
# no correction, which is direct evidence the true boundary has been passed rather than
# a number picked in advance and hoped to be big enough.
MIN_CLEAN_STREAK = 100 # user can alter

_require_positive_int("CHUNK", CHUNK)

# N_snps and Np are derived from the data itself, not set by hand, so there is no
# way for them to silently drift out of sync with the files they describe.
pheno_cols = pl.read_parquet(parquet_path, n_rows=0).columns[1:]
Np         = len(pheno_cols)
N_snps     = np.load(z_path, mmap_mode='r').shape[0]
N_SNPS     = N_snps            # all SNPs, no sampling

Sigma       = None
score_mm    = None
Z_mm        = None
pheno_names = None


# Stage 1: real cumulative joint test

def _init_worker(sigma, score_path_w, z_path_w, n_snps_w, np_w):
    global Sigma, score_mm, Z_mm
    Sigma    = sigma
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
    snp_scores = np.array(score_mm[idx, :], dtype=np.float64)
    z_vec      = np.array(Z_mm[idx, :],     dtype=np.float64)

    order      = np.argsort(snp_scores)[::-1]
    z_s        = z_vec[order]
    Sig_s      = Sigma[np.ix_(order, order)] + RIDGE * np.eye(Np)
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
    global Sigma, score_mm, Z_mm, pheno_names

    print("=" * 70)
    print("STAGE 1: cumulative joint test on the real toy data")
    print("=" * 70)

    print("Loading LDSC matrix...", flush=True)
    # pheno_names comes from parquet_path, NOT from the LDSC csv's own column order --
    # score_mm/Z_mm's columns follow parquet_path's phenotype order (that's what they were
    # built from), so if the LDSC csv's columns happened to be in a different order,
    # using ldsc_df.columns directly here would both permute Sigma relative to
    # score_mm/Z_mm and silently mislabel the output table's phenotype columns with
    # the wrong names, since pheno_names is also what labels out_table_parquet below.
    pheno_names = pl.read_parquet(parquet_path, n_rows=0).columns[1:]
    assert len(pheno_names) == Np, f"{len(pheno_names)} phenotype columns vs Np={Np}"
    ldsc_df = pd.read_csv(ldsc_path, index_col=0)
    assert set(ldsc_df.columns) == set(pheno_names), (
        "LDSC csv columns and parquet phenotype columns are not the same set -- "
        "check ldsc_path and parquet_path point at the same phenotype panel.")
    Sigma = ldsc_df.loc[pheno_names, pheno_names].values.astype(np.float64)
    assert Sigma.shape == (Np, Np), f"unexpected Sigma shape {Sigma.shape}"

    print("Opening score and Z memmaps...", flush=True)
    score_mm = np.memmap(score_path, dtype='float32', mode='r', shape=(N_snps, Np))
    Z_mm     = np.load(z_path, mmap_mode='r')

    snp_indices = np.arange(N_SNPS)

    print("Loading SNP IDs...", flush=True)
    snp_ids = (
        pl.scan_parquet(parquet_path).select("ID").collect().to_series().to_numpy()
    )
    assert len(snp_ids) == N_snps, f"ID count {len(snp_ids)} != N_snps {N_snps}"

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
                 initargs=(Sigma, score_path, z_path, N_snps, Np)) as pool:
        for i, r in enumerate(tqdm(
                pool.imap(_process_snp, snp_indices, chunksize=200),
                total=N_SNPS, desc="  SNPs")):
            peak_ks[i], peak_scores[i], peak_logps[i] = r[0], r[1], r[2]
            fallback_any[i], peak_fallback[i]         = r[3], r[4]
            stop_ks[i], stopped[i]                    = r[5], r[6]
            if r[7] is not None:
                kept_curves.append(r[7])

    curves = np.array(kept_curves) if kept_curves else np.empty((0, Np), dtype=np.float32)
    print(f"\n  computed in {_elapsed(t0)}", flush=True)
    print(f"  curves retained for plotting: {len(curves):,}", flush=True)

    print(f"\nNormal-approximation fallback:", flush=True)
    print(f"  fired anywhere on a curve : {int(fallback_any.sum()):,} of {N_SNPS:,}",
          flush=True)
    print(f"  fired at the PEAK value   : {int(peak_fallback.sum()):,}", flush=True)
    if peak_fallback.any():
        print("  [!] affected peaks are unreliable - recompute with mpmath:", flush=True)
        print(f"      max affected peak = {peak_logps[peak_fallback].max():.3f}",
              flush=True)
    else:
        print("  all reported peak values come from the exact chi-squared computation.",
              flush=True)

    print(f"\nEarly stopping:", flush=True)
    print(f"  triggered : {int(stopped.sum()):,} ({100*stopped.mean():.1f}%)", flush=True)
    print(f"  stop k    : mean={stop_ks.mean():.1f}, median={np.median(stop_ks):.0f}, "
          f"min={stop_ks.min()}, max={stop_ks.max()}", flush=True)

    print(f"\nPeak k         : mean={peak_ks.mean():.1f}, median={np.median(peak_ks):.0f}, "
          f"min={peak_ks.min()}, max={peak_ks.max()}", flush=True)
    print(f"Peak score     : mean={peak_scores.mean():.4f}, "
          f"median={np.median(peak_scores):.4f}", flush=True)
    qs = [50, 90, 95, 99, 99.9, 99.99]
    qv = np.percentile(peak_logps, qs)
    print(f"Peak -log10(p) :", flush=True)
    for q, v in zip(qs, qv):
        print(f"  p{q:<6} = {v:8.3f}", flush=True)
    print(f"  max    = {peak_logps.max():8.3f}", flush=True)

    # Calibration against a full null run (skipped in this toy example)
    p_emp = None
    null_max, null_logps = None, None
    if os.path.exists(null_parquet):
        null_df    = pl.read_parquet(null_parquet)
        null_logps = null_df["peak_log10p"].to_numpy()
        n_null     = len(null_logps)
        null_max   = null_logps.max()

        print(f"\n--- Calibration against {n_null:,} null SNPs ---", flush=True)
        print(f"Null peak -log10(p): median={np.median(null_logps):.3f}, "
              f"p99={np.percentile(null_logps, 99):.3f}, max={null_max:.3f}", flush=True)

        null_sorted = np.sort(null_logps)
        rank        = np.searchsorted(null_sorted, peak_logps, side='left')
        ge_count    = n_null - rank
        p_emp       = (ge_count + 1) / (n_null + 1)

        n_exceed = int((peak_logps > null_max).sum())
        exp_fp   = N_SNPS / (n_null + 1)
        print(f"\nReal SNPs exceeding the null maximum : {n_exceed:,} "
              f"({100*n_exceed/N_SNPS:.4f}% of {N_SNPS:,})", flush=True)
        print(f"Expected false positives at this threshold : {exp_fp:.2f}", flush=True)
        print(f"Enrichment : {n_exceed/max(exp_fp, 1e-12):.1f}x", flush=True)
        for thr in (0.05, 0.01, 0.001, 1.0/(n_null+1)):
            print(f"  p_emp <= {thr:.2e} : {int((p_emp <= thr).sum()):,}", flush=True)
        print(f"Resolution floor of the empirical p-value: {1.0/(n_null+1):.2e}",
              flush=True)
    else:
        print(f"\n[!] Null file not found at {null_parquet} - skipping calibration. "
              f"This toy example does not include a null run.", flush=True)

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

    # df is in original file order (snp_indices = np.arange(N_SNPS) in stage 1),
    # so the row's own position IS its index into score_mm/Z_mm.
    n_dup = df.height - df["SNP_ID"].n_unique()
    if n_dup:
        print(f"  [!] {n_dup} duplicate SNP_ID values in the output - "
              f"name-based lookup would have been unsafe; using row index instead.")

    df_indexed = df.with_row_index("row_idx")
    candidates = df_indexed.sort("peak_log10p", descending=True)
    total_candidates = candidates.height
    print(f"Checking candidates by reported peak -log10(p), widening until "
          f"{MIN_CLEAN_STREAK} consecutive candidates in a row need no correction "
          f"({total_candidates:,} candidates available)...")

    n_corrected = 0
    n_checked = 0
    corrections = {}       # row index in df -> corrected peak_log10p
    correction_ranks = []  # rank (0 = most extreme) among all candidates
    clean_streak = 0

    for rank, row in enumerate(candidates.iter_rows(named=True)):
        idx = row["row_idx"]
        k   = int(row["peak_k"])

        snp_scores = np.asarray(score_mm[idx, :], dtype=np.float64)
        z_vec      = np.asarray(Z_mm[idx, :],      dtype=np.float64)
        order      = np.argsort(snp_scores)[::-1][:k]

        Sig_k = Sigma[np.ix_(order, order)] + RIDGE * np.eye(k)
        z_k   = z_vec[order]

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

    print(f"\nChecked {n_checked:,} of {total_candidates:,} candidates")
    print(f"Corrected {n_corrected} of {n_checked:,} checked")
    if correction_ranks:
        last_rank = max(correction_ranks)
        print(f"Rank (0-indexed, by peak_log10p) of the last SNP requiring "
              f"correction: {last_rank}")
        if n_checked == total_candidates and clean_streak < MIN_CLEAN_STREAK:
            print(f"  Reached the end of all candidates before completing a "
                  f"{MIN_CLEAN_STREAK} candidate clean streak (only {clean_streak} "
                  f"available past the last correction). Every candidate has now "
                  f"been checked, so the boundary is still fully confirmed, just by "
                  f"exhaustion rather than by streak length.")
        else:
            print(f"  Boundary confirmed: {n_checked - 1 - last_rank} consecutive "
                  f"clean candidates checked past it "
                  f"(target streak: {MIN_CLEAN_STREAK}).")
    else:
        print(f"No corrections needed in the first {n_checked} candidates checked "
              f"(the most extreme SNPs). Nothing in this dataset needed correction.")

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
        n_null     = len(null_logps)
        null_max   = null_logps.max()
        null_sorted = np.sort(null_logps)

        peak_logps = df["peak_log10p"].to_numpy()
        rank       = np.searchsorted(null_sorted, peak_logps, side='left')
        p_emp      = (n_null - rank + 1) / (n_null + 1)
        df = df.with_columns(pl.Series("p_emp", p_emp))

        n_exceed = int((peak_logps > null_max).sum())
        print(f"\nCorrected: SNPs exceeding the null maximum ({null_max:.3f}): "
              f"{n_exceed:,} ({100*n_exceed/N_snps:.4f}% of {N_snps:,})")

        hits = df.filter(pl.col("peak_log10p") > null_max).sort(
            "peak_log10p", descending=True)
        hits.write_csv(out_hits_corrected, separator="\t")
        print(f"Corrected hits -> {out_hits_corrected}")
    else:
        print(f"\n[!] Null file not found at {null_parquet} - skipping calibration "
              f"recompute. This toy example does not include a null run.")

    df.write_parquet(out_parquet_corrected)
    print(f"\nCorrected summary -> {out_parquet_corrected}")

    return df


# Stage 3: final output table

def run_stage3_final_table(corrected_df):
    print("\n" + "=" * 70)
    print("STAGE 3: final output table (SNP x phenotype scores + test results)")
    print("=" * 70)
    t0 = time.time()

    snp_ids   = corrected_df["SNP_ID"].to_numpy()
    peak_k    = corrected_df["peak_k"].to_numpy().astype(np.int32)
    peak_logp = corrected_df["peak_log10p"].to_numpy().astype(np.float64)

    print("Loading joint (all-100-phenotype) p-values...", flush=True)
    pv = np.load(pvalue_matrix_npy)
    assert pv.shape[0] == N_snps, f"pvalue_matrix rows {pv.shape[0]} != {N_snps}"
    joint_p = pv[:, 1].astype(np.float64)   # column 1 = 'joint_pval' from p_values_matrix.py

    with np.errstate(divide='ignore'):
        joint_neglog10p = -np.log10(joint_p)

    zero_idx = np.where(joint_p == 0.0)[0]
    print(f"Rescuing {len(zero_idx)} underflowed joint p-values via mpmath "
          f"(none expected at this scale)...", flush=True)
    # No ridge here: must match p_values_matrix.py exactly (plain cholesky(Sigma),
    # no regularisation), since this rescues specific values of the same statistic
    # that script already computed, not a new one. Sigma is the raw COR loaded in
    # stage 1 (RIDGE is only ever added per-subset inside _process_snp), so it is
    # already exactly the matrix p_values_matrix.py used.
    L = cholesky(Sigma, lower=True, check_finite=False)
    for idx in zero_idx:
        z = np.asarray(Z_mm[idx, :], dtype=np.float64)
        y = solve_triangular(L, z, lower=True, check_finite=False)
        S = float(y @ y)
        joint_neglog10p[idx] = exact_neg_log10p(S, Np)

    print("\nAssembling the output table in chunks from the score matrix...", flush=True)
    out_cols = (["SNP_ID"] + list(pheno_names) +
                ["n_optimal_phenotypes", "neglog10p_optimal_set",
                 "neglog10p_joint_all_phenotypes"])

    parts = []
    for start in range(0, N_snps, CHUNK):
        end = min(start + CHUNK, N_snps)
        block = np.ascontiguousarray(score_mm[start:end, :])
        df = pl.from_numpy(block, schema=pheno_names, orient="row")
        df = df.with_columns([
            pl.Series("SNP_ID", snp_ids[start:end]),
            pl.Series("n_optimal_phenotypes", peak_k[start:end]),
            pl.Series("neglog10p_optimal_set", peak_logp[start:end]),
            pl.Series("neglog10p_joint_all_phenotypes", joint_neglog10p[start:end]),
        ]).select(out_cols)
        parts.append(df)
        print(f"  chunk {start:>7,}-{end:>7,}  ({_elapsed(t0)})", flush=True)

    out_df = pl.concat(parts, rechunk=False)
    assert out_df.height == N_snps
    assert out_df.width == 1 + Np + 3

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