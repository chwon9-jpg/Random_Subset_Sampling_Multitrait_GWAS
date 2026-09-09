import sys
import numpy as np
import pandas as pd
import polars as pl
from scipy.linalg import cholesky, solve_triangular

base_out   = "/path/to/output/directory"
score_path = "/path/to/score_matrix.dat"
z_path     = "/path/to/Z_matrix.npy"
ldsc_path  = "/path/to/LDSC_intercept_matrix.csv"
save_path  = "/path/to/df_allSNPs_allphenos.parquet"
corrected  = f"{base_out}/cumulative_joint_test_real_corrected.parquet" # Make sure base_out is set correctly

N_snps, Np = np.load(z_path, mmap_mode='r').shape[0], pl.read_parquet(save_path, n_rows=0).width - 1
RIDGE_MARGIN = 1e-4
GWS_Z        = 5.45     # |z| at p = 5e-8
TOP_N        = None        # User to define how many of the most extreme SNPs (by peak_log10p) to inspect
log_path     = f"{base_out}/inspect_large_k_snps_output.txt"


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


if __name__ == '__main__':
    _stdout, _stderr = sys.stdout, sys.stderr
    log_file = open(log_path, "w")
    sys.stdout = _Tee(_stdout, log_file)
    sys.stderr = _Tee(_stderr, log_file)

    print("Loading LDSC matrix, phenotype columns, and corrected output parquet...", flush=True)
    pheno_cols = pl.read_parquet(save_path, n_rows=0).columns[1:]
    COR_df         = pd.read_csv(ldsc_path, sep='\t', index_col=0)
    COR_df.index   = COR_df.index.str.removeprefix('z_')
    COR_df.columns = COR_df.columns.str.removeprefix('z_')
    Sigma = COR_df.loc[pheno_cols, pheno_cols].values.astype(np.float64)
    assert Sigma.shape == (Np, Np)

    lambda_min_full = np.linalg.eigvalsh(Sigma).min()
    RIDGE = max(1e-4, RIDGE_MARGIN - lambda_min_full)
    print(f"Full 78x78 Sigma: min eigenvalue = {lambda_min_full:.6f}  ->  RIDGE = {RIDGE:.6f} "
          f"(matches what the real pipeline run used)")

    df = pl.read_parquet(corrected).with_row_index("row_idx")
    score_mm = np.memmap(score_path, dtype='float32', mode='r', shape=(N_snps, Np))
    Z_mm     = np.load(z_path, mmap_mode='r')

    ev_full = np.linalg.eigvalsh(Sigma)
    print(f"\nFull 78x78 LDSC matrix: min eigenvalue = {ev_full.min():.4e}, max = {ev_full.max():.2f}, "
          f"condition number = {ev_full.max()/max(abs(ev_full.min()),1e-12):.3e}")
    print(f"  eigenvalues below 1e-2: {(ev_full < 1e-2).sum()} of {Np}")

    # Identify the TOP_N most extreme SNPs by peak_log10p, sorting the corrected parquet's
    # non-NaN rows. peak_log10p is NaN for SNPs with zero jointly-observed phenotypes (see
    # cumulative_joint_test_pipeline_table.py), and Polars' `>` doesn't use IEEE-754 NaN
    # semantics the way numpy's does, so is_not_nan() is filtered explicitly rather than
    # trusting a bare descending sort to push NaN rows to the bottom.
    extreme = (df.filter(pl.col("peak_log10p").is_not_nan())
                 .sort("peak_log10p", descending=True)
                 .head(TOP_N))
    TARGETS = [{"row_idx": int(r["row_idx"]), "peak_k": int(r["peak_k"])}
               for r in extreme.iter_rows(named=True)]

    print(f"\nTop {TOP_N} most extreme SNPs by peak_log10p:")
    for r in extreme.iter_rows(named=True):
        print(f"  row_idx={r['row_idx']:<10} peak_k={r['peak_k']:<4} "
              f"peak_log10p={r['peak_log10p']:>12.3f}  "
              f"was_fallback_corrected={bool(r['was_fallback_corrected'])}")

    for t in TARGETS:
        idx, k = t["row_idx"], t["peak_k"]
        row = df.filter(pl.col("row_idx") == idx)
        print(f"\n{'='*72}")
        print(f"row_idx={idx}   peak_k={k}")
        print(f"  reported peak -log10(p) = {row['peak_log10p'][0]:.3f}   "
              f"was_fallback_corrected = {bool(row['was_fallback_corrected'][0])}")
        print('='*72)

        # Restrict to jointly-observed phenotypes, exactly like _process_snp
        snp_scores_full = np.asarray(score_mm[idx, :], dtype=np.float64)
        z_vec_full      = np.asarray(Z_mm[idx, :], dtype=np.float64)
        observed = ~np.isnan(snp_scores_full) & ~np.isnan(z_vec_full)
        obs_idx  = np.where(observed)[0]
        m = len(obs_idx)
        print(f"  jointly-observed phenotypes for this SNP: {m} of {Np}")

        scores_obs = snp_scores_full[obs_idx]
        z_obs      = z_vec_full[obs_idx]
        Sig_obs    = Sigma[np.ix_(obs_idx, obs_idx)]

        rank_order = np.argsort(scores_obs)[::-1][:k]      # top-k by score, matching the real run
        z_k        = z_obs[rank_order]
        pheno_k    = [pheno_cols[obs_idx[i]] for i in rank_order]

        # 1. Marginal z-scores
        print(f"\n1. MARGINAL z-SCORES (within the peak subset of {k} phenotypes)")
        print(f"   max|z|          = {np.abs(z_k).max():.4f}")
        print(f"   mean|z|         = {np.abs(z_k).mean():.4f}")
        print(f"   |z| > 1.96      = {(np.abs(z_k) > 1.96).sum()} of {k} "
              f"(expected ~{0.05*k:.0f} under H0)")
        print(f"   |z| > {GWS_Z}     = {(np.abs(z_k) > GWS_Z).sum()}  "
              f"(genome-wide univariate threshold)")
        sum_z2 = float(z_k @ z_k)
        print(f"   sum z^2         = {sum_z2:.1f}   (expected ~{k} if z ~ N(0,I))")
        print(f"   ratio to expect = {sum_z2/k:.2f}x")

        # 2. Joint statistic
        Sig_sub = Sigma[np.ix_(obs_idx, obs_idx)][np.ix_(rank_order, rank_order)] + RIDGE * np.eye(k)
        ev_k    = np.linalg.eigvalsh(Sig_sub)
        L       = cholesky(Sig_sub, lower=True, check_finite=False)
        Y       = solve_triangular(L, z_k, lower=True, check_finite=False)
        S       = float(Y @ Y)

        print(f"\n2. JOINT STATISTIC")
        print(f"   subset eigenvalues: min={ev_k.min():.4e}, max={ev_k.max():.2f}")
        print(f"   S = ||L^-1 z||^2  = {S:.1f}   (expected ~{k} under H0)")
        print(f"   inflation S/k     = {S/k:.2f}x")
        print(f"   for reference, sum z^2 / k = {sum_z2/k:.2f}x")
        print(f"   -> the joint test amplifies the raw signal by a further "
              f"{S/sum_z2:.2f}x")

        # 3. Concentration across components
        comp = Y**2
        srt  = np.sort(comp)[::-1]
        print(f"\n3. CONCENTRATION OF S ACROSS ITS {k} COMPONENTS")
        print(f"   largest single component : {srt[0]:.1f}  "
              f"({100*srt[0]/S:.1f}% of S)")
        print(f"   top 5 components         : {100*srt[:5].sum()/S:.1f}% of S")
        print(f"   top 10 components        : {100*srt[:10].sum()/S:.1f}% of S")
        print(f"   under H0, the largest of {k} chi2_1 draws is typically ~"
              f"{2*np.log(k):.1f}")
        print(f"   -> largest component is {srt[0]/(2*np.log(k)):.1f}x that")

        # 4. Do large contributions align with small eigenvalues? 
        evals, evecs = np.linalg.eigh(Sig_sub)
        proj    = evecs.T @ z_k
        contrib = proj**2 / evals

        top5 = np.argsort(contrib)[::-1][:5]
        print(f"\n4. AMPLIFICATION BY EIGEN-DIRECTION")
        print(f"   contribution of direction i to S is (v_i . z)^2 / lambda_i")
        print(f"   {'rank':>4} {'lambda':>12} {'(v.z)^2':>12} {'contrib':>12} {'% of S':>8}")
        for r, i in enumerate(top5):
            print(f"   {r:>4} {evals[i]:>12.4e} {proj[i]**2:>12.4f} "
                  f"{contrib[i]:>12.1f} {100*contrib[i]/S:>7.1f}%")

        frac_small = contrib[evals < 1e-2].sum() / S
        n_small    = int((evals < 1e-2).sum())
        print(f"\n   directions with lambda < 1e-2: {n_small} of {k}, "
              f"contributing {100*frac_small:.1f}% of S")
        print(f"   (if these few directions carry a large share of S, the statistic")
        print(f"    is driven by near-degenerate directions rather than by broad signal)")

        # 5. Which phenotypes make up the smallest eigen-direction 
        smallest_dir = evecs[:, np.argmin(evals)]
        top_loadings = np.argsort(np.abs(smallest_dir))[::-1][:4]
        print(f"\n5. PHENOTYPES LOADING MOST HEAVILY ON THE SMALLEST EIGEN-DIRECTION "
              f"(lambda={evals.min():.4e})")
        for i in top_loadings:
            print(f"   {pheno_k[i]:<55} loading={smallest_dir[i]:+.3f}")

    print(f"\nLog saved -> {log_path}")
    sys.stdout, sys.stderr = _stdout, _stderr
    log_file.close()