import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_out    = "/path/to/output/directory"
final_path  = f"{base_out}/final_output_table.parquet"
null_path   = f"{base_out}/cumulative_joint_test_null.parquet"
pv_path     = "/path/to/pvalue_matrix.npy"
contam_path = f"{base_out}/eigenvalue_contamination_scan.parquet"
fig_path    = f"{base_out}/qqplot_optimal_subset_vs_univariate.png"

GWS = -np.log10(5e-8)

if __name__ == '__main__':
    df = pl.scan_parquet(final_path).select(
        ["n_optimal_phenotypes", "neglog10p_optimal_set", "neglog10p_joint_observed_phenotypes"]
    ).collect().with_row_index("row_idx")

    pv = np.load(pv_path)
    with np.errstate(divide='ignore'):
        uni_neglog10p = -np.log10(pv[:, 0])

    opt_k      = df["n_optimal_phenotypes"].to_numpy()
    opt_logp   = df["neglog10p_optimal_set"].to_numpy()
    joint_logp = df["neglog10p_joint_observed_phenotypes"].to_numpy()
    row_idx    = df["row_idx"].to_numpy()
    N          = df.height

    null_max = float(pl.read_parquet(null_path)["peak_log10p"].max())
    print(f"Null max (optimal-subset significance threshold): {null_max:.6f}")

    uni_sig   = uni_neglog10p > GWS
    joint_sig = joint_logp > GWS
    opt_sig   = opt_logp > null_max
    print(f"optimal-subset significant: {opt_sig.sum():,}")

    cats = {
        "None significant":              (~uni_sig) & (~opt_sig) & (~joint_sig),
        "Univariate only":                 uni_sig  & (~opt_sig) & (~joint_sig),
        "Joint only":                    (~uni_sig) & (~opt_sig) &   joint_sig,
        "Optimal subset only":           (~uni_sig) &   opt_sig  & (~joint_sig),
        "Univariate + Joint":              uni_sig  & (~opt_sig) &   joint_sig,
        "Univariate + Optimal subset":     uni_sig  &   opt_sig  & (~joint_sig),
        "Joint + Optimal subset":       (~uni_sig) &   opt_sig  &   joint_sig,
        "All three":                       uni_sig  &   opt_sig  &   joint_sig,
    }
    print("\nContingency table")
    for k, v in cats.items():
        c = int(v.sum())
        print(f"  {k:<30} {c:>10,} ({100*c/N:.4f}%)")

    subset_specific      = opt_sig & (~joint_sig)
    subset_specific_only = subset_specific & (~uni_sig)
    print(f"\nSubset-specific (optimal-sig, joint-not-sig): {subset_specific.sum():,}")
    print(f"  of these, not univariate-sig either:         {subset_specific_only.sum():,}")
    print(f"  of these, also univariate-sig:                {(subset_specific & uni_sig).sum():,}")

    contam_rows = set(
        pl.read_parquet(contam_path).filter(pl.col("frac_small") > 0.5)["row_idx"].to_list()
    )
    ss_rows = set(row_idx[subset_specific].tolist())
    n_contam_in_ss = len(contam_rows & ss_rows)
    print(f"\nOf the {subset_specific.sum():,} subset-specific SNPs, "
          f"{n_contam_in_ss} are eigenvalue-degenerate artifacts "
          f"(scan_eigenvalue_contamination.py's frac_small>0.5 flag) -> "
          f"{subset_specific.sum() - n_contam_in_ss:,} clean subset-specific discoveries.")

    print(f"\nn_optimal_phenotypes (peak k):")
    print(f"  subset-specific (median):        {np.nanmedian(opt_k[subset_specific]):.0f}")
    print(f"  joint+optimal both sig (median): {np.nanmedian(opt_k[opt_sig & joint_sig]):.0f}")

    joint_gt_opt = joint_logp > opt_logp
    print(f"\nneglog10p_joint_observed_phenotypes > neglog10p_optimal_set: "
          f"{joint_gt_opt.sum():,} ({100*joint_gt_opt.sum()/N:.4f}%)")

    # --- Figure ---
    rng = np.random.default_rng(0)
    sample_size = min(400_000, N)
    sample_idx = rng.choice(N, size=sample_size, replace=False)
    plot_idx = np.union1d(sample_idx, np.where(opt_sig)[0])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(uni_neglog10p[plot_idx], joint_logp[plot_idx], s=3, alpha=0.2,
               color='tab:blue', label='Joint (observed phenotypes)', rasterized=True)
    ax.scatter(uni_neglog10p[plot_idx], opt_logp[plot_idx], s=3, alpha=0.2,
               color='tab:red', label='Optimal subset (cumulative test)', rasterized=True)
    ax.plot([0, 300], [0, 300], 'k--', lw=1, label='y = x')
    ax.axhline(GWS, color='gray', linestyle=':', lw=1, label='GWS threshold (p<5e-08)')
    ax.axhline(null_max, color='darkred', linestyle=':', lw=1,
               label=f'Null max ({null_max:.2f}), optimal-subset criterion')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_xlabel(r'$-\log_{10}(p_{univariate})$')
    ax.set_ylabel(r'$-\log_{10}(p)$')
    ax.set_title('Joint and optimal-subset $-\\log_{10}(p)$ vs univariate $-\\log_{10}(p)$ (all SNPs)')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"\nSaved -> {fig_path}")
