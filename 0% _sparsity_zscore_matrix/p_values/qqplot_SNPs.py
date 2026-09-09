import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir   = "/path/to/p_values"
THRESHOLD = 5e-8

mat   = np.load(f"{out_dir}/pvalue_matrix.npy")
order = np.argsort(mat[:, 0])
mat   = mat[order]

log_uni    = -np.log10(np.clip(mat[:, 0], 1e-300, 1.0))
log_joint  = -np.log10(np.clip(mat[:, 1], 1e-300, 1.0))
log_subset = -np.log10(np.clip(mat[:, 2], 1e-300, 1.0))

print(f"Plotting all {len(mat):,} SNPs")
plot_mask = np.ones(len(mat), dtype=bool)

gws     = -np.log10(THRESHOLD)
max_val = max(log_uni[plot_mask].max(), log_joint[plot_mask].max(), log_subset[plot_mask].max())

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(log_uni[plot_mask], log_joint[plot_mask],
           s=1, alpha=0.3, color='steelblue', label='Joint (all phenotypes)', rasterized=True)
ax.scatter(log_uni[plot_mask], log_subset[plot_mask],
           s=1, alpha=0.3, color='tomato', label='Min subset (Bonferroni)', rasterized=True)

ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, label='y = x')
ax.axvline(gws, color='grey', linestyle=':', linewidth=0.8, label=f'GWS threshold (p < {THRESHOLD})')
ax.axhline(gws, color='grey', linestyle=':', linewidth=0.8)

ax.set_xlabel(r"$-\log_{10}(p_{\mathrm{uni}})$", fontsize=13)
ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=13)
ax.set_title("Joint and subset p-values vs univariate p-value (all SNPs)", fontsize=13)
ax.legend(fontsize=10, markerscale=6)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = f"{out_dir}/qqplot_allSNPs.pdf"
plt.savefig(out_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved: {out_path}")
