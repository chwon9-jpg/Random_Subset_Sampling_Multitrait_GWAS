'''
Picks a single random SNP and scatters that one SNP's scores/|z| values across 
the 100 phenotypes — it's per-SNP, not per-phenotype, despite the similar-looking scatter+OLS code.
'''

import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

score_path   = "/path/to/score_matrix.dat"
z_path       = "/path/to/Z_matrix.npy"
parquet_path = "/path/to/df_toy_allSNPs_allphenos.parquet"
out_path     = "/path/to/discordant_SNP_profile.png"

N_snps = 20_000 
Np     = 100

np.random.seed(17)
snp_idx = np.random.randint(0, N_snps)
print(f"Selected SNP index: {snp_idx:,}")

pheno_cols = pl.read_parquet(parquet_path, n_rows=0).columns[1:]

score = np.memmap(score_path, dtype='float32', mode='r', shape=(N_snps, Np))
Z     = np.load(z_path, mmap_mode='r')

snp_scores = np.array(score[snp_idx, :], dtype=np.float64)
snp_abs_z  = np.abs(np.array(Z[snp_idx, :], dtype=np.float64))

snp_id = (
    pl.scan_parquet(parquet_path)
    .select("ID")
    .slice(snp_idx, 1)
    .collect()
    .item()
)
print(f"SNP ID: {snp_id}")

# Scatter plot
slope, intercept, r, p, _ = stats.linregress(snp_scores, snp_abs_z)
x_line = np.linspace(snp_scores.min(), snp_scores.max(), 300)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')

ax.scatter(snp_scores, snp_abs_z,
           s=14, alpha=0.65, color='steelblue', edgecolors='none',
           label='100 phenotypes', zorder=3)
ax.plot(x_line, y_line, color='firebrick', lw=2.0, zorder=4,
        label=f'OLS fit  ($r$ = {r:.3f})')

ax.set_xlabel('Phenotype score  ($S_1$)', fontsize=12)
ax.set_ylabel('$|z|$', fontsize=12)
ax.set_title(
    f'Phenotype score vs. |z| across 100 phenotypes\nSNP: {snp_id}',
    fontsize=12, fontweight='bold', pad=10)
ax.legend(fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
print(f"Saved → {out_path}")
