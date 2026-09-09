import os
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

score_path = "/path/to/score_matrix.dat"
z_path     = "/path/to/Z_matrix.npy"
save_path  = "/path/to/df_allSNPs_allphenos.parquet"
out_dir    = "/path/to/output/directory"

pheno_cols = pl.read_parquet(save_path, n_rows=0).columns[1:]
Np = len(pheno_cols)

Z = np.load(z_path, mmap_mode='r')
N_snps = Z.shape[0]
if Z.shape[1] != Np:
    raise SystemExit(f"{Z.shape[1]} columns in {z_path} vs {Np} phenotype columns in {save_path}.")

_nbytes = os.path.getsize(score_path)
if _nbytes != N_snps * Np * 4:
    raise SystemExit(f"{score_path}: {_nbytes:,} B doesn't match N_snps={N_snps:,} x Np={Np} float32.")
score = np.memmap(score_path, dtype='float32', mode='r', shape=(N_snps, Np))

np.random.seed(7)
snp_idx = np.random.randint(0, N_snps)
# snp_idx = 34
print(f"Selected SNP index: {snp_idx:,}")

out_path = f"{out_dir}/score_matrix_SNProw{snp_idx}.png"

snp_scores = np.array(score[snp_idx, :], dtype=np.float64)
snp_abs_z  = np.abs(np.array(Z[snp_idx, :], dtype=np.float64))

valid = ~np.isnan(snp_scores) & ~np.isnan(snp_abs_z)
n_valid = int(valid.sum())
if n_valid < Np:
    missing_phenos = [pheno_cols[i] for i in range(Np) if not valid[i]]
    print(f"{Np - n_valid} of {Np} phenotypes excluded for this SNP "
          f"(missing z-score and/or score): {missing_phenos}")
if n_valid < 2:
    raise SystemExit(f"Only {n_valid} valid phenotype(s) for SNP index {snp_idx} -- "
                      f"not enough to fit a regression. Try a different seed.")

snp_scores = snp_scores[valid]
snp_abs_z  = snp_abs_z[valid]

snp_id = f"row {snp_idx:,}"
print(f"SNP: {snp_id} ({n_valid}/{Np} phenotypes plotted)")

# Scatter plot
slope, intercept, r, p, _ = stats.linregress(snp_scores, snp_abs_z)
x_line = np.linspace(snp_scores.min(), snp_scores.max(), 300)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')

ax.scatter(snp_scores, snp_abs_z,
           s=14, alpha=0.65, color='steelblue', edgecolors='none',
           label=f'{n_valid} phenotypes', zorder=3)
ax.plot(x_line, y_line, color='firebrick', lw=2.0, zorder=4,
        label=f'OLS fit  ($r$ = {r:.3f})')

ax.set_xlabel('Phenotype score  ($S_1$)', fontsize=12)
ax.set_ylabel('$|z|$', fontsize=12)
ax.set_title(
    f'Phenotype score vs. |z| across {n_valid} of {Np} phenotypes\nSNP: {snp_id}',
    fontsize=12, fontweight='bold', pad=10)
ax.legend(fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
print(f"Saved -> {out_path}")