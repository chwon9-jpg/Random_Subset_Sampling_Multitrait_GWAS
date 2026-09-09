## Instructions on the order of running the scripts to avoid errors

This directory lets you verify your own environment against **known-good expected
outputs** before pointing any script at the full-scale sparse data. Every
folder below contains one stage's script(s) plus the exact files they produced
on this toy dataset. Run a script yourself and compare your output against
what's already here; matching output means your setup is sound.

The sampling script (step 2 below) takes the LDSC intercept matrix as an
input, so LDSC derivation has to happen first in principle. In practice,
`derive_LDSC_intercept_matrix.py` is still in progress and not yet checked
into this repo, so there's no demo of step 1 to run here either.
`LDSC_intercept_matrix.csv` (see below), provided inside
`dynamic_sampling_method_example/`, is a pre-supplied convenience file, not
something any script in this folder produced, and this toy example
effectively starts at step 2 until the derivation script lands.


### The toy dataset

`dynamic_sampling_method_example/Z_matrix.npy`: 20,000 SNPs x 78 phenotypes,
real missing z-scores (~0.98% of entries NaN, averaged per SNP; this is what
"sparse" means in this pipeline's name, as opposed to the separately-maintained
0%-sparsity toy example). `dynamic_sampling_method_example/df_toy_allSNPs_allphenos.parquet`
carries the SNP IDs and phenotype column order; `LDSC_intercept_matrix.csv` in
the same folder is the pre-computed LDSC intercept matrix these 78 phenotypes
share. Every script below reads these three files directly by path — there's
no top-level convenience copy in this directory.


### The pipeline — run in this order

```
1. derive_LDSC_intercept_matrix.py (IN PROGRESS, not yet in this repo)
   LDSC_intercept_matrix.csv here is a pre-supplied convenience file, not
   produced by any script in this folder; see the caveat above
            |
            v
2. dynamic_sampling_loop.py
   (dynamic_sampling_method_example/)
            |  
            v
3. generate_score_matrix.py
   (score_matrix_example/)
            |
            +------------------------------+--------------------------------------+
            v                              v                                      v
4a. plot_discordant_SNP.py     4b. p_values_matrix.py         4c. cumulative_joint_test_null_pipeline.py
    (score_matrix_example/ —       (p_values_example/)             (cumulative_joint_testing_example/ — only
     standalone, nothing                |                           needs LDSC_intercept_matrix.csv + Z_matrix.npy,
     downstream needs it)               +--> optional:              not step 2's output)
        |                               |    contingency_table.py                 |
        |                               |    qqplot_SNPs.py                       |
        |                               |                                         |
        +-------------------------------+-----------------------------------------+
                                        |
                                        v
5.            cumulative_joint_test_pipeline_3panelfigure.py  
                                   OR
              cumulative_joint_test_pipeline_table.py
   (cumulative_joint_testing_example/ -- table variant REQUIRES 3b's pvalue_matrix.npy)
            |
            v
6. Diagnostics (optional, run after step 5):
   6a. inspect_large_k_snps.py             (standalone; needs step 5's corrected parquet)
   6b. scan_eigenvalue_contamination.py    (needs step 4c's null parquet + step 5's
                                             corrected parquet)
   6c. qqplot_optimal_subset_vs_univariate.py
       (needs the TABLE variant of step 5, step 4c's null parquet, 6b's output,
        and 4b's pvalue_matrix.npy)
```
**1) `derive_LDSC_intercept_matrix.py`** (IN PROGRESS)
Not yet checked into this repo, so there's no demo of this step in this toy
directory either. `LDSC_intercept_matrix.csv`, used throughout steps 2
onward, is a pre-supplied convenience file rather than something derived
here. Once the derivation script lands, this will demonstrate it the same
way every other step demonstrates its own script; until then, start at
step 2.

**2) `dynamic_sampling_method_example/dynamic_sampling_loop.py`**
Draws random phenotype subsets from the 78-phenotype panel and computes
`stat_matrix.dat` / `sel_matrix.npy` / `n_snp_used.npy`. Config used:
`n_workers=8, n_chunks=16, SNP_BATCH=5_000`. The data's own ~0.98% average
missingness pushed the automatically-computed subset count to `Ns=11,000`
(printed at runtime, not something you set directly). `stat_matrix.dat` and
`sel_matrix.npy` aren't kept in this folder once steps 3 and 4b have consumed
them; see the deletion prompt in step 3.

**3) `score_matrix_example/generate_score_matrix.py`**
Consumes step 1's `stat_matrix.dat` + `sel_matrix.npy`, produces
`score_matrix.dat`. Config used: `N_WORKERS=32, BLAS_THREADS=1,
SNP_BATCH=5_000, VERIFY_N_SAMPLE=10_000`. After the partial verification
passes, this script interactively offers to delete `stat_matrix.dat` /
`sel_matrix.npy`. That is why they aren't present in
`dynamic_sampling_method_example/` anymore, only `score_matrix.dat` is.

**4) Three branches:**
- **4a — `score_matrix_example/plot_discordant_SNP.py`**: standalone diagnostic
  scatter for one random SNP (seeded), off of step 3's `score_matrix.dat`.
  `score_matrix_SNProw16921_77.png` and `score_matrix_SNProw7624_78.png` here
  are two such reference plots. Nothing downstream reads its output.
- **4b — `p_values_example/p_values_matrix.py`**: consumes `stat_matrix.dat`
  directly (not `score_matrix.dat`), produces `pvalue_matrix.npy`/`.csv`.
  Config used: `N_WORKERS=8, BLAS_THREADS=1, SNP_BATCH=5_000`.
  `contingency_table.py` and `qqplot_SNPs.py` are optional visualizations of
  that output. `contingency_table_output.txt` here shows the reference
  numbers (e.g. 1,757 of 20,000 SNPs significant under all three criteria at
  p < 5e-8; 101 SNPs untested for the subset criterion).
- **4c — `cumulative_joint_testing_example/cumulative_joint_test_null_pipeline.py`**:
  generates its own null (no-signal) dataset internally, doesn't touch step
  3's output at all. Config used: `N_snps=20_000, Ns=11_000, N_CHUNKS=100,
  N_WORKERS_SUB=32, N_WORKERS_CJT=16, SNP_BATCH=5_000, SCORE_SNP_BATCH=5_000`.
  Produces `cumulative_joint_test_null.parquet`/`.png`; the calibration
  ceiling (max `peak_log10p` ≈ 11.59 on this toy run) that step 5's real
  peaks get compared against.
- **If you intend to run the table variant of step 5, you must run 4b
  first** since its `pvalue_matrix.npy` is a direct input.

**5) Pick one — `cumulative_joint_testing_example/cumulative_joint_test_pipeline_3panelfigure.py`
or `cumulative_joint_test_pipeline_table.py`**
Config used (both): `N_WORKERS=16, KEEP_EVERY=10, MIN_CLEAN_STREAK=100`
(table variant also: `CHUNK=5_000`). Both run the real cumulative joint test
on `score_matrix.dat`, then the adaptive fallback correction. On this
dataset, 222 of 20,000 SNPs needed correction, with a clean boundary at rank
221 (this is the exact dataset the `MIN_CLEAN_STREAK` comment inside these
scripts refers to). A large share of SNPs (10,721 of 20,000) exceed the null
calibration ceiling here which is expected on this particular toy panel and
includes several near-duplicate/derived phenotype pairs that inflate the
LDSC correlation structure; don't take that fraction as a general benchmark
for real data. The figure variant ends in a 3-panel summary figure
(`cumulative_joint_test_real.png`); the table variant ends in
`final_output_table.parquet` (one row per SNP, all 78 per-phenotype scores
plus `n_optimal_phenotypes`, `neglog10p_optimal_set`, and the joint-all-
observed-phenotypes statistic from 4b).

**6) Diagnostics, optional, run after step 5:**
- **6a — `inspect_large_k_snps.py`**: standalone single-SNP diagnostic.
  Auto-identifies its `TOP_N` (3 here) most extreme SNPs by `peak_log10p`
  from step 5's corrected parquet, instead of requiring them hand-picked and
  hardcoded. On this toy run: row_idx 15204 (peak_log10p=6,146.6), 4828
  (1,945.2), and 8481 (1,759.0), all `was_fallback_corrected=True`. All
  three show the same mechanism: a single near-null eigen-direction of the
  LDSC matrix, loading almost entirely on the collinear lipid panel
  (LDL_DIRECT / CHOLESTEROL / APOLIPOPROTEIN_B / HDL_CHOLESTEROL), accounts
  for 93.2%, 79.7%, and 96.1% of each SNP's joint statistic respectively.
  Output saved to `inspect_large_k_snps_output.txt`.
- **6b — `scan_eigenvalue_contamination.py`**: extends 6a's per-SNP check to
  every SNP clearing step 4c's null threshold, not just three hand-picked
  ones. Config used: `N_WORKERS=16`. Its hit filter guards against NaN
  `peak_log10p` explicitly (`is_not_nan()`), since Polars' `>` doesn't use
  IEEE-754 NaN semantics the way numpy's does; that's why it scans 10,620
  hits rather than the 10,721 step 5 reports, a 101-row gap accounted for
  entirely by SNPs with zero jointly-observed phenotypes. Of those 10,620,
  8,877 (83.59%) are flagged contaminated (`frac_small > 0.5`), rising
  sharply with `peak_k`: 3.6% contaminated at k in [0,10) versus 98.6% at k
  in [75,79) (6,968 of the 10,620 hits, the single largest band). Produces
  `eigenvalue_contamination_scan.parquet`; log saved to
  `scan_eigenvalue_contamination_output.txt`.
- **6c — `qqplot_optimal_subset_vs_univariate.py`**: compares univariate,
  joint (observed-phenotype-restricted), and optimal-subset p-values across
  all 20,000 SNPs. Requires the TABLE variant of step 5 specifically
  (`final_output_table.parquet`), plus 6b's
  `eigenvalue_contamination_scan.parquet`. On this toy run: 10,620 SNPs
  optimal-subset significant, of which 205 are "subset-specific" (significant
  only via the optimal subset, not the joint test); cross-referencing 6b's
  flag, 204 of those 205 are clean discoveries and 1 is an eigenvalue-
  degenerate artifact. Subset-specific hits have a much lower median
  `peak_k` (17) than hits significant under both the joint and optimal-
  subset criteria (76), consistent with 6b's finding that high-k hits are
  the contaminated ones. Saves `qqplot_optimal_subset_vs_univariate.png`;
  doesn't write a log file, unlike 6a and 6b.