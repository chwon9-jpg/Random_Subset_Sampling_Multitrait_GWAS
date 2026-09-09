## Instructions on the order of running the scripts to avoid errors

This directory holds the actual scripts for the sparse (real missing z-scores)
multitrait-testing pipeline as opposed to `toy_example_expected_outputs/`,
which lets you validate your setup against known-good outputs first. Every
script here ships with `/path/to/...` placeholder paths in its configuration
block; you fill those in with your own data before running. Each folder below
also has its own `RUN_CONFIG.txt` with the fuller detail (exact config notes,
output file shapes/dtypes). This file is just the map of how the folders fit
together and in what order.

Every "User to define None" numeric config value (worker counts, batch sizes,
etc.) is validated before anything else runs: leaving one as `None`, setting a
non-int, or a non-positive value all fail immediately with a message naming
exactly which variable is wrong; not a crash buried deep inside a worker
process. Fill in the whole configuration block before running, not just enough
to get past the first error.


### Before any of this: Z_matrix.npy, the phenotype parquet, and the LDSC intercept matrix

Every script below reads a z-score matrix (`Z_matrix.npy`, shape `(N_snps, Np)`,
`NaN` for missing entries), a Parquet with an `ID` column plus one column
per phenotype in the same order (`df_allSNPs_allphenos.parquet`), and an LDSC
intercept matrix (`LDSC_intercept_matrix.csv`).

The true Step 1 of this pipeline is deriving that LDSC intercept matrix, but
that script is still in progress and not yet checked into this repo (soon to
be added). Until it lands, you need `LDSC_intercept_matrix.csv` from
somewhere else, and the pipeline below effectively starts at Step 2.

If you're starting from a plain-text z-score file rather than `Z_matrix.npy`
plus the parquet, `convert_zscore_to_npy_and_parquet.py` (repo root) converts
a csv/tsv/txt z-score matrix into both files in one pass, auto-detecting
whether the first column holds SNP IDs. Nothing in this directory creates
them from scratch.

Once `derive_LDSC_intercept_matrix.py` is available, it will need
`Z_matrix.npy` plus an LD scores vector as input. `derive_ld_scores_from_reference_panel.py`
(repo root) builds that vector, one LD score per SNP in `Z_matrix.npy`'s row
order, by looking your SNP IDs up against a reference LD score panel (1000G
Phase 3 is bundled; other chromosome-prefixed panels are supported by
adjusting `FILENAME_PATTERN`). Nothing in this directory needs LD scores
directly; only the LDSC-derivation step will, once it exists.


### The full pipeline: run in this order

```
1. derive_LDSC_intercept_matrix.py (IN PROGRESS, not yet in this repo)
   will be pairwise-complete, so missing entries in Z_matrix.npy are fine;
   needs Z_matrix.npy + an LD scores .npy file (see above)
        |
        v
2. dynamic_sampling_loop.py
   (dynamic_sampling_method/: needs Z_matrix.npy, the parquet, and
    LDSC_intercept_matrix.csv from step 1)
        |
        +-------------------------------+-----------------------------------------+
        v                               v                                         v
3a. generate_score_matrix.py   3b. p_values_matrix.py        3c. cumulative_joint_test_null_pipeline.py
    (score_matrix/)                 (p_values/)                    (cumulative_joint_testing/: only needs
        |                               |                           step 1's LDSC matrix + Z_matrix, not
        v                               +--> optional:              step 2's output; the calibration
    (optional)                          |    contingency_table.py   result is the actual point of step 4,
    plot_discordant_SNP.py              |    qqplot_SNPs.py         but it isn't a hard prerequisite)
    (standalone, nothing                |                                         |
     downstream needs it)               |                                         |
        |                               |                                         |
        +-------------------------------+-----------------------------------------+
                        | 
                        v
4. cumulative_joint_test_pipeline_3panelfigure.py   (pick ONE)
   OR
   cumulative_joint_test_pipeline_table.py
   (cumulative_joint_testing/: both need 3a's score_matrix.dat; the table
    variant additionally REQUIRES 3b's pvalue_matrix.npy; both pick up 3c's
    null parquet automatically if present, for the calibration reference)
                  |
                  v
5. Diagnostics (optional), run after step 4:
   inspect_large_k_snps.py            (standalone; needs step 4's corrected parquet)
   scan_eigenvalue_contamination.py   (needs step 3c's null parquet + step 4's
                                        corrected parquet)
   qqplot_optimal_subset_vs_univariate.py
       (needs the TABLE variant of step 4, step 3c's null parquet, the scan
        script's output above, and 3b's pvalue_matrix.npy)
```

**1) `derive_LDSC_intercept_matrix.py`** (IN PROGRESS)
Not yet checked into this repo, so there's nothing to run for this step yet.
Once available, it will derive the LDSC intercept matrix
(`LDSC_intercept_matrix.csv`) from your z-score matrix and an LD scores
vector, pairwise-complete, so missing entries in `Z_matrix.npy` are fine.
Everything downstream needs this file; until it lands, supply
`LDSC_intercept_matrix.csv` yourself and start at step 2.

**2) `dynamic_sampling_method/dynamic_sampling_loop.py`**
Draws random phenotype subsets and computes `stat_matrix.dat` / `sel_matrix.npy`
/ `n_snp_used.npy` from `Z_matrix.npy` and step 1's LDSC matrix. The number of
subsets (`Ns`) is computed automatically from the data's own measured
missingness, not something you set directly.

**3) Three branches off step 2's output (and step 1's, for 3c):**
- **3a) `score_matrix/generate_score_matrix.py`**: consumes `stat_matrix.dat`
  + `sel_matrix.npy`, produces `score_matrix.dat`. `score_matrix/plot_discordant_SNP.py`
  is an optional standalone diagnostic plot off of its output; nothing later
  reads it.
- **3b) `p_values/p_values_matrix.py`**: consumes `stat_matrix.dat` directly
  (not `score_matrix.dat`), produces `pvalue_matrix.npy`/`.csv` (univariate,
  joint-all-phenotype, and min-subset p-values per SNP).
  `p_values/contingency_table.py` and `p_values/qqplot_SNPs.py` are optional
  visualizations of that output.
- **3c) `cumulative_joint_testing/cumulative_joint_test_null_pipeline.py`**:
  generates its own null (no-signal) dataset internally and writes
  `cumulative_joint_test_null.parquet`/`.png`: the calibration ceiling step
  4's real peaks get judged against. Only needs step 1's LDSC matrix and
  `Z_matrix.npy`; doesn't touch step 2's output at all. Not a hard
  prerequisite for step 4 (it's skipped automatically if missing), but run it
  anyway, since the calibration is the actual point of this pipeline.
- **If you intend to run the table variant of step 4, you must run 3b first**:
  its `pvalue_matrix.npy` is a direct input. If you only intend to run the
  figure variant, step 3b is optional.
  

**4) Pick one: `cumulative_joint_testing/cumulative_joint_test_pipeline_3panelfigure.py`
or `cumulative_joint_testing/cumulative_joint_test_pipeline_table.py`**
Both run the real cumulative joint test on `score_matrix.dat` (3a), then an
adaptive fallback correction (walks the ranked candidates until
`MIN_CLEAN_STREAK` consecutive ones need no correction, not a fixed guessed
cutoff). The figure variant ends in a 3-panel summary figure; the table
variant ends in `final_output_table.parquet` (one row per SNP, all
per-phenotype scores plus the optimal-subset and joint-all-phenotype
statistics) and additionally requires 3b's `pvalue_matrix.npy`.

**5) Diagnostics, optional, run after step 4 (all in `cumulative_joint_testing/`):**
- `inspect_large_k_snps.py`: standalone single-SNP diagnostic. Needs step 4's
  `cumulative_joint_test_real_corrected.parquet` (either variant writes it),
  plus the same `Z_matrix.npy` / `score_matrix.dat` / LDSC matrix / phenotype
  parquet everything else uses. Auto-identifies its own `TOP_N` most extreme
  SNPs by `peak_log10p` from the corrected parquet, instead of requiring
  them hand-picked and hardcoded.
- `scan_eigenvalue_contamination.py`: extends that same per-SNP check to
  every SNP clearing step 3c's null threshold, not just a handful. Needs
  step 3c's `cumulative_joint_test_null.parquet` and step 4's corrected
  parquet. Writes `eigenvalue_contamination_scan.parquet`.
- `qqplot_optimal_subset_vs_univariate.py`: compares univariate, joint, and
  optimal-subset p-values across all SNPs. Needs the TABLE variant's
  `final_output_table.parquet` specifically (the figure variant doesn't
  produce that file), the scan script's output above, and 3b's
  `pvalue_matrix.npy`. Writes `qqplot_optimal_subset_vs_univariate.png`.


### Every script's paths are placeholders

Every script in this directory is checked in with `/path/to/...` placeholder
paths in its configuration block, not real values. You fill those in
yourself before running.