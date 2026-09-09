## Instructions on the order of running the scripts to avoid errors

This directory holds the actual scripts for the 0% sparsity (dense, no missing
z-scores) multitrait-testing pipeline, as opposed to `toy_example_expected_outputs/`,
which lets you validate your setup against known-good outputs first. Every
script here ships with `/path/to/...` placeholder paths in its configuration
block; you fill those in with your own data before running. Unlike
`sparse_zscore_matrix/`, the `RUN_CONFIG.txt` exist only in each of the folders
in `toy_example_expected_outputs/`.

Every script here now validates its own "User to define None" values
individually, the same way `sparse_zscore_matrix/` does: leaving any one
unset, or setting a non-int or non-positive value, fails immediately with a
message naming exactly which variable is wrong, before any data is touched.
A few scripts additionally derive values that used to be filled in by hand
(`N_snps`, `Np`) directly from the data itself, so there's nothing to set
for those at all anymore. This is called out below wherever it applies.


### Before any of this: Z_matrix.npy, the phenotype parquet, and the LDSC intercept matrix

Every script below reads a z-score matrix (`Z_matrix.npy`, shape
`(N_snps, Np)`, with NO missing entries since this pipeline assumes full
observation), a parquet with an `ID` column plus one column per phenotype in
the same order (`df_allSNPs_allphenos.parquet`), and an LDSC intercept matrix
(`LDSC_intercept_matrix.csv`).

The true Step 1 of this pipeline is deriving that LDSC intercept matrix, but
that script is still in progress and not yet checked into this repo (soon to
be added). Until it lands, you need `LDSC_intercept_matrix.csv` from
somewhere else, and the pipeline below effectively starts at Step 2.

If you're starting from a plain-text z-score file rather than `Z_matrix.npy`
plus the parquet, `convert_zscore_to_npy_and_parquet.py` (repo root, main
branch) converts a csv/tsv/txt z-score matrix into both files in one pass,
auto-detecting whether the first column holds SNP IDs.

Once `derive_LDSC_intercept_matrix.py` is available, it will need
`Z_matrix.npy` plus an LD scores vector as input. `derive_ld_scores_from_reference_panel.py`
(repo root, main branch) builds that vector, one LD score per SNP in
`Z_matrix.npy`'s row order, by looking your SNP IDs up against a reference LD
score panel (1000G Phase 3 is bundled; other chromosome-prefixed panels are
supported by adjusting `FILENAME_PATTERN`). Nothing in this folder needs LD
scores directly; only the LDSC-derivation step will, once it exists.


### The pipeline: run in this order

```
1. derive_LDSC_intercept_matrix.py   (IN PROGRESS, not yet in this folder)
   produces LDSC_intercept_matrix.csv
        |
        v
2. sampling_loop.py
   (sampling_method/)
        |
        v
3. generate_score_matrix.py
   (score_matrix/)
        |
        +--------------------------------+----------------------------------------+
        v                                v                                        v
4a. plot_discordant_SNP.py   4b. p_values_matrix.py         4c. cumulative_joint_test_null_pipeline.py
    (score_matrix/, standalone)    (p_values/)                    (cumulative_joint_testing/, only needs
        |                               |                            LDSC_intercept_matrix.csv and the
        |                               +--> optional:                parquet, not step 3's output)
        |                               |    contingency_table.py                 |
        |                               |    qqplot_SNPs.py                       |
        |                               |                                         |
        +-------------------------------+-----------------------------------------+
                                        |
                                        v
5. cumulative_joint_test_pipeline_3panelfigure.py   (pick one)
   OR
   cumulative_joint_test_pipeline_table.py
   (cumulative_joint_testing/, table variant requires 4b's pvalue_matrix.npy)
```

**1) `derive_LDSC_intercept_matrix.py`** (IN PROGRESS)
Not yet in this folder. Will derive `LDSC_intercept_matrix.csv` from
`Z_matrix.npy` and an LD scores vector (see `derive_ld_scores_from_reference_panel.py`
above). Everything below needs its output; until it's added, supply
`LDSC_intercept_matrix.csv` yourself.

**2) `sampling_method/sampling_loop.py`**
Draws random phenotype subsets and computes `stat_matrix.dat` and
`sel_matrix.npy` from `Z_matrix.npy` and `LDSC_intercept_matrix.csv`. Unlike
the sparse pipeline, the number of subsets is fixed at `Ns = 10,000` rather
than computed from measured missingness, since this dataset has none. Subset
sizes are drawn uniformly between 5 and 90 phenotypes; that upper bound is
hardcoded rather than derived from `Np`, so even with all 100 phenotypes
available, no subset larger than 90 phenotypes is ever drawn.

**3) `score_matrix/generate_score_matrix.py`**
Consumes step 2's `stat_matrix.dat` and `sel_matrix.npy`, produces
`score_matrix.dat`. The sample size for the post-run partial verification is
its own named, validated config variable, `VERIFY_SAMPLE_SIZE` (it's
min-ed against `N_snps`, so any value works even on small data).

**4) Three branches:**
- **4a) `score_matrix/plot_discordant_SNP.py`**: a standalone diagnostic
  scatter for one random SNP (seeded), built off of step 3's
  `score_matrix.dat`. `N_snps` and `Np` are derived from `Z_matrix.npy`'s
  shape and the phenotype parquet's columns, with a `SystemExit` if
  `score_matrix.dat`'s file size or `Z_matrix.npy`'s column count don't
  match. There is nothing to set by hand here anymore. Its `parquet_path`
  placeholder reads `df_toy_allSNPs_allphenos.parquet`, unlike every other
  script in this directory, which expects `df_allSNPs_allphenos.parquet`;
  point it at whichever file you actually have. Its output filename is also
  fixed (`discordant_SNP_profile.png`); it does not embed the sampled SNP's
  index the way the sparse pipeline's version does.
- **4b) `p_values/p_values_matrix.py`**: consumes `stat_matrix.dat` directly,
  not `score_matrix.dat`, and produces `pvalue_matrix.npy`/`.csv` with three
  columns (univariate, joint across all phenotypes, and minimum subset
  p-values). There is no missing data handling here (no NaN masking, no
  observed phenotype counts), since the dense pipeline assumes every
  phenotype is observed for every SNP. `contingency_table.py` and
  `qqplot_SNPs.py` are optional visualizations of that output.
  `contingency_table.py` now saves the same table it prints to
  `contingency_table_output.txt`, matching the sparse pipeline's version.
  `qqplot_SNPs.py`'s joint series legend now just reads "Joint (all
  phenotypes)", with no dataset specific phenotype count hardcoded into it.
- **4c) `cumulative_joint_testing/cumulative_joint_test_null_pipeline.py`**:
  generates its own null dataset (no true signal) internally, and does not
  touch step 3's output at all. `Np` is derived from the phenotype parquet's
  columns, not set by hand. `N_snps` still has no natural source to derive
  from here (this stage doesn't read a real `Z_matrix.npy` at all, unlike
  the sparse pipeline's null script, so there's no real SNP count to match
  against for calibration resolution). It's still a plain "User to define"
  choice, just now validated like everything else. `N_WORKERS_SUB` (Stage
  B, joblib threads) is checked first and drives `POLARS_MAX_THREADS`;
  `N_WORKERS_CJT` (Stage D, `mp.Pool`) is intentionally independent, not
  derived from `N_WORKERS_SUB`: the toy run's own recorded config uses
  `N_WORKERS_SUB=8, N_WORKERS_CJT=16`, so don't assume one should track the
  other. `N_snps`, `N_CHUNKS`, `N_WORKERS_SUB`, `N_WORKERS_CJT`, and
  `SNP_BATCH` are all now validated individually up front.
- **If you intend to run the table variant of step 5, you must run 4b
  first**, since its `pvalue_matrix.npy` is a direct input.

**5) Pick one: `cumulative_joint_testing/cumulative_joint_test_pipeline_3panelfigure.py`
or `cumulative_joint_test_pipeline_table.py`**
Both run the real cumulative joint test on `score_matrix.dat`, then a
fallback correction. Both now use the same adaptive `MIN_CLEAN_STREAK`
search as the sparse pipeline: rather than checking a fixed number of top
candidates, which can silently miss a real correction just past its edge,
they walk down the ranked list until `MIN_CLEAN_STREAK` (100) consecutive
candidates in a row need no correction at all, direct evidence the true
boundary has been passed rather than a number picked in advance and hoped
to be big enough. `N_WORKERS` is validated individually and drives
`POLARS_MAX_THREADS` (no more `"USER_TO_DEFINE"` literal to hand-edit); like
step 4a, `N_snps` and `Np` are derived from `Z_matrix.npy` and the
phenotype parquet rather than set by hand. The table variant's `CHUNK` is
likewise validated individually. The figure variant ends in a 3 panel
summary figure; the table variant ends in `final_output_table.parquet`
(one row per SNP, all per phenotype scores plus `n_optimal_phenotypes`,
`neglog10p_optimal_set`, and the joint p-value across all phenotypes from
4b) and additionally requires 4b's `pvalue_matrix.npy`.


### Every script's paths are placeholders

Every script in this directory is checked in with `/path/to/...` placeholder
paths in its configuration block, not real values. You fill those in
yourself before running.
