## Instructions on the order of running the scripts to avoid errors

This directory lets the user verify their own environment against **known-good expected
outputs** before pointing any script at the full-scale data. Every folder below
contains the script(s) it demonstrates plus a `RUN_CONFIG.txt`: the exact
configuration used, the exact printed output, and every file the script produced,
with sizes. Users run a script and diff their output against its `RUN_CONFIG.txt`;
if they match, their setup is sound.

The sampling script (step 2 below) takes the LDSC intercept matrix as an input, so
LDSC derivation has to happen first in principle. In practice, `derive_LDSC_intercept_matrix.py`
is still in progress and not yet checked into this repo, so there's no demo of
step 1 to run here either. `LDSC_toy_matrix.csv` (see below) is a pre-supplied
convenience file, not something any script in this folder produced, and this
toy example effectively starts at step 2 until the derivation script lands.


### Shared toy data

`LDSC_toy_matrix.csv` and `zscore_matrix_20KSNPs_100phenotypes.npy` at the top level
of this directory are convenience copies of the same files used throughout the
pipeline below (`sampling_method_expected_output/Z_matrix.npy` for the z-score
matrix; `LDSC_toy_matrix.csv` has no separate copy inside `sampling_method_expected_output/`,
it's read straight from here). Both are identical content, just easier to find
without digging into a subfolder.


### The full pipeline: run in this order

```
1. derive_LDSC_intercept_matrix.py (IN PROGRESS, not yet in this repo)
   LDSC_toy_matrix.csv here is a pre-supplied convenience file, not
   produced by any script in this folder; see the caveat above
        |
        v
2. sampling_script_toy_run.py
        |
        v
3. generate_score_matrix_toy_run.py
        |
        +-------------------------------------------+
        v                                           v
4a. p_values_matrix_toy_run.py           4b. plot_discordant_SNP.py
    (+ optional: contingency_table.py,       (score_vs_|z|/: standalone,
     qqplot_20k_SNPs.py)                      nothing downstream needs it)
        |
        | (required only if the user plan to run 6b)
        v
5. cumulative_joint_test_null_pipeline_toy_run.py
   (not a hard prerequisite for step 6; those scripts skip calibration
   automatically if this hasn't been run, but run it first anyway,
   since the calibration result is the actual point of step 6)
        |
        +----------------------------------------------------------+
        v                                                          v
6a. cumulative_joint_test_pipeline_toy_run       6b. cumulative_joint_test_pipeline_toy_run
    _3panelfigure.py                                 _table.py
    (ends in the 3-panel summary figure)             (ends in a per-SNP table; REQUIRES
                                                        4a's pvalue_matrix.npy as input)
```

**1) `derive_LDSC_intercept_matrix.py`** (IN PROGRESS)
Not yet checked into this repo, so there's no demo of this step in this toy
directory either. `LDSC_toy_matrix.csv`, used throughout steps 2 onward, is a
pre-supplied convenience file rather than something derived here. Once the
derivation script lands, this will demonstrate it the same way every other
step demonstrates its own script; until then, start at step 2.

**2) `sampling_method_expected_output/sampling_script_toy_run.py`**
Draws the 10,000 random phenotype subsets and computes `stat_matrix.dat` /
`sel_matrix.npy` from the toy z-score matrix and LDSC intercept matrix (the
top-level `LDSC_toy_matrix.csv`).

**3) `score_matrix_expected_output/generate_score_matrix/generate_score_matrix_toy_run.py`**
Consumes `stat_matrix.dat` + `sel_matrix.npy` from step 2, produces `score_matrix.dat`.

**4) Two independent, optional branches off of step 3's output:**
- **4a) `p_values_expected_output/compute_p_values/p_values_matrix_toy_run.py`**:
  produces `pvalue_matrix.npy`/`.csv` (univariate, joint-all-100, and min-subset
  p-values per SNP). `contingency_table/contingency_table.py` and
  `qqplot/qqplot_20k_SNPs.py` are optional visualizations of that output: run
  either, both, or neither.
- **4b) `score_matrix_expected_output/score_vs_|z|/plot_discordant_SNP.py`**: a
  standalone diagnostic scatter plot for one random SNP. Nothing later in the
  pipeline reads its output, so it's safe to skip entirely if the user is not
  interested in it.
- **If the user intend to run step 6b (the table variant) later, the user must run 4a's
  `p_values_matrix_toy_run.py`**: its `pvalue_matrix.npy` is a direct input to 6b.
  If the user only intend to run 6a (the figure variant), step 4 is entirely optional.

**5) `cumulative_joint_testing_expected_output/cumulative_joint_test_null_pipeline_toy_run.py`**
Generates a null (no-signal) dataset with the same phenotype correlation structure,
runs the identical sampling loop and cumulative joint test on it, and writes
`cumulative_joint_test_null_toy.parquet`/`.png`: the calibration ceiling that the
real peaks in step 6 are judged against. Takes ~2 minutes (dominated by its own
sampling loop); everything else in this pipeline finishes in seconds.

**6) Pick one: running both just repeats steps 2 and 3 of their shared work:**
- **6a) `cumulative_joint_test_pipeline_toy_run_3panelfigure.py`**: real cumulative
  joint test, then fallback correction, then a final 3-panel figure (with the null
  reference lines from step 5, if it was run). Reports how many toy SNPs (230 of
  20,000, in the reference run) exceed the null ceiling: signal that selection
  alone can't explain.
- **6b) `cumulative_joint_test_pipeline_toy_run_table.py`**: same real cumulative
  joint test and fallback correction as 6a, but ends in
  `final_output_table_toy.parquet`: one row per SNP with all 100 per-phenotype
  scores plus `n_optimal_phenotypes`, `neglog10p_optimal_set`, and
  `neglog10p_joint_all_phenotypes` (the last of which comes from 4a's
  `pvalue_matrix.npy`, joint_pval column).