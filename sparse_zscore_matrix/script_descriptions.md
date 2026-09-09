# Script descriptions: sparse_zscore_matrix

This file goes one level deeper than `READ_FIRST.md` and each folder's
`RUN_CONFIG.txt`. Those tell you the order to run things in and which
config values to set; this file explains what each script actually
computes and how it gets there, for anyone who wants to understand the
method itself, not just operate it. It covers every script under
`sparse_zscore_matrix/`, excluding `toy_example_expected_outputs/`, in
pipeline order.

The core idea running through all of it: testing every phenotype jointly
against every SNP at once is the obvious thing to try, but it dilutes a
signal that is only carried by a handful of related phenotypes, since
adding a hundred unrelated ones to the same test averages the real signal
away. Testing phenotypes one at a time avoids dilution but a SNP with a
moderate coordinated effect across several correlated phenotypes can fail
every single univariate test. The method here threads that gap: draw many
random phenotype subsets, test each one jointly, and use the pattern of
which subsets score well to rank phenotypes for each SNP individually.
Then, for each SNP, test the top ranked phenotype alone, the top two
together, the top three, and so on, and take whichever prefix of that
ranking gives the strongest joint statistic. A null run calibrates how
strong that peak needs to be before it counts as more than what the
ranking and selection procedure would produce from noise alone.


## Before the pipeline: `derive_LDSC_intercept_matrix.py`

Not yet in this repo (see `READ_FIRST.md`). Once available it will derive
the LDSC intercept matrix every script below treats as a fixed input, a
correlation like matrix `Sigma` describing how correlated two phenotypes'
summary statistics are purely from sample overlap and shared population
structure, independent of any real genetic effect. That matrix is what
every random subset test, every joint test, and the cumulative test's own
covariance structure are all built on.


## `dynamic_sampling_method/dynamic_sampling_loop.py`

Draws `Ns` random phenotype subsets and, for each one, tests every SNP
jointly against that subset, producing `stat_matrix.dat` (one rescaled
statistic per subset per SNP) and `sel_matrix.npy` (which phenotypes each
subset contained).

The LDSC matrix is loaded tab separated with its `z_` prefixed labels
stripped, then reordered to match the phenotype parquet's column order
exactly, since a mismatch there would silently permute `Sigma` relative to
the z score matrix it is meant to describe. Its minimum eigenvalue is
checked, and if negative (estimation noise rather than a real problem), a
ridge is added to the whole matrix's diagonal, sized so the ridged
matrix's minimum eigenvalue clears a small positive margin
(`RIDGE_MARGIN`, `1e-4`). By Cauchy's interlacing theorem, fixing the
minimum eigenvalue of the full matrix this way guarantees every principal
submatrix built from it (that is, every random subset's own restricted
covariance matrix) is positive definite too, so no later Cholesky
decomposition on any subset can fail for lack of this fix.

`Ns` itself is not a fixed number. It is computed from the data's own
average per SNP missingness (`sparsity`, the fraction of phenotypes that
are `NaN`, averaged across all SNPs): a sparser matrix means more SNPs get
excluded from any given subset (a subset only tests a SNP if that SNP has
non missing z scores for every phenotype the subset drew), so more subsets
are needed to reach a comparably stable score per phenotype. `Ns` scales
linearly between 1.1 times a base count of 10,000 at 20% sparsity or below
and 1.5 times that base count at 80% sparsity or above, flat outside that
band.

For each subset, drawn independently and in parallel across `n_workers`
threads split into `n_chunks` reproducible chunks: a random size `k`
between 5 and `Np` phenotypes is chosen, then `k` phenotypes are drawn
without replacement. The Cholesky factor of the LDSC matrix restricted to
those `k` phenotypes is computed once. Then, in `SNP_BATCH` sized blocks,
every SNP is checked for whether all `k` selected phenotypes are observed
for it; SNPs missing even one are excluded from this subset entirely
(their statistic stays `NaN`). For the SNPs that pass, the z vector
restricted to the subset is decorrelated by solving the triangular system
against the Cholesky factor, and the sum of squares of the result is a
chi squared statistic with `k` degrees of freedom under the null that
there is no coordinated signal in this subset.

That statistic is then rescaled onto a common chi squared, one degree of
freedom footing: its p value under chi squared with `k` degrees of freedom
is computed, then that same p value is converted back into whatever
statistic a chi squared, one degree of freedom test would have produced
for it. This rescaling is the whole reason subsets of very different sizes
can later be pooled together in `generate_score_matrix.py`: a strong
result from a five phenotype subset and a strong result from a
seventy phenotype subset land on the same numeric scale once rescaled,
even though their raw statistics started on completely different scales.


## `score_matrix/generate_score_matrix.py`

Turns the Ns times N_snps grid of per subset statistics into one
per SNP, per phenotype score: `score_matrix.dat`, shape N_snps by Np. This
score is what the cumulative joint test later ranks a SNP's phenotypes by.

For a given SNP and phenotype, the score is the mean, across every subset
that both contained that phenotype and was not excluded for missingness on
that SNP, of the subset's rescaled statistic for that SNP. In other words,
each subset's joint significance gets attributed back proportionally to
every phenotype that took part in it, and averaging across many random
subsets turns that attribution into a stable per phenotype signal. The
implementation computes this as two matrix products per SNP batch: a
numerator (the sum of statistics across subsets containing each
phenotype, with missing entries zeroed first so they cannot contribute)
and a denominator (the count of subsets that both contained that phenotype
and were valid for that SNP), then divides. A phenotype that was never
both selected and valid for a given SNP produces `0/0`, which becomes
`NaN` by design rather than an arbitrary numeric default.

Work is split by SNP range across `N_WORKERS` processes, each further
batching its own range into `SNP_BATCH` sized pieces to bound memory. Once
finished, the script samples `VERIFY_N_SAMPLE` evenly spaced rows of the
completed `score_matrix.dat` and reports the NaN excluded min, max, and
mean, what fraction of sampled entries are `NaN` (expected to roughly
track the z score matrix's own missingness rate), how many sampled rows
are entirely zero (never expected, worth investigating if it happens at
all), and how many are entirely `NaN` (a SNP with no valid score for any
phenotype at all, unusual but not automatically a bug). If the check
passes, it offers to delete `stat_matrix.dat` and `sel_matrix.npy`
interactively, since neither is needed again once `score_matrix.dat`
exists; run without an attached terminal (for example under a batch
scheduler), it leaves both files in place rather than guessing.


## `score_matrix/plot_discordant_SNP.py`

A standalone sanity check, not a pipeline dependency: picks one SNP at
random (seed fixed at 7) and plots its per phenotype score against its per
phenotype absolute z value, with an ordinary least squares fit line. The
expectation this plot tests visually is simple: a phenotype with a large
absolute z for this SNP should tend to have gotten a higher score too,
since a phenotype genuinely associated with a SNP will more often land in
the small, significant subsets that drive its score up. A scatter with no
relationship at all, or a strongly negative one, would be a sign something
upstream is broken.

`N_snps` and `Np` are read directly from `Z_matrix.npy`'s shape and the
phenotype parquet's column count, with an immediate error if
`score_matrix.dat`'s file size does not match those dimensions, so a
mismatched or stale intermediate file is caught here rather than producing
a confusing failure later. Phenotypes with a missing score or missing z
for the sampled SNP are excluded from the plot and named in the console
output; at least two valid phenotypes are required to fit a line at all.
The output file name embeds the sampled SNP's row index, so repeated runs
with different seeds do not overwrite one another.


## `p_values/p_values_matrix.py`

Computes three per SNP p values directly and exactly, with no random
subsets involved, as a fast and independent point of comparison for
everything downstream that does use subsets.

The first, `min_univariate_pval`, is simply the smallest single phenotype
p value for that SNP, from a standard one degree of freedom test on each
phenotype's z score independently; missing phenotypes are dropped from the
minimum rather than treated as anything else.

The second, `joint_pval`, tests every phenotype that SNP actually has
observed, jointly, all at once, using a chi squared test whose degrees of
freedom equal however many phenotypes that specific SNP has (recorded
alongside it as `n_pheno_observed`). Because missingness patterns repeat
heavily within a batch of SNPs (most are either fully observed or missing
the same one or two high missingness phenotypes), SNPs are grouped by
their exact observed phenotype pattern before this test runs, so each
distinct pattern gets one Cholesky decomposition and one batched solve
covering every SNP that shares it, rather than repeating that work once
per SNP.

The third, `min_subset_pval_bonf`, reuses the random subset statistics
already computed in `stat_matrix.dat`: it takes the single most extreme
rescaled statistic any subset produced for that SNP, converts it to a p
value, and applies a Bonferroni correction by multiplying by however many
subsets actually tested that SNP (`n_subsets_used`), not a flat count of
all `Ns` subsets, since a SNP excluded from many subsets by missingness
should only be penalized for the ones that actually included it.

The LDSC matrix here gets the same tab separated parsing, `z_` prefix
stripping, and data driven ridge described above, so its Cholesky factor
and every restricted submatrix built from it are guaranteed valid.


## `p_values/contingency_table.py`

Cross tabulates the three `p_values_matrix.py` criteria at a genome wide
significance threshold (`p < 5e-8` by default), reporting how many SNPs
fall into each of the eight combinations: none significant, each criterion
alone, each pair together, and all three at once. Separately reports how
many SNPs were untested (`NaN`) under each criterion, since those are
folded into "not significant" in the main counts but are a genuinely
different situation from "tested and found not significant." Prints to the
console and saves the identical text to `contingency_table_output.txt`.


## `p_values/qqplot_SNPs.py`

Plots joint and subset `-log10(p)` against univariate `-log10(p)` for
every SNP, with a `y = x` reference line and the genome wide threshold
marked on both axes, so points sitting well above the line or the
threshold on the joint or subset axis but not the univariate one are
visually the SNPs those tests are finding that a univariate scan alone
would miss. `NaN` p values (a SNP untested for a given criterion, most
often the subset one, if it was simply never validly drawn into any
subset) are left in place rather than dropped beforehand; matplotlib
omits them from the scatter automatically, and the axis bounds are
computed with `nanmax` specifically, since an ordinary maximum would
propagate a single `NaN` into the whole figure's axis range.


## `cumulative_joint_testing/cumulative_joint_test_null_pipeline.py`

Builds the null reference the two real data pipeline scripts below are
calibrated against. Because the cumulative joint test always reports
whichever prefix of the ranked phenotype list gave the strongest result,
its peak is inflated even under pure noise, simply from having many chances
to get lucky; comparing a real SNP's peak against a fixed textbook
threshold would overstate how surprising it is. This script instead
simulates SNPs with no true signal at all, runs the identical pipeline on
them, and records how extreme their peaks get purely from that selection
effect.

Four stages, run in order:

Stage A generates `N_snps` synthetic z vectors by drawing from `N(0,
Sigma)` directly (a standard normal vector transformed by the LDSC
matrix's own Cholesky factor), then injects missingness to match the real
data: for each synthetic SNP, a real SNP is sampled at random and its
exact missing phenotype pattern is copied onto the synthetic z vector.
Without this step the null calibration would always test the full
phenotype panel, while real SNPs often have fewer observed phenotypes, and
fewer available terms changes the peak `-log10(p)` distribution even under
a genuine null, biasing the calibration.

Stage B runs the same random subset sampling loop as
`dynamic_sampling_loop.py`, on this synthetic z matrix, producing a null
`stat_matrix.dat` and `sel_matrix.npy`.

Stage C aggregates those into a null `score_matrix.dat`, by the same
averaging logic as `generate_score_matrix.py`.

Stage D runs the cumulative joint test itself on the null score and z
matrices, in parallel across a process pool. For each simulated SNP: its
jointly observed phenotypes are found, ranked by score in descending
order, and the joint statistic is built up one phenotype at a time via an
incremental Cholesky update, adding each newly ranked phenotype's
contribution without recomputing the whole factorization from scratch.
`-log10(p)` is recorded at every step, and the running best value is
tracked. Once past a minimum of `EARLY_STOP_MIN_K` (5) phenotypes, if the
current value has fallen more than `EARLY_STOP_TOL` (30%) below the best
value seen so far, the loop stops early, since a curve that has already
declined that much rarely climbs back to a new peak, which saves
substantial compute across the whole run. The exact chi squared survival
function is used unless it underflows to zero in floating point, in which
case a normal approximation is used instead and that fact is recorded.
Restricting to jointly observed phenotypes has to happen before ranking,
not after, because `numpy`'s sort places `NaN` values at the end when
ascending, so reversing that order for a descending ranking would put
`NaN` scored (unobserved) phenotypes first if they were not excluded
beforehand, corrupting the whole curve starting from the very first step.

The result is `cumulative_joint_test_null.parquet` (one row per simulated
SNP) and a three panel figure summarizing the peak curves, how many
simulated SNPs are still contributing at each rank, and the distribution
of peak ranks. The single largest `peak_log10p` value in that parquet is
the calibration ceiling both real data scripts below compare their own
results against.


## `cumulative_joint_testing/cumulative_joint_test_pipeline_3panelfigure.py` and `cumulative_joint_test_pipeline_table.py`

These two scripts share their first two stages exactly and differ only in
the third.

Stage 1 runs the same incremental cumulative joint test described above
for the null pipeline's Stage D, but on the real `score_matrix.dat` and
`Z_matrix.npy` instead of simulated ones. In addition to tracking whether
the normal approximation fallback fired anywhere on a SNP's curve, it
separately tracks whether it fired specifically at the reported peak,
since a peak reached through that fallback is considered unreliable and
gets recomputed exactly in Stage 2 rather than trusted as is. If a null
parquet from the null pipeline is present, an empirical p value is
computed for every real SNP by its rank against the null distribution, and
the script reports how many real SNPs exceed the null's own maximum,
compares that to how many would be expected by chance alone at that
threshold, and writes the SNPs above the ceiling to a tab separated file.

Stage 2 addresses the same underflow issue from a different angle: the
normal approximation fallback used in Stage 1 is only reliable when it
does not actually need to be used, so every reported peak near the
underflow boundary (in double precision floating point, close to a
`peak_log10p` of about 307.65) deserves scrutiny, but only the most
extreme candidates are actually at risk. Rather than checking a fixed,
guessed number of the top candidates, which can silently miss a real
correction just past its edge, this stage walks down the full list sorted
by reported peak value, recomputing each one's exact value at eighty
significant digits of precision via `mpmath` (immune to the double
precision underflow that motivated the fallback in the first place), and
stops once `MIN_CLEAN_STREAK` (100) consecutive candidates in a row needed
no correction at all. That streak is direct evidence the true boundary
between values that needed correction and values that did not has
actually been passed, rather than a number chosen in advance and hoped to
be large enough. Empirical p values and the hit list are then recomputed
against the null distribution using these corrected peak values.

Stage 3, in the figure variant, draws a three panel summary from the
corrected data: the peak curves' mean, median, and interquartile range
(for whichever SNPs had their full curve retained, one in every
`KEEP_EVERY`), with the null distribution's own median and maximum
overlaid as reference lines when available; how many SNPs are still
contributing to the curve at each rank; and the corrected distribution of
peak ranks across every SNP.

Stage 3, in the table variant, instead assembles one row per SNP
containing every one of that SNP's per phenotype scores, its corrected
`n_optimal_phenotypes` and `neglog10p_optimal_set` from Stage 2, and a
directly comparable joint statistic pulled from `p_values_matrix.py`'s
output: `n_pheno_observed_joint` and `neglog10p_joint_observed_phenotypes`,
so the optimal subset result sits next to the plain joint result for the
same SNP in the same row. Any joint p value that itself underflowed to
exactly zero in `pvalue_matrix.npy` is individually rescued using the same
high precision construction `p_values_matrix.py`'s joint column used: the
same restricted and ridged covariance matrix, the same degrees of freedom
equal to that SNP's own observed phenotype count, checked against an
assertion that the recomputed count matches what was stored, which would
otherwise catch a mismatched z score matrix between the two scripts
immediately rather than silently mixing incompatible data. The table is
assembled in `CHUNK` sized blocks of SNPs to bound memory while writing.


## `cumulative_joint_testing/inspect_large_k_snps.py`

A hands on diagnostic for a small number of individual SNPs, meant to
answer a question the summary tables cannot: for a SNP with an extreme
reported peak, is that peak driven by broad coordinated signal across many
phenotypes, or is it an artifact of amplification through a nearly
singular direction of the covariance matrix.

Its `TOP_N` most extreme SNPs are identified automatically by sorting the
corrected parquet's non `NaN` `peak_log10p` values in descending order,
rather than requiring them picked out and hardcoded by hand. Since Polars'
comparison operators do not follow IEEE 754 `NaN` semantics the way
`numpy`'s do, the sort is preceded by an explicit filter for non `NaN`
rows rather than trusting a bare sort to push them to one end.

For each target SNP, the script reconstructs precisely the subset and
ranking the real pipeline used for it (restrict to jointly observed
phenotypes, rank by score descending, take the top `peak_k`), then reports
five things in turn. First, the marginal z scores within that subset:
maximum and mean absolute value, how many clear the nominal and genome
wide significance thresholds, and the sum of squared z values relative to
its expectation under the null, a first read on whether there is broad
univariate signal here at all. Second, the joint statistic itself,
computed through an eigendecomposition rather than a Cholesky
factorization specifically so its value can be broken down by direction
afterward, along with how much it is inflated relative to its degrees of
freedom and how much further amplification the joint test adds beyond
what the raw sum of squared z values alone would suggest. Third, how
concentrated that statistic is: what share comes from its single largest
component, compared against what the largest of that many independent
chi squared draws would typically look like under the null. Fourth,
whether the largest contributions align with small eigenvalues of the
subset's covariance matrix, since the contribution of each eigen direction
is its projected z value squared divided by its own eigenvalue, so a
direction with a tiny eigenvalue gets amplified enormously by even a small
projection onto it, real signal or not; this reports what share of the
statistic comes from directions below an eigenvalue of `1e-2`, the exact
same threshold `scan_eigenvalue_contamination.py` applies at scale. Fifth,
which phenotypes load most heavily onto the single smallest eigen
direction, since that direction is the one most likely to be amplifying
noise rather than reflecting genuine coordinated signal, and its loadings
often reveal a cluster of biochemically redundant phenotypes rather than
anything biologically new.


## `cumulative_joint_testing/scan_eigenvalue_contamination.py`

Extends the fourth check from `inspect_large_k_snps.py`, the small
eigenvalue attribution, from a handful of hand picked SNPs to every SNP
that cleared the null threshold, giving a dataset wide picture of how much
of the optimal subset test's discoveries reflect broad genuine signal
versus amplification through near singular covariance directions.

Hits are found by filtering the corrected parquet to `peak_log10p` above
the null's own maximum, with an explicit `is_not_nan()` guard on that
comparison. This matters because Polars' `>` operator does not follow
IEEE 754 `NaN` semantics the way `numpy`'s does: a bare comparison would
silently pull SNPs with zero jointly observed phenotypes into the hit set,
and the eigendecomposition on their empty subset would then crash the
whole scan partway through.

For each hit, the script reconstructs that SNP's own peak subset exactly
as the real pipeline built it, then computes the same eigendecomposition
based breakdown as `inspect_large_k_snps.py`: `frac_small`, the share of
the joint statistic coming from directions with eigenvalue below
`SMALL_EV` (`1e-2`), and `top1_frac`, the share from the single largest
contributing direction alone. A SNP is flagged contaminated if
`frac_small` exceeds `CONTAM_FRAC` (0.5), meaning the majority of its
joint statistic traces back to nearly degenerate directions rather than
broad signal spread across many phenotypes. The ridge added to the
covariance matrix is computed once from the full matrix's minimum
eigenvalue, the same data driven construction used everywhere else in this
pipeline, which by Cauchy interlacing guarantees every subset's own
restricted covariance matrix stays positive definite, so the
eigendecomposition never runs into a spuriously negative eigenvalue from
numerical noise alone.

The scan runs in parallel across `N_WORKERS` processes and writes
`eigenvalue_contamination_scan.parquet`, one row per hit, joined back to
its `SNP_ID`, `peak_log10p`, and empirical p value. The printed summary
reports the overall contamination rate, the percentiles of `frac_small`
across all hits, and the contamination rate broken down by band of
`peak_k`, since contamination concentrates heavily among SNPs whose
optimal subset includes most or all of the phenotype panel: the closer a
subset gets to the full panel, the more likely it is to include whatever
small group of phenotypes happens to form a near singular direction of the
covariance matrix.


## `cumulative_joint_testing/qqplot_optimal_subset_vs_univariate.py`

The headline comparison: does the optimal subset cumulative test surface
real signal that the univariate and full joint tests miss, and how much of
whatever it finds survives the contamination check above.

For every SNP, three `-log10(p)` values are compared: the univariate value
(the strongest single phenotype association, taken from
`pvalue_matrix.npy`'s first column), the joint value (restricted to that
SNP's own observed phenotypes, taken from `final_output_table.parquet`,
itself sourced from `pvalue_matrix.npy`'s second column when the table was
built), and the optimal subset value (the cumulative test's own corrected
peak, also from `final_output_table.parquet`). Significance for the
univariate and joint columns uses the standard genome wide threshold,
`p < 5e-8`. Significance for the optimal subset column instead uses the
null pipeline's own empirical maximum, since that peak is inflated by
exactly the selection effect the null calibration exists to correct for,
and applying the standard threshold directly to it would flag the
overwhelming majority of all SNPs as significant.

An eight way contingency table is built from these three criteria, the
same structure as `contingency_table.py` but for these three columns
instead. The script then isolates "subset specific" SNPs: significant
under the optimal subset criterion but not the joint one, the strongest
evidence available that the ranking and early stopping procedure is
finding something a fixed, all phenotypes joint test dilutes away. Those
subset specific SNPs are cross referenced against
`scan_eigenvalue_contamination.py`'s own output by row index, so the
headline count of subset specific discoveries is immediately split into
how many are contamination artifacts and how many are clean.

The resulting figure scatters joint and optimal subset `-log10(p)` against
univariate `-log10(p)`, with a `y = x` line, the genome wide threshold, and
the null calibrated optimal subset threshold all marked. The background
sample of points is capped at `min(400_000, N)` so the figure stays
tractable even on datasets with far more than 400,000 SNPs, while every
significant optimal subset point is still guaranteed to appear regardless
of that cap.
