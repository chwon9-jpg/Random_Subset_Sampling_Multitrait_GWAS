# Random Subset Sampling for Multitrait GWAS

This repository accompanies the article *Random Subset Sampling for Multitrait
GWAS: A Scalable Alternative to Exhaustive Phenotype Testing*, written during
Christopher Won's internship. It contains two sets of scripts, corresponding to
two versions of the input data: a z-score matrix with no missing entries
(`0%_sparsity_zscore_matrix/`) and one with missing entries (`sparse_zscore_matrix/`).

**Only `0%_sparsity_zscore_matrix/` was used to produce the results and figures in
the article.** The sparse case is implemented and functional, but its results are
not reported in the article; within the 3-month internship there was time to run
the pipeline but not to write up a second set of results alongside the first.

Genome wide association studies normally test one SNP against one phenotype
at a time. When phenotypes are correlated, as with the 1,010 brain MRI
phenotypes analyzed in the article, a SNP with a moderate coordinated effect
across many of them can fail to reach significance in any single univariate
test, yet be strongly supported when those phenotypes are tested jointly.
Testing every phenotype together in one joint test avoids that problem but
reintroduces the dilution it was meant to solve, since a signal carried by
only a small group of related phenotypes gets averaged away once every
phenotype is included. Exhaustively testing every possible phenotype subset
is combinatorially infeasible (roughly 10^304 subsets for 1,010
phenotypes). The method implemented here instead draws Ns = 10,000 random
phenotype subsets, tests each jointly using the LDSC intercept matrix as the
null covariance, and rescales every subset's statistic onto a common
chi squared, 1 degree of freedom scale so that subsets of different sizes
stay comparable. From there, a per phenotype score attributes each subset's
joint signal back to individual phenotypes, and a cumulative joint test
ranks phenotypes by that score and tests nested prefixes of the ranking,
calibrated against a null run of the same size, to find SNPs whose peak
signal cannot be explained by the ranking procedure alone. Applied at full scale to 9,611,261 SNPs across all 1,010 phenotypes, the method
finds 666 SNPs significant under subset-based testing alone and 173,033 SNPs whose
cumulative joint signal cannot be attributed to the ranking procedure, with the
two independently-constructed results cross-validating each other. See the
abstract (Article_Random_Subset_Sampling_for_Multitrait_GWAS.pdf) for full detail.

The exact stages (sampling, score matrix, p values, LDSC intercept
derivation, cumulative joint testing) and the order to run them in are
documented in each folder's own README, not repeated here. Start with the
`READ_FIRST.md` (or `README_toy_example.md`, inside a toy folder) in whichever
folder you're working in.


## Repository layout

Two independent, parallel pipeline implementations, depending on whether your
z score matrix has missing entries:

- `sparse_zscore_matrix/`: for real data with missing z scores. Pairwise
  complete throughout. This is the more actively maintained implementation;
  see its `READ_FIRST.md` for the order to run things in, and its
  `script_descriptions.md` for a detailed walkthrough of what each script
  actually computes and how, phenotype by phenotype and stage by stage,
  beyond just the run order.
- `0% _sparsity_zscore_matrix/`: for data with no missing z scores. Simpler
  and faster since there is no missingness bookkeeping. Configuration
  validation and the fallback correction step's adaptive search now match
  the sparse pipeline throughout: every worker count and batch size is
  checked individually before anything runs, with a message naming exactly
  which one is wrong, and both cumulative joint testing scripts widen their
  candidate window until a clean streak confirms the correction boundary
  rather than checking a fixed guessed count. One small leftover remains, a
  path placeholder naming inconsistency in `plot_discordant_SNP.py`; its
  `READ_FIRST.md` documents it explicitly.

Both directories are organized the same way internally: a sampling stage, a
score matrix stage, a p values stage, an LDSC intercept derivation stage (to be added), and
a cumulative joint testing stage. Each also ships its own
`toy_example_expected_outputs/` folder, a small dataset already run through
the whole pipeline, so you can diff your own run's output against a known
good reference before pointing any script at real data.


## Repo root scripts

Two scripts live at the top level because both pipelines depend on them
before any folder specific script can run:

- `convert_zscore_to_npy_and_parquet.py`: converts a plain text z score
  matrix (csv, tsv, or txt) into the `Z_matrix.npy` plus
  `df_allSNPs_allphenos.parquet` pair every downstream script expects.
  Automatically detects whether the first column holds SNP IDs.
- `derive_ld_scores_from_reference_panel.py`: builds the `ld_scores.npy`
  vector, one LD score per SNP in your `Z_matrix.npy`'s row order, by looking
  your SNP IDs up against a reference LD score panel. This is what
  `derive_LDSC_intercept_matrix.py` needs as its LD score input. 1000G
  Phase 3 is bundled; eur_w_ld_chr and other chromosome prefixed panels are
  supported by adjusting `FILENAME_PATTERN`.


## Requirements

Python 3.14, with dependencies pinned in `requirements.txt`:
`pip install -r requirements.txt`.

This repository also uses [Git LFS](https://git-lfs.com/) to store the large
binary files in the `toy_example_expected_outputs/` folders (`.dat`, `.npy`,
and `.parquet`, some several hundred MB each). Install `git-lfs` before
cloning, then run `git lfs install` once per machine. If you already cloned
this repository without `git-lfs` installed, those files will show up as
small text pointers instead of real data; install `git-lfs` and run
`git lfs pull` from inside the repository to fetch the actual content.
`git lfs install` only needs to be run once per machine, not once per repo
or per file: any `.dat`, `.npy`, or `.parquet` file added here later, by
anyone, is picked up automatically with no extra steps, as long as
`git-lfs` has been installed on that machine at some point. Only a genuinely
new large file type not already covered by `.gitattributes` would need a new
`git lfs track` rule added.


## Status

`derive_LDSC_intercept_matrix.py` is still in progress in both pipelines;
each `READ_FIRST.md` and `README_toy_example.md` says so explicitly where it
matters. Everything else described in a folder's own README reflects the
current, working behavior of that folder's scripts.