######################################################################################################
#                           User Configuration required
# These are None on purpose: set them for the reference panel and z-score matrix
# being used rather than inheriting values from the last run.
#
#   REFERENCE_PANEL_DIR : directory holding one LD score file per chromosome.
#   FILENAME_PATTERN    : the per-chromosome filename, with "{chrom}" where the
#                          chromosome number goes. Different reference panel
#                          distributions name these differently. Adjust to match
#                          whichever panel you're pointing at:
#                            1000G Phase 3 (as bundled in this repo) : "LDscore.{chrom}.l2.ldscore.gz"
#                            eur_w_ld_chr (standard LDSC download)   : "{chrom}.l2.ldscore.gz"
#                            chr-prefixed panels                     : "chr{chrom}.l2.ldscore.gz"
#   CHROMS               : which chromosome numbers to read. Autosomes 1-22 by
#                           default; narrow this if your panel only covers some.
#   SNP_ID_SOURCE        : the SNP ID list defining your Z_matrix.npy's row order. This is
#                          either a Parquet file with an "ID" column (the
#                          df_allSNPs_allphenos.parquet convention used elsewhere
#                          in this repo), or a plain .txt file, one SNP ID per line.
#   OUTPUT_PATH          : where ld_scores.npy is written.
#######################################################################################################


REFERENCE_PANEL_DIR = "/path/to/reference_panel/LDscore_dir"
FILENAME_PATTERN    = "LDscore.{chrom}.l2.ldscore.gz"
CHROMS              = list(range(1, 23))
SNP_ID_SOURCE       = "/path/to/df_allSNPs_allphenos.parquet"
OUTPUT_PATH         = "/path/to/ld_scores.npy"

if None in (REFERENCE_PANEL_DIR, SNP_ID_SOURCE, OUTPUT_PATH):
    raise SystemExit("Set REFERENCE_PANEL_DIR, SNP_ID_SOURCE and OUTPUT_PATH in "
                      "the configuration block above.")

import os
import numpy as np
import polars as pl

# Step 1: your SNP ID list, in Z_matrix.npy's row order
if SNP_ID_SOURCE.lower().endswith(".parquet"):
    snp_ids = pl.read_parquet(SNP_ID_SOURCE)["ID"].to_list()
else:
    with open(SNP_ID_SOURCE) as f:
        snp_ids = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(snp_ids):,} SNP IDs from {SNP_ID_SOURCE}")

# Step 2: reference panel LD scores, one chromosome at a time
frames = []
missing_chroms = []
for chrom in CHROMS:
    path = os.path.join(REFERENCE_PANEL_DIR, FILENAME_PATTERN.format(chrom=chrom))
    if not os.path.exists(path):
        missing_chroms.append(chrom)
        continue
    frames.append(pl.read_csv(path, separator="\t", columns=["SNP", "L2"]))

if missing_chroms:
    print(f"[!] {len(missing_chroms)} chromosome file(s) not found, skipped: "
          f"{missing_chroms}")
if not frames:
    raise SystemExit(f"No reference panel files found in {REFERENCE_PANEL_DIR} "
                      f"matching pattern {FILENAME_PATTERN!r} for CHROMS={CHROMS}.")

ld_df = pl.concat(frames)
n_before_dedup = ld_df.height
ld_df = ld_df.filter(~pl.col("SNP").is_duplicated())
n_dup = n_before_dedup - ld_df.height
if n_dup:
    print(f"Dropped {n_dup:,} SNP IDs that appear more than once in the "
          f"reference panel (ambiguous which L2 value is correct).")
print(f"Reference panel: {ld_df.height:,} usable SNPs across "
      f"{len(frames)} chromosome file(s).")

# Step 3: look up each of your SNPs against the reference, preserving order
snp_order = pl.DataFrame({"ID": snp_ids}).with_row_index("_order")
joined = (
    snp_order
    .join(ld_df, left_on="ID", right_on="SNP", how="left")
    .sort("_order")
)
ld_scores = joined["L2"].to_numpy().astype(np.float64)   # unmatched becomes NaN

n_matched = int(np.sum(~np.isnan(ld_scores)))
n_missing = len(ld_scores) - n_matched

# Step 4: save
np.save(OUTPUT_PATH, ld_scores)

print(f"\nMatched   : {n_matched:,} of {len(ld_scores):,} SNPs "
      f"({100*n_matched/len(ld_scores):.2f}%)")
print(f"Unmatched : {n_missing:,} SNPs -> NaN in ld_scores.npy "
      f"(derive_LDSC_intercept_matrix.py skips these automatically)")
print(f"Saved {OUTPUT_PATH}  shape={ld_scores.shape}  dtype={ld_scores.dtype}")
print(f"\nThis length must match Z_matrix.npy's row count exactly — "
      f"derive_LDSC_intercept_matrix.py checks that and will raise if it doesn't.")