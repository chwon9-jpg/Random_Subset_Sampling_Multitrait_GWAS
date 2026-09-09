###################################################################################
#                           User Configuration required
# These are None on purpose: set them for the file being converted rather than
# inheriting values from the last file this was run on.
#
#   INPUT_PATH    : the .csv/.tsv/.txt file to convert. Header row required.
#   OUTPUT_DIR    : where Z_matrix.npy and the companion Parquet are written.
#   HAS_ID_COLUMN : True/False to force whether column 0 holds SNP IDs rather than
#                   phenotype values, or None to auto-detect (see step 2 above).
#   SEP           : column delimiter, or None to infer from INPUT_PATH's extension
#                   (.csv -> ",", anything else -> tab).
###################################################################################

INPUT_PATH    = "/path/to/zscore_matrix.csv/.tsv/.txt"
OUTPUT_DIR    = "/path/to/output/directory"
HAS_ID_COLUMN = None
SEP           = None

if None in (INPUT_PATH, OUTPUT_DIR):
    raise SystemExit("Set INPUT_PATH and OUTPUT_DIR in the configuration block above.")

import os
import numpy as np
import polars as pl

os.makedirs(OUTPUT_DIR, exist_ok=True)

sep = SEP if SEP is not None else ("," if INPUT_PATH.lower().endswith(".csv") else "\t")

header = pl.read_csv(INPUT_PATH, separator=sep, n_rows=0).columns
if len(header) < 2:
    raise SystemExit(f"Only {len(header)} column(s) found with separator {sep!r} - "
                      f"check SEP matches the file's actual delimiter.")

# Auto-detect an ID column: read a small sample and see whether column 0 parses as
# a float the way every other column must. A genuine phenotype column that happens
# to fail here would fail identically on the full read below, so this sample is
# only ever wrong in the direction of correctly detecting a non-numeric column.
if HAS_ID_COLUMN is None:
    sample = pl.read_csv(INPUT_PATH, separator=sep, n_rows=100)
    try:
        sample[header[0]].cast(pl.Float64)
        has_id_column = False
    except pl.exceptions.InvalidOperationError:
        has_id_column = True
    print(f"HAS_ID_COLUMN not set - auto-detected column 0 ({header[0]!r}) as "
          f"{'an ID column' if has_id_column else 'a phenotype column'}.")
else:
    has_id_column = HAS_ID_COLUMN

pheno_cols = header[1:] if has_id_column else header
schema_overrides = {c: pl.Float32 for c in pheno_cols}
df = pl.read_csv(INPUT_PATH, separator=sep, schema_overrides=schema_overrides)

if has_id_column:
    snp_ids = df[header[0]].cast(pl.Utf8).to_list()
else:
    snp_ids = [f"SNP_{i+1}" for i in range(df.height)]
    print(f"No ID column: synthesized sequential IDs SNP_1..SNP_{df.height:,} "
          f"for the companion Parquet.")

Z = df.select(pheno_cols).to_numpy().astype(np.float32)

z_path = os.path.join(OUTPUT_DIR, "Z_matrix.npy")
np.save(z_path, Z)

parquet_path = os.path.join(OUTPUT_DIR, "df_allSNPs_allphenos.parquet")
out_df = pl.DataFrame({"ID": snp_ids})
for i, name in enumerate(pheno_cols):
    out_df = out_df.with_columns(pl.Series(name, Z[:, i]))
out_df.write_parquet(parquet_path)

n_missing = int(np.isnan(Z).sum())
print(f"\nWrote {z_path}")
print(f"  shape={Z.shape}  dtype={Z.dtype}")
print(f"  missing entries: {n_missing:,} of {Z.size:,} "
      f"({100*n_missing/Z.size:.2f}%)")
print(f"Wrote {parquet_path}")
print(f"  columns: ID + {len(pheno_cols)} phenotypes")
print(f"  first 3 phenotype names: {pheno_cols[:3]}")
print(f"  first 3 SNP IDs        : {snp_ids[:3]}")