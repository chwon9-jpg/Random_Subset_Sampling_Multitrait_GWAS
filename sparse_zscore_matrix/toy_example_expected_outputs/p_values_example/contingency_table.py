import numpy as np
import pandas as pd

out_dir     = "/path/to/output/directory"
out_txt_path = f"{out_dir}/contingency_table_output.txt"
THRESHOLD   = 5e-8

_lines = []
def log(msg=""):
    print(msg)
    _lines.append(str(msg))

mat = np.load(f"{out_dir}/pvalue_matrix.npy")

# 5 columns: 0 = min_univariate_pval, 1 = joint_pval, 2 = n_pheno_observed, 3 = min_subset_pval_bonf, 4 = n_subsets_used.
uni_p    = mat[:, 0]
joint_p  = mat[:, 1]
subset_p = mat[:, 3]

uni_nan    = np.isnan(uni_p)
joint_nan  = np.isnan(joint_p)
subset_nan = np.isnan(subset_p)

uni    = (uni_p    < THRESHOLD) & ~uni_nan
joint  = (joint_p  < THRESHOLD) & ~joint_nan
subset = (subset_p < THRESHOLD) & ~subset_nan

categories = {
    "None":                (~uni) & (~joint) & (~subset),
    "Uni only":            uni    & (~joint) & (~subset),
    "Joint only":          (~uni) & joint    & (~subset),
    "Subsets only":        (~uni) & (~joint) & subset,
    "Uni + Joint":         uni    & joint    & (~subset),
    "Uni + Subsets":       uni    & (~joint) & subset,
    "Joint + Subsets":     (~uni) & joint    & subset,
    "All three":           uni    & joint    & subset,
}

N = len(mat)

counts = {label: mask.sum() for label, mask in categories.items()}

index = ["Uni", "Joint", "Subsets"]
table = pd.DataFrame(index=index, columns=index, dtype=object)
table[:] = ""

def fmt(label):
    c = counts[label]
    return f"{c:,} ({c/N*100:.3f}%)"

table.loc["Uni",     "Uni"]     = fmt("Uni only")
table.loc["Joint",   "Joint"]   = fmt("Joint only")
table.loc["Subsets", "Subsets"] = fmt("Subsets only")
table.loc["Uni",     "Joint"]   = fmt("Uni + Joint")
table.loc["Uni",     "Subsets"] = fmt("Uni + Subsets")
table.loc["Joint",   "Subsets"] = fmt("Joint + Subsets")

log("Contingency table (count / proportion of all SNPs):")
log(table.to_string())
log(f"\nAll three significant:  {fmt('All three')}")
log(f"None significant:       {fmt('None')}")
log(f"\nTotal SNPs: {N:,}")

# Alternate view of the same 8 categorie in list format
list_order = ["None", "Uni only", "Joint only", "Subsets only",
              "Uni + Joint", "Uni + Subsets", "Joint + Subsets", "All three"]
list_df = pd.DataFrame({
    "Category":        list_order,
    "Count":           [f"{counts[c]:,}" for c in list_order],
    "Proportion (%)":  [f"{counts[c]/N*100:.3f}" for c in list_order],
})
log("\nSame counts, list style:")
log(list_df.to_string(index=False))
log(f"\nTotal SNPs: {N:,}")

def fmt_nan(mask):
    c = int(mask.sum())
    return f"{c:,} ({c/N*100:.4f}%)"

log(f"\nUntested (NaN) per criterion -- folded into 'not significant' above, reported separately here:")
log(f"  Uni untested:     {fmt_nan(uni_nan)}")
log(f"  Joint untested:   {fmt_nan(joint_nan)}")
log(f"  Subsets untested: {fmt_nan(subset_nan)}")

with open(out_txt_path, "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nSaved -> {out_txt_path}")