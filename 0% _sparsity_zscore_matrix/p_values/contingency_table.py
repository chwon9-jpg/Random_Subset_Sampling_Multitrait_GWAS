import numpy as np
import pandas as pd

out_dir      = "/path/to/p_values"
out_txt_path = f"{out_dir}/contingency_table_output.txt"
THRESHOLD    = 5e-8

_lines = []
def log(msg=""):
    print(msg)
    _lines.append(str(msg))

mat = np.load("/path/to/pvalue_matrix.npy")
uni    = mat[:, 0] < THRESHOLD
joint  = mat[:, 1] < THRESHOLD
subset = mat[:, 2] < THRESHOLD

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

with open(out_txt_path, "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nSaved -> {out_txt_path}")