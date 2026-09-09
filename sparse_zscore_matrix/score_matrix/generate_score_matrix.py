def _require_positive_int(name, value):
    if value is None:
        raise SystemExit(f"Set {name} in the configuration block above.")
    if not isinstance(value, int) or isinstance(value, bool):
        raise SystemExit(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if value <= 0:
        raise SystemExit(f"{name} must be a positive integer, got {value}")

N_WORKERS, BLAS_THREADS = None, None
SNP_BATCH = None # user to define
VERIFY_N_SAMPLE = None # user to define, dependent on |SNPs|

base_dir = "/path/to/output/directory"
stat_path = "/pasteur/helix/projects/GGS_CONNECT/WKD_CHRISTOPHER/toy_example_expected_outputs/sparse_zscore_matrix/sampling_loop/stat_matrix.dat"
sel_path  = "/pasteur/helix/projects/GGS_CONNECT/WKD_CHRISTOPHER/toy_example_expected_outputs/sparse_zscore_matrix/sampling_loop/sel_matrix.npy"
out_path  = f"{base_dir}/score_matrix.dat"

_require_positive_int("N_WORKERS", N_WORKERS)
_require_positive_int("BLAS_THREADS", BLAS_THREADS)
_require_positive_int("SNP_BATCH", SNP_BATCH)
_require_positive_int("VERIFY_N_SAMPLE", VERIFY_N_SAMPLE)

import os
for env in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]: os.environ[env] = str(BLAS_THREADS)

import time, math, numpy as np
from joblib import Parallel, delayed

# Shapes taken from the inputs. stat_matrix is a raw .dat with no header, so its SNP count
# is recovered from the file size; the check catches a stale or truncated file.
Ns, Np  = np.load(sel_path, mmap_mode='r').shape
_nbytes = os.path.getsize(stat_path)
if _nbytes % (Ns * 4):
    raise SystemExit(f"{stat_path}: {_nbytes:,} B is not a whole number of rows for Ns={Ns} float32.")
N_snps  = _nbytes // (Ns * 4)

def process_snp_range(snp_start, snp_end, sel_float, Np):
    stat = np.memmap(stat_path, dtype='float32', mode='r',  shape=(Ns, N_snps))
    out  = np.memmap(out_path,  dtype='float32', mode='r+', shape=(N_snps, Np))

    tot_b = math.ceil((snp_end - snp_start) / SNP_BATCH)
    w_start = time.time()

    for b_i, b_start in enumerate(range(snp_start, snp_end, SNP_BATCH), 1):
        b_end = min(b_start + SNP_BATCH, snp_end)
        t0 = time.time()

        stat_batch = np.array(stat[:, b_start:b_end], dtype=np.float32)   # (Ns, batch)
        valid      = ~np.isnan(stat_batch)                                 # (Ns, batch)
        np.nan_to_num(stat_batch, copy=False, nan=0.0)                    # NaN -> 0, in place
        valid_f32  = valid.astype(np.float32)                              # (Ns, batch)

        numerator = stat_batch.T @ sel_float   # (batch, Np)
        denom     = valid_f32.T   @ sel_float   # (batch, Np) — per-SNP, per-phenotype count

        with np.errstate(invalid='ignore'):    # 0/0 -> NaN is the intended result here
            out[b_start:b_end, :] = numerator / denom

        print(f"[worker {snp_start:,}–{snp_end:,}] batch {b_i}/{tot_b} ({100*b_i/tot_b:.1f}%) | SNPs {b_start:,}–{b_end:,} | {time.time()-t0:.1f}s", flush=True)

    out.flush()
    print(f"[worker {snp_start:,}–{snp_end:,}] DONE in {(time.time()-w_start)/60:.1f} min", flush=True)

if __name__ == '__main__':
    sel = np.load(sel_path)
    sel_float = sel.astype(np.float32)
    del sel

    print(f"Np={Np} | N_snps={N_snps:,} | Ns={Ns}\nWorkers={N_WORKERS} | SNP_BATCH={SNP_BATCH:,} | BLAS={BLAS_THREADS}")
    print(f"Peak RAM (array-only floor, actual has run ~1.3x higher -- see config comment): "
          f"{N_WORKERS * (Ns * SNP_BATCH * 9) / 1e9:.0f} GB | Output: {N_snps * Np * 4 / 1e9:.1f} GB")

    # Initialize file
    np.memmap(out_path, dtype='float32', mode='w+', shape=(N_snps, Np))

    chunk = math.ceil(N_snps / N_WORKERS)
    ranges = [(i * chunk, min((i + 1) * chunk, N_snps)) for i in range(N_WORKERS)]
    print(f"SNP ranges per worker: {ranges}")

    Parallel(n_jobs=N_WORKERS, backend='loky', verbose=5)(
        delayed(process_snp_range)(s, e, sel_float, Np) for s, e in ranges
    )
    print("Done. score_matrix.dat written.")


    print("\nRunning partial verification of score_matrix.dat...")
    out_check  = np.memmap(out_path, dtype='float32', mode='r', shape=(N_snps, Np))
    sample_idx = np.linspace(0, N_snps - 1, num=min(VERIFY_N_SAMPLE, N_snps), dtype=np.int64)
    sample     = np.asarray(out_check[sample_idx, :])
    n_allzero  = int((sample.sum(axis=1) == 0).sum())
    n_allnan   = int(np.isnan(sample).all(axis=1).sum())
    n_nan      = int(np.isnan(sample).sum())
    del out_check

    print(f"  sampled {len(sample_idx):,} of {N_snps:,} rows")
    print(f"  min={np.nanmin(sample):.4f} max={np.nanmax(sample):.4f} mean={np.nanmean(sample):.4f}  (NaN-excluded)")
    print(f"  NaN entries in sample     : {n_nan:,} of {sample.size:,} ({100*n_nan/sample.size:.2f}%) "
          f"— expected to roughly track the z-score matrix's own missingness rate")
    print(f"  all-zero rows in sample   : {n_allzero}  (unexpected at any sparsity — investigate if nonzero)")
    print(f"  all-NaN rows in sample    : {n_allnan}  (a SNP with no valid score for any phenotype — unusual, worth checking, not automatically a bug)")

    if n_allzero:
        print("\n[!] Partial check found all-zero rows - NOT offering to delete "
              "stat_matrix.dat / sel_matrix.npy. Investigate score_matrix.dat "
              "before rerunning; the inputs have been left in place so the run "
              "can be repeated without regenerating them.")
    else:
        print("\nPartial check passed (no all-zero rows).")
        print(f"stat_matrix.dat ({os.path.getsize(stat_path) / 1e9:.1f} GB) and "
              f"sel_matrix.npy ({os.path.getsize(sel_path) / 1e6:.1f} MB) were only "
              f"inputs to the matrix product above and are not needed again once "
              f"score_matrix.dat exists.")
        try:
            answer = input("Delete stat_matrix.dat and sel_matrix.npy now? [y/N]: ").strip().lower()
        except EOFError:
            # No attached terminal (e.g. running under sbatch) - default to the
            # safe, non-destructive choice rather than guessing.
            print("  no interactive input available (e.g. running under sbatch) - "
                  "leaving both files in place. Delete them by hand, or rerun this "
                  "script interactively, if you want them removed.")
            answer = "n"

        if answer in ("y", "yes"):
            os.remove(stat_path)
            os.remove(sel_path)
            print(f"Deleted {stat_path}")
            print(f"Deleted {sel_path}")
        else:
            print("Left stat_matrix.dat and sel_matrix.npy in place.")