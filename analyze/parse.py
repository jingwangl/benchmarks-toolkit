import argparse, pandas as pd, numpy as np, os, glob
ap = argparse.ArgumentParser()
ap.add_argument("inputs", nargs="+")
ap.add_argument("-o", "--output", required=True)
args = ap.parse_args()

dfs = []
for pat in args.inputs:
    for p in glob.glob(pat):
        if p.endswith(".csv"):
            dfs.append(pd.read_csv(p))
if not dfs:
    raise SystemExit("No CSVs found")

df = pd.concat(dfs, ignore_index=True)
# Normalize columns for aggregation
if "bench" not in df.columns:
    raise SystemExit("CSV must include 'bench' column")

def agg_compute(d):
    # supports cpp_compute and py_io
    if "wall_ms" not in d: return None
    out = {
        "p50": np.percentile(d["wall_ms"], 50),
        "p95": np.percentile(d["wall_ms"], 95),
        "p99": np.percentile(d["wall_ms"], 99),
        "mean": d["wall_ms"].mean()
    }
    return pd.Series(out)

group_cols = []
if "config" in df.columns: group_cols.append("config")
if "threads" in df.columns: group_cols.append("threads")
if "block_kb" in df.columns: group_cols.append("block_kb")

g = df.groupby(["bench"] + group_cols, dropna=False).apply(agg_compute).reset_index()
g.to_csv(args.output, index=False)
print("Wrote", args.output)
