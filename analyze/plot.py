import argparse, pandas as pd
import matplotlib.pyplot as plt
import os

ap = argparse.ArgumentParser()
ap.add_argument("metrics_csv")
ap.add_argument("-o", "--outdir", required=True)
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
df = pd.read_csv(args.metrics_csv)

def save_plot(sub, title, filename):
    plt.figure()
    for key, d in sub.groupby(sub.columns[:-4].tolist()):  # group by non-metric columns
        label = ", ".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        plt.plot(d.index, d["p95"], marker="o", label=label)
    plt.title(title)
    plt.xlabel("configurations")
    plt.ylabel("p95 wall ms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, filename))
    plt.close()

# Simple plots per bench
for bench in df["bench"].unique():
    sub = df[df["bench"]==bench].copy()
    sub = sub.reset_index(drop=True)
    save_plot(sub, f"{bench} p95", f"{bench}_p95.png")
print("Plots written to", args.outdir)
