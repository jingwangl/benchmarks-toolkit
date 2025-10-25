#!/usr/bin/env bash
# Usage: run.sh <O2|O3> <threads> [iterations]
set -euo pipefail
opt=${1:-O2}
threads=${2:-1}
iters=${3:-90000000}  # Fixed workload targeting ~30ms single-thread on typical PC
cd "$(dirname "$0")"
bin="build/cpp_compute_${opt}"
if [ ! -x "$bin" ]; then
  echo "Binary $bin not found. Run build.sh first." >&2
  exit 1
fi
label="cpp_compute_${opt}_t${threads}"
out="$(realpath ../../out)/cpp_compute_raw.csv"
mkdir -p "$(dirname "$out")"
# header if empty
[ -f "$out" ] || echo "bench,config,threads,wall_ms,iterations" > "$out"

# run 5 times and append for better statistical significance
for i in 1 2 3 4 5; do
  line=$("$bin" "$iters" "$threads")
  wall=$(echo "$line" | awk -F'[,=]' '{for(i=1;i<=NF;i++){if($i~"wall_ms"){print $(i+1); exit}}}')
  echo "cpp_compute,${opt},${threads},${wall},${iters}" >> "$out"
done
echo "Wrote results to $out"
