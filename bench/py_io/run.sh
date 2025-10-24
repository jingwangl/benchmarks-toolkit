#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out="$(realpath ../../out)/py_io_raw.csv"
mkdir -p "$(dirname "$out")"
echo "bench,block_kb,wall_ms,count" > "$out"
for kb in 4 64 512; do
  for i in 1 2 3; do
    line=$(python3 io_bench.py --block_kb $kb --count 20 2>/dev/null || true)  # Reduced for home computer compatibility
    wall=$(echo "$line" | awk -F'[,=]' '{for(i=1;i<=NF;i++){if($i~"wall_ms"){print $(i+1); exit}}}')
    echo "py_io,${kb},${wall},20" >> "$out"
  done
done
echo "Wrote results to $out"
